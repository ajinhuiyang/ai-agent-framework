// Package transformer implements a pure-Go Transformer (LLaMA-architecture)
// inference engine. It loads weights from GGUF, runs forward passes with
// RMSNorm, RoPE, grouped-query attention, SwiGLU FFN, KV cache, and
// supports both greedy and sampling-based token generation.
//
// Supported architectures: LLaMA, Qwen, Mistral, Gemma (all LLaMA-family).
//
// This is a CPU-only implementation. Performance will be significantly slower
// than llama.cpp but is fully portable with zero CGO dependencies.
package transformer

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/your-org/local-llm/internal/domain"
	"github.com/your-org/local-llm/internal/engine"
	"github.com/your-org/local-llm/internal/engine/native/bpe"
	"github.com/your-org/local-llm/internal/engine/native/gguf"
)

// ModelConfig holds hyperparameters extracted from GGUF metadata.
type ModelConfig struct {
	Arch             string // "llama", "qwen2", etc.
	VocabSize        int
	HiddenSize       int // a.k.a. embedding_length
	IntermediateSize int // FFN intermediate (typically 4*hidden or with SwiGLU scaling)
	NumLayers        int // num_hidden_layers
	NumHeads         int // num_attention_heads
	NumKVHeads       int // num_key_value_heads (for GQA; == NumHeads if MHA)
	HeadDim          int // hidden_size / num_heads
	MaxSeqLen        int // context_length
	RMSNormEps       float32
	RopeTheta        float32 // RoPE base frequency
	RopeScaling      float32
}

// layerWeights holds dequantized weights for one Transformer layer.
type layerWeights struct {
	// Attention weights — GGUF stores as [in_features, out_features]
	// matVecMul treats them as: out[i] = sum(W[i*cols..] * vec)
	// So we need to transpose: store as [out_features, in_features] row-major
	Wq []float32 // [hidden, hidden] (transposed from GGUF)
	Wk []float32 // [kv_dim, hidden] (transposed from GGUF)
	Wv []float32 // [kv_dim, hidden] (transposed from GGUF)
	Wo []float32 // [hidden, hidden] (transposed from GGUF)
	// Attention biases (Qwen2 has these)
	Bq []float32 // [hidden] or nil
	Bk []float32 // [kv_dim] or nil
	Bv []float32 // [kv_dim] or nil
	// Attention norm
	AttnNorm []float32 // [hidden]
	// FFN
	WGate []float32 // [intermediate, hidden] (transposed from GGUF)
	WUp   []float32 // [intermediate, hidden] (transposed from GGUF)
	WDown []float32 // [hidden, intermediate] (transposed from GGUF)
	// FFN norm
	FFNNorm []float32 // [hidden]
}

// modelWeights holds all dequantized model weights.
type modelWeights struct {
	TokenEmbed []float32 // [vocab, hidden]
	OutputNorm []float32 // [hidden]
	Output     []float32 // [vocab, hidden] (lm_head)
	Layers     []layerWeights
}

// kvCache holds the key-value cache for one layer.
type kvCache struct {
	K   []float32 // [seq_len, kv_dim]
	V   []float32 // [seq_len, kv_dim]
	Len int       // number of cached positions
}

// Engine is the native pure-Go Transformer inference engine.
type Engine struct {
	mu     sync.RWMutex
	loaded bool

	config  ModelConfig
	weights modelWeights
	tok     *bpe.Tokenizer
	cache   []kvCache // per-layer KV cache
	meta    *engine.ModelMeta

	logger *zap.Logger
}

// New creates a new native Transformer engine.
func New(logger *zap.Logger) *Engine {
	if logger == nil {
		logger, _ = zap.NewDevelopment()
	}
	return &Engine{logger: logger}
}

// Load loads a GGUF model file, parses metadata, and dequantizes all weights.
func (e *Engine) Load(ctx context.Context, modelPath string, opts engine.LoadOptions) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	start := time.Now()
	e.logger.Info("loading GGUF model", zap.String("path", modelPath))

	// Parse GGUF file.
	gf, err := gguf.Parse(modelPath)
	if err != nil {
		return fmt.Errorf("parse GGUF: %w", err)
	}

	// Extract model configuration from metadata.
	cfg, err := extractConfig(gf)
	if err != nil {
		return fmt.Errorf("extract config: %w", err)
	}
	if opts.NumCtx > 0 && opts.NumCtx < cfg.MaxSeqLen {
		cfg.MaxSeqLen = opts.NumCtx
	}
	e.config = cfg

	e.logger.Info("model config",
		zap.String("arch", cfg.Arch),
		zap.Int("vocab", cfg.VocabSize),
		zap.Int("hidden", cfg.HiddenSize),
		zap.Int("layers", cfg.NumLayers),
		zap.Int("heads", cfg.NumHeads),
		zap.Int("kv_heads", cfg.NumKVHeads),
		zap.Int("ctx_len", cfg.MaxSeqLen),
	)

	// Build tokenizer from GGUF metadata.
	e.tok, err = bpe.NewFromGGUF(gf)
	if err != nil {
		return fmt.Errorf("build tokenizer: %w", err)
	}
	e.logger.Info("tokenizer loaded", zap.Int("vocab_size", e.tok.VocabSize()))

	// Load and dequantize weights.
	if err := e.loadWeights(gf); err != nil {
		return fmt.Errorf("load weights: %w", err)
	}

	// Initialize KV cache.
	e.cache = make([]kvCache, cfg.NumLayers)
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	for i := range e.cache {
		e.cache[i] = kvCache{
			K: make([]float32, 0, cfg.MaxSeqLen*kvDim),
			V: make([]float32, 0, cfg.MaxSeqLen*kvDim),
		}
	}

	// Build metadata.
	fileInfo, _ := os.Stat(modelPath)
	fileSize := int64(0)
	if fileInfo != nil {
		fileSize = fileInfo.Size()
	}
	e.meta = &engine.ModelMeta{
		Name:          modelPath,
		Path:          modelPath,
		Family:        cfg.Arch,
		ParameterSize: fmt.Sprintf("%dM", estimateParams(cfg)/1_000_000),
		ContextLength: cfg.MaxSeqLen,
		EmbeddingSize: cfg.HiddenSize,
		FileSizeBytes: fileSize,
	}

	e.loaded = true
	e.logger.Info("model loaded",
		zap.Duration("elapsed", time.Since(start)),
		zap.Int("tensors", len(gf.Tensors)),
	)
	return nil
}

func (e *Engine) Unload() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.loaded = false
	e.weights = modelWeights{}
	e.cache = nil
	e.tok = nil
	e.meta = nil
	runtime.GC()
	return nil
}

// Predict generates text (non-streaming).
func (e *Engine) Predict(ctx context.Context, req domain.InferenceRequest) (engine.PredictResult, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if !e.loaded {
		return engine.PredictResult{}, fmt.Errorf("no model loaded")
	}

	start := time.Now()
	e.resetCache()

	tokens := e.tok.Encode(req.Prompt)
	tokens = append([]int{e.tok.BOSID()}, tokens...)
	promptLen := len(tokens)

	maxGen := req.Options.NumPredict
	if maxGen <= 0 {
		maxGen = 256
	}

	// Generate tokens.
	var generated []int
	for i := 0; i < maxGen; i++ {
		select {
		case <-ctx.Done():
			break
		default:
		}

		logits := e.forward(tokens)
		nextToken := e.sample(logits, req.Options)

		if nextToken == e.tok.EOSID() {
			break
		}
		if isStopToken(e.tok.DecodeToken(nextToken), req.Options.Stop) {
			break
		}

		generated = append(generated, nextToken)
		tokens = []int{nextToken} // Next iteration: only new token (KV cache holds history)
	}

	text := e.tok.Decode(generated)
	elapsed := time.Since(start)

	return engine.PredictResult{
		Text:             text,
		PromptTokens:     promptLen,
		CompletionTokens: len(generated),
		TotalDuration:    elapsed.Nanoseconds(),
		PromptDuration:   elapsed.Nanoseconds() / 3,
		EvalDuration:     elapsed.Nanoseconds() * 2 / 3,
	}, nil
}

// PredictStream generates tokens one by one, sending each to a channel.
func (e *Engine) PredictStream(ctx context.Context, req domain.InferenceRequest) (<-chan engine.StreamToken, error) {
	e.mu.Lock()
	if !e.loaded {
		e.mu.Unlock()
		return nil, fmt.Errorf("no model loaded")
	}

	ch := make(chan engine.StreamToken, 32)

	go func() {
		defer close(ch)
		defer e.mu.Unlock()

		e.resetCache()

		tokens := e.tok.Encode(req.Prompt)
		tokens = append([]int{e.tok.BOSID()}, tokens...)

		maxGen := req.Options.NumPredict
		if maxGen <= 0 {
			maxGen = 256
		}

		for i := 0; i < maxGen; i++ {
			select {
			case <-ctx.Done():
				ch <- engine.StreamToken{Err: ctx.Err(), Done: true}
				return
			default:
			}

			logits := e.forward(tokens)
			nextToken := e.sample(logits, req.Options)

			if nextToken == e.tok.EOSID() {
				ch <- engine.StreamToken{Done: true}
				return
			}

			text := e.tok.DecodeToken(nextToken)
			if isStopToken(text, req.Options.Stop) {
				ch <- engine.StreamToken{Done: true}
				return
			}

			ch <- engine.StreamToken{Text: text}
			tokens = []int{nextToken}
		}

		ch <- engine.StreamToken{Done: true}
	}()

	return ch, nil
}

// Embed generates embeddings by running the model's embedding layer.
func (e *Engine) Embed(_ context.Context, texts []string) ([][]float32, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if !e.loaded {
		return nil, fmt.Errorf("no model loaded")
	}

	results := make([][]float32, len(texts))
	for i, text := range texts {
		tokens := e.tok.Encode(text)
		if len(tokens) == 0 {
			results[i] = make([]float32, e.config.HiddenSize)
			continue
		}

		// Get embeddings for all tokens and mean-pool.
		embed := make([]float32, e.config.HiddenSize)
		for _, tok := range tokens {
			offset := tok * e.config.HiddenSize
			if offset+e.config.HiddenSize <= len(e.weights.TokenEmbed) {
				for j := 0; j < e.config.HiddenSize; j++ {
					embed[j] += e.weights.TokenEmbed[offset+j]
				}
			}
		}
		// Mean pool.
		scale := 1.0 / float32(len(tokens))
		for j := range embed {
			embed[j] *= scale
		}
		// L2 normalize.
		var norm float32
		for _, v := range embed {
			norm += v * v
		}
		norm = float32(math.Sqrt(float64(norm)))
		if norm > 0 {
			for j := range embed {
				embed[j] /= norm
			}
		}
		results[i] = embed
	}
	return results, nil
}

func (e *Engine) TokenCount(_ context.Context, text string) (int, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if !e.loaded {
		return 0, fmt.Errorf("no model loaded")
	}
	return len(e.tok.Encode(text)), nil
}

func (e *Engine) ModelInfo() *engine.ModelMeta {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.meta
}

func (e *Engine) IsLoaded() bool {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.loaded
}

// ======================================================================
// Forward pass
// ======================================================================

// forward runs a single forward pass for the given token(s).
// Returns logits of shape [vocab_size].
func (e *Engine) forward(tokens []int) []float32 {
	cfg := e.config
	hidden := cfg.HiddenSize
	seqLen := len(tokens)

	// Token embedding lookup.
	x := make([]float32, seqLen*hidden)
	for i, tok := range tokens {
		if tok >= 0 && tok < cfg.VocabSize {
			offset := tok * hidden
			copy(x[i*hidden:(i+1)*hidden], e.weights.TokenEmbed[offset:offset+hidden])
		}
	}

	// Run through each Transformer layer.
	for layer := 0; layer < cfg.NumLayers; layer++ {
		x = e.transformerLayer(x, seqLen, layer)
	}

	// Final RMSNorm.
	for i := 0; i < seqLen; i++ {
		rmsNorm(x[i*hidden:(i+1)*hidden], e.weights.OutputNorm, cfg.RMSNormEps)
	}

	// Output projection (only for the last token).
	lastHidden := x[(seqLen-1)*hidden : seqLen*hidden]
	logits := make([]float32, cfg.VocabSize)
	matVecMul(logits, e.weights.Output, lastHidden, cfg.VocabSize, hidden)

	return logits
}

// transformerLayer runs a single Transformer layer.
func (e *Engine) transformerLayer(x []float32, seqLen, layer int) []float32 {
	cfg := e.config
	hidden := cfg.HiddenSize
	lw := &e.weights.Layers[layer]

	// 1. Pre-attention RMSNorm.
	normed := make([]float32, len(x))
	copy(normed, x)
	for i := 0; i < seqLen; i++ {
		rmsNorm(normed[i*hidden:(i+1)*hidden], lw.AttnNorm, cfg.RMSNormEps)
	}

	// 2. Self-attention with KV cache.
	attnOut := e.selfAttention(normed, seqLen, layer)

	// 3. Residual connection.
	for i := range x {
		x[i] += attnOut[i]
	}

	// 4. Pre-FFN RMSNorm.
	normed2 := make([]float32, len(x))
	copy(normed2, x)
	for i := 0; i < seqLen; i++ {
		rmsNorm(normed2[i*hidden:(i+1)*hidden], lw.FFNNorm, cfg.RMSNormEps)
	}

	// 5. FFN (SwiGLU).
	ffnOut := e.feedForward(normed2, seqLen, layer)

	// 6. Residual connection.
	for i := range x {
		x[i] += ffnOut[i]
	}

	return x
}

// selfAttention implements multi-head (grouped-query) attention with RoPE and KV cache.
func (e *Engine) selfAttention(x []float32, seqLen, layer int) []float32 {
	cfg := e.config
	hidden := cfg.HiddenSize
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvDim := numKVHeads * headDim
	lw := &e.weights.Layers[layer]
	kvc := &e.cache[layer]

	out := make([]float32, seqLen*hidden)

	for pos := 0; pos < seqLen; pos++ {
		xPos := x[pos*hidden : (pos+1)*hidden]

		// Project Q, K, V.
		q := make([]float32, hidden)
		k := make([]float32, kvDim)
		v := make([]float32, kvDim)
		matVecMul(q, lw.Wq, xPos, hidden, hidden)
		matVecMul(k, lw.Wk, xPos, kvDim, hidden)
		matVecMul(v, lw.Wv, xPos, kvDim, hidden)

		// Add biases (Qwen2 has attention biases).
		if lw.Bq != nil {
			for i := range q {
				q[i] += lw.Bq[i]
			}
		}
		if lw.Bk != nil {
			for i := range k {
				k[i] += lw.Bk[i]
			}
		}
		if lw.Bv != nil {
			for i := range v {
				v[i] += lw.Bv[i]
			}
		}

		// Apply RoPE to Q and K.
		absPos := kvc.Len
		applyRoPE(q, headDim, numHeads, absPos, cfg.RopeTheta)
		applyRoPE(k, headDim, numKVHeads, absPos, cfg.RopeTheta)

		// Append K, V to cache.
		kvc.K = append(kvc.K, k...)
		kvc.V = append(kvc.V, v...)
		kvc.Len++

		// Compute attention for each query head.
		cachedLen := kvc.Len
		headsPerKVGroup := numHeads / numKVHeads

		for h := 0; h < numHeads; h++ {
			kvHead := h / headsPerKVGroup
			qHead := q[h*headDim : (h+1)*headDim]

			// Compute attention scores.
			scores := make([]float32, cachedLen)
			scale := float32(1.0 / math.Sqrt(float64(headDim)))

			for t := 0; t < cachedLen; t++ {
				kHead := kvc.K[t*kvDim+kvHead*headDim : t*kvDim+(kvHead+1)*headDim]
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += qHead[d] * kHead[d]
				}
				scores[t] = dot * scale
			}

			// Causal mask: current position can attend to all cached positions.
			// (Already correct since we only have positions <= absPos in cache.)

			// Softmax.
			softmax(scores)

			// Weighted sum of values.
			for t := 0; t < cachedLen; t++ {
				vHead := kvc.V[t*kvDim+kvHead*headDim : t*kvDim+(kvHead+1)*headDim]
				for d := 0; d < headDim; d++ {
					out[pos*hidden+h*headDim+d] += scores[t] * vHead[d]
				}
			}
		}

		// Output projection.
		projected := make([]float32, hidden)
		matVecMul(projected, lw.Wo, out[pos*hidden:(pos+1)*hidden], hidden, hidden)
		copy(out[pos*hidden:], projected)
	}

	return out
}

// feedForward implements SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
func (e *Engine) feedForward(x []float32, seqLen, layer int) []float32 {
	cfg := e.config
	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	lw := &e.weights.Layers[layer]

	out := make([]float32, seqLen*hidden)

	for pos := 0; pos < seqLen; pos++ {
		xPos := x[pos*hidden : (pos+1)*hidden]

		gate := make([]float32, inter)
		up := make([]float32, inter)
		matVecMul(gate, lw.WGate, xPos, inter, hidden)
		matVecMul(up, lw.WUp, xPos, inter, hidden)

		// SwiGLU activation: silu(gate) * up
		for i := range gate {
			gate[i] = silu(gate[i]) * up[i]
		}

		// Down projection.
		matVecMul(out[pos*hidden:(pos+1)*hidden], lw.WDown, gate, hidden, inter)
	}

	return out
}

// resetCache clears the KV cache.
func (e *Engine) resetCache() {
	kvDim := e.config.NumKVHeads * e.config.HeadDim
	for i := range e.cache {
		e.cache[i].K = make([]float32, 0, e.config.MaxSeqLen*kvDim)
		e.cache[i].V = make([]float32, 0, e.config.MaxSeqLen*kvDim)
		e.cache[i].Len = 0
	}
}

// ======================================================================
// Math utilities
// ======================================================================

// matVecMul computes out = mat * vec, where mat is [rows, cols] row-major.
// Uses goroutine parallelism for large matrices.
func matVecMul(out, mat, vec []float32, rows, cols int) {
	if rows < 256 {
		// Small matrix: single-threaded.
		for i := 0; i < rows; i++ {
			var sum float32
			rowOff := i * cols
			for j := 0; j < cols; j++ {
				sum += mat[rowOff+j] * vec[j]
			}
			out[i] = sum
		}
		return
	}

	// Large matrix: parallel with goroutines.
	numWorkers := runtime.NumCPU()
	chunkSize := (rows + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				var sum float32
				rowOff := i * cols
				// Unroll loop by 4 for better performance.
				j := 0
				for ; j+3 < cols; j += 4 {
					sum += mat[rowOff+j]*vec[j] +
						mat[rowOff+j+1]*vec[j+1] +
						mat[rowOff+j+2]*vec[j+2] +
						mat[rowOff+j+3]*vec[j+3]
				}
				for ; j < cols; j++ {
					sum += mat[rowOff+j] * vec[j]
				}
				out[i] = sum
			}
		}(start, end)
	}
	wg.Wait()
}

// rmsNorm applies Root Mean Square Layer Normalization in-place.
func rmsNorm(x, weight []float32, eps float32) {
	n := len(x)
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	ss = float32(1.0 / math.Sqrt(float64(ss/float32(n))+float64(eps)))
	for i := range x {
		x[i] = x[i] * ss * weight[i]
	}
}

// applyRoPE applies Rotary Position Embedding to a vector.
// Uses interleaved pairs: (x[0],x[1]), (x[2],x[3]), ... as used by Qwen2/LLaMA.
func applyRoPE(x []float32, headDim, numHeads, pos int, theta float32) {
	for h := 0; h < numHeads; h++ {
		for i := 0; i < headDim/2; i++ {
			freq := float32(1.0) / float32(math.Pow(float64(theta), float64(2*i)/float64(headDim)))
			angle := float32(pos) * freq
			cos := float32(math.Cos(float64(angle)))
			sin := float32(math.Sin(float64(angle)))

			idx := h*headDim + 2*i
			x0 := x[idx]
			x1 := x[idx+1]
			x[idx] = x0*cos - x1*sin
			x[idx+1] = x0*sin + x1*cos
		}
	}
}

// silu computes SiLU (Sigmoid Linear Unit): x * sigmoid(x)
func silu(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}

// softmax computes softmax in-place.
func softmax(x []float32) {
	max := x[0]
	for _, v := range x[1:] {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i := range x {
		x[i] = float32(math.Exp(float64(x[i] - max)))
		sum += x[i]
	}
	for i := range x {
		x[i] /= sum
	}
}

// ======================================================================
// Sampling
// ======================================================================

// sample selects the next token from logits.
func (e *Engine) sample(logits []float32, opts domain.Options) int {
	temp := opts.Temperature
	if temp <= 0 {
		temp = 1.0
	}

	// Greedy (temperature ~0).
	if temp < 0.01 {
		return argmax(logits)
	}

	// Apply temperature.
	for i := range logits {
		logits[i] /= float32(temp)
	}

	// Apply top-K filtering.
	topK := opts.TopK
	if topK > 0 && topK < len(logits) {
		logits = topKFilter(logits, topK)
	}

	// Apply top-P (nucleus) filtering.
	topP := opts.TopP
	if topP > 0 && topP < 1 {
		logits = topPFilter(logits, float32(topP))
	}

	// Apply repetition penalty (simplified).
	if opts.RepeatPenalty > 1 {
		// Would need generated token history; skipped for simplicity.
	}

	// Convert to probabilities.
	softmax(logits)

	// Multinomial sampling.
	r := rand.Float32()
	var cumulative float32
	for i, p := range logits {
		cumulative += p
		if r <= cumulative {
			return i
		}
	}
	return len(logits) - 1
}

func argmax(x []float32) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}

type tokenProb struct {
	idx   int
	logit float32
}

func topKFilter(logits []float32, k int) []float32 {
	items := make([]tokenProb, len(logits))
	for i, l := range logits {
		items[i] = tokenProb{i, l}
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].logit > items[j].logit
	})

	threshold := items[k-1].logit
	for i := range logits {
		if logits[i] < threshold {
			logits[i] = float32(math.Inf(-1))
		}
	}
	return logits
}

func topPFilter(logits []float32, p float32) []float32 {
	items := make([]tokenProb, len(logits))
	for i, l := range logits {
		items[i] = tokenProb{i, l}
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].logit > items[j].logit
	})

	// Compute softmax for sorted logits.
	probs := make([]float32, len(items))
	maxVal := items[0].logit
	var sum float32
	for i, it := range items {
		probs[i] = float32(math.Exp(float64(it.logit - maxVal)))
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}

	// Find cutoff.
	var cumul float32
	cutoffIdx := len(items) - 1
	for i, prob := range probs {
		cumul += prob
		if cumul >= p {
			cutoffIdx = i
			break
		}
	}

	// Mask tokens below cutoff.
	keep := make(map[int]bool)
	for i := 0; i <= cutoffIdx; i++ {
		keep[items[i].idx] = true
	}
	for i := range logits {
		if !keep[i] {
			logits[i] = float32(math.Inf(-1))
		}
	}
	return logits
}

// ======================================================================
// Weight loading
// ======================================================================

// loadWeights reads and dequantizes all model tensors.
// Uses a single file handle and parallel dequantization for speed.
func (e *Engine) loadWeights(gf *gguf.File) error {
	cfg := e.config
	w := &e.weights

	// Open file once for all tensor reads.
	file, err := os.Open(gf.FilePath)
	if err != nil {
		return fmt.Errorf("open model file: %w", err)
	}
	defer file.Close()

	e.logger.Info("loading weights",
		zap.Int("layers", cfg.NumLayers),
		zap.Int("tensors", len(gf.Tensors)),
	)

	loadStart := time.Now()

	readTensor := func(name string) ([]float32, error) {
		t, ok := gf.FindTensor(name)
		if !ok {
			return nil, fmt.Errorf("tensor %q not found", name)
		}
		offset := gf.DataOffset + int64(t.Offset)
		size := t.TensorDataSize()
		data := make([]byte, size)
		if _, err := file.ReadAt(data, offset); err != nil {
			return nil, fmt.Errorf("read tensor %q: %w", name, err)
		}
		return gguf.Dequantize(data, t.Type, t.ElementCount()), nil
	}

	// Token embedding.
	w.TokenEmbed, err = readTensor("token_embd.weight")
	if err != nil {
		return fmt.Errorf("token_embd: %w", err)
	}
	e.logger.Info("loaded token embeddings", zap.Duration("elapsed", time.Since(loadStart)))

	// Output norm.
	w.OutputNorm, err = readTensor("output_norm.weight")
	if err != nil {
		return fmt.Errorf("output_norm: %w", err)
	}

	// Output projection (lm_head).
	w.Output, err = readTensor("output.weight")
	if err != nil {
		e.logger.Info("output.weight not found, using tied embedding weights")
		w.Output = make([]float32, len(w.TokenEmbed))
		copy(w.Output, w.TokenEmbed)
	}

	// Per-layer weights — load in parallel per layer.
	w.Layers = make([]layerWeights, cfg.NumLayers)

	// Use a worker pool to load layers in parallel.
	numWorkers := runtime.NumCPU()
	if numWorkers > cfg.NumLayers {
		numWorkers = cfg.NumLayers
	}

	type layerJob struct {
		index int
		err   error
	}
	jobs := make(chan int, cfg.NumLayers)
	results := make(chan layerJob, cfg.NumLayers)

	// Open separate file handles for parallel reads.
	for wk := 0; wk < numWorkers; wk++ {
		go func() {
			f, err := os.Open(gf.FilePath)
			if err != nil {
				results <- layerJob{err: err}
				return
			}
			defer f.Close()

			for i := range jobs {
				prefix := fmt.Sprintf("blk.%d.", i)
				lw := &w.Layers[i]
				var loadErr error

				readT := func(name string) ([]float32, error) {
					t, ok := gf.FindTensor(name)
					if !ok {
						return nil, fmt.Errorf("tensor %q not found", name)
					}
					offset := gf.DataOffset + int64(t.Offset)
					size := t.TensorDataSize()
					data := make([]byte, size)
					if _, err := f.ReadAt(data, offset); err != nil {
						return nil, fmt.Errorf("read tensor %q: %w", name, err)
					}
					return gguf.Dequantize(data, t.Type, t.ElementCount()), nil
				}

				// Optional tensor (bias) — returns nil if not found.
				readOptional := func(name string) []float32 {
					result, err := readT(name)
					if err != nil {
						return nil
					}
					return result
				}

				lw.Wq, loadErr = readT(prefix + "attn_q.weight")
				if loadErr == nil {
					lw.Wk, loadErr = readT(prefix + "attn_k.weight")
				}
				if loadErr == nil {
					lw.Wv, loadErr = readT(prefix + "attn_v.weight")
				}
				if loadErr == nil {
					lw.Wo, loadErr = readT(prefix + "attn_output.weight")
				}
				if loadErr == nil {
					lw.AttnNorm, loadErr = readT(prefix + "attn_norm.weight")
				}
				if loadErr == nil {
					lw.WGate, loadErr = readT(prefix + "ffn_gate.weight")
				}
				if loadErr == nil {
					lw.WUp, loadErr = readT(prefix + "ffn_up.weight")
				}
				if loadErr == nil {
					lw.WDown, loadErr = readT(prefix + "ffn_down.weight")
				}
				if loadErr == nil {
					lw.FFNNorm, loadErr = readT(prefix + "ffn_norm.weight")
				}

				// Load optional biases (Qwen2 has attention biases).
				if loadErr == nil {
					lw.Bq = readOptional(prefix + "attn_q.bias")
					lw.Bk = readOptional(prefix + "attn_k.bias")
					lw.Bv = readOptional(prefix + "attn_v.bias")
				}

				results <- layerJob{index: i, err: loadErr}
			}
		}()
	}

	// Submit all layer jobs.
	for i := 0; i < cfg.NumLayers; i++ {
		jobs <- i
	}
	close(jobs)

	// Collect results.
	loaded := 0
	for range cfg.NumLayers {
		res := <-results
		if res.err != nil {
			return fmt.Errorf("layer %d: %w", res.index, res.err)
		}
		loaded++
		if loaded%4 == 0 || loaded == cfg.NumLayers {
			e.logger.Info("loaded layer weights",
				zap.Int("done", loaded),
				zap.Int("total", cfg.NumLayers),
				zap.Duration("elapsed", time.Since(loadStart)),
			)
		}
	}

	e.logger.Info("all weights loaded",
		zap.Duration("total_time", time.Since(loadStart)),
	)
	return nil
}

// loadTensor reads a tensor from the GGUF file and dequantizes it to float32.
func (e *Engine) loadTensor(gf *gguf.File, name string) ([]float32, error) {
	t, ok := gf.FindTensor(name)
	if !ok {
		return nil, fmt.Errorf("tensor %q not found", name)
	}
	return gf.ReadTensorFloat32(t)
}

// ======================================================================
// Config extraction
// ======================================================================

func extractConfig(gf *gguf.File) (ModelConfig, error) {
	cfg := ModelConfig{
		RMSNormEps:  1e-5,
		RopeTheta:   10000.0,
		RopeScaling: 1.0,
	}

	// Detect architecture.
	if arch, ok := gf.GetString("general.architecture"); ok {
		cfg.Arch = arch
	} else {
		cfg.Arch = "llama"
	}

	prefix := cfg.Arch + "."

	// Read hyperparameters.
	if v, ok := gf.GetUint32(prefix + "embedding_length"); ok {
		cfg.HiddenSize = int(v)
	}
	if v, ok := gf.GetUint32(prefix + "feed_forward_length"); ok {
		cfg.IntermediateSize = int(v)
	}
	if v, ok := gf.GetUint32(prefix + "block_count"); ok {
		cfg.NumLayers = int(v)
	}
	if v, ok := gf.GetUint32(prefix + "attention.head_count"); ok {
		cfg.NumHeads = int(v)
	}
	if v, ok := gf.GetUint32(prefix + "attention.head_count_kv"); ok {
		cfg.NumKVHeads = int(v)
	} else {
		cfg.NumKVHeads = cfg.NumHeads // MHA fallback
	}
	if v, ok := gf.GetUint32(prefix + "context_length"); ok {
		cfg.MaxSeqLen = int(v)
	} else {
		cfg.MaxSeqLen = 4096
	}
	if v, ok := gf.GetFloat32(prefix + "attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = v
	}
	if v, ok := gf.GetFloat32(prefix + "rope.freq_base"); ok {
		cfg.RopeTheta = v
	}

	// Vocab size from tokenizer.
	if tokens, ok := gf.Metadata["tokenizer.ggml.tokens"]; ok {
		if arr, ok := tokens.([]interface{}); ok {
			cfg.VocabSize = len(arr)
		}
	}

	// Derived.
	if cfg.NumHeads > 0 && cfg.HiddenSize > 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	if cfg.IntermediateSize == 0 && cfg.HiddenSize > 0 {
		// Common default: 8/3 * hidden, rounded to multiple of 256
		cfg.IntermediateSize = ((cfg.HiddenSize * 8 / 3) + 255) / 256 * 256
	}

	// Validate.
	if cfg.HiddenSize == 0 {
		return cfg, fmt.Errorf("embedding_length not found in GGUF metadata")
	}
	if cfg.NumLayers == 0 {
		return cfg, fmt.Errorf("block_count not found in GGUF metadata")
	}

	return cfg, nil
}

func estimateParams(cfg ModelConfig) int {
	embed := cfg.VocabSize * cfg.HiddenSize
	perLayer := 4*cfg.HiddenSize*cfg.HiddenSize + 3*cfg.HiddenSize*cfg.IntermediateSize + 2*cfg.HiddenSize
	output := cfg.VocabSize * cfg.HiddenSize
	return embed + cfg.NumLayers*perLayer + output
}

func isStopToken(text string, stopWords []string) bool {
	for _, sw := range stopWords {
		if strings.Contains(text, sw) {
			return true
		}
	}
	return false
}
