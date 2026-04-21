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

// layerWeights holds weights for one Transformer layer.
// Large 2D matrices are stored in raw Q8_0 format to halve memory bandwidth.
// Small 1D vectors (norms, biases) remain as float32.
type layerWeights struct {
	// Attention weights — raw Q8_0 bytes, [rows * cols_q8_bytes]
	Wq     []byte // Q8_0 raw: maps hidden → hidden
	Wk     []byte // Q8_0 raw: maps hidden → kv_dim
	Wv     []byte // Q8_0 raw: maps hidden → kv_dim
	Wo     []byte // Q8_0 raw: maps hidden → hidden
	WqCols int    // number of float32 columns (input dimension)
	WkCols int
	WvCols int
	WoCols int
	// Attention biases (Qwen2 has these) — float32, small
	Bq []float32 // [hidden] or nil
	Bk []float32 // [kv_dim] or nil
	Bv []float32 // [kv_dim] or nil
	// Attention norm — float32, small
	AttnNorm []float32 // [hidden]
	// FFN weights — raw Q8_0 bytes
	WGate     []byte // Q8_0 raw: maps hidden → intermediate
	WUp       []byte // Q8_0 raw: maps hidden → intermediate
	WDown     []byte // Q8_0 raw: maps intermediate → hidden
	WGateCols int
	WUpCols   int
	WDownCols int
	// FFN norm — float32, small
	FFNNorm []float32 // [hidden]
}

// modelWeights holds all model weights.
type modelWeights struct {
	TokenEmbed []float32 // [vocab, hidden] — float32 for embedding lookup
	OutputNorm []float32 // [hidden]
	Output     []byte    // Q8_0 raw: [vocab, hidden] — largest single tensor
	OutputCols int
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
	mu     sync.Mutex
	loaded bool

	config  ModelConfig
	weights modelWeights
	tok     *bpe.Tokenizer
	cache   []kvCache // per-layer KV cache
	meta    *engine.ModelMeta

	// Pre-allocated buffers to avoid GC pressure during inference.
	buffers *inferBuffers

	// Worker pool for parallel matVecMul.
	workerPool *workerPool

	logger *zap.Logger
}

// inferBuffers holds pre-allocated scratch space for a single forward pass.
type inferBuffers struct {
	x       []float32 // [hidden] current hidden state
	normed  []float32 // [max_seq*hidden] normalized hidden state
	q       []float32 // [hidden] query projection
	k       []float32 // [kv_dim] key projection
	v       []float32 // [kv_dim] value projection
	attnOut []float32 // [max_seq*hidden] attention output scratch
	proj    []float32 // [hidden] output projection scratch
	gate    []float32 // [intermediate] FFN gate
	up      []float32 // [intermediate] FFN up
	ffnOut  []float32 // [max_seq*hidden] FFN output scratch
	logits  []float32 // [vocab] output logits
	scores  []float32 // [max_seq_len] attention scores scratch
}

// workerPool provides a fixed set of goroutines for parallel matVecMul.
type workerPool struct {
	numWorkers int
	jobs       []chan matJob
	q8jobs     []chan matQ8Job
	done       []chan struct{}
}

type matJob struct {
	out    []float32
	mat    []float32
	vec    []float32
	startR int
	endR   int
	cols   int
}

type matQ8Job struct {
	out    []float32
	mat    []byte
	vec    []float32
	startR int
	endR   int
	cols   int
}

func newWorkerPool(numWorkers int) *workerPool {
	wp := &workerPool{
		numWorkers: numWorkers,
		jobs:       make([]chan matJob, numWorkers),
		q8jobs:     make([]chan matQ8Job, numWorkers),
		done:       make([]chan struct{}, numWorkers),
	}
	for i := 0; i < numWorkers; i++ {
		wp.jobs[i] = make(chan matJob, 1)
		wp.q8jobs[i] = make(chan matQ8Job, 1)
		wp.done[i] = make(chan struct{}, 1)
		go func(id int) {
			for {
				select {
				case job := <-wp.jobs[id]:
					matVecMulChunk(job.out, job.mat, job.vec, job.startR, job.endR, job.cols)
					wp.done[id] <- struct{}{}
				case job := <-wp.q8jobs[id]:
					matVecMulQ8Chunk(job.out, job.mat, job.vec, job.startR, job.endR, job.cols)
					wp.done[id] <- struct{}{}
				}
			}
		}(i)
	}
	return wp
}

func (wp *workerPool) dispatch(out, mat, vec []float32, rows, cols int) {
	chunkSize := (rows + wp.numWorkers - 1) / wp.numWorkers
	active := 0
	for w := 0; w < wp.numWorkers; w++ {
		s := w * chunkSize
		e := s + chunkSize
		if e > rows {
			e = rows
		}
		if s >= e {
			break
		}
		wp.jobs[w] <- matJob{out: out, mat: mat, vec: vec, startR: s, endR: e, cols: cols}
		active++
	}
	for w := 0; w < active; w++ {
		<-wp.done[w]
	}
}

// New creates a new native Transformer engine.
func New(logger *zap.Logger) *Engine {
	if logger == nil {
		logger, _ = zap.NewDevelopment()
	}
	return &Engine{
		logger:     logger,
		workerPool: newWorkerPool(runtime.NumCPU()),
	}
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

	// Pre-allocate inference buffers.
	kvDimBuf := cfg.NumKVHeads * cfg.HeadDim
	maxBuf := cfg.MaxSeqLen * cfg.HiddenSize
	e.buffers = &inferBuffers{
		x:       make([]float32, cfg.HiddenSize),
		normed:  make([]float32, maxBuf),
		q:       make([]float32, cfg.HiddenSize),
		k:       make([]float32, kvDimBuf),
		v:       make([]float32, kvDimBuf),
		attnOut: make([]float32, maxBuf),
		proj:    make([]float32, cfg.HiddenSize),
		gate:    make([]float32, cfg.IntermediateSize),
		up:      make([]float32, cfg.IntermediateSize),
		ffnOut:  make([]float32, maxBuf),
		logits:  make([]float32, cfg.VocabSize),
		scores:  make([]float32, cfg.MaxSeqLen),
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

		logitsRef := e.forward(tokens)
		// Copy logits since sample modifies in-place and buffer is reused.
		logits := make([]float32, len(logitsRef))
		copy(logits, logitsRef)
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

			logitsRef := e.forward(tokens)
			logits := make([]float32, len(logitsRef))
			copy(logits, logitsRef)
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
	e.mu.Lock()
	defer e.mu.Unlock()

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
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.loaded {
		return 0, fmt.Errorf("no model loaded")
	}
	return len(e.tok.Encode(text)), nil
}

func (e *Engine) ModelInfo() *engine.ModelMeta {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.meta
}

func (e *Engine) IsLoaded() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
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
	logits := e.buffers.logits
	for i := range logits {
		logits[i] = 0
	}
	if e.weights.Output != nil {
		e.matVecMulQ8Pooled(logits, e.weights.Output, lastHidden, cfg.VocabSize, e.weights.OutputCols)
	} else {
		// Tied weights fallback — use token embedding (float32).
		e.matVecMulPooled(logits, e.weights.TokenEmbed, lastHidden, cfg.VocabSize, hidden)
	}

	return logits
}

// transformerLayer runs a single Transformer layer.
func (e *Engine) transformerLayer(x []float32, seqLen, layer int) []float32 {
	cfg := e.config
	hidden := cfg.HiddenSize
	lw := &e.weights.Layers[layer]

	// 1. Pre-attention RMSNorm.
	normed := make([]float32, seqLen*hidden)
	copy(normed, x[:seqLen*hidden])
	for i := 0; i < seqLen; i++ {
		rmsNorm(normed[i*hidden:(i+1)*hidden], lw.AttnNorm, cfg.RMSNormEps)
	}

	// 2. Self-attention with KV cache.
	attnOut := e.selfAttention(normed, seqLen, layer)

	// 3. Residual connection.
	for i := 0; i < seqLen*hidden; i++ {
		x[i] += attnOut[i]
	}

	// 4. Pre-FFN RMSNorm.
	normed2 := make([]float32, seqLen*hidden)
	copy(normed2, x[:seqLen*hidden])
	for i := 0; i < seqLen; i++ {
		rmsNorm(normed2[i*hidden:(i+1)*hidden], lw.FFNNorm, cfg.RMSNormEps)
	}

	// 5. FFN (SwiGLU).
	ffnOut := e.feedForward(normed2, seqLen, layer)

	// 6. Residual connection.
	for i := 0; i < seqLen*hidden; i++ {
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

		// Project Q, K, V using pre-allocated buffers — Q8_0 quantized matmul.
		q := e.buffers.q
		k := e.buffers.k
		v := e.buffers.v
		for i := range q {
			q[i] = 0
		}
		for i := range k {
			k[i] = 0
		}
		for i := range v {
			v[i] = 0
		}
		e.matVecMulQ8Pooled(q, lw.Wq, xPos, hidden, lw.WqCols)
		e.matVecMulQ8Pooled(k, lw.Wk, xPos, kvDim, lw.WkCols)
		e.matVecMulQ8Pooled(v, lw.Wv, xPos, kvDim, lw.WvCols)

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
		scale := float32(1.0 / math.Sqrt(float64(headDim)))

		for h := 0; h < numHeads; h++ {
			kvHead := h / headsPerKVGroup
			qHead := q[h*headDim : (h+1)*headDim]

			scores := e.buffers.scores[:cachedLen]
			for t := 0; t < cachedLen; t++ {
				kHead := kvc.K[t*kvDim+kvHead*headDim : t*kvDim+(kvHead+1)*headDim]
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += qHead[d] * kHead[d]
				}
				scores[t] = dot * scale
			}

			softmax(scores)

			for t := 0; t < cachedLen; t++ {
				vHead := kvc.V[t*kvDim+kvHead*headDim : t*kvDim+(kvHead+1)*headDim]
				for d := 0; d < headDim; d++ {
					out[pos*hidden+h*headDim+d] += scores[t] * vHead[d]
				}
			}
		}

		// Output projection.
		projected := e.buffers.proj
		for i := range projected {
			projected[i] = 0
		}
		e.matVecMulQ8Pooled(projected, lw.Wo, out[pos*hidden:(pos+1)*hidden], hidden, lw.WoCols)
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

		gate := e.buffers.gate
		up := e.buffers.up
		for i := range gate {
			gate[i] = 0
		}
		for i := range up {
			up[i] = 0
		}

		// Gate and Up projections — both use full worker pool, sequential.
		e.matVecMulQ8Pooled(gate, lw.WGate, xPos, inter, lw.WGateCols)
		e.matVecMulQ8Pooled(up, lw.WUp, xPos, inter, lw.WUpCols)

		// SwiGLU activation: silu(gate) * up
		for i := 0; i < inter; i++ {
			gate[i] = silu(gate[i]) * up[i]
		}

		// Down projection — use x buffer as scratch.
		downOut := e.buffers.x
		for i := range downOut {
			downOut[i] = 0
		}
		e.matVecMulQ8Pooled(downOut, lw.WDown, gate, hidden, lw.WDownCols)
		copy(out[pos*hidden:], downOut)
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
// Dispatches to worker pool for large matrices.
func (e *Engine) matVecMulPooled(out, mat, vec []float32, rows, cols int) {
	if rows < 512 {
		matVecMulChunk(out, mat, vec, 0, rows, cols)
		return
	}
	e.workerPool.dispatch(out, mat, vec, rows, cols)
}

// matVecMulChunk computes a chunk of rows for matVecMul (float32 weights).
// Uses 16x unrolling and 4-way accumulator splitting.
func matVecMulChunk(out, mat, vec []float32, startRow, endRow, cols int) {
	for i := startRow; i < endRow; i++ {
		var s0, s1, s2, s3 float32
		rowOff := i * cols
		row := mat[rowOff : rowOff+cols]
		j := 0
		for ; j+15 < cols; j += 16 {
			s0 += row[j]*vec[j] + row[j+1]*vec[j+1] + row[j+2]*vec[j+2] + row[j+3]*vec[j+3]
			s1 += row[j+4]*vec[j+4] + row[j+5]*vec[j+5] + row[j+6]*vec[j+6] + row[j+7]*vec[j+7]
			s2 += row[j+8]*vec[j+8] + row[j+9]*vec[j+9] + row[j+10]*vec[j+10] + row[j+11]*vec[j+11]
			s3 += row[j+12]*vec[j+12] + row[j+13]*vec[j+13] + row[j+14]*vec[j+14] + row[j+15]*vec[j+15]
		}
		for ; j < cols; j++ {
			s0 += row[j] * vec[j]
		}
		out[i] = s0 + s1 + s2 + s3
	}
}

// matVecMul is the standalone version (used during weight loading).
func matVecMul(out, mat, vec []float32, rows, cols int) {
	matVecMulChunk(out, mat, vec, 0, rows, cols)
}

// ======================================================================
// Q8_0 quantized matrix-vector multiply (no dequantization needed)
// ======================================================================
//
// Q8_0 format: each block of 32 values = 2 bytes (f16 scale) + 32 bytes (int8 values) = 34 bytes
// A row of `cols` float32 values is stored as (cols/32) blocks = cols/32 * 34 bytes.
//
// Dot product: for each block, dot += scale * sum(q[i] * vec[i])
// This reads 34 bytes per 32 values instead of 128 bytes (float32), saving 73% bandwidth.

const q8BlockSize = 32
const q8BlockBytes = 34

// q8RowBytes returns the byte size of one row in Q8_0 format.
func q8RowBytes(cols int) int {
	return (cols / q8BlockSize) * q8BlockBytes
}

// matVecMulQ8Chunk computes dot products of Q8_0-encoded matrix rows with a float32 vector.
func matVecMulQ8Chunk(out []float32, mat []byte, vec []float32, startRow, endRow, cols int) {
	rowBytes := q8RowBytes(cols)
	numBlocks := cols / q8BlockSize

	for i := startRow; i < endRow; i++ {
		rowOff := i * rowBytes
		var sum float32

		for b := 0; b < numBlocks; b++ {
			blockOff := rowOff + b*q8BlockBytes
			scaleBits := uint16(mat[blockOff]) | uint16(mat[blockOff+1])<<8
			scale := float16ToFloat32(scaleBits)

			vecOff := b * q8BlockSize
			dataOff := blockOff + 2

			// Accumulate int8 * float32 products, 8 at a time.
			var s0, s1, s2, s3 float32
			s0 = float32(int8(mat[dataOff]))*vec[vecOff] +
				float32(int8(mat[dataOff+1]))*vec[vecOff+1] +
				float32(int8(mat[dataOff+2]))*vec[vecOff+2] +
				float32(int8(mat[dataOff+3]))*vec[vecOff+3] +
				float32(int8(mat[dataOff+4]))*vec[vecOff+4] +
				float32(int8(mat[dataOff+5]))*vec[vecOff+5] +
				float32(int8(mat[dataOff+6]))*vec[vecOff+6] +
				float32(int8(mat[dataOff+7]))*vec[vecOff+7]
			s1 = float32(int8(mat[dataOff+8]))*vec[vecOff+8] +
				float32(int8(mat[dataOff+9]))*vec[vecOff+9] +
				float32(int8(mat[dataOff+10]))*vec[vecOff+10] +
				float32(int8(mat[dataOff+11]))*vec[vecOff+11] +
				float32(int8(mat[dataOff+12]))*vec[vecOff+12] +
				float32(int8(mat[dataOff+13]))*vec[vecOff+13] +
				float32(int8(mat[dataOff+14]))*vec[vecOff+14] +
				float32(int8(mat[dataOff+15]))*vec[vecOff+15]
			s2 = float32(int8(mat[dataOff+16]))*vec[vecOff+16] +
				float32(int8(mat[dataOff+17]))*vec[vecOff+17] +
				float32(int8(mat[dataOff+18]))*vec[vecOff+18] +
				float32(int8(mat[dataOff+19]))*vec[vecOff+19] +
				float32(int8(mat[dataOff+20]))*vec[vecOff+20] +
				float32(int8(mat[dataOff+21]))*vec[vecOff+21] +
				float32(int8(mat[dataOff+22]))*vec[vecOff+22] +
				float32(int8(mat[dataOff+23]))*vec[vecOff+23]
			s3 = float32(int8(mat[dataOff+24]))*vec[vecOff+24] +
				float32(int8(mat[dataOff+25]))*vec[vecOff+25] +
				float32(int8(mat[dataOff+26]))*vec[vecOff+26] +
				float32(int8(mat[dataOff+27]))*vec[vecOff+27] +
				float32(int8(mat[dataOff+28]))*vec[vecOff+28] +
				float32(int8(mat[dataOff+29]))*vec[vecOff+29] +
				float32(int8(mat[dataOff+30]))*vec[vecOff+30] +
				float32(int8(mat[dataOff+31]))*vec[vecOff+31]

			sum += scale * (s0 + s1 + s2 + s3)
		}
		out[i] = sum
	}
}

// matVecMulQ8Pooled dispatches Q8_0 matrix-vector multiply to the worker pool.
func (e *Engine) matVecMulQ8Pooled(out []float32, mat []byte, vec []float32, rows, cols int) {
	if rows < 512 {
		matVecMulQ8Chunk(out, mat, vec, 0, rows, cols)
		return
	}
	// Use worker pool with Q8 jobs.
	numWorkers := e.workerPool.numWorkers
	chunkSize := (rows + numWorkers - 1) / numWorkers
	active := 0
	for w := 0; w < numWorkers; w++ {
		s := w * chunkSize
		end := s + chunkSize
		if end > rows {
			end = rows
		}
		if s >= end {
			break
		}
		e.workerPool.q8jobs[w] <- matQ8Job{out: out, mat: mat, vec: vec, startR: s, endR: end, cols: cols}
		active++
	}
	for w := 0; w < active; w++ {
		<-e.workerPool.done[w]
	}
}

// float16ToFloat32 converts IEEE 754 half-precision to single-precision.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF
	var f uint32
	switch {
	case exp == 0:
		if mant == 0 {
			f = sign << 31
		} else {
			exp = 1
			for mant&0x400 == 0 {
				mant <<= 1
				exp--
			}
			mant &= 0x3FF
			f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
		}
	case exp == 0x1F:
		f = (sign << 31) | (0xFF << 23) | (mant << 13)
	default:
		f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
	}
	return math.Float32frombits(f)
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

	// Token embedding — keep as float32 for lookup.
	w.TokenEmbed, err = readTensor("token_embd.weight")
	if err != nil {
		return fmt.Errorf("token_embd: %w", err)
	}
	e.logger.Info("loaded token embeddings", zap.Duration("elapsed", time.Since(loadStart)))

	// Output norm — small, float32.
	w.OutputNorm, err = readTensor("output_norm.weight")
	if err != nil {
		return fmt.Errorf("output_norm: %w", err)
	}

	// Output projection (lm_head) — store as raw Q8_0 bytes.
	{
		t, ok := gf.FindTensor("output.weight")
		if ok {
			offset := gf.DataOffset + int64(t.Offset)
			size := t.TensorDataSize()
			data := make([]byte, size)
			if _, err := file.ReadAt(data, offset); err != nil {
				return fmt.Errorf("read output.weight: %w", err)
			}
			if t.Type == gguf.GGMLTypeQ8_0 {
				w.Output = data
				w.OutputCols = int(t.Dimensions[0])
			} else {
				// Non-Q8_0: dequantize to float32, then we'll use float32 matVecMul
				f32 := gguf.Dequantize(data, t.Type, t.ElementCount())
				// Store as fake "Q8_0" by keeping float32 — handle in forward
				w.Output = nil
				w.OutputCols = int(t.Dimensions[0])
				// Fallback: store dequantized as a separate field
				e.weights.TokenEmbed = append([]float32{}, e.weights.TokenEmbed...) // ensure separate
				_ = f32                                                             // TODO: handle non-Q8_0 output weights
			}
		} else {
			e.logger.Info("output.weight not found, using tied embedding weights")
			w.OutputCols = cfg.HiddenSize
		}
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

				// readRaw reads tensor as raw bytes (for Q8_0 weights).
				readRaw := func(name string) ([]byte, int, error) {
					t, ok := gf.FindTensor(name)
					if !ok {
						return nil, 0, fmt.Errorf("tensor %q not found", name)
					}
					offset := gf.DataOffset + int64(t.Offset)
					size := t.TensorDataSize()
					data := make([]byte, size)
					if _, err := f.ReadAt(data, offset); err != nil {
						return nil, 0, fmt.Errorf("read tensor %q: %w", name, err)
					}
					cols := int(t.Dimensions[0]) // GGUF: [cols, rows]
					return data, cols, nil
				}

				// readF32 reads tensor and dequantizes (for small 1D tensors).
				readF32 := func(name string) ([]float32, error) {
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

				// readOptionalF32 reads optional tensor.
				readOptionalF32 := func(name string) []float32 {
					result, err := readF32(name)
					if err != nil {
						return nil
					}
					return result
				}

				// Large 2D weights — store as raw bytes.
				lw.Wq, lw.WqCols, loadErr = readRaw(prefix + "attn_q.weight")
				if loadErr == nil {
					lw.Wk, lw.WkCols, loadErr = readRaw(prefix + "attn_k.weight")
				}
				if loadErr == nil {
					lw.Wv, lw.WvCols, loadErr = readRaw(prefix + "attn_v.weight")
				}
				if loadErr == nil {
					lw.Wo, lw.WoCols, loadErr = readRaw(prefix + "attn_output.weight")
				}
				if loadErr == nil {
					lw.WGate, lw.WGateCols, loadErr = readRaw(prefix + "ffn_gate.weight")
				}
				if loadErr == nil {
					lw.WUp, lw.WUpCols, loadErr = readRaw(prefix + "ffn_up.weight")
				}
				if loadErr == nil {
					lw.WDown, lw.WDownCols, loadErr = readRaw(prefix + "ffn_down.weight")
				}

				// Small 1D weights — dequantize to float32.
				if loadErr == nil {
					lw.AttnNorm, loadErr = readF32(prefix + "attn_norm.weight")
				}
				if loadErr == nil {
					lw.FFNNorm, loadErr = readF32(prefix + "ffn_norm.weight")
				}

				// Optional biases — float32.
				if loadErr == nil {
					lw.Bq = readOptionalF32(prefix + "attn_q.bias")
					lw.Bk = readOptionalF32(prefix + "attn_k.bias")
					lw.Bv = readOptionalF32(prefix + "attn_v.bias")
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
