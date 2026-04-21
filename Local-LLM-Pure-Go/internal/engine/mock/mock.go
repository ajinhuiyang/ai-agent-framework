// Package mock provides a mock inference engine for development and testing.
// It generates simulated responses without loading any real model, useful for
// verifying the API layer and integration with LLM-Generation.
package mock

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/your-org/local-llm/internal/domain"
	"github.com/your-org/local-llm/internal/engine"
)

// Engine is a mock inference engine.
type Engine struct {
	mu     sync.RWMutex
	loaded bool
	meta   *engine.ModelMeta
}

// New creates a new mock engine.
func New() *Engine {
	return &Engine{}
}

func (e *Engine) Load(_ context.Context, modelPath string, opts engine.LoadOptions) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Simulate load delay.
	time.Sleep(100 * time.Millisecond)

	e.loaded = true
	e.meta = &engine.ModelMeta{
		Name:          "mock-model",
		Path:          modelPath,
		Family:        "mock",
		ParameterSize: "7B",
		QuantLevel:    "Q4_K_M",
		ContextLength: opts.NumCtx,
		EmbeddingSize: 4096,
		FileSizeBytes: 4_000_000_000,
	}
	return nil
}

func (e *Engine) Unload() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.loaded = false
	e.meta = nil
	return nil
}

func (e *Engine) Predict(ctx context.Context, req domain.InferenceRequest) (engine.PredictResult, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if !e.loaded {
		return engine.PredictResult{}, fmt.Errorf("no model loaded")
	}

	start := time.Now()

	// Generate a mock response based on the prompt.
	response := e.generateResponse(req.Prompt, req.Options)

	// Simulate token generation delay.
	tokenCount := len([]rune(response)) / 2 // rough estimate
	time.Sleep(time.Duration(tokenCount) * 5 * time.Millisecond)

	elapsed := time.Since(start)

	promptTokens := e.estimateTokens(req.Prompt)

	return engine.PredictResult{
		Text:             response,
		PromptTokens:     promptTokens,
		CompletionTokens: tokenCount,
		TotalDuration:    elapsed.Nanoseconds(),
		PromptDuration:   elapsed.Nanoseconds() / 3,
		EvalDuration:     elapsed.Nanoseconds() * 2 / 3,
	}, nil
}

func (e *Engine) PredictStream(ctx context.Context, req domain.InferenceRequest) (<-chan engine.StreamToken, error) {
	e.mu.RLock()
	if !e.loaded {
		e.mu.RUnlock()
		return nil, fmt.Errorf("no model loaded")
	}
	e.mu.RUnlock()

	response := e.generateResponse(req.Prompt, req.Options)
	words := strings.Fields(response)

	ch := make(chan engine.StreamToken, 32)
	go func() {
		defer close(ch)
		for i, word := range words {
			select {
			case <-ctx.Done():
				ch <- engine.StreamToken{Err: ctx.Err(), Done: true}
				return
			default:
			}

			text := word
			if i < len(words)-1 {
				text += " "
			}

			// Simulate token generation delay.
			time.Sleep(time.Duration(20+rand.Intn(30)) * time.Millisecond)

			ch <- engine.StreamToken{
				Text: text,
				Done: i == len(words)-1,
			}
		}
	}()

	return ch, nil
}

func (e *Engine) Embed(_ context.Context, texts []string) ([][]float32, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if !e.loaded {
		return nil, fmt.Errorf("no model loaded")
	}

	dim := e.meta.EmbeddingSize
	if dim == 0 {
		dim = 4096
	}

	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		vec := make([]float32, dim)
		// Generate deterministic pseudo-embeddings based on text content.
		seed := int64(0)
		for _, r := range text {
			seed = seed*31 + int64(r)
		}
		rng := rand.New(rand.NewSource(seed))
		var norm float64
		for j := range vec {
			vec[j] = rng.Float32()*2 - 1
			norm += float64(vec[j]) * float64(vec[j])
		}
		// L2 normalize.
		norm = math.Sqrt(norm)
		for j := range vec {
			vec[j] /= float32(norm)
		}
		embeddings[i] = vec
	}

	return embeddings, nil
}

func (e *Engine) TokenCount(_ context.Context, text string) (int, error) {
	return e.estimateTokens(text), nil
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

// estimateTokens provides a rough token count estimation.
func (e *Engine) estimateTokens(text string) int {
	// Rough heuristic: ~1.5 tokens per CJK character, ~0.75 per English word
	runes := []rune(text)
	count := 0
	for _, r := range runes {
		if r > 0x4E00 && r < 0x9FFF {
			count += 2 // CJK character ≈ 1-2 tokens
		} else if r == ' ' || r == '\n' {
			count++ // word boundary
		}
	}
	if count == 0 {
		count = len(strings.Fields(text))
	}
	return count
}

// generateResponse creates a simulated response.
func (e *Engine) generateResponse(prompt string, opts domain.Options) string {
	lower := strings.ToLower(prompt)

	// Simple pattern matching for mock responses.
	switch {
	case strings.Contains(lower, "hello") || strings.Contains(lower, "你好"):
		return "你好！我是本地部署的 AI 助手，基于 Local-LLM 推理引擎运行。有什么我可以帮你的吗？"
	case strings.Contains(lower, "weather") || strings.Contains(lower, "天气"):
		return "根据我的知识，北京今天多云转晴，气温 12-22°C，东北风 3-4 级，适合户外活动。需要了解其他城市的天气吗？"
	case strings.Contains(lower, "json"):
		return `{"answer": "这是一个 JSON 格式的模拟响应", "confidence": 0.95, "source": "mock-engine"}`
	default:
		return fmt.Sprintf("这是来自 Local-LLM 模拟引擎的回复。我收到了你的请求（%d 个字符）。"+
			"在生产环境中，这里会由 llama.cpp 引擎处理你的 GGUF 模型推理请求。"+
			"当前使用的采样参数：temperature=%.2f, top_p=%.2f, top_k=%d。",
			len([]rune(prompt)),
			opts.Temperature, opts.TopP, opts.TopK)
	}
}
