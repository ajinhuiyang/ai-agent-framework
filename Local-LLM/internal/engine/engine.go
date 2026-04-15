// Package engine defines the interface for LLM inference backends.
//
// Implementations include:
//   - llamacpp: Production backend using llama.cpp via CGO bindings
//   - mock:     Development/testing backend with simulated responses
package engine

import (
	"context"

	"github.com/your-org/local-llm/internal/domain"
)

// Engine is the core inference interface. All backends must implement this.
type Engine interface {
	// Load loads a model from the given file path.
	Load(ctx context.Context, modelPath string, opts LoadOptions) error

	// Unload releases the currently loaded model from memory.
	Unload() error

	// Predict generates text from a prompt (non-streaming).
	Predict(ctx context.Context, req domain.InferenceRequest) (PredictResult, error)

	// PredictStream generates text token by token.
	PredictStream(ctx context.Context, req domain.InferenceRequest) (<-chan StreamToken, error)

	// Embed generates embeddings for the given texts.
	Embed(ctx context.Context, texts []string) ([][]float32, error)

	// TokenCount returns the number of tokens in the given text.
	TokenCount(ctx context.Context, text string) (int, error)

	// ModelInfo returns information about the currently loaded model.
	ModelInfo() *ModelMeta

	// IsLoaded returns whether a model is currently loaded.
	IsLoaded() bool
}

// LoadOptions configures how a model is loaded.
type LoadOptions struct {
	NumCtx    int // Context window size
	NumGPU    int // Number of GPU layers
	NumThread int // Number of CPU threads
	BatchSize int // Batch size for prompt processing
}

// PredictResult is the result of a non-streaming prediction.
type PredictResult struct {
	Text             string // Generated text
	PromptTokens     int    // Number of tokens in the prompt
	CompletionTokens int    // Number of generated tokens
	TotalDuration    int64  // Nanoseconds total
	LoadDuration     int64  // Nanoseconds for model loading
	PromptDuration   int64  // Nanoseconds for prompt processing
	EvalDuration     int64  // Nanoseconds for token generation
}

// StreamToken is a single token in a streaming response.
type StreamToken struct {
	Text string // Decoded token text
	Done bool   // Whether this is the last token
	Err  error  // Non-nil if an error occurred
}

// ModelMeta holds metadata about a loaded model.
type ModelMeta struct {
	Name          string
	Path          string
	Family        string
	ParameterSize string
	QuantLevel    string
	ContextLength int
	EmbeddingSize int
	FileSizeBytes int64
}
