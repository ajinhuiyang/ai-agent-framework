// Package domain defines the core data types for the Local-LLM inference service.
// All API types are designed to be Ollama-compatible so that LLM-Generation
// can call this service exactly the same way it calls a real Ollama instance.
package domain

import "time"

// ---------------------------------------------------------------------------
// Ollama /api/chat
// ---------------------------------------------------------------------------

// ChatRequest mirrors Ollama POST /api/chat request body.
type ChatRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
	Stream   *bool         `json:"stream,omitempty"` // default true in Ollama
	Format   string        `json:"format,omitempty"` // "json" for JSON mode
	Options  *Options      `json:"options,omitempty"`

	// Keep-alive duration (not implemented but accepted for compat).
	KeepAlive string `json:"keep_alive,omitempty"`
}

// ChatMessage is a single message in a chat conversation.
type ChatMessage struct {
	Role    string `json:"role"` // "system", "user", "assistant"
	Content string `json:"content"`
}

// ChatResponse mirrors Ollama /api/chat response (non-streaming full response,
// or individual streaming chunks).
type ChatResponse struct {
	Model           string      `json:"model"`
	CreatedAt       time.Time   `json:"created_at"`
	Message         ChatMessage `json:"message"`
	Done            bool        `json:"done"`
	DoneReason      string      `json:"done_reason,omitempty"`
	TotalDuration   int64       `json:"total_duration,omitempty"` // nanoseconds
	LoadDuration    int64       `json:"load_duration,omitempty"`  // nanoseconds
	PromptEvalCount int         `json:"prompt_eval_count,omitempty"`
	EvalCount       int         `json:"eval_count,omitempty"`
	EvalDuration    int64       `json:"eval_duration,omitempty"` // nanoseconds
}

// ---------------------------------------------------------------------------
// Ollama /api/generate
// ---------------------------------------------------------------------------

// GenerateRequest mirrors Ollama POST /api/generate request body.
type GenerateRequest struct {
	Model   string   `json:"model"`
	Prompt  string   `json:"prompt"`
	System  string   `json:"system,omitempty"`
	Stream  *bool    `json:"stream,omitempty"`
	Format  string   `json:"format,omitempty"`
	Options *Options `json:"options,omitempty"`
	Context []int    `json:"context,omitempty"` // previous token context for continuation
}

// GenerateResponse mirrors Ollama /api/generate response.
type GenerateResponse struct {
	Model           string    `json:"model"`
	CreatedAt       time.Time `json:"created_at"`
	Response        string    `json:"response"`
	Done            bool      `json:"done"`
	DoneReason      string    `json:"done_reason,omitempty"`
	Context         []int     `json:"context,omitempty"`
	TotalDuration   int64     `json:"total_duration,omitempty"`
	LoadDuration    int64     `json:"load_duration,omitempty"`
	PromptEvalCount int       `json:"prompt_eval_count,omitempty"`
	EvalCount       int       `json:"eval_count,omitempty"`
	EvalDuration    int64     `json:"eval_duration,omitempty"`
}

// ---------------------------------------------------------------------------
// Ollama /api/embed
// ---------------------------------------------------------------------------

// EmbedRequest mirrors Ollama POST /api/embed request body.
type EmbedRequest struct {
	Model string      `json:"model"`
	Input interface{} `json:"input"` // string or []string
}

// EmbedResponse mirrors Ollama /api/embed response.
type EmbedResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float32 `json:"embeddings"`
}

// ---------------------------------------------------------------------------
// Ollama /api/tags (list models)
// ---------------------------------------------------------------------------

// TagsResponse mirrors Ollama GET /api/tags response.
type TagsResponse struct {
	Models []ModelInfo `json:"models"`
}

// ModelInfo describes a locally available model.
type ModelInfo struct {
	Name       string    `json:"name"`
	Model      string    `json:"model"`
	ModifiedAt time.Time `json:"modified_at"`
	Size       int64     `json:"size"`
	Digest     string    `json:"digest"`
	Details    Details   `json:"details"`
}

// Details holds model metadata.
type Details struct {
	ParentModel       string   `json:"parent_model"`
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

// ---------------------------------------------------------------------------
// Ollama /api/show (model info)
// ---------------------------------------------------------------------------

// ShowRequest mirrors Ollama POST /api/show request.
type ShowRequest struct {
	Name string `json:"name"`
}

// ShowResponse mirrors Ollama /api/show response.
type ShowResponse struct {
	ModelFile  string  `json:"modelfile"`
	Parameters string  `json:"parameters"`
	Template   string  `json:"template"`
	Details    Details `json:"details"`
}

// ---------------------------------------------------------------------------
// Shared: Generation Options (compatible with Ollama)
// ---------------------------------------------------------------------------

// Options holds generation parameters. Field names match Ollama's API.
type Options struct {
	// Sampling
	Temperature   float64 `json:"temperature,omitempty"`
	TopP          float64 `json:"top_p,omitempty"`
	TopK          int     `json:"top_k,omitempty"`
	RepeatPenalty float64 `json:"repeat_penalty,omitempty"`
	Seed          int     `json:"seed,omitempty"`

	// Length
	NumPredict int `json:"num_predict,omitempty"` // max tokens to generate
	NumCtx     int `json:"num_ctx,omitempty"`     // context window size

	// Stop sequences
	Stop []string `json:"stop,omitempty"`

	// Hardware
	NumGPU    int `json:"num_gpu,omitempty"`
	NumThread int `json:"num_thread,omitempty"`
}

// ---------------------------------------------------------------------------
// Internal: Inference types
// ---------------------------------------------------------------------------

// InferenceRequest is the engine-internal request (unified from chat/generate).
type InferenceRequest struct {
	Prompt  string  // Full formatted prompt text
	Options Options // Sampling/generation parameters
	Stream  bool    // Whether to stream tokens
}

// Token represents a single generated token.
type Token struct {
	Text string // Decoded token text
	ID   int    // Token ID (if available)
}
