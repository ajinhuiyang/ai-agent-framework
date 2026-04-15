// Package llm defines the LLM provider abstraction and message types.
package llm

import "context"

// Role constants for chat messages.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
)

// Message represents a chat message for the LLM.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// CompletionRequest contains parameters for an LLM completion call.
type CompletionRequest struct {
	Messages    []Message `json:"messages"`
	Model       string    `json:"model,omitempty"`
	Temperature float64   `json:"temperature,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	TopP        float64   `json:"top_p,omitempty"`
	Stop        []string  `json:"stop,omitempty"`

	// ResponseFormat can request structured output (e.g., JSON mode)
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

// ResponseFormat specifies the desired output format.
type ResponseFormat struct {
	Type string `json:"type"` // "json_object" or "text"
}

// CompletionResponse contains the LLM's response.
type CompletionResponse struct {
	Content      string `json:"content"`
	Model        string `json:"model"`
	FinishReason string `json:"finish_reason"`
	Usage        Usage  `json:"usage"`
}

// Usage contains token usage information.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Provider is the interface that all LLM backends must implement.
type Provider interface {
	// Complete sends a chat completion request and returns the response.
	Complete(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)

	// CompleteJSON is a convenience method that forces JSON output.
	CompleteJSON(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)

	// Name returns the provider's identifier.
	Name() string

	// HealthCheck verifies the provider is reachable.
	HealthCheck(ctx context.Context) error
}
