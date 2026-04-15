// Package llm defines the unified interface for LLM providers.
package llm

import (
	"context"

	"github.com/your-org/llm-generation/internal/domain"
)

// Provider is the interface that all LLM backends must implement.
type Provider interface {
	// Complete performs a chat completion request.
	Complete(ctx context.Context, messages []domain.Message, config *domain.GenerateConfig) (*CompletionResult, error)

	// CompleteStream performs a streaming chat completion request.
	CompleteStream(ctx context.Context, messages []domain.Message, config *domain.GenerateConfig) (<-chan StreamEvent, error)

	// Name returns the provider identifier.
	Name() string

	// Models returns the list of available models.
	Models() []string

	// HealthCheck verifies the provider is reachable.
	HealthCheck(ctx context.Context) error
}

// CompletionResult is the result of a non-streaming completion.
type CompletionResult struct {
	Content      string
	Model        string
	FinishReason string
	Usage        domain.Usage
}

// StreamEvent is a single event in a streaming completion.
type StreamEvent struct {
	Content      string
	Done         bool
	FinishReason string
	Usage        *domain.Usage // Only set on final event.
	Err          error         // Non-nil if an error occurred.
}
