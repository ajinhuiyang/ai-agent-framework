// Package qwen implements the LLM provider for Alibaba Qwen (通义千问 DashScope).
// Qwen DashScope provides an OpenAI-compatible API, so this is a thin wrapper.
package qwen

import (
	"context"

	"github.com/your-org/llm-generation/internal/domain"
	"github.com/your-org/llm-generation/internal/llm"
	"github.com/your-org/llm-generation/internal/llm/openai"
)

// Provider wraps the OpenAI-compatible provider for Qwen/DashScope.
type Provider struct {
	inner *openai.Provider
}

// New creates a new Qwen provider.
// DashScope OpenAI-compatible endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1
func New(apiKey, baseURL, model string) *Provider {
	if baseURL == "" {
		baseURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
	}
	if model == "" {
		model = "qwen-turbo"
	}
	return &Provider{
		inner: openai.New(apiKey, baseURL, model, "qwen"),
	}
}

func (p *Provider) Complete(ctx context.Context, messages []domain.Message, config *domain.GenerateConfig) (*llm.CompletionResult, error) {
	return p.inner.Complete(ctx, messages, config)
}

func (p *Provider) CompleteStream(ctx context.Context, messages []domain.Message, config *domain.GenerateConfig) (<-chan llm.StreamEvent, error) {
	return p.inner.CompleteStream(ctx, messages, config)
}

func (p *Provider) Name() string                          { return "qwen" }
func (p *Provider) Models() []string                      { return p.inner.Models() }
func (p *Provider) HealthCheck(ctx context.Context) error { return p.inner.HealthCheck(ctx) }
