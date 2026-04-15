// Package zhipu implements the LLM provider for ZhipuAI (智谱清言 GLM).
// ZhipuAI provides an OpenAI-compatible API, so this is a thin wrapper
// around the OpenAI provider with ZhipuAI-specific defaults.
package zhipu

import (
	"context"

	"github.com/your-org/llm-generation/internal/domain"
	"github.com/your-org/llm-generation/internal/llm"
	"github.com/your-org/llm-generation/internal/llm/openai"
)

// Provider wraps the OpenAI-compatible provider for ZhipuAI.
type Provider struct {
	inner *openai.Provider
}

// New creates a new ZhipuAI provider.
// ZhipuAI uses OpenAI-compatible API at https://open.bigmodel.cn/api/paas/v4
func New(apiKey, baseURL, model string) *Provider {
	if baseURL == "" {
		baseURL = "https://open.bigmodel.cn/api/paas/v4"
	}
	if model == "" {
		model = "glm-4-flash"
	}
	return &Provider{
		inner: openai.New(apiKey, baseURL, model, "zhipu"),
	}
}

func (p *Provider) Complete(ctx context.Context, messages []domain.Message, config *domain.GenerateConfig) (*llm.CompletionResult, error) {
	return p.inner.Complete(ctx, messages, config)
}

func (p *Provider) CompleteStream(ctx context.Context, messages []domain.Message, config *domain.GenerateConfig) (<-chan llm.StreamEvent, error) {
	return p.inner.CompleteStream(ctx, messages, config)
}

func (p *Provider) Name() string                          { return "zhipu" }
func (p *Provider) Models() []string                      { return p.inner.Models() }
func (p *Provider) HealthCheck(ctx context.Context) error { return p.inner.HealthCheck(ctx) }
