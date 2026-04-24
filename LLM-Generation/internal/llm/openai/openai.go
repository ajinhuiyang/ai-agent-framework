// Package openai implements the LLM provider using the OpenAI-compatible API.
// Works with OpenAI, Ollama (OpenAI-compat mode), vLLM, Azure, etc.
package openai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	goai "github.com/sashabaranov/go-openai"

	"github.com/your-org/llm-generation/internal/domain"
	"github.com/your-org/llm-generation/internal/llm"
)

// Provider implements llm.Provider using the OpenAI chat completions API.
type Provider struct {
	client *goai.Client
	model  string
	name   string
}

// New creates a new OpenAI-compatible provider.
func New(apiKey, baseURL, model, name string) *Provider {
	config := goai.DefaultConfig(apiKey)
	if baseURL != "" {
		config.BaseURL = baseURL
	}
	// Local LLM inference can take minutes; set a generous timeout.
	config.HTTPClient = &http.Client{
		Timeout: 10 * time.Minute,
	}
	if name == "" {
		name = "openai"
	}
	return &Provider{
		client: goai.NewClientWithConfig(config),
		model:  model,
		name:   name,
	}
}

func (p *Provider) Complete(ctx context.Context, messages []domain.Message, config *domain.GenerateConfig) (*llm.CompletionResult, error) {
	model := p.model
	if config != nil && config.Model != "" {
		model = config.Model
	}

	req := goai.ChatCompletionRequest{
		Model:    model,
		Messages: toOpenAIMessages(messages),
	}

	if config != nil {
		if config.Temperature > 0 {
			req.Temperature = float32(config.Temperature)
		}
		if config.MaxTokens > 0 {
			req.MaxTokens = config.MaxTokens
		}
		if config.TopP > 0 {
			req.TopP = float32(config.TopP)
		}
		if len(config.StopWords) > 0 {
			req.Stop = config.StopWords
		}
	}

	resp, err := p.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("openai completion failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("openai returned empty choices")
	}

	return &llm.CompletionResult{
		Content:      resp.Choices[0].Message.Content,
		Model:        resp.Model,
		FinishReason: string(resp.Choices[0].FinishReason),
		Usage: domain.Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}, nil
}

func (p *Provider) CompleteStream(ctx context.Context, messages []domain.Message, config *domain.GenerateConfig) (<-chan llm.StreamEvent, error) {
	model := p.model
	if config != nil && config.Model != "" {
		model = config.Model
	}

	req := goai.ChatCompletionRequest{
		Model:    model,
		Messages: toOpenAIMessages(messages),
		Stream:   true,
	}

	if config != nil {
		if config.Temperature > 0 {
			req.Temperature = float32(config.Temperature)
		}
		if config.MaxTokens > 0 {
			req.MaxTokens = config.MaxTokens
		}
		if config.TopP > 0 {
			req.TopP = float32(config.TopP)
		}
		if len(config.StopWords) > 0 {
			req.Stop = config.StopWords
		}
	}

	stream, err := p.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("openai stream failed: %w", err)
	}

	ch := make(chan llm.StreamEvent, 64)
	go func() {
		defer close(ch)
		defer stream.Close()

		for {
			resp, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				ch <- llm.StreamEvent{Done: true, FinishReason: "stop"}
				return
			}
			if err != nil {
				ch <- llm.StreamEvent{Err: err, Done: true}
				return
			}

			if len(resp.Choices) > 0 {
				choice := resp.Choices[0]
				event := llm.StreamEvent{
					Content: choice.Delta.Content,
				}
				if choice.FinishReason != "" {
					event.Done = true
					event.FinishReason = string(choice.FinishReason)
				}
				ch <- event
				if event.Done {
					return
				}
			}
		}
	}()

	return ch, nil
}

func (p *Provider) Name() string     { return p.name }
func (p *Provider) Models() []string { return []string{p.model} }

func (p *Provider) HealthCheck(ctx context.Context) error {
	_, err := p.client.ListModels(ctx)
	if err != nil {
		return fmt.Errorf("openai health check failed: %w", err)
	}
	return nil
}

func toOpenAIMessages(messages []domain.Message) []goai.ChatCompletionMessage {
	msgs := make([]goai.ChatCompletionMessage, len(messages))
	for i, m := range messages {
		msgs[i] = goai.ChatCompletionMessage{
			Role:    m.Role,
			Content: m.Content,
		}
	}
	return msgs
}
