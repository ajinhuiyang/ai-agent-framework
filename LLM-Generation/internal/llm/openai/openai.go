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
	// 不设置 http.Client.Timeout: 它覆盖整个请求生命周期（包括读取流式 body），
	// 会在本地大模型推理时间较长时强制断开 SSE 流。
	// 超时由调用方的 context.WithTimeout 控制。
	config.HTTPClient = &http.Client{
		Transport: &http.Transport{
			MaxIdleConnsPerHost: 10,
			IdleConnTimeout:     90 * time.Second,
		},
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
		// repetition_penalty → OpenAI 标准字段 frequency_penalty + presence_penalty
		// Local-LLM C++ 会读取这两个字段并转换为 repetition_penalty
		if config.RepetitionPenalty > 0 && config.RepetitionPenalty != 1.0 {
			// frequency_penalty 控制基于频率的惩罚 (OpenAI 范围 0~2)
			// presence_penalty 控制出现过的 token 的惩罚 (OpenAI 范围 -2~2)
			// 我们将 repetition_penalty 拆分: presence 控制是否惩罚，frequency 控制力度
			penalty := float32(config.RepetitionPenalty - 1.0) // 1.15 → 0.15
			req.FrequencyPenalty = penalty * 2.0               // 0.15 → 0.3
			req.PresencePenalty = penalty * 1.5                // 0.15 → 0.225
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
		// repetition_penalty → OpenAI 标准字段 (与 Complete 一致)
		if config.RepetitionPenalty > 0 && config.RepetitionPenalty != 1.0 {
			penalty := float32(config.RepetitionPenalty - 1.0)
			req.FrequencyPenalty = penalty * 2.0
			req.PresencePenalty = penalty * 1.5
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

		// 记录最后收到的 finish_reason，用于 EOF 时的兜底
		lastFinishReason := ""

		for {
			resp, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				// EOF 表示流结束。如果之前已经通过 finish_reason chunk 发送了 Done 事件，
				// 这里就不需要再发了。如果没有（某些 LLM 不发 finish_reason chunk），
				// 则发送兜底 Done 事件。
				if lastFinishReason == "" {
					lastFinishReason = "stop"
				}
				ch <- llm.StreamEvent{Done: true, FinishReason: lastFinishReason}
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
					lastFinishReason = string(choice.FinishReason)
					event.Done = true
					event.FinishReason = lastFinishReason
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
