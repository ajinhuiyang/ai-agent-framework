// Package openai implements the LLM provider using the OpenAI-compatible API.
// This works with OpenAI, Azure OpenAI, Ollama, vLLM, and any other
// service that implements the OpenAI chat completions API.
package openai

import (
	"context"
	"fmt"

	"github.com/sashabaranov/go-openai"

	"github.com/your-org/nlu/internal/llm"
)

// Provider implements llm.Provider using the OpenAI-compatible API.
type Provider struct {
	client      *openai.Client
	model       string
	temperature float64
	maxTokens   int
}

// Config holds configuration for the OpenAI provider.
type Config struct {
	APIKey      string
	BaseURL     string
	Model       string
	OrgID       string
	Temperature float64
	MaxTokens   int
}

// New creates a new OpenAI-compatible LLM provider.
func New(cfg Config) *Provider {
	clientCfg := openai.DefaultConfig(cfg.APIKey)

	if cfg.BaseURL != "" {
		clientCfg.BaseURL = cfg.BaseURL
	}

	if cfg.OrgID != "" {
		clientCfg.OrgID = cfg.OrgID
	}

	client := openai.NewClientWithConfig(clientCfg)

	return &Provider{
		client:      client,
		model:       cfg.Model,
		temperature: cfg.Temperature,
		maxTokens:   cfg.MaxTokens,
	}
}

// Name returns the provider identifier.
func (p *Provider) Name() string {
	return "openai"
}

// Complete sends a chat completion request.
func (p *Provider) Complete(ctx context.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	messages := make([]openai.ChatCompletionMessage, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	model := p.model
	if req.Model != "" {
		model = req.Model
	}

	temperature := float32(p.temperature)
	if req.Temperature > 0 {
		temperature = float32(req.Temperature)
	}

	maxTokens := p.maxTokens
	if req.MaxTokens > 0 {
		maxTokens = req.MaxTokens
	}

	chatReq := openai.ChatCompletionRequest{
		Model:       model,
		Messages:    messages,
		Temperature: temperature,
		MaxTokens:   maxTokens,
	}

	if req.TopP > 0 {
		topP := float32(req.TopP)
		chatReq.TopP = topP
	}

	if len(req.Stop) > 0 {
		chatReq.Stop = req.Stop
	}

	if req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object" {
		chatReq.ResponseFormat = &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatTypeJSONObject,
		}
	}

	resp, err := p.client.CreateChatCompletion(ctx, chatReq)
	if err != nil {
		return nil, fmt.Errorf("openai completion failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("openai returned no choices")
	}

	choice := resp.Choices[0]
	return &llm.CompletionResponse{
		Content:      choice.Message.Content,
		Model:        resp.Model,
		FinishReason: string(choice.FinishReason),
		Usage: llm.Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}, nil
}

// CompleteJSON forces JSON output from the model.
func (p *Provider) CompleteJSON(ctx context.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	req.ResponseFormat = &llm.ResponseFormat{Type: "json_object"}

	// Ensure system message mentions JSON output
	hasJSONInstruction := false
	for _, msg := range req.Messages {
		if msg.Role == llm.RoleSystem {
			hasJSONInstruction = true
			break
		}
	}

	if !hasJSONInstruction {
		// Prepend a system message requesting JSON
		req.Messages = append([]llm.Message{
			{Role: llm.RoleSystem, Content: "You must respond in valid JSON format."},
		}, req.Messages...)
	}

	return p.Complete(ctx, req)
}

// HealthCheck verifies the provider is reachable.
func (p *Provider) HealthCheck(ctx context.Context) error {
	_, err := p.client.ListModels(ctx)
	if err != nil {
		return fmt.Errorf("openai health check failed: %w", err)
	}
	return nil
}
