// Package ollama implements the LLM provider using the Ollama native API.
package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/your-org/llm-generation/internal/domain"
	"github.com/your-org/llm-generation/internal/llm"
)

// Provider implements llm.Provider using the Ollama /api/chat endpoint.
type Provider struct {
	baseURL string
	model   string
	client  *http.Client
}

// New creates a new Ollama provider.
func New(baseURL, model string) *Provider {
	return &Provider{
		baseURL: baseURL,
		model:   model,
		client:  &http.Client{},
	}
}

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
	Options  *chatOptions  `json:"options,omitempty"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatOptions struct {
	Temperature float64 `json:"temperature,omitempty"`
	NumPredict  int     `json:"num_predict,omitempty"`
	TopP        float64 `json:"top_p,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
}

type chatResponse struct {
	Model           string      `json:"model"`
	Message         chatMessage `json:"message"`
	Done            bool        `json:"done"`
	DoneReason      string      `json:"done_reason,omitempty"`
	PromptEvalCount int         `json:"prompt_eval_count,omitempty"`
	EvalCount       int         `json:"eval_count,omitempty"`
}

func (p *Provider) Complete(ctx context.Context, messages []domain.Message, config *domain.GenerateConfig) (*llm.CompletionResult, error) {
	model := p.model
	if config != nil && config.Model != "" {
		model = config.Model
	}

	reqBody := chatRequest{
		Model:    model,
		Messages: toChatMessages(messages),
		Stream:   false,
		Options:  buildOptions(config),
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama chat request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var result chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &llm.CompletionResult{
		Content:      result.Message.Content,
		Model:        result.Model,
		FinishReason: result.DoneReason,
		Usage: domain.Usage{
			PromptTokens:     result.PromptEvalCount,
			CompletionTokens: result.EvalCount,
			TotalTokens:      result.PromptEvalCount + result.EvalCount,
		},
	}, nil
}

func (p *Provider) CompleteStream(ctx context.Context, messages []domain.Message, config *domain.GenerateConfig) (<-chan llm.StreamEvent, error) {
	model := p.model
	if config != nil && config.Model != "" {
		model = config.Model
	}

	reqBody := chatRequest{
		Model:    model,
		Messages: toChatMessages(messages),
		Stream:   true,
		Options:  buildOptions(config),
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama stream request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("ollama stream returned status %d: %s", resp.StatusCode, string(respBody))
	}

	ch := make(chan llm.StreamEvent, 64)
	go func() {
		defer close(ch)
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			var chunk chatResponse
			if err := json.Unmarshal(scanner.Bytes(), &chunk); err != nil {
				ch <- llm.StreamEvent{Err: err, Done: true}
				return
			}

			event := llm.StreamEvent{
				Content: chunk.Message.Content,
				Done:    chunk.Done,
			}
			if chunk.Done {
				event.FinishReason = chunk.DoneReason
				event.Usage = &domain.Usage{
					PromptTokens:     chunk.PromptEvalCount,
					CompletionTokens: chunk.EvalCount,
					TotalTokens:      chunk.PromptEvalCount + chunk.EvalCount,
				}
			}
			ch <- event
			if chunk.Done {
				return
			}
		}
		if err := scanner.Err(); err != nil {
			ch <- llm.StreamEvent{Err: err, Done: true}
		}
	}()

	return ch, nil
}

func (p *Provider) Name() string     { return "ollama" }
func (p *Provider) Models() []string { return []string{p.model} }

func (p *Provider) HealthCheck(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+"/api/tags", nil)
	if err != nil {
		return err
	}
	resp, err := p.client.Do(req)
	if err != nil {
		return fmt.Errorf("ollama health check failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ollama returned status %d", resp.StatusCode)
	}
	return nil
}

func toChatMessages(messages []domain.Message) []chatMessage {
	msgs := make([]chatMessage, len(messages))
	for i, m := range messages {
		msgs[i] = chatMessage{Role: m.Role, Content: m.Content}
	}
	return msgs
}

func buildOptions(config *domain.GenerateConfig) *chatOptions {
	if config == nil {
		return nil
	}
	opts := &chatOptions{}
	if config.Temperature > 0 {
		opts.Temperature = config.Temperature
	}
	if config.MaxTokens > 0 {
		opts.NumPredict = config.MaxTokens
	}
	if config.TopP > 0 {
		opts.TopP = config.TopP
	}
	if config.TopK > 0 {
		opts.TopK = config.TopK
	}
	return opts
}
