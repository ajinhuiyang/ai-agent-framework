// Package ollama implements the LLM provider using the Ollama REST API directly.
// While Ollama also supports OpenAI-compatible endpoints, this package provides
// native Ollama API integration for full access to Ollama-specific features.
package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/your-org/nlu/internal/llm"
)

// Provider implements llm.Provider using native Ollama API.
type Provider struct {
	baseURL     string
	model       string
	httpClient  *http.Client
	temperature float64
	maxTokens   int
}

// Config holds configuration for the Ollama provider.
type Config struct {
	BaseURL     string
	Model       string
	Timeout     time.Duration
	Temperature float64
	MaxTokens   int
}

// New creates a new Ollama provider.
func New(cfg Config) *Provider {
	if cfg.Timeout == 0 {
		cfg.Timeout = 600 * time.Second // 10 minutes for local inference
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "http://localhost:11434"
	}

	return &Provider{
		baseURL: cfg.BaseURL,
		model:   cfg.Model,
		httpClient: &http.Client{
			Timeout: cfg.Timeout,
		},
		temperature: cfg.Temperature,
		maxTokens:   cfg.MaxTokens,
	}
}

// ollamaChatRequest is the native Ollama chat API request.
type ollamaChatRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
	Format   string          `json:"format,omitempty"`
	Options  *ollamaOptions  `json:"options,omitempty"`
}

type ollamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ollamaOptions struct {
	Temperature float64  `json:"temperature,omitempty"`
	NumPredict  int      `json:"num_predict,omitempty"`
	TopP        float64  `json:"top_p,omitempty"`
	Stop        []string `json:"stop,omitempty"`
}

// ollamaChatResponse is the native Ollama chat API response.
type ollamaChatResponse struct {
	Model           string        `json:"model"`
	Message         ollamaMessage `json:"message"`
	Done            bool          `json:"done"`
	TotalDuration   int64         `json:"total_duration"`
	PromptEvalCount int           `json:"prompt_eval_count"`
	EvalCount       int           `json:"eval_count"`
}

// Name returns the provider identifier.
func (p *Provider) Name() string {
	return "ollama"
}

// Complete sends a chat completion request to Ollama.
func (p *Provider) Complete(ctx context.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	messages := make([]ollamaMessage, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = ollamaMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	model := p.model
	if req.Model != "" {
		model = req.Model
	}

	temp := p.temperature
	if req.Temperature > 0 {
		temp = req.Temperature
	}

	maxTok := p.maxTokens
	if req.MaxTokens > 0 {
		maxTok = req.MaxTokens
	}

	ollamaReq := ollamaChatRequest{
		Model:    model,
		Messages: messages,
		Stream:   false,
		Options: &ollamaOptions{
			Temperature: temp,
			NumPredict:  maxTok,
		},
	}

	if req.TopP > 0 {
		ollamaReq.Options.TopP = req.TopP
	}

	if len(req.Stop) > 0 {
		ollamaReq.Options.Stop = req.Stop
	}

	if req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object" {
		ollamaReq.Format = "json"
	}

	body, err := json.Marshal(ollamaReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal ollama request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create ollama request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama request failed: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(httpResp.Body)
		return nil, fmt.Errorf("ollama returned status %d: %s", httpResp.StatusCode, string(respBody))
	}

	var ollamaResp ollamaChatResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&ollamaResp); err != nil {
		return nil, fmt.Errorf("failed to decode ollama response: %w", err)
	}

	return &llm.CompletionResponse{
		Content:      ollamaResp.Message.Content,
		Model:        ollamaResp.Model,
		FinishReason: "stop",
		Usage: llm.Usage{
			PromptTokens:     ollamaResp.PromptEvalCount,
			CompletionTokens: ollamaResp.EvalCount,
			TotalTokens:      ollamaResp.PromptEvalCount + ollamaResp.EvalCount,
		},
	}, nil
}

// CompleteJSON forces JSON output from Ollama.
func (p *Provider) CompleteJSON(ctx context.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	req.ResponseFormat = &llm.ResponseFormat{Type: "json_object"}

	// Ensure system message mentions JSON output
	hasSystem := false
	for _, msg := range req.Messages {
		if msg.Role == llm.RoleSystem {
			hasSystem = true
			break
		}
	}
	if !hasSystem {
		req.Messages = append([]llm.Message{
			{Role: llm.RoleSystem, Content: "You must respond in valid JSON format."},
		}, req.Messages...)
	}

	return p.Complete(ctx, req)
}

// HealthCheck verifies Ollama is reachable.
func (p *Provider) HealthCheck(ctx context.Context) error {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+"/api/tags", nil)
	if err != nil {
		return fmt.Errorf("ollama health check failed: %w", err)
	}

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("ollama health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ollama health check returned status %d", resp.StatusCode)
	}

	return nil
}
