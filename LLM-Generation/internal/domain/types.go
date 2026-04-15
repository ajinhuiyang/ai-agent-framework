// Package domain defines the core data types for the LLM Generation service.
package domain

import "time"

// GenerateRequest represents a request to generate content.
type GenerateRequest struct {
	// Prompt is the main user input / question.
	Prompt string `json:"prompt"`

	// SystemPrompt overrides the default system prompt.
	SystemPrompt string `json:"system_prompt,omitempty"`

	// Context is additional context retrieved from RAG or NLU.
	Context []ContextItem `json:"context,omitempty"`

	// NLUResult is the structured intent/entity result from the NLU service.
	NLUResult *NLUResult `json:"nlu_result,omitempty"`

	// ConversationID links to an ongoing conversation for multi-turn.
	ConversationID string `json:"conversation_id,omitempty"`

	// History provides explicit conversation history.
	History []Message `json:"history,omitempty"`

	// Config provides runtime generation parameters.
	Config *GenerateConfig `json:"config,omitempty"`

	// Stream indicates whether to use streaming response.
	Stream bool `json:"stream,omitempty"`

	// Provider specifies which LLM backend to use (optional, uses default if empty).
	Provider string `json:"provider,omitempty"`
}

// ContextItem is a piece of context from RAG retrieval.
type ContextItem struct {
	Content string  `json:"content"`
	Source  string  `json:"source,omitempty"`
	Score   float64 `json:"score,omitempty"`
}

// NLUResult is a simplified view of the NLU service output.
type NLUResult struct {
	Intent     string            `json:"intent"`
	Confidence float64           `json:"confidence"`
	Entities   []Entity          `json:"entities,omitempty"`
	Sentiment  string            `json:"sentiment,omitempty"`
	Slots      map[string]string `json:"slots,omitempty"`
}

// Entity represents a named entity from NLU.
type Entity struct {
	Type  string `json:"type"`
	Value string `json:"value"`
}

// Message represents a single conversation turn.
type Message struct {
	Role    string `json:"role"` // "system", "user", "assistant"
	Content string `json:"content"`
}

// GenerateConfig provides runtime parameters for generation.
type GenerateConfig struct {
	Temperature float64  `json:"temperature,omitempty"`
	MaxTokens   int      `json:"max_tokens,omitempty"`
	TopP        float64  `json:"top_p,omitempty"`
	TopK        int      `json:"top_k,omitempty"`
	StopWords   []string `json:"stop_words,omitempty"`
	Model       string   `json:"model,omitempty"`
}

// GenerateResponse is the complete generation result.
type GenerateResponse struct {
	Content        string    `json:"content"`
	ConversationID string    `json:"conversation_id,omitempty"`
	Provider       string    `json:"provider"`
	Model          string    `json:"model"`
	Usage          Usage     `json:"usage"`
	FinishReason   string    `json:"finish_reason"`
	CreatedAt      time.Time `json:"created_at"`
}

// StreamChunk represents a single chunk in streaming response.
type StreamChunk struct {
	Content      string `json:"content"`
	Done         bool   `json:"done"`
	FinishReason string `json:"finish_reason,omitempty"`
}

// Usage tracks token consumption.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Conversation represents a multi-turn conversation session.
type Conversation struct {
	ID        string    `json:"id"`
	Messages  []Message `json:"messages"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// ProviderInfo describes an available LLM provider.
type ProviderInfo struct {
	Name      string   `json:"name"`
	Models    []string `json:"models,omitempty"`
	IsDefault bool     `json:"is_default"`
	Status    string   `json:"status"` // "healthy", "unhealthy", "unknown"
}
