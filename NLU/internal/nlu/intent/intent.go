// Package intent implements intent recognition using LLM.
package intent

import (
	"context"
	"encoding/json"
	"fmt"

	"go.uber.org/zap"

	"github.com/your-org/nlu/internal/domain"
	"github.com/your-org/nlu/internal/llm"
	"github.com/your-org/nlu/internal/prompt"
)

// Recognizer performs intent recognition using an LLM provider.
type Recognizer struct {
	provider      llm.Provider
	promptManager *prompt.Manager
	logger        *zap.Logger
}

// New creates a new intent recognizer.
func New(provider llm.Provider, promptManager *prompt.Manager, logger *zap.Logger) *Recognizer {
	return &Recognizer{
		provider:      provider,
		promptManager: promptManager,
		logger:        logger,
	}
}

// recognizeRequest holds template data for intent recognition.
type recognizeRequest struct {
	Text    string
	Intents []domain.IntentSchema
	Context string
}

// llmIntentResponse is the expected JSON structure from the LLM.
type llmIntentResponse struct {
	Intent     string  `json:"intent"`
	Confidence float64 `json:"confidence"`
	SubIntent  string  `json:"sub_intent,omitempty"`
	Candidates []struct {
		Intent     string  `json:"intent"`
		Confidence float64 `json:"confidence"`
	} `json:"candidates,omitempty"`
	Reasoning string `json:"reasoning,omitempty"`
}

// Recognize identifies the intent of the given text.
func (r *Recognizer) Recognize(ctx context.Context, text string, schema *domain.DomainSchema, dialogContext string) (*domain.IntentResult, error) {
	r.logger.Debug("recognizing intent", zap.String("text", text))

	// Build prompt data
	data := recognizeRequest{
		Text:    text,
		Context: dialogContext,
	}
	if schema != nil {
		data.Intents = schema.Intents
	}

	// Render the prompt template
	promptStr, err := r.promptManager.Render("intent_recognition", data)
	if err != nil {
		return nil, fmt.Errorf("failed to render intent prompt: %w", err)
	}

	// Call LLM
	resp, err := r.provider.CompleteJSON(ctx, &llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "You are an intent recognition system. Always respond in valid JSON format."},
			{Role: llm.RoleUser, Content: promptStr},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("LLM intent recognition failed: %w", err)
	}

	// Parse the LLM's JSON response
	var llmResp llmIntentResponse
	if err := json.Unmarshal([]byte(resp.Content), &llmResp); err != nil {
		r.logger.Warn("failed to parse LLM intent response, attempting recovery",
			zap.String("raw_response", resp.Content),
			zap.Error(err),
		)
		return nil, fmt.Errorf("failed to parse intent response: %w", err)
	}

	// Build result
	result := &domain.IntentResult{
		TopIntent: domain.Intent{
			Name:       llmResp.Intent,
			Confidence: llmResp.Confidence,
			SubIntent:  llmResp.SubIntent,
		},
	}

	for _, c := range llmResp.Candidates {
		result.Candidates = append(result.Candidates, domain.Intent{
			Name:       c.Intent,
			Confidence: c.Confidence,
		})
	}

	r.logger.Info("intent recognized",
		zap.String("intent", result.TopIntent.Name),
		zap.Float64("confidence", result.TopIntent.Confidence),
	)

	return result, nil
}
