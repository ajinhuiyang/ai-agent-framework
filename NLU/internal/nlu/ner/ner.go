// Package ner implements named entity recognition using LLM.
package ner

import (
	"context"
	"encoding/json"
	"fmt"

	"go.uber.org/zap"

	"github.com/your-org/nlu/internal/domain"
	"github.com/your-org/nlu/internal/llm"
	"github.com/your-org/nlu/internal/prompt"
)

// Extractor performs named entity recognition using an LLM provider.
type Extractor struct {
	provider      llm.Provider
	promptManager *prompt.Manager
	logger        *zap.Logger
}

// New creates a new NER extractor.
func New(provider llm.Provider, promptManager *prompt.Manager, logger *zap.Logger) *Extractor {
	return &Extractor{
		provider:      provider,
		promptManager: promptManager,
		logger:        logger,
	}
}

// extractRequest holds template data for NER.
type extractRequest struct {
	Text        string
	EntityTypes []domain.EntityTypeSchema
	Language    string
}

// llmNERResponse is the expected JSON structure from the LLM.
type llmNERResponse struct {
	Entities []struct {
		Type       string  `json:"type"`
		Value      string  `json:"value"`
		Start      int     `json:"start"`
		End        int     `json:"end"`
		Confidence float64 `json:"confidence"`
		Normalized string  `json:"normalized,omitempty"`
	} `json:"entities"`
}

// Extract identifies named entities in the given text.
func (e *Extractor) Extract(ctx context.Context, text string, schema *domain.DomainSchema, language string) (*domain.NERResult, error) {
	e.logger.Debug("extracting entities", zap.String("text", text))

	data := extractRequest{
		Text:     text,
		Language: language,
	}
	if schema != nil {
		data.EntityTypes = schema.EntityTypes
	}

	promptStr, err := e.promptManager.Render("ner", data)
	if err != nil {
		return nil, fmt.Errorf("failed to render NER prompt: %w", err)
	}

	resp, err := e.provider.CompleteJSON(ctx, &llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "You are a named entity recognition system. Always respond in valid JSON format."},
			{Role: llm.RoleUser, Content: promptStr},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("LLM NER failed: %w", err)
	}

	var llmResp llmNERResponse
	if err := json.Unmarshal([]byte(resp.Content), &llmResp); err != nil {
		e.logger.Warn("failed to parse LLM NER response",
			zap.String("raw_response", resp.Content),
			zap.Error(err),
		)
		return nil, fmt.Errorf("failed to parse NER response: %w", err)
	}

	result := &domain.NERResult{}
	for _, ent := range llmResp.Entities {
		result.Entities = append(result.Entities, domain.Entity{
			Type:       ent.Type,
			Value:      ent.Value,
			Start:      ent.Start,
			End:        ent.End,
			Confidence: ent.Confidence,
			Normalized: ent.Normalized,
		})
	}

	e.logger.Info("entities extracted", zap.Int("count", len(result.Entities)))
	return result, nil
}
