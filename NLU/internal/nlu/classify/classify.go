// Package classify implements text classification using LLM.
package classify

import (
	"context"
	"encoding/json"
	"fmt"

	"go.uber.org/zap"

	"github.com/your-org/nlu/internal/domain"
	"github.com/your-org/nlu/internal/llm"
	"github.com/your-org/nlu/internal/prompt"
)

// Classifier performs text classification using an LLM provider.
type Classifier struct {
	provider      llm.Provider
	promptManager *prompt.Manager
	logger        *zap.Logger
}

// New creates a new text classifier.
func New(provider llm.Provider, promptManager *prompt.Manager, logger *zap.Logger) *Classifier {
	return &Classifier{
		provider:      provider,
		promptManager: promptManager,
		logger:        logger,
	}
}

// classifyRequest holds template data for text classification.
type classifyRequest struct {
	Text       string
	Categories []string
	MultiLabel bool
}

// llmClassifyResponse is the expected JSON structure from the LLM.
type llmClassifyResponse struct {
	TopCategory struct {
		Label      string  `json:"label"`
		Confidence float64 `json:"confidence"`
	} `json:"top_category"`
	Categories []struct {
		Label      string  `json:"label"`
		Confidence float64 `json:"confidence"`
	} `json:"categories"`
	IsMultiLabel bool `json:"is_multi_label"`
}

// ClassifyRequest contains parameters for text classification.
type ClassifyRequest struct {
	Text       string
	Categories []string
	MultiLabel bool
}

// Classify classifies the given text into categories.
func (c *Classifier) Classify(ctx context.Context, req *ClassifyRequest) (*domain.ClassificationResult, error) {
	c.logger.Debug("classifying text", zap.String("text", req.Text))

	data := classifyRequest{
		Text:       req.Text,
		Categories: req.Categories,
		MultiLabel: req.MultiLabel,
	}

	promptStr, err := c.promptManager.Render("text_classification", data)
	if err != nil {
		return nil, fmt.Errorf("failed to render classification prompt: %w", err)
	}

	resp, err := c.provider.CompleteJSON(ctx, &llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "You are a text classification system. Always respond in valid JSON format."},
			{Role: llm.RoleUser, Content: promptStr},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("LLM text classification failed: %w", err)
	}

	var llmResp llmClassifyResponse
	if err := json.Unmarshal([]byte(resp.Content), &llmResp); err != nil {
		c.logger.Warn("failed to parse LLM classification response",
			zap.String("raw_response", resp.Content),
			zap.Error(err),
		)
		return nil, fmt.Errorf("failed to parse classification response: %w", err)
	}

	result := &domain.ClassificationResult{
		TopCategory: domain.TextCategory{
			Label:      llmResp.TopCategory.Label,
			Confidence: llmResp.TopCategory.Confidence,
		},
		IsMultiLabel: llmResp.IsMultiLabel,
	}

	for _, cat := range llmResp.Categories {
		result.Categories = append(result.Categories, domain.TextCategory{
			Label:      cat.Label,
			Confidence: cat.Confidence,
		})
	}

	c.logger.Info("text classified",
		zap.String("category", result.TopCategory.Label),
		zap.Float64("confidence", result.TopCategory.Confidence),
	)

	return result, nil
}
