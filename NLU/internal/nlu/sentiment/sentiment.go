// Package sentiment implements sentiment analysis using LLM.
package sentiment

import (
	"context"
	"encoding/json"
	"fmt"

	"go.uber.org/zap"

	"github.com/your-org/nlu/internal/domain"
	"github.com/your-org/nlu/internal/llm"
	"github.com/your-org/nlu/internal/prompt"
)

// Analyzer performs sentiment analysis using an LLM provider.
type Analyzer struct {
	provider      llm.Provider
	promptManager *prompt.Manager
	logger        *zap.Logger
}

// New creates a new sentiment analyzer.
func New(provider llm.Provider, promptManager *prompt.Manager, logger *zap.Logger) *Analyzer {
	return &Analyzer{
		provider:      provider,
		promptManager: promptManager,
		logger:        logger,
	}
}

// analyzeRequest holds template data for sentiment analysis.
type analyzeRequest struct {
	Text     string
	Language string
}

// llmSentimentResponse is the expected JSON structure from the LLM.
type llmSentimentResponse struct {
	Label      string  `json:"label"`
	Score      float64 `json:"score"`
	Confidence float64 `json:"confidence"`
	Aspects    []struct {
		Aspect     string  `json:"aspect"`
		Label      string  `json:"label"`
		Score      float64 `json:"score"`
		Confidence float64 `json:"confidence"`
	} `json:"aspects,omitempty"`
	Emotions map[string]float64 `json:"emotions,omitempty"`
}

// Analyze performs sentiment analysis on the given text.
func (a *Analyzer) Analyze(ctx context.Context, text string, language string) (*domain.SentimentResult, error) {
	a.logger.Debug("analyzing sentiment", zap.String("text", text))

	data := analyzeRequest{
		Text:     text,
		Language: language,
	}

	promptStr, err := a.promptManager.Render("sentiment_analysis", data)
	if err != nil {
		return nil, fmt.Errorf("failed to render sentiment prompt: %w", err)
	}

	resp, err := a.provider.CompleteJSON(ctx, &llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "You are a sentiment analysis system. Always respond in valid JSON format."},
			{Role: llm.RoleUser, Content: promptStr},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("LLM sentiment analysis failed: %w", err)
	}

	var llmResp llmSentimentResponse
	if err := json.Unmarshal([]byte(resp.Content), &llmResp); err != nil {
		a.logger.Warn("failed to parse LLM sentiment response",
			zap.String("raw_response", resp.Content),
			zap.Error(err),
		)
		return nil, fmt.Errorf("failed to parse sentiment response: %w", err)
	}

	result := &domain.SentimentResult{
		Label:      domain.SentimentLabel(llmResp.Label),
		Score:      llmResp.Score,
		Confidence: llmResp.Confidence,
		Emotions:   llmResp.Emotions,
	}

	for _, asp := range llmResp.Aspects {
		result.Aspects = append(result.Aspects, domain.AspectSentiment{
			Aspect:     asp.Aspect,
			Label:      domain.SentimentLabel(asp.Label),
			Score:      asp.Score,
			Confidence: asp.Confidence,
		})
	}

	a.logger.Info("sentiment analyzed",
		zap.String("label", string(result.Label)),
		zap.Float64("score", result.Score),
	)

	return result, nil
}
