// Package slot implements slot filling for task-oriented dialogue using LLM.
package slot

import (
	"context"
	"encoding/json"
	"fmt"

	"go.uber.org/zap"

	"github.com/your-org/nlu/internal/domain"
	"github.com/your-org/nlu/internal/llm"
	"github.com/your-org/nlu/internal/prompt"
)

// Filler performs slot filling using an LLM provider.
type Filler struct {
	provider      llm.Provider
	promptManager *prompt.Manager
	logger        *zap.Logger
}

// New creates a new slot filler.
func New(provider llm.Provider, promptManager *prompt.Manager, logger *zap.Logger) *Filler {
	return &Filler{
		provider:      provider,
		promptManager: promptManager,
		logger:        logger,
	}
}

// fillRequest holds template data for slot filling.
type fillRequest struct {
	Text          string
	Intent        string
	Slots         []domain.SlotDefinition
	FilledSlots   map[string]interface{}
	DialogHistory string
}

// llmSlotResponse is the expected JSON structure from the LLM.
type llmSlotResponse struct {
	FilledSlots []struct {
		Name       string      `json:"name"`
		Value      interface{} `json:"value"`
		Confidence float64     `json:"confidence"`
		Source     string      `json:"source"`
	} `json:"filled_slots"`
	MissingSlots []string `json:"missing_slots"`
	NextPrompt   string   `json:"next_prompt"`
	AllFilled    bool     `json:"all_filled"`
}

// FillRequest contains parameters for a slot filling operation.
type FillRequest struct {
	Text          string
	Intent        string
	Slots         []domain.SlotDefinition
	FilledSlots   map[string]domain.SlotValue
	DialogHistory string
}

// Fill extracts slot values from the given text.
func (f *Filler) Fill(ctx context.Context, req *FillRequest) (*domain.SlotFillingResult, error) {
	f.logger.Debug("filling slots",
		zap.String("text", req.Text),
		zap.String("intent", req.Intent),
	)

	// Build existing filled slots map for the prompt
	existingSlots := make(map[string]interface{})
	if req.FilledSlots != nil {
		for name, sv := range req.FilledSlots {
			existingSlots[name] = sv.Value
		}
	}

	data := fillRequest{
		Text:          req.Text,
		Intent:        req.Intent,
		Slots:         req.Slots,
		FilledSlots:   existingSlots,
		DialogHistory: req.DialogHistory,
	}

	promptStr, err := f.promptManager.Render("slot_filling", data)
	if err != nil {
		return nil, fmt.Errorf("failed to render slot filling prompt: %w", err)
	}

	resp, err := f.provider.CompleteJSON(ctx, &llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "You are a slot filling system for task-oriented dialogue. Always respond in valid JSON format."},
			{Role: llm.RoleUser, Content: promptStr},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("LLM slot filling failed: %w", err)
	}

	var llmResp llmSlotResponse
	if err := json.Unmarshal([]byte(resp.Content), &llmResp); err != nil {
		f.logger.Warn("failed to parse LLM slot filling response",
			zap.String("raw_response", resp.Content),
			zap.Error(err),
		)
		return nil, fmt.Errorf("failed to parse slot filling response: %w", err)
	}

	result := &domain.SlotFillingResult{
		MissingSlots: llmResp.MissingSlots,
		NextPrompt:   llmResp.NextPrompt,
		AllFilled:    llmResp.AllFilled,
	}

	for _, s := range llmResp.FilledSlots {
		result.FilledSlots = append(result.FilledSlots, domain.SlotValue{
			Name:       s.Name,
			Value:      s.Value,
			Confidence: s.Confidence,
			Source:     s.Source,
		})
	}

	f.logger.Info("slots filled",
		zap.Int("filled_count", len(result.FilledSlots)),
		zap.Int("missing_count", len(result.MissingSlots)),
		zap.Bool("all_filled", result.AllFilled),
	)

	return result, nil
}
