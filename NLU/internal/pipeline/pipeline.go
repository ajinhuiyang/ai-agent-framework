// Package pipeline implements the NLU pipeline engine that orchestrates
// multiple NLU capabilities (intent, NER, slot filling, sentiment, classification)
// and manages their execution order and data flow.
//
// Intent recognition, entity extraction, and sentiment analysis use a rule-based
// engine (zero LLM calls) so that only the final answer generation needs LLM.
package pipeline

import (
	"context"
	"fmt"
	"time"

	"go.uber.org/zap"

	"github.com/your-org/nlu/internal/domain"
	"github.com/your-org/nlu/internal/nlu/classify"
	"github.com/your-org/nlu/internal/nlu/dialog"
	"github.com/your-org/nlu/internal/nlu/rules"
	"github.com/your-org/nlu/internal/nlu/slot"
)

// Capability represents a supported NLU capability.
type Capability string

const (
	CapIntent    Capability = "intent"
	CapNER       Capability = "ner"
	CapSlot      Capability = "slot"
	CapSentiment Capability = "sentiment"
	CapClassify  Capability = "classify"
)

// Engine orchestrates the NLU pipeline.
type Engine struct {
	rulesEngine    *rules.Engine // Rule-based: intent, NER, sentiment (instant, no LLM)
	slotFiller     *slot.Filler  // LLM-based slot filling (only when needed)
	textClassifier *classify.Classifier
	dialogManager  *dialog.Manager
	schema         *domain.DomainSchema
	logger         *zap.Logger
	defaultCaps    []Capability
}

// Config holds pipeline engine configuration.
type Config struct {
	RulesEngine    *rules.Engine
	SlotFiller     *slot.Filler
	TextClassifier *classify.Classifier
	DialogManager  *dialog.Manager
	Schema         *domain.DomainSchema
	Logger         *zap.Logger
	DefaultCaps    []string
}

// NewEngine creates a new NLU pipeline engine.
func NewEngine(cfg Config) *Engine {
	defaultCaps := make([]Capability, 0, len(cfg.DefaultCaps))
	for _, c := range cfg.DefaultCaps {
		defaultCaps = append(defaultCaps, Capability(c))
	}

	return &Engine{
		rulesEngine:    cfg.RulesEngine,
		slotFiller:     cfg.SlotFiller,
		textClassifier: cfg.TextClassifier,
		dialogManager:  cfg.DialogManager,
		schema:         cfg.Schema,
		logger:         cfg.Logger,
		defaultCaps:    defaultCaps,
	}
}

// SetSchema updates the domain schema at runtime.
func (e *Engine) SetSchema(schema *domain.DomainSchema) {
	e.schema = schema
}

// Process runs the NLU pipeline on the given request.
// Intent, NER, and sentiment use rule-based matching (instant, no LLM calls).
func (e *Engine) Process(ctx context.Context, req *domain.NLURequest) (*domain.NLUResult, error) {
	start := time.Now()

	result := &domain.NLUResult{
		Text:     req.Text,
		Language: req.Language,
	}

	// Determine which capabilities to run
	caps := e.resolveCaps(req.Capabilities)
	e.logger.Info("processing NLU request",
		zap.String("text", req.Text),
		zap.Strings("capabilities", capStrings(caps)),
	)

	// Get dialog context if session ID is provided
	if req.SessionID != "" && e.dialogManager != nil {
		_ = e.dialogManager.GetOrCreateSession(req.SessionID)
	}

	// Run rule-based capabilities (instant, no LLM)
	for _, cap := range caps {
		switch cap {
		case CapIntent:
			if e.rulesEngine != nil {
				result.Intent = e.rulesEngine.RecognizeIntent(req.Text)
			}

		case CapNER:
			if e.rulesEngine != nil {
				result.Entities = e.rulesEngine.ExtractEntities(req.Text)
			}

		case CapSentiment:
			if e.rulesEngine != nil {
				result.Sentiment = e.rulesEngine.AnalyzeSentiment(req.Text)
			}

		case CapClassify:
			if e.textClassifier != nil {
				categories := e.getCategories(req)
				classResult, err := e.textClassifier.Classify(ctx, &classify.ClassifyRequest{
					Text:       req.Text,
					Categories: categories,
					MultiLabel: req.ClassifyConfig != nil && req.ClassifyConfig.MultiLabel,
				})
				if err != nil {
					e.logger.Error("text classification failed", zap.Error(err))
					result.Errors = append(result.Errors, fmt.Sprintf("classify: %v", err))
				} else {
					result.Classification = classResult
				}
			}
		}
	}

	// Slot filling (depends on intent result, may use LLM)
	if containsCap(caps, CapSlot) && e.slotFiller != nil {
		intentName := ""
		if result.Intent != nil {
			intentName = result.Intent.TopIntent.Name
		}

		slots := e.getSlotDefinitions(req, intentName)
		if len(slots) > 0 {
			var filledSlots map[string]domain.SlotValue
			if req.SessionID != "" && e.dialogManager != nil {
				state := e.dialogManager.GetDialogState(req.SessionID)
				filledSlots = state.FilledSlots
			}

			dialogHistory := ""
			if req.SessionID != "" && e.dialogManager != nil {
				dialogHistory = e.dialogManager.GetDialogHistory(req.SessionID)
			}

			slotResult, err := e.slotFiller.Fill(ctx, &slot.FillRequest{
				Text:          req.Text,
				Intent:        intentName,
				Slots:         slots,
				FilledSlots:   filledSlots,
				DialogHistory: dialogHistory,
			})
			if err != nil {
				e.logger.Error("slot filling failed", zap.Error(err))
				result.Errors = append(result.Errors, fmt.Sprintf("slot: %v", err))
			} else {
				result.SlotFilling = slotResult

				if req.SessionID != "" && e.dialogManager != nil {
					e.dialogManager.UpdateSlots(req.SessionID, slotResult.FilledSlots)
				}
			}
		}
	}

	// Update dialog history
	if req.SessionID != "" && e.dialogManager != nil {
		e.dialogManager.AddUserTurn(req.SessionID, req.Text, result)
		result.DialogState = e.dialogManager.GetDialogState(req.SessionID)
	}

	result.ProcessingTime = time.Since(start).Milliseconds()

	e.logger.Info("NLU processing complete (rule-based)",
		zap.Int64("processing_time_ms", result.ProcessingTime),
		zap.Int("error_count", len(result.Errors)),
	)

	return result, nil
}

// resolveCaps determines which capabilities to run.
func (e *Engine) resolveCaps(requested []string) []Capability {
	if len(requested) == 0 {
		return e.defaultCaps
	}

	caps := make([]Capability, 0, len(requested))
	for _, r := range requested {
		caps = append(caps, Capability(r))
	}
	return caps
}

// getCategories resolves classification categories from request or schema.
func (e *Engine) getCategories(req *domain.NLURequest) []string {
	if req.ClassifyConfig != nil && len(req.ClassifyConfig.Categories) > 0 {
		return req.ClassifyConfig.Categories
	}
	if e.schema != nil && len(e.schema.Categories) > 0 {
		return e.schema.Categories
	}
	return []string{"general", "question", "command", "feedback", "other"}
}

// getSlotDefinitions resolves slot definitions from request or schema.
func (e *Engine) getSlotDefinitions(req *domain.NLURequest, intentName string) []domain.SlotDefinition {
	if req.SlotConfig != nil && len(req.SlotConfig.SlotDefinitions) > 0 {
		return req.SlotConfig.SlotDefinitions
	}

	if e.schema != nil && intentName != "" {
		for _, intentSchema := range e.schema.Intents {
			if intentSchema.Name == intentName && len(intentSchema.Slots) > 0 {
				var slots []domain.SlotDefinition
				for _, slotName := range intentSchema.Slots {
					for _, sd := range e.schema.SlotDefinitions {
						if sd.Name == slotName {
							slots = append(slots, sd)
						}
					}
				}
				return slots
			}
		}
	}

	if e.schema != nil {
		return e.schema.SlotDefinitions
	}

	return nil
}

func containsCap(caps []Capability, target Capability) bool {
	for _, c := range caps {
		if c == target {
			return true
		}
	}
	return false
}

func capStrings(caps []Capability) []string {
	s := make([]string, len(caps))
	for i, c := range caps {
		s[i] = string(c)
	}
	return s
}
