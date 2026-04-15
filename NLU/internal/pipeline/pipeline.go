// Package pipeline implements the NLU pipeline engine that orchestrates
// multiple NLU capabilities (intent, NER, slot filling, sentiment, classification)
// and manages their execution order and data flow.
package pipeline

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/your-org/nlu/internal/domain"
	"github.com/your-org/nlu/internal/nlu/classify"
	"github.com/your-org/nlu/internal/nlu/dialog"
	"github.com/your-org/nlu/internal/nlu/intent"
	"github.com/your-org/nlu/internal/nlu/ner"
	"github.com/your-org/nlu/internal/nlu/sentiment"
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
	intentRecognizer  *intent.Recognizer
	nerExtractor      *ner.Extractor
	slotFiller        *slot.Filler
	sentimentAnalyzer *sentiment.Analyzer
	textClassifier    *classify.Classifier
	dialogManager     *dialog.Manager
	schema            *domain.DomainSchema
	logger            *zap.Logger
	defaultCaps       []Capability
}

// Config holds pipeline engine configuration.
type Config struct {
	IntentRecognizer  *intent.Recognizer
	NERExtractor      *ner.Extractor
	SlotFiller        *slot.Filler
	SentimentAnalyzer *sentiment.Analyzer
	TextClassifier    *classify.Classifier
	DialogManager     *dialog.Manager
	Schema            *domain.DomainSchema
	Logger            *zap.Logger
	DefaultCaps       []string
}

// NewEngine creates a new NLU pipeline engine.
func NewEngine(cfg Config) *Engine {
	defaultCaps := make([]Capability, 0, len(cfg.DefaultCaps))
	for _, c := range cfg.DefaultCaps {
		defaultCaps = append(defaultCaps, Capability(c))
	}

	return &Engine{
		intentRecognizer:  cfg.IntentRecognizer,
		nerExtractor:      cfg.NERExtractor,
		slotFiller:        cfg.SlotFiller,
		sentimentAnalyzer: cfg.SentimentAnalyzer,
		textClassifier:    cfg.TextClassifier,
		dialogManager:     cfg.DialogManager,
		schema:            cfg.Schema,
		logger:            cfg.Logger,
		defaultCaps:       defaultCaps,
	}
}

// SetSchema updates the domain schema at runtime.
func (e *Engine) SetSchema(schema *domain.DomainSchema) {
	e.schema = schema
}

// Process runs the NLU pipeline on the given request.
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
	var dialogHistory string
	if req.SessionID != "" && e.dialogManager != nil {
		_ = e.dialogManager.GetOrCreateSession(req.SessionID)
		dialogHistory = e.dialogManager.GetDialogHistory(req.SessionID)
	}

	// Determine which capabilities can run in parallel vs sequentially
	// Phase 1 (parallel): intent, ner, sentiment, classify
	// Phase 2 (depends on intent): slot filling
	var wg sync.WaitGroup
	var mu sync.Mutex

	// Phase 1: Run independent capabilities in parallel
	for _, cap := range caps {
		switch cap {
		case CapIntent:
			wg.Add(1)
			go func() {
				defer wg.Done()
				intentResult, err := e.intentRecognizer.Recognize(ctx, req.Text, e.schema, dialogHistory)
				mu.Lock()
				defer mu.Unlock()
				if err != nil {
					e.logger.Error("intent recognition failed", zap.Error(err))
					result.Errors = append(result.Errors, fmt.Sprintf("intent: %v", err))
				} else {
					result.Intent = intentResult
				}
			}()

		case CapNER:
			wg.Add(1)
			go func() {
				defer wg.Done()
				nerResult, err := e.nerExtractor.Extract(ctx, req.Text, e.schema, req.Language)
				mu.Lock()
				defer mu.Unlock()
				if err != nil {
					e.logger.Error("NER failed", zap.Error(err))
					result.Errors = append(result.Errors, fmt.Sprintf("ner: %v", err))
				} else {
					result.Entities = nerResult
				}
			}()

		case CapSentiment:
			wg.Add(1)
			go func() {
				defer wg.Done()
				sentResult, err := e.sentimentAnalyzer.Analyze(ctx, req.Text, req.Language)
				mu.Lock()
				defer mu.Unlock()
				if err != nil {
					e.logger.Error("sentiment analysis failed", zap.Error(err))
					result.Errors = append(result.Errors, fmt.Sprintf("sentiment: %v", err))
				} else {
					result.Sentiment = sentResult
				}
			}()

		case CapClassify:
			wg.Add(1)
			go func() {
				defer wg.Done()
				categories := e.getCategories(req)
				classResult, err := e.textClassifier.Classify(ctx, &classify.ClassifyRequest{
					Text:       req.Text,
					Categories: categories,
					MultiLabel: req.ClassifyConfig != nil && req.ClassifyConfig.MultiLabel,
				})
				mu.Lock()
				defer mu.Unlock()
				if err != nil {
					e.logger.Error("text classification failed", zap.Error(err))
					result.Errors = append(result.Errors, fmt.Sprintf("classify: %v", err))
				} else {
					result.Classification = classResult
				}
			}()
		}
	}

	wg.Wait()

	// Phase 2: Slot filling (depends on intent result)
	if containsCap(caps, CapSlot) && e.slotFiller != nil {
		intentName := ""
		if result.Intent != nil {
			intentName = result.Intent.TopIntent.Name
		}

		slots := e.getSlotDefinitions(req, intentName)
		if len(slots) > 0 {
			// Get existing filled slots from dialog context
			var filledSlots map[string]domain.SlotValue
			if req.SessionID != "" && e.dialogManager != nil {
				state := e.dialogManager.GetDialogState(req.SessionID)
				filledSlots = state.FilledSlots
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

				// Update dialog slots
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

	e.logger.Info("NLU processing complete",
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
		cap := Capability(r)
		if e.isCapAvailable(cap) {
			caps = append(caps, cap)
		}
	}
	return caps
}

// isCapAvailable checks if a capability's handler is configured.
func (e *Engine) isCapAvailable(cap Capability) bool {
	switch cap {
	case CapIntent:
		return e.intentRecognizer != nil
	case CapNER:
		return e.nerExtractor != nil
	case CapSlot:
		return e.slotFiller != nil
	case CapSentiment:
		return e.sentimentAnalyzer != nil
	case CapClassify:
		return e.textClassifier != nil
	default:
		return false
	}
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
	// First check request-level slot config
	if req.SlotConfig != nil && len(req.SlotConfig.SlotDefinitions) > 0 {
		return req.SlotConfig.SlotDefinitions
	}

	// Then check schema for intent-specific slots
	if e.schema != nil && intentName != "" {
		for _, intentSchema := range e.schema.Intents {
			if intentSchema.Name == intentName && len(intentSchema.Slots) > 0 {
				// Find matching slot definitions
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

	// Fall back to all schema slot definitions
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
