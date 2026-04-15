// Package orchestrator coordinates the LLM generation pipeline:
// prompt building → provider selection → generation → conversation management.
package orchestrator

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"

	"github.com/your-org/llm-generation/internal/domain"
	"github.com/your-org/llm-generation/internal/llm"
	"github.com/your-org/llm-generation/internal/prompt"
)

// Orchestrator is the main generation engine.
type Orchestrator struct {
	providers   map[string]llm.Provider
	defaultProv string
	promptMgr   *prompt.Manager
	logger      *zap.Logger

	// Conversation store (in-memory).
	convMu    sync.RWMutex
	convStore map[string]*domain.Conversation

	// Default generation config.
	defaultConfig domain.GenerateConfig

	maxTurns int
}

// New creates a new Orchestrator.
func New(
	providers map[string]llm.Provider,
	defaultProv string,
	promptMgr *prompt.Manager,
	logger *zap.Logger,
	defaultConfig domain.GenerateConfig,
	maxTurns int,
) *Orchestrator {
	return &Orchestrator{
		providers:     providers,
		defaultProv:   defaultProv,
		promptMgr:     promptMgr,
		logger:        logger,
		convStore:     make(map[string]*domain.Conversation),
		defaultConfig: defaultConfig,
		maxTurns:      maxTurns,
	}
}

// Generate performs a non-streaming generation.
func (o *Orchestrator) Generate(ctx context.Context, req domain.GenerateRequest) (*domain.GenerateResponse, error) {
	// Select provider.
	provider, err := o.getProvider(req.Provider)
	if err != nil {
		return nil, err
	}

	// Merge config.
	config := o.mergeConfig(req.Config)

	// Load conversation history if applicable.
	if req.ConversationID != "" {
		o.loadConversationHistory(&req)
	}

	// Build messages.
	messages := o.promptMgr.BuildMessages(req)

	o.logger.Info("generating response",
		zap.String("provider", provider.Name()),
		zap.Int("messages", len(messages)),
	)

	// Call LLM.
	result, err := provider.Complete(ctx, messages, config)
	if err != nil {
		return nil, fmt.Errorf("generation failed: %w", err)
	}

	// Update conversation.
	convID := req.ConversationID
	if convID != "" {
		o.updateConversation(convID, req.Prompt, result.Content)
	}

	return &domain.GenerateResponse{
		Content:        result.Content,
		ConversationID: convID,
		Provider:       provider.Name(),
		Model:          result.Model,
		Usage:          result.Usage,
		FinishReason:   result.FinishReason,
		CreatedAt:      time.Now(),
	}, nil
}

// GenerateStream performs a streaming generation.
func (o *Orchestrator) GenerateStream(ctx context.Context, req domain.GenerateRequest) (<-chan domain.StreamChunk, string, error) {
	provider, err := o.getProvider(req.Provider)
	if err != nil {
		return nil, "", err
	}

	config := o.mergeConfig(req.Config)

	if req.ConversationID != "" {
		o.loadConversationHistory(&req)
	}

	messages := o.promptMgr.BuildMessages(req)

	streamCh, err := provider.CompleteStream(ctx, messages, config)
	if err != nil {
		return nil, "", fmt.Errorf("stream generation failed: %w", err)
	}

	// Transform llm.StreamEvent to domain.StreamChunk and accumulate full content.
	outCh := make(chan domain.StreamChunk, 64)
	convID := req.ConversationID

	go func() {
		defer close(outCh)
		var fullContent string

		for event := range streamCh {
			if event.Err != nil {
				outCh <- domain.StreamChunk{
					Content:      fmt.Sprintf("[error: %v]", event.Err),
					Done:         true,
					FinishReason: "error",
				}
				return
			}

			fullContent += event.Content
			outCh <- domain.StreamChunk{
				Content:      event.Content,
				Done:         event.Done,
				FinishReason: event.FinishReason,
			}

			if event.Done {
				// Update conversation after streaming completes.
				if convID != "" {
					o.updateConversation(convID, req.Prompt, fullContent)
				}
				return
			}
		}
	}()

	return outCh, provider.Name(), nil
}

// CreateConversation creates a new conversation session.
func (o *Orchestrator) CreateConversation() string {
	id := uuid.New().String()
	o.convMu.Lock()
	defer o.convMu.Unlock()
	o.convStore[id] = &domain.Conversation{
		ID:        id,
		Messages:  []domain.Message{},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	return id
}

// GetConversation retrieves a conversation by ID.
func (o *Orchestrator) GetConversation(id string) (*domain.Conversation, bool) {
	o.convMu.RLock()
	defer o.convMu.RUnlock()
	conv, ok := o.convStore[id]
	return conv, ok
}

// DeleteConversation removes a conversation.
func (o *Orchestrator) DeleteConversation(id string) {
	o.convMu.Lock()
	defer o.convMu.Unlock()
	delete(o.convStore, id)
}

// ListProviders returns info about available providers.
func (o *Orchestrator) ListProviders(ctx context.Context) []domain.ProviderInfo {
	var infos []domain.ProviderInfo
	for name, p := range o.providers {
		status := "healthy"
		if err := p.HealthCheck(ctx); err != nil {
			status = "unhealthy"
		}
		infos = append(infos, domain.ProviderInfo{
			Name:      name,
			Models:    p.Models(),
			IsDefault: name == o.defaultProv,
			Status:    status,
		})
	}
	return infos
}

func (o *Orchestrator) getProvider(name string) (llm.Provider, error) {
	if name == "" {
		name = o.defaultProv
	}
	p, ok := o.providers[name]
	if !ok {
		return nil, fmt.Errorf("unknown provider: %s (available: %v)", name, o.providerNames())
	}
	return p, nil
}

func (o *Orchestrator) providerNames() []string {
	names := make([]string, 0, len(o.providers))
	for name := range o.providers {
		names = append(names, name)
	}
	return names
}

func (o *Orchestrator) mergeConfig(reqConfig *domain.GenerateConfig) *domain.GenerateConfig {
	config := o.defaultConfig
	if reqConfig != nil {
		if reqConfig.Temperature > 0 {
			config.Temperature = reqConfig.Temperature
		}
		if reqConfig.MaxTokens > 0 {
			config.MaxTokens = reqConfig.MaxTokens
		}
		if reqConfig.TopP > 0 {
			config.TopP = reqConfig.TopP
		}
		if reqConfig.TopK > 0 {
			config.TopK = reqConfig.TopK
		}
		if len(reqConfig.StopWords) > 0 {
			config.StopWords = reqConfig.StopWords
		}
		if reqConfig.Model != "" {
			config.Model = reqConfig.Model
		}
	}
	return &config
}

func (o *Orchestrator) loadConversationHistory(req *domain.GenerateRequest) {
	o.convMu.RLock()
	defer o.convMu.RUnlock()

	conv, ok := o.convStore[req.ConversationID]
	if !ok {
		return
	}

	// Prepend conversation history before any explicit history.
	if len(conv.Messages) > 0 {
		history := make([]domain.Message, len(conv.Messages))
		copy(history, conv.Messages)
		req.History = append(history, req.History...)
	}
}

func (o *Orchestrator) updateConversation(convID, userMsg, assistantMsg string) {
	o.convMu.Lock()
	defer o.convMu.Unlock()

	conv, ok := o.convStore[convID]
	if !ok {
		conv = &domain.Conversation{
			ID:        convID,
			CreatedAt: time.Now(),
		}
		o.convStore[convID] = conv
	}

	conv.Messages = append(conv.Messages,
		domain.Message{Role: "user", Content: userMsg},
		domain.Message{Role: "assistant", Content: assistantMsg},
	)
	conv.UpdatedAt = time.Now()

	// Enforce max turns (sliding window).
	if o.maxTurns > 0 && len(conv.Messages) > o.maxTurns*2 {
		conv.Messages = conv.Messages[len(conv.Messages)-o.maxTurns*2:]
	}
}
