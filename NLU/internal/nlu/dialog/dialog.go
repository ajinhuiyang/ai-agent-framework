// Package dialog implements multi-turn dialog context management.
package dialog

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"

	"github.com/your-org/nlu/internal/domain"
	"github.com/your-org/nlu/internal/llm"
	"github.com/your-org/nlu/internal/prompt"
)

// Manager manages multi-turn dialog sessions.
type Manager struct {
	provider      llm.Provider
	promptManager *prompt.Manager
	logger        *zap.Logger
	sessions      map[string]*domain.DialogContext
	mu            sync.RWMutex
	maxTurns      int
}

// New creates a new dialog manager.
func New(provider llm.Provider, promptManager *prompt.Manager, logger *zap.Logger, maxTurns int) *Manager {
	if maxTurns <= 0 {
		maxTurns = 20
	}
	return &Manager{
		provider:      provider,
		promptManager: promptManager,
		logger:        logger,
		sessions:      make(map[string]*domain.DialogContext),
		maxTurns:      maxTurns,
	}
}

// GetOrCreateSession returns an existing session or creates a new one.
func (m *Manager) GetOrCreateSession(sessionID string) *domain.DialogContext {
	m.mu.Lock()
	defer m.mu.Unlock()

	if session, ok := m.sessions[sessionID]; ok {
		return session
	}

	if sessionID == "" {
		sessionID = uuid.New().String()
	}

	session := &domain.DialogContext{
		SessionID:    sessionID,
		Turns:        make([]domain.DialogTurn, 0),
		CurrentSlots: make(map[string]domain.SlotValue),
		Metadata:     make(map[string]string),
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
	}

	m.sessions[sessionID] = session
	return session
}

// AddUserTurn adds a user message to the dialog history.
func (m *Manager) AddUserTurn(sessionID string, content string, nluResult *domain.NLUResult) {
	m.mu.Lock()
	defer m.mu.Unlock()

	session, ok := m.sessions[sessionID]
	if !ok {
		return
	}

	session.Turns = append(session.Turns, domain.DialogTurn{
		Role:      "user",
		Content:   content,
		Timestamp: time.Now(),
		NLUResult: nluResult,
	})

	// Trim if exceeding max turns
	if len(session.Turns) > m.maxTurns {
		session.Turns = session.Turns[len(session.Turns)-m.maxTurns:]
	}

	session.UpdatedAt = time.Now()
}

// AddAssistantTurn adds an assistant response to the dialog history.
func (m *Manager) AddAssistantTurn(sessionID string, content string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	session, ok := m.sessions[sessionID]
	if !ok {
		return
	}

	session.Turns = append(session.Turns, domain.DialogTurn{
		Role:      "assistant",
		Content:   content,
		Timestamp: time.Now(),
	})

	if len(session.Turns) > m.maxTurns {
		session.Turns = session.Turns[len(session.Turns)-m.maxTurns:]
	}

	session.UpdatedAt = time.Now()
}

// GetDialogHistory returns a formatted dialog history string.
func (m *Manager) GetDialogHistory(sessionID string) string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	session, ok := m.sessions[sessionID]
	if !ok || len(session.Turns) == 0 {
		return ""
	}

	var sb strings.Builder
	for _, turn := range session.Turns {
		sb.WriteString(fmt.Sprintf("[%s]: %s\n", turn.Role, turn.Content))
	}
	return sb.String()
}

// GetDialogState computes the current dialog state.
func (m *Manager) GetDialogState(sessionID string) *domain.DialogState {
	m.mu.RLock()
	defer m.mu.RUnlock()

	session, ok := m.sessions[sessionID]
	if !ok {
		return &domain.DialogState{}
	}

	state := &domain.DialogState{
		FilledSlots: session.CurrentSlots,
		TurnCount:   len(session.Turns),
	}

	// Determine current intent from the most recent user turn with NLU result
	for i := len(session.Turns) - 1; i >= 0; i-- {
		turn := session.Turns[i]
		if turn.Role == "user" && turn.NLUResult != nil && turn.NLUResult.Intent != nil {
			state.CurrentIntent = turn.NLUResult.Intent.TopIntent.Name
			break
		}
	}

	return state
}

// UpdateSlots merges new slot values into the session.
func (m *Manager) UpdateSlots(sessionID string, slots []domain.SlotValue) {
	m.mu.Lock()
	defer m.mu.Unlock()

	session, ok := m.sessions[sessionID]
	if !ok {
		return
	}

	for _, sv := range slots {
		session.CurrentSlots[sv.Name] = sv
	}
	session.UpdatedAt = time.Now()
}

// GetContextSummary uses the LLM to summarize the dialog context.
func (m *Manager) GetContextSummary(ctx context.Context, sessionID string, currentInput string) (string, error) {
	m.mu.RLock()
	session, ok := m.sessions[sessionID]
	m.mu.RUnlock()

	if !ok || len(session.Turns) == 0 {
		return "", nil
	}

	// Build template data
	type turnData struct {
		Role    string
		Content string
	}
	turns := make([]turnData, len(session.Turns))
	for i, t := range session.Turns {
		turns[i] = turnData{Role: t.Role, Content: t.Content}
	}

	data := struct {
		Turns        []turnData
		CurrentInput string
	}{
		Turns:        turns,
		CurrentInput: currentInput,
	}

	promptStr, err := m.promptManager.Render("dialog_context", data)
	if err != nil {
		return "", fmt.Errorf("failed to render dialog context prompt: %w", err)
	}

	resp, err := m.provider.Complete(ctx, &llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: promptStr},
		},
	})
	if err != nil {
		return "", fmt.Errorf("failed to get context summary: %w", err)
	}

	return resp.Content, nil
}

// DeleteSession removes a dialog session.
func (m *Manager) DeleteSession(sessionID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.sessions, sessionID)
}

// CleanupExpiredSessions removes sessions older than the given duration.
func (m *Manager) CleanupExpiredSessions(maxAge time.Duration) int {
	m.mu.Lock()
	defer m.mu.Unlock()

	cutoff := time.Now().Add(-maxAge)
	removed := 0
	for id, session := range m.sessions {
		if session.UpdatedAt.Before(cutoff) {
			delete(m.sessions, id)
			removed++
		}
	}
	return removed
}
