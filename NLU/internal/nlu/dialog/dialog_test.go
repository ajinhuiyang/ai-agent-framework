package dialog

import (
	"testing"
	"time"

	"go.uber.org/zap"

	"github.com/your-org/nlu/internal/domain"
	"github.com/your-org/nlu/internal/prompt"
)

func newTestManager() *Manager {
	logger, _ := zap.NewDevelopment()
	pm := prompt.NewManager()
	return New(nil, pm, logger, 10)
}

func TestManager_GetOrCreateSession(t *testing.T) {
	m := newTestManager()

	// Create a new session
	session := m.GetOrCreateSession("test-session-1")
	if session.SessionID != "test-session-1" {
		t.Errorf("expected session ID %q, got %q", "test-session-1", session.SessionID)
	}

	// Get the same session
	session2 := m.GetOrCreateSession("test-session-1")
	if session2.SessionID != session.SessionID {
		t.Error("expected same session")
	}

	// Auto-generate session ID
	session3 := m.GetOrCreateSession("")
	if session3.SessionID == "" {
		t.Error("expected auto-generated session ID")
	}
}

func TestManager_AddTurns(t *testing.T) {
	m := newTestManager()
	session := m.GetOrCreateSession("test-session")

	m.AddUserTurn("test-session", "Hello!", nil)
	m.AddAssistantTurn("test-session", "Hi there!")

	if len(session.Turns) != 2 {
		t.Fatalf("expected 2 turns, got %d", len(session.Turns))
	}

	if session.Turns[0].Role != "user" {
		t.Errorf("expected user role, got %q", session.Turns[0].Role)
	}
	if session.Turns[0].Content != "Hello!" {
		t.Errorf("expected %q, got %q", "Hello!", session.Turns[0].Content)
	}
	if session.Turns[1].Role != "assistant" {
		t.Errorf("expected assistant role, got %q", session.Turns[1].Role)
	}
}

func TestManager_MaxTurns(t *testing.T) {
	m := newTestManager() // maxTurns = 10

	m.GetOrCreateSession("test-session")

	// Add 15 turns
	for i := 0; i < 15; i++ {
		m.AddUserTurn("test-session", "message", nil)
	}

	session := m.GetOrCreateSession("test-session")
	if len(session.Turns) > 10 {
		t.Errorf("expected at most 10 turns, got %d", len(session.Turns))
	}
}

func TestManager_GetDialogHistory(t *testing.T) {
	m := newTestManager()
	m.GetOrCreateSession("test-session")

	// Empty history
	history := m.GetDialogHistory("test-session")
	if history != "" {
		t.Error("expected empty history")
	}

	m.AddUserTurn("test-session", "Hello", nil)
	m.AddAssistantTurn("test-session", "Hi!")

	history = m.GetDialogHistory("test-session")
	if history == "" {
		t.Error("expected non-empty history")
	}
}

func TestManager_GetDialogState(t *testing.T) {
	m := newTestManager()
	m.GetOrCreateSession("test-session")

	nluResult := &domain.NLUResult{
		Intent: &domain.IntentResult{
			TopIntent: domain.Intent{Name: "greeting", Confidence: 0.95},
		},
	}

	m.AddUserTurn("test-session", "Hello", nluResult)

	state := m.GetDialogState("test-session")
	if state.CurrentIntent != "greeting" {
		t.Errorf("expected intent %q, got %q", "greeting", state.CurrentIntent)
	}
	if state.TurnCount != 1 {
		t.Errorf("expected 1 turn, got %d", state.TurnCount)
	}
}

func TestManager_UpdateSlots(t *testing.T) {
	m := newTestManager()
	m.GetOrCreateSession("test-session")

	m.UpdateSlots("test-session", []domain.SlotValue{
		{Name: "city", Value: "Shanghai", Confidence: 0.9},
		{Name: "date", Value: "tomorrow", Confidence: 0.85},
	})

	state := m.GetDialogState("test-session")
	if len(state.FilledSlots) != 2 {
		t.Fatalf("expected 2 filled slots, got %d", len(state.FilledSlots))
	}

	if state.FilledSlots["city"].Value != "Shanghai" {
		t.Errorf("expected city %q, got %v", "Shanghai", state.FilledSlots["city"].Value)
	}
}

func TestManager_DeleteSession(t *testing.T) {
	m := newTestManager()
	m.GetOrCreateSession("test-session")

	m.DeleteSession("test-session")

	state := m.GetDialogState("test-session")
	if state.TurnCount != 0 {
		t.Error("expected empty state after deletion")
	}
}

func TestManager_CleanupExpiredSessions(t *testing.T) {
	m := newTestManager()

	m.GetOrCreateSession("session-1")
	m.GetOrCreateSession("session-2")

	// Force session-1 to be old
	m.mu.Lock()
	m.sessions["session-1"].UpdatedAt = time.Now().Add(-2 * time.Hour)
	m.mu.Unlock()

	removed := m.CleanupExpiredSessions(1 * time.Hour)
	if removed != 1 {
		t.Errorf("expected 1 removed, got %d", removed)
	}

	// session-2 should still exist
	state := m.GetDialogState("session-2")
	if state.TurnCount != 0 {
		t.Error("session-2 should still exist")
	}

	// session-1 should be gone
	history := m.GetDialogHistory("session-1")
	if history != "" {
		t.Error("session-1 should be deleted")
	}
}

func TestManager_NonExistentSession(t *testing.T) {
	m := newTestManager()

	// These should not panic
	m.AddUserTurn("nonexistent", "hello", nil)
	m.AddAssistantTurn("nonexistent", "hi")
	m.UpdateSlots("nonexistent", nil)

	history := m.GetDialogHistory("nonexistent")
	if history != "" {
		t.Error("expected empty history for nonexistent session")
	}

	state := m.GetDialogState("nonexistent")
	if state.TurnCount != 0 {
		t.Error("expected empty state for nonexistent session")
	}
}
