package llm

import (
	"testing"
)

func TestProviderFactory(t *testing.T) {
	factory := NewProviderFactory()

	// List should be empty initially
	if len(factory.List()) != 0 {
		t.Errorf("expected 0 providers, got %d", len(factory.List()))
	}

	// Get non-existent provider
	_, err := factory.Get("openai")
	if err == nil {
		t.Error("expected error for non-existent provider")
	}

	// Register a mock provider
	factory.Register("test", nil)
	providers := factory.List()
	if len(providers) != 1 {
		t.Errorf("expected 1 provider, got %d", len(providers))
	}

	// Get registered (nil) provider
	p, err := factory.Get("test")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if p != nil {
		t.Error("expected nil provider")
	}
}
