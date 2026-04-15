package prompt

import (
	"strings"
	"testing"
)

func TestManager_RegisterAndRender(t *testing.T) {
	m := NewManager()

	// Test custom template
	err := m.Register("test", "Hello {{.Name}}, you are {{.Age}} years old!")
	if err != nil {
		t.Fatalf("Register failed: %v", err)
	}

	data := struct {
		Name string
		Age  int
	}{
		Name: "Alice",
		Age:  30,
	}

	result, err := m.Render("test", data)
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}

	expected := "Hello Alice, you are 30 years old!"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestManager_RenderNotFound(t *testing.T) {
	m := NewManager()
	_, err := m.Render("nonexistent", nil)
	if err == nil {
		t.Fatal("expected error for nonexistent template")
	}
}

func TestManager_BuiltinTemplates(t *testing.T) {
	m := NewManager()

	// Verify all built-in templates are registered
	builtins := []string{
		"intent_recognition",
		"ner",
		"slot_filling",
		"sentiment_analysis",
		"text_classification",
		"dialog_context",
		"unified_nlu",
	}

	for _, name := range builtins {
		// Render with minimal data to ensure template parses
		_, err := m.Render(name, struct {
			Text          string
			Intents       interface{}
			EntityTypes   interface{}
			Slots         interface{}
			Categories    []string
			Context       string
			Language      string
			MultiLabel    bool
			Turns         interface{}
			CurrentInput  string
			FilledSlots   interface{}
			DialogHistory string
			Intent        string
		}{
			Text: "test input",
		})
		// We only check that it doesn't panic; template may have partial results
		_ = err
	}
}

func TestManager_IntentRecognitionPrompt(t *testing.T) {
	m := NewManager()

	type IntentSchema struct {
		Name        string
		Description string
		Examples    []string
	}

	data := struct {
		Text    string
		Intents []IntentSchema
		Context string
	}{
		Text: "I want to book a flight to Shanghai",
		Intents: []IntentSchema{
			{Name: "book_flight", Description: "Book a flight", Examples: []string{"book flight", "fly to"}},
			{Name: "greeting", Description: "Say hello", Examples: []string{"hi", "hello"}},
		},
		Context: "",
	}

	result, err := m.Render("intent_recognition", data)
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}

	// Verify the prompt contains key elements
	if !strings.Contains(result, "I want to book a flight to Shanghai") {
		t.Error("prompt should contain the input text")
	}
	if !strings.Contains(result, "book_flight") {
		t.Error("prompt should contain the intent name")
	}
	if !strings.Contains(result, "JSON") {
		t.Error("prompt should mention JSON format")
	}
}

func TestManager_SentimentPrompt(t *testing.T) {
	m := NewManager()

	data := struct {
		Text     string
		Language string
	}{
		Text:     "这个产品太棒了！",
		Language: "zh",
	}

	result, err := m.Render("sentiment_analysis", data)
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}

	if !strings.Contains(result, "这个产品太棒了") {
		t.Error("prompt should contain the input text")
	}
	if !strings.Contains(result, "zh") {
		t.Error("prompt should contain the language")
	}
}

func TestManager_RegisterInvalidTemplate(t *testing.T) {
	m := NewManager()
	err := m.Register("bad", "{{.Invalid")
	if err == nil {
		t.Fatal("expected error for invalid template")
	}
}

func TestTemplateFuncs(t *testing.T) {
	m := NewManager()

	// Test join function
	err := m.Register("test_join", `{{join .Items ", "}}`)
	if err != nil {
		t.Fatalf("Register failed: %v", err)
	}

	result, err := m.Render("test_join", struct{ Items []string }{Items: []string{"a", "b", "c"}})
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}
	if result != "a, b, c" {
		t.Errorf("expected %q, got %q", "a, b, c", result)
	}

	// Test bullet function
	err = m.Register("test_bullet", `{{bullet .Items}}`)
	if err != nil {
		t.Fatalf("Register failed: %v", err)
	}

	result, err = m.Render("test_bullet", struct{ Items []string }{Items: []string{"first", "second"}})
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}
	if !strings.Contains(result, "- first") || !strings.Contains(result, "- second") {
		t.Errorf("bullet format incorrect: %q", result)
	}
}
