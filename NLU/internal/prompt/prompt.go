// Package prompt provides template-based prompt generation for NLU tasks.
package prompt

import (
	"bytes"
	"fmt"
	"strings"
	"text/template"
)

// Template wraps Go's text/template with NLU-specific helpers.
type Template struct {
	name string
	tmpl *template.Template
	raw  string
}

// Manager manages multiple prompt templates.
type Manager struct {
	templates map[string]*Template
}

// NewManager creates a new prompt template manager with built-in NLU templates.
func NewManager() *Manager {
	m := &Manager{
		templates: make(map[string]*Template),
	}
	m.registerBuiltinTemplates()
	return m
}

// Register adds a custom template.
func (m *Manager) Register(name, tmplStr string) error {
	t, err := template.New(name).Funcs(templateFuncs()).Parse(tmplStr)
	if err != nil {
		return fmt.Errorf("failed to parse template %q: %w", name, err)
	}
	m.templates[name] = &Template{
		name: name,
		tmpl: t,
		raw:  tmplStr,
	}
	return nil
}

// Render renders a template by name with the given data.
func (m *Manager) Render(name string, data interface{}) (string, error) {
	t, ok := m.templates[name]
	if !ok {
		return "", fmt.Errorf("template %q not found", name)
	}
	var buf bytes.Buffer
	if err := t.tmpl.Execute(&buf, data); err != nil {
		return "", fmt.Errorf("failed to render template %q: %w", name, err)
	}
	return strings.TrimSpace(buf.String()), nil
}

// templateFuncs returns custom template functions for NLU prompts.
func templateFuncs() template.FuncMap {
	return template.FuncMap{
		"join":  strings.Join,
		"lower": strings.ToLower,
		"upper": strings.ToUpper,
		"quote": func(s string) string { return fmt.Sprintf("%q", s) },
		"bullet": func(items []string) string {
			var sb strings.Builder
			for _, item := range items {
				sb.WriteString("- ")
				sb.WriteString(item)
				sb.WriteString("\n")
			}
			return sb.String()
		},
		"numbered": func(items []string) string {
			var sb strings.Builder
			for i, item := range items {
				sb.WriteString(fmt.Sprintf("%d. %s\n", i+1, item))
			}
			return sb.String()
		},
	}
}

// --- Built-in NLU Prompt Templates ---

func (m *Manager) registerBuiltinTemplates() {
	// Intent Recognition
	_ = m.Register("intent_recognition", intentRecognitionPrompt)
	// Named Entity Recognition
	_ = m.Register("ner", nerPrompt)
	// Slot Filling
	_ = m.Register("slot_filling", slotFillingPrompt)
	// Sentiment Analysis
	_ = m.Register("sentiment_analysis", sentimentAnalysisPrompt)
	// Text Classification
	_ = m.Register("text_classification", textClassificationPrompt)
	// Dialog Context Summary
	_ = m.Register("dialog_context", dialogContextPrompt)
	// Unified NLU
	_ = m.Register("unified_nlu", unifiedNLUPrompt)
}

// --- Intent Recognition Prompt ---
const intentRecognitionPrompt = `You are an intent recognition system. Analyze the user's input and identify the intent.

{{if .Intents}}Available intents:
{{range .Intents}}- {{.Name}}{{if .Description}}: {{.Description}}{{end}}
{{if .Examples}}  Examples: {{join .Examples ", "}}
{{end}}{{end}}{{end}}

{{if .Context}}Conversation context:
{{.Context}}
{{end}}

User input: "{{.Text}}"

Respond in JSON format:
{
  "intent": "<intent_name>",
  "confidence": <0.0-1.0>,
  "sub_intent": "<optional_sub_intent>",
  "candidates": [
    {"intent": "<alt_intent>", "confidence": <score>}
  ],
  "reasoning": "<brief explanation>"
}`

// --- NER Prompt ---
const nerPrompt = `You are a named entity recognition (NER) system. Extract all named entities from the given text.

{{if .EntityTypes}}Entity types to look for:
{{range .EntityTypes}}- {{.Name}}{{if .Description}}: {{.Description}}{{end}}
{{if .Examples}}  Examples: {{join .Examples ", "}}
{{end}}{{end}}{{end}}

{{if .Language}}Language: {{.Language}}
{{end}}

Text: "{{.Text}}"

Respond in JSON format:
{
  "entities": [
    {
      "type": "<entity_type>",
      "value": "<extracted_text>",
      "start": <start_char_offset>,
      "end": <end_char_offset>,
      "confidence": <0.0-1.0>,
      "normalized": "<canonical_form_if_applicable>"
    }
  ]
}`

// --- Slot Filling Prompt ---
const slotFillingPrompt = `You are a slot filling system for task-oriented dialogue. Extract slot values from the user's input.

{{if .Intent}}Current intent: {{.Intent}}
{{end}}

Required slots:
{{range .Slots}}- {{.Name}} (type: {{.Type}}){{if .Description}}: {{.Description}}{{end}}{{if .Enum}} [allowed: {{join .Enum ", "}}]{{end}}{{if not .Required}} (optional){{end}}
{{end}}

{{if .FilledSlots}}Already filled slots:
{{range $name, $val := .FilledSlots}}- {{$name}}: {{$val}}
{{end}}{{end}}

{{if .DialogHistory}}Recent conversation:
{{.DialogHistory}}
{{end}}

User input: "{{.Text}}"

Respond in JSON format:
{
  "filled_slots": [
    {
      "name": "<slot_name>",
      "value": "<extracted_value>",
      "confidence": <0.0-1.0>,
      "source": "user"
    }
  ],
  "missing_slots": ["<slot_name>"],
  "next_prompt": "<question to ask for next missing slot>",
  "all_filled": <true|false>
}`

// --- Sentiment Analysis Prompt ---
const sentimentAnalysisPrompt = `You are a sentiment analysis system. Analyze the sentiment of the given text.

{{if .Language}}Language: {{.Language}}
{{end}}

Text: "{{.Text}}"

Respond in JSON format:
{
  "label": "<positive|negative|neutral|mixed>",
  "score": <-1.0 to 1.0>,
  "confidence": <0.0-1.0>,
  "aspects": [
    {
      "aspect": "<aspect_name>",
      "label": "<positive|negative|neutral>",
      "score": <-1.0 to 1.0>,
      "confidence": <0.0-1.0>
    }
  ],
  "emotions": {
    "joy": <0.0-1.0>,
    "anger": <0.0-1.0>,
    "sadness": <0.0-1.0>,
    "fear": <0.0-1.0>,
    "surprise": <0.0-1.0>,
    "disgust": <0.0-1.0>
  }
}`

// --- Text Classification Prompt ---
const textClassificationPrompt = `You are a text classification system. Classify the given text into one or more categories.

Available categories:
{{range .Categories}}- {{.}}
{{end}}

{{if .MultiLabel}}This is multi-label classification. A text can belong to multiple categories.
{{else}}Choose exactly ONE best-fitting category.
{{end}}

Text: "{{.Text}}"

Respond in JSON format:
{
  "top_category": {
    "label": "<category_name>",
    "confidence": <0.0-1.0>
  },
  "categories": [
    {"label": "<category>", "confidence": <score>}
  ],
  "is_multi_label": {{if .MultiLabel}}true{{else}}false{{end}}
}`

// --- Dialog Context Prompt ---
const dialogContextPrompt = `Summarize the following conversation context for NLU processing.

Conversation history:
{{range .Turns}}[{{.Role}}]: {{.Content}}
{{end}}

Current user input: "{{.CurrentInput}}"

Provide a brief summary that captures:
1. The main topic/intent of the conversation
2. Key entities and slot values mentioned
3. Any context changes or topic switches

Respond in JSON format:
{
  "summary": "<brief_summary>",
  "main_intent": "<detected_overall_intent>",
  "accumulated_entities": [{"type": "<type>", "value": "<value>"}],
  "topic_changed": <true|false>
}`

// --- Unified NLU Prompt ---
const unifiedNLUPrompt = `You are a comprehensive Natural Language Understanding (NLU) system. Perform all the following analyses on the given text simultaneously.

{{if .Intents}}Available intents:
{{range .Intents}}- {{.Name}}{{if .Description}}: {{.Description}}{{end}}
{{end}}{{end}}

{{if .EntityTypes}}Entity types:
{{range .EntityTypes}}- {{.Name}}{{if .Description}}: {{.Description}}{{end}}
{{end}}{{end}}

{{if .Slots}}Slots to fill:
{{range .Slots}}- {{.Name}} ({{.Type}}){{if .Description}}: {{.Description}}{{end}}
{{end}}{{end}}

{{if .Categories}}Classification categories:
{{range .Categories}}- {{.}}
{{end}}{{end}}

{{if .Context}}Conversation context:
{{.Context}}
{{end}}

{{if .Language}}Language: {{.Language}}
{{end}}

User input: "{{.Text}}"

Perform ALL of the following analyses and respond in JSON format:
{
  "intent": {
    "name": "<intent>",
    "confidence": <0.0-1.0>,
    "candidates": [{"name": "<alt>", "confidence": <score>}]
  },
  "entities": [
    {"type": "<type>", "value": "<text>", "start": <int>, "end": <int>, "confidence": <0.0-1.0>}
  ],
  "sentiment": {
    "label": "<positive|negative|neutral|mixed>",
    "score": <-1.0 to 1.0>,
    "confidence": <0.0-1.0>
  },
  "classification": {
    "label": "<category>",
    "confidence": <0.0-1.0>
  }
}`
