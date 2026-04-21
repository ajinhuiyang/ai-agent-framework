// Package prompt manages prompt templates for the LLM Generation service.
package prompt

import (
	"bytes"
	"fmt"
	"strings"
	"text/template"

	"github.com/your-org/llm-generation/internal/domain"
)

// Manager handles prompt template rendering.
type Manager struct {
	defaultSystem string
	ragSystem     string
	templates     map[string]string // named templates (code_analyze, code_generate, etc.)
}

// New creates a new prompt Manager.
func New(defaultSystem, ragSystem string, templates map[string]string) *Manager {
	if defaultSystem == "" {
		defaultSystem = "You are a helpful, accurate, and concise AI assistant."
	}
	if templates == nil {
		templates = make(map[string]string)
	}
	return &Manager{
		defaultSystem: defaultSystem,
		ragSystem:     ragSystem,
		templates:     templates,
	}
}

// ResolveTemplate looks up a named template. If the key matches a registered
// template name, returns the template content. Otherwise returns the key as-is
// (treating it as a literal system prompt).
func (m *Manager) ResolveTemplate(key string) string {
	if tmpl, ok := m.templates[key]; ok {
		return tmpl
	}
	return key
}

// BuildMessages constructs the full message list for a generation request.
func (m *Manager) BuildMessages(req domain.GenerateRequest) []domain.Message {
	var messages []domain.Message

	// 1. System prompt
	systemPrompt := m.buildSystemPrompt(req)
	messages = append(messages, domain.Message{
		Role:    "system",
		Content: systemPrompt,
	})

	// 2. Conversation history
	for _, msg := range req.History {
		messages = append(messages, msg)
	}

	// 3. User prompt (with NLU context injected if available)
	userPrompt := m.buildUserPrompt(req)
	messages = append(messages, domain.Message{
		Role:    "user",
		Content: userPrompt,
	})

	return messages
}

// buildSystemPrompt determines the appropriate system prompt.
func (m *Manager) buildSystemPrompt(req domain.GenerateRequest) string {
	// Explicit system prompt takes priority.
	if req.SystemPrompt != "" {
		return req.SystemPrompt
	}

	// If RAG context is provided, use RAG system prompt.
	if len(req.Context) > 0 && m.ragSystem != "" {
		contextText := m.formatContext(req.Context)
		return m.renderTemplate(m.ragSystem, map[string]string{
			"Context": contextText,
		})
	}

	return m.defaultSystem
}

// buildUserPrompt enhances the user prompt with NLU metadata if available.
func (m *Manager) buildUserPrompt(req domain.GenerateRequest) string {
	prompt := req.Prompt

	// If we have NLU results, add structured context.
	if req.NLUResult != nil {
		var parts []string
		parts = append(parts, fmt.Sprintf("[Intent: %s (%.2f)]", req.NLUResult.Intent, req.NLUResult.Confidence))

		if len(req.NLUResult.Entities) > 0 {
			var entities []string
			for _, e := range req.NLUResult.Entities {
				entities = append(entities, fmt.Sprintf("%s=%s", e.Type, e.Value))
			}
			parts = append(parts, fmt.Sprintf("[Entities: %s]", strings.Join(entities, ", ")))
		}

		if len(req.NLUResult.Slots) > 0 {
			var slots []string
			for k, v := range req.NLUResult.Slots {
				slots = append(slots, fmt.Sprintf("%s=%s", k, v))
			}
			parts = append(parts, fmt.Sprintf("[Slots: %s]", strings.Join(slots, ", ")))
		}

		prompt = strings.Join(parts, " ") + "\n\n" + prompt
	}

	// If RAG context is provided but not in system prompt, add inline.
	if len(req.Context) > 0 && req.SystemPrompt != "" {
		contextText := m.formatContext(req.Context)
		prompt = "Reference context:\n" + contextText + "\n\nQuestion: " + prompt
	}

	return prompt
}

// formatContext formats RAG context items into a readable string.
func (m *Manager) formatContext(items []domain.ContextItem) string {
	var parts []string
	for i, item := range items {
		part := fmt.Sprintf("[%d] %s", i+1, item.Content)
		if item.Source != "" {
			part += fmt.Sprintf(" (source: %s)", item.Source)
		}
		parts = append(parts, part)
	}
	return strings.Join(parts, "\n\n")
}

// renderTemplate renders a Go template string with the given data.
func (m *Manager) renderTemplate(tmplStr string, data map[string]string) string {
	tmpl, err := template.New("prompt").Parse(tmplStr)
	if err != nil {
		return tmplStr
	}
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return tmplStr
	}
	return buf.String()
}
