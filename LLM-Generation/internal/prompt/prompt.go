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
		defaultSystem = "You are a helpful code assistant. Answer in the same language as the user's question. If the user asks in Chinese, reply in Chinese."
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
		// 先尝试将 SystemPrompt 当作 template key 来解析 (e.g. "code_generate")
		if tmpl, ok := m.templates[req.SystemPrompt]; ok {
			// 将 default_system（含语言约束等通用规则）与具体模板拼接。
			// 使用明确的分隔，让模型理解这是两层指令。
			return m.defaultSystem + "\n## 本次任务要求\n" + tmpl
		}
		// 不是已知 key，当作字面 system prompt 使用
		return m.defaultSystem + "\n## 本次任务要求\n" + req.SystemPrompt
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

// buildUserPrompt returns the user prompt without NLU metadata injection.
// NLU metadata (intent, entities) was previously prepended to the prompt,
// but testing shows it adds unnecessary tokens without improving quality.
// The intent is already used for system prompt template selection.
func (m *Manager) buildUserPrompt(req domain.GenerateRequest) string {
	prompt := req.Prompt

	// If RAG context is provided but not in system prompt, add inline.
	if len(req.Context) > 0 && req.SystemPrompt != "" {
		contextText := m.formatContext(req.Context)
		prompt = "Reference context:\n" + contextText + "\n\nQuestion: " + prompt
	}

	return prompt
}

// formatContext formats RAG context items into a readable string.
// Limits to top 3 results to keep system prompt concise and reduce prefill tokens.
func (m *Manager) formatContext(items []domain.ContextItem) string {
	var parts []string
	limit := 3
	if len(items) < limit {
		limit = len(items)
	}
	for i := 0; i < limit; i++ {
		part := fmt.Sprintf("[%d] %s", i+1, items[i].Content)
		if items[i].Source != "" {
			part += fmt.Sprintf(" (source: %s)", items[i].Source)
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
