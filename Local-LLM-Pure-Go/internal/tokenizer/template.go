// Package tokenizer provides chat template formatting.
// It converts a list of ChatMessages into a single prompt string
// that the model expects, based on the model's chat template format.
package tokenizer

import (
	"fmt"
	"strings"

	"github.com/your-org/local-llm/internal/domain"
)

// ChatTemplate defines how to format messages for a specific model family.
type ChatTemplate struct {
	Name        string
	BOS         string // Beginning of sequence token
	EOS         string // End of sequence token
	SystemStart string
	SystemEnd   string
	UserStart   string
	UserEnd     string
	AssistStart string
	AssistEnd   string
}

// Predefined chat templates for common model families.
var (
	// ChatML format (used by Qwen, many fine-tuned models)
	ChatML = ChatTemplate{
		Name:        "chatml",
		BOS:         "",
		EOS:         "",
		SystemStart: "<|im_start|>system\n",
		SystemEnd:   "<|im_end|>\n",
		UserStart:   "<|im_start|>user\n",
		UserEnd:     "<|im_end|>\n",
		AssistStart: "<|im_start|>assistant\n",
		AssistEnd:   "<|im_end|>\n",
	}

	// Llama 2 / Llama 3 format
	Llama3 = ChatTemplate{
		Name:        "llama3",
		BOS:         "<|begin_of_text|>",
		EOS:         "<|end_of_text|>",
		SystemStart: "<|start_header_id|>system<|end_header_id|>\n\n",
		SystemEnd:   "<|eot_id|>",
		UserStart:   "<|start_header_id|>user<|end_header_id|>\n\n",
		UserEnd:     "<|eot_id|>",
		AssistStart: "<|start_header_id|>assistant<|end_header_id|>\n\n",
		AssistEnd:   "<|eot_id|>",
	}

	// Mistral / Mixtral format
	Mistral = ChatTemplate{
		Name:        "mistral",
		BOS:         "<s>",
		EOS:         "</s>",
		SystemStart: "[INST] ",
		SystemEnd:   " ",
		UserStart:   "",
		UserEnd:     " [/INST]",
		AssistStart: "",
		AssistEnd:   "</s>",
	}

	// Gemma format
	Gemma = ChatTemplate{
		Name:        "gemma",
		BOS:         "<bos>",
		EOS:         "<eos>",
		SystemStart: "",
		SystemEnd:   "",
		UserStart:   "<start_of_turn>user\n",
		UserEnd:     "<end_of_turn>\n",
		AssistStart: "<start_of_turn>model\n",
		AssistEnd:   "<end_of_turn>\n",
	}

	// Plain format (no special tokens, for simple completion models)
	Plain = ChatTemplate{
		Name:        "plain",
		SystemStart: "### System:\n",
		SystemEnd:   "\n\n",
		UserStart:   "### User:\n",
		UserEnd:     "\n\n",
		AssistStart: "### Assistant:\n",
		AssistEnd:   "\n\n",
	}
)

// templateRegistry maps template names to templates.
var templateRegistry = map[string]ChatTemplate{
	"chatml":  ChatML,
	"llama3":  Llama3,
	"mistral": Mistral,
	"gemma":   Gemma,
	"plain":   Plain,
}

// GetTemplate returns a chat template by name. Falls back to ChatML.
func GetTemplate(name string) ChatTemplate {
	name = strings.ToLower(name)
	if t, ok := templateRegistry[name]; ok {
		return t
	}
	return ChatML // Default
}

// DetectTemplate attempts to detect the correct template based on model name.
func DetectTemplate(modelName string) ChatTemplate {
	lower := strings.ToLower(modelName)

	switch {
	case strings.Contains(lower, "qwen"):
		return ChatML
	case strings.Contains(lower, "llama-3"), strings.Contains(lower, "llama3"):
		return Llama3
	case strings.Contains(lower, "llama"):
		return Llama3
	case strings.Contains(lower, "mistral"), strings.Contains(lower, "mixtral"):
		return Mistral
	case strings.Contains(lower, "gemma"):
		return Gemma
	default:
		return ChatML
	}
}

// FormatChat formats a list of chat messages into a prompt string.
func FormatChat(tmpl ChatTemplate, messages []domain.ChatMessage) string {
	var b strings.Builder

	if tmpl.BOS != "" {
		b.WriteString(tmpl.BOS)
	}

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			b.WriteString(tmpl.SystemStart)
			b.WriteString(msg.Content)
			b.WriteString(tmpl.SystemEnd)
		case "user":
			b.WriteString(tmpl.UserStart)
			b.WriteString(msg.Content)
			b.WriteString(tmpl.UserEnd)
		case "assistant":
			b.WriteString(tmpl.AssistStart)
			b.WriteString(msg.Content)
			b.WriteString(tmpl.AssistEnd)
		default:
			// Unknown role, treat as user.
			b.WriteString(tmpl.UserStart)
			b.WriteString(fmt.Sprintf("[%s] %s", msg.Role, msg.Content))
			b.WriteString(tmpl.UserEnd)
		}
	}

	// Add the assistant start token to prime the model for generation.
	b.WriteString(tmpl.AssistStart)

	return b.String()
}

// FormatGenerate formats a generate request (system + prompt) into a prompt string.
func FormatGenerate(tmpl ChatTemplate, system, prompt string) string {
	var messages []domain.ChatMessage
	if system != "" {
		messages = append(messages, domain.ChatMessage{Role: "system", Content: system})
	}
	messages = append(messages, domain.ChatMessage{Role: "user", Content: prompt})
	return FormatChat(tmpl, messages)
}
