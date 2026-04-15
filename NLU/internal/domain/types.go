// Package domain defines core NLU domain models and types.
package domain

import "time"

// --- Intent Recognition ---

// Intent represents a recognized user intent.
type Intent struct {
	Name       string  `json:"name"`                 // Intent identifier, e.g., "book_flight"
	Confidence float64 `json:"confidence"`           // Confidence score [0, 1]
	SubIntent  string  `json:"sub_intent,omitempty"` // Optional hierarchical sub-intent
}

// IntentResult is the output of intent recognition.
type IntentResult struct {
	TopIntent  Intent   `json:"top_intent"`
	Candidates []Intent `json:"candidates,omitempty"` // Ranked alternative intents
}

// --- Named Entity Recognition ---

// Entity represents a recognized named entity.
type Entity struct {
	Type       string  `json:"type"`                 // Entity type, e.g., "PERSON", "DATE", "LOCATION"
	Value      string  `json:"value"`                // Extracted text value
	Start      int     `json:"start"`                // Start character offset in original text
	End        int     `json:"end"`                  // End character offset
	Confidence float64 `json:"confidence"`           // Confidence score [0, 1]
	Normalized string  `json:"normalized,omitempty"` // Normalized/canonical value
}

// NERResult is the output of named entity recognition.
type NERResult struct {
	Entities []Entity `json:"entities"`
}

// --- Slot Filling ---

// SlotDefinition defines an expected slot in a dialog frame.
type SlotDefinition struct {
	Name        string   `json:"name"`                  // Slot name, e.g., "departure_city"
	Type        string   `json:"type"`                  // Expected type, e.g., "city", "date", "number"
	Required    bool     `json:"required"`              // Whether this slot must be filled
	Prompt      string   `json:"prompt,omitempty"`      // Prompt to ask user if slot is missing
	Validators  []string `json:"validators,omitempty"`  // Validation rules
	Enum        []string `json:"enum,omitempty"`        // Allowed values
	Default     string   `json:"default,omitempty"`     // Default value
	Description string   `json:"description,omitempty"` // Human-readable description
}

// SlotValue represents a filled slot.
type SlotValue struct {
	Name       string      `json:"name"`
	Value      interface{} `json:"value"`
	Confidence float64     `json:"confidence"`
	Source     string      `json:"source"` // "user", "context", "default"
	Confirmed  bool        `json:"confirmed"`
}

// SlotFillingResult is the output of slot filling.
type SlotFillingResult struct {
	FilledSlots  []SlotValue `json:"filled_slots"`
	MissingSlots []string    `json:"missing_slots,omitempty"`
	NextPrompt   string      `json:"next_prompt,omitempty"` // Prompt for next missing slot
	AllFilled    bool        `json:"all_filled"`
}

// --- Sentiment Analysis ---

// SentimentLabel represents sentiment polarity.
type SentimentLabel string

const (
	SentimentPositive SentimentLabel = "positive"
	SentimentNegative SentimentLabel = "negative"
	SentimentNeutral  SentimentLabel = "neutral"
	SentimentMixed    SentimentLabel = "mixed"
)

// SentimentResult is the output of sentiment analysis.
type SentimentResult struct {
	Label      SentimentLabel     `json:"label"`
	Score      float64            `json:"score"` // Overall sentiment score [-1, 1]
	Confidence float64            `json:"confidence"`
	Aspects    []AspectSentiment  `json:"aspects,omitempty"`  // Aspect-based sentiment
	Emotions   map[string]float64 `json:"emotions,omitempty"` // Fine-grained emotions
}

// AspectSentiment represents sentiment towards a specific aspect.
type AspectSentiment struct {
	Aspect     string         `json:"aspect"`
	Label      SentimentLabel `json:"label"`
	Score      float64        `json:"score"`
	Confidence float64        `json:"confidence"`
}

// --- Text Classification ---

// TextCategory represents a classification label.
type TextCategory struct {
	Label      string  `json:"label"`
	Confidence float64 `json:"confidence"`
}

// ClassificationResult is the output of text classification.
type ClassificationResult struct {
	TopCategory  TextCategory   `json:"top_category"`
	Categories   []TextCategory `json:"categories,omitempty"` // All ranked categories
	IsMultiLabel bool           `json:"is_multi_label"`
}

// --- Dialog Management ---

// DialogTurn represents one turn of conversation.
type DialogTurn struct {
	Role      string     `json:"role"` // "user" or "assistant"
	Content   string     `json:"content"`
	Timestamp time.Time  `json:"timestamp"`
	NLUResult *NLUResult `json:"nlu_result,omitempty"` // Attached NLU result for user turns
}

// DialogContext holds the state of a multi-turn conversation.
type DialogContext struct {
	SessionID    string               `json:"session_id"`
	Turns        []DialogTurn         `json:"turns"`
	CurrentSlots map[string]SlotValue `json:"current_slots,omitempty"`
	Metadata     map[string]string    `json:"metadata,omitempty"`
	CreatedAt    time.Time            `json:"created_at"`
	UpdatedAt    time.Time            `json:"updated_at"`
}

// DialogState represents the current dialog state.
type DialogState struct {
	CurrentIntent string               `json:"current_intent,omitempty"`
	FilledSlots   map[string]SlotValue `json:"filled_slots,omitempty"`
	PendingSlots  []string             `json:"pending_slots,omitempty"`
	TurnCount     int                  `json:"turn_count"`
	Completed     bool                 `json:"completed"`
}

// --- Unified NLU Result ---

// NLUResult is the unified output combining all NLU capabilities.
type NLUResult struct {
	Text           string                `json:"text"`               // Original input text
	Language       string                `json:"language,omitempty"` // Detected language
	Intent         *IntentResult         `json:"intent,omitempty"`
	Entities       *NERResult            `json:"entities,omitempty"`
	SlotFilling    *SlotFillingResult    `json:"slot_filling,omitempty"`
	Sentiment      *SentimentResult      `json:"sentiment,omitempty"`
	Classification *ClassificationResult `json:"classification,omitempty"`
	DialogState    *DialogState          `json:"dialog_state,omitempty"`
	ProcessingTime int64                 `json:"processing_time_ms"`   // Total processing time in ms
	ModelUsed      string                `json:"model_used,omitempty"` // Which LLM model was used
	Errors         []string              `json:"errors,omitempty"`     // Non-fatal errors
}

// --- NLU Request ---

// NLURequest is the input request for NLU processing.
type NLURequest struct {
	Text           string            `json:"text" binding:"required"`
	SessionID      string            `json:"session_id,omitempty"`
	Language       string            `json:"language,omitempty"`
	Capabilities   []string          `json:"capabilities,omitempty"` // Which NLU capabilities to run: "intent", "ner", "slot", "sentiment", "classify"
	IntentConfig   *IntentConfig     `json:"intent_config,omitempty"`
	SlotConfig     *SlotConfig       `json:"slot_config,omitempty"`
	ClassifyConfig *ClassifyConfig   `json:"classify_config,omitempty"`
	Context        map[string]string `json:"context,omitempty"` // Additional context
}

// IntentConfig provides runtime configuration for intent recognition.
type IntentConfig struct {
	CandidateIntents []string `json:"candidate_intents,omitempty"` // Limit to these intents
	TopN             int      `json:"top_n,omitempty"`             // Number of candidates to return
}

// SlotConfig provides runtime configuration for slot filling.
type SlotConfig struct {
	SlotDefinitions []SlotDefinition `json:"slot_definitions,omitempty"`
}

// ClassifyConfig provides runtime configuration for text classification.
type ClassifyConfig struct {
	Categories []string `json:"categories,omitempty"` // Target categories
	MultiLabel bool     `json:"multi_label,omitempty"`
}

// --- Schema / Domain Config ---

// DomainSchema defines the NLU domain configuration (intents, entities, slots).
type DomainSchema struct {
	Name            string             `json:"name" yaml:"name"`
	Description     string             `json:"description,omitempty" yaml:"description"`
	Intents         []IntentSchema     `json:"intents" yaml:"intents"`
	EntityTypes     []EntityTypeSchema `json:"entity_types" yaml:"entity_types"`
	SlotDefinitions []SlotDefinition   `json:"slot_definitions,omitempty" yaml:"slot_definitions"`
	Categories      []string           `json:"categories,omitempty" yaml:"categories"`
}

// IntentSchema defines a supported intent.
type IntentSchema struct {
	Name        string   `json:"name" yaml:"name"`
	Description string   `json:"description,omitempty" yaml:"description"`
	Examples    []string `json:"examples,omitempty" yaml:"examples"` // Few-shot examples
	Slots       []string `json:"slots,omitempty" yaml:"slots"`       // Associated slot names
}

// EntityTypeSchema defines a supported entity type.
type EntityTypeSchema struct {
	Name        string   `json:"name" yaml:"name"`
	Description string   `json:"description,omitempty" yaml:"description"`
	Examples    []string `json:"examples,omitempty" yaml:"examples"`
}

// --- Ask (Full Pipeline: NLU → RAG → LLM-Generation) ---

// AskRequest is the input for the full orchestrated pipeline.
type AskRequest struct {
	Text      string `json:"text" binding:"required"` // User question
	SessionID string `json:"session_id,omitempty"`    // Multi-turn conversation ID
	Language  string `json:"language,omitempty"`
	TopK      int    `json:"top_k,omitempty"`    // RAG search top-K (default 5)
	Provider  string `json:"provider,omitempty"` // LLM provider override
	Stream    bool   `json:"stream,omitempty"`   // Stream the final response
}

// AskResponse is the full orchestrated pipeline response.
type AskResponse struct {
	Answer   string     `json:"answer"`            // Final generated answer
	NLU      *NLUResult `json:"nlu,omitempty"`     // NLU analysis result
	Sources  []Source   `json:"sources,omitempty"` // RAG retrieved sources
	Provider string     `json:"provider,omitempty"`
	Model    string     `json:"model,omitempty"`
}

// Source is a piece of context retrieved by RAG.
type Source struct {
	Content string  `json:"content"`
	Source  string  `json:"source,omitempty"`
	Score   float64 `json:"score"`
}
