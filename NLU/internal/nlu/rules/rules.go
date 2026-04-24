// Package rules implements rule-based NLU (intent, NER, sentiment) using keyword
// matching instead of LLM calls. This makes NLU processing instant (~0ms) so
// that only the final answer generation needs an LLM call.
//
// The rules are loaded from the domain.yaml schema at startup.
package rules

import (
	"strings"
	"unicode"

	"go.uber.org/zap"

	"github.com/your-org/nlu/internal/domain"
)

// Engine provides rule-based intent recognition, entity extraction, and sentiment analysis.
type Engine struct {
	schema   *domain.DomainSchema
	logger   *zap.Logger
	entities []entityRule
}

// entityRule is a precompiled entity matching rule.
type entityRule struct {
	Type  string
	Value string
	Lower string // precomputed lowercase for matching
}

// New creates a new rule-based NLU engine.
func New(schema *domain.DomainSchema, logger *zap.Logger) *Engine {
	e := &Engine{
		schema: schema,
		logger: logger,
	}
	e.compileEntityRules()
	return e
}

// compileEntityRules precomputes lowercase entity values for fast matching.
func (e *Engine) compileEntityRules() {
	if e.schema == nil {
		return
	}
	for _, et := range e.schema.EntityTypes {
		for _, ex := range et.Examples {
			e.entities = append(e.entities, entityRule{
				Type:  et.Name,
				Value: ex,
				Lower: strings.ToLower(ex),
			})
		}
	}
	// Sort by length descending so longer matches win (e.g. "Spring Boot" before "Spring")
	for i := 0; i < len(e.entities); i++ {
		for j := i + 1; j < len(e.entities); j++ {
			if len(e.entities[j].Lower) > len(e.entities[i].Lower) {
				e.entities[i], e.entities[j] = e.entities[j], e.entities[i]
			}
		}
	}
}

// ---------- Intent Recognition ----------

// RecognizeIntent matches the user text against domain.yaml intent examples using
// keyword overlap scoring. Returns the best matching intent.
func (e *Engine) RecognizeIntent(text string) *domain.IntentResult {
	if e.schema == nil || len(e.schema.Intents) == 0 {
		return &domain.IntentResult{
			TopIntent: domain.Intent{Name: "unknown", Confidence: 0.0},
		}
	}

	lower := strings.ToLower(text)
	var bestIntent string
	bestScore := 0.0
	var candidates []domain.Intent

	for _, intent := range e.schema.Intents {
		if intent.Name == "unknown" {
			continue
		}
		score := e.scoreIntent(lower, intent)
		if score > 0 {
			candidates = append(candidates, domain.Intent{
				Name:       intent.Name,
				Confidence: score,
			})
		}
		if score > bestScore {
			bestScore = score
			bestIntent = intent.Name
		}
	}

	if bestIntent == "" {
		bestIntent = "ask_question" // default fallback for programming questions
		bestScore = 0.3
	}

	// Sort candidates by confidence descending
	for i := 0; i < len(candidates); i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].Confidence > candidates[i].Confidence {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	result := &domain.IntentResult{
		TopIntent: domain.Intent{
			Name:       bestIntent,
			Confidence: bestScore,
		},
		Candidates: candidates,
	}

	e.logger.Debug("intent recognized (rules)",
		zap.String("intent", bestIntent),
		zap.Float64("confidence", bestScore),
	)

	return result
}

// scoreIntent computes a match score between user text and an intent's examples.
func (e *Engine) scoreIntent(textLower string, intent domain.IntentSchema) float64 {
	if len(intent.Examples) == 0 {
		return 0
	}

	bestMatch := 0.0
	for _, example := range intent.Examples {
		exLower := strings.ToLower(example)
		score := e.textSimilarity(textLower, exLower)
		if score > bestMatch {
			bestMatch = score
		}
	}
	return bestMatch
}

// textSimilarity computes keyword overlap ratio between two strings.
func (e *Engine) textSimilarity(text, example string) float64 {
	// Exact substring match
	if strings.Contains(text, example) || strings.Contains(example, text) {
		shorter := len(example)
		longer := len(text)
		if shorter > longer {
			shorter, longer = longer, shorter
		}
		return 0.5 + 0.5*float64(shorter)/float64(longer)
	}

	// Word overlap
	textWords := tokenize(text)
	exWords := tokenize(example)
	if len(exWords) == 0 {
		return 0
	}

	matches := 0
	for _, ew := range exWords {
		for _, tw := range textWords {
			if tw == ew {
				matches++
				break
			}
		}
	}

	if matches == 0 {
		return 0
	}
	// Jaccard-like score
	return float64(matches) / float64(len(exWords))
}

// tokenize splits text into word tokens, handling both Chinese and English.
// Chinese text is split into bigrams (2-char sliding window) for better matching,
// English text is split on whitespace/punctuation as before.
func tokenize(text string) []string {
	// Split on common delimiters
	fields := strings.FieldsFunc(text, func(r rune) bool {
		return r == ' ' || r == '\t' || r == ',' || r == '.' || r == '?' ||
			r == '!' || r == '、' || r == '，' || r == '。' || r == '？' || r == '！'
	})
	result := make([]string, 0, len(fields)*2)
	for _, f := range fields {
		f = strings.TrimSpace(f)
		if f == "" {
			continue
		}
		lower := strings.ToLower(f)

		// Check if this segment contains CJK characters
		runes := []rune(lower)
		hasCJK := false
		for _, r := range runes {
			if isCJK(r) {
				hasCJK = true
				break
			}
		}

		if hasCJK {
			// For CJK text: emit individual characters and bigrams
			for i, r := range runes {
				if isCJK(r) {
					// Single character
					result = append(result, string(r))
					// Bigram (2-char window)
					if i+1 < len(runes) && isCJK(runes[i+1]) {
						result = append(result, string(runes[i:i+2]))
					}
				}
			}
			// Also add the full segment for exact matching
			result = append(result, lower)
		} else {
			result = append(result, lower)
		}
	}
	return result
}

// isCJK returns true if the rune is a CJK Unified Ideograph.
func isCJK(r rune) bool {
	return unicode.Is(unicode.Han, r)
}

// ---------- Entity Extraction ----------

// ExtractEntities finds known entities in the text by matching against
// domain.yaml entity_types examples.
func (e *Engine) ExtractEntities(text string) *domain.NERResult {
	result := &domain.NERResult{}
	if e.schema == nil {
		return result
	}

	lower := strings.ToLower(text)
	matched := make(map[string]bool) // avoid duplicate matches

	for _, rule := range e.entities {
		if matched[rule.Lower] {
			continue
		}
		idx := strings.Index(lower, rule.Lower)
		if idx >= 0 {
			matched[rule.Lower] = true
			result.Entities = append(result.Entities, domain.Entity{
				Type:       rule.Type,
				Value:      rule.Value,
				Start:      idx,
				End:        idx + len(rule.Value),
				Confidence: 1.0,
			})
		}
	}

	e.logger.Debug("entities extracted (rules)", zap.Int("count", len(result.Entities)))
	return result
}

// ---------- Sentiment Analysis ----------

// negative/positive keyword lists for simple rule-based sentiment.
var (
	negativeKeywords = []string{
		// Chinese
		"报错", "错误", "失败", "异常", "崩溃", "出问题", "不行", "不工作", "有问题",
		"挂了", "卡死", "死循环", "泄漏", "溢出", "超时", "拒绝", "无法",
		// English
		"error", "fail", "crash", "bug", "broken", "stuck", "timeout",
		"exception", "panic", "leak", "overflow", "refused", "cannot", "unable",
	}
	positiveKeywords = []string{
		// Chinese
		"感谢", "谢谢", "太好了", "完美", "不错", "很好", "厉害", "优秀",
		// English
		"thanks", "thank", "great", "perfect", "awesome", "excellent", "nice", "good",
	}
)

// AnalyzeSentiment performs keyword-based sentiment analysis.
func (e *Engine) AnalyzeSentiment(text string) *domain.SentimentResult {
	lower := strings.ToLower(text)

	negCount := 0
	posCount := 0
	for _, kw := range negativeKeywords {
		if strings.Contains(lower, kw) {
			negCount++
		}
	}
	for _, kw := range positiveKeywords {
		if strings.Contains(lower, kw) {
			posCount++
		}
	}

	var label domain.SentimentLabel
	var score float64

	switch {
	case negCount > 0 && posCount > 0:
		label = domain.SentimentMixed
		score = 0.0
	case negCount > 0:
		label = domain.SentimentNegative
		score = -0.5 - 0.1*float64(negCount)
		if score < -1 {
			score = -1
		}
	case posCount > 0:
		label = domain.SentimentPositive
		score = 0.5 + 0.1*float64(posCount)
		if score > 1 {
			score = 1
		}
	default:
		label = domain.SentimentNeutral
		score = 0.0
	}

	e.logger.Debug("sentiment analyzed (rules)",
		zap.String("label", string(label)),
		zap.Float64("score", score),
	)

	return &domain.SentimentResult{
		Label:      label,
		Score:      score,
		Confidence: 0.8,
	}
}
