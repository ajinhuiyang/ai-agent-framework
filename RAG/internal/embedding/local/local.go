// Package local implements a pure Go embedding provider using TF-IDF with
// feature hashing. No external models or services required.
//
// How it works:
//   - Text is tokenized into words (supports Chinese + English)
//   - Each word is hashed to a fixed-dimension bucket (feature hashing / hashing trick)
//   - TF-IDF weighting is applied: TF = word count / total words, IDF learned from corpus
//   - Output vector is L2-normalized for cosine similarity compatibility
//
// This gives decent keyword-based retrieval without any ML model.
package local

import (
	"context"
	"hash/fnv"
	"math"
	"strings"
	"sync"
	"unicode"
)

// Provider implements the embedding.Provider interface using TF-IDF with feature hashing.
type Provider struct {
	dimension int
	mu        sync.RWMutex
	docCount  int            // total documents seen
	docFreq   map[uint32]int // how many documents contain each hash bucket
}

// New creates a new local TF-IDF embedding provider.
// dimension controls the vector size (higher = more accurate but more memory).
// Recommended: 2048-8192.
func New(dimension int) *Provider {
	if dimension <= 0 {
		dimension = 4096
	}
	return &Provider{
		dimension: dimension,
		docFreq:   make(map[uint32]int),
	}
}

func (p *Provider) Name() string   { return "local" }
func (p *Provider) Dimension() int { return p.dimension }

// Embed converts a single text into a TF-IDF vector.
func (p *Provider) Embed(_ context.Context, text string) ([]float32, error) {
	tokens := tokenize(text)
	vec := p.computeVector(tokens)
	return vec, nil
}

// EmbedBatch converts multiple texts and updates IDF statistics.
func (p *Provider) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	// First pass: update IDF statistics from this batch
	p.mu.Lock()
	for _, text := range texts {
		tokens := tokenize(text)
		seen := make(map[uint32]bool)
		for _, tok := range tokens {
			h := hashToken(tok, p.dimension)
			if !seen[h] {
				seen[h] = true
				p.docFreq[h]++
			}
		}
		p.docCount++
	}
	p.mu.Unlock()

	// Second pass: compute vectors
	results := make([][]float32, len(texts))
	for i, text := range texts {
		tokens := tokenize(text)
		results[i] = p.computeVector(tokens)
	}
	return results, nil
}

// computeVector builds a TF-IDF vector for the given tokens.
func (p *Provider) computeVector(tokens []string) []float32 {
	vec := make([]float32, p.dimension)
	if len(tokens) == 0 {
		return vec
	}

	// Count term frequency
	tf := make(map[uint32]int)
	for _, tok := range tokens {
		h := hashToken(tok, p.dimension)
		tf[h]++
	}

	// Compute TF-IDF
	p.mu.RLock()
	docCount := p.docCount
	if docCount < 1 {
		docCount = 1
	}
	for h, count := range tf {
		termFreq := float64(count) / float64(len(tokens))
		df := p.docFreq[h]
		if df < 1 {
			df = 1
		}
		idf := math.Log(float64(docCount)/float64(df)) + 1.0
		vec[h] = float32(termFreq * idf)
	}
	p.mu.RUnlock()

	// L2 normalize for cosine similarity
	normalize(vec)
	return vec
}

// hashToken maps a token string to a bucket index using FNV hash.
func hashToken(token string, dim int) uint32 {
	h := fnv.New32a()
	h.Write([]byte(token))
	return h.Sum32() % uint32(dim)
}

// normalize applies L2 normalization in-place.
func normalize(vec []float32) {
	var norm float64
	for _, v := range vec {
		norm += float64(v) * float64(v)
	}
	if norm < 1e-12 {
		return
	}
	norm = math.Sqrt(norm)
	for i := range vec {
		vec[i] = float32(float64(vec[i]) / norm)
	}
}

// tokenize splits text into lowercase word tokens.
// Handles both Chinese (character-level) and English (word-level).
func tokenize(text string) []string {
	text = strings.ToLower(text)
	var tokens []string
	var current strings.Builder

	for _, r := range text {
		if unicode.Is(unicode.Han, r) {
			// Chinese character: flush current word, add character as its own token
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			tokens = append(tokens, string(r))
		} else if unicode.IsLetter(r) || unicode.IsDigit(r) {
			current.WriteRune(r)
		} else {
			// Separator
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
		}
	}
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}

	// Filter stop words
	filtered := make([]string, 0, len(tokens))
	for _, t := range tokens {
		if len(t) > 0 && !isStopWord(t) {
			filtered = append(filtered, t)
		}
	}
	return filtered
}

var stopWords = map[string]bool{
	// English
	"a": true, "an": true, "the": true, "is": true, "are": true, "was": true,
	"were": true, "be": true, "been": true, "being": true, "have": true,
	"has": true, "had": true, "do": true, "does": true, "did": true,
	"will": true, "would": true, "could": true, "should": true, "may": true,
	"might": true, "shall": true, "can": true, "to": true, "of": true,
	"in": true, "for": true, "on": true, "with": true, "at": true, "by": true,
	"from": true, "as": true, "into": true, "through": true, "and": true,
	"but": true, "or": true, "not": true, "no": true, "if": true, "then": true,
	"it": true, "its": true, "this": true, "that": true, "i": true, "me": true,
	"my": true, "we": true, "you": true, "your": true, "he": true, "she": true,
	// Chinese
	"的": true, "了": true, "在": true, "是": true, "我": true, "有": true,
	"和": true, "就": true, "不": true, "人": true, "都": true, "一": true,
	"一个": true, "上": true, "也": true, "很": true, "到": true, "说": true,
	"要": true, "去": true, "你": true, "会": true, "着": true, "没有": true,
	"看": true, "好": true, "自己": true, "这": true, "他": true, "吗": true,
}

func isStopWord(word string) bool {
	return stopWords[word]
}
