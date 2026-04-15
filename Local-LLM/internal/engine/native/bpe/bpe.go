// Package bpe implements a Byte-Pair Encoding tokenizer that reads vocabulary
// directly from GGUF file metadata.
//
// GGUF files store tokenizer data in metadata keys:
//
//	tokenizer.ggml.model       = "gpt2" | "llama" | ...
//	tokenizer.ggml.tokens      = [string array of tokens]
//	tokenizer.ggml.scores      = [float32 array of token scores/priorities]
//	tokenizer.ggml.merges      = [string array of merge rules] (for BPE)
//	tokenizer.ggml.bos_token_id = uint32
//	tokenizer.ggml.eos_token_id = uint32
package bpe

import (
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/your-org/local-llm/internal/engine/native/gguf"
)

// Tokenizer implements BPE tokenization using vocabulary from a GGUF file.
type Tokenizer struct {
	// Token ID → string
	idToToken []string
	// String → token ID
	tokenToID map[string]int
	// Token scores (used for BPE merge priority)
	scores []float32
	// Merge rules: "token1 token2" → merged token
	merges map[string]int // merge pair → rank (lower = higher priority)
	// Special tokens: "<|im_start|>" → token ID (built lazily)
	specialTokens map[string]int

	bosID int
	eosID int
	padID int

	model string // "gpt2", "llama", etc.
}

// NewFromGGUF creates a tokenizer from GGUF file metadata.
func NewFromGGUF(gf *gguf.File) (*Tokenizer, error) {
	t := &Tokenizer{
		tokenToID: make(map[string]int),
		merges:    make(map[string]int),
		bosID:     1, // default
		eosID:     2, // default
		padID:     -1,
	}

	// Read tokenizer model type.
	if model, ok := gf.GetString("tokenizer.ggml.model"); ok {
		t.model = model
	}

	// Read tokens.
	tokensRaw, ok := gf.Metadata["tokenizer.ggml.tokens"]
	if !ok {
		return nil, fmt.Errorf("missing tokenizer.ggml.tokens in GGUF metadata")
	}
	tokensArr, ok := tokensRaw.([]interface{})
	if !ok {
		return nil, fmt.Errorf("tokenizer.ggml.tokens is not an array")
	}

	t.idToToken = make([]string, len(tokensArr))
	for i, v := range tokensArr {
		s, ok := v.(string)
		if !ok {
			s = fmt.Sprintf("<token_%d>", i)
		}
		t.idToToken[i] = s
		t.tokenToID[s] = i
	}

	// Read scores.
	if scoresRaw, ok := gf.Metadata["tokenizer.ggml.scores"]; ok {
		if scoresArr, ok := scoresRaw.([]interface{}); ok {
			t.scores = make([]float32, len(scoresArr))
			for i, v := range scoresArr {
				switch val := v.(type) {
				case float32:
					t.scores[i] = val
				case float64:
					t.scores[i] = float32(val)
				}
			}
		}
	}

	// Read merges (for BPE models like GPT-2).
	if mergesRaw, ok := gf.Metadata["tokenizer.ggml.merges"]; ok {
		if mergesArr, ok := mergesRaw.([]interface{}); ok {
			for rank, v := range mergesArr {
				if s, ok := v.(string); ok {
					t.merges[s] = rank
				}
			}
		}
	}

	// Read special token IDs.
	if v, ok := gf.GetUint32("tokenizer.ggml.bos_token_id"); ok {
		t.bosID = int(v)
	}
	if v, ok := gf.GetUint32("tokenizer.ggml.eos_token_id"); ok {
		t.eosID = int(v)
	}
	if v, ok := gf.GetUint32("tokenizer.ggml.padding_token_id"); ok {
		t.padID = int(v)
	}

	return t, nil
}

// Encode converts text to a list of token IDs.
// Handles special tokens (like <|im_start|>) by splitting on them first.
func (t *Tokenizer) Encode(text string) []int {
	if text == "" {
		return nil
	}

	// Split text on special tokens, encode each segment separately.
	return t.encodeWithSpecialTokens(text)
}

// encodeWithSpecialTokens splits on special tokens and encodes segments.
func (t *Tokenizer) encodeWithSpecialTokens(text string) []int {
	// Build list of special tokens to look for (tokens containing <| or similar patterns).
	if t.specialTokens == nil {
		t.buildSpecialTokens()
	}

	var result []int
	remaining := text

	for len(remaining) > 0 {
		// Find the earliest special token in remaining text.
		bestPos := -1
		bestLen := 0
		bestID := -1

		for token, id := range t.specialTokens {
			pos := strings.Index(remaining, token)
			if pos >= 0 && (bestPos < 0 || pos < bestPos || (pos == bestPos && len(token) > bestLen)) {
				bestPos = pos
				bestLen = len(token)
				bestID = id
			}
		}

		if bestPos < 0 {
			// No more special tokens, encode the rest.
			result = append(result, t.encodeSegment(remaining)...)
			break
		}

		// Encode text before the special token.
		if bestPos > 0 {
			result = append(result, t.encodeSegment(remaining[:bestPos])...)
		}

		// Add the special token ID directly.
		result = append(result, bestID)

		// Move past the special token.
		remaining = remaining[bestPos+bestLen:]
	}

	return result
}

// encodeSegment encodes a text segment (no special tokens) using BPE or SentencePiece.
func (t *Tokenizer) encodeSegment(text string) []int {
	if text == "" {
		return nil
	}
	if len(t.merges) > 0 {
		return t.encodeBPE(text)
	}
	return t.encodeSentencePiece(text)
}

// buildSpecialTokens collects all tokens that look like special tokens.
func (t *Tokenizer) buildSpecialTokens() {
	t.specialTokens = make(map[string]int)
	for i, tok := range t.idToToken {
		if len(tok) > 2 && strings.HasPrefix(tok, "<|") && strings.HasSuffix(tok, "|>") {
			t.specialTokens[tok] = i
		} else if len(tok) > 2 && strings.HasPrefix(tok, "<") && strings.HasSuffix(tok, ">") &&
			!strings.ContainsAny(tok, " \t\n") && len(tok) <= 30 {
			// Also catch tokens like <s>, </s>, <unk>, etc.
			t.specialTokens[tok] = i
		}
	}
}

// encodeBPE implements GPT-2 style BPE encoding.
func (t *Tokenizer) encodeBPE(text string) []int {
	// Start with individual UTF-8 bytes/characters as initial tokens.
	symbols := t.textToSymbols(text)

	// Iteratively merge the highest-priority pair.
	for {
		bestPair := ""
		bestRank := len(t.merges) + 1
		bestIdx := -1

		for i := 0; i < len(symbols)-1; i++ {
			pair := symbols[i] + " " + symbols[i+1]
			if rank, ok := t.merges[pair]; ok && rank < bestRank {
				bestRank = rank
				bestPair = pair
				bestIdx = i
				_ = bestPair
			}
		}

		if bestIdx < 0 {
			break
		}

		// Merge the pair.
		merged := symbols[bestIdx] + symbols[bestIdx+1]
		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, merged)
		if bestIdx+2 < len(symbols) {
			newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		}
		symbols = newSymbols
	}

	// Convert symbols to token IDs.
	ids := make([]int, 0, len(symbols))
	for _, sym := range symbols {
		if id, ok := t.tokenToID[sym]; ok {
			ids = append(ids, id)
		} else {
			// Unknown token: encode as individual bytes.
			for _, b := range []byte(sym) {
				byteToken := fmt.Sprintf("<0x%02X>", b)
				if id, ok := t.tokenToID[byteToken]; ok {
					ids = append(ids, id)
				}
			}
		}
	}

	return ids
}

// encodeSentencePiece implements SentencePiece-style encoding using scores.
// This is used by LLaMA, Qwen, and other models.
func (t *Tokenizer) encodeSentencePiece(text string) []int {
	// Prepend space for SentencePiece convention (▁ = \u2581).
	text = "\u2581" + strings.ReplaceAll(text, " ", "\u2581")

	// Start with individual characters.
	symbols := make([]spSymbol, 0, utf8.RuneCountInString(text))
	for _, r := range text {
		symbols = append(symbols, spSymbol{text: string(r)})
	}

	// Iteratively merge the pair with the highest score.
	for {
		bestScore := float32(-1e18)
		bestIdx := -1

		for i := 0; i < len(symbols)-1; i++ {
			merged := symbols[i].text + symbols[i+1].text
			if id, ok := t.tokenToID[merged]; ok {
				score := float32(0)
				if id < len(t.scores) {
					score = t.scores[id]
				}
				if score > bestScore {
					bestScore = score
					bestIdx = i
				}
			}
		}

		if bestIdx < 0 {
			break
		}

		// Merge.
		merged := symbols[bestIdx].text + symbols[bestIdx+1].text
		newSymbols := make([]spSymbol, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, spSymbol{text: merged})
		if bestIdx+2 < len(symbols) {
			newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		}
		symbols = newSymbols
	}

	// Convert to IDs.
	ids := make([]int, 0, len(symbols))
	for _, sym := range symbols {
		if id, ok := t.tokenToID[sym.text]; ok {
			ids = append(ids, id)
		} else {
			// Fallback: encode as UTF-8 bytes.
			for _, b := range []byte(sym.text) {
				byteToken := fmt.Sprintf("<0x%02X>", b)
				if id, ok := t.tokenToID[byteToken]; ok {
					ids = append(ids, id)
				}
			}
		}
	}

	return ids
}

type spSymbol struct {
	text string
}

// textToSymbols splits text into initial BPE symbols using GPT-2 byte encoding.
// Each byte is mapped to a unique Unicode character to avoid control characters.
func (t *Tokenizer) textToSymbols(text string) []string {
	var symbols []string
	for _, b := range []byte(text) {
		symbols = append(symbols, string(byteToRune(b)))
	}
	return symbols
}

// Decode converts token IDs back to text.
// Handles GPT-2 byte-level encoding: token strings use special Unicode chars
// that need to be mapped back to raw bytes.
func (t *Tokenizer) Decode(ids []int) string {
	var tokenBytes []byte
	for _, id := range ids {
		if id >= 0 && id < len(t.idToToken) {
			token := t.idToToken[id]
			// Convert GPT-2 byte-encoded Unicode chars back to raw bytes.
			for _, r := range token {
				if b, ok := runeToByte(r); ok {
					tokenBytes = append(tokenBytes, b)
				} else {
					// Pass through as UTF-8.
					tokenBytes = append(tokenBytes, []byte(string(r))...)
				}
			}
		}
	}
	return string(tokenBytes)
}

// DecodeToken converts a single token ID to text.
func (t *Tokenizer) DecodeToken(id int) string {
	return t.Decode([]int{id})
}

// VocabSize returns the vocabulary size.
func (t *Tokenizer) VocabSize() int { return len(t.idToToken) }

// BOSID returns the beginning-of-sequence token ID.
func (t *Tokenizer) BOSID() int { return t.bosID }

// EOSID returns the end-of-sequence token ID.
func (t *Tokenizer) EOSID() int { return t.eosID }

// TokenID returns the ID for a given token string, or -1 if not found.
func (t *Tokenizer) TokenID(token string) int {
	if id, ok := t.tokenToID[token]; ok {
		return id
	}
	return -1
}

// TopTokens returns the top-N tokens by score (for debugging).
func (t *Tokenizer) TopTokens(n int) []string {
	type scored struct {
		token string
		score float32
	}
	var items []scored
	for i, tok := range t.idToToken {
		s := float32(0)
		if i < len(t.scores) {
			s = t.scores[i]
		}
		items = append(items, scored{tok, s})
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].score > items[j].score
	})
	if n > len(items) {
		n = len(items)
	}
	result := make([]string, n)
	for i := 0; i < n; i++ {
		result[i] = items[i].token
	}
	return result
}

// GPT-2 byte-level encoding tables.
// Maps byte values to Unicode code points to avoid control characters.
// See: https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9

var (
	byteToRuneTable [256]rune
	runeToByteTable map[rune]byte
)

func init() {
	runeToByteTable = make(map[rune]byte)
	n := 0
	for b := 0; b < 256; b++ {
		// Printable ASCII and Latin-1 supplement characters are kept as-is.
		if (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF) {
			byteToRuneTable[b] = rune(b)
		} else {
			// Control chars and other bytes get mapped to U+0100+ range.
			byteToRuneTable[b] = rune(256 + n)
			n++
		}
		runeToByteTable[byteToRuneTable[b]] = byte(b)
	}
}

// byteToRune converts a byte to its GPT-2 Unicode representation.
func byteToRune(b byte) rune {
	return byteToRuneTable[b]
}

// runeToByte converts a GPT-2 Unicode character back to its byte value.
func runeToByte(r rune) (byte, bool) {
	b, ok := runeToByteTable[r]
	return b, ok
}
