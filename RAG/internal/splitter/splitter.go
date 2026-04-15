// Package splitter handles splitting documents into smaller chunks for embedding.
package splitter

import (
	"strings"
	"unicode/utf8"

	"github.com/google/uuid"
	"github.com/your-org/rag/internal/domain"
)

// Splitter splits documents into chunks.
type Splitter struct {
	chunkSize int
	overlap   int
	separator string
}

// New creates a new text splitter.
func New(chunkSize, overlap int, separator string) *Splitter {
	if chunkSize <= 0 {
		chunkSize = 512
	}
	if overlap < 0 {
		overlap = 0
	}
	if overlap >= chunkSize {
		overlap = chunkSize / 4
	}
	if separator == "" {
		separator = "\n\n"
	}
	return &Splitter{
		chunkSize: chunkSize,
		overlap:   overlap,
		separator: separator,
	}
}

// Split splits a document into chunks.
func (s *Splitter) Split(doc domain.Document) []domain.Chunk {
	text := strings.TrimSpace(doc.Content)
	if text == "" {
		return nil
	}

	// First split by separator (paragraph breaks, etc.)
	paragraphs := strings.Split(text, s.separator)

	var chunks []domain.Chunk
	var currentChunk strings.Builder
	idx := 0

	for _, para := range paragraphs {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}

		// If adding this paragraph exceeds chunk size, flush current chunk.
		if currentChunk.Len() > 0 && utf8.RuneCountInString(currentChunk.String())+utf8.RuneCountInString(para)+1 > s.chunkSize {
			chunk := s.makeChunk(doc.ID, currentChunk.String(), idx)
			chunks = append(chunks, chunk)
			idx++

			// Apply overlap: keep the tail of the current chunk.
			if s.overlap > 0 {
				tail := s.getTail(currentChunk.String(), s.overlap)
				currentChunk.Reset()
				if tail != "" {
					currentChunk.WriteString(tail)
				}
			} else {
				currentChunk.Reset()
			}
		}

		// If a single paragraph exceeds chunk size, split it further.
		if utf8.RuneCountInString(para) > s.chunkSize {
			// Flush anything accumulated.
			if currentChunk.Len() > 0 {
				chunk := s.makeChunk(doc.ID, currentChunk.String(), idx)
				chunks = append(chunks, chunk)
				idx++
				currentChunk.Reset()
			}

			// Split the long paragraph into fixed-size pieces.
			runes := []rune(para)
			for i := 0; i < len(runes); i += s.chunkSize - s.overlap {
				end := i + s.chunkSize
				if end > len(runes) {
					end = len(runes)
				}
				piece := string(runes[i:end])
				chunk := s.makeChunk(doc.ID, piece, idx)
				chunks = append(chunks, chunk)
				idx++
			}
			continue
		}

		if currentChunk.Len() > 0 {
			currentChunk.WriteString(" ")
		}
		currentChunk.WriteString(para)
	}

	// Flush remaining.
	if currentChunk.Len() > 0 {
		chunk := s.makeChunk(doc.ID, currentChunk.String(), idx)
		chunks = append(chunks, chunk)
	}

	return chunks
}

func (s *Splitter) makeChunk(docID, content string, index int) domain.Chunk {
	return domain.Chunk{
		ID:         uuid.New().String(),
		DocumentID: docID,
		Content:    content,
		Index:      index,
	}
}

// getTail returns the last n runes of the text as overlap context.
func (s *Splitter) getTail(text string, n int) string {
	runes := []rune(text)
	if len(runes) <= n {
		return text
	}
	return string(runes[len(runes)-n:])
}
