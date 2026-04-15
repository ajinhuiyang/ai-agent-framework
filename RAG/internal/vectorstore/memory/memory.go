// Package memory implements an in-memory vector store using brute-force cosine similarity.
// Suitable for development, testing, and small-scale deployments.
package memory

import (
	"context"
	"math"
	"sort"
	"sync"

	"github.com/your-org/rag/internal/domain"
)

// Store is an in-memory vector store.
type Store struct {
	mu          sync.RWMutex
	collections map[string]*collection
}

type collection struct {
	chunks []domain.Chunk
}

// New creates a new in-memory vector store.
func New() *Store {
	return &Store{
		collections: make(map[string]*collection),
	}
}

func (s *Store) Insert(_ context.Context, collName string, chunks []domain.Chunk) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	coll, ok := s.collections[collName]
	if !ok {
		coll = &collection{}
		s.collections[collName] = coll
	}
	coll.chunks = append(coll.chunks, chunks...)
	return nil
}

func (s *Store) Search(_ context.Context, collName string, queryVec []float32, topK int, minScore float64) ([]domain.SearchResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	coll, ok := s.collections[collName]
	if !ok {
		return nil, nil
	}

	type scored struct {
		chunk domain.Chunk
		score float64
	}

	var results []scored
	for _, chunk := range coll.chunks {
		if len(chunk.Embedding) == 0 {
			continue
		}
		score := cosineSimilarity(queryVec, chunk.Embedding)
		if score >= minScore {
			results = append(results, scored{chunk: chunk, score: score})
		}
	}

	// Sort by score descending.
	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	if topK > 0 && len(results) > topK {
		results = results[:topK]
	}

	searchResults := make([]domain.SearchResult, len(results))
	for i, r := range results {
		// Clear embedding from response to reduce payload.
		r.chunk.Embedding = nil
		searchResults[i] = domain.SearchResult{
			Chunk:    r.chunk,
			Score:    r.score,
			Distance: 1 - r.score,
		}
	}
	return searchResults, nil
}

func (s *Store) Delete(_ context.Context, collName string, chunkIDs []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	coll, ok := s.collections[collName]
	if !ok {
		return nil
	}

	idSet := make(map[string]struct{}, len(chunkIDs))
	for _, id := range chunkIDs {
		idSet[id] = struct{}{}
	}

	filtered := make([]domain.Chunk, 0, len(coll.chunks))
	for _, chunk := range coll.chunks {
		if _, remove := idSet[chunk.ID]; !remove {
			filtered = append(filtered, chunk)
		}
	}
	coll.chunks = filtered
	return nil
}

func (s *Store) ListCollections(_ context.Context) ([]domain.CollectionInfo, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	infos := make([]domain.CollectionInfo, 0, len(s.collections))
	for name, coll := range s.collections {
		dim := 0
		if len(coll.chunks) > 0 && len(coll.chunks[0].Embedding) > 0 {
			dim = len(coll.chunks[0].Embedding)
		}
		infos = append(infos, domain.CollectionInfo{
			Name:       name,
			ChunkCount: len(coll.chunks),
			Dimension:  dim,
		})
	}
	return infos, nil
}

func (s *Store) DeleteCollection(_ context.Context, collName string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.collections, collName)
	return nil
}

func (s *Store) Count(_ context.Context, collName string) (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	coll, ok := s.collections[collName]
	if !ok {
		return 0, nil
	}
	return len(coll.chunks), nil
}

// cosineSimilarity computes the cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}
