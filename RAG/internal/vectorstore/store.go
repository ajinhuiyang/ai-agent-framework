// Package vectorstore defines the interface for vector storage and retrieval.
package vectorstore

import (
	"context"

	"github.com/your-org/rag/internal/domain"
)

// Store is the interface for vector storage backends.
type Store interface {
	// Insert adds chunks with embeddings to the store.
	Insert(ctx context.Context, collection string, chunks []domain.Chunk) error

	// Search finds the most similar chunks to the given query vector.
	Search(ctx context.Context, collection string, queryVec []float32, topK int, minScore float64) ([]domain.SearchResult, error)

	// Delete removes chunks by their IDs.
	Delete(ctx context.Context, collection string, chunkIDs []string) error

	// ListCollections returns all available collections.
	ListCollections(ctx context.Context) ([]domain.CollectionInfo, error)

	// DeleteCollection removes an entire collection.
	DeleteCollection(ctx context.Context, collection string) error

	// Count returns the number of chunks in a collection.
	Count(ctx context.Context, collection string) (int, error)
}
