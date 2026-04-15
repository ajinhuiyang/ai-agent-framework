// Package embedding defines the interface for text embedding (vectorization).
package embedding

import "context"

// Provider generates vector embeddings from text.
type Provider interface {
	// Embed converts a single text into a vector embedding.
	Embed(ctx context.Context, text string) ([]float32, error)

	// EmbedBatch converts multiple texts into vector embeddings.
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)

	// Dimension returns the embedding vector dimension.
	Dimension() int

	// Name returns the provider name.
	Name() string
}
