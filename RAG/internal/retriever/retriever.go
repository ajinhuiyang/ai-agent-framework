// Package retriever orchestrates the full RAG retrieval pipeline:
// query embedding → vector search → optional reranking → filtered results.
package retriever

import (
	"context"
	"fmt"
	"time"

	"go.uber.org/zap"

	"github.com/google/uuid"
	"github.com/your-org/rag/internal/domain"
	"github.com/your-org/rag/internal/embedding"
	"github.com/your-org/rag/internal/splitter"
	"github.com/your-org/rag/internal/vectorstore"
)

// Retriever is the main RAG retrieval engine.
type Retriever struct {
	embedder embedding.Provider
	store    vectorstore.Store
	splitter *splitter.Splitter
	logger   *zap.Logger

	defaultTopK     int
	defaultMinScore float64
}

// New creates a new Retriever.
func New(
	embedder embedding.Provider,
	store vectorstore.Store,
	splitter *splitter.Splitter,
	logger *zap.Logger,
	defaultTopK int,
	defaultMinScore float64,
) *Retriever {
	return &Retriever{
		embedder:        embedder,
		store:           store,
		splitter:        splitter,
		logger:          logger,
		defaultTopK:     defaultTopK,
		defaultMinScore: defaultMinScore,
	}
}

// Ingest processes documents: split → embed → store.
func (r *Retriever) Ingest(ctx context.Context, req domain.IngestRequest) (*domain.IngestResponse, error) {
	collection := req.Collection
	if collection == "" {
		collection = "default"
	}

	var allChunks []domain.Chunk

	for _, docInput := range req.Documents {
		doc := domain.Document{
			ID:        uuid.New().String(),
			Content:   docInput.Content,
			Source:    docInput.Source,
			Metadata:  docInput.Metadata,
			CreatedAt: time.Now(),
		}

		// Split document into chunks.
		chunks := r.splitter.Split(doc)

		// Copy metadata to chunks.
		for i := range chunks {
			chunks[i].Metadata = make(map[string]string)
			if doc.Source != "" {
				chunks[i].Metadata["source"] = doc.Source
			}
			for k, v := range doc.Metadata {
				chunks[i].Metadata[k] = v
			}
		}

		allChunks = append(allChunks, chunks...)
	}

	if len(allChunks) == 0 {
		return &domain.IngestResponse{Collection: collection}, nil
	}

	// Batch embed all chunks.
	texts := make([]string, len(allChunks))
	for i, chunk := range allChunks {
		texts[i] = chunk.Content
	}

	r.logger.Info("embedding chunks", zap.Int("count", len(texts)))
	embeddings, err := r.embedder.EmbedBatch(ctx, texts)
	if err != nil {
		return nil, fmt.Errorf("batch embedding failed: %w", err)
	}

	for i := range allChunks {
		allChunks[i].Embedding = embeddings[i]
	}

	// Store chunks with embeddings.
	if err := r.store.Insert(ctx, collection, allChunks); err != nil {
		return nil, fmt.Errorf("vector store insert failed: %w", err)
	}

	// Collect unique document IDs.
	docIDSet := make(map[string]struct{})
	for _, chunk := range allChunks {
		docIDSet[chunk.DocumentID] = struct{}{}
	}
	docIDs := make([]string, 0, len(docIDSet))
	for id := range docIDSet {
		docIDs = append(docIDs, id)
	}

	r.logger.Info("ingestion complete",
		zap.Int("documents", len(docIDs)),
		zap.Int("chunks", len(allChunks)),
		zap.String("collection", collection),
	)

	return &domain.IngestResponse{
		DocumentIDs: docIDs,
		ChunkCount:  len(allChunks),
		Collection:  collection,
	}, nil
}

// Search performs semantic search: embed query → vector search → filter → return.
func (r *Retriever) Search(ctx context.Context, req domain.SearchRequest) (*domain.SearchResponse, error) {
	start := time.Now()

	collection := req.Collection
	if collection == "" {
		collection = "default"
	}

	topK := req.TopK
	if topK <= 0 {
		topK = r.defaultTopK
	}

	minScore := req.MinScore
	if minScore <= 0 {
		minScore = r.defaultMinScore
	}

	// Embed the query.
	queryVec, err := r.embedder.Embed(ctx, req.Query)
	if err != nil {
		return nil, fmt.Errorf("query embedding failed: %w", err)
	}

	// Vector search.
	results, err := r.store.Search(ctx, collection, queryVec, topK, minScore)
	if err != nil {
		return nil, fmt.Errorf("vector search failed: %w", err)
	}

	// Apply metadata filters if provided.
	if len(req.Filters) > 0 {
		results = filterResults(results, req.Filters)
	}

	elapsed := time.Since(start)
	r.logger.Info("search complete",
		zap.String("query", req.Query),
		zap.Int("results", len(results)),
		zap.Duration("elapsed", elapsed),
	)

	return &domain.SearchResponse{
		Results:     results,
		Query:       req.Query,
		TotalFound:  len(results),
		TimeTakenMs: elapsed.Milliseconds(),
	}, nil
}

// filterResults applies metadata-based filtering on search results.
func filterResults(results []domain.SearchResult, filters map[string]string) []domain.SearchResult {
	var filtered []domain.SearchResult
	for _, r := range results {
		match := true
		for k, v := range filters {
			if r.Chunk.Metadata[k] != v {
				match = false
				break
			}
		}
		if match {
			filtered = append(filtered, r)
		}
	}
	return filtered
}
