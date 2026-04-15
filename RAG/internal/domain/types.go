// Package domain defines the core data types for the RAG service.
package domain

import "time"

// Document represents a source document that can be indexed and searched.
type Document struct {
	ID        string            `json:"id"`
	Content   string            `json:"content"`
	Source    string            `json:"source"` // file path, URL, etc.
	Metadata  map[string]string `json:"metadata,omitempty"`
	CreatedAt time.Time         `json:"created_at"`
}

// Chunk represents a split piece of a document, ready for embedding.
type Chunk struct {
	ID         string            `json:"id"`
	DocumentID string            `json:"document_id"`
	Content    string            `json:"content"`
	Index      int               `json:"index"` // position in original document
	Metadata   map[string]string `json:"metadata,omitempty"`
	Embedding  []float32         `json:"embedding,omitempty"` // vector representation
}

// SearchRequest represents a retrieval query.
type SearchRequest struct {
	Query      string            `json:"query"`
	TopK       int               `json:"top_k,omitempty"`      // number of results to return
	MinScore   float64           `json:"min_score,omitempty"`  // minimum similarity threshold
	Filters    map[string]string `json:"filters,omitempty"`    // metadata-based filtering
	Collection string            `json:"collection,omitempty"` // target collection/namespace
	Rerank     bool              `json:"rerank,omitempty"`     // whether to apply reranking
}

// SearchResult represents a single retrieval hit.
type SearchResult struct {
	Chunk    Chunk   `json:"chunk"`
	Score    float64 `json:"score"`    // similarity score [0, 1]
	Distance float64 `json:"distance"` // vector distance
}

// SearchResponse is the complete response of a retrieval operation.
type SearchResponse struct {
	Results     []SearchResult `json:"results"`
	Query       string         `json:"query"`
	TotalFound  int            `json:"total_found"`
	TimeTakenMs int64          `json:"time_taken_ms"`
}

// IngestRequest represents a request to add documents to the knowledge base.
type IngestRequest struct {
	Documents  []DocumentInput `json:"documents"`
	Collection string          `json:"collection,omitempty"`
	ChunkSize  int             `json:"chunk_size,omitempty"`
	Overlap    int             `json:"overlap,omitempty"`
}

// DocumentInput is a simplified document for ingestion.
type DocumentInput struct {
	Content  string            `json:"content"`
	Source   string            `json:"source,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// IngestResponse is the result of an ingestion operation.
type IngestResponse struct {
	DocumentIDs []string `json:"document_ids"`
	ChunkCount  int      `json:"chunk_count"`
	Collection  string   `json:"collection"`
}

// CollectionInfo describes a vector collection.
type CollectionInfo struct {
	Name       string `json:"name"`
	ChunkCount int    `json:"chunk_count"`
	Dimension  int    `json:"dimension"`
}

// WebSearchRequest represents a request for online search augmentation.
type WebSearchRequest struct {
	Query   string `json:"query"`
	MaxURLs int    `json:"max_urls,omitempty"`
}

// WebSearchResult represents a single web search hit.
type WebSearchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet"`
	Content string `json:"content,omitempty"` // fetched full content
}
