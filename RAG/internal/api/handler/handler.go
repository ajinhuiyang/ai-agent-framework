// Package handler implements HTTP request handlers for the RAG service.
package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/your-org/rag/internal/domain"
	"github.com/your-org/rag/internal/loader/pdf"
	"github.com/your-org/rag/internal/loader/search"
	"github.com/your-org/rag/internal/loader/text"
	"github.com/your-org/rag/internal/loader/web"
	"github.com/your-org/rag/internal/retriever"
	"github.com/your-org/rag/internal/vectorstore"
)

// Handler handles HTTP requests for the RAG service.
type Handler struct {
	retriever *retriever.Retriever
	store     vectorstore.Store
	logger    *zap.Logger
}

// New creates a new Handler.
func New(r *retriever.Retriever, store vectorstore.Store, logger *zap.Logger) *Handler {
	return &Handler{
		retriever: r,
		store:     store,
		logger:    logger,
	}
}

// APIResponse is the standard response envelope.
type APIResponse struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data,omitempty"`
	Error   *APIError   `json:"error,omitempty"`
}

// APIError represents an error response.
type APIError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// HealthCheck handles GET /health.
func (h *Handler) HealthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, APIResponse{
		Success: true,
		Data:    gin.H{"status": "healthy", "service": "rag"},
	})
}

// Ingest handles POST /api/v1/rag/ingest — add documents to the knowledge base.
func (h *Handler) Ingest(c *gin.Context) {
	var req domain.IngestRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, APIResponse{
			Error: &APIError{Code: 400, Message: "invalid request: " + err.Error()},
		})
		return
	}

	if len(req.Documents) == 0 {
		c.JSON(http.StatusBadRequest, APIResponse{
			Error: &APIError{Code: 400, Message: "documents array is required"},
		})
		return
	}

	result, err := h.retriever.Ingest(c.Request.Context(), req)
	if err != nil {
		h.logger.Error("ingestion failed", zap.Error(err))
		c.JSON(http.StatusInternalServerError, APIResponse{
			Error: &APIError{Code: 500, Message: "ingestion failed: " + err.Error()},
		})
		return
	}

	c.JSON(http.StatusOK, APIResponse{Success: true, Data: result})
}

// Search handles POST /api/v1/rag/search — perform semantic search.
func (h *Handler) Search(c *gin.Context) {
	var req domain.SearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, APIResponse{
			Error: &APIError{Code: 400, Message: "invalid request: " + err.Error()},
		})
		return
	}

	if req.Query == "" {
		c.JSON(http.StatusBadRequest, APIResponse{
			Error: &APIError{Code: 400, Message: "query is required"},
		})
		return
	}

	result, err := h.retriever.Search(c.Request.Context(), req)
	if err != nil {
		h.logger.Error("search failed", zap.Error(err))
		c.JSON(http.StatusInternalServerError, APIResponse{
			Error: &APIError{Code: 500, Message: "search failed: " + err.Error()},
		})
		return
	}

	c.JSON(http.StatusOK, APIResponse{Success: true, Data: result})
}

// ListCollections handles GET /api/v1/rag/collections — list all collections.
func (h *Handler) ListCollections(c *gin.Context) {
	collections, err := h.store.ListCollections(c.Request.Context())
	if err != nil {
		h.logger.Error("list collections failed", zap.Error(err))
		c.JSON(http.StatusInternalServerError, APIResponse{
			Error: &APIError{Code: 500, Message: "failed to list collections"},
		})
		return
	}

	c.JSON(http.StatusOK, APIResponse{Success: true, Data: collections})
}

// DeleteCollection handles DELETE /api/v1/rag/collections/:name — delete a collection.
func (h *Handler) DeleteCollection(c *gin.Context) {
	name := c.Param("name")
	if name == "" {
		c.JSON(http.StatusBadRequest, APIResponse{
			Error: &APIError{Code: 400, Message: "collection name is required"},
		})
		return
	}

	if err := h.store.DeleteCollection(c.Request.Context(), name); err != nil {
		h.logger.Error("delete collection failed", zap.Error(err), zap.String("collection", name))
		c.JSON(http.StatusInternalServerError, APIResponse{
			Error: &APIError{Code: 500, Message: "failed to delete collection"},
		})
		return
	}

	c.JSON(http.StatusOK, APIResponse{
		Success: true,
		Data:    gin.H{"message": "collection deleted", "name": name},
	})
}

// --- Loader endpoints: fetch content from external sources and ingest ---

// FetchWeb handles POST /api/v1/rag/fetch/web — scrape a URL and ingest its content.
func (h *Handler) FetchWeb(c *gin.Context) {
	var req struct {
		URL        string `json:"url" binding:"required"`
		Collection string `json:"collection,omitempty"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, APIResponse{
			Error: &APIError{Code: 400, Message: "invalid request: " + err.Error()},
		})
		return
	}

	scraper := web.New()
	results, err := scraper.Load(c.Request.Context(), req.URL)
	if err != nil {
		h.logger.Error("web fetch failed", zap.Error(err))
		c.JSON(http.StatusInternalServerError, APIResponse{
			Error: &APIError{Code: 500, Message: "web fetch failed: " + err.Error()},
		})
		return
	}

	// Convert to IngestRequest and ingest.
	ingestReq := domain.IngestRequest{Collection: req.Collection}
	for _, r := range results {
		meta := map[string]string{"title": r.Title}
		ingestReq.Documents = append(ingestReq.Documents, domain.DocumentInput{
			Content: r.Content, Source: r.Source, Metadata: meta,
		})
	}
	ingestResp, err := h.retriever.Ingest(c.Request.Context(), ingestReq)
	if err != nil {
		c.JSON(http.StatusInternalServerError, APIResponse{
			Error: &APIError{Code: 500, Message: "ingest failed: " + err.Error()},
		})
		return
	}

	c.JSON(http.StatusOK, APIResponse{
		Success: true,
		Data: gin.H{
			"fetched": len(results),
			"ingest":  ingestResp,
		},
	})
}

// FetchSearch handles POST /api/v1/rag/fetch/search — search the web and ingest results.
func (h *Handler) FetchSearch(c *gin.Context) {
	var req struct {
		Query      string `json:"query" binding:"required"`
		Engine     string `json:"engine,omitempty"` // "google" or "bing", default "google"
		TopN       int    `json:"top_n,omitempty"`  // fetch full content for top N results
		Collection string `json:"collection,omitempty"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, APIResponse{
			Error: &APIError{Code: 400, Message: "invalid request: " + err.Error()},
		})
		return
	}

	engine := search.Google
	if req.Engine == "bing" {
		engine = search.Bing
	}
	topN := req.TopN
	if topN <= 0 {
		topN = 3
	}

	searcher := search.New(engine)

	var results []struct{ Title, Content, Source string }

	if topN > 0 {
		// Fetch full page content for top N results.
		loaded, err := searcher.LoadWithContent(c.Request.Context(), req.Query, topN)
		if err != nil {
			h.logger.Warn("search with content failed, falling back to snippets", zap.Error(err))
		}
		for _, r := range loaded {
			results = append(results, struct{ Title, Content, Source string }{r.Title, r.Content, r.Source})
		}
	}

	// If no full content results, use snippets.
	if len(results) == 0 {
		snippets, err := searcher.Load(c.Request.Context(), req.Query)
		if err != nil {
			c.JSON(http.StatusInternalServerError, APIResponse{
				Error: &APIError{Code: 500, Message: "search failed: " + err.Error()},
			})
			return
		}
		for _, r := range snippets {
			results = append(results, struct{ Title, Content, Source string }{r.Title, r.Content, r.Source})
		}
	}

	// Ingest results.
	var docs []domain.DocumentInput
	for _, r := range results {
		if r.Content == "" {
			continue
		}
		meta := map[string]string{"title": r.Title}
		docs = append(docs, domain.DocumentInput{Content: r.Content, Source: r.Source, Metadata: meta})
	}

	var ingestResp *domain.IngestResponse
	if len(docs) > 0 {
		ingestReq := domain.IngestRequest{Documents: docs, Collection: req.Collection}
		var ingestErr error
		ingestResp, ingestErr = h.retriever.Ingest(c.Request.Context(), ingestReq)
		if ingestErr != nil {
			c.JSON(http.StatusInternalServerError, APIResponse{
				Error: &APIError{Code: 500, Message: "ingest failed: " + ingestErr.Error()},
			})
			return
		}
	}

	c.JSON(http.StatusOK, APIResponse{
		Success: true,
		Data: gin.H{
			"query":   req.Query,
			"fetched": len(results),
			"ingest":  ingestResp,
		},
	})
}

// FetchFile handles POST /api/v1/rag/fetch/file — parse local files and ingest.
func (h *Handler) FetchFile(c *gin.Context) {
	var req struct {
		Path       string `json:"path" binding:"required"` // File or directory path
		Collection string `json:"collection,omitempty"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, APIResponse{
			Error: &APIError{Code: 400, Message: "invalid request: " + err.Error()},
		})
		return
	}

	var allResults []struct{ Title, Content, Source string }

	// Try text parser first.
	textParser := text.New()
	textResults, err := textParser.Load(c.Request.Context(), req.Path)
	if err == nil {
		for _, r := range textResults {
			allResults = append(allResults, struct{ Title, Content, Source string }{r.Title, r.Content, r.Source})
		}
	}

	// Try PDF parser for .pdf files.
	pdfParser := pdf.New()
	pdfResults, err := pdfParser.Load(c.Request.Context(), req.Path)
	if err == nil {
		for _, r := range pdfResults {
			allResults = append(allResults, struct{ Title, Content, Source string }{r.Title, r.Content, r.Source})
		}
	}

	if len(allResults) == 0 {
		c.JSON(http.StatusBadRequest, APIResponse{
			Error: &APIError{Code: 400, Message: "no content could be extracted from: " + req.Path},
		})
		return
	}

	ingestReq := domain.IngestRequest{Collection: req.Collection}
	for _, r := range allResults {
		meta := map[string]string{"title": r.Title}
		ingestReq.Documents = append(ingestReq.Documents, domain.DocumentInput{
			Content: r.Content, Source: r.Source, Metadata: meta,
		})
	}

	ingestResp, err := h.retriever.Ingest(c.Request.Context(), ingestReq)
	if err != nil {
		c.JSON(http.StatusInternalServerError, APIResponse{
			Error: &APIError{Code: 500, Message: "ingest failed: " + err.Error()},
		})
		return
	}

	c.JSON(http.StatusOK, APIResponse{
		Success: true,
		Data: gin.H{
			"parsed": len(allResults),
			"ingest": ingestResp,
		},
	})
}
