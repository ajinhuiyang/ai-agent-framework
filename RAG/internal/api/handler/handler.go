// Package handler implements HTTP request handlers for the RAG service.
package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/your-org/rag/internal/domain"
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
