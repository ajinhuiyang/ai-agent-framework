// Package handler implements HTTP request handlers for the LLM Generation service.
package handler

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/your-org/llm-generation/internal/domain"
	"github.com/your-org/llm-generation/internal/orchestrator"
)

// Handler handles HTTP requests for the LLM Generation service.
type Handler struct {
	orch   *orchestrator.Orchestrator
	logger *zap.Logger
}

// New creates a new Handler.
func New(orch *orchestrator.Orchestrator, logger *zap.Logger) *Handler {
	return &Handler{orch: orch, logger: logger}
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
		Data:    gin.H{"status": "healthy", "service": "llm-generation"},
	})
}

// Generate handles POST /api/v1/llm/generate — generate content.
func (h *Handler) Generate(c *gin.Context) {
	var req domain.GenerateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, APIResponse{
			Error: &APIError{Code: 400, Message: "invalid request: " + err.Error()},
		})
		return
	}

	if req.Prompt == "" {
		c.JSON(http.StatusBadRequest, APIResponse{
			Error: &APIError{Code: 400, Message: "prompt is required"},
		})
		return
	}

	// If streaming is requested, use SSE.
	if req.Stream {
		h.handleStream(c, req)
		return
	}

	result, err := h.orch.Generate(c.Request.Context(), req)
	if err != nil {
		h.logger.Error("generation failed", zap.Error(err))
		c.JSON(http.StatusInternalServerError, APIResponse{
			Error: &APIError{Code: 500, Message: "generation failed: " + err.Error()},
		})
		return
	}

	c.JSON(http.StatusOK, APIResponse{Success: true, Data: result})
}

// handleStream handles streaming response via Server-Sent Events (SSE).
func (h *Handler) handleStream(c *gin.Context, req domain.GenerateRequest) {
	streamCh, providerName, err := h.orch.GenerateStream(c.Request.Context(), req)
	if err != nil {
		h.logger.Error("stream generation failed", zap.Error(err))
		c.JSON(http.StatusInternalServerError, APIResponse{
			Error: &APIError{Code: 500, Message: "stream failed: " + err.Error()},
		})
		return
	}

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Provider", providerName)

	c.Writer.Flush()
	flusher, _ := c.Writer.(http.Flusher)

	for chunk := range streamCh {
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(c.Writer, "data: %s\n\n", data)
		if flusher != nil {
			flusher.Flush()
		}
		if chunk.Done {
			break
		}
	}
}

// CreateConversation handles POST /api/v1/llm/conversations — create a new conversation.
func (h *Handler) CreateConversation(c *gin.Context) {
	id := h.orch.CreateConversation()
	c.JSON(http.StatusOK, APIResponse{
		Success: true,
		Data:    gin.H{"conversation_id": id},
	})
}

// GetConversation handles GET /api/v1/llm/conversations/:id — get conversation history.
func (h *Handler) GetConversation(c *gin.Context) {
	id := c.Param("id")
	conv, ok := h.orch.GetConversation(id)
	if !ok {
		c.JSON(http.StatusNotFound, APIResponse{
			Error: &APIError{Code: 404, Message: "conversation not found"},
		})
		return
	}
	c.JSON(http.StatusOK, APIResponse{Success: true, Data: conv})
}

// DeleteConversation handles DELETE /api/v1/llm/conversations/:id — delete a conversation.
func (h *Handler) DeleteConversation(c *gin.Context) {
	id := c.Param("id")
	h.orch.DeleteConversation(id)
	c.JSON(http.StatusOK, APIResponse{
		Success: true,
		Data:    gin.H{"message": "conversation deleted"},
	})
}

// ListProviders handles GET /api/v1/llm/providers — list available LLM providers.
func (h *Handler) ListProviders(c *gin.Context) {
	infos := h.orch.ListProviders(c.Request.Context())
	c.JSON(http.StatusOK, APIResponse{Success: true, Data: infos})
}
