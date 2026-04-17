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

// --- Code-specific endpoints ---

// CodeAnalyze handles POST /api/v1/llm/code/analyze
func (h *Handler) CodeAnalyze(c *gin.Context) {
	h.handleCode(c, "analyze", "code_analyze")
}

// CodeGenerate handles POST /api/v1/llm/code/generate
func (h *Handler) CodeGenerate(c *gin.Context) {
	h.handleCode(c, "generate", "code_generate")
}

// CodeExplain handles POST /api/v1/llm/code/explain
func (h *Handler) CodeExplain(c *gin.Context) {
	h.handleCode(c, "explain", "code_explain")
}

// CodeRefactor handles POST /api/v1/llm/code/refactor
func (h *Handler) CodeRefactor(c *gin.Context) {
	h.handleCode(c, "refactor", "code_refactor")
}

// CodeTest handles POST /api/v1/llm/code/test
func (h *Handler) CodeTest(c *gin.Context) {
	h.handleCode(c, "test", "code_test")
}

// CodeReview handles POST /api/v1/llm/code/review
func (h *Handler) CodeReview(c *gin.Context) {
	h.handleCode(c, "review", "code_review")
}

// handleCode is the shared handler for all code operations.
func (h *Handler) handleCode(c *gin.Context, codeType, systemPromptKey string) {
	var req domain.CodeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, APIResponse{
			Error: &APIError{Code: 400, Message: "invalid request: " + err.Error()},
		})
		return
	}

	// Build the user prompt based on operation type.
	var userPrompt string
	lang := req.Language
	if lang == "" {
		lang = "unknown"
	}

	switch codeType {
	case "analyze":
		if req.Code == "" {
			c.JSON(http.StatusBadRequest, APIResponse{Error: &APIError{Code: 400, Message: "code is required"}})
			return
		}
		userPrompt = fmt.Sprintf("Analyze the following %s code:\n\n```%s\n%s\n```", lang, lang, req.Code)

	case "generate":
		if req.Description == "" {
			c.JSON(http.StatusBadRequest, APIResponse{Error: &APIError{Code: 400, Message: "description is required"}})
			return
		}
		userPrompt = fmt.Sprintf("Generate %s code for the following requirement:\n\n%s", lang, req.Description)

	case "explain":
		if req.Code == "" {
			c.JSON(http.StatusBadRequest, APIResponse{Error: &APIError{Code: 400, Message: "code is required"}})
			return
		}
		userPrompt = fmt.Sprintf("Explain the following %s code line by line:\n\n```%s\n%s\n```", lang, lang, req.Code)

	case "refactor":
		if req.Code == "" {
			c.JSON(http.StatusBadRequest, APIResponse{Error: &APIError{Code: 400, Message: "code is required"}})
			return
		}
		userPrompt = fmt.Sprintf("Refactor and improve the following %s code:\n\n```%s\n%s\n```", lang, lang, req.Code)

	case "test":
		if req.Code == "" {
			c.JSON(http.StatusBadRequest, APIResponse{Error: &APIError{Code: 400, Message: "code is required"}})
			return
		}
		fw := req.TestFramework
		if fw == "" {
			fw = "standard library"
		}
		userPrompt = fmt.Sprintf("Generate unit tests using %s for the following %s code:\n\n```%s\n%s\n```", fw, lang, lang, req.Code)

	case "review":
		if req.Code == "" && req.Diff == "" {
			c.JSON(http.StatusBadRequest, APIResponse{Error: &APIError{Code: 400, Message: "code or diff is required"}})
			return
		}
		if req.Diff != "" {
			userPrompt = fmt.Sprintf("Review the following code diff:\n\n```diff\n%s\n```", req.Diff)
		} else {
			userPrompt = fmt.Sprintf("Review the following %s code:\n\n```%s\n%s\n```", lang, lang, req.Code)
		}
	}

	// Generate using the orchestrator with the appropriate system prompt.
	genReq := domain.GenerateRequest{
		Prompt:       userPrompt,
		SystemPrompt: systemPromptKey, // Orchestrator/prompt manager resolves this
		Provider:     req.Provider,
		Config:       req.Config,
	}

	result, err := h.orch.Generate(c.Request.Context(), genReq)
	if err != nil {
		h.logger.Error("code operation failed", zap.String("type", codeType), zap.Error(err))
		c.JSON(http.StatusInternalServerError, APIResponse{
			Error: &APIError{Code: 500, Message: codeType + " failed: " + err.Error()},
		})
		return
	}

	c.JSON(http.StatusOK, APIResponse{
		Success: true,
		Data: domain.CodeResponse{
			Result:   result.Content,
			Type:     codeType,
			Language: req.Language,
			Provider: result.Provider,
			Model:    result.Model,
			Usage:    result.Usage,
		},
	})
}
