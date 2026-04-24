// Package handler implements HTTP request handlers for the NLU API.
package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/your-org/nlu/internal/client"
	"github.com/your-org/nlu/internal/domain"
	"github.com/your-org/nlu/internal/nlu/dialog"
	"github.com/your-org/nlu/internal/pipeline"
)

// NLUHandler handles NLU API requests.
type NLUHandler struct {
	engine        *pipeline.Engine
	dialogManager *dialog.Manager
	ragClient     *client.RAGClient
	llmClient     *client.LLMClient
	logger        *zap.Logger
}

// NewNLUHandler creates a new NLU handler.
func NewNLUHandler(
	engine *pipeline.Engine,
	dialogManager *dialog.Manager,
	ragClient *client.RAGClient,
	llmClient *client.LLMClient,
	logger *zap.Logger,
) *NLUHandler {
	return &NLUHandler{
		engine:        engine,
		dialogManager: dialogManager,
		ragClient:     ragClient,
		llmClient:     llmClient,
		logger:        logger,
	}
}

// --- Response Types ---

// APIResponse is the standard API response wrapper.
type APIResponse struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data,omitempty"`
	Error   *APIError   `json:"error,omitempty"`
}

// APIError represents an API error.
type APIError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// --- Handlers ---

// Process handles POST /api/v1/nlu/process
// This is the main NLU endpoint that runs the full pipeline.
func (h *NLUHandler) Process(c *gin.Context) {
	var req domain.NLURequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error: &APIError{
				Code:    http.StatusBadRequest,
				Message: "invalid request: " + err.Error(),
			},
		})
		return
	}

	if req.Text == "" {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error: &APIError{
				Code:    http.StatusBadRequest,
				Message: "text field is required",
			},
		})
		return
	}

	result, err := h.engine.Process(c.Request.Context(), &req)
	if err != nil {
		h.logger.Error("NLU processing failed", zap.Error(err))
		c.JSON(http.StatusInternalServerError, APIResponse{
			Success: false,
			Error: &APIError{
				Code:    http.StatusInternalServerError,
				Message: "NLU processing failed: " + err.Error(),
			},
		})
		return
	}

	c.JSON(http.StatusOK, APIResponse{
		Success: true,
		Data:    result,
	})
}

// IntentOnly handles POST /api/v1/nlu/intent
// Runs only intent recognition.
func (h *NLUHandler) IntentOnly(c *gin.Context) {
	var req domain.NLURequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   &APIError{Code: http.StatusBadRequest, Message: err.Error()},
		})
		return
	}

	req.Capabilities = []string{"intent"}
	result, err := h.engine.Process(c.Request.Context(), &req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   &APIError{Code: http.StatusInternalServerError, Message: err.Error()},
		})
		return
	}

	c.JSON(http.StatusOK, APIResponse{Success: true, Data: result.Intent})
}

// NEROnly handles POST /api/v1/nlu/ner
// Runs only named entity recognition.
func (h *NLUHandler) NEROnly(c *gin.Context) {
	var req domain.NLURequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   &APIError{Code: http.StatusBadRequest, Message: err.Error()},
		})
		return
	}

	req.Capabilities = []string{"ner"}
	result, err := h.engine.Process(c.Request.Context(), &req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   &APIError{Code: http.StatusInternalServerError, Message: err.Error()},
		})
		return
	}

	c.JSON(http.StatusOK, APIResponse{Success: true, Data: result.Entities})
}

// SentimentOnly handles POST /api/v1/nlu/sentiment
func (h *NLUHandler) SentimentOnly(c *gin.Context) {
	var req domain.NLURequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   &APIError{Code: http.StatusBadRequest, Message: err.Error()},
		})
		return
	}

	req.Capabilities = []string{"sentiment"}
	result, err := h.engine.Process(c.Request.Context(), &req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   &APIError{Code: http.StatusInternalServerError, Message: err.Error()},
		})
		return
	}

	c.JSON(http.StatusOK, APIResponse{Success: true, Data: result.Sentiment})
}

// ClassifyOnly handles POST /api/v1/nlu/classify
func (h *NLUHandler) ClassifyOnly(c *gin.Context) {
	var req domain.NLURequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   &APIError{Code: http.StatusBadRequest, Message: err.Error()},
		})
		return
	}

	req.Capabilities = []string{"classify"}
	result, err := h.engine.Process(c.Request.Context(), &req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   &APIError{Code: http.StatusInternalServerError, Message: err.Error()},
		})
		return
	}

	c.JSON(http.StatusOK, APIResponse{Success: true, Data: result.Classification})
}

// SlotFill handles POST /api/v1/nlu/slot
func (h *NLUHandler) SlotFill(c *gin.Context) {
	var req domain.NLURequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   &APIError{Code: http.StatusBadRequest, Message: err.Error()},
		})
		return
	}

	req.Capabilities = []string{"intent", "slot"}
	result, err := h.engine.Process(c.Request.Context(), &req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   &APIError{Code: http.StatusInternalServerError, Message: err.Error()},
		})
		return
	}

	c.JSON(http.StatusOK, APIResponse{
		Success: true,
		Data: gin.H{
			"intent":       result.Intent,
			"slot_filling": result.SlotFilling,
		},
	})
}

// --- Dialog Management Handlers ---

// GetDialogState handles GET /api/v1/dialog/:session_id
func (h *NLUHandler) GetDialogState(c *gin.Context) {
	sessionID := c.Param("session_id")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   &APIError{Code: http.StatusBadRequest, Message: "session_id is required"},
		})
		return
	}

	state := h.dialogManager.GetDialogState(sessionID)
	c.JSON(http.StatusOK, APIResponse{Success: true, Data: state})
}

// DeleteDialog handles DELETE /api/v1/dialog/:session_id
func (h *NLUHandler) DeleteDialog(c *gin.Context) {
	sessionID := c.Param("session_id")
	h.dialogManager.DeleteSession(sessionID)
	c.JSON(http.StatusOK, APIResponse{Success: true, Data: gin.H{"message": "session deleted"}})
}

// --- Health Check ---

// HealthCheck handles GET /health
func (h *NLUHandler) HealthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":  "ok",
		"service": "nlu",
	})
}

// --- Full Orchestration: NLU → RAG → LLM-Generation ---

// Ask handles POST /api/v1/nlu/ask
// This is the full orchestration endpoint that:
// 1. Runs NLU to understand the user's intent and entities
// 2. Calls RAG to search for relevant context
// 3. Calls LLM-Generation with NLU results + RAG context to produce the final answer
func (h *NLUHandler) Ask(c *gin.Context) {
	var req domain.AskRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   &APIError{Code: 400, Message: "invalid request: " + err.Error()},
		})
		return
	}
	if req.Text == "" {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   &APIError{Code: 400, Message: "text is required"},
		})
		return
	}

	ctx := c.Request.Context()

	// ========== Step 1: NLU — Fast rule-based analysis (no LLM call) ==========
	h.logger.Info("ask: step 1 — NLU processing (rule-based)", zap.String("text", req.Text))

	nluReq := &domain.NLURequest{
		Text:         req.Text,
		Language:     req.Language,
		SessionID:    req.SessionID,
		Capabilities: []string{"intent", "ner", "sentiment"},
	}
	nluResult, err := h.engine.Process(ctx, nluReq)
	if err != nil {
		h.logger.Error("ask: NLU failed", zap.Error(err))
		nluResult = &domain.NLUResult{Text: req.Text, Errors: []string{"NLU failed: " + err.Error()}}
	}

	h.logger.Info("ask: NLU done",
		zap.Any("intent", nluResult.Intent),
		zap.Int("entities", countEntities(nluResult)),
	)

	// ========== Step 2: RAG — Search for relevant context ==========
	h.logger.Info("ask: step 2 — RAG search")

	var sources []domain.Source
	if h.ragClient != nil {
		topK := req.TopK
		if topK <= 0 {
			topK = 5
		}
		ragReq := client.SearchRequest{
			Query: req.Text,
			TopK:  topK,
		}
		ragResult, err := h.ragClient.Search(ctx, ragReq)
		if err != nil {
			h.logger.Warn("ask: RAG search failed (continuing without context)", zap.Error(err))
		} else if ragResult != nil {
			for _, r := range ragResult.Results {
				sources = append(sources, domain.Source{
					Content: r.Chunk.Content,
					Source:  r.Chunk.Metadata["source"],
					Score:   r.Score,
				})
			}
			h.logger.Info("ask: RAG done", zap.Int("results", len(sources)))
		}
	} else {
		h.logger.Warn("ask: RAG client not configured, skipping retrieval")
	}

	// ========== Step 3: LLM-Generation — Generate final answer ==========
	h.logger.Info("ask: step 3 — LLM generation")

	if h.llmClient == nil {
		c.JSON(http.StatusServiceUnavailable, APIResponse{
			Success: false,
			Error:   &APIError{Code: 503, Message: "LLM generation service not configured"},
		})
		return
	}

	// Build NLU result reference for LLM-Generation.
	var nluRef *client.NLUResultRef
	if nluResult.Intent != nil {
		nluRef = &client.NLUResultRef{
			Intent:     nluResult.Intent.TopIntent.Name,
			Confidence: nluResult.Intent.TopIntent.Confidence,
		}
		if nluResult.Entities != nil {
			for _, e := range nluResult.Entities.Entities {
				nluRef.Entities = append(nluRef.Entities, client.EntityRef{
					Type:  e.Type,
					Value: e.Value,
				})
			}
		}
		if nluResult.Sentiment != nil {
			nluRef.Sentiment = string(nluResult.Sentiment.Label)
		}
	}

	// Build context items from RAG results.
	var contextItems []client.ContextItem
	for _, s := range sources {
		contextItems = append(contextItems, client.ContextItem{
			Content: s.Content,
			Source:  s.Source,
			Score:   s.Score,
		})
	}

	genReq := client.GenerateRequest{
		Prompt:         req.Text,
		Context:        contextItems,
		NLUResult:      nluRef,
		ConversationID: req.SessionID,
		Provider:       req.Provider,
	}

	genResult, err := h.llmClient.Generate(ctx, genReq)
	if err != nil {
		h.logger.Error("ask: LLM generation failed", zap.Error(err))
		c.JSON(http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   &APIError{Code: 500, Message: "generation failed: " + err.Error()},
		})
		return
	}

	h.logger.Info("ask: complete",
		zap.String("provider", genResult.Provider),
		zap.Int("answer_len", len(genResult.Content)),
	)

	// ========== Return combined result ==========
	c.JSON(http.StatusOK, APIResponse{
		Success: true,
		Data: domain.AskResponse{
			Answer:   genResult.Content,
			NLU:      nluResult,
			Sources:  sources,
			Provider: genResult.Provider,
			Model:    genResult.Model,
		},
	})
}

func countEntities(r *domain.NLUResult) int {
	if r == nil || r.Entities == nil {
		return 0
	}
	return len(r.Entities.Entities)
}
