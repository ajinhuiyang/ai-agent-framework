// Package handler implements Ollama-compatible HTTP API handlers.
// Every endpoint mirrors Ollama's API so that LLM-Generation's
// Ollama provider can call this service without any changes.
package handler

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/your-org/local-llm/internal/config"
	"github.com/your-org/local-llm/internal/domain"
	"github.com/your-org/local-llm/internal/model"
	"github.com/your-org/local-llm/internal/sampler"
	"github.com/your-org/local-llm/internal/tokenizer"
)

// Handler handles Ollama-compatible API requests.
type Handler struct {
	manager      *model.Manager
	defaults     config.SamplingConfig
	defaultModel string
	logger       *zap.Logger
}

// New creates a new Handler.
func New(mgr *model.Manager, defaults config.SamplingConfig, defaultModel string, logger *zap.Logger) *Handler {
	return &Handler{
		manager:      mgr,
		defaults:     defaults,
		defaultModel: defaultModel,
		logger:       logger,
	}
}

// ---------------------------------------------------------------------------
// POST /api/chat — Ollama chat completion
// ---------------------------------------------------------------------------

// Chat handles the Ollama /api/chat endpoint.
func (h *Handler) Chat(c *gin.Context) {
	var req domain.ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request: " + err.Error()})
		return
	}

	modelName := req.Model
	if modelName == "" {
		modelName = h.defaultModel
	}

	// Ensure model is loaded.
	if err := h.manager.EnsureLoaded(c.Request.Context(), modelName); err != nil {
		h.logger.Error("failed to load model", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Merge sampling options.
	opts := sampler.Merge(req.Options, h.defaults)

	// Detect chat template and format prompt.
	tmpl := tokenizer.DetectTemplate(modelName)
	prompt := tokenizer.FormatChat(tmpl, req.Messages)

	inferReq := domain.InferenceRequest{
		Prompt:  prompt,
		Options: opts,
	}

	// Determine streaming (Ollama defaults to stream=true).
	stream := true
	if req.Stream != nil {
		stream = *req.Stream
	}

	if stream {
		h.chatStream(c, modelName, inferReq, tmpl)
	} else {
		h.chatNonStream(c, modelName, inferReq)
	}
}

func (h *Handler) chatNonStream(c *gin.Context, modelName string, req domain.InferenceRequest) {
	eng := h.manager.GetEngine()
	result, err := eng.Predict(c.Request.Context(), req)
	if err != nil {
		h.logger.Error("prediction failed", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	resp := domain.ChatResponse{
		Model:           modelName,
		CreatedAt:       time.Now().UTC(),
		Message:         domain.ChatMessage{Role: "assistant", Content: result.Text},
		Done:            true,
		DoneReason:      "stop",
		TotalDuration:   result.TotalDuration,
		LoadDuration:    result.LoadDuration,
		PromptEvalCount: result.PromptTokens,
		EvalCount:       result.CompletionTokens,
		EvalDuration:    result.EvalDuration,
	}

	c.JSON(http.StatusOK, resp)
}

func (h *Handler) chatStream(c *gin.Context, modelName string, req domain.InferenceRequest, tmpl tokenizer.ChatTemplate) {
	eng := h.manager.GetEngine()
	req.Stream = true

	streamCh, err := eng.PredictStream(c.Request.Context(), req)
	if err != nil {
		h.logger.Error("stream prediction failed", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.Header("Content-Type", "application/x-ndjson")
	c.Header("Cache-Control", "no-cache")

	flusher, _ := c.Writer.(http.Flusher)
	var totalContent string
	tokenCount := 0

	for token := range streamCh {
		if token.Err != nil {
			errResp := domain.ChatResponse{
				Model:     modelName,
				CreatedAt: time.Now().UTC(),
				Message:   domain.ChatMessage{Role: "assistant", Content: ""},
				Done:      true,
			}
			data, _ := json.Marshal(errResp)
			fmt.Fprintf(c.Writer, "%s\n", data)
			if flusher != nil {
				flusher.Flush()
			}
			return
		}

		totalContent += token.Text
		tokenCount++

		resp := domain.ChatResponse{
			Model:     modelName,
			CreatedAt: time.Now().UTC(),
			Message:   domain.ChatMessage{Role: "assistant", Content: token.Text},
			Done:      token.Done,
		}

		if token.Done {
			resp.DoneReason = "stop"
			resp.EvalCount = tokenCount
		}

		data, _ := json.Marshal(resp)
		fmt.Fprintf(c.Writer, "%s\n", data)
		if flusher != nil {
			flusher.Flush()
		}
	}
}

// ---------------------------------------------------------------------------
// POST /api/generate — Ollama text generation
// ---------------------------------------------------------------------------

// Generate handles the Ollama /api/generate endpoint.
func (h *Handler) Generate(c *gin.Context) {
	var req domain.GenerateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request: " + err.Error()})
		return
	}

	modelName := req.Model
	if modelName == "" {
		modelName = h.defaultModel
	}

	if err := h.manager.EnsureLoaded(c.Request.Context(), modelName); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	opts := sampler.Merge(req.Options, h.defaults)
	tmpl := tokenizer.DetectTemplate(modelName)
	prompt := tokenizer.FormatGenerate(tmpl, req.System, req.Prompt)

	inferReq := domain.InferenceRequest{
		Prompt:  prompt,
		Options: opts,
	}

	stream := true
	if req.Stream != nil {
		stream = *req.Stream
	}

	if stream {
		h.generateStream(c, modelName, inferReq)
	} else {
		h.generateNonStream(c, modelName, inferReq)
	}
}

func (h *Handler) generateNonStream(c *gin.Context, modelName string, req domain.InferenceRequest) {
	eng := h.manager.GetEngine()
	result, err := eng.Predict(c.Request.Context(), req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, domain.GenerateResponse{
		Model:           modelName,
		CreatedAt:       time.Now().UTC(),
		Response:        result.Text,
		Done:            true,
		DoneReason:      "stop",
		TotalDuration:   result.TotalDuration,
		LoadDuration:    result.LoadDuration,
		PromptEvalCount: result.PromptTokens,
		EvalCount:       result.CompletionTokens,
		EvalDuration:    result.EvalDuration,
	})
}

func (h *Handler) generateStream(c *gin.Context, modelName string, req domain.InferenceRequest) {
	eng := h.manager.GetEngine()
	req.Stream = true

	streamCh, err := eng.PredictStream(c.Request.Context(), req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.Header("Content-Type", "application/x-ndjson")
	c.Header("Cache-Control", "no-cache")

	flusher, _ := c.Writer.(http.Flusher)
	tokenCount := 0

	for token := range streamCh {
		tokenCount++
		resp := domain.GenerateResponse{
			Model:     modelName,
			CreatedAt: time.Now().UTC(),
			Response:  token.Text,
			Done:      token.Done,
		}
		if token.Done {
			resp.DoneReason = "stop"
			resp.EvalCount = tokenCount
		}

		data, _ := json.Marshal(resp)
		fmt.Fprintf(c.Writer, "%s\n", data)
		if flusher != nil {
			flusher.Flush()
		}
	}
}

// ---------------------------------------------------------------------------
// POST /api/embed — Ollama embeddings
// ---------------------------------------------------------------------------

// Embed handles the Ollama /api/embed endpoint.
func (h *Handler) Embed(c *gin.Context) {
	var req domain.EmbedRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request: " + err.Error()})
		return
	}

	modelName := req.Model
	if modelName == "" {
		modelName = h.defaultModel
	}

	if err := h.manager.EnsureLoaded(c.Request.Context(), modelName); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Normalize input to []string.
	var texts []string
	switch v := req.Input.(type) {
	case string:
		texts = []string{v}
	case []interface{}:
		for _, item := range v {
			if s, ok := item.(string); ok {
				texts = append(texts, s)
			}
		}
	default:
		c.JSON(http.StatusBadRequest, gin.H{"error": "input must be a string or array of strings"})
		return
	}

	eng := h.manager.GetEngine()
	embeddings, err := eng.Embed(c.Request.Context(), texts)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, domain.EmbedResponse{
		Model:      modelName,
		Embeddings: embeddings,
	})
}

// ---------------------------------------------------------------------------
// GET /api/tags — List local models
// ---------------------------------------------------------------------------

// Tags handles the Ollama GET /api/tags endpoint.
func (h *Handler) Tags(c *gin.Context) {
	models := h.manager.ListModels()
	c.JSON(http.StatusOK, domain.TagsResponse{Models: models})
}

// ---------------------------------------------------------------------------
// POST /api/show — Show model info
// ---------------------------------------------------------------------------

// Show handles the Ollama POST /api/show endpoint.
func (h *Handler) Show(c *gin.Context) {
	var req domain.ShowRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := h.manager.EnsureLoaded(c.Request.Context(), req.Name); err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "model not found: " + req.Name})
		return
	}

	meta := h.manager.GetEngine().ModelInfo()
	resp := domain.ShowResponse{
		Details: domain.Details{
			Family:            meta.Family,
			ParameterSize:     meta.ParameterSize,
			QuantizationLevel: meta.QuantLevel,
			Format:            "gguf",
		},
	}

	c.JSON(http.StatusOK, resp)
}

// ---------------------------------------------------------------------------
// HEAD / and GET / — Root health check (Ollama returns "Ollama is running")
// ---------------------------------------------------------------------------

// Root handles the root endpoint, mimicking Ollama's response.
func (h *Handler) Root(c *gin.Context) {
	c.String(http.StatusOK, "Local-LLM is running")
}

// ---------------------------------------------------------------------------
// GET /api/version — Version info
// ---------------------------------------------------------------------------

// Version handles the version endpoint.
func (h *Handler) Version(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"version": "0.1.0"})
}

// ---------------------------------------------------------------------------
// Unused context variable suppressor
// ---------------------------------------------------------------------------
var _ context.Context
