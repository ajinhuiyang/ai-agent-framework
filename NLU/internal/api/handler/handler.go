// Package handler implements HTTP request handlers for the NLU API.
package handler

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

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

	// 默认开启流式响应：Local-LLM 推理速度有限，流式可大幅降低首 token 延迟，
	// 避免客户端因等待全量输出而超时。
	// Stream 是 *bool: nil = 用户未传(默认开启流式), true = 显式开启, false = 显式关闭
	streamEnabled := true
	if req.Stream != nil {
		streamEnabled = *req.Stream
	}

	ctx := c.Request.Context()

	// 设置请求级别超时，确保在 WriteTimeout 之前返回错误响应
	// 而不是让 Go http.Server 的 WriteTimeout 直接关闭 TCP 连接 (导致 empty reply)
	// 注意：本地大模型推理速度很慢（14B Q4 在 CPU 上约 0.16 tok/s），
	// 生成一个完整回答可能需要 10-30 分钟，因此超时需要设得足够大。
	ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()

	// ========== Step 1 & 2: NLU + RAG 并行执行 (互不依赖) ==========
	h.logger.Info("ask: step 1+2 — NLU + RAG in parallel", zap.String("text", req.Text))

	var nluResult *domain.NLUResult
	var sources []domain.Source
	var wg sync.WaitGroup

	// NLU: 规则引擎 (通常 <1ms)
	wg.Add(1)
	go func() {
		defer wg.Done()
		nluReq := &domain.NLURequest{
			Text:         req.Text,
			Language:     req.Language,
			SessionID:    req.SessionID,
			Capabilities: []string{"intent", "ner", "sentiment"},
		}
		result, err := h.engine.Process(ctx, nluReq)
		if err != nil {
			h.logger.Error("ask: NLU failed", zap.Error(err))
			result = &domain.NLUResult{Text: req.Text, Errors: []string{"NLU failed: " + err.Error()}}
		}
		nluResult = result
	}()

	// RAG: 语义检索
	// 流式模式下给 RAG 设置短超时 (2s)，避免拖慢首 token 延迟；
	// 非流式模式下等待完成。
	wg.Add(1)
	go func() {
		defer wg.Done()
		if h.ragClient == nil {
			h.logger.Warn("ask: RAG client not configured, skipping retrieval")
			return
		}
		ragTimeout := 30 * time.Second
		if streamEnabled {
			ragTimeout = 2 * time.Second
		}
		ragCtx, ragCancel := context.WithTimeout(ctx, ragTimeout)
		defer ragCancel()

		topK := req.TopK
		if topK <= 0 {
			topK = 3 // 减少 RAG 结果数量, 降低 prompt token 数
		}
		ragReq := client.SearchRequest{
			Query: req.Text,
			TopK:  topK,
		}
		ragResult, err := h.ragClient.Search(ragCtx, ragReq)
		if err != nil {
			h.logger.Warn("ask: RAG search failed (continuing without context)", zap.Error(err))
			return
		}
		if ragResult != nil {
			for _, r := range ragResult.Results {
				sources = append(sources, domain.Source{
					Content: r.Chunk.Content,
					Source:  r.Chunk.Metadata["source"],
					Score:   r.Score,
				})
			}
		}
	}()

	wg.Wait()

	h.logger.Info("ask: NLU+RAG done",
		zap.Any("intent", nluResult.Intent),
		zap.Int("entities", countEntities(nluResult)),
		zap.Int("rag_results", len(sources)),
	)

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

	// 根据 NLU 识别的 intent 设置专门的 system prompt key，
	// LLM-Generation 的 prompt manager 会将 key 解析为对应的模板内容。
	if nluRef != nil && nluRef.Intent != "" {
		genReq.SystemPrompt = nluRef.Intent // e.g. "code_generate", "code_explain"
	}

	// ========== 流式 vs 非流式 ==========
	if streamEnabled {
		h.askStream(c, ctx, genReq, nluResult, sources)
		return
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
		zap.Bool("truncated", genResult.Truncated),
	)

	// ========== Return combined result ==========
	c.JSON(http.StatusOK, APIResponse{
		Success: true,
		Data: domain.AskResponse{
			Answer:    genResult.Content,
			NLU:       nluResult,
			Sources:   sources,
			Provider:  genResult.Provider,
			Model:     genResult.Model,
			Truncated: genResult.Truncated,
		},
	})
}

func countEntities(r *domain.NLUResult) int {
	if r == nil || r.Entities == nil {
		return 0
	}
	return len(r.Entities.Entities)
}

// askStream 处理流式 Ask 请求：先发 NLU 元数据，再流式推送 LLM 生成的内容
func (h *NLUHandler) askStream(c *gin.Context, ctx context.Context,
	genReq client.GenerateRequest, nluResult *domain.NLUResult, sources []domain.Source) {

	streamCh, err := h.llmClient.GenerateStream(ctx, genReq)
	if err != nil {
		h.logger.Error("ask: LLM stream failed", zap.Error(err))
		c.JSON(http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   &APIError{Code: 500, Message: "stream failed: " + err.Error()},
		})
		return
	}

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no") // 禁用 nginx/反向代理 buffering
	c.Writer.Flush()

	flusher, _ := c.Writer.(http.Flusher)

	// 先发送 NLU 元数据 (让客户端立刻知道 intent/entities)
	meta := struct {
		Type    string            `json:"type"`
		NLU     *domain.NLUResult `json:"nlu,omitempty"`
		Sources []domain.Source   `json:"sources,omitempty"`
	}{
		Type:    "metadata",
		NLU:     nluResult,
		Sources: sources,
	}
	metaJSON, _ := json.Marshal(meta)
	fmt.Fprintf(c.Writer, "data: %s\n\n", metaJSON)
	if flusher != nil {
		flusher.Flush()
	}

	// 流式推送 LLM 生成的内容，同时累积完整回复
	var fullContent string
	gotDone := false
	for chunk := range streamCh {
		fullContent += chunk.Content

		if chunk.Done {
			gotDone = true
			// 优先使用 LLM-Generation 传回的修复后完整内容
			finalContent := fullContent
			if chunk.FullContent != "" {
				finalContent = chunk.FullContent
			}
			// 最后一个事件：附带完整回复内容和 finish_reason
			event := struct {
				Type         string `json:"type"`
				Content      string `json:"content"`
				Done         bool   `json:"done"`
				FullContent  string `json:"full_content"`
				FinishReason string `json:"finish_reason,omitempty"`
			}{
				Type:         "content",
				Content:      chunk.Content,
				Done:         true,
				FullContent:  finalContent,
				FinishReason: chunk.FinishReason,
			}
			data, _ := json.Marshal(event)
			fmt.Fprintf(c.Writer, "data: %s\n\n", data)
			if flusher != nil {
				flusher.Flush()
			}
			break
		}

		event := struct {
			Type    string `json:"type"`
			Content string `json:"content"`
			Done    bool   `json:"done"`
		}{
			Type:    "content",
			Content: chunk.Content,
			Done:    false,
		}
		data, _ := json.Marshal(event)
		fmt.Fprintf(c.Writer, "data: %s\n\n", data)
		if flusher != nil {
			flusher.Flush()
		}
	}

	// 兜底：如果 channel 关闭但没收到 done 事件，仍然发送完整回复
	if !gotDone && fullContent != "" {
		h.logger.Warn("ask: stream channel closed without done event, sending fallback",
			zap.Int("content_len", len(fullContent)),
		)
		event := struct {
			Type         string `json:"type"`
			Content      string `json:"content"`
			Done         bool   `json:"done"`
			FullContent  string `json:"full_content"`
			FinishReason string `json:"finish_reason,omitempty"`
		}{
			Type:         "content",
			Content:      "",
			Done:         true,
			FullContent:  fullContent,
			FinishReason: "disconnect",
		}
		data, _ := json.Marshal(event)
		fmt.Fprintf(c.Writer, "data: %s\n\n", data)
		if flusher != nil {
			flusher.Flush()
		}
	} else if !gotDone && fullContent == "" {
		// 没有收到任何内容就断开了
		h.logger.Error("ask: stream channel closed with no content")
		event := struct {
			Type         string `json:"type"`
			Content      string `json:"content"`
			Done         bool   `json:"done"`
			FinishReason string `json:"finish_reason,omitempty"`
		}{
			Type:         "content",
			Content:      "[error: LLM generation returned no content]",
			Done:         true,
			FinishReason: "error",
		}
		data, _ := json.Marshal(event)
		fmt.Fprintf(c.Writer, "data: %s\n\n", data)
		if flusher != nil {
			flusher.Flush()
		}
	}
}
