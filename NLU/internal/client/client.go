// Package client provides HTTP clients for calling peer services (RAG, LLM-Generation).
package client

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// RAGClient is an HTTP client for the RAG service.
type RAGClient struct {
	baseURL string
	client  *http.Client
}

// NewRAGClient creates a new RAG service client.
func NewRAGClient(baseURL string, timeout time.Duration) *RAGClient {
	if timeout == 0 {
		timeout = 30 * time.Second
	}
	return &RAGClient{
		baseURL: baseURL,
		client: &http.Client{
			Timeout: timeout,
			Transport: &http.Transport{
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
}

// SearchRequest is the request body for RAG search.
type SearchRequest struct {
	Query      string            `json:"query"`
	TopK       int               `json:"top_k,omitempty"`
	MinScore   float64           `json:"min_score,omitempty"`
	Filters    map[string]string `json:"filters,omitempty"`
	Collection string            `json:"collection,omitempty"`
}

// SearchResult is a single search hit from RAG.
type SearchResult struct {
	Chunk    ChunkResult `json:"chunk"`
	Score    float64     `json:"score"`
	Distance float64     `json:"distance"`
}

// ChunkResult is the chunk data from a search result.
type ChunkResult struct {
	ID         string            `json:"id"`
	DocumentID string            `json:"document_id"`
	Content    string            `json:"content"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// SearchResponse is the complete RAG search response.
type SearchResponse struct {
	Results     []SearchResult `json:"results"`
	Query       string         `json:"query"`
	TotalFound  int            `json:"total_found"`
	TimeTakenMs int64          `json:"time_taken_ms"`
}

// Search calls the RAG service to perform semantic search.
func (c *RAGClient) Search(ctx context.Context, req SearchRequest) (*SearchResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal search request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/v1/rag/search", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create search request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("RAG search request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("RAG search returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var apiResp struct {
		Success bool            `json:"success"`
		Data    *SearchResponse `json:"data"`
		Error   *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return nil, fmt.Errorf("decode search response: %w", err)
	}
	if !apiResp.Success || apiResp.Data == nil {
		msg := "unknown error"
		if apiResp.Error != nil {
			msg = apiResp.Error.Message
		}
		return nil, fmt.Errorf("RAG search failed: %s", msg)
	}

	return apiResp.Data, nil
}

// HealthCheck checks if the RAG service is healthy.
func (c *RAGClient) HealthCheck(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("RAG health check failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("RAG returned status %d", resp.StatusCode)
	}
	return nil
}

// -----------------------------------------------------------
// LLM Generation Client
// -----------------------------------------------------------

// LLMClient is an HTTP client for the LLM Generation service.
type LLMClient struct {
	baseURL      string
	client       *http.Client // 用于普通请求 (带 Timeout)
	streamClient *http.Client // 用于 SSE 流式请求 (无 Timeout, 依赖 context 控制取消)
}

// NewLLMClient creates a new LLM Generation service client.
func NewLLMClient(baseURL string, timeout time.Duration) *LLMClient {
	if timeout == 0 {
		timeout = 120 * time.Second
	}
	transport := &http.Transport{
		MaxIdleConnsPerHost: 10,
		IdleConnTimeout:     90 * time.Second,
	}
	return &LLMClient{
		baseURL: baseURL,
		client: &http.Client{
			Timeout:   timeout,
			Transport: transport,
		},
		// SSE 流式请求不能用 http.Client.Timeout —— 它覆盖整个请求生命周期（包括读取
		// 流式 body），会在推理时间过长时直接杀死连接。超时由调用方的 context 控制。
		streamClient: &http.Client{
			Transport: transport,
		},
	}
}

// GenerateRequest is the request body for LLM generation.
type GenerateRequest struct {
	Prompt         string        `json:"prompt"`
	SystemPrompt   string        `json:"system_prompt,omitempty"`
	Context        []ContextItem `json:"context,omitempty"`
	NLUResult      *NLUResultRef `json:"nlu_result,omitempty"`
	ConversationID string        `json:"conversation_id,omitempty"`
	History        []Message     `json:"history,omitempty"`
	Config         *GenConfig    `json:"config,omitempty"`
	Stream         bool          `json:"stream,omitempty"`
	Provider       string        `json:"provider,omitempty"`
}

// ContextItem is a piece of context from RAG.
type ContextItem struct {
	Content string  `json:"content"`
	Source  string  `json:"source,omitempty"`
	Score   float64 `json:"score,omitempty"`
}

// NLUResultRef is a reference to NLU analysis results.
type NLUResultRef struct {
	Intent     string            `json:"intent"`
	Confidence float64           `json:"confidence"`
	Entities   []EntityRef       `json:"entities,omitempty"`
	Sentiment  string            `json:"sentiment,omitempty"`
	Slots      map[string]string `json:"slots,omitempty"`
}

// EntityRef is a named entity reference.
type EntityRef struct {
	Type  string `json:"type"`
	Value string `json:"value"`
}

// Message is a conversation message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// GenConfig provides generation parameters.
type GenConfig struct {
	Temperature float64 `json:"temperature,omitempty"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Model       string  `json:"model,omitempty"`
}

// GenerateResponse is the LLM generation result.
type GenerateResponse struct {
	Content        string `json:"content"`
	ConversationID string `json:"conversation_id,omitempty"`
	Provider       string `json:"provider"`
	Model          string `json:"model"`
	FinishReason   string `json:"finish_reason"`
	Truncated      bool   `json:"-"` // derived: true if finish_reason == "length"
}

// Generate calls the LLM Generation service to produce content.
func (c *LLMClient) Generate(ctx context.Context, req GenerateRequest) (*GenerateResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal generate request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/v1/llm/generate", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create generate request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("LLM generate request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("LLM generate returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var apiResp struct {
		Success bool              `json:"success"`
		Data    *GenerateResponse `json:"data"`
		Error   *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return nil, fmt.Errorf("decode generate response: %w", err)
	}
	if !apiResp.Success || apiResp.Data == nil {
		msg := "unknown error"
		if apiResp.Error != nil {
			msg = apiResp.Error.Message
		}
		return nil, fmt.Errorf("LLM generate failed: %s", msg)
	}

	// 标记是否因 max_tokens 截断
	if apiResp.Data.FinishReason == "length" {
		apiResp.Data.Truncated = true
	}

	return apiResp.Data, nil
}

// StreamChunk is a single chunk from streaming LLM generation.
type StreamChunk struct {
	Content      string `json:"content"`
	Done         bool   `json:"done"`
	FinishReason string `json:"finish_reason,omitempty"`
	FullContent  string `json:"full_content,omitempty"`
	Provider     string `json:"provider,omitempty"`
	Model        string `json:"model,omitempty"`
}

// GenerateStream calls the LLM Generation service with streaming enabled.
// Returns a channel that emits parsed SSE chunks.
func (c *LLMClient) GenerateStream(ctx context.Context, req GenerateRequest) (<-chan StreamChunk, error) {
	req.Stream = true
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal generate request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/v1/llm/generate", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create generate request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := c.streamClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("LLM stream request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("LLM stream returned status %d: %s", resp.StatusCode, string(respBody))
	}

	ch := make(chan StreamChunk, 64)
	go func() {
		defer close(ch)
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 64*1024), 256*1024) // 增大 buffer 防止大 chunk 截断
		receivedDone := false
		for scanner.Scan() {
			line := scanner.Text()
			// SSE format: "data: {...}"
			if len(line) < 6 || line[:6] != "data: " {
				continue
			}
			data := line[6:]
			if data == "[DONE]" {
				ch <- StreamChunk{Done: true}
				receivedDone = true
				return
			}
			var chunk StreamChunk
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue
			}
			ch <- chunk
			if chunk.Done {
				receivedDone = true
				return
			}
		}
		// scanner 结束但未收到 done 信号（连接断开/出错），发送兜底 done
		if !receivedDone {
			ch <- StreamChunk{Done: true, FinishReason: "disconnect"}
		}
	}()

	return ch, nil
}

// HealthCheck checks if the LLM Generation service is healthy.
func (c *LLMClient) HealthCheck(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("LLM health check failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("LLM returned status %d", resp.StatusCode)
	}
	return nil
}
