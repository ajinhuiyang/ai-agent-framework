// Package ollama implements embedding via the Ollama native API.
package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// Provider implements embedding.Provider using the Ollama /api/embed endpoint.
type Provider struct {
	baseURL   string
	model     string
	dimension int
	client    *http.Client
}

// New creates a new Ollama embedding provider.
func New(baseURL, model string, dimension int) *Provider {
	return &Provider{
		baseURL:   baseURL,
		model:     model,
		dimension: dimension,
		client:    &http.Client{},
	}
}

type embedRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type embedBatchRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type embedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

func (p *Provider) Embed(ctx context.Context, text string) ([]float32, error) {
	reqBody := embedRequest{Model: p.model, Input: text}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal embed request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create embed request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama embed request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama embed returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var result embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode embed response: %w", err)
	}
	if len(result.Embeddings) == 0 {
		return nil, fmt.Errorf("ollama embed returned empty embeddings")
	}

	return result.Embeddings[0], nil
}

func (p *Provider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	// Ollama supports batch embedding via array input.
	reqBody := embedBatchRequest{Model: p.model, Input: texts}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal batch embed request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create batch embed request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama batch embed request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama batch embed returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var result embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode batch embed response: %w", err)
	}

	return result.Embeddings, nil
}

func (p *Provider) Dimension() int { return p.dimension }
func (p *Provider) Name() string   { return "ollama" }
