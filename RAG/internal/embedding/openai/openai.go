// Package openai implements embedding via the OpenAI-compatible API.
// Works with OpenAI, Ollama (OpenAI-compat mode), vLLM, Azure OpenAI, etc.
package openai

import (
	"context"
	"fmt"

	goai "github.com/sashabaranov/go-openai"
)

// Provider implements embedding.Provider using the OpenAI embeddings API.
type Provider struct {
	client    *goai.Client
	model     string
	dimension int
}

// New creates a new OpenAI embedding provider.
func New(apiKey, baseURL, model string, dimension int) *Provider {
	config := goai.DefaultConfig(apiKey)
	if baseURL != "" {
		config.BaseURL = baseURL
	}
	return &Provider{
		client:    goai.NewClientWithConfig(config),
		model:     model,
		dimension: dimension,
	}
}

func (p *Provider) Embed(ctx context.Context, text string) ([]float32, error) {
	resp, err := p.client.CreateEmbeddings(ctx, goai.EmbeddingRequestStrings{
		Input: []string{text},
		Model: goai.EmbeddingModel(p.model),
	})
	if err != nil {
		return nil, fmt.Errorf("openai embedding failed: %w", err)
	}
	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("openai embedding returned empty data")
	}
	return resp.Data[0].Embedding, nil
}

func (p *Provider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	// Process in batches of 100 to avoid API limits.
	const batchSize = 100
	var allEmbeddings [][]float32

	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[i:end]

		resp, err := p.client.CreateEmbeddings(ctx, goai.EmbeddingRequestStrings{
			Input: batch,
			Model: goai.EmbeddingModel(p.model),
		})
		if err != nil {
			return nil, fmt.Errorf("openai batch embedding failed at index %d: %w", i, err)
		}

		for _, d := range resp.Data {
			allEmbeddings = append(allEmbeddings, d.Embedding)
		}
	}

	return allEmbeddings, nil
}

func (p *Provider) Dimension() int { return p.dimension }
func (p *Provider) Name() string   { return "openai" }
