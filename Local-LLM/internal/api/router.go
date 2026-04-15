// Package api sets up the HTTP router for the Local-LLM service.
// All routes mirror the Ollama API for full compatibility.
package api

import (
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/your-org/local-llm/internal/api/handler"
	"github.com/your-org/local-llm/internal/api/middleware"
)

// NewRouter creates and configures the Gin router with Ollama-compatible routes.
func NewRouter(h *handler.Handler, logger *zap.Logger) *gin.Engine {
	r := gin.New()

	r.Use(middleware.Recovery(logger))
	r.Use(middleware.Logger(logger))
	r.Use(middleware.CORS())

	// Root — Ollama returns "Ollama is running" here.
	r.GET("/", h.Root)
	r.HEAD("/", h.Root)

	// Ollama API routes
	r.POST("/api/chat", h.Chat)         // Chat completion
	r.POST("/api/generate", h.Generate) // Text generation
	r.POST("/api/embed", h.Embed)       // Embeddings
	r.GET("/api/tags", h.Tags)          // List models
	r.POST("/api/show", h.Show)         // Show model info
	r.GET("/api/version", h.Version)    // Version

	return r
}
