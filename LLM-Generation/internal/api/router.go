// Package api sets up the HTTP router for the LLM Generation service.
package api

import (
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/your-org/llm-generation/internal/api/handler"
	"github.com/your-org/llm-generation/internal/api/middleware"
)

// NewRouter creates and configures the Gin router.
func NewRouter(h *handler.Handler, logger *zap.Logger) *gin.Engine {
	r := gin.New()

	// Middleware
	r.Use(middleware.Recovery(logger))
	r.Use(middleware.Logger(logger))
	r.Use(middleware.CORS())

	// Health check
	r.GET("/health", h.HealthCheck)

	// LLM Generation API v1
	v1 := r.Group("/api/v1/llm")
	{
		// Generation
		v1.POST("/generate", h.Generate)

		// Conversations
		v1.POST("/conversations", h.CreateConversation)
		v1.GET("/conversations/:id", h.GetConversation)
		v1.DELETE("/conversations/:id", h.DeleteConversation)

		// Provider management
		v1.GET("/providers", h.ListProviders)
	}

	return r
}
