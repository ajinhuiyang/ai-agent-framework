// Package api sets up the HTTP router for the RAG service.
package api

import (
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/your-org/rag/internal/api/handler"
	"github.com/your-org/rag/internal/api/middleware"
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

	// RAG API v1
	v1 := r.Group("/api/v1/rag")
	{
		v1.POST("/ingest", h.Ingest)                        // Add documents
		v1.POST("/search", h.Search)                        // Semantic search
		v1.GET("/collections", h.ListCollections)           // List collections
		v1.DELETE("/collections/:name", h.DeleteCollection) // Delete collection

		// Content fetching — scrape external sources and auto-ingest
		fetch := v1.Group("/fetch")
		{
			fetch.POST("/web", h.FetchWeb)       // Scrape a URL
			fetch.POST("/search", h.FetchSearch) // Search engine scraping
			fetch.POST("/file", h.FetchFile)     // Parse local files
		}
	}

	return r
}
