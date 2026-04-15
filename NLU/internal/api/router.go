// Package api provides the HTTP router and server setup.
package api

import (
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/your-org/nlu/internal/api/handler"
	"github.com/your-org/nlu/internal/api/middleware"
)

// NewRouter creates and configures the Gin router with all NLU routes.
func NewRouter(h *handler.NLUHandler, logger *zap.Logger, mode string) *gin.Engine {
	if mode == "release" {
		gin.SetMode(gin.ReleaseMode)
	} else if mode == "test" {
		gin.SetMode(gin.TestMode)
	}

	r := gin.New()

	// Global middleware
	r.Use(middleware.Recovery(logger))
	r.Use(middleware.Logger(logger))
	r.Use(middleware.CORS())
	r.Use(middleware.RequestID())

	// Health check
	r.GET("/health", h.HealthCheck)

	// API v1
	v1 := r.Group("/api/v1")
	{
		// NLU endpoints
		nlu := v1.Group("/nlu")
		{
			nlu.POST("/ask", h.Ask)                 // Full pipeline: NLU → RAG → LLM-Generation
			nlu.POST("/process", h.Process)         // Full NLU pipeline
			nlu.POST("/intent", h.IntentOnly)       // Intent recognition only
			nlu.POST("/ner", h.NEROnly)             // NER only
			nlu.POST("/sentiment", h.SentimentOnly) // Sentiment analysis only
			nlu.POST("/classify", h.ClassifyOnly)   // Text classification only
			nlu.POST("/slot", h.SlotFill)           // Slot filling (with intent)
		}

		// Dialog management endpoints
		dlg := v1.Group("/dialog")
		{
			dlg.GET("/:session_id", h.GetDialogState)
			dlg.DELETE("/:session_id", h.DeleteDialog)
		}
	}

	return r
}
