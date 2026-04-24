// RAG Service - Retrieval Augmented Generation
//
// This service handles document ingestion, text splitting, embedding,
// vector storage, and semantic search.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/your-org/rag/internal/api"
	"github.com/your-org/rag/internal/api/handler"
	"github.com/your-org/rag/internal/config"
	embeddingLocal "github.com/your-org/rag/internal/embedding/local"
	embeddingOllama "github.com/your-org/rag/internal/embedding/ollama"
	embeddingOpenAI "github.com/your-org/rag/internal/embedding/openai"
	"github.com/your-org/rag/internal/retriever"
	"github.com/your-org/rag/internal/splitter"
	"github.com/your-org/rag/internal/vectorstore/memory"
)

func main() {
	configPath := flag.String("config", "", "path to config file")
	flag.Parse()

	// Load configuration.
	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}

	// Initialize logger.
	var logger *zap.Logger
	if cfg.Logging.Format == "json" {
		logger, err = zap.NewProduction()
	} else {
		logger, err = zap.NewDevelopment()
	}
	if err != nil {
		log.Fatalf("failed to init logger: %v", err)
	}
	defer logger.Sync()

	// Create embedding provider.
	var embedder interface {
		Embed(ctx context.Context, text string) ([]float32, error)
		EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
		Dimension() int
		Name() string
	}

	switch cfg.Embedding.Provider {
	case "openai":
		embedder = embeddingOpenAI.New(
			cfg.Embedding.OpenAI.APIKey,
			cfg.Embedding.OpenAI.BaseURL,
			cfg.Embedding.OpenAI.Model,
			cfg.Embedding.OpenAI.Dimension,
		)
	case "ollama":
		embedder = embeddingOllama.New(
			cfg.Embedding.Ollama.BaseURL,
			cfg.Embedding.Ollama.Model,
			cfg.Embedding.Ollama.Dimension,
		)
	case "local":
		embedder = embeddingLocal.New(cfg.Embedding.Local.Dimension)
	default:
		log.Fatalf("unsupported embedding provider: %s", cfg.Embedding.Provider)
	}

	logger.Info("embedding provider initialized", zap.String("provider", embedder.Name()))

	// Create vector store.
	store := memory.New()
	logger.Info("vector store initialized", zap.String("type", "memory"))

	// Create text splitter.
	sp := splitter.New(cfg.Splitter.ChunkSize, cfg.Splitter.ChunkOverlap, cfg.Splitter.Separator)

	// Create retriever.
	ret := retriever.New(
		embedder,
		store,
		sp,
		logger,
		cfg.Retriever.DefaultTopK,
		cfg.Retriever.MinScore,
	)

	// Create HTTP handler and router.
	h := handler.New(ret, store, logger)

	gin.SetMode(cfg.Server.Mode)
	router := api.NewRouter(h, logger)

	// Start HTTP server.
	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	srv := &http.Server{
		Addr:         addr,
		Handler:      router,
		ReadTimeout:  time.Duration(cfg.Server.ReadTimeout) * time.Second,
		WriteTimeout: time.Duration(cfg.Server.WriteTimeout) * time.Second,
	}

	go func() {
		logger.Info("RAG service starting", zap.String("addr", addr))
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("server failed", zap.Error(err))
		}
	}()

	// Graceful shutdown.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("shutting down RAG service...")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		logger.Fatal("server forced to shutdown", zap.Error(err))
	}
	logger.Info("RAG service stopped")
}
