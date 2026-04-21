// LLM Generation Service
//
// This service provides unified LLM content generation with support for
// multiple backends (OpenAI, Ollama, ZhipuAI, Qwen), streaming, and
// multi-turn conversation management.
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

	"github.com/your-org/llm-generation/internal/api"
	"github.com/your-org/llm-generation/internal/api/handler"
	"github.com/your-org/llm-generation/internal/config"
	"github.com/your-org/llm-generation/internal/domain"
	"github.com/your-org/llm-generation/internal/llm"
	llmOllama "github.com/your-org/llm-generation/internal/llm/ollama"
	llmOpenAI "github.com/your-org/llm-generation/internal/llm/openai"
	llmQwen "github.com/your-org/llm-generation/internal/llm/qwen"
	llmZhipu "github.com/your-org/llm-generation/internal/llm/zhipu"
	"github.com/your-org/llm-generation/internal/orchestrator"
	"github.com/your-org/llm-generation/internal/prompt"
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

	// Create LLM providers.
	providers := make(map[string]llm.Provider)

	// Always register OpenAI-compatible provider.
	if cfg.LLM.OpenAI.BaseURL != "" {
		providers["openai"] = llmOpenAI.New(
			cfg.LLM.OpenAI.APIKey,
			cfg.LLM.OpenAI.BaseURL,
			cfg.LLM.OpenAI.Model,
			"openai",
		)
		logger.Info("registered provider", zap.String("name", "openai"), zap.String("model", cfg.LLM.OpenAI.Model))
	}

	// Register Ollama if configured.
	if cfg.LLM.Ollama.BaseURL != "" {
		providers["ollama"] = llmOllama.New(
			cfg.LLM.Ollama.BaseURL,
			cfg.LLM.Ollama.Model,
		)
		logger.Info("registered provider", zap.String("name", "ollama"), zap.String("model", cfg.LLM.Ollama.Model))
	}

	// Register ZhipuAI if API key is set.
	if cfg.LLM.Zhipu.APIKey != "" {
		providers["zhipu"] = llmZhipu.New(
			cfg.LLM.Zhipu.APIKey,
			cfg.LLM.Zhipu.BaseURL,
			cfg.LLM.Zhipu.Model,
		)
		logger.Info("registered provider", zap.String("name", "zhipu"), zap.String("model", cfg.LLM.Zhipu.Model))
	}

	// Register Qwen/DashScope if API key is set.
	if cfg.LLM.Qwen.APIKey != "" {
		providers["qwen"] = llmQwen.New(
			cfg.LLM.Qwen.APIKey,
			cfg.LLM.Qwen.BaseURL,
			cfg.LLM.Qwen.Model,
		)
		logger.Info("registered provider", zap.String("name", "qwen"), zap.String("model", cfg.LLM.Qwen.Model))
	}

	if len(providers) == 0 {
		log.Fatal("no LLM providers configured")
	}

	// Create prompt manager with all templates from config.
	promptMgr := prompt.New(cfg.Prompt.DefaultSystem, cfg.Prompt.RAGSystem, cfg.Prompt.TemplateMap())

	// Create orchestrator.
	defaultConfig := domain.GenerateConfig{
		Temperature: cfg.LLM.Temperature,
		MaxTokens:   cfg.LLM.MaxTokens,
	}

	orch := orchestrator.New(
		providers,
		cfg.LLM.DefaultProvider,
		promptMgr,
		logger,
		defaultConfig,
		cfg.Conversation.MaxTurns,
	)

	// Create HTTP handler and router.
	h := handler.New(orch, promptMgr, logger)

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
		logger.Info("LLM Generation service starting",
			zap.String("addr", addr),
			zap.Int("providers", len(providers)),
			zap.String("default_provider", cfg.LLM.DefaultProvider),
		)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("server failed", zap.Error(err))
		}
	}()

	// Graceful shutdown.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("shutting down LLM Generation service...")
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		logger.Fatal("server forced to shutdown", zap.Error(err))
	}
	logger.Info("LLM Generation service stopped")
}
