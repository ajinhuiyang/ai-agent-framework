// Package main is the entry point for the NLU microservice.
package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"gopkg.in/yaml.v3"

	"github.com/your-org/nlu/internal/api"
	"github.com/your-org/nlu/internal/api/handler"
	"github.com/your-org/nlu/internal/client"
	"github.com/your-org/nlu/internal/config"
	"github.com/your-org/nlu/internal/domain"
	"github.com/your-org/nlu/internal/llm"
	llmollama "github.com/your-org/nlu/internal/llm/ollama"
	llmopenai "github.com/your-org/nlu/internal/llm/openai"
	"github.com/your-org/nlu/internal/nlu/classify"
	"github.com/your-org/nlu/internal/nlu/dialog"
	"github.com/your-org/nlu/internal/nlu/intent"
	"github.com/your-org/nlu/internal/nlu/ner"
	"github.com/your-org/nlu/internal/nlu/sentiment"
	"github.com/your-org/nlu/internal/nlu/slot"
	"github.com/your-org/nlu/internal/pipeline"
	"github.com/your-org/nlu/internal/prompt"
)

func main() {
	// Parse command-line flags
	configPath := flag.String("config", "", "path to config file")
	flag.Parse()

	// Load configuration
	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load config: %v\n", err)
		os.Exit(1)
	}

	// Initialize logger
	logger, err := initLogger(cfg.Logging)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to initialize logger: %v\n", err)
		os.Exit(1)
	}
	defer logger.Sync()

	logger.Info("starting NLU service",
		zap.String("provider", cfg.LLM.Provider),
		zap.Int("port", cfg.Server.Port),
	)

	// Initialize LLM provider
	provider, err := initLLMProvider(cfg.LLM)
	if err != nil {
		logger.Fatal("failed to initialize LLM provider", zap.Error(err))
	}
	logger.Info("LLM provider initialized", zap.String("provider", provider.Name()))

	// Initialize prompt manager
	promptManager := prompt.NewManager()

	// Load domain schema from domain.yaml
	var domainSchema *domain.DomainSchema
	if cfg.NLU.DomainSchemaPath != "" {
		schemaData, err := os.ReadFile(cfg.NLU.DomainSchemaPath)
		if err != nil {
			logger.Warn("failed to read domain schema file, pipeline will use defaults",
				zap.String("path", cfg.NLU.DomainSchemaPath),
				zap.Error(err),
			)
		} else {
			var schema domain.DomainSchema
			if err := yaml.Unmarshal(schemaData, &schema); err != nil {
				logger.Warn("failed to parse domain schema, pipeline will use defaults",
					zap.Error(err),
				)
			} else {
				domainSchema = &schema
				logger.Info("domain schema loaded",
					zap.String("name", schema.Name),
					zap.Int("intents", len(schema.Intents)),
					zap.Int("entity_types", len(schema.EntityTypes)),
					zap.Int("slot_definitions", len(schema.SlotDefinitions)),
					zap.Int("categories", len(schema.Categories)),
				)
			}
		}
	}

	// Initialize NLU modules
	intentRecognizer := intent.New(provider, promptManager, logger)
	nerExtractor := ner.New(provider, promptManager, logger)
	slotFiller := slot.New(provider, promptManager, logger)
	sentimentAnalyzer := sentiment.New(provider, promptManager, logger)
	textClassifier := classify.New(provider, promptManager, logger)
	dialogManager := dialog.New(provider, promptManager, logger, cfg.NLU.MaxDialogTurns)

	// Initialize NLU pipeline engine
	engine := pipeline.NewEngine(pipeline.Config{
		IntentRecognizer:  intentRecognizer,
		NERExtractor:      nerExtractor,
		SlotFiller:        slotFiller,
		SentimentAnalyzer: sentimentAnalyzer,
		TextClassifier:    textClassifier,
		DialogManager:     dialogManager,
		Schema:            domainSchema,
		Logger:            logger,
		DefaultCaps:       cfg.NLU.DefaultCapabilities,
	})

	// Initialize HTTP clients for peer services
	ragClient := client.NewRAGClient(
		cfg.Services.RAG.BaseURL,
		time.Duration(cfg.Services.RAG.Timeout)*time.Second,
	)
	llmClient := client.NewLLMClient(
		cfg.Services.LLMGeneration.BaseURL,
		time.Duration(cfg.Services.LLMGeneration.Timeout)*time.Second,
	)

	logger.Info("peer service clients initialized",
		zap.String("rag_url", cfg.Services.RAG.BaseURL),
		zap.String("llm_url", cfg.Services.LLMGeneration.BaseURL),
	)

	// Initialize HTTP handler and router
	h := handler.NewNLUHandler(engine, dialogManager, ragClient, llmClient, logger)
	router := api.NewRouter(h, logger, cfg.Server.Mode)

	// Create HTTP server
	srv := &http.Server{
		Addr:         fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port),
		Handler:      router,
		ReadTimeout:  time.Duration(cfg.Server.ReadTimeout) * time.Second,
		WriteTimeout: time.Duration(cfg.Server.WriteTimeout) * time.Second,
	}

	// Start server in a goroutine
	go func() {
		logger.Info("NLU service listening", zap.String("addr", srv.Addr))
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("server failed", zap.Error(err))
		}
	}()

	// Graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("shutting down NLU service...")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		logger.Error("server forced to shutdown", zap.Error(err))
	}

	logger.Info("NLU service stopped")
}

// initLogger creates a zap logger based on config.
func initLogger(cfg config.LoggingConfig) (*zap.Logger, error) {
	var zapCfg zap.Config
	if cfg.Format == "console" {
		zapCfg = zap.NewDevelopmentConfig()
	} else {
		zapCfg = zap.NewProductionConfig()
	}

	// Set log level
	switch cfg.Level {
	case "debug":
		zapCfg.Level = zap.NewAtomicLevelAt(zapcore.DebugLevel)
	case "info":
		zapCfg.Level = zap.NewAtomicLevelAt(zapcore.InfoLevel)
	case "warn":
		zapCfg.Level = zap.NewAtomicLevelAt(zapcore.WarnLevel)
	case "error":
		zapCfg.Level = zap.NewAtomicLevelAt(zapcore.ErrorLevel)
	default:
		zapCfg.Level = zap.NewAtomicLevelAt(zapcore.InfoLevel)
	}

	return zapCfg.Build()
}

// initLLMProvider creates an LLM provider based on config.
func initLLMProvider(cfg config.LLMConfig) (llm.Provider, error) {
	switch cfg.Provider {
	case "openai":
		return llmopenai.New(llmopenai.Config{
			APIKey:      cfg.OpenAI.APIKey,
			BaseURL:     cfg.OpenAI.BaseURL,
			Model:       cfg.OpenAI.Model,
			OrgID:       cfg.OpenAI.OrgID,
			Temperature: cfg.Temperature,
			MaxTokens:   cfg.MaxTokens,
		}), nil

	case "ollama":
		return llmollama.New(llmollama.Config{
			BaseURL:     cfg.Ollama.BaseURL,
			Model:       cfg.Ollama.Model,
			Timeout:     time.Duration(cfg.Timeout) * time.Second,
			Temperature: cfg.Temperature,
			MaxTokens:   cfg.MaxTokens,
		}), nil

	default:
		return nil, fmt.Errorf("unsupported LLM provider: %s", cfg.Provider)
	}
}
