// Local-LLM Inference Service
//
// A local model inference server with Ollama-compatible API.
// Loads GGUF models and serves chat, generate, and embed endpoints
// so that LLM-Generation can use it as an Ollama backend.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/your-org/local-llm/internal/api"
	"github.com/your-org/local-llm/internal/api/handler"
	"github.com/your-org/local-llm/internal/config"
	"github.com/your-org/local-llm/internal/engine"
	"github.com/your-org/local-llm/internal/engine/mock"
	"github.com/your-org/local-llm/internal/engine/native/transformer"
	"github.com/your-org/local-llm/internal/model"
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

	// Determine thread count.
	numThread := cfg.Inference.NumThread
	if numThread <= 0 {
		numThread = runtime.NumCPU()
	}

	// Create inference engine.
	// Always use native Transformer engine. If model loading fails at runtime,
	// the error will be returned to the caller.
	var eng engine.Engine
	engineType := "native"

	// Resolve models directory to absolute path.
	modelsDir := cfg.Models.Dir
	if !filepath.IsAbs(modelsDir) {
		// Try relative to current working directory first.
		if absPath, err := filepath.Abs(modelsDir); err == nil {
			if _, err := os.Stat(absPath); err == nil {
				modelsDir = absPath
			}
		}
		// If not found, try relative to executable location.
		if _, err := os.Stat(modelsDir); err != nil {
			if exePath, err := os.Executable(); err == nil {
				candidate := filepath.Join(filepath.Dir(exePath), modelsDir)
				if _, err := os.Stat(candidate); err == nil {
					modelsDir = candidate
				}
			}
		}
	}

	// Check if models directory has GGUF files.
	hasModels := false
	if entries, err := os.ReadDir(modelsDir); err == nil {
		for _, e := range entries {
			name := e.Name()
			if strings.HasSuffix(name, ".gguf") || strings.HasSuffix(name, ".bin") {
				hasModels = true
				break
			}
		}
	}

	if hasModels {
		eng = transformer.New(logger)
		logger.Info("inference engine initialized",
			zap.String("type", "native (pure Go Transformer)"),
			zap.String("models_dir", modelsDir),
			zap.Int("threads", numThread),
		)
	} else {
		eng = mock.New()
		engineType = "mock"
		logger.Warn("no GGUF models found, using mock engine",
			zap.String("models_dir", modelsDir),
			zap.String("hint", "place .gguf files in the models directory for real inference"),
		)
	}

	// Override config models dir with resolved path.
	cfg.Models.Dir = modelsDir

	// Create load options.
	loadOpts := engine.LoadOptions{
		NumCtx:    cfg.Inference.NumCtx,
		NumGPU:    cfg.Inference.NumGPU,
		NumThread: numThread,
		BatchSize: cfg.Inference.BatchSize,
	}

	// Create model manager.
	mgr := model.New(eng, cfg.Models.Dir, loadOpts, logger)

	// Scan for available models.
	if err := mgr.Scan(); err != nil {
		logger.Warn("model scan failed", zap.Error(err))
	}

	// Pre-load default model in background (don't block HTTP server).
	if cfg.Models.Default != "" {
		go func() {
			logger.Info("pre-loading default model (background)", zap.String("model", cfg.Models.Default))
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
			defer cancel()
			if err := mgr.EnsureLoaded(ctx, cfg.Models.Default); err != nil {
				logger.Error("failed to pre-load default model",
					zap.String("model", cfg.Models.Default),
					zap.Error(err),
				)
			} else {
				logger.Info("default model loaded and ready", zap.String("model", cfg.Models.Default))
			}
		}()
	}

	// Create HTTP handler and router.
	h := handler.New(mgr, cfg.Sampling, cfg.Models.Default, logger)

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
		logger.Info("Local-LLM service starting",
			zap.String("addr", addr),
			zap.String("engine", engineType),
			zap.String("default_model", cfg.Models.Default),
			zap.String("models_dir", cfg.Models.Dir),
		)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("server failed", zap.Error(err))
		}
	}()

	// Graceful shutdown.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("shutting down Local-LLM service...")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Unload model before shutting down.
	if eng.IsLoaded() {
		logger.Info("unloading model...")
		eng.Unload()
	}

	if err := srv.Shutdown(ctx); err != nil {
		logger.Fatal("server forced to shutdown", zap.Error(err))
	}
	logger.Info("Local-LLM service stopped")
}
