// Package model manages model discovery, loading, and lifecycle.
package model

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/your-org/local-llm/internal/domain"
	"github.com/your-org/local-llm/internal/engine"
)

// Manager handles model lifecycle: discovery, loading, caching, unloading.
type Manager struct {
	mu        sync.RWMutex
	engine    engine.Engine
	modelsDir string
	logger    *zap.Logger
	loadOpts  engine.LoadOptions

	// Model registry: maps model name → file path.
	registry map[string]ModelEntry

	// Currently loaded model name.
	currentModel string
}

// ModelEntry describes a model available on disk.
type ModelEntry struct {
	Name       string
	Path       string
	Size       int64
	ModifiedAt time.Time
}

// New creates a new model Manager.
func New(eng engine.Engine, modelsDir string, loadOpts engine.LoadOptions, logger *zap.Logger) *Manager {
	return &Manager{
		engine:    eng,
		modelsDir: modelsDir,
		logger:    logger,
		loadOpts:  loadOpts,
		registry:  make(map[string]ModelEntry),
	}
}

// Scan discovers all GGUF model files in the models directory and registers them.
func (m *Manager) Scan() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	entries, err := os.ReadDir(m.modelsDir)
	if err != nil {
		if os.IsNotExist(err) {
			m.logger.Warn("models directory does not exist, creating", zap.String("dir", m.modelsDir))
			if mkErr := os.MkdirAll(m.modelsDir, 0755); mkErr != nil {
				return fmt.Errorf("create models dir: %w", mkErr)
			}
			return nil
		}
		return fmt.Errorf("scan models dir: %w", err)
	}

	count := 0
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		ext := strings.ToLower(filepath.Ext(name))
		if ext != ".gguf" && ext != ".bin" {
			continue
		}

		info, err := entry.Info()
		if err != nil {
			continue
		}

		// Derive model name: "qwen2.5-7b-q4_k_m.gguf" → "qwen2.5-7b-q4_k_m"
		modelName := strings.TrimSuffix(name, ext)
		// Also register with a colon alias: "qwen2.5:7b"
		modelPath := filepath.Join(m.modelsDir, name)

		m.registry[modelName] = ModelEntry{
			Name:       modelName,
			Path:       modelPath,
			Size:       info.Size(),
			ModifiedAt: info.ModTime(),
		}
		count++
	}

	m.logger.Info("model scan complete", zap.Int("found", count), zap.String("dir", m.modelsDir))
	return nil
}

// RegisterModel manually registers a model (useful for models outside the dir).
func (m *Manager) RegisterModel(name, path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("model file not found: %w", err)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.registry[name] = ModelEntry{
		Name:       name,
		Path:       path,
		Size:       info.Size(),
		ModifiedAt: info.ModTime(),
	}
	return nil
}

// EnsureLoaded makes sure the specified model is loaded. If a different model
// is loaded, it is unloaded first.
func (m *Manager) EnsureLoaded(ctx context.Context, modelName string) error {
	// Fast path: check with read lock first (no blocking during inference).
	m.mu.RLock()
	if m.currentModel == modelName && m.engine.IsLoaded() {
		m.mu.RUnlock()
		return nil
	}
	m.mu.RUnlock()

	// Slow path: need to load, acquire write lock.
	m.mu.Lock()
	defer m.mu.Unlock()

	// Double-check after acquiring write lock.
	if m.currentModel == modelName && m.engine.IsLoaded() {
		return nil
	}

	// Find model in registry.
	entry, ok := m.registry[modelName]
	if !ok {
		// Try fuzzy match: "qwen2.5:7b" might match "qwen2.5-7b-q4_k_m"
		for name, e := range m.registry {
			normalized := strings.ReplaceAll(modelName, ":", "-")
			if strings.Contains(strings.ToLower(name), strings.ToLower(normalized)) {
				entry = e
				ok = true
				break
			}
		}
	}

	if !ok {
		// If no models found and using mock engine, auto-register a placeholder.
		if len(m.registry) == 0 {
			entry = ModelEntry{
				Name:       modelName,
				Path:       filepath.Join(m.modelsDir, modelName+".gguf"),
				Size:       0,
				ModifiedAt: time.Now(),
			}
			m.registry[modelName] = entry
		} else {
			return fmt.Errorf("model %q not found (available: %v)", modelName, m.listNamesLocked())
		}
	}

	// Unload current model if different.
	if m.engine.IsLoaded() {
		m.logger.Info("unloading model", zap.String("model", m.currentModel))
		if err := m.engine.Unload(); err != nil {
			m.logger.Warn("failed to unload model", zap.Error(err))
		}
	}

	// Load the requested model.
	m.logger.Info("loading model",
		zap.String("model", entry.Name),
		zap.String("path", entry.Path),
	)

	if err := m.engine.Load(ctx, entry.Path, m.loadOpts); err != nil {
		return fmt.Errorf("failed to load model %q: %w", entry.Name, err)
	}

	m.currentModel = modelName
	m.logger.Info("model loaded", zap.String("model", modelName))
	return nil
}

// CurrentModel returns the name of the currently loaded model.
func (m *Manager) CurrentModel() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.currentModel
}

// ListModels returns info about all registered models.
func (m *Manager) ListModels() []domain.ModelInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var models []domain.ModelInfo
	for _, entry := range m.registry {
		info := domain.ModelInfo{
			Name:       entry.Name,
			Model:      entry.Name,
			ModifiedAt: entry.ModifiedAt,
			Size:       entry.Size,
			Details: domain.Details{
				Format: "gguf",
				Family: "llama",
			},
		}
		// If this model is currently loaded, enrich with engine metadata.
		if entry.Name == m.currentModel {
			if meta := m.engine.ModelInfo(); meta != nil {
				info.Details.Family = meta.Family
				info.Details.ParameterSize = meta.ParameterSize
				info.Details.QuantizationLevel = meta.QuantLevel
			}
		}
		models = append(models, info)
	}
	return models
}

// ListNames returns all registered model names.
func (m *Manager) ListNames() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.listNamesLocked()
}

// listNamesLocked returns model names without acquiring the lock (caller must hold it).
func (m *Manager) listNamesLocked() []string {
	names := make([]string, 0, len(m.registry))
	for name := range m.registry {
		names = append(names, name)
	}
	return names
}

// GetEngine returns the underlying inference engine (for direct predict calls).
func (m *Manager) GetEngine() engine.Engine {
	return m.engine
}
