package config

import (
	"os"
	"testing"
)

func TestLoad_Defaults(t *testing.T) {
	cfg, err := Load("")
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if cfg.Server.Port != 8080 {
		t.Errorf("expected port 8080, got %d", cfg.Server.Port)
	}

	if cfg.Server.Host != "0.0.0.0" {
		t.Errorf("expected host 0.0.0.0, got %s", cfg.Server.Host)
	}

	if cfg.LLM.Provider != "openai" {
		t.Errorf("expected provider openai, got %s", cfg.LLM.Provider)
	}

	if cfg.LLM.Temperature != 0.1 {
		t.Errorf("expected temperature 0.1, got %f", cfg.LLM.Temperature)
	}

	if cfg.LLM.MaxTokens != 2048 {
		t.Errorf("expected max_tokens 2048, got %d", cfg.LLM.MaxTokens)
	}

	if cfg.Logging.Level != "info" {
		t.Errorf("expected log level info, got %s", cfg.Logging.Level)
	}

	if len(cfg.NLU.DefaultCapabilities) == 0 {
		t.Error("expected default capabilities to be set")
	}
}

func TestLoad_EnvOverride(t *testing.T) {
	os.Setenv("NLU_SERVER_PORT", "9090")
	os.Setenv("NLU_LLM_PROVIDER", "ollama")
	defer os.Unsetenv("NLU_SERVER_PORT")
	defer os.Unsetenv("NLU_LLM_PROVIDER")

	cfg, err := Load("")
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if cfg.Server.Port != 9090 {
		t.Errorf("expected port 9090, got %d", cfg.Server.Port)
	}

	if cfg.LLM.Provider != "ollama" {
		t.Errorf("expected provider ollama, got %s", cfg.LLM.Provider)
	}
}

func TestLoad_ConfigFile(t *testing.T) {
	// Try to load from the project's config file
	cfg, err := Load("../../configs/config.yaml")
	if err != nil {
		// Config file may not exist in test environment, which is OK
		t.Skipf("skipping config file test: %v", err)
	}

	if cfg.Server.Port <= 0 {
		t.Errorf("expected valid port, got %d", cfg.Server.Port)
	}
}

func TestLoad_NonExistentFile(t *testing.T) {
	_, err := Load("/nonexistent/path/config.yaml")
	if err == nil {
		t.Error("expected error for non-existent config file")
	}
}
