// Package config handles application configuration via Viper.
package config

import (
	"fmt"
	"strings"

	"github.com/spf13/viper"
)

// Config is the root application configuration.
type Config struct {
	Server   ServerConfig   `mapstructure:"server"`
	LLM      LLMConfig      `mapstructure:"llm"`
	NLU      NLUConfig      `mapstructure:"nlu"`
	Logging  LoggingConfig  `mapstructure:"logging"`
	Services ServicesConfig `mapstructure:"services"`
}

// ServerConfig holds HTTP server settings.
type ServerConfig struct {
	Host         string `mapstructure:"host"`
	Port         int    `mapstructure:"port"`
	ReadTimeout  int    `mapstructure:"read_timeout"`  // seconds
	WriteTimeout int    `mapstructure:"write_timeout"` // seconds
	Mode         string `mapstructure:"mode"`          // "debug", "release", "test"
}

// LLMConfig holds LLM provider settings.
type LLMConfig struct {
	Provider    string       `mapstructure:"provider"` // "openai", "ollama"
	OpenAI      OpenAIConfig `mapstructure:"openai"`
	Ollama      OllamaConfig `mapstructure:"ollama"`
	MaxRetries  int          `mapstructure:"max_retries"`
	Timeout     int          `mapstructure:"timeout"` // seconds
	Temperature float64      `mapstructure:"temperature"`
	MaxTokens   int          `mapstructure:"max_tokens"`
}

// OpenAIConfig holds OpenAI API settings.
type OpenAIConfig struct {
	APIKey  string `mapstructure:"api_key"`
	BaseURL string `mapstructure:"base_url"` // Custom endpoint for compatible APIs
	Model   string `mapstructure:"model"`
	OrgID   string `mapstructure:"org_id"`
}

// OllamaConfig holds Ollama settings.
type OllamaConfig struct {
	BaseURL string `mapstructure:"base_url"`
	Model   string `mapstructure:"model"`
}

// NLUConfig holds NLU pipeline settings.
type NLUConfig struct {
	DefaultCapabilities []string `mapstructure:"default_capabilities"` // Default NLU capabilities to run
	MaxDialogTurns      int      `mapstructure:"max_dialog_turns"`
	DomainSchemaPath    string   `mapstructure:"domain_schema_path"`
	PromptTemplatePath  string   `mapstructure:"prompt_template_path"`
	CacheEnabled        bool     `mapstructure:"cache_enabled"`
	CacheTTL            int      `mapstructure:"cache_ttl"` // seconds
}

// LoggingConfig holds logging settings.
type LoggingConfig struct {
	Level  string `mapstructure:"level"`  // "debug", "info", "warn", "error"
	Format string `mapstructure:"format"` // "json", "console"
}

// ServicesConfig holds peer service connection settings.
type ServicesConfig struct {
	RAG           ServiceEndpoint `mapstructure:"rag"`
	LLMGeneration ServiceEndpoint `mapstructure:"llm_generation"`
}

// ServiceEndpoint holds connection details for a peer service.
type ServiceEndpoint struct {
	BaseURL string `mapstructure:"base_url"`
	Timeout int    `mapstructure:"timeout"` // seconds
}

// Load reads configuration from file and environment variables.
func Load(configPath string) (*Config, error) {
	v := viper.New()

	// Set defaults
	setDefaults(v)

	// Read config file
	if configPath != "" {
		v.SetConfigFile(configPath)
	} else {
		v.SetConfigName("config")
		v.SetConfigType("yaml")
		v.AddConfigPath("./configs")
		v.AddConfigPath(".")
	}

	// Read environment variables (NLU_SERVER_PORT, NLU_LLM_PROVIDER, etc.)
	v.SetEnvPrefix("NLU")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	if err := v.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}
		// Config file not found; rely on defaults and env vars
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	return &cfg, nil
}

func setDefaults(v *viper.Viper) {
	// Server
	v.SetDefault("server.host", "0.0.0.0")
	v.SetDefault("server.port", 8080)
	v.SetDefault("server.read_timeout", 600)
	v.SetDefault("server.write_timeout", 600)
	v.SetDefault("server.mode", "debug")

	// LLM — defaults point to Local-LLM C++ on port 11434
	v.SetDefault("llm.provider", "openai")
	v.SetDefault("llm.max_retries", 3)
	v.SetDefault("llm.timeout", 600)
	v.SetDefault("llm.temperature", 0.1)
	v.SetDefault("llm.max_tokens", 2048)
	v.SetDefault("llm.openai.api_key", "not-needed")
	v.SetDefault("llm.openai.base_url", "http://localhost:11434/v1")
	v.SetDefault("llm.openai.model", "Qwen2.5-Coder-7B-Instruct-Q4_K_M")
	v.SetDefault("llm.ollama.base_url", "http://localhost:11434")
	v.SetDefault("llm.ollama.model", "qwen2.5-coder:7b-instruct-q4_K_M")

	// NLU
	v.SetDefault("nlu.default_capabilities", []string{"intent", "ner", "sentiment"})
	v.SetDefault("nlu.max_dialog_turns", 20)
	v.SetDefault("nlu.domain_schema_path", "./configs/domain.yaml")
	v.SetDefault("nlu.prompt_template_path", "./configs/prompts")
	v.SetDefault("nlu.cache_enabled", true)
	v.SetDefault("nlu.cache_ttl", 300)

	// Logging
	v.SetDefault("logging.level", "info")
	v.SetDefault("logging.format", "json")

	// Peer services
	v.SetDefault("services.rag.base_url", "http://localhost:8081")
	v.SetDefault("services.rag.timeout", 600)
	v.SetDefault("services.llm_generation.base_url", "http://localhost:8082")
	v.SetDefault("services.llm_generation.timeout", 600)
}
