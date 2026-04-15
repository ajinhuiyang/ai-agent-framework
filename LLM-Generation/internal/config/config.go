// Package config handles configuration loading for the LLM Generation service.
package config

import (
	"fmt"
	"strings"

	"github.com/spf13/viper"
)

// Config is the root configuration struct.
type Config struct {
	Server       ServerConfig       `mapstructure:"server"`
	LLM          LLMConfig          `mapstructure:"llm"`
	Conversation ConversationConfig `mapstructure:"conversation"`
	Prompt       PromptConfig       `mapstructure:"prompt"`
	Logging      LoggingConfig      `mapstructure:"logging"`
}

type ServerConfig struct {
	Host         string `mapstructure:"host"`
	Port         int    `mapstructure:"port"`
	ReadTimeout  int    `mapstructure:"read_timeout"`
	WriteTimeout int    `mapstructure:"write_timeout"`
	Mode         string `mapstructure:"mode"`
}

type LLMConfig struct {
	DefaultProvider string       `mapstructure:"default_provider"`
	Temperature     float64      `mapstructure:"temperature"`
	MaxTokens       int          `mapstructure:"max_tokens"`
	OpenAI          OpenAIConfig `mapstructure:"openai"`
	Ollama          OllamaConfig `mapstructure:"ollama"`
	Zhipu           ZhipuConfig  `mapstructure:"zhipu"`
	Qwen            QwenConfig   `mapstructure:"qwen"`
}

type OpenAIConfig struct {
	APIKey  string `mapstructure:"api_key"`
	BaseURL string `mapstructure:"base_url"`
	Model   string `mapstructure:"model"`
}

type OllamaConfig struct {
	BaseURL string `mapstructure:"base_url"`
	Model   string `mapstructure:"model"`
}

type ZhipuConfig struct {
	APIKey  string `mapstructure:"api_key"`
	BaseURL string `mapstructure:"base_url"`
	Model   string `mapstructure:"model"`
}

type QwenConfig struct {
	APIKey  string `mapstructure:"api_key"`
	BaseURL string `mapstructure:"base_url"`
	Model   string `mapstructure:"model"`
}

type ConversationConfig struct {
	MaxTurns      int `mapstructure:"max_turns"`
	ExpireMinutes int `mapstructure:"expire_minutes"`
	MaxSessions   int `mapstructure:"max_sessions"`
}

type PromptConfig struct {
	DefaultSystem string `mapstructure:"default_system"`
	RAGSystem     string `mapstructure:"rag_system"`
}

type LoggingConfig struct {
	Level  string `mapstructure:"level"`
	Format string `mapstructure:"format"`
}

// Load reads configuration from the given file path, with environment variable overrides.
func Load(configPath string) (*Config, error) {
	v := viper.New()

	// Defaults
	v.SetDefault("server.host", "0.0.0.0")
	v.SetDefault("server.port", 8082)
	v.SetDefault("server.read_timeout", 60)
	v.SetDefault("server.write_timeout", 120)
	v.SetDefault("server.mode", "debug")
	v.SetDefault("llm.default_provider", "ollama")
	v.SetDefault("llm.temperature", 0.7)
	v.SetDefault("llm.max_tokens", 4096)
	v.SetDefault("llm.openai.base_url", "http://localhost:11434/v1")
	v.SetDefault("llm.openai.model", "qwen2.5-0.5b-instruct-q8_0")
	v.SetDefault("llm.ollama.base_url", "http://localhost:11434")
	v.SetDefault("llm.ollama.model", "qwen2.5-0.5b-instruct-q8_0")
	v.SetDefault("conversation.max_turns", 50)
	v.SetDefault("conversation.expire_minutes", 60)
	v.SetDefault("conversation.max_sessions", 10000)
	v.SetDefault("logging.level", "info")
	v.SetDefault("logging.format", "console")

	// Config file
	if configPath != "" {
		v.SetConfigFile(configPath)
	} else {
		v.SetConfigName("config")
		v.SetConfigType("yaml")
		v.AddConfigPath("./configs")
		v.AddConfigPath(".")
	}

	// Environment variables: LLM_GEN_SERVER_PORT, etc.
	v.SetEnvPrefix("LLM_GEN")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	if err := v.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	return &cfg, nil
}
