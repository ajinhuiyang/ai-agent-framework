// Package config handles configuration loading for the RAG service.
package config

import (
	"fmt"
	"strings"

	"github.com/spf13/viper"
)

// Config is the root configuration struct.
type Config struct {
	Server      ServerConfig      `mapstructure:"server"`
	Embedding   EmbeddingConfig   `mapstructure:"embedding"`
	VectorStore VectorStoreConfig `mapstructure:"vectorstore"`
	Splitter    SplitterConfig    `mapstructure:"splitter"`
	Retriever   RetrieverConfig   `mapstructure:"retriever"`
	WebSearch   WebSearchConfig   `mapstructure:"web_search"`
	Logging     LoggingConfig     `mapstructure:"logging"`
}

type ServerConfig struct {
	Host         string `mapstructure:"host"`
	Port         int    `mapstructure:"port"`
	ReadTimeout  int    `mapstructure:"read_timeout"`
	WriteTimeout int    `mapstructure:"write_timeout"`
	Mode         string `mapstructure:"mode"`
}

type EmbeddingConfig struct {
	Provider string          `mapstructure:"provider"`
	OpenAI   EmbeddingOpenAI `mapstructure:"openai"`
	Ollama   EmbeddingOllama `mapstructure:"ollama"`
}

type EmbeddingOpenAI struct {
	APIKey    string `mapstructure:"api_key"`
	BaseURL   string `mapstructure:"base_url"`
	Model     string `mapstructure:"model"`
	Dimension int    `mapstructure:"dimension"`
}

type EmbeddingOllama struct {
	BaseURL   string `mapstructure:"base_url"`
	Model     string `mapstructure:"model"`
	Dimension int    `mapstructure:"dimension"`
}

type VectorStoreConfig struct {
	Type   string       `mapstructure:"type"`
	Milvus MilvusConfig `mapstructure:"milvus"`
}

type MilvusConfig struct {
	Address  string `mapstructure:"address"`
	Database string `mapstructure:"database"`
}

type SplitterConfig struct {
	ChunkSize    int    `mapstructure:"chunk_size"`
	ChunkOverlap int    `mapstructure:"chunk_overlap"`
	Separator    string `mapstructure:"separator"`
}

type RetrieverConfig struct {
	DefaultTopK   int     `mapstructure:"default_top_k"`
	MinScore      float64 `mapstructure:"min_score"`
	RerankEnabled bool    `mapstructure:"rerank_enabled"`
}

type WebSearchConfig struct {
	Enabled bool   `mapstructure:"enabled"`
	APIKey  string `mapstructure:"api_key"`
	Engine  string `mapstructure:"engine"`
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
	v.SetDefault("server.port", 8081)
	v.SetDefault("server.read_timeout", 30)
	v.SetDefault("server.write_timeout", 30)
	v.SetDefault("server.mode", "debug")
	v.SetDefault("embedding.provider", "ollama")
	v.SetDefault("embedding.openai.dimension", 768)
	v.SetDefault("embedding.ollama.base_url", "http://localhost:11434")
	v.SetDefault("embedding.ollama.model", "qwen2.5-coder-0.5b-q8_0")
	v.SetDefault("embedding.ollama.dimension", 768)
	v.SetDefault("vectorstore.type", "memory")
	v.SetDefault("splitter.chunk_size", 512)
	v.SetDefault("splitter.chunk_overlap", 64)
	v.SetDefault("splitter.separator", "\n\n")
	v.SetDefault("retriever.default_top_k", 5)
	v.SetDefault("retriever.min_score", 0.3)
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

	// Environment variables: RAG_SERVER_PORT, RAG_EMBEDDING_PROVIDER, etc.
	v.SetEnvPrefix("RAG")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	if err := v.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}
		// Config file not found is acceptable; use defaults + env vars.
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	return &cfg, nil
}
