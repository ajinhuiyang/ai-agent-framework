// Package config handles configuration loading for the Local-LLM service.
package config

import (
	"fmt"
	"strings"

	"github.com/spf13/viper"
)

// Config is the root configuration.
type Config struct {
	Server    ServerConfig    `mapstructure:"server"`
	Models    ModelsConfig    `mapstructure:"models"`
	Inference InferenceConfig `mapstructure:"inference"`
	Sampling  SamplingConfig  `mapstructure:"sampling"`
	Logging   LoggingConfig   `mapstructure:"logging"`
}

type ServerConfig struct {
	Host         string `mapstructure:"host"`
	Port         int    `mapstructure:"port"`
	ReadTimeout  int    `mapstructure:"read_timeout"`
	WriteTimeout int    `mapstructure:"write_timeout"`
	Mode         string `mapstructure:"mode"`
}

type ModelsConfig struct {
	Dir     string `mapstructure:"dir"`
	Default string `mapstructure:"default"`
}

type InferenceConfig struct {
	NumCtx    int `mapstructure:"num_ctx"`
	NumThread int `mapstructure:"num_thread"`
	NumGPU    int `mapstructure:"num_gpu"`
	BatchSize int `mapstructure:"batch_size"`
}

type SamplingConfig struct {
	Temperature   float64 `mapstructure:"temperature"`
	TopP          float64 `mapstructure:"top_p"`
	TopK          int     `mapstructure:"top_k"`
	RepeatPenalty float64 `mapstructure:"repeat_penalty"`
	NumPredict    int     `mapstructure:"num_predict"`
	Seed          int     `mapstructure:"seed"`
}

type LoggingConfig struct {
	Level  string `mapstructure:"level"`
	Format string `mapstructure:"format"`
}

// Load reads configuration from file and environment variables.
func Load(configPath string) (*Config, error) {
	v := viper.New()

	// Defaults
	v.SetDefault("server.host", "0.0.0.0")
	v.SetDefault("server.port", 11434)
	v.SetDefault("server.read_timeout", 300)
	v.SetDefault("server.write_timeout", 300)
	v.SetDefault("server.mode", "debug")
	v.SetDefault("models.dir", "./models")
	v.SetDefault("models.default", "qwen2.5-0.5b-instruct-q8_0")
	v.SetDefault("inference.num_ctx", 4096)
	v.SetDefault("inference.num_thread", 0)
	v.SetDefault("inference.num_gpu", 0)
	v.SetDefault("inference.batch_size", 512)
	v.SetDefault("sampling.temperature", 0.7)
	v.SetDefault("sampling.top_p", 0.9)
	v.SetDefault("sampling.top_k", 40)
	v.SetDefault("sampling.repeat_penalty", 1.1)
	v.SetDefault("sampling.num_predict", 2048)
	v.SetDefault("sampling.seed", -1)
	v.SetDefault("logging.level", "info")
	v.SetDefault("logging.format", "console")

	if configPath != "" {
		v.SetConfigFile(configPath)
	} else {
		v.SetConfigName("config")
		v.SetConfigType("yaml")
		v.AddConfigPath("./configs")
		v.AddConfigPath(".")
	}

	v.SetEnvPrefix("LOCAL_LLM")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	if err := v.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("failed to read config: %w", err)
		}
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	return &cfg, nil
}
