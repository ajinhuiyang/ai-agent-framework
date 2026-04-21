// Package sampler provides sampling parameter utilities.
// It merges request-level options with global defaults.
package sampler

import (
	"github.com/your-org/local-llm/internal/config"
	"github.com/your-org/local-llm/internal/domain"
)

// Merge merges request options with global defaults.
// Request values take precedence over defaults when non-zero.
func Merge(reqOpts *domain.Options, defaults config.SamplingConfig) domain.Options {
	result := domain.Options{
		Temperature:   defaults.Temperature,
		TopP:          defaults.TopP,
		TopK:          defaults.TopK,
		RepeatPenalty: defaults.RepeatPenalty,
		NumPredict:    defaults.NumPredict,
		Seed:          defaults.Seed,
	}

	if reqOpts == nil {
		return result
	}

	if reqOpts.Temperature > 0 {
		result.Temperature = reqOpts.Temperature
	}
	if reqOpts.TopP > 0 {
		result.TopP = reqOpts.TopP
	}
	if reqOpts.TopK > 0 {
		result.TopK = reqOpts.TopK
	}
	if reqOpts.RepeatPenalty > 0 {
		result.RepeatPenalty = reqOpts.RepeatPenalty
	}
	if reqOpts.NumPredict > 0 {
		result.NumPredict = reqOpts.NumPredict
	}
	if reqOpts.Seed != 0 {
		result.Seed = reqOpts.Seed
	}
	if reqOpts.NumCtx > 0 {
		result.NumCtx = reqOpts.NumCtx
	}
	if len(reqOpts.Stop) > 0 {
		result.Stop = reqOpts.Stop
	}
	if reqOpts.NumGPU > 0 {
		result.NumGPU = reqOpts.NumGPU
	}
	if reqOpts.NumThread > 0 {
		result.NumThread = reqOpts.NumThread
	}

	return result
}
