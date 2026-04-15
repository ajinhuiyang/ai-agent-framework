// Package llamacpp provides an inference engine implementation using llama.cpp
// via CGO bindings.
//
// This is the PRODUCTION backend. To use it, you need:
//
//  1. Install llama.cpp: git clone https://github.com/ggerganov/llama.cpp
//  2. Build it: cd llama.cpp && make
//  3. Use a Go binding library such as:
//     - github.com/go-skynet/go-llama.cpp (MIT License)
//     - github.com/mudler/go-ggllm.cpp (MIT License)
//  4. Set CGO_ENABLED=1 and link against llama.cpp libraries
//
// Build command:
//
//	CGO_ENABLED=1 go build -tags llamacpp ./cmd/server
//
// This file provides the structure and interface. The actual CGO calls
// are behind build tags so the project compiles without llama.cpp installed.
package llamacpp

// NOTE: This is a skeleton implementation.
// The actual llama.cpp CGO integration requires:
//
//   #cgo LDFLAGS: -L/path/to/llama.cpp -lllama -lm -lstdc++
//   #cgo CFLAGS: -I/path/to/llama.cpp
//   #include "llama.h"
//
// For a ready-to-use Go binding, see:
//   https://github.com/go-skynet/go-llama.cpp (MIT, recommended)
//
// Example integration:
//
//   import llama "github.com/go-skynet/go-llama.cpp"
//
//   type Engine struct {
//       model *llama.LLama
//   }
//
//   func (e *Engine) Load(ctx context.Context, path string, opts LoadOptions) error {
//       model, err := llama.New(path, llama.SetContext(opts.NumCtx),
//           llama.SetGPULayers(opts.NumGPU), llama.SetThreads(opts.NumThread))
//       e.model = model
//       return err
//   }
//
//   func (e *Engine) Predict(ctx context.Context, req InferenceRequest) (PredictResult, error) {
//       text, err := e.model.Predict(req.Prompt,
//           llama.SetTemperature(req.Options.Temperature),
//           llama.SetTopP(req.Options.TopP),
//           llama.SetTopK(req.Options.TopK),
//           llama.SetTokens(req.Options.NumPredict),
//           llama.SetStopWords(req.Options.Stop...))
//       return PredictResult{Text: text}, err
//   }
//
//   func (e *Engine) PredictStream(ctx context.Context, req InferenceRequest) (<-chan StreamToken, error) {
//       ch := make(chan StreamToken, 32)
//       go func() {
//           defer close(ch)
//           e.model.Predict(req.Prompt,
//               llama.SetTemperature(req.Options.Temperature),
//               llama.SetTokenCallback(func(token string) bool {
//                   select {
//                   case <-ctx.Done():
//                       return false
//                   case ch <- StreamToken{Text: token}:
//                       return true
//                   }
//               }))
//           ch <- StreamToken{Done: true}
//       }()
//       return ch, nil
//   }
//
//   func (e *Engine) Embed(ctx context.Context, texts []string) ([][]float32, error) {
//       var result [][]float32
//       for _, text := range texts {
//           emb, err := e.model.Embeddings(text)
//           if err != nil { return nil, err }
//           result = append(result, emb)
//       }
//       return result, nil
//   }
