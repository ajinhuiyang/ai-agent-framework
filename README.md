# GoInfer

Get up and running with local AI — pure Go, zero dependencies.

A complete AI inference and application stack written entirely in Go. Load GGUF models, run Transformer inference, understand intent, retrieve knowledge, and generate answers — all without CGO, Python, or external services.

English | [中文](./README_CN.md)

---

### Run in 30 seconds

The only file you need to download before running is the model (506 MB):

```bash
mkdir -p Local-LLM/models && curl -L -o Local-LLM/models/qwen2.5-0.5b-instruct-q8_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
```

Then start the four services in separate terminals and you're ready:

```bash
cd Local-LLM      && go run ./cmd/server   # :11434  Inference engine
cd RAG             && go run ./cmd/server   # :8081   Knowledge retrieval
cd LLM-Generation  && go run ./cmd/server   # :8082   Content generation
cd NLU             && go run ./cmd/server   # :8080   Intent understanding
```

```bash
curl -s http://localhost:8080/api/v1/nlu/ask \
  -H "Content-Type: application/json" \
  -d '{"text": "What is Go?"}' | jq
```

---

## Overview

GoInfer is composed of four independent microservices that work together:

```
User Request
     │
     ▼
 NLU (:8080)          Understands intent, extracts entities
     │
     ▼
 RAG (:8081)          Retrieves relevant knowledge
     │
     ▼
 LLM-Generation (:8082)   Orchestrates prompt and generates answer
     │
     ▼
 Local-LLM (:11434)  Runs model inference locally (Ollama-compatible API)
     │
     ▼
 Answer
```

## Features

- **Pure Go inference engine** — no CGO, no C/C++ dependencies, no Python
- **GGUF model support** — load and run quantized models (Q4_0, Q8_0, F16, F32)
- **Ollama-compatible API** — drop-in replacement, works with existing Ollama clients
- **Full NLU pipeline** — intent recognition, entity extraction, sentiment analysis, slot filling
- **RAG retrieval** — document ingestion, text splitting, embedding, semantic search
- **Multi-provider LLM** — unified interface for local models, OpenAI, Ollama, ZhipuAI, Qwen
- **Microservice architecture** — deploy together or independently
- **Fully offline** — runs entirely on your machine, no data leaves your network

## Quickstart

### 1. Download a model

```bash
mkdir -p Local-LLM/models
curl -L -o Local-LLM/models/qwen2.5-0.5b-instruct-q8_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
```

See [Supported Models](#supported-models) for more options.

### 2. Start all services

```bash
# Terminal 1 — Local inference engine
cd Local-LLM && go run ./cmd/server

# Terminal 2 — RAG service
cd RAG && go run ./cmd/server

# Terminal 3 — LLM generation service
cd LLM-Generation && go run ./cmd/server

# Terminal 4 — NLU service (orchestrates everything)
cd NLU && go run ./cmd/server
```

### 3. Ask a question

```bash
curl -s http://localhost:8080/api/v1/nlu/ask \
  -H "Content-Type: application/json" \
  -d '{"text": "What is Go?"}' | jq
```

Response includes the generated answer, NLU analysis, and RAG sources:

```json
{
  "success": true,
  "data": {
    "answer": "Go is a statically typed, compiled programming language...",
    "nlu": {
      "intent": {"top_intent": {"name": "ask_question", "confidence": 0.95}},
      "entities": {"entities": [{"type": "PRODUCT", "value": "Go"}]}
    },
    "sources": [],
    "provider": "ollama",
    "model": "qwen2.5-0.5b-instruct-q8_0"
  }
}
```

## REST API

### Local-LLM (Ollama-compatible)

Chat with a model:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5-0.5b-instruct-q8_0",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'
```

Generate embeddings:

```bash
curl http://localhost:11434/api/embed -d '{
  "model": "qwen2.5-0.5b-instruct-q8_0",
  "input": "Hello world"
}'
```

List models:

```bash
curl http://localhost:11434/api/tags
```

### NLU

Full pipeline (NLU → RAG → LLM):

```bash
curl http://localhost:8080/api/v1/nlu/ask -d '{"text": "Hello"}'
```

Intent recognition only:

```bash
curl http://localhost:8080/api/v1/nlu/intent -d '{"text": "Book a flight to Shanghai"}'
```

Entity extraction only:

```bash
curl http://localhost:8080/api/v1/nlu/ner -d '{"text": "Jay Chou concert in Beijing tomorrow"}'
```

Sentiment analysis:

```bash
curl http://localhost:8080/api/v1/nlu/sentiment -d '{"text": "This product is amazing!"}'
```

### RAG

Ingest documents:

```bash
curl http://localhost:8081/api/v1/rag/ingest -d '{
  "documents": [{"content": "Go was designed at Google in 2007.", "source": "wiki"}]
}'
```

Semantic search:

```bash
curl http://localhost:8081/api/v1/rag/search -d '{"query": "Who designed Go?"}'
```

### LLM-Generation

Generate with context:

```bash
curl http://localhost:8082/api/v1/llm/generate -d '{
  "prompt": "What is Go?",
  "provider": "ollama",
  "context": [{"content": "Go was designed at Google.", "source": "wiki"}]
}'
```

## Architecture

```
GoInfer/
├── NLU/                 # Intent recognition, NER, sentiment, slot filling
│   ├── cmd/server/      # HTTP server entry point
│   ├── internal/
│   │   ├── nlu/         # LLM-powered NLU capabilities
│   │   ├── pipeline/    # Parallel execution engine
│   │   ├── client/      # HTTP clients for RAG & LLM-Generation
│   │   └── api/         # REST API handlers
│   └── configs/         # Domain schema, prompt templates
│
├── RAG/                 # Retrieval Augmented Generation
│   ├── internal/
│   │   ├── embedding/   # Text vectorization (OpenAI, Ollama)
│   │   ├── vectorstore/ # Vector storage (memory, extensible to Milvus)
│   │   ├── splitter/    # Document chunking
│   │   └── retriever/   # Search orchestration
│   └── configs/
│
├── LLM-Generation/      # Content generation with multi-backend support
│   ├── internal/
│   │   ├── llm/         # Provider interface + implementations
│   │   │   ├── openai/  # OpenAI-compatible API
│   │   │   ├── ollama/  # Ollama native API
│   │   │   ├── zhipu/   # ZhipuAI (GLM)
│   │   │   └── qwen/    # Qwen (DashScope)
│   │   ├── orchestrator/ # Generation pipeline
│   │   └── prompt/      # Template engine
│   └── configs/
│
└── Local-LLM/           # Pure Go local inference engine
    ├── internal/
    │   ├── engine/
    │   │   ├── native/
    │   │   │   ├── gguf/        # GGUF file parser
    │   │   │   ├── bpe/         # BPE tokenizer
    │   │   │   └── transformer/ # Transformer inference
    │   │   └── mock/            # Mock engine for testing
    │   ├── model/       # Model lifecycle management
    │   ├── tokenizer/   # Chat template engine
    │   └── sampler/     # Sampling parameters
    ├── models/          # GGUF model files (not tracked in git)
    └── configs/
```

## Supported Models

Any GGUF-format model based on the LLaMA architecture family:

| Model | Parameters | Download Size (Q8_0) | Notes |
|-------|-----------|---------------------|-------|
| Qwen2.5-0.5B-Instruct | 0.5B | ~640 MB | Recommended for testing |
| Qwen2.5-1.5B-Instruct | 1.5B | ~1.6 GB | Good balance of speed and quality |
| Qwen2.5-3B-Instruct | 3B | ~3.2 GB | Better quality, slower |
| Llama-3.2-1B-Instruct | 1B | ~1.1 GB | Meta's compact model |
| Gemma-2-2B-IT | 2B | ~2.5 GB | Google's instruction-tuned model |

Download from [HuggingFace](https://huggingface.co) and place in `Local-LLM/models/`.

> **Note:** The pure Go inference engine is CPU-only. For models larger than 3B, consider using [Ollama](https://ollama.com) as the backend instead — GoInfer's services are fully compatible.

## Performance

Benchmarked on Apple M5 (10-core CPU), Qwen2.5-0.5B-Instruct Q8_0:

| Metric | Value |
|--------|-------|
| Model load time | ~500 ms |
| Inference speed | ~20 tokens/sec |
| Memory usage | ~1.5 GB |
| First token latency | ~100 ms |

## Configuration

Each service reads from `configs/config.yaml` and supports environment variable overrides.

### Local-LLM

```yaml
models:
  dir: "./models"
  default: "qwen2.5-0.5b-instruct-q8_0"

inference:
  num_ctx: 2048
  num_thread: 0    # 0 = auto-detect CPU cores

sampling:
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  num_predict: 256
```

### Service Ports

| Service | Default Port | Environment Variable |
|---------|-------------|---------------------|
| NLU | 8080 | `NLU_SERVER_PORT` |
| RAG | 8081 | `RAG_SERVER_PORT` |
| LLM-Generation | 8082 | `LLM_GEN_SERVER_PORT` |
| Local-LLM | 11434 | `LOCAL_LLM_SERVER_PORT` |

## How It Works

### Pure Go Inference Pipeline

```
GGUF File
  → Parse header & metadata (gguf.go)
  → Extract vocabulary & build BPE tokenizer (bpe.go)
  → Load & dequantize weights: Q8_0/Q4_0/F16/F32 → float32 (gguf.go)
  → Encode prompt with chat template (template.go)
  → Run Transformer forward pass (transformer.go):
      Token Embedding → [N × Transformer Block] → RMSNorm → LM Head → Logits
      Each block:
        ├── RMSNorm → GQA Self-Attention (RoPE + KV Cache) → Residual
        └── RMSNorm → SwiGLU FFN (gate·up → silu → down) → Residual
  → Sample next token (temperature, top-k, top-p)
  → Decode token back to text
  → Repeat until EOS or max tokens
```

### Full Pipeline Flow

```
"What is Go?" → NLU analyzes intent & entities
              → RAG searches knowledge base
              → LLM-Generation builds prompt with context
              → Local-LLM runs inference
              → "Go is a programming language..."
```

## Dependencies

All dependencies are permissively licensed (MIT/Apache-2.0/BSD):

| Library | License | Purpose |
|---------|---------|---------|
| [gin-gonic/gin](https://github.com/gin-gonic/gin) | MIT | HTTP framework |
| [sashabaranov/go-openai](https://github.com/sashabaranov/go-openai) | Apache-2.0 | OpenAI-compatible API client |
| [spf13/viper](https://github.com/spf13/viper) | MIT | Configuration management |
| [uber-go/zap](https://github.com/uber-go/zap) | MIT | Structured logging |
| [google/uuid](https://github.com/google/uuid) | BSD-3 | UUID generation |

## Compared to Ollama

| | GoInfer | Ollama |
|---|---------|--------|
| Inference engine | Pure Go | C++ (llama.cpp via CGO) |
| CGO required | No | Yes |
| NLU (intent/entity/sentiment) | Built-in | No |
| RAG (retrieval) | Built-in | No |
| Multi-service architecture | Yes (4 services) | Monolithic |
| Cross-compile | `go build` | Requires C toolchain |
| Inference speed | ~20 tok/s (0.5B) | ~100+ tok/s (0.5B) |
| Production-ready | Experimental | Yes |

GoInfer trades raw inference speed for portability and a complete application stack.

## How to Call Each Service

### Full Pipeline (recommended)

Send a question to NLU, it automatically calls RAG and LLM-Generation, returns the final answer:

```bash
curl -s -X POST http://localhost:8080/api/v1/nlu/ask \
  -H "Content-Type: application/json" \
  -d '{"text": "What is Go?"}' | jq
```

### Call Services Individually

**Local-LLM — chat directly with the model:**

```bash
curl -s -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-1.5b-instruct-q8_0",
    "stream": false,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**NLU — analyze intent only (no generation):**

```bash
curl -s -X POST http://localhost:8080/api/v1/nlu/intent \
  -H "Content-Type: application/json" \
  -d '{"text": "Book a flight to Shanghai"}'
```

**RAG — search the knowledge base:**

```bash
curl -s -X POST http://localhost:8081/api/v1/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Go?"}'
```

**LLM-Generation — generate with custom context:**

```bash
curl -s -X POST http://localhost:8082/api/v1/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is Go?",
    "provider": "ollama",
    "context": [{"content": "Go was designed at Google in 2007.", "source": "wiki"}]
  }'
```

### API Endpoint Reference

| Service | Port | Method | Endpoint | Description |
|---------|------|--------|----------|-------------|
| NLU | 8080 | POST | `/api/v1/nlu/ask` | Full pipeline: NLU → RAG → LLM |
| NLU | 8080 | POST | `/api/v1/nlu/intent` | Intent recognition |
| NLU | 8080 | POST | `/api/v1/nlu/ner` | Entity extraction |
| NLU | 8080 | POST | `/api/v1/nlu/sentiment` | Sentiment analysis |
| NLU | 8080 | POST | `/api/v1/nlu/slot` | Slot filling |
| NLU | 8080 | GET | `/health` | Health check |
| RAG | 8081 | POST | `/api/v1/rag/ingest` | Ingest documents |
| RAG | 8081 | POST | `/api/v1/rag/search` | Semantic search |
| RAG | 8081 | GET | `/api/v1/rag/collections` | List collections |
| LLM-Gen | 8082 | POST | `/api/v1/llm/generate` | Generate content |
| LLM-Gen | 8082 | POST | `/api/v1/llm/conversations` | Create conversation |
| LLM-Gen | 8082 | GET | `/api/v1/llm/providers` | List LLM providers |
| Local-LLM | 11434 | POST | `/api/chat` | Chat completion |
| Local-LLM | 11434 | POST | `/api/generate` | Text generation |
| Local-LLM | 11434 | POST | `/api/embed` | Generate embeddings |
| Local-LLM | 11434 | GET | `/api/tags` | List models |

## How to Switch Models

### Step 1: Download a new GGUF model

Find models on [HuggingFace](https://huggingface.co) — search for the model name + "GGUF". Download the `.gguf` file into `Local-LLM/models/`:

```bash
# Example: switch to Qwen2.5-1.5B for better quality
curl -L -o Local-LLM/models/qwen2.5-1.5b-instruct-q8_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf
```

The filename (without `.gguf`) becomes the model name used in API calls.

### Step 2: Update configuration

Edit `Local-LLM/configs/config.yaml` — change `default` to the new model name:

```yaml
models:
  dir: "./models"
  default: "qwen2.5-1.5b-instruct-q8_0"   # ← change this
```

Also update `LLM-Generation/configs/config.yaml`:

```yaml
llm:
  ollama:
    model: "qwen2.5-1.5b-instruct-q8_0"   # ← match the model name
```

### Step 3: Restart services

```bash
# Restart Local-LLM (must restart to load the new model)
# Ctrl+C the running process, then:
cd Local-LLM && go run ./cmd/server

# Restart LLM-Generation to pick up the new config
cd LLM-Generation && go run ./cmd/server

# NLU and RAG do not need to restart
```

### Step 4: Verify

```bash
# Check the model is loaded
curl -s http://localhost:11434/api/tags | jq '.models[].name'

# Test it
curl -s -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-1.5b-instruct-q8_0","stream":false,"messages":[{"role":"user","content":"Hello"}]}'
```

### Recommended Models

| Model | Size | Speed | Quality | Download |
|-------|------|-------|---------|----------|
| Qwen2.5-0.5B-Instruct Q8_0 | 640 MB | ~20 tok/s | Basic | `curl -L -o Local-LLM/models/qwen2.5-0.5b-instruct-q8_0.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf` |
| Qwen2.5-1.5B-Instruct Q8_0 | 1.8 GB | ~7 tok/s | Good | `curl -L -o Local-LLM/models/qwen2.5-1.5b-instruct-q8_0.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf` |
| Qwen2.5-3B-Instruct Q8_0 | 3.4 GB | ~3 tok/s | Better | `curl -L -o Local-LLM/models/qwen2.5-3b-instruct-q8_0.gguf https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q8_0.gguf` |

### Using a Custom Model

You can use any GGUF model from HuggingFace. The download URL pattern is:

```
https://huggingface.co/{org}/{repo}/resolve/main/{filename}.gguf
```

For example, to find Qwen models:
1. Go to https://huggingface.co/Qwen
2. Search for the model with "GGUF" in the name
3. Click the `.gguf` file and copy the download URL
4. Download and place in `Local-LLM/models/`

> **Tip:** Choose Q8_0 quantization for the best balance of size and quality. Q4_0 is smaller but lower quality.

## License

MIT
