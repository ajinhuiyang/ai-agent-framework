# GoInfer

纯 Go 本地 AI 推理与应用全栈，零外部依赖。

完全使用 Go 语言构建的 AI 推理和应用系统。加载 GGUF 模型、运行 Transformer 推理、理解用户意图、检索知识、生成回答 —— 无需 CGO、Python 或任何外部服务。

[English](./README.md) | 中文

---

### 30 秒跑通

运行前唯一需要下载的是模型文件（506 MB）：

```bash
mkdir -p Local-LLM/models && curl -L -o Local-LLM/models/qwen2.5-0.5b-instruct-q8_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
```

然后分别在四个终端启动服务，即可使用：

```bash
cd Local-LLM      && go run ./cmd/server   # :11434  推理引擎
cd RAG             && go run ./cmd/server   # :8081   知识检索
cd LLM-Generation  && go run ./cmd/server   # :8082   内容生成
cd NLU             && go run ./cmd/server   # :8080   意图理解
```

```bash
curl -s http://localhost:8080/api/v1/nlu/ask \
  -H "Content-Type: application/json" \
  -d '{"text": "Go语言是什么？"}' | jq
```

---

## 目录

- [项目概览](#项目概览)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
  - [下载模型](#1-下载模型)
  - [启动服务](#2-启动所有服务)
  - [发起请求](#3-发起请求)
- [REST API 参考](#rest-api-参考)
  - [Local-LLM（Ollama 兼容）](#local-llmollama-兼容)
  - [NLU 自然语言理解](#nlu-自然语言理解)
  - [RAG 检索增强](#rag-检索增强生成)
  - [LLM-Generation 内容生成](#llm-generation-内容生成)
- [系统架构](#系统架构)
- [项目结构](#项目结构)
- [支持的模型](#支持的模型)
- [性能指标](#性能指标)
- [配置说明](#配置说明)
- [工作原理](#工作原理)
  - [推理流水线](#纯-go-推理流水线)
  - [全链路流程](#全链路流程)
- [依赖库](#依赖库)
- [与 Ollama 的对比](#与-ollama-的对比)
- [如何请求各服务](#如何请求各服务)
  - [全链路调用](#全链路调用推荐)
  - [单独调用各服务](#单独调用各服务)
  - [API 端点速查表](#api-端点速查表)
- [如何更换模型](#如何更换模型)
  - [下载模型](#第一步下载新的-gguf-模型)
  - [修改配置](#第二步修改配置)
  - [重启服务](#第三步重启服务)
  - [验证](#第四步验证)
  - [推荐模型](#推荐模型及下载命令)
- [许可证](#许可证)

---

## 项目概览

GoInfer 由四个独立的微服务组成，协同工作：

```
用户请求
   │
   ▼
NLU (:8080)            理解意图、抽取实体、分析情感
   │
   ▼
RAG (:8081)            从知识库中检索相关内容
   │
   ▼
LLM-Generation (:8082) 编排提示词、生成最终回答
   │
   ▼
Local-LLM (:11434)     本地运行模型推理（Ollama 兼容 API）
   │
   ▼
返回回答
```

四个服务可以一起部署，也可以独立运行。

---

## 核心特性

- **纯 Go 推理引擎** —— 无 CGO、无 C/C++ 依赖、无 Python
- **GGUF 模型支持** —— 加载并运行量化模型（Q4_0、Q8_0、F16、F32）
- **Ollama 兼容 API** —— 可直接替代 Ollama，现有客户端无需修改
- **完整 NLU 管线** —— 意图识别、实体抽取、情感分析、槽位填充
- **RAG 检索增强** —— 文档摄入、文本分块、向量化、语义搜索
- **多后端 LLM 适配** —— 统一接口对接本地模型、OpenAI、Ollama、智谱、通义千问
- **微服务架构** —— 各服务独立部署、独立扩展
- **完全离线** —— 全部在本地运行，数据不出网络

---

## 快速开始

### 1. 下载模型

```bash
mkdir -p Local-LLM/models
curl -L -o Local-LLM/models/qwen2.5-0.5b-instruct-q8_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
```

更多模型请参考 [支持的模型](#支持的模型)。

### 2. 启动所有服务

```bash
# 终端 1 —— 本地推理引擎
cd Local-LLM && go run ./cmd/server

# 终端 2 —— RAG 检索服务
cd RAG && go run ./cmd/server

# 终端 3 —— LLM 生成服务
cd LLM-Generation && go run ./cmd/server

# 终端 4 —— NLU 服务（编排所有服务）
cd NLU && go run ./cmd/server
```

### 3. 发起请求

```bash
curl -s http://localhost:8080/api/v1/nlu/ask \
  -H "Content-Type: application/json" \
  -d '{"text": "Go语言是什么？"}' | jq
```

返回结果包含生成的回答、NLU 分析和 RAG 来源：

```json
{
  "success": true,
  "data": {
    "answer": "Go 是一种静态类型的编译型编程语言...",
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

---

## REST API 参考

### Local-LLM（Ollama 兼容）

**对话：**

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5-0.5b-instruct-q8_0",
  "messages": [{"role": "user", "content": "你好！"}],
  "stream": false
}'
```

**向量化：**

```bash
curl http://localhost:11434/api/embed -d '{
  "model": "qwen2.5-0.5b-instruct-q8_0",
  "input": "你好世界"
}'
```

**列出模型：**

```bash
curl http://localhost:11434/api/tags
```

### NLU 自然语言理解

**全链路（NLU → RAG → LLM 一步到位）：**

```bash
curl http://localhost:8080/api/v1/nlu/ask \
  -d '{"text": "今天北京天气怎么样"}'
```

**仅意图识别：**

```bash
curl http://localhost:8080/api/v1/nlu/intent \
  -d '{"text": "帮我订一张去上海的机票"}'
```

**仅实体抽取：**

```bash
curl http://localhost:8080/api/v1/nlu/ner \
  -d '{"text": "周杰伦明天下午3点在北京开演唱会"}'
```

**仅情感分析：**

```bash
curl http://localhost:8080/api/v1/nlu/sentiment \
  -d '{"text": "这个产品太棒了，质量非常好！"}'
```

**槽位填充：**

```bash
curl http://localhost:8080/api/v1/nlu/slot -d '{
  "text": "帮我订后天从北京到上海的机票，两个人",
  "slot_config": {
    "slot_definitions": [
      {"name": "departure_city", "type": "city", "required": true},
      {"name": "arrival_city", "type": "city", "required": true},
      {"name": "departure_date", "type": "date", "required": true},
      {"name": "passenger_count", "type": "number", "required": false}
    ]
  }
}'
```

### RAG 检索增强生成

**摄入文档：**

```bash
curl http://localhost:8081/api/v1/rag/ingest -d '{
  "documents": [
    {"content": "Go 语言由 Google 于 2007 年设计。", "source": "wiki"},
    {"content": "Go 的并发模型基于 goroutine 和 channel。", "source": "tutorial"}
  ]
}'
```

**语义搜索：**

```bash
curl http://localhost:8081/api/v1/rag/search \
  -d '{"query": "Go 语言是谁设计的？"}'
```

**列出集合：**

```bash
curl http://localhost:8081/api/v1/rag/collections
```

### LLM-Generation 内容生成

**带上下文生成：**

```bash
curl http://localhost:8082/api/v1/llm/generate -d '{
  "prompt": "Go语言是什么？",
  "provider": "ollama",
  "context": [{"content": "Go 由 Google 设计。", "source": "wiki"}]
}'
```

**多轮对话：**

```bash
# 创建会话
curl -X POST http://localhost:8082/api/v1/llm/conversations

# 在会话中生成
curl http://localhost:8082/api/v1/llm/generate -d '{
  "prompt": "继续",
  "conversation_id": "<返回的会话ID>"
}'
```

**查看可用后端：**

```bash
curl http://localhost:8082/api/v1/llm/providers
```

---

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                      用户请求                              │
└──────────────┬───────────────────────────────────────────┘
               ▼
┌──────────────────────────────┐
│     NLU 意图理解 (:8080)      │  意图识别 / 实体抽取 / 情感分析
│                              │  槽位填充 / 多轮对话管理
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│     RAG 检索增强 (:8081)      │  文档分块 / 向量化 / 语义搜索
│                              │  相关性过滤 / 知识召回
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  LLM-Generation 生成 (:8082)  │  提示词编排 / 多后端适配
│                              │  流式输出 / 会话管理
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│   Local-LLM 推理 (:11434)    │  GGUF 解析 / Transformer 推理
│                              │  纯 Go 实现 / Ollama 兼容 API
└──────────────────────────────┘
```

---

## 项目结构

```
GoInfer/
├── NLU/                         # 自然语言理解服务
│   ├── cmd/server/              # 服务入口
│   ├── internal/
│   │   ├── nlu/                 # LLM 驱动的 NLU 能力
│   │   │   ├── intent/          #   意图识别
│   │   │   ├── ner/             #   命名实体识别
│   │   │   ├── sentiment/       #   情感分析
│   │   │   ├── slot/            #   槽位填充
│   │   │   └── dialog/          #   多轮对话管理
│   │   ├── pipeline/            # 并行执行引擎
│   │   ├── client/              # RAG / LLM-Generation HTTP 客户端
│   │   └── api/                 # REST API 处理器
│   └── configs/                 # 领域配置、提示词模板
│
├── RAG/                         # 检索增强生成服务
│   ├── internal/
│   │   ├── embedding/           # 文本向量化
│   │   │   ├── openai/          #   OpenAI 兼容接口
│   │   │   └── ollama/          #   Ollama 原生接口
│   │   ├── vectorstore/         # 向量存储
│   │   │   └── memory/          #   内存实现（可扩展 Milvus）
│   │   ├── splitter/            # 文档分块器
│   │   └── retriever/           # 检索编排器
│   └── configs/
│
├── LLM-Generation/              # 内容生成服务
│   ├── internal/
│   │   ├── llm/                 # LLM Provider 统一接口
│   │   │   ├── openai/          #   OpenAI 兼容 API
│   │   │   ├── ollama/          #   Ollama 原生 API
│   │   │   ├── zhipu/           #   智谱清言 (GLM)
│   │   │   └── qwen/            #   通义千问 (DashScope)
│   │   ├── orchestrator/        # 生成编排器
│   │   └── prompt/              # 提示词模板引擎
│   └── configs/
│
└── Local-LLM/                   # 纯 Go 本地推理引擎
    ├── internal/
    │   ├── engine/
    │   │   ├── native/
    │   │   │   ├── gguf/        #   GGUF 文件解析器
    │   │   │   ├── bpe/         #   BPE 分词器
    │   │   │   └── transformer/ #   Transformer 推理引擎
    │   │   └── mock/            #   模拟引擎（开发测试用）
    │   ├── model/               # 模型生命周期管理
    │   ├── tokenizer/           # Chat 模板引擎
    │   └── sampler/             # 采样参数管理
    ├── models/                  # GGUF 模型文件（不纳入版本控制）
    └── configs/
```

---

## 支持的模型

支持所有基于 LLaMA 架构的 GGUF 格式模型：

| 模型 | 参数量 | 下载大小 (Q8_0) | 说明 |
|------|-------|----------------|------|
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) | 0.5B | ~640 MB | 推荐用于测试，速度最快 |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) | 1.5B | ~1.6 GB | 速度与质量的平衡 |
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) | 3B | ~3.2 GB | 质量更好，速度较慢 |
| [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 1B | ~1.1 GB | Meta 的紧凑模型 |
| [Gemma-2-2B-IT](https://huggingface.co/google/gemma-2-2b-it) | 2B | ~2.5 GB | Google 的指令微调模型 |

从 [HuggingFace](https://huggingface.co) 下载后放入 `Local-LLM/models/` 目录即可。

> **注意：** 纯 Go 推理引擎仅支持 CPU。对于 3B 以上的模型，建议使用 [Ollama](https://ollama.com) 作为后端 —— GoInfer 的所有服务与 Ollama 完全兼容，无需修改代码。

---

## 性能指标

测试环境：Apple M5（10 核 CPU），模型 Qwen2.5-0.5B-Instruct Q8_0：

| 指标 | 数值 |
|------|------|
| 模型加载时间 | ~500 ms |
| 推理速度 | ~20 tokens/秒 |
| 内存占用 | ~1.5 GB |
| 首 token 延迟 | ~100 ms |

---

## 配置说明

每个服务从 `configs/config.yaml` 读取配置，同时支持环境变量覆盖。

### Local-LLM 配置

```yaml
models:
  dir: "./models"                         # 模型文件目录
  default: "qwen2.5-0.5b-instruct-q8_0"  # 默认模型

inference:
  num_ctx: 2048    # 上下文窗口大小
  num_thread: 0    # CPU 线程数（0 = 自动检测）

sampling:
  temperature: 0.7   # 采样温度
  top_p: 0.9         # 核采样概率
  top_k: 40          # Top-K 采样
  num_predict: 256   # 最大生成 token 数
```

### 服务端口

| 服务 | 默认端口 | 环境变量 |
|------|---------|---------|
| NLU | 8080 | `NLU_SERVER_PORT` |
| RAG | 8081 | `RAG_SERVER_PORT` |
| LLM-Generation | 8082 | `LLM_GEN_SERVER_PORT` |
| Local-LLM | 11434 | `LOCAL_LLM_SERVER_PORT` |

---

## 工作原理

### 纯 Go 推理流水线

```
GGUF 模型文件
  → 解析文件头和元数据 (gguf.go)
  → 提取词表、构建 BPE 分词器 (bpe.go)
  → 加载权重并反量化：Q8_0/Q4_0/F16/F32 → float32 (gguf.go)
  → 使用 Chat 模板编码提示词 (template.go)
  → 运行 Transformer 前向推理 (transformer.go)：
      Token Embedding → [N × Transformer Block] → RMSNorm → LM Head → Logits
      每个 Block：
        ├── RMSNorm → GQA Self-Attention (RoPE + KV Cache) → 残差连接
        └── RMSNorm → SwiGLU FFN (gate·up → silu → down) → 残差连接
  → 采样下一个 token（temperature / top-k / top-p）
  → 解码 token 为文本
  → 重复直到 EOS 或达到最大 token 数
```

### 全链路流程

```
"Go语言是什么？"
       │
       ▼
  ┌─ NLU 分析 ──────────────────────────────────┐
  │  意图：ask_question (置信度 0.95)              │
  │  实体：[{类型: PRODUCT, 值: "Go"}]             │
  │  情感：neutral                                │
  └──────────────────────────────┬───────────────┘
                                 ▼
  ┌─ RAG 检索 ──────────────────────────────────┐
  │  查询："Go语言是什么？"                         │
  │  召回：[{内容: "Go 由 Google 设计...", 分数: 0.92}] │
  └──────────────────────────────┬───────────────┘
                                 ▼
  ┌─ LLM-Generation 生成 ───────────────────────┐
  │  系统提示：你是知识库助手...                      │
  │  上下文：[RAG 召回结果]                         │
  │  NLU 标注：[Intent=ask_question, Entity=Go]   │
  │  用户问题：Go语言是什么？                        │
  └──────────────────────────────┬───────────────┘
                                 ▼
  ┌─ Local-LLM 推理 ────────────────────────────┐
  │  模型：qwen2.5-0.5b-instruct-q8_0            │
  │  推理速度：~20 tok/s                           │
  └──────────────────────────────┬───────────────┘
                                 ▼
  "Go 是一种由 Google 设计的静态类型编译型编程语言..."
```

---

## 依赖库

所有依赖均为宽松许可（MIT / Apache-2.0 / BSD）：

| 库 | 许可证 | 用途 |
|----|-------|------|
| [gin-gonic/gin](https://github.com/gin-gonic/gin) | MIT | HTTP 路由框架 |
| [sashabaranov/go-openai](https://github.com/sashabaranov/go-openai) | Apache-2.0 | OpenAI 兼容 API 客户端 |
| [spf13/viper](https://github.com/spf13/viper) | MIT | 配置管理 |
| [uber-go/zap](https://github.com/uber-go/zap) | MIT | 结构化日志 |
| [google/uuid](https://github.com/google/uuid) | BSD-3 | UUID 生成 |

---

## 与 Ollama 的对比

| | GoInfer | Ollama |
|---|---------|--------|
| 推理引擎 | 纯 Go | C++ (llama.cpp via CGO) |
| 需要 CGO | 否 | 是 |
| NLU（意图/实体/情感） | 内置 | 无 |
| RAG（知识检索） | 内置 | 无 |
| 多服务架构 | 是（4 个独立服务） | 单体 |
| 交叉编译 | `go build` 即可 | 需要 C 工具链 |
| 推理速度 | ~20 tok/s (0.5B) | ~100+ tok/s (0.5B) |
| 生产就绪 | 实验性 | 是 |

GoInfer 用推理速度换取了可移植性和完整的应用栈。

---

## 如何请求各服务

### 全链路调用（推荐）

向 NLU 发送一个问题，它自动编排 RAG 检索 + LLM 生成，返回最终回答：

```bash
curl -s -X POST http://localhost:8080/api/v1/nlu/ask \
  -H "Content-Type: application/json" \
  -d '{"text": "Go语言是什么？"}' | jq
```

### 单独调用各服务

**Local-LLM —— 直接和模型对话：**

```bash
curl -s -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-1.5b-instruct-q8_0",
    "stream": false,
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

**NLU —— 仅分析意图（不生成回答）：**

```bash
curl -s -X POST http://localhost:8080/api/v1/nlu/intent \
  -H "Content-Type: application/json" \
  -d '{"text": "帮我订一张去上海的机票"}'
```

**RAG —— 搜索知识库：**

```bash
curl -s -X POST http://localhost:8081/api/v1/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Go语言是什么？"}'
```

**LLM-Generation —— 带上下文生成：**

```bash
curl -s -X POST http://localhost:8082/api/v1/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Go语言是什么？",
    "provider": "ollama",
    "context": [{"content": "Go 由 Google 于 2007 年设计。", "source": "wiki"}]
  }'
```

### API 端点速查表

| 服务 | 端口 | 方法 | 端点 | 说明 |
|------|------|------|------|------|
| NLU | 8080 | POST | `/api/v1/nlu/ask` | 全链路：NLU → RAG → LLM |
| NLU | 8080 | POST | `/api/v1/nlu/intent` | 意图识别 |
| NLU | 8080 | POST | `/api/v1/nlu/ner` | 实体抽取 |
| NLU | 8080 | POST | `/api/v1/nlu/sentiment` | 情感分析 |
| NLU | 8080 | POST | `/api/v1/nlu/slot` | 槽位填充 |
| NLU | 8080 | GET | `/health` | 健康检查 |
| RAG | 8081 | POST | `/api/v1/rag/ingest` | 摄入文档 |
| RAG | 8081 | POST | `/api/v1/rag/search` | 语义搜索 |
| RAG | 8081 | GET | `/api/v1/rag/collections` | 列出集合 |
| LLM-Gen | 8082 | POST | `/api/v1/llm/generate` | 生成内容 |
| LLM-Gen | 8082 | POST | `/api/v1/llm/conversations` | 创建会话 |
| LLM-Gen | 8082 | GET | `/api/v1/llm/providers` | 列出可用后端 |
| Local-LLM | 11434 | POST | `/api/chat` | 对话补全 |
| Local-LLM | 11434 | POST | `/api/generate` | 文本生成 |
| Local-LLM | 11434 | POST | `/api/embed` | 生成向量 |
| Local-LLM | 11434 | GET | `/api/tags` | 列出模型 |

---

## 如何更换模型

### 第一步：下载新的 GGUF 模型

在 [HuggingFace](https://huggingface.co) 上搜索模型名 + "GGUF"，下载 `.gguf` 文件到 `Local-LLM/models/` 目录：

```bash
# 示例：切换到 Qwen2.5-1.5B 以获得更好的回答质量
curl -L -o Local-LLM/models/qwen2.5-1.5b-instruct-q8_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf
```

文件名（去掉 `.gguf` 后缀）就是 API 调用时使用的模型名。

### 第二步：修改配置

编辑 `Local-LLM/configs/config.yaml`，将 `default` 改为新模型名：

```yaml
models:
  dir: "./models"
  default: "qwen2.5-1.5b-instruct-q8_0"   # ← 改这里
```

同时编辑 `LLM-Generation/configs/config.yaml`：

```yaml
llm:
  ollama:
    model: "qwen2.5-1.5b-instruct-q8_0"   # ← 与模型名保持一致
```

### 第三步：重启服务

```bash
# 重启 Local-LLM（必须重启才能加载新模型）
# 在运行的终端按 Ctrl+C 停止，然后重新启动：
cd Local-LLM && go run ./cmd/server

# 重启 LLM-Generation 使其读取新配置
cd LLM-Generation && go run ./cmd/server

# NLU 和 RAG 不需要重启
```

### 第四步：验证

```bash
# 检查模型是否加载成功
curl -s http://localhost:11434/api/tags | jq '.models[].name'

# 测试对话
curl -s -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-1.5b-instruct-q8_0","stream":false,"messages":[{"role":"user","content":"你好"}]}'
```

### 推荐模型及下载命令

| 模型 | 大小 | 速度 | 质量 | 下载命令 |
|------|------|------|------|---------|
| Qwen2.5-0.5B-Instruct Q8_0 | 640 MB | ~20 tok/s | 基础 | `curl -L -o Local-LLM/models/qwen2.5-0.5b-instruct-q8_0.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf` |
| Qwen2.5-1.5B-Instruct Q8_0 | 1.8 GB | ~7 tok/s | 良好 | `curl -L -o Local-LLM/models/qwen2.5-1.5b-instruct-q8_0.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf` |
| Qwen2.5-3B-Instruct Q8_0 | 3.4 GB | ~3 tok/s | 更好 | `curl -L -o Local-LLM/models/qwen2.5-3b-instruct-q8_0.gguf https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q8_0.gguf` |

### 使用其他模型

你可以使用 HuggingFace 上任何 GGUF 格式的模型。下载链接的通用格式为：

```
https://huggingface.co/{组织名}/{仓库名}/resolve/main/{文件名}.gguf
```

查找步骤：
1. 打开 https://huggingface.co
2. 搜索模型名 + "GGUF"（如 "Qwen2.5 1.5B GGUF"）
3. 进入仓库，点击 `.gguf` 文件，复制下载链接
4. 下载后放入 `Local-LLM/models/` 目录
5. 修改配置文件中的 `default` 和 `model` 字段
6. 重启 Local-LLM 和 LLM-Generation

> **提示：** 建议选择 Q8_0 量化版本，在文件大小和质量之间有最好的平衡。Q4_0 更小但质量略差。

---

## 许可证

[MIT](./LICENSE)
