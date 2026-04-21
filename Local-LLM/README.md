# Local-LLM

本地大模型推理引擎，纯 C++ 实现，零第三方依赖。

支持加载 GGUF 格式模型文件，执行 Transformer 推理，并通过 OpenAI 兼容的 HTTP API 对外提供服务。

## 架构

```
GGUF 模型文件 → 解析 → 反量化 → Transformer 推理 → 生成文本 → HTTP API
```

### 模块说明

| 模块 | 文件 | 职责 |
|------|------|------|
| GGUF 解析器 | `gguf_parser.h/cpp` | 解析 GGUF 文件格式，mmap 映射张量数据 |
| 反量化 | `dequantize.h/cpp` | 支持 Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/F16/Q2_K~Q6_K 反量化 |
| 张量运算 | `tensor.h` | Tensor 类、矩阵运算、RMSNorm、RoPE、Softmax、SiLU 等 |
| Tokenizer | `tokenizer.h/cpp` | BPE/SentencePiece 分词，从 GGUF 元数据加载词表 |
| Transformer | `transformer.h/cpp` | 完整 Transformer 推理：Attention + FFN + KV Cache |
| HTTP 服务 | `http_server.h/cpp` | OpenAI 兼容 API，支持流式 SSE 响应 |
| 主程序 | `main.cpp` | CLI 入口：服务器模式 / 交互模式 / 单次推理 |

## 构建

### 依赖

- CMake >= 3.16
- C++17 编译器 (GCC 7+, Clang 5+, AppleClang 10+)
- macOS 自动使用 Accelerate framework

### 编译

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

编译产物为 `build/localllm`。

## 使用

### 1. HTTP API 服务器模式 (默认)

```bash
./localllm -m /path/to/model.gguf -p 8080
```

启动后提供 OpenAI 兼容 API：

```bash
# Chat Completions
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 256
  }'

# 流式响应
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'

# Text Completions
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 100}'

# 健康检查
curl http://localhost:8080/health

# 模型列表
curl http://localhost:8080/v1/models
```

### 2. 交互式聊天模式

```bash
./localllm -m model.gguf -i
```

### 3. 单次推理

```bash
./localllm -m model.gguf --prompt "What is the meaning of life?"
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model, -m` | GGUF 模型文件路径 | (必填) |
| `--port, -p` | HTTP 服务端口 | 8080 |
| `--prompt` | 单次推理的提示文本 | - |
| `--max-tokens` | 最大生成 token 数 | 256 |
| `--temperature` | 采样温度 (0=贪心) | 0.7 |
| `--top-p` | Top-p 核采样阈值 | 0.9 |
| `--interactive, -i` | 交互式聊天模式 | - |
| `--server, -s` | HTTP 服务器模式 | 默认 |

## 支持的量化格式

- **F32** - 全精度浮点
- **F16** - 半精度浮点
- **Q4_0 / Q4_1** - 4-bit 量化
- **Q5_0 / Q5_1** - 5-bit 量化
- **Q8_0** - 8-bit 量化
- **Q2_K ~ Q6_K** - K-quant 系列量化

## 支持的模型架构

- **LLaMA** 系列 (LLaMA 2/3, Mistral, Qwen 等基于 LLaMA 架构的模型)

模型需要为 GGUF 格式。可以使用 [llama.cpp](https://github.com/ggerganov/llama.cpp) 的转换工具将其他格式转为 GGUF。

## 项目结构

```
Local-LLM/
├── CMakeLists.txt          # 构建配置
├── README.md               # 本文件
├── include/                 # 头文件
│   ├── gguf_parser.h       # GGUF 文件解析
│   ├── dequantize.h        # 反量化接口
│   ├── tensor.h            # 张量 + 数学运算
│   ├── tokenizer.h         # BPE 分词器
│   ├── transformer.h       # Transformer 推理
│   └── http_server.h       # HTTP 服务
├── src/                     # 源文件
│   ├── gguf_parser.cpp
│   ├── dequantize.cpp
│   ├── tokenizer.cpp
│   ├── transformer.cpp
│   ├── http_server.cpp
│   └── main.cpp            # 入口
└── build/                   # 构建目录
```
