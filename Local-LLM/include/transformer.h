#pragma once

#include "gguf_parser.h"
#include "tensor.h"
#include "tokenizer.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace localllm {

// 模型配置 (从 GGUF 元数据加载)
struct ModelConfig {
    std::string architecture = "llama"; // llama, qwen2, ...
    std::string name;

    int32_t vocab_size = 32000;
    int32_t hidden_size = 4096;       // embedding dimension
    int32_t intermediate_size = 11008; // FFN hidden dimension
    int32_t num_layers = 32;
    int32_t num_heads = 32;           // attention heads
    int32_t num_kv_heads = 32;        // GQA: key-value heads (可能 < num_heads)
    int32_t head_dim = 128;           // hidden_size / num_heads
    int32_t max_seq_len = 2048;
    int32_t context_len = 4096;       // 实际使用的最大上下文长度 (限制KV cache)
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    int32_t rope_dim = 0;            // 0 = head_dim
    bool has_attn_bias = false;       // Qwen2 等模型有 attention bias

    // 从 GGUF 加载
    void load_from_gguf(const GGUFParser& parser);
    void print() const;
};

// Transformer 层的权重
struct TransformerLayerWeights {
    // Attention
    Tensor wq;          // query projection   [hidden_size, num_heads * head_dim]
    Tensor wk;          // key projection     [hidden_size, num_kv_heads * head_dim]
    Tensor wv;          // value projection   [hidden_size, num_kv_heads * head_dim]
    Tensor wo;          // output projection  [num_heads * head_dim, hidden_size]

    // Attention bias (Qwen2 等)
    Tensor bq;          // query bias  [num_heads * head_dim]
    Tensor bk;          // key bias    [num_kv_heads * head_dim]
    Tensor bv;          // value bias  [num_kv_heads * head_dim]

    // Attention norm
    Tensor attn_norm;   // RMSNorm weight [hidden_size]

    // FFN (SwiGLU for LLaMA)
    Tensor w1;          // gate projection   [hidden_size, intermediate_size]
    Tensor w2;          // down projection   [intermediate_size, hidden_size]
    Tensor w3;          // up projection     [hidden_size, intermediate_size]

    // FFN norm
    Tensor ffn_norm;    // RMSNorm weight [hidden_size]
};

// 模型权重
struct ModelWeights {
    Tensor token_embedding;  // [vocab_size, hidden_size]
    Tensor output_norm;      // RMSNorm [hidden_size]
    Tensor output;           // language model head [hidden_size, vocab_size]

    std::vector<TransformerLayerWeights> layers;
};

// KV Cache: 存储每层的 key 和 value
struct KVCache {
    std::vector<Tensor> key_cache;    // [num_layers] x [max_seq_len, num_kv_heads * head_dim]
    std::vector<Tensor> value_cache;  // [num_layers] x [max_seq_len, num_kv_heads * head_dim]
    int32_t seq_len = 0;              // 当前已缓存的序列长度

    void init(const ModelConfig& config);
    void clear();
};

// Transformer 推理引擎
class Transformer {
public:
    Transformer() = default;

    // 加载模型
    bool load_model(const std::string& model_path);

    // 前向传播: 输入 token -> 输出 logits [vocab_size]
    // pos: 当前 token 在序列中的位置
    const std::vector<float>& forward(int32_t token, int32_t pos);

    // 生成文本
    using TokenCallback = std::function<bool(int32_t token, const std::string& text)>;
    struct GenerateResult {
        std::string text;
        int prompt_tokens = 0;
        int completion_tokens = 0;
    };
    GenerateResult generate(const std::string& prompt,
                         int max_tokens = 2048,
                         float temperature = 0.7f,
                         float top_p = 0.9f,
                         TokenCallback callback = nullptr,
                         const std::string& system_prompt = "",
                         float repetition_penalty = 1.3f);

    // 清除 KV cache (开始新对话)
    void reset();

    // 获取 tokenizer
    const Tokenizer& tokenizer() const { return tokenizer_; }
    const ModelConfig& config() const { return config_; }

private:
    // 单层 Transformer 前向传播 (单 token, decode 阶段)
    void forward_layer(int layer, float* x, int32_t pos);

    // Attention 前向 (单 token)
    void forward_attention(int layer, float* x, float* x_out, int32_t pos);

    // FFN 前向 (SwiGLU, 单 token)
    void forward_ffn(int layer, float* x, float* x_out);

    // ---- Batch 前向 (多 token, prefill 阶段) ----
    // X: [hidden_size, seq_len] 列主序, pos_start: 起始位置
    void forward_layer_batch(int layer, float* X, int64_t seq_len, int32_t pos_start);
    void forward_attention_batch(int layer, float* X, float* X_out, int64_t seq_len, int32_t pos_start);
    void forward_ffn_batch(int layer, float* X, float* X_out, int64_t seq_len);

    // Batch prefill: 处理所有 prompt tokens, 返回最后一个 token 的 logits
    std::vector<float> forward_batch(const std::vector<int32_t>& tokens, int32_t pos_start);

    ModelConfig config_;
    ModelWeights weights_;
    KVCache kv_cache_;
    Tokenizer tokenizer_;
    GGUFParser parser_;

    // 临时 buffer
    std::vector<float> x_buf_;      // [hidden_size]
    std::vector<float> x_buf2_;     // [hidden_size]
    std::vector<float> attn_out_;   // [hidden_size]
    std::vector<float> attn_buf_;   // [num_heads, max_seq_len]
    std::vector<float> ffn_buf1_;   // [intermediate_size]
    std::vector<float> ffn_buf2_;   // [intermediate_size]
    std::vector<float> logits_;     // [vocab_size]
    std::vector<float> x_embed_;    // [hidden_size] 持久 embedding buffer

    // Q, K, V 临时 buffer
    std::vector<float> q_buf_;
    std::vector<float> k_buf_;
    std::vector<float> v_buf_;

    // Batch prefill 临时 buffer (按需分配)
    std::vector<float> batch_x_buf_;   // [hidden_size * seq_len]
    std::vector<float> batch_x_buf2_;  // [hidden_size * seq_len]
    std::vector<float> batch_norm_buf_;  // [hidden_size * seq_len] RMSNorm 输出
    std::vector<float> batch_attn_proj_; // [hidden_size * seq_len] Wo 投影输出
    std::vector<float> batch_q_buf_;   // [num_heads * head_dim * seq_len]
    std::vector<float> batch_k_buf_;   // [num_kv_heads * head_dim * seq_len]
    std::vector<float> batch_v_buf_;   // [num_kv_heads * head_dim * seq_len]
    std::vector<float> batch_ffn1_;    // [intermediate_size * seq_len]
    std::vector<float> batch_ffn2_;    // [intermediate_size * seq_len]

    // RoPE 预计算频率表 [head_dim/2]
    std::vector<float> rope_freq_;
};

} // namespace localllm
