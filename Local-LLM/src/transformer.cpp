#include "transformer.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>

namespace localllm {

// ======================== ModelConfig ========================

void ModelConfig::load_from_gguf(const GGUFParser& parser) {
    architecture = parser.get_string("general.architecture", "llama");
    name = parser.get_string("general.name", "unknown");

    std::string prefix = architecture + ".";

    vocab_size = parser.get_uint32("tokenizer.ggml.tokens", vocab_size);
    // 从 token 数组获取实际词表大小
    if (parser.has_metadata("tokenizer.ggml.tokens")) {
        const auto& tokens_val = parser.get_metadata("tokenizer.ggml.tokens");
        if (tokens_val.type == GGUFMetadataValueType::ARRAY) {
            vocab_size = static_cast<int32_t>(tokens_val.arr_values.size());
        }
    }

    hidden_size = parser.get_uint32(prefix + "embedding_length", hidden_size);
    intermediate_size = parser.get_uint32(prefix + "feed_forward_length", intermediate_size);
    num_layers = parser.get_uint32(prefix + "block_count", num_layers);
    num_heads = parser.get_uint32(prefix + "attention.head_count", num_heads);
    num_kv_heads = parser.get_uint32(prefix + "attention.head_count_kv", num_heads);
    max_seq_len = parser.get_uint32(prefix + "context_length", max_seq_len);
    rms_norm_eps = parser.get_float32(prefix + "attention.layer_norm_rms_epsilon", rms_norm_eps);
    rope_theta = parser.get_float32(prefix + "rope.freq_base", rope_theta);

    head_dim = hidden_size / num_heads;
    rope_dim = head_dim;

    // 限制实际使用的上下文长度, 避免 KV cache 占用过多内存
    context_len = std::min(max_seq_len, (int32_t)4096);

    // 检测是否有 attention bias (通过查找张量)
    has_attn_bias = (parser.find_tensor("blk.0.attn_q.bias") != nullptr);
}

void ModelConfig::print() const {
    std::cout << "=== Model Configuration ===" << std::endl;
    std::cout << "  Architecture:      " << architecture << std::endl;
    std::cout << "  Name:              " << name << std::endl;
    std::cout << "  Vocab size:        " << vocab_size << std::endl;
    std::cout << "  Hidden size:       " << hidden_size << std::endl;
    std::cout << "  Intermediate size: " << intermediate_size << std::endl;
    std::cout << "  Num layers:        " << num_layers << std::endl;
    std::cout << "  Num heads:         " << num_heads << std::endl;
    std::cout << "  Num KV heads:      " << num_kv_heads << std::endl;
    std::cout << "  Head dim:          " << head_dim << std::endl;
    std::cout << "  Max seq len:       " << max_seq_len << std::endl;
    std::cout << "  Context len:       " << context_len << " (actual KV cache)" << std::endl;
    std::cout << "  RMS norm eps:      " << rms_norm_eps << std::endl;
    std::cout << "  RoPE theta:        " << rope_theta << std::endl;
    std::cout << "  Attention bias:    " << (has_attn_bias ? "yes" : "no") << std::endl;
    std::cout << "===========================" << std::endl;
}

// ======================== KVCache ========================

void KVCache::init(const ModelConfig& config) {
    seq_len = 0;
    int kv_dim = config.num_kv_heads * config.head_dim;
    int ctx_len = config.context_len;

    key_cache.resize(config.num_layers);
    value_cache.resize(config.num_layers);

    for (int i = 0; i < config.num_layers; ++i) {
        key_cache[i] = Tensor({ctx_len, kv_dim});
        value_cache[i] = Tensor({ctx_len, kv_dim});
    }

    size_t kv_mb = config.num_layers * 2ULL * ctx_len * kv_dim * 4 / 1024 / 1024;
    std::cout << "[KVCache] Initialized: " << config.num_layers << " layers, "
              << "KV dim: " << kv_dim << ", Context len: " << ctx_len
              << " (" << kv_mb << " MB)" << std::endl;
}

void KVCache::clear() {
    seq_len = 0;
    for (auto& k : key_cache) k.zero();
    for (auto& v : value_cache) v.zero();
}

// ======================== Transformer ========================

bool Transformer::load_model(const std::string& model_path) {
    std::cout << "[Model] Loading model from: " << model_path << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // 1. 解析 GGUF 文件
    if (!parser_.load(model_path)) {
        std::cerr << "[Model] Failed to parse GGUF file" << std::endl;
        return false;
    }

    // 2. 加载配置
    config_.load_from_gguf(parser_);
    config_.print();

    // 3. 加载 tokenizer
    if (!tokenizer_.load_from_gguf(parser_)) {
        std::cerr << "[Model] Failed to load tokenizer" << std::endl;
        return false;
    }

    // 更新 vocab_size
    if (tokenizer_.vocab_size() > 0) {
        config_.vocab_size = tokenizer_.vocab_size();
    }

    // 4. 加载模型权重
    std::cout << "[Model] Loading weights..." << std::endl;

    try {
        // Token embedding: 需要反量化因为要按 token ID 索引行
        weights_.token_embedding = Tensor::from_gguf(parser_, "token_embd.weight", true);
        std::cout << "[Model]   token_embd.weight: loaded (dequantized)" << std::endl;

        // Output norm: 小张量, 反量化
        weights_.output_norm = Tensor::from_gguf(parser_, "output_norm.weight", true);
        std::cout << "[Model]   output_norm.weight: loaded" << std::endl;

        // Output (LM head): 大矩阵, 保持量化
        const GGUFTensorInfo* output_tensor = parser_.find_tensor("output.weight");
        if (output_tensor) {
            weights_.output = Tensor::from_gguf(parser_, "output.weight");
            std::cout << "[Model]   output.weight: loaded (quantized)" << std::endl;
        } else {
            weights_.output = weights_.token_embedding;
            std::cout << "[Model]   output.weight: shared with token_embd" << std::endl;
        }

        // 逐层加载 Transformer 权重
        weights_.layers.resize(config_.num_layers);
        for (int i = 0; i < config_.num_layers; ++i) {
            std::string prefix = "blk." + std::to_string(i) + ".";
            auto& layer = weights_.layers[i];

            // Norm 权重: 小张量, 反量化
            layer.attn_norm = Tensor::from_gguf(parser_, prefix + "attn_norm.weight", true);

            // 大权重矩阵: 保持量化
            layer.wq = Tensor::from_gguf(parser_, prefix + "attn_q.weight");
            layer.wk = Tensor::from_gguf(parser_, prefix + "attn_k.weight");
            layer.wv = Tensor::from_gguf(parser_, prefix + "attn_v.weight");
            layer.wo = Tensor::from_gguf(parser_, prefix + "attn_output.weight");

            // Bias: 小张量, 反量化
            if (config_.has_attn_bias) {
                if (parser_.find_tensor(prefix + "attn_q.bias")) {
                    layer.bq = Tensor::from_gguf(parser_, prefix + "attn_q.bias", true);
                }
                if (parser_.find_tensor(prefix + "attn_k.bias")) {
                    layer.bk = Tensor::from_gguf(parser_, prefix + "attn_k.bias", true);
                }
                if (parser_.find_tensor(prefix + "attn_v.bias")) {
                    layer.bv = Tensor::from_gguf(parser_, prefix + "attn_v.bias", true);
                }
            }

            layer.ffn_norm = Tensor::from_gguf(parser_, prefix + "ffn_norm.weight", true);
            // FFN 大矩阵: 保持量化
            layer.w1 = Tensor::from_gguf(parser_, prefix + "ffn_gate.weight");
            layer.w2 = Tensor::from_gguf(parser_, prefix + "ffn_down.weight");
            layer.w3 = Tensor::from_gguf(parser_, prefix + "ffn_up.weight");

            if ((i + 1) % 8 == 0 || i == config_.num_layers - 1) {
                std::cout << "[Model]   Layer " << (i + 1) << "/" << config_.num_layers << " loaded" << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[Model] Error loading weights: " << e.what() << std::endl;
        return false;
    }

    // 5. 初始化 KV cache
    kv_cache_.init(config_);

    // 6. 分配临时 buffer
    x_buf_.resize(config_.hidden_size);
    x_buf2_.resize(config_.hidden_size);
    attn_out_.resize(config_.hidden_size);
    attn_buf_.resize(config_.num_heads * config_.context_len);
    ffn_buf1_.resize(config_.intermediate_size);
    ffn_buf2_.resize(config_.intermediate_size);
    logits_.resize(config_.vocab_size);
    q_buf_.resize(config_.num_heads * config_.head_dim);
    k_buf_.resize(config_.num_kv_heads * config_.head_dim);
    v_buf_.resize(config_.num_kv_heads * config_.head_dim);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "[Model] Model loaded successfully in " << elapsed << " ms" << std::endl;

    return true;
}

void Transformer::forward_attention(int layer, float* x, float* x_out, int32_t pos) {
    auto& lw = weights_.layers[layer];
    int dim = config_.hidden_size;
    int head_dim = config_.head_dim;
    int n_heads = config_.num_heads;
    int n_kv_heads = config_.num_kv_heads;
    int kv_dim = n_kv_heads * head_dim;
    int kv_mul = n_heads / n_kv_heads;

    // Q, K, V 投影
    mat_vec_mul_tensor(lw.wq, x, q_buf_.data(), n_heads * head_dim, dim);
    mat_vec_mul_tensor(lw.wk, x, k_buf_.data(), kv_dim, dim);
    mat_vec_mul_tensor(lw.wv, x, v_buf_.data(), kv_dim, dim);

    // 加 bias (Qwen2)
    if (config_.has_attn_bias) {
        if (!lw.bq.empty()) vec_add_inplace(q_buf_.data(), lw.bq.data(), n_heads * head_dim);
        if (!lw.bk.empty()) vec_add_inplace(k_buf_.data(), lw.bk.data(), kv_dim);
        if (!lw.bv.empty()) vec_add_inplace(v_buf_.data(), lw.bv.data(), kv_dim);
    }

    // RoPE: 先对所有 KV heads 旋转, 再对 Q heads 旋转
    for (int h = 0; h < n_kv_heads; ++h) {
        float* k_head = k_buf_.data() + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(config_.rope_theta, (float)i / head_dim);
            float val = pos * freq;
            float cos_val = cosf(val);
            float sin_val = sinf(val);
            float k0 = k_head[i], k1 = k_head[i + 1];
            k_head[i]     = k0 * cos_val - k1 * sin_val;
            k_head[i + 1] = k0 * sin_val + k1 * cos_val;
        }
    }
    for (int h = 0; h < n_heads; ++h) {
        float* q_head = q_buf_.data() + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(config_.rope_theta, (float)i / head_dim);
            float val = pos * freq;
            float cos_val = cosf(val);
            float sin_val = sinf(val);
            float q0 = q_head[i], q1 = q_head[i + 1];
            q_head[i]     = q0 * cos_val - q1 * sin_val;
            q_head[i + 1] = q0 * sin_val + q1 * cos_val;
        }
    }

    // 将 K, V 写入 cache
    memcpy(kv_cache_.key_cache[layer].row_data(pos), k_buf_.data(), kv_dim * sizeof(float));
    memcpy(kv_cache_.value_cache[layer].row_data(pos), v_buf_.data(), kv_dim * sizeof(float));

    // Attention
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    for (int h = 0; h < n_heads; ++h) {
        float* q_head = q_buf_.data() + h * head_dim;
        float* attn_scores = attn_buf_.data() + h * config_.context_len;
        int kv_h = h / kv_mul;

        // Q @ K^T
        for (int t = 0; t <= pos; ++t) {
            const float* k_t = kv_cache_.key_cache[layer].row_data(t) + kv_h * head_dim;
#ifdef USE_ACCELERATE
            float dot;
            vDSP_dotpr(q_head, 1, k_t, 1, &dot, (vDSP_Length)head_dim);
            attn_scores[t] = dot * scale;
#else
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) score += q_head[d] * k_t[d];
            attn_scores[t] = score * scale;
#endif
        }

        softmax(attn_scores, pos + 1);

        // Weighted sum of values
        float* out_head = x_out + h * head_dim;
        memset(out_head, 0, head_dim * sizeof(float));
        for (int t = 0; t <= pos; ++t) {
            const float* v_t = kv_cache_.value_cache[layer].row_data(t) + kv_h * head_dim;
            float w = attn_scores[t];
#ifdef USE_ACCELERATE
            // out_head += w * v_t
            vDSP_vsma(v_t, 1, &w, out_head, 1, out_head, 1, (vDSP_Length)head_dim);
#else
            for (int d = 0; d < head_dim; ++d) out_head[d] += w * v_t[d];
#endif
        }
    }
}

void Transformer::forward_ffn(int layer, float* x, float* x_out) {
    auto& lw = weights_.layers[layer];
    int dim = config_.hidden_size;
    int hidden_dim = config_.intermediate_size;

    // SwiGLU FFN: W1/W3 合并并行, 共享 x 输入
    float* buf1 = ffn_buf1_.data();
    float* buf2 = ffn_buf2_.data();

    // 用线程池同时计算 W1 和 W3 的各一半行
    ThreadPool::instance().parallel_for(hidden_dim * 2, [&](int64_t start, int64_t end) {
        for (int64_t idx = start; idx < end; ++idx) {
            if (idx < hidden_dim) {
                // W1 的第 idx 行
                int64_t row = idx;
                buf1[row] = vec_dot_quant(lw.w1.quant_row_data(row), x, dim, lw.w1.type());
            } else {
                // W3 的第 (idx - hidden_dim) 行
                int64_t row = idx - hidden_dim;
                buf2[row] = vec_dot_quant(lw.w3.quant_row_data(row), x, dim, lw.w3.type());
            }
        }
    });

    // SiLU(gate) * up
    for (int i = 0; i < hidden_dim; ++i) {
        buf1[i] = (buf1[i] / (1.0f + expf(-buf1[i]))) * buf2[i];
    }

    // W2 @ (gate * up) -> x_out
    mat_vec_mul_tensor(lw.w2, buf1, x_out, dim, hidden_dim);
}

void Transformer::forward_layer(int layer, float* x, int32_t pos) {
    auto& lw = weights_.layers[layer];
    int dim = config_.hidden_size;

    rmsnorm(x_buf_.data(), x, lw.attn_norm.data(), dim, config_.rms_norm_eps);
    forward_attention(layer, x_buf_.data(), x_buf2_.data(), pos);
    mat_vec_mul_tensor(weights_.layers[layer].wo, x_buf2_.data(), attn_out_.data(), dim, dim);
    vec_add_inplace(x, attn_out_.data(), dim);

    rmsnorm(x_buf_.data(), x, lw.ffn_norm.data(), dim, config_.rms_norm_eps);
    forward_ffn(layer, x_buf_.data(), x_buf2_.data());
    vec_add_inplace(x, x_buf2_.data(), dim);
}

std::vector<float> Transformer::forward(int32_t token, int32_t pos) {
    int dim = config_.hidden_size;

    if (x_embed_.size() != (size_t)dim) x_embed_.resize(dim);
    memcpy(x_embed_.data(), weights_.token_embedding.row_data(token), dim * sizeof(float));

    for (int l = 0; l < config_.num_layers; ++l) {
        forward_layer(l, x_embed_.data(), pos);
    }

    rmsnorm(x_embed_.data(), x_embed_.data(), weights_.output_norm.data(), dim, config_.rms_norm_eps);
    logits_.resize(config_.vocab_size);
    mat_vec_mul_tensor(weights_.output, x_embed_.data(), logits_.data(), config_.vocab_size, dim);

    return logits_;
}

void Transformer::reset() {
    kv_cache_.clear();
}

// ======================== 采样 ========================

static int32_t sample_top_p(const std::vector<float>& logits, float temperature, float top_p) {
    int n = static_cast<int>(logits.size());

    if (temperature <= 0.0f) {
        // Greedy
        return static_cast<int32_t>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    // 应用 temperature
    std::vector<float> probs(n);
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        probs[i] = expf((logits[i] - max_logit) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < n; ++i) {
        probs[i] /= sum;
    }

    // Top-p (nucleus) sampling
    std::vector<std::pair<float, int>> prob_idx(n);
    for (int i = 0; i < n; ++i) {
        prob_idx[i] = {probs[i], i};
    }
    std::sort(prob_idx.begin(), prob_idx.end(), std::greater<>());

    float cumulative = 0.0f;
    int cutoff = n;
    for (int i = 0; i < n; ++i) {
        cumulative += prob_idx[i].first;
        if (cumulative >= top_p) {
            cutoff = i + 1;
            break;
        }
    }

    // 重新归一化
    float renorm_sum = 0.0f;
    for (int i = 0; i < cutoff; ++i) {
        renorm_sum += prob_idx[i].first;
    }

    // 随机采样
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, renorm_sum);
    float r = dist(rng);

    float acc = 0.0f;
    for (int i = 0; i < cutoff; ++i) {
        acc += prob_idx[i].first;
        if (acc >= r) {
            return prob_idx[i].second;
        }
    }

    return prob_idx[0].second;
}

std::string Transformer::generate(const std::string& prompt,
                                   int max_tokens,
                                   float temperature,
                                   float top_p,
                                   TokenCallback callback) {
    // 构建 prompt tokens
    std::vector<int32_t> prompt_tokens;

    // Qwen2 chat template: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    if (config_.architecture == "qwen2") {
        // 查找特殊 token IDs
        auto find_token = [&](const std::string& text) -> int32_t {
            // 尝试直接在词表中查找
            for (int32_t i = 0; i < tokenizer_.vocab_size(); ++i) {
                if (tokenizer_.get_token_text(i) == text) return i;
            }
            return -1;
        };

        int32_t im_start = find_token("<|im_start|>");
        int32_t im_end = find_token("<|im_end|>");
        int32_t nl_token = find_token("\n");
        if (nl_token < 0) {
            // GPT2 tokenizer: '\n' 可能编码为 'Ċ'
            nl_token = find_token("Ċ");
        }

        if (im_start >= 0 && im_end >= 0) {
            // system turn
            prompt_tokens.push_back(im_start);
            auto sys_tokens = tokenizer_.encode("system", false);
            prompt_tokens.insert(prompt_tokens.end(), sys_tokens.begin(), sys_tokens.end());
            if (nl_token >= 0) prompt_tokens.push_back(nl_token);
            auto sys_msg = tokenizer_.encode("You are a helpful assistant.", false);
            prompt_tokens.insert(prompt_tokens.end(), sys_msg.begin(), sys_msg.end());
            prompt_tokens.push_back(im_end);
            if (nl_token >= 0) prompt_tokens.push_back(nl_token);

            // user turn
            prompt_tokens.push_back(im_start);
            auto user_tokens = tokenizer_.encode("user", false);
            prompt_tokens.insert(prompt_tokens.end(), user_tokens.begin(), user_tokens.end());
            if (nl_token >= 0) prompt_tokens.push_back(nl_token);
            auto user_msg = tokenizer_.encode(prompt, false);
            prompt_tokens.insert(prompt_tokens.end(), user_msg.begin(), user_msg.end());
            prompt_tokens.push_back(im_end);
            if (nl_token >= 0) prompt_tokens.push_back(nl_token);

            // assistant turn start
            prompt_tokens.push_back(im_start);
            auto asst_tokens = tokenizer_.encode("assistant", false);
            prompt_tokens.insert(prompt_tokens.end(), asst_tokens.begin(), asst_tokens.end());
            if (nl_token >= 0) prompt_tokens.push_back(nl_token);
        } else {
            // fallback: 无 chat template
            prompt_tokens = tokenizer_.encode(prompt, true);
        }
    } else {
        prompt_tokens = tokenizer_.encode(prompt, true);
    }

    std::cout << "[Generate] Prompt tokens: " << prompt_tokens.size() << std::endl;

    std::string generated_text;
    std::vector<int32_t> output_tokens;

    auto start = std::chrono::high_resolution_clock::now();
    int total_tokens = 0;

    // Prefill: 处理所有 prompt tokens
    int32_t pos = 0;
    std::vector<float> logits;

    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        logits = forward(prompt_tokens[i], pos);
        pos++;
        total_tokens++;
    }

    // Decode: 自回归生成
    for (int i = 0; i < max_tokens; ++i) {
        int32_t next_token = sample_top_p(logits, temperature, top_p);

        // 检查是否结束
        if (tokenizer_.is_eos(next_token)) {
            break;
        }

        output_tokens.push_back(next_token);
        std::string token_text = tokenizer_.decode(next_token);
        generated_text += token_text;

        // 回调
        if (callback) {
            if (!callback(next_token, token_text)) {
                break; // 用户取消
            }
        }

        // 前向传播下一个 token
        logits = forward(next_token, pos);
        pos++;
        total_tokens++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    float tokens_per_sec = (elapsed_ms > 0) ? (total_tokens * 1000.0f / elapsed_ms) : 0;

    std::cout << "\n[Generate] " << output_tokens.size() << " tokens generated"
              << " (" << tokens_per_sec << " tok/s)"
              << " in " << elapsed_ms << " ms" << std::endl;

    return generated_text;
}

} // namespace localllm
