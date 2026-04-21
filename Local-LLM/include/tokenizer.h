#pragma once

#include "gguf_parser.h"
#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace localllm {

// BPE Tokenizer: 从 GGUF 元数据加载词表
class Tokenizer {
public:
    Tokenizer() = default;

    // 从 GGUF 解析器加载词表
    bool load_from_gguf(const GGUFParser& parser);

    // 编码: 文本 -> token IDs
    std::vector<int32_t> encode(const std::string& text, bool add_bos = true) const;

    // 解码: token ID -> 文本
    std::string decode(int32_t token_id) const;

    // 解码: token IDs -> 文本
    std::string decode(const std::vector<int32_t>& tokens) const;

    // 词表大小
    int32_t vocab_size() const { return static_cast<int32_t>(id_to_token_.size()); }

    // 特殊 token
    int32_t bos_token() const { return bos_id_; }
    int32_t eos_token() const { return eos_id_; }
    int32_t pad_token() const { return pad_id_; }
    int32_t unk_token() const { return unk_id_; }

    bool is_eos(int32_t token_id) const {
        return token_id == eos_id_ || token_id == eot_id_ || token_id == im_end_id_;
    }

    // token 文本
    const std::string& get_token_text(int32_t id) const { return id_to_token_[id]; }

private:
    // BPE 合并
    struct BPEMerge {
        std::string left;
        std::string right;
        int rank;
    };

    // BPE 编码实现
    std::vector<int32_t> bpe_encode(const std::string& text) const;

    // 词表
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<float> token_scores_;
    std::vector<int32_t> token_types_; // 0=normal, 1=unknown, 2=control, 3=user_defined, ...

    // BPE 合并规则
    std::map<std::pair<std::string, std::string>, int> merges_;

    // 特殊 token IDs
    int32_t bos_id_ = 1;
    int32_t eos_id_ = 2;
    int32_t pad_id_ = -1;
    int32_t unk_id_ = 0;
    int32_t eot_id_ = -1; // end of turn (for chat models)
    int32_t im_end_id_ = -1; // <|im_end|> (Qwen2)

    // tokenizer 类型
    std::string model_type_; // "llama", "gpt2", etc.
};

} // namespace localllm
