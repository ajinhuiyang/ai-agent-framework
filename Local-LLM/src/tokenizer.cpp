#include "tokenizer.h"

#include <cassert>
#include <codecvt>
#include <iostream>
#include <locale>
#include <sstream>

namespace localllm {

bool Tokenizer::load_from_gguf(const GGUFParser& parser) {
    // 获取 tokenizer 模型类型
    model_type_ = parser.get_string("tokenizer.ggml.model", "llama");
    std::cout << "[Tokenizer] Model type: " << model_type_ << std::endl;

    // 加载词表 tokens
    if (!parser.has_metadata("tokenizer.ggml.tokens")) {
        std::cerr << "[Tokenizer] No tokens found in GGUF metadata" << std::endl;
        return false;
    }

    auto tokens = parser.get_string_array("tokenizer.ggml.tokens");
    if (tokens.empty()) {
        std::cerr << "[Tokenizer] Empty token list" << std::endl;
        return false;
    }

    id_to_token_ = std::move(tokens);
    for (int32_t i = 0; i < static_cast<int32_t>(id_to_token_.size()); ++i) {
        token_to_id_[id_to_token_[i]] = i;
    }

    std::cout << "[Tokenizer] Loaded " << id_to_token_.size() << " tokens" << std::endl;

    // 加载 token scores (用于 SentencePiece / Unigram tokenizer)
    if (parser.has_metadata("tokenizer.ggml.scores")) {
        token_scores_ = parser.get_float_array("tokenizer.ggml.scores");
    }

    // 加载 token types
    if (parser.has_metadata("tokenizer.ggml.token_type")) {
        const auto& val = parser.get_metadata("tokenizer.ggml.token_type");
        if (val.type == GGUFMetadataValueType::ARRAY) {
            for (const auto& v : val.arr_values) {
                token_types_.push_back(static_cast<int32_t>(v.val_int32));
            }
        }
    }

    // 加载 BPE merges (如果存在)
    if (parser.has_metadata("tokenizer.ggml.merges")) {
        auto merge_strs = parser.get_string_array("tokenizer.ggml.merges");
        for (int i = 0; i < static_cast<int>(merge_strs.size()); ++i) {
            const auto& m = merge_strs[i];
            auto space_pos = m.find(' ');
            if (space_pos != std::string::npos) {
                std::string left = m.substr(0, space_pos);
                std::string right = m.substr(space_pos + 1);
                merges_[{left, right}] = i;
            }
        }
        std::cout << "[Tokenizer] Loaded " << merges_.size() << " BPE merges" << std::endl;
    }

    // 加载特殊 token IDs
    bos_id_ = parser.get_uint32("tokenizer.ggml.bos_token_id", 1);
    eos_id_ = parser.get_uint32("tokenizer.ggml.eos_token_id", 2);
    pad_id_ = parser.get_uint32("tokenizer.ggml.padding_token_id", 0);
    unk_id_ = parser.get_uint32("tokenizer.ggml.unknown_token_id", 0);

    // 查找 eot token
    auto it = token_to_id_.find("<|eot_id|>");
    if (it != token_to_id_.end()) {
        eot_id_ = it->second;
    } else {
        it = token_to_id_.find("<|end_of_turn|>");
        if (it != token_to_id_.end()) {
            eot_id_ = it->second;
        }
    }
    // Qwen2: <|im_end|> 也是停止 token
    it = token_to_id_.find("<|im_end|>");
    if (it != token_to_id_.end()) {
        im_end_id_ = it->second;
    }

    std::cout << "[Tokenizer] BOS: " << bos_id_ << ", EOS: " << eos_id_ << std::endl;

    return true;
}

// SentencePiece 风格的 BPE 编码
std::vector<int32_t> Tokenizer::bpe_encode(const std::string& text) const {
    if (text.empty()) return {};

    // 如果是 SentencePiece 模型 (llama 类型), 使用 score-based greedy
    if (model_type_ == "llama" && !token_scores_.empty()) {
        // SentencePiece unigram 方法: 贪心最长匹配
        std::vector<int32_t> tokens;
        size_t i = 0;

        while (i < text.size()) {
            int32_t best_id = unk_id_;
            size_t best_len = 0;
            float best_score = -1e30f;

            // 尝试匹配最长的 token
            for (size_t len = 1; len <= text.size() - i && len <= 128; ++len) {
                std::string sub = text.substr(i, len);
                auto it = token_to_id_.find(sub);
                if (it != token_to_id_.end()) {
                    int32_t id = it->second;
                    float score = (id < static_cast<int32_t>(token_scores_.size()))
                                  ? token_scores_[id] : 0.0f;
                    if (len > best_len || (len == best_len && score > best_score)) {
                        best_id = id;
                        best_len = len;
                        best_score = score;
                    }
                }
            }

            if (best_len == 0) {
                // 无法匹配, 跳过一个字节
                best_len = 1;
                // 尝试查找字节级别的 fallback token
                unsigned char c = text[i];
                char hex_token[16];
                snprintf(hex_token, sizeof(hex_token), "<0x%02X>", c);
                auto it = token_to_id_.find(hex_token);
                if (it != token_to_id_.end()) {
                    best_id = it->second;
                } else {
                    best_id = unk_id_;
                }
            }

            tokens.push_back(best_id);
            i += best_len;
        }

        return tokens;
    }

    // GPT-2 / BPE 风格: 使用合并规则
    // 初始化: 每个字符一个 token
    std::vector<std::string> pieces;
    for (size_t i = 0; i < text.size();) {
        // UTF-8 字符长度
        unsigned char c = text[i];
        int char_len = 1;
        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;
        pieces.push_back(text.substr(i, char_len));
        i += char_len;
    }

    // 反复应用优先级最高的合并
    while (pieces.size() > 1) {
        int best_rank = INT_MAX;
        int best_pos = -1;

        for (int i = 0; i < static_cast<int>(pieces.size()) - 1; ++i) {
            auto it = merges_.find({pieces[i], pieces[i + 1]});
            if (it != merges_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = i;
            }
        }

        if (best_pos < 0) break; // 没有更多合并

        pieces[best_pos] = pieces[best_pos] + pieces[best_pos + 1];
        pieces.erase(pieces.begin() + best_pos + 1);
    }

    // 转换为 token IDs
    std::vector<int32_t> result;
    for (const auto& p : pieces) {
        auto it = token_to_id_.find(p);
        if (it != token_to_id_.end()) {
            result.push_back(it->second);
        } else {
            result.push_back(unk_id_);
        }
    }

    return result;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int32_t> tokens;

    if (add_bos && bos_id_ >= 0) {
        tokens.push_back(bos_id_);
    }

    // SentencePiece 模型需要在句首加空格
    std::string processed_text = text;
    if (model_type_ == "llama") {
        // SentencePiece 把空格替换为特殊字符 (U+2581 = \xe2\x96\x81)
        std::string sp_text;
        sp_text += "\xe2\x96\x81"; // 开头加空格标记
        for (char c : processed_text) {
            if (c == ' ') {
                sp_text += "\xe2\x96\x81";
            } else {
                sp_text += c;
            }
        }
        processed_text = sp_text;
    }

    auto encoded = bpe_encode(processed_text);
    tokens.insert(tokens.end(), encoded.begin(), encoded.end());

    return tokens;
}

std::string Tokenizer::decode(int32_t token_id) const {
    if (token_id < 0 || token_id >= static_cast<int32_t>(id_to_token_.size())) {
        return "";
    }

    std::string token = id_to_token_[token_id];

    // 跳过特殊 token
    if (token_id == bos_id_ || token_id == eos_id_ || token_id == pad_id_) {
        return "";
    }
    // 跳过 chat template 特殊 tokens
    if (token.size() > 2 && token[0] == '<' && token[1] == '|' && token.back() == '>') {
        return "";
    }

    // 处理字节级 token: <0xNN>
    if (token.size() == 6 && token[0] == '<' && token[1] == '0' && token[2] == 'x' && token[5] == '>') {
        char c = static_cast<char>(std::stoul(token.substr(3, 2), nullptr, 16));
        return std::string(1, c);
    }

    // GPT2 字节级 BPE: Unicode 字符 -> 原始字节
    // GPT2 把每个字节映射到一个 Unicode 字符:
    //   0x21-0x7E -> 保持不变 (! 到 ~)
    //   0xA1-0xAC -> 保持不变
    //   0xAE-0xFF -> 保持不变
    //   其余字节 (0x00-0x20, 0x7F-0xA0, 0xAD) -> 映射到 U+0100 开始的字符
    if (model_type_ == "gpt2") {
        std::string result;
        size_t i = 0;
        while (i < token.size()) {
            unsigned char c = token[i];

            if (c < 0x80) {
                // ASCII
                result += (char)c;
                i++;
            } else if ((c & 0xE0) == 0xC0 && i + 1 < token.size()) {
                // 2-byte UTF-8
                unsigned char c2 = token[i + 1];
                uint32_t cp = ((c & 0x1F) << 6) | (c2 & 0x3F);

                // GPT2 映射: U+0100 到 U+0143 对应字节 0x00-0x20, 0x7F-0xA0, 0xAD
                if (cp >= 0x0100 && cp <= 0x0143) {
                    // GPT2 的 bytes_to_unicode 映射的逆映射
                    static const uint8_t gpt2_byte_decoder[] = {
                        // U+0100 -> 0x00, U+0101 -> 0x01, ..., U+0120 -> 0x20
                        0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,
                        0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F,
                        0x20, // U+0120 -> space
                        0x7F, // U+0121 -> DEL
                        0x80,0x81,0x82,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8A,0x8B,0x8C,0x8D,0x8E,0x8F,
                        0x90,0x91,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0x9B,0x9C,0x9D,0x9E,0x9F,
                        0xA0, // U+0141
                        0xAD, // U+0142
                        // U+0143 也映射到什么? 实际上 GPT2 只用了 0x00-0xFF 对应的 256 个 codepoint
                    };
                    int idx = cp - 0x0100;
                    if (idx < (int)sizeof(gpt2_byte_decoder)) {
                        result += (char)gpt2_byte_decoder[idx];
                    } else {
                        result += (char)(c);
                        result += (char)(c2);
                    }
                } else {
                    // 普通 2-byte UTF-8 字符, 保持不变
                    result += (char)c;
                    result += (char)c2;
                }
                i += 2;
            } else if ((c & 0xF0) == 0xE0 && i + 2 < token.size()) {
                // 3-byte UTF-8, 保持不变
                result += token[i];
                result += token[i + 1];
                result += token[i + 2];
                i += 3;
            } else if ((c & 0xF8) == 0xF0 && i + 3 < token.size()) {
                // 4-byte UTF-8, 保持不变
                result += token[i];
                result += token[i + 1];
                result += token[i + 2];
                result += token[i + 3];
                i += 4;
            } else {
                result += (char)c;
                i++;
            }
        }
        return result;
    }

    // SentencePiece: 将 \xe2\x96\x81 替换回空格
    std::string result;
    for (size_t i = 0; i < token.size();) {
        if (i + 2 < token.size() &&
            (unsigned char)token[i] == 0xe2 &&
            (unsigned char)token[i + 1] == 0x96 &&
            (unsigned char)token[i + 2] == 0x81) {
            result += ' ';
            i += 3;
        } else {
            result += token[i];
            i++;
        }
    }

    return result;
}

std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string result;
    for (auto id : tokens) {
        result += decode(id);
    }
    return result;
}

} // namespace localllm
