#include "tokenizer.h"

#include <cassert>
#include <codecvt>
#include <iostream>
#include <locale>
#include <sstream>
#include <unordered_map>

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
    // GPT-2 的字节级 BPE: 先把每个字节映射到 Unicode 字符，再做 BPE
    static const auto& byte_map = []() -> const std::vector<std::string>& {
        static std::vector<std::string> m(256);
        static bool init = false;
        if (!init) {
            uint32_t n = 0;
            for (int b = 0; b < 256; b++) {
                uint32_t cp;
                if ((b >= 0x21 && b <= 0x7E) ||
                    (b >= 0xA1 && b <= 0xAC) ||
                    (b >= 0xAE && b <= 0xFF)) {
                    cp = static_cast<uint32_t>(b);
                } else {
                    cp = 0x0100 + n;
                    n++;
                }
                // Encode Unicode codepoint to UTF-8 string
                std::string s;
                if (cp < 0x80) {
                    s += (char)cp;
                } else if (cp < 0x800) {
                    s += (char)(0xC0 | (cp >> 6));
                    s += (char)(0x80 | (cp & 0x3F));
                } else {
                    s += (char)(0xE0 | (cp >> 12));
                    s += (char)(0x80 | ((cp >> 6) & 0x3F));
                    s += (char)(0x80 | (cp & 0x3F));
                }
                m[b] = s;
            }
            init = true;
        }
        return m;
    }();

    // 将原始字节序列转换为 GPT-2 Unicode 字符序列
    std::vector<std::string> pieces;
    for (size_t i = 0; i < text.size(); i++) {
        unsigned char b = text[i];
        pieces.push_back(byte_map[b]);
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
    // GPT2/Qwen 把每个字节(0x00-0xFF)映射到一个唯一的 Unicode 码点，
    // decode 时需要逆映射还原。
    if (model_type_ == "gpt2") {
        // 构建完整的逆映射表: Unicode 码点 -> 原始字节
        static const auto& inv_map = []() -> const std::unordered_map<uint32_t, uint8_t>& {
            static std::unordered_map<uint32_t, uint8_t> m;
            if (m.empty()) {
                // 复现 GPT-2 的 bytes_to_unicode():
                // 某些字节直接映射到自身的码点，其余映射到 U+0100 开始
                uint32_t n = 0;
                for (int b = 0; b < 256; b++) {
                    if ((b >= 0x21 && b <= 0x7E) ||
                        (b >= 0xA1 && b <= 0xAC) ||
                        (b >= 0xAE && b <= 0xFF)) {
                        m[static_cast<uint32_t>(b)] = static_cast<uint8_t>(b);
                    } else {
                        m[0x0100 + n] = static_cast<uint8_t>(b);
                        n++;
                    }
                }
            }
            return m;
        }();

        // 先解码 UTF-8 字符串为 Unicode 码点序列，再通过逆映射还原字节
        std::string result;
        size_t i = 0;
        while (i < token.size()) {
            unsigned char c = token[i];
            uint32_t cp = 0;
            int len = 0;

            if (c < 0x80) {
                cp = c; len = 1;
            } else if ((c & 0xE0) == 0xC0 && i + 1 < token.size()) {
                cp = ((c & 0x1F) << 6) | (token[i+1] & 0x3F);
                len = 2;
            } else if ((c & 0xF0) == 0xE0 && i + 2 < token.size()) {
                cp = ((c & 0x0F) << 12) | ((token[i+1] & 0x3F) << 6) | (token[i+2] & 0x3F);
                len = 3;
            } else if ((c & 0xF8) == 0xF0 && i + 3 < token.size()) {
                cp = ((c & 0x07) << 18) | ((token[i+1] & 0x3F) << 12) | ((token[i+2] & 0x3F) << 6) | (token[i+3] & 0x3F);
                len = 4;
            } else {
                result += (char)c;
                i++;
                continue;
            }

            auto it = inv_map.find(cp);
            if (it != inv_map.end()) {
                result += (char)it->second;
            } else {
                // 不在映射表中，保持原 UTF-8 编码
                for (int j = 0; j < len; j++) result += token[i + j];
            }
            i += len;
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
