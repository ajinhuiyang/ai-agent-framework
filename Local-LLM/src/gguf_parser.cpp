#include "gguf_parser.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace localllm {

// ======================== GGMLType 特性表 ========================

GGMLTypeTraits get_type_traits(GGMLType type) {
    // block_size: 每块元素个数, type_size: 每块字节数
    switch (type) {
        case GGMLType::F32:    return {1, 4, false};
        case GGMLType::F16:    return {1, 2, false};
        case GGMLType::Q4_0:   return {32, 18, true};   // 32 个 4-bit + 1 个 f16 scale = 16+2
        case GGMLType::Q4_1:   return {32, 20, true};   // 32 个 4-bit + f16 scale + f16 min = 16+2+2
        case GGMLType::Q5_0:   return {32, 22, true};
        case GGMLType::Q5_1:   return {32, 24, true};
        case GGMLType::Q8_0:   return {32, 34, true};   // 32 个 int8 + 1 个 f16 scale = 32+2
        case GGMLType::Q8_1:   return {32, 36, true};
        case GGMLType::Q2_K:   return {256, 84, true};
        case GGMLType::Q3_K:   return {256, 110, true};
        case GGMLType::Q4_K:   return {256, 144, true};
        case GGMLType::Q5_K:   return {256, 176, true};
        case GGMLType::Q6_K:   return {256, 210, true};
        case GGMLType::Q8_K:   return {256, 292, true};
        case GGMLType::I8:     return {1, 1, false};
        case GGMLType::I16:    return {1, 2, false};
        case GGMLType::I32:    return {1, 4, false};
        case GGMLType::I64:    return {1, 8, false};
        case GGMLType::F64:    return {1, 8, false};
        default:
            throw std::runtime_error("Unsupported GGML type: " + std::to_string(static_cast<int>(type)));
    }
}

// ======================== GGUFParser 实现 ========================

GGUFParser::~GGUFParser() {
    if (mmap_addr_ && mmap_size_ > 0) {
        munmap(mmap_addr_, mmap_size_);
        mmap_addr_ = nullptr;
        mmap_size_ = 0;
    }
}

template<typename T>
T GGUFParser::read_val(FILE* f) {
    T val;
    if (fread(&val, sizeof(T), 1, f) != 1) {
        throw std::runtime_error("Failed to read value from GGUF file");
    }
    return val;
}

std::string GGUFParser::read_string(FILE* f) {
    uint64_t len = read_val<uint64_t>(f);
    if (len > 1024 * 1024) { // 安全上限 1MB
        throw std::runtime_error("String too long in GGUF file: " + std::to_string(len));
    }
    std::string s(len, '\0');
    if (len > 0 && fread(&s[0], 1, len, f) != len) {
        throw std::runtime_error("Failed to read string from GGUF file");
    }
    return s;
}

GGUFMetadataValue GGUFParser::read_metadata_value(FILE* f, GGUFMetadataValueType type) {
    GGUFMetadataValue val;
    val.type = type;

    switch (type) {
        case GGUFMetadataValueType::UINT8:
            val.val_uint8 = read_val<uint8_t>(f);
            break;
        case GGUFMetadataValueType::INT8:
            val.val_int8 = read_val<int8_t>(f);
            break;
        case GGUFMetadataValueType::UINT16:
            val.val_uint16 = read_val<uint16_t>(f);
            break;
        case GGUFMetadataValueType::INT16:
            val.val_int16 = read_val<int16_t>(f);
            break;
        case GGUFMetadataValueType::UINT32:
            val.val_uint32 = read_val<uint32_t>(f);
            break;
        case GGUFMetadataValueType::INT32:
            val.val_int32 = read_val<int32_t>(f);
            break;
        case GGUFMetadataValueType::FLOAT32:
            val.val_float32 = read_val<float>(f);
            break;
        case GGUFMetadataValueType::BOOL:
            val.val_bool = read_val<uint8_t>(f) != 0;
            break;
        case GGUFMetadataValueType::STRING:
            val.val_string = read_string(f);
            break;
        case GGUFMetadataValueType::UINT64:
            val.val_uint64 = read_val<uint64_t>(f);
            break;
        case GGUFMetadataValueType::INT64:
            val.val_int64 = read_val<int64_t>(f);
            break;
        case GGUFMetadataValueType::FLOAT64:
            val.val_float64 = read_val<double>(f);
            break;
        case GGUFMetadataValueType::ARRAY: {
            val.arr_type = static_cast<GGUFMetadataValueType>(read_val<uint32_t>(f));
            uint64_t arr_len = read_val<uint64_t>(f);
            val.arr_values.reserve(arr_len);
            for (uint64_t i = 0; i < arr_len; ++i) {
                val.arr_values.push_back(read_metadata_value(f, val.arr_type));
            }
            break;
        }
        default:
            throw std::runtime_error("Unknown GGUF metadata type: " + std::to_string(static_cast<int>(type)));
    }
    return val;
}

bool GGUFParser::load(const std::string& filepath) {
    filepath_ = filepath;

    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        std::cerr << "[GGUF] Failed to open file: " << filepath << std::endl;
        return false;
    }

    // 获取文件大小
    fseek(f, 0, SEEK_END);
    file_size_ = ftell(f);
    fseek(f, 0, SEEK_SET);

    try {
        // 1. 读取头部
        header_.magic = read_val<uint32_t>(f);
        if (header_.magic != GGUF_MAGIC) {
            std::cerr << "[GGUF] Invalid magic number: 0x" << std::hex << header_.magic
                      << " (expected 0x" << GGUF_MAGIC << ")" << std::dec << std::endl;
            fclose(f);
            return false;
        }

        header_.version = read_val<uint32_t>(f);
        if (header_.version < 2 || header_.version > 3) {
            std::cerr << "[GGUF] Unsupported version: " << header_.version << std::endl;
            fclose(f);
            return false;
        }

        header_.tensor_count = read_val<uint64_t>(f);
        header_.metadata_kv_count = read_val<uint64_t>(f);

        std::cout << "[GGUF] Version: " << header_.version
                  << ", Tensors: " << header_.tensor_count
                  << ", Metadata KV: " << header_.metadata_kv_count << std::endl;

        // 2. 读取元数据键值对
        for (uint64_t i = 0; i < header_.metadata_kv_count; ++i) {
            std::string key = read_string(f);
            auto vtype = static_cast<GGUFMetadataValueType>(read_val<uint32_t>(f));
            GGUFMetadataValue value = read_metadata_value(f, vtype);
            metadata_[key] = std::move(value);
        }

        // 3. 读取张量信息
        tensors_.resize(header_.tensor_count);
        for (uint64_t i = 0; i < header_.tensor_count; ++i) {
            auto& ti = tensors_[i];
            ti.name = read_string(f);
            ti.n_dimensions = read_val<uint32_t>(f);
            ti.dimensions.resize(ti.n_dimensions);
            for (uint32_t d = 0; d < ti.n_dimensions; ++d) {
                ti.dimensions[d] = read_val<uint64_t>(f);
            }
            ti.type = static_cast<GGMLType>(read_val<uint32_t>(f));
            ti.offset = read_val<uint64_t>(f);
        }

        // 4. 计算数据段起始位置 (对齐到 32 字节)
        long current_pos = ftell(f);
        size_t alignment = 32; // GGUF v3 默认对齐

        // 检查是否有自定义对齐
        if (has_metadata("general.alignment")) {
            alignment = get_uint32("general.alignment", 32);
        }

        size_t data_offset = ((current_pos + alignment - 1) / alignment) * alignment;

        fclose(f);

        // 5. 使用 mmap 映射文件
        int fd = open(filepath.c_str(), O_RDONLY);
        if (fd < 0) {
            std::cerr << "[GGUF] Failed to open file for mmap: " << filepath << std::endl;
            return false;
        }

        mmap_size_ = file_size_;
        mmap_addr_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);

        if (mmap_addr_ == MAP_FAILED) {
            std::cerr << "[GGUF] mmap failed" << std::endl;
            mmap_addr_ = nullptr;
            return false;
        }

        // madvise 建议内核顺序读取
        madvise(mmap_addr_, mmap_size_, MADV_SEQUENTIAL);

        data_start_ = static_cast<uint8_t*>(mmap_addr_) + data_offset;

        std::cout << "[GGUF] File loaded successfully. Data offset: " << data_offset
                  << ", File size: " << file_size_ << std::endl;

        // 打印一些关键元数据
        if (has_metadata("general.architecture")) {
            std::cout << "[GGUF] Architecture: " << get_string("general.architecture") << std::endl;
        }
        if (has_metadata("general.name")) {
            std::cout << "[GGUF] Model name: " << get_string("general.name") << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "[GGUF] Error parsing file: " << e.what() << std::endl;
        fclose(f);
        return false;
    }

    return true;
}

bool GGUFParser::has_metadata(const std::string& key) const {
    return metadata_.find(key) != metadata_.end();
}

const GGUFMetadataValue& GGUFParser::get_metadata(const std::string& key) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) {
        throw std::runtime_error("Metadata key not found: " + key);
    }
    return it->second;
}

std::string GGUFParser::get_string(const std::string& key, const std::string& default_val) const {
    if (!has_metadata(key)) return default_val;
    const auto& val = get_metadata(key);
    if (val.type != GGUFMetadataValueType::STRING) return default_val;
    return val.val_string;
}

uint32_t GGUFParser::get_uint32(const std::string& key, uint32_t default_val) const {
    if (!has_metadata(key)) return default_val;
    const auto& val = get_metadata(key);
    switch (val.type) {
        case GGUFMetadataValueType::UINT32: return val.val_uint32;
        case GGUFMetadataValueType::INT32:  return static_cast<uint32_t>(val.val_int32);
        case GGUFMetadataValueType::UINT16: return val.val_uint16;
        case GGUFMetadataValueType::UINT8:  return val.val_uint8;
        default: return default_val;
    }
}

int32_t GGUFParser::get_int32(const std::string& key, int32_t default_val) const {
    if (!has_metadata(key)) return default_val;
    const auto& val = get_metadata(key);
    switch (val.type) {
        case GGUFMetadataValueType::INT32:  return val.val_int32;
        case GGUFMetadataValueType::UINT32: return static_cast<int32_t>(val.val_uint32);
        default: return default_val;
    }
}

uint64_t GGUFParser::get_uint64(const std::string& key, uint64_t default_val) const {
    if (!has_metadata(key)) return default_val;
    const auto& val = get_metadata(key);
    switch (val.type) {
        case GGUFMetadataValueType::UINT64: return val.val_uint64;
        case GGUFMetadataValueType::UINT32: return val.val_uint32;
        case GGUFMetadataValueType::INT64:  return static_cast<uint64_t>(val.val_int64);
        default: return default_val;
    }
}

float GGUFParser::get_float32(const std::string& key, float default_val) const {
    if (!has_metadata(key)) return default_val;
    const auto& val = get_metadata(key);
    if (val.type != GGUFMetadataValueType::FLOAT32) return default_val;
    return val.val_float32;
}

bool GGUFParser::get_bool(const std::string& key, bool default_val) const {
    if (!has_metadata(key)) return default_val;
    const auto& val = get_metadata(key);
    if (val.type != GGUFMetadataValueType::BOOL) return default_val;
    return val.val_bool;
}

std::vector<std::string> GGUFParser::get_string_array(const std::string& key) const {
    std::vector<std::string> result;
    if (!has_metadata(key)) return result;
    const auto& val = get_metadata(key);
    if (val.type != GGUFMetadataValueType::ARRAY) return result;
    for (const auto& v : val.arr_values) {
        if (v.type == GGUFMetadataValueType::STRING) {
            result.push_back(v.val_string);
        }
    }
    return result;
}

std::vector<float> GGUFParser::get_float_array(const std::string& key) const {
    std::vector<float> result;
    if (!has_metadata(key)) return result;
    const auto& val = get_metadata(key);
    if (val.type != GGUFMetadataValueType::ARRAY) return result;
    for (const auto& v : val.arr_values) {
        if (v.type == GGUFMetadataValueType::FLOAT32) {
            result.push_back(v.val_float32);
        }
    }
    return result;
}

const GGUFTensorInfo* GGUFParser::find_tensor(const std::string& name) const {
    for (const auto& t : tensors_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

const void* GGUFParser::get_tensor_data(const GGUFTensorInfo& info) const {
    if (!data_start_) return nullptr;
    return data_start_ + info.offset;
}

} // namespace localllm
