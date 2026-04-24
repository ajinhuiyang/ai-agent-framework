#pragma once

#include "platform.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace localllm {

// GGUF 文件格式常量
constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF" as little-endian uint32
constexpr uint32_t GGUF_VERSION_3 = 3;

// GGUF 元数据值类型
enum class GGUFMetadataValueType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

// GGML 张量类型 (量化格式)
enum class GGMLType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // Q4_2 deprecated
    // Q4_3 deprecated
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    COUNT,
};

// 每种量化类型的块大小和类型大小
struct GGMLTypeTraits {
    size_t block_size;   // 每个量化块包含的元素数
    size_t type_size;    // 每个量化块的字节数
    bool is_quantized;
};

GGMLTypeTraits get_type_traits(GGMLType type);

// GGUF 字符串
struct GGUFString {
    uint64_t len;
    std::string data;
};

// GGUF 元数据值 (简化版, 用 variant 更好但为了兼容性用 union)
struct GGUFMetadataValue {
    GGUFMetadataValueType type;

    uint8_t val_uint8;
    int8_t val_int8;
    uint16_t val_uint16;
    int16_t val_int16;
    uint32_t val_uint32;
    int32_t val_int32;
    float val_float32;
    bool val_bool;
    std::string val_string;
    uint64_t val_uint64;
    int64_t val_int64;
    double val_float64;

    // 数组
    GGUFMetadataValueType arr_type;
    std::vector<GGUFMetadataValue> arr_values;

    GGUFMetadataValue() : type(GGUFMetadataValueType::UINT32) {}
};

// GGUF 张量信息
struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dimensions;
    std::vector<uint64_t> dimensions;
    GGMLType type;
    uint64_t offset; // 相对于数据段起始的偏移

    // 计算张量元素总数
    uint64_t num_elements() const {
        uint64_t n = 1;
        for (auto d : dimensions) n *= d;
        return n;
    }

    // 计算张量字节大小
    size_t byte_size() const {
        auto traits = get_type_traits(type);
        uint64_t ne = num_elements();
        return (ne / traits.block_size) * traits.type_size;
    }
};

// GGUF 文件头
struct GGUFHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
};

// GGUF 文件解析器
class GGUFParser {
public:
    GGUFParser() = default;
    ~GGUFParser();

    // 解析 GGUF 文件
    bool load(const std::string& filepath);

    // 获取元数据
    const std::map<std::string, GGUFMetadataValue>& metadata() const { return metadata_; }
    bool has_metadata(const std::string& key) const;
    const GGUFMetadataValue& get_metadata(const std::string& key) const;

    // 便捷方法: 获取不同类型的元数据值
    std::string get_string(const std::string& key, const std::string& default_val = "") const;
    uint32_t get_uint32(const std::string& key, uint32_t default_val = 0) const;
    int32_t get_int32(const std::string& key, int32_t default_val = 0) const;
    uint64_t get_uint64(const std::string& key, uint64_t default_val = 0) const;
    float get_float32(const std::string& key, float default_val = 0.0f) const;
    bool get_bool(const std::string& key, bool default_val = false) const;
    std::vector<std::string> get_string_array(const std::string& key) const;
    std::vector<float> get_float_array(const std::string& key) const;

    // 获取张量信息
    const std::vector<GGUFTensorInfo>& tensors() const { return tensors_; }
    const GGUFTensorInfo* find_tensor(const std::string& name) const;

    // 获取张量数据指针 (mmap 后的原始数据)
    const void* get_tensor_data(const GGUFTensorInfo& info) const;

    // 文件信息
    const GGUFHeader& header() const { return header_; }
    size_t file_size() const { return file_size_; }
    const std::string& filepath() const { return filepath_; }

private:
    // 读取辅助函数
    template<typename T>
    T read_val(FILE* f);
    std::string read_string(FILE* f);
    GGUFMetadataValue read_metadata_value(FILE* f, GGUFMetadataValueType type);

    GGUFHeader header_{};
    std::map<std::string, GGUFMetadataValue> metadata_;
    std::vector<GGUFTensorInfo> tensors_;

    // 内存映射 (跨平台)
    platform::MappedFile mapped_file_;
    uint8_t* data_start_ = nullptr; // 张量数据段起始地址

    std::string filepath_;
    size_t file_size_ = 0;
};

} // namespace localllm
