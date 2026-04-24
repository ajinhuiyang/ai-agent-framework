#pragma once

// Metal GPU compute context for Local-LLM
// Provides C++ interface for GPU-accelerated matrix-vector multiplication.
// Auto-detects GPU availability at runtime.

#include "gguf_parser.h"
#include <cstdint>
#include <string>

namespace localllm {

class MetalContext {
public:
    static MetalContext& instance();

    // 是否可用 (GPU 检测成功 + shader 加载成功)
    bool available() const;

    // GPU 信息
    std::string device_name() const;

    // 量化矩阵-向量乘法: y[m] = W[m,n] @ x[n]
    // W_data: 量化权重原始数据指针
    // x: float 输入向量 (长度 n)
    // y: float 输出向量 (长度 m)
    // 返回 true 表示 GPU 完成, false 表示需要 fallback 到 CPU
    bool mat_vec_mul(const void* W_data, const float* x, float* y,
                     int64_t m, int64_t n, GGMLType type);

    // float 矩阵-向量乘法
    bool mat_vec_mul_f32(const float* W, const float* x, float* y,
                         int64_t m, int64_t n);

    // 为经常使用的 buffer 预分配 GPU 内存
    void preallocate(int64_t max_rows, int64_t max_cols);

private:
    MetalContext();
    ~MetalContext();
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace localllm
