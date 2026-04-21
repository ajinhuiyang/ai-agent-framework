#pragma once

#include "gguf_parser.h"
#include <cstdint>
#include <vector>

namespace localllm {

// F16 转 F32 的辅助函数
float f16_to_f32(uint16_t h);
uint16_t f32_to_f16(float f);

// ======================== 反量化函数 ========================
// 将量化数据块解码为 float32 数组

// Q4_0: 每32个元素一个块, 4-bit 量化 + fp16 scale
// 块结构: [fp16 scale (2 bytes)] [32 x 4-bit values (16 bytes)] = 18 bytes
void dequantize_q4_0(const void* src, float* dst, int64_t n_elements);

// Q4_1: 类似 Q4_0 但多了一个 fp16 min 值
// 块结构: [fp16 scale (2 bytes)] [fp16 min (2 bytes)] [32 x 4-bit values (16 bytes)] = 20 bytes
void dequantize_q4_1(const void* src, float* dst, int64_t n_elements);

// Q5_0: 5-bit 量化
void dequantize_q5_0(const void* src, float* dst, int64_t n_elements);

// Q5_1: 5-bit 量化 + min
void dequantize_q5_1(const void* src, float* dst, int64_t n_elements);

// Q8_0: 8-bit 量化, 每32个元素一个块
// 块结构: [fp16 scale (2 bytes)] [32 x int8 values (32 bytes)] = 34 bytes
void dequantize_q8_0(const void* src, float* dst, int64_t n_elements);

// F16: IEEE 754 半精度浮点
void dequantize_f16(const void* src, float* dst, int64_t n_elements);

// Q2_K: K-quant 2-bit
void dequantize_q2_k(const void* src, float* dst, int64_t n_elements);

// Q3_K: K-quant 3-bit
void dequantize_q3_k(const void* src, float* dst, int64_t n_elements);

// Q4_K: K-quant 4-bit
void dequantize_q4_k(const void* src, float* dst, int64_t n_elements);

// Q5_K: K-quant 5-bit
void dequantize_q5_k(const void* src, float* dst, int64_t n_elements);

// Q6_K: K-quant 6-bit
void dequantize_q6_k(const void* src, float* dst, int64_t n_elements);

// 通用反量化入口: 根据类型自动选择对应的反量化函数
void dequantize(const void* src, float* dst, int64_t n_elements, GGMLType type);

// 反量化整个张量到 float 数组
std::vector<float> dequantize_tensor(const void* data, int64_t n_elements, GGMLType type);

// ======================== 量化向量点积 ========================
// 直接对量化数据和 float 向量计算点积, 无需中间反量化缓冲区
// 返回 sum( dequant(src)[i] * y[i] )

float vec_dot_q4_k(const void* src, const float* y, int64_t n);
float vec_dot_q6_k(const void* src, const float* y, int64_t n);
float vec_dot_q8_0(const void* src, const float* y, int64_t n);
float vec_dot_q4_0(const void* src, const float* y, int64_t n);

// 通用量化 dot 入口
float vec_dot_quant(const void* src, const float* y, int64_t n, GGMLType type);

} // namespace localllm
