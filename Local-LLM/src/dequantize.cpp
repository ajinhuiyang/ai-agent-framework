#include "dequantize.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define USE_NEON 1
#endif

namespace localllm {

// ======================== F16 <-> F32 转换 ========================

float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x03FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            uint32_t result = sign;
            float f;
            memcpy(&f, &result, 4);
            return f;
        }
        exponent = 1;
        while (!(mantissa & 0x0400)) {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= 0x03FF;
        exponent = exponent + (127 - 15);
        uint32_t result = sign | (exponent << 23) | (mantissa << 13);
        float f;
        memcpy(&f, &result, 4);
        return f;
    } else if (exponent == 31) {
        uint32_t result = sign | 0x7F800000 | (mantissa << 13);
        float f;
        memcpy(&f, &result, 4);
        return f;
    }

    exponent = exponent + (127 - 15);
    uint32_t result = sign | (exponent << 23) | (mantissa << 13);
    float f;
    memcpy(&f, &result, 4);
    return f;
}

uint16_t f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);

    uint16_t sign = (bits >> 16) & 0x8000;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x007FFFFF;

    if (exponent <= 0) {
        if (exponent < -10) return sign;
        mantissa = (mantissa | 0x00800000) >> (1 - exponent);
        return sign | (uint16_t)(mantissa >> 13);
    } else if (exponent == 0xFF - 127 + 15) {
        if (mantissa == 0) return sign | 0x7C00;
        return sign | 0x7E00;
    }

    if (exponent > 30) return sign | 0x7C00;

    return sign | (uint16_t)(exponent << 10) | (uint16_t)(mantissa >> 13);
}

// ======================== 块结构定义 (严格按 llama.cpp) ========================

#pragma pack(push, 1)

// QK = 32
struct BlockQ4_0 {
    uint16_t d;         // fp16 scale
    uint8_t qs[16];     // 32 x 4-bit (low nibble = first 16, high nibble = last 16)
};

struct BlockQ4_1 {
    uint16_t d;         // fp16 scale
    uint16_t m;         // fp16 min
    uint8_t qs[16];
};

struct BlockQ5_0 {
    uint16_t d;
    uint8_t qh[4];      // 5th bit for 32 values
    uint8_t qs[16];
};

struct BlockQ5_1 {
    uint16_t d;
    uint16_t m;
    uint8_t qh[4];
    uint8_t qs[16];
};

struct BlockQ8_0 {
    uint16_t d;
    int8_t qs[32];
};

// K-quants: QK_K = 256

struct BlockQ2K {
    uint8_t scales[16];
    uint8_t qs[64];
    uint16_t d;
    uint16_t dmin;
};

struct BlockQ3K {
    uint8_t hmask[32];  // high bit mask
    uint8_t qs[64];     // 2-bit quants
    uint8_t scales[12]; // packed scales
    uint16_t d;         // super-block scale
};

struct BlockQ4K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct BlockQ5K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];     // high bits
    uint8_t qs[128];    // low 4 bits
};

struct BlockQ6K {
    uint8_t ql[128];    // low 4 bits of quants
    uint8_t qh[64];     // high 2 bits of quants
    int8_t scales[16];
    uint16_t d;
};

#pragma pack(pop)

// ======================== helper: get_scale_min_k4 ========================
// 严格来自 llama.cpp
static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j]     >> 6) << 4);
    }
}

// ======================== Q4_0 ========================
// 严格对照 llama.cpp dequantize_row_q4_0

void dequantize_q4_0(const void* src, float* dst, int64_t n_elements) {
    const auto* x = static_cast<const BlockQ4_0*>(src);
    const int nb = n_elements / 32;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);

        for (int j = 0; j < 16; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >> 4) - 8;

            dst[i * 32 + j + 0]  = x0 * d;
            dst[i * 32 + j + 16] = x1 * d;
        }
    }
}

// ======================== Q4_1 ========================

void dequantize_q4_1(const void* src, float* dst, int64_t n_elements) {
    const auto* x = static_cast<const BlockQ4_1*>(src);
    const int nb = n_elements / 32;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        const float m = f16_to_f32(x[i].m);

        for (int j = 0; j < 16; ++j) {
            const int x0 = x[i].qs[j] & 0x0F;
            const int x1 = x[i].qs[j] >> 4;

            dst[i * 32 + j + 0]  = x0 * d + m;
            dst[i * 32 + j + 16] = x1 * d + m;
        }
    }
}

// ======================== Q5_0 ========================
// 严格对照 llama.cpp: qh bit extraction

void dequantize_q5_0(const void* src, float* dst, int64_t n_elements) {
    const auto* x = static_cast<const BlockQ5_0*>(src);
    const int nb = n_elements / 32;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < 16; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;

            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >> 4) | xh_1) - 16;

            dst[i * 32 + j + 0]  = x0 * d;
            dst[i * 32 + j + 16] = x1 * d;
        }
    }
}

// ======================== Q5_1 ========================

void dequantize_q5_1(const void* src, float* dst, int64_t n_elements) {
    const auto* x = static_cast<const BlockQ5_1*>(src);
    const int nb = n_elements / 32;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        const float m = f16_to_f32(x[i].m);

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < 16; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;

            const int x0 = (x[i].qs[j] & 0x0F) | xh_0;
            const int x1 = (x[i].qs[j] >> 4) | xh_1;

            dst[i * 32 + j + 0]  = x0 * d + m;
            dst[i * 32 + j + 16] = x1 * d + m;
        }
    }
}

// ======================== Q8_0 ========================

void dequantize_q8_0(const void* src, float* dst, int64_t n_elements) {
    const auto* x = static_cast<const BlockQ8_0*>(src);
    const int nb = n_elements / 32;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        for (int j = 0; j < 32; ++j) {
            dst[i * 32 + j] = x[i].qs[j] * d;
        }
    }
}

// ======================== F16 ========================

void dequantize_f16(const void* src, float* dst, int64_t n_elements) {
    const auto* data = static_cast<const uint16_t*>(src);
    for (int64_t i = 0; i < n_elements; ++i) {
        dst[i] = f16_to_f32(data[i]);
    }
}

// ======================== Q2_K ========================
// 严格对照 llama.cpp dequantize_row_q2_K

void dequantize_q2_k(const void* src, float* dst, int64_t n_elements) {
    const auto* x = static_cast<const BlockQ2K*>(src);
    const int nb = n_elements / 256;

    for (int i = 0; i < nb; i++) {
        const float d   = f16_to_f32(x[i].d);
        const float min = f16_to_f32(x[i].dmin);

        const uint8_t* q = x[i].qs;
        float* y = dst + i * 256;

        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = x[i].scales[is++];
                float dl = d * (sc & 0xF);
                float ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
                }

                sc = x[i].scales[is++];
                dl = d * (sc & 0xF);
                ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;
                }

                shift += 2;
            }
            q += 32;
        }
    }
}

// ======================== Q3_K ========================
// 严格对照 llama.cpp dequantize_row_q3_K

void dequantize_q3_k(const void* src, float* dst, int64_t n_elements) {
    const auto* x = static_cast<const BlockQ3K*>(src);
    const int nb = n_elements / 256;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t* scales = (const int8_t*)aux;

    for (int i = 0; i < nb; i++) {
        const float d_all = f16_to_f32(x[i].d);

        const uint8_t* q  = x[i].qs;
        const uint8_t* hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        float* y = dst + i * 256;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

// ======================== Q4_K ========================
// 严格对照 llama.cpp dequantize_row_q4_K

void dequantize_q4_k(const void* src, float* dst, int64_t n_elements) {
    const auto* x = static_cast<const BlockQ4K*>(src);
    const int nb = n_elements / 256;

    for (int i = 0; i < nb; i++) {
        const uint8_t* q = x[i].qs;

        const float d   = f16_to_f32(x[i].d);
        const float min = f16_to_f32(x[i].dmin);

        float* y = dst + i * 256;
        int is = 0;
        uint8_t sc, m;

        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;

            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4)  - m2;

            q += 32;
            is += 2;
        }
    }
}

// ======================== Q5_K ========================
// 严格对照 llama.cpp dequantize_row_q5_K

void dequantize_q5_k(const void* src, float* dst, int64_t n_elements) {
    const auto* x = static_cast<const BlockQ5K*>(src);
    const int nb = n_elements / 256;

    for (int i = 0; i < nb; i++) {
        const uint8_t* ql = x[i].qs;
        const uint8_t* qh = x[i].qh;

        const float d   = f16_to_f32(x[i].d);
        const float min = f16_to_f32(x[i].dmin);

        float* y = dst + i * 256;
        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;

        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;

            for (int l = 0; l < 32; ++l) {
                *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *y++ = d2 * ((ql[l] >> 4)  + (qh[l] & u2 ? 16 : 0)) - m2;
            }

            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

// ======================== Q6_K ========================
// 严格对照 llama.cpp dequantize_row_q6_K

void dequantize_q6_k(const void* src, float* dst, int64_t n_elements) {
    const auto* x = static_cast<const BlockQ6K*>(src);
    const int nb = n_elements / 256;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);

        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t*  sc = x[i].scales;

        float* y = dst + i * 256;

        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;

                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// ======================== 通用入口 ========================

void dequantize(const void* src, float* dst, int64_t n_elements, GGMLType type) {
    switch (type) {
        case GGMLType::F32:
            memcpy(dst, src, n_elements * sizeof(float));
            break;
        case GGMLType::F16:
            dequantize_f16(src, dst, n_elements);
            break;
        case GGMLType::Q4_0:
            dequantize_q4_0(src, dst, n_elements);
            break;
        case GGMLType::Q4_1:
            dequantize_q4_1(src, dst, n_elements);
            break;
        case GGMLType::Q5_0:
            dequantize_q5_0(src, dst, n_elements);
            break;
        case GGMLType::Q5_1:
            dequantize_q5_1(src, dst, n_elements);
            break;
        case GGMLType::Q8_0:
            dequantize_q8_0(src, dst, n_elements);
            break;
        case GGMLType::Q2_K:
            dequantize_q2_k(src, dst, n_elements);
            break;
        case GGMLType::Q3_K:
            dequantize_q3_k(src, dst, n_elements);
            break;
        case GGMLType::Q4_K:
            dequantize_q4_k(src, dst, n_elements);
            break;
        case GGMLType::Q5_K:
            dequantize_q5_k(src, dst, n_elements);
            break;
        case GGMLType::Q6_K:
            dequantize_q6_k(src, dst, n_elements);
            break;
        default:
            throw std::runtime_error("Unsupported quantization type for dequantize: "
                                     + std::to_string(static_cast<int>(type)));
    }
}

std::vector<float> dequantize_tensor(const void* data, int64_t n_elements, GGMLType type) {
    std::vector<float> result(n_elements);
    dequantize(data, result.data(), n_elements, type);
    return result;
}

// ======================== 量化向量点积 ========================
// NEON SIMD 加速: 在反量化同时累加点积

float vec_dot_q4_k(const void* src, const float* y, int64_t n) {
    const auto* x = static_cast<const BlockQ4K*>(src);
    const int nb = n / 256;
    float sumf = 0.0f;

#ifdef USE_NEON
    for (int i = 0; i < nb; i++) {
        const uint8_t* q = x[i].qs;
        const float d    = f16_to_f32(x[i].d);
        const float dmin = f16_to_f32(x[i].dmin);
        const float* yb  = y + i * 256;

        int is = 0;
        uint8_t sc, m_val;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m_val);
            const float d1 = d * sc, m1 = dmin * m_val;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m_val);
            const float d2 = d * sc, m2 = dmin * m_val;

            // 用 NEON 处理 32 个元素
            float32x4_t vs1 = vdupq_n_f32(0.0f);
            float32x4_t vs2 = vdupq_n_f32(0.0f);
            float32x4_t vsy1 = vdupq_n_f32(0.0f);
            float32x4_t vsy2 = vdupq_n_f32(0.0f);

            for (int l = 0; l < 32; l += 4) {
                // 加载 4 个 quant 字节, 拆成低 4bit 和高 4bit
                float ql0 = q[l] & 0xF, ql1 = q[l+1] & 0xF, ql2 = q[l+2] & 0xF, ql3 = q[l+3] & 0xF;
                float qh0 = q[l] >> 4,  qh1 = q[l+1] >> 4,  qh2 = q[l+2] >> 4,  qh3 = q[l+3] >> 4;

                float32x4_t vql = {ql0, ql1, ql2, ql3};
                float32x4_t vqh = {qh0, qh1, qh2, qh3};

                float32x4_t vy1 = vld1q_f32(yb + j + l);
                float32x4_t vy2 = vld1q_f32(yb + j + l + 32);

                vs1 = vfmaq_f32(vs1, vy1, vql);
                vs2 = vfmaq_f32(vs2, vy2, vqh);
                vsy1 = vaddq_f32(vsy1, vy1);
                vsy2 = vaddq_f32(vsy2, vy2);
            }

            float s1 = vaddvq_f32(vs1);
            float s2 = vaddvq_f32(vs2);
            float sy1 = vaddvq_f32(vsy1);
            float sy2 = vaddvq_f32(vsy2);

            sumf += d1 * s1 - m1 * sy1 + d2 * s2 - m2 * sy2;
            q += 32;
            is += 2;
        }
    }
#else
    for (int i = 0; i < nb; i++) {
        const uint8_t* q = x[i].qs;
        const float d   = f16_to_f32(x[i].d);
        const float dmin = f16_to_f32(x[i].dmin);
        const float* yb = y + i * 256;

        int is = 0;
        uint8_t sc, m_val;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m_val);
            const float d1 = d * sc, m1 = dmin * m_val;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m_val);
            const float d2 = d * sc, m2 = dmin * m_val;

            float s1 = 0.0f, s2 = 0.0f, sy1 = 0.0f, sy2 = 0.0f;
            for (int l = 0; l < 32; ++l) {
                s1 += yb[j + l]      * (q[l] & 0xF);
                s2 += yb[j + l + 32] * (q[l] >> 4);
                sy1 += yb[j + l];
                sy2 += yb[j + l + 32];
            }
            sumf += d1 * s1 - m1 * sy1 + d2 * s2 - m2 * sy2;
            q += 32;
            is += 2;
        }
    }
#endif
    return sumf;
}

float vec_dot_q6_k(const void* src, const float* y, int64_t n) {
    const auto* x = static_cast<const BlockQ6K*>(src);
    const int nb = n / 256;
    float sumf = 0.0f;

#ifdef USE_NEON
    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t*  sc = x[i].scales;
        const float* yb = y + i * 256;

        for (int n2 = 0; n2 < 256; n2 += 128) {
            // 处理前16元素和后16元素分别用不同scale
            for (int half = 0; half < 2; half++) {
                // half=0: l=0..15, half=1: l=16..31
                int lbase = half * 16;
                float dsc0 = d * sc[half + 0];
                float dsc2 = d * sc[half + 2];
                float dsc4 = d * sc[half + 4];
                float dsc6 = d * sc[half + 6];

                float32x4_t vd0 = vdupq_n_f32(dsc0);
                float32x4_t vd2 = vdupq_n_f32(dsc2);
                float32x4_t vd4 = vdupq_n_f32(dsc4);
                float32x4_t vd6 = vdupq_n_f32(dsc6);
                float32x4_t v32 = vdupq_n_f32(32.0f);

                float32x4_t acc1 = vdupq_n_f32(0);
                float32x4_t acc2 = vdupq_n_f32(0);
                float32x4_t acc3 = vdupq_n_f32(0);
                float32x4_t acc4 = vdupq_n_f32(0);

                for (int l = lbase; l < lbase + 16; l += 4) {
                    // 解码 4 个 6-bit 值 (标量, 然后向量化)
                    float32x4_t vq1, vq2, vq3, vq4;
                    {
                        int32_t q1a[4], q2a[4], q3a[4], q4a[4];
                        for (int k = 0; k < 4; k++) {
                            int ll = l + k;
                            q1a[k] = (int)((ql[ll+ 0] & 0xF) | (((qh[ll] >> 0) & 3) << 4));
                            q2a[k] = (int)((ql[ll+32] & 0xF) | (((qh[ll] >> 2) & 3) << 4));
                            q3a[k] = (int)((ql[ll+ 0] >> 4)  | (((qh[ll] >> 4) & 3) << 4));
                            q4a[k] = (int)((ql[ll+32] >> 4)  | (((qh[ll] >> 6) & 3) << 4));
                        }
                        vq1 = vsubq_f32(vcvtq_f32_s32(vld1q_s32(q1a)), v32);
                        vq2 = vsubq_f32(vcvtq_f32_s32(vld1q_s32(q2a)), v32);
                        vq3 = vsubq_f32(vcvtq_f32_s32(vld1q_s32(q3a)), v32);
                        vq4 = vsubq_f32(vcvtq_f32_s32(vld1q_s32(q4a)), v32);
                    }

                    // 乘 scale
                    vq1 = vmulq_f32(vq1, vd0);
                    vq2 = vmulq_f32(vq2, vd2);
                    vq3 = vmulq_f32(vq3, vd4);
                    vq4 = vmulq_f32(vq4, vd6);

                    // 加载 y 并 FMA
                    acc1 = vfmaq_f32(acc1, vld1q_f32(yb + n2 + l +  0), vq1);
                    acc2 = vfmaq_f32(acc2, vld1q_f32(yb + n2 + l + 32), vq2);
                    acc3 = vfmaq_f32(acc3, vld1q_f32(yb + n2 + l + 64), vq3);
                    acc4 = vfmaq_f32(acc4, vld1q_f32(yb + n2 + l + 96), vq4);
                }
                sumf += vaddvq_f32(acc1) + vaddvq_f32(acc2) + vaddvq_f32(acc3) + vaddvq_f32(acc4);
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
#else
    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t*  sc = x[i].scales;
        const float* yb = y + i * 256;

        for (int n2 = 0; n2 < 256; n2 += 128) {
            float s1 = 0, s2 = 0, s3 = 0, s4 = 0;
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;

                s1 += yb[n2 + l +  0] * (d * sc[is + 0] * q1);
                s2 += yb[n2 + l + 32] * (d * sc[is + 2] * q2);
                s3 += yb[n2 + l + 64] * (d * sc[is + 4] * q3);
                s4 += yb[n2 + l + 96] * (d * sc[is + 6] * q4);
            }
            sumf += s1 + s2 + s3 + s4;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
#endif
    return sumf;
}

float vec_dot_q8_0(const void* src, const float* y, int64_t n) {
    const auto* x = static_cast<const BlockQ8_0*>(src);
    const int nb = n / 32;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        float s = 0.0f;
        for (int j = 0; j < 32; ++j) {
            s += y[i * 32 + j] * x[i].qs[j];
        }
        sumf += d * s;
    }
    return sumf;
}

float vec_dot_q4_0(const void* src, const float* y, int64_t n) {
    const auto* x = static_cast<const BlockQ4_0*>(src);
    const int nb = n / 32;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        const float* yb = y + i * 32;
        float s = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >> 4) - 8;
            s += yb[j]      * x0;
            s += yb[j + 16] * x1;
        }
        sumf += d * s;
    }
    return sumf;
}

float vec_dot_quant(const void* src, const float* y, int64_t n, GGMLType type) {
    switch (type) {
        case GGMLType::Q4_K: return vec_dot_q4_k(src, y, n);
        case GGMLType::Q6_K: return vec_dot_q6_k(src, y, n);
        case GGMLType::Q8_0: return vec_dot_q8_0(src, y, n);
        case GGMLType::Q4_0: return vec_dot_q4_0(src, y, n);
        default: return -1.0f; // 标记: 需要 fallback 到 dequant+dot
    }
}

} // namespace localllm
