// Metal Compute Shaders for Local-LLM
// GPU-accelerated matrix-vector multiplication with quantized weights
//
// Optimized: Each row uses a full SIMD group (32 threads) for parallel reduction.
// This gives ~10-20x speedup over the naive one-thread-per-row approach.

#include <metal_stdlib>
using namespace metal;

// ======================== Block structures ========================

struct block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct block_q8_0 {
    half d;
    int8_t qs[32];
};

struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    half d;
};

// ======================== Float mat-vec-mul ========================
// 每行用 32 个线程 (一个 SIMD group) 并行计算

kernel void mat_vec_mul_f32(
    device const float* W     [[buffer(0)]],
    device const float* x     [[buffer(1)]],
    device       float* y     [[buffer(2)]],
    constant   uint&    ncols [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_index_in_threadgroup]]
) {
    uint row = tgid;
    uint offset = row * ncols;

    float sum = 0.0f;
    for (uint j = lid; j < ncols; j += 32) {
        sum += W[offset + j] * x[j];
    }

    // SIMD group reduce
    sum += simd_shuffle_xor(sum, 16);
    sum += simd_shuffle_xor(sum, 8);
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);

    if (lid == 0) {
        y[row] = sum;
    }
}

// ======================== Q4_0 mat-vec-mul ========================

kernel void mat_vec_mul_q4_0(
    device const block_q4_0* W [[buffer(0)]],
    device const float*      x [[buffer(1)]],
    device       float*      y [[buffer(2)]],
    constant   uint&     ncols [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_index_in_threadgroup]]
) {
    uint nb = ncols / 32;
    uint row_offset = tgid * nb;

    float sum = 0.0f;
    for (uint b = lid; b < nb; b += 32) {
        device const block_q4_0& blk = W[row_offset + b];
        float d = float(blk.d);
        uint x_base = b * 32;

        for (uint j = 0; j < 16; j++) {
            uint8_t byte = blk.qs[j];
            float v0 = (float(byte & 0xF) - 8.0f) * d;
            float v1 = (float(byte >> 4)   - 8.0f) * d;
            sum += v0 * x[x_base + j] + v1 * x[x_base + j + 16];
        }
    }

    sum += simd_shuffle_xor(sum, 16);
    sum += simd_shuffle_xor(sum, 8);
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);

    if (lid == 0) {
        y[tgid] = sum;
    }
}

// ======================== Q8_0 mat-vec-mul ========================

kernel void mat_vec_mul_q8_0(
    device const block_q8_0* W [[buffer(0)]],
    device const float*      x [[buffer(1)]],
    device       float*      y [[buffer(2)]],
    constant   uint&     ncols [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_index_in_threadgroup]]
) {
    uint nb = ncols / 32;
    uint row_offset = tgid * nb;

    float sum = 0.0f;
    for (uint b = lid; b < nb; b += 32) {
        device const block_q8_0& blk = W[row_offset + b];
        float d = float(blk.d);
        uint x_base = b * 32;

        for (uint j = 0; j < 32; j++) {
            sum += float(blk.qs[j]) * d * x[x_base + j];
        }
    }

    sum += simd_shuffle_xor(sum, 16);
    sum += simd_shuffle_xor(sum, 8);
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);

    if (lid == 0) {
        y[tgid] = sum;
    }
}

// ======================== Q4_K mat-vec-mul ========================
// Most critical kernel for Q4_K_M models

kernel void mat_vec_mul_q4_k(
    device const block_q4_K* W [[buffer(0)]],
    device const float*      x [[buffer(1)]],
    device       float*      y [[buffer(2)]],
    constant   uint&     ncols [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_index_in_threadgroup]]
) {
    uint nb = ncols / 256;
    uint row_offset = tgid * nb;

    float sum = 0.0f;
    for (uint b = lid; b < nb; b += 32) {
        device const block_q4_K& blk = W[row_offset + b];
        float d = float(blk.d);
        float dmin = float(blk.dmin);
        uint x_base = b * 256;

        for (uint sb = 0; sb < 8; sb++) {
            uint8_t sc_byte, m_byte;
            if (sb < 4) {
                sc_byte = blk.scales[sb] & 0x3F;
                m_byte  = blk.scales[sb + 4] & 0x3F;
            } else {
                uint idx = sb - 4;
                sc_byte = (blk.scales[idx] >> 6) | ((blk.scales[idx + 8] & 0xF) << 2);
                m_byte  = (blk.scales[idx + 4] >> 6) | ((blk.scales[idx + 8] >> 4) << 2);
            }
            float sc = d * float(sc_byte);
            float mn = dmin * float(m_byte);

            uint qs_offset = sb * 16;
            uint x_off = x_base + sb * 32;

            for (uint j = 0; j < 16; j++) {
                uint8_t byte = blk.qs[qs_offset + j];
                float v0 = sc * float(byte & 0xF) - mn;
                float v1 = sc * float(byte >> 4)  - mn;
                sum += v0 * x[x_off + j] + v1 * x[x_off + j + 16];
            }
        }
    }

    sum += simd_shuffle_xor(sum, 16);
    sum += simd_shuffle_xor(sum, 8);
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);

    if (lid == 0) {
        y[tgid] = sum;
    }
}

// ======================== Q6_K mat-vec-mul ========================

kernel void mat_vec_mul_q6_k(
    device const block_q6_K* W [[buffer(0)]],
    device const float*      x [[buffer(1)]],
    device       float*      y [[buffer(2)]],
    constant   uint&     ncols [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_index_in_threadgroup]]
) {
    uint nb = ncols / 256;
    uint row_offset = tgid * nb;

    float sum = 0.0f;
    for (uint b = lid; b < nb; b += 32) {
        device const block_q6_K& blk = W[row_offset + b];
        float d = float(blk.d);
        uint x_base = b * 256;

        for (uint sb = 0; sb < 16; sb++) {
            float sc = d * float(blk.scales[sb]);
            uint ql_off = sb * 8;
            uint qh_off = sb * 4;
            uint x_off  = x_base + sb * 16;

            for (uint j = 0; j < 16; j++) {
                uint8_t ql_byte = blk.ql[ql_off + j / 2];
                uint8_t qh_byte = blk.qh[qh_off + j / 4];
                uint8_t ql_val = (j % 2 == 0) ? (ql_byte & 0xF) : (ql_byte >> 4);
                uint8_t qh_val = (qh_byte >> (2 * (j % 4))) & 0x3;
                int8_t  q = int8_t(ql_val | (qh_val << 4)) - 32;
                sum += sc * float(q) * x[x_off + j];
            }
        }
    }

    sum += simd_shuffle_xor(sum, 16);
    sum += simd_shuffle_xor(sum, 8);
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);

    if (lid == 0) {
        y[tgid] = sum;
    }
}
