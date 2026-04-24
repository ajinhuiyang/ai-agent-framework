// Metal Compute Shaders for Local-LLM
// GPU-accelerated matrix-vector multiplication with quantized weights
//
// These kernels replace the CPU-bound mat_vec_mul_tensor for the hot path:
// 28 layers x 7 matrices per layer = 196 mat-vec-muls per token.

#include <metal_stdlib>
using namespace metal;

// ======================== Q4_K block structure ========================
// K-quant 4-bit: 256 elements per super-block
// Layout: d(f16) dmin(f16) scales[12](uint8) qs[128](uint8)

struct block_q4_K {
    half d;             // super-block scale
    half dmin;          // super-block min
    uint8_t scales[12]; // sub-block scales and mins (packed)
    uint8_t qs[128];    // quantized values (4-bit, 2 per byte)
};

// ======================== Q8_0 block structure ========================
// 32 elements per block: d(f16) + 32 x int8

struct block_q8_0 {
    half d;          // scale
    int8_t qs[32];   // quantized values
};

// ======================== Q4_0 block structure ========================
// 32 elements per block: d(f16) + 16 bytes (32 x 4-bit)

struct block_q4_0 {
    half d;          // scale
    uint8_t qs[16];  // quantized values (4-bit, 2 per byte)
};

// ======================== Q6_K block structure ========================
// 256 elements per super-block

struct block_q6_K {
    uint8_t ql[128];   // lower 4 bits of quantized values
    uint8_t qh[64];    // upper 2 bits of quantized values
    int8_t  scales[16]; // scales
    half d;             // super-block scale
};

// ======================== Float mat-vec-mul ========================
// y[row] = dot(W[row, :], x[:])  for each row in [row_start, row_start + n_rows)
// W is row-major [M x N], x is [N], y is [M]

kernel void mat_vec_mul_f32(
    device const float* W     [[buffer(0)]],
    device const float* x     [[buffer(1)]],
    device       float* y     [[buffer(2)]],
    constant   uint&    ncols [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    float sum = 0.0f;
    uint offset = tid * ncols;
    for (uint j = 0; j < ncols; j++) {
        sum += W[offset + j] * x[j];
    }
    y[tid] = sum;
}

// ======================== Q4_0 mat-vec-mul ========================

kernel void mat_vec_mul_q4_0(
    device const block_q4_0* W [[buffer(0)]],
    device const float*      x [[buffer(1)]],
    device       float*      y [[buffer(2)]],
    constant   uint&     ncols [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Each row has ncols elements = ncols/32 blocks
    uint nb = ncols / 32;
    uint row_offset = tid * nb;

    float sum = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const block_q4_0& blk = W[row_offset + b];
        float d = float(blk.d);
        uint x_base = b * 32;

        for (uint j = 0; j < 16; j++) {
            uint8_t byte = blk.qs[j];
            float v0 = (float(byte & 0xF) - 8.0f) * d;
            float v1 = (float(byte >> 4)   - 8.0f) * d;
            sum += v0 * x[x_base + j]      +
                   v1 * x[x_base + j + 16];
        }
    }
    y[tid] = sum;
}

// ======================== Q8_0 mat-vec-mul ========================

kernel void mat_vec_mul_q8_0(
    device const block_q8_0* W [[buffer(0)]],
    device const float*      x [[buffer(1)]],
    device       float*      y [[buffer(2)]],
    constant   uint&     ncols [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint nb = ncols / 32;
    uint row_offset = tid * nb;

    float sum = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const block_q8_0& blk = W[row_offset + b];
        float d = float(blk.d);
        uint x_base = b * 32;

        for (uint j = 0; j < 32; j++) {
            sum += float(blk.qs[j]) * d * x[x_base + j];
        }
    }
    y[tid] = sum;
}

// ======================== Q4_K mat-vec-mul ========================
// Most complex: K-quant with sub-block scales

kernel void mat_vec_mul_q4_k(
    device const block_q4_K* W [[buffer(0)]],
    device const float*      x [[buffer(1)]],
    device       float*      y [[buffer(2)]],
    constant   uint&     ncols [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // 256 elements per super-block
    uint nb = ncols / 256;
    uint row_offset = tid * nb;

    float sum = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const block_q4_K& blk = W[row_offset + b];
        float d = float(blk.d);
        float dmin = float(blk.dmin);
        uint x_base = b * 256;

        // Process 8 sub-blocks of 32 elements each
        for (uint sb = 0; sb < 8; sb++) {
            // Decode scale and min for this sub-block
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
                sum += v0 * x[x_off + j] +
                       v1 * x[x_off + j + 16];
            }
        }
    }
    y[tid] = sum;
}

// ======================== Q6_K mat-vec-mul ========================

kernel void mat_vec_mul_q6_k(
    device const block_q6_K* W [[buffer(0)]],
    device const float*      x [[buffer(1)]],
    device       float*      y [[buffer(2)]],
    constant   uint&     ncols [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint nb = ncols / 256;
    uint row_offset = tid * nb;

    float sum = 0.0f;
    for (uint b = 0; b < nb; b++) {
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
    y[tid] = sum;
}

// ======================== RMSNorm ========================

kernel void rmsnorm_kernel(
    device const float* x      [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device       float* y      [[buffer(2)]],
    constant   uint&    n      [[buffer(3)]],
    constant   float&   eps    [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    // 单线程计算 (n 通常只有 3584, 不值得多线程 reduce)
    if (tid != 0) return;

    float ss = 0.0f;
    for (uint i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    ss = 1.0f / sqrt(ss / float(n) + eps);
    for (uint i = 0; i < n; i++) {
        y[i] = x[i] * ss * weight[i];
    }
}

// ======================== Element-wise add ========================

kernel void vec_add_kernel(
    device       float* y [[buffer(0)]],
    device const float* x [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    y[tid] += x[tid];
}

// ======================== SiLU activation ========================

kernel void silu_mul_kernel(
    device float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    float g = gate[tid];
    gate[tid] = (g / (1.0f + exp(-g))) * up[tid];
}
