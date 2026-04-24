#pragma once

#include "gguf_parser.h"
#include "dequantize.h"
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

#ifdef USE_METAL
#include "metal_compute.h"
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace localllm {

// ======================== 线程池 ========================

class ThreadPool {
public:
    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    int num_threads() const { return n_threads_; }

    // 并行执行: 将 [0, total) 分给多个线程
    // 使用过分拆分 (over-decomposition) 实现自动负载均衡
    void parallel_for(int64_t total, const std::function<void(int64_t, int64_t)>& fn) {
        if (total <= 0) return;
        if (n_threads_ <= 1 || total < 32) {
            fn(0, total);
            return;
        }

        // 拆成比线程数更多的 chunk, 实现负载均衡
        // 慢线程 (E核) 做少点, 快线程 (P核) 多抢
        int n_chunks = n_threads_ * 4;
        if (n_chunks > (int)total) n_chunks = (int)total;
        int64_t chunk = (total + n_chunks - 1) / n_chunks;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_remaining_ = 0;
            for (int t = 0; t < n_chunks; ++t) {
                int64_t start = t * chunk;
                int64_t end = std::min(start + chunk, total);
                if (start >= total) break;
                task_queue_.push_back({fn, start, end});
                tasks_remaining_++;
            }
        }
        cv_work_.notify_all();

        std::unique_lock<std::mutex> lock(mutex_);
        cv_done_.wait(lock, [this] { return tasks_remaining_ == 0; });
    }

private:
    ThreadPool() {
        n_threads_ = static_cast<int>(std::thread::hardware_concurrency());
        if (n_threads_ <= 0) n_threads_ = 4;

        // 获取性能核数
        n_perf_cores_ = n_threads_;
#if defined(__APPLE__)
        int perf_cores = 0;
        size_t sz = sizeof(perf_cores);
        if (sysctlbyname("hw.perflevel0.logicalcpu", &perf_cores, &sz, nullptr, 0) == 0 && perf_cores > 0) {
            n_perf_cores_ = perf_cores;
        }
#endif

        for (int i = 0; i < n_threads_; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_work_.notify_all();
        for (auto& w : workers_) w.join();
    }

    struct Task {
        std::function<void(int64_t, int64_t)> fn;
        int64_t start, end;
    };

    void worker_loop() {
        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_work_.wait(lock, [this] { return stop_ || !task_queue_.empty(); });
                if (stop_ && task_queue_.empty()) return;
                task = task_queue_.back();
                task_queue_.pop_back();
            }
            task.fn(task.start, task.end);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                tasks_remaining_--;
            }
            cv_done_.notify_one();
        }
    }

    int n_threads_;
    int n_perf_cores_ = 4;
    std::vector<std::thread> workers_;
    std::vector<Task> task_queue_;
    std::mutex mutex_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    int tasks_remaining_ = 0;
    bool stop_ = false;
};

// ======================== Tensor ========================

class Tensor {
public:
    Tensor() = default;

    Tensor(const std::vector<int64_t>& shape)
        : shape_(shape), type_(GGMLType::F32) {
        int64_t n = num_elements();
        data_.resize(n, 0.0f);
    }

    Tensor(const std::vector<int64_t>& shape, const std::vector<float>& data)
        : shape_(shape), data_(data), type_(GGMLType::F32) {}

    Tensor(const std::vector<int64_t>& shape, std::vector<float>&& data)
        : shape_(shape), data_(std::move(data)), type_(GGMLType::F32) {}

    static Tensor from_gguf(const GGUFParser& parser, const std::string& name, bool force_dequant = false) {
        const GGUFTensorInfo* info = parser.find_tensor(name);
        if (!info) {
            throw std::runtime_error("Tensor not found: " + name);
        }

        const void* raw_data = parser.get_tensor_data(*info);
        int64_t ne = info->num_elements();

        std::vector<int64_t> shape;
        if (info->n_dimensions == 1) {
            shape = {static_cast<int64_t>(info->dimensions[0])};
        } else if (info->n_dimensions == 2) {
            shape = {static_cast<int64_t>(info->dimensions[1]),
                     static_cast<int64_t>(info->dimensions[0])};
        } else {
            for (int d = info->n_dimensions - 1; d >= 0; --d) {
                shape.push_back(static_cast<int64_t>(info->dimensions[d]));
            }
        }

        if (info->type == GGMLType::F32 || force_dequant) {
            Tensor t;
            t.shape_ = shape;
            t.type_ = GGMLType::F32;
            if (info->type == GGMLType::F32) {
                t.data_.resize(ne);
                memcpy(t.data_.data(), raw_data, ne * sizeof(float));
            } else {
                t.data_ = dequantize_tensor(raw_data, ne, info->type);
            }
            return t;
        }

        Tensor t;
        t.shape_ = shape;
        t.type_ = info->type;
        t.quant_data_ = static_cast<const uint8_t*>(raw_data);
        t.quant_size_ = info->byte_size();
        // 预计算行字节数
        auto traits = get_type_traits(t.type_);
        t.row_bytes_ = (shape.back() / traits.block_size) * traits.type_size;
        return t;
    }

    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t dim(int i) const { return shape_[i]; }
    int ndim() const { return static_cast<int>(shape_.size()); }
    GGMLType type() const { return type_; }
    bool is_quantized() const { return type_ != GGMLType::F32 && quant_data_ != nullptr; }

    int64_t num_elements() const {
        int64_t n = 1;
        for (auto d : shape_) n *= d;
        return n;
    }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    size_t size_bytes() const {
        if (is_quantized()) return quant_size_;
        return data_.size() * sizeof(float);
    }

    float& operator[](int64_t i) { return data_[i]; }
    const float& operator[](int64_t i) const { return data_[i]; }

    float& at(int64_t row, int64_t col) { return data_[row * shape_[1] + col]; }
    const float& at(int64_t row, int64_t col) const { return data_[row * shape_[1] + col]; }

    float* row_data(int64_t row) { return data_.data() + row * shape_.back(); }
    const float* row_data(int64_t row) const { return data_.data() + row * shape_.back(); }

    const uint8_t* quant_row_data(int64_t row) const {
        return quant_data_ + row * row_bytes_;
    }

    void dequant_row(int64_t row, float* dst) const {
        if (type_ == GGMLType::F32) {
            memcpy(dst, row_data(row), shape_.back() * sizeof(float));
            return;
        }
        dequantize(quant_row_data(row), dst, shape_.back(), type_);
    }

    bool empty() const { return data_.empty() && quant_data_ == nullptr; }

    void resize(const std::vector<int64_t>& shape) {
        shape_ = shape;
        type_ = GGMLType::F32;
        data_.resize(num_elements(), 0.0f);
    }

    void fill(float val) { std::fill(data_.begin(), data_.end(), val); }
    void zero() { fill(0.0f); }

private:
    std::vector<int64_t> shape_;
    std::vector<float> data_;
    const uint8_t* quant_data_ = nullptr;
    size_t quant_size_ = 0;
    size_t row_bytes_ = 0;     // 预计算的量化行字节数
    GGMLType type_ = GGMLType::F32;
};

// ======================== 基础线性代数 ========================

inline void mat_vec_mul(const float* A, const float* x, float* y, int64_t m, int64_t n) {
#ifdef USE_METAL
    if (MetalContext::instance().mat_vec_mul_f32(A, x, y, m, n)) return;
#endif
#ifdef USE_ACCELERATE
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)m, (int)n,
                1.0f, A, (int)n, x, 1, 0.0f, y, 1);
#else
    for (int64_t i = 0; i < m; ++i) {
        float sum = 0.0f;
        const float* row = A + i * n;
        for (int64_t j = 0; j < n; ++j) sum += row[j] * x[j];
        y[i] = sum;
    }
#endif
}

// 量化 mat-vec-mul, 线程池并行
// 对支持的量化类型使用融合 dot product (跳过反量化中间缓冲区)
// 对不支持的类型 fallback 到 dequant + dot
inline void mat_vec_mul_tensor(const Tensor& W, const float* x, float* y, int64_t m, int64_t n) {
    if (!W.is_quantized()) {
        mat_vec_mul(W.data(), x, y, m, n);
        return;
    }

#ifdef USE_METAL
    // 优先尝试 GPU
    if (MetalContext::instance().mat_vec_mul(
            W.quant_row_data(0), x, y, m, n, W.type())) {
        return;
    }
    // GPU 不支持此量化类型或执行失败, fallback 到 CPU
#endif

    GGMLType wtype = W.type();
    // 检查是否有融合 dot product 实现
    bool has_fused = (wtype == GGMLType::Q4_K || wtype == GGMLType::Q6_K ||
                      wtype == GGMLType::Q8_0 || wtype == GGMLType::Q4_0);

    ThreadPool::instance().parallel_for(m, [&](int64_t row_start, int64_t row_end) {
        if (has_fused) {
            for (int64_t i = row_start; i < row_end; ++i) {
                y[i] = vec_dot_quant(W.quant_row_data(i), x, n, wtype);
            }
        } else {
            // Fallback: dequant + dot
            std::vector<float> buf(n);
            for (int64_t i = row_start; i < row_end; ++i) {
                W.dequant_row(i, buf.data());
#ifdef USE_ACCELERATE
                vDSP_dotpr(buf.data(), 1, x, 1, &y[i], (vDSP_Length)n);
#else
                float sum = 0.0f;
                for (int64_t j = 0; j < n; ++j) sum += buf[j] * x[j];
                y[i] = sum;
#endif
            }
        }
    });
}

// ======================== Batch 矩阵乘法 (Prefill 加速) ========================
// Y[m, seq_len] = W[m, n] @ X[n, seq_len]
// W: 权重矩阵 (可量化), X: 多个 token 的输入 (列主序), Y: 输出 (列主序)
// 将多个 mat-vec-mul 合并为一个 mat-mat-mul, 大幅提升 prefill 吞吐

// Float 矩阵乘法: Y[m, k] = A[m, n] @ B[n, k]
// A: row-major [m x n], B: column-major [n x k], Y: column-major [m x k]
inline void mat_mat_mul(const float* A, const float* B, float* Y,
                        int64_t m, int64_t n, int64_t k) {
#ifdef USE_ACCELERATE
    // cblas_sgemm: C = alpha * A @ B + beta * C
    // A: row-major [m x n], B: col-major [n x k], C: col-major [m x k]
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                (int)m, (int)k, (int)n,
                1.0f, A, (int)m, B, (int)n,
                0.0f, Y, (int)m);
#else
    // Naive fallback: 逐列计算 Y[:,j] = A @ B[:,j]
    for (int64_t j = 0; j < k; ++j) {
        const float* b_col = B + j * n;
        float* y_col = Y + j * m;
        for (int64_t i = 0; i < m; ++i) {
            float sum = 0.0f;
            const float* a_row = A + i * n;
            for (int64_t p = 0; p < n; ++p) sum += a_row[p] * b_col[p];
            y_col[i] = sum;
        }
    }
#endif
}

// 量化权重的 batch mat-mul: Y[m, seq_len] = W_quant[m, n] @ X[n, seq_len]
// 对每一列 (token) 做量化 mat-vec-mul, 利用线程池并行化
// 比逐 token 串行快, 因为所有 token 的同一行可以共享反量化结果
inline void mat_mat_mul_tensor(const Tensor& W, const float* X, float* Y,
                               int64_t m, int64_t n, int64_t seq_len) {
    if (seq_len == 1) {
        // 单 token: 直接走优化过的 mat-vec-mul
        mat_vec_mul_tensor(W, X, Y, m, n);
        return;
    }

    if (!W.is_quantized()) {
        // Float 权重: 直接用 BLAS mat-mat-mul
        mat_mat_mul(W.data(), X, Y, m, n, seq_len);
        return;
    }

    // 量化权重: 先反量化每行, 然后一次性处理所有 seq_len 个 token
    // 这比逐 token 分别调 mat_vec_mul_tensor 快, 因为反量化只做一次
    ThreadPool::instance().parallel_for(m, [&](int64_t row_start, int64_t row_end) {
        std::vector<float> row_buf(n);
        for (int64_t i = row_start; i < row_end; ++i) {
            W.dequant_row(i, row_buf.data());
            for (int64_t j = 0; j < seq_len; ++j) {
                const float* x_col = X + j * n;  // 第 j 个 token
                float* y_col = Y + j * m;        // 第 j 个输出
#ifdef USE_ACCELERATE
                vDSP_dotpr(row_buf.data(), 1, x_col, 1, &y_col[i], (vDSP_Length)n);
#else
                float sum = 0.0f;
                for (int64_t p = 0; p < n; ++p) sum += row_buf[p] * x_col[p];
                y_col[i] = sum;
#endif
            }
        }
    });
}

inline void vec_add(float* y, const float* a, const float* b, int64_t n) {
#ifdef USE_ACCELERATE
    vDSP_vadd(a, 1, b, 1, y, 1, (vDSP_Length)n);
#else
    for (int64_t i = 0; i < n; ++i) y[i] = a[i] + b[i];
#endif
}

inline void vec_add_inplace(float* y, const float* x, int64_t n) {
#ifdef USE_ACCELERATE
    // y = y + x
    vDSP_vadd(y, 1, x, 1, y, 1, (vDSP_Length)n);
#else
    for (int64_t i = 0; i < n; ++i) y[i] += x[i];
#endif
}

inline void vec_mul(float* y, const float* a, const float* b, int64_t n) {
#ifdef USE_ACCELERATE
    vDSP_vmul(a, 1, b, 1, y, 1, (vDSP_Length)n);
#else
    for (int64_t i = 0; i < n; ++i) y[i] = a[i] * b[i];
#endif
}

inline void vec_scale(float* y, const float* x, float scale, int64_t n) {
#ifdef USE_ACCELERATE
    vDSP_vsmul(x, 1, &scale, y, 1, (vDSP_Length)n);
#else
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * scale;
#endif
}

inline float vec_dot(const float* a, const float* b, int64_t n) {
#ifdef USE_ACCELERATE
    float result;
    vDSP_dotpr(a, 1, b, 1, &result, (vDSP_Length)n);
    return result;
#else
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

inline void rmsnorm(float* y, const float* x, const float* weight, int64_t n, float eps = 1e-5f) {
#ifdef USE_ACCELERATE
    float ss;
    vDSP_dotpr(x, 1, x, 1, &ss, (vDSP_Length)n);
    ss = 1.0f / sqrtf(ss / n + eps);
    // y = x * ss * weight
    vDSP_vsmul(x, 1, &ss, y, 1, (vDSP_Length)n);
    vDSP_vmul(y, 1, weight, 1, y, 1, (vDSP_Length)n);
#else
    float ss = 0.0f;
    for (int64_t i = 0; i < n; ++i) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * ss * weight[i];
#endif
}

inline void layernorm(float* y, const float* x, const float* weight, const float* bias,
                      int64_t n, float eps = 1e-5f) {
    float mean = 0.0f;
    for (int64_t i = 0; i < n; ++i) mean += x[i];
    mean /= n;
    float var = 0.0f;
    for (int64_t i = 0; i < n; ++i) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float scale = 1.0f / sqrtf(var + eps);
    for (int64_t i = 0; i < n; ++i) {
        y[i] = (x[i] - mean) * scale * weight[i];
        if (bias) y[i] += bias[i];
    }
}

inline void silu(float* y, const float* x, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

inline void gelu(float* y, const float* x, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = 0.5f * x[i] * (1.0f + tanhf(0.7978845608f * (x[i] + 0.044715f * x[i] * x[i] * x[i])));
    }
}

inline void softmax(float* x, int64_t n) {
    float max_val = x[0];
    for (int64_t i = 1; i < n; ++i) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    float inv_sum = 1.0f / sum;
#ifdef USE_ACCELERATE
    vDSP_vsmul(x, 1, &inv_sum, x, 1, (vDSP_Length)n);
#else
    for (int64_t i = 0; i < n; ++i) x[i] *= inv_sum;
#endif
}

inline void rope(float* q, float* k, int head_dim, int pos, float theta = 10000.0f) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(theta, (float)i / head_dim);
        float val = pos * freq;
        float cos_val = cosf(val);
        float sin_val = sinf(val);

        float q0 = q[i], q1 = q[i + 1];
        q[i]     = q0 * cos_val - q1 * sin_val;
        q[i + 1] = q0 * sin_val + q1 * cos_val;

        float k0 = k[i], k1 = k[i + 1];
        k[i]     = k0 * cos_val - k1 * sin_val;
        k[i + 1] = k0 * sin_val + k1 * cos_val;
    }
}

} // namespace localllm
