// Metal GPU compute implementation for Local-LLM
// Objective-C++ (.mm) to use Metal framework APIs.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_compute.h"
#include <iostream>
#include <unordered_map>

namespace localllm {

// ======================== Impl ========================

struct MetalContext::Impl {
    id<MTLDevice>       device       = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLLibrary>      library      = nil;

    // Pipeline states for each kernel
    id<MTLComputePipelineState> pso_f32  = nil;
    id<MTLComputePipelineState> pso_q4_0 = nil;
    id<MTLComputePipelineState> pso_q8_0 = nil;
    id<MTLComputePipelineState> pso_q4_k = nil;
    id<MTLComputePipelineState> pso_q6_k = nil;

    // Preallocated buffers for x and y to avoid per-call allocation
    id<MTLBuffer> x_buf  = nil;
    id<MTLBuffer> y_buf  = nil;
    int64_t x_buf_size   = 0;
    int64_t y_buf_size   = 0;

    bool is_available = false;
    std::string dev_name;

    bool init() {
        @autoreleasepool {
            // 1. 获取默认 GPU 设备
            device = MTLCreateSystemDefaultDevice();
            if (!device) {
                std::cerr << "[Metal] No Metal-capable GPU found" << std::endl;
                return false;
            }
            dev_name = [[device name] UTF8String];
            std::cout << "[Metal] GPU device: " << dev_name << std::endl;

            // 2. 创建命令队列
            commandQueue = [device newCommandQueue];
            if (!commandQueue) {
                std::cerr << "[Metal] Failed to create command queue" << std::endl;
                return false;
            }

            // 3. 加载 shader: 运行时从 .metal 源文件编译
            NSError* error = nil;

            // 尝试从源文件编译
            NSString* srcPath = @METAL_SHADER_SOURCE;
            NSString* shaderSrc = [NSString stringWithContentsOfFile:srcPath
                                                            encoding:NSUTF8StringEncoding
                                                               error:&error];
            if (shaderSrc) {
                MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
                opts.mathMode = MTLMathModeRelaxed;
                library = [device newLibraryWithSource:shaderSrc options:opts error:&error];
                if (!library) {
                    std::cerr << "[Metal] Failed to compile shaders: "
                              << [[error localizedDescription] UTF8String] << std::endl;
                    return false;
                }
                std::cout << "[Metal] Shaders compiled from source" << std::endl;
            } else {
                std::cerr << "[Metal] Shader source not found at: "
                          << [srcPath UTF8String] << std::endl;
                return false;
            }

            // 4. 创建各 kernel 的 pipeline state
            pso_f32  = createPSO("mat_vec_mul_f32");
            pso_q4_0 = createPSO("mat_vec_mul_q4_0");
            pso_q8_0 = createPSO("mat_vec_mul_q8_0");
            pso_q4_k = createPSO("mat_vec_mul_q4_k");
            pso_q6_k = createPSO("mat_vec_mul_q6_k");

            if (!pso_q4_k) {
                std::cerr << "[Metal] Failed to create critical pipeline states" << std::endl;
                return false;
            }

            is_available = true;
            std::cout << "[Metal] GPU acceleration initialized successfully" << std::endl;
            return true;
        }
    }

    id<MTLComputePipelineState> createPSO(const char* name) {
        @autoreleasepool {
            NSString* funcName = [NSString stringWithUTF8String:name];
            id<MTLFunction> func = [library newFunctionWithName:funcName];
            if (!func) {
                std::cerr << "[Metal] Kernel not found: " << name << std::endl;
                return nil;
            }
            NSError* error = nil;
            id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func error:&error];
            if (!pso) {
                std::cerr << "[Metal] Failed to create PSO for " << name << ": "
                          << [[error localizedDescription] UTF8String] << std::endl;
                return nil;
            }
            return pso;
        }
    }

    void ensureBuffers(int64_t x_bytes, int64_t y_bytes) {
        if (x_bytes > x_buf_size) {
            x_buf_size = x_bytes * 2; // 预留 2 倍避免频繁重分配
            x_buf = [device newBufferWithLength:x_buf_size options:MTLResourceStorageModeShared];
        }
        if (y_bytes > y_buf_size) {
            y_buf_size = y_bytes * 2;
            y_buf = [device newBufferWithLength:y_buf_size options:MTLResourceStorageModeShared];
        }
    }

    bool dispatch(id<MTLComputePipelineState> pso,
                  const void* W_data, int64_t W_bytes,
                  const float* x, int64_t n,
                  float* y, int64_t m) {
        @autoreleasepool {
            if (!pso) return false;

            // 准备 buffers
            // W: 使用 no-copy shared buffer (权重数据已 mmap, 直接让 GPU 读)
            id<MTLBuffer> W_buf = [device newBufferWithBytesNoCopy:(void*)W_data
                                                           length:W_bytes
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
            if (!W_buf) {
                // Fallback: 如果 no-copy 失败 (地址不对齐等), 用 copy 方式
                W_buf = [device newBufferWithBytes:W_data length:W_bytes options:MTLResourceStorageModeShared];
            }
            if (!W_buf) return false;

            int64_t x_bytes = n * sizeof(float);
            int64_t y_bytes = m * sizeof(float);
            ensureBuffers(x_bytes, y_bytes);

            // 拷贝 x 到 GPU buffer
            memcpy([x_buf contents], x, x_bytes);

            // ncols 常量
            uint32_t ncols = (uint32_t)n;

            // 创建命令
            id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

            [encoder setComputePipelineState:pso];
            [encoder setBuffer:W_buf offset:0 atIndex:0];
            [encoder setBuffer:x_buf offset:0 atIndex:1];
            [encoder setBuffer:y_buf offset:0 atIndex:2];
            [encoder setBytes:&ncols length:sizeof(ncols) atIndex:3];

            // 每行一个线程
            MTLSize gridSize = MTLSizeMake(m, 1, 1);
            NSUInteger threadGroupSize = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)m);
            // 对齐到 32 的倍数 (warp size)
            if (threadGroupSize > 32) threadGroupSize = (threadGroupSize / 32) * 32;
            MTLSize tgSize = MTLSizeMake(threadGroupSize, 1, 1);

            [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
            [encoder endEncoding];

            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            // 拷贝结果回 CPU
            memcpy(y, [y_buf contents], y_bytes);

            return true;
        }
    }

    void cleanup() {
        pso_f32  = nil;
        pso_q4_0 = nil;
        pso_q8_0 = nil;
        pso_q4_k = nil;
        pso_q6_k = nil;
        x_buf = nil;
        y_buf = nil;
        library = nil;
        commandQueue = nil;
        device = nil;
    }
};

// ======================== Public API ========================

MetalContext& MetalContext::instance() {
    static MetalContext ctx;
    return ctx;
}

MetalContext::MetalContext() {
    impl_ = new Impl();
    impl_->init();
}

MetalContext::~MetalContext() {
    if (impl_) {
        impl_->cleanup();
        delete impl_;
    }
}

bool MetalContext::available() const {
    return impl_ && impl_->is_available;
}

std::string MetalContext::device_name() const {
    return impl_ ? impl_->dev_name : "";
}

void MetalContext::preallocate(int64_t max_rows, int64_t max_cols) {
    if (!impl_ || !impl_->is_available) return;
    impl_->ensureBuffers(max_cols * sizeof(float), max_rows * sizeof(float));
}

bool MetalContext::mat_vec_mul_f32(const float* W, const float* x, float* y,
                                    int64_t m, int64_t n) {
    if (!impl_ || !impl_->is_available) return false;
    int64_t W_bytes = m * n * sizeof(float);
    return impl_->dispatch(impl_->pso_f32, W, W_bytes, x, n, y, m);
}

bool MetalContext::mat_vec_mul(const void* W_data, const float* x, float* y,
                                int64_t m, int64_t n, GGMLType type) {
    if (!impl_ || !impl_->is_available) return false;

    // 选择对应的 kernel 和计算权重数据大小
    id<MTLComputePipelineState> pso = nil;
    int64_t W_bytes = 0;

    auto traits = get_type_traits(type);

    switch (type) {
        case GGMLType::Q4_0:
            pso = impl_->pso_q4_0;
            W_bytes = (m * n / traits.block_size) * traits.type_size;
            break;
        case GGMLType::Q8_0:
            pso = impl_->pso_q8_0;
            W_bytes = (m * n / traits.block_size) * traits.type_size;
            break;
        case GGMLType::Q4_K:
            pso = impl_->pso_q4_k;
            W_bytes = (m * n / traits.block_size) * traits.type_size;
            break;
        case GGMLType::Q6_K:
            pso = impl_->pso_q6_k;
            W_bytes = (m * n / traits.block_size) * traits.type_size;
            break;
        default:
            return false; // 不支持的类型, fallback 到 CPU
    }

    if (!pso) return false;

    return impl_->dispatch(pso, W_data, W_bytes, x, n, y, m);
}

} // namespace localllm
