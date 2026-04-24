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

    // Weight buffer cache: pointer -> MTLBuffer (avoid recreating every dispatch)
    std::unordered_map<const void*, id<MTLBuffer>> weight_cache;

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

            // W buffer: 从缓存中获取或创建 (权重不变, 可复用)
            id<MTLBuffer> W_buf = nil;
            auto it = weight_cache.find(W_data);
            if (it != weight_cache.end()) {
                W_buf = it->second;
            } else {
                W_buf = [device newBufferWithBytesNoCopy:(void*)W_data
                                                 length:W_bytes
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
                if (!W_buf) {
                    W_buf = [device newBufferWithBytes:W_data length:W_bytes options:MTLResourceStorageModeShared];
                }
                if (W_buf) {
                    weight_cache[W_data] = W_buf;
                }
            }
            if (!W_buf) return false;

            int64_t x_bytes = n * sizeof(float);
            int64_t y_bytes = m * sizeof(float);
            ensureBuffers(x_bytes, y_bytes);

            // 拷贝 x 到 GPU shared buffer
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

            // 每行一个 threadgroup (32 线程), 行内并行 reduce
            MTLSize numThreadgroups = MTLSizeMake(m, 1, 1);  // m 行 = m 个 threadgroup
            MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);  // SIMD group size

            [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerGroup];
            [encoder endEncoding];

            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            // 拷贝结果回 CPU
            memcpy(y, [y_buf contents], y_bytes);

            return true;
        }
    }

    void cleanup() {
        weight_cache.clear();
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

bool MetalContext::mat_vec_mul_batch(const float* x, int64_t n,
                                     const BatchOp* ops, int num_ops) {
    if (!impl_ || !impl_->is_available || num_ops == 0) return false;

    @autoreleasepool {
        // 准备 x buffer
        int64_t x_bytes = n * sizeof(float);
        impl_->ensureBuffers(x_bytes, 0);
        memcpy([impl_->x_buf contents], x, x_bytes);
        uint32_t ncols = (uint32_t)n;

        // 为每个 op 准备 y buffer
        struct OpInfo {
            id<MTLComputePipelineState> pso;
            id<MTLBuffer> W_buf;
            id<MTLBuffer> y_buf;
            float* y_cpu;
            int64_t y_bytes;
        };
        std::vector<OpInfo> infos(num_ops);

        for (int i = 0; i < num_ops; i++) {
            auto& op = ops[i];
            auto& info = infos[i];

            // 选择 PSO
            switch (op.type) {
                case GGMLType::Q4_0: info.pso = impl_->pso_q4_0; break;
                case GGMLType::Q8_0: info.pso = impl_->pso_q8_0; break;
                case GGMLType::Q4_K: info.pso = impl_->pso_q4_k; break;
                case GGMLType::Q6_K: info.pso = impl_->pso_q6_k; break;
                default: return false;
            }
            if (!info.pso) return false;

            // W buffer (从缓存获取)
            auto it = impl_->weight_cache.find(op.W_data);
            if (it != impl_->weight_cache.end()) {
                info.W_buf = it->second;
            } else {
                info.W_buf = [impl_->device newBufferWithBytesNoCopy:(void*)op.W_data
                                                              length:op.W_bytes
                                                             options:MTLResourceStorageModeShared
                                                         deallocator:nil];
                if (!info.W_buf) {
                    info.W_buf = [impl_->device newBufferWithBytes:op.W_data
                                                           length:op.W_bytes
                                                          options:MTLResourceStorageModeShared];
                }
                if (info.W_buf) impl_->weight_cache[op.W_data] = info.W_buf;
            }
            if (!info.W_buf) return false;

            // y buffer
            info.y_bytes = op.m * sizeof(float);
            info.y_buf = [impl_->device newBufferWithLength:info.y_bytes
                                                   options:MTLResourceStorageModeShared];
            info.y_cpu = op.y;
        }

        // 创建单个 command buffer, 编码所有操作
        id<MTLCommandBuffer> cmdBuf = [impl_->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        for (int i = 0; i < num_ops; i++) {
            auto& info = infos[i];
            auto& op = ops[i];

            [encoder setComputePipelineState:info.pso];
            [encoder setBuffer:info.W_buf offset:0 atIndex:0];
            [encoder setBuffer:impl_->x_buf offset:0 atIndex:1];
            [encoder setBuffer:info.y_buf offset:0 atIndex:2];
            [encoder setBytes:&ncols length:sizeof(ncols) atIndex:3];

            MTLSize gridSize = MTLSizeMake(op.m, 1, 1);
            NSUInteger tgSize = MIN((NSUInteger)info.pso.maxTotalThreadsPerThreadgroup, (NSUInteger)op.m);
            if (tgSize > 32) tgSize = (tgSize / 32) * 32;

            [encoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        }

        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        // 拷贝所有结果回 CPU
        for (int i = 0; i < num_ops; i++) {
            memcpy(infos[i].y_cpu, [infos[i].y_buf contents], infos[i].y_bytes);
        }

        return true;
    }
}

} // namespace localllm
