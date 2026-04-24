cmake_minimum_required(VERSION 3.16)
project(LocalLLM VERSION 1.0.0 LANGUAGES CXX)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# 默认 Release 构建
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# macOS 特定: 启用 Accelerate framework (可选 BLAS 加速)
if(APPLE)
    option(USE_ACCELERATE "Use Apple Accelerate framework for BLAS" ON)
    if(USE_ACCELERATE)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if(ACCELERATE_FRAMEWORK)
            add_definitions(-DUSE_ACCELERATE -DACCELERATE_NEW_LAPACK)
            message(STATUS "Using Apple Accelerate framework")
        endif()
    endif()

    # Metal GPU 加速 (自动检测)
    option(USE_METAL "Use Apple Metal for GPU acceleration" ON)
    if(USE_METAL)
        find_library(METAL_FRAMEWORK Metal)
        find_library(FOUNDATION_FRAMEWORK Foundation)
        if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
            enable_language(OBJCXX)
            add_definitions(-DUSE_METAL)
            message(STATUS "Metal GPU acceleration: ENABLED")
        else()
            set(USE_METAL OFF)
            message(STATUS "Metal GPU acceleration: DISABLED (framework not found)")
        endif()
    endif()
endif()

# 线程库
find_package(Threads REQUIRED)

# 头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 源文件
set(SOURCES
    src/gguf_parser.cpp
    src/dequantize.cpp
    src/tokenizer.cpp
    src/transformer.cpp
    src/http_server.cpp
    src/platform.cpp
)

# Metal 源文件
if(APPLE AND USE_METAL)
    list(APPEND SOURCES src/metal_compute.mm)
endif()

# 静态库
add_library(localllm_lib STATIC ${SOURCES})
target_include_directories(localllm_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(localllm_lib PUBLIC Threads::Threads)

if(APPLE AND ACCELERATE_FRAMEWORK)
    target_link_libraries(localllm_lib PUBLIC ${ACCELERATE_FRAMEWORK})
endif()

if(APPLE AND USE_METAL)
    target_link_libraries(localllm_lib PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    # Metal shader 路径 (运行时从源文件编译)
    target_compile_definitions(localllm_lib PUBLIC
        METAL_SHADER_SOURCE="${CMAKE_SOURCE_DIR}/src/kernels.metal"
    )
endif()

# 可执行文件
add_executable(localllm src/main.cpp)
target_link_libraries(localllm PRIVATE localllm_lib)

# 安装
install(TARGETS localllm RUNTIME DESTINATION bin)

# 打印构建信息
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
