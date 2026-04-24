#pragma once

// Cross-platform abstractions for Local-LLM
// Provides unified API for: mmap, sockets, threading
// Supports: macOS, Linux, Windows

#include <cstddef>
#include <cstdint>
#include <string>

// ======================== Platform detection ========================

#if defined(_WIN32) || defined(_WIN64)
    #define LOCALLLM_WINDOWS 1
    #define LOCALLLM_POSIX   0
#elif defined(__APPLE__) || defined(__linux__)
    #define LOCALLLM_WINDOWS 0
    #define LOCALLLM_POSIX   1
#else
    #error "Unsupported platform"
#endif

// ======================== Memory-mapped file ========================

namespace localllm {
namespace platform {

struct MappedFile {
    void*  addr = nullptr;
    size_t size = 0;

#if LOCALLLM_WINDOWS
    void* file_handle = nullptr;   // HANDLE
    void* mapping_handle = nullptr; // HANDLE
#endif
};

// Map a file into memory (read-only). Returns true on success.
bool mmap_file(const std::string& path, MappedFile& mf);

// Unmap a previously mapped file.
void munmap_file(MappedFile& mf);

// Advise the OS about access patterns (best-effort, no-op on unsupported platforms).
void madvise_sequential(MappedFile& mf);

// ======================== Socket abstraction ========================

#if LOCALLLM_WINDOWS
using socket_t = uintptr_t; // SOCKET is UINT_PTR on Windows
constexpr socket_t INVALID_SOCK = ~((socket_t)0); // INVALID_SOCKET
#else
using socket_t = int;
constexpr socket_t INVALID_SOCK = -1;
#endif

// Must be called before any socket operations (no-op on POSIX).
bool socket_init();

// Cleanup socket library (no-op on POSIX).
void socket_cleanup();

// Create a TCP server socket, bind and listen.
socket_t socket_listen(int port, int backlog = 128);

// Accept a connection. Returns client socket.
socket_t socket_accept(socket_t server_sock);

// Read up to `len` bytes. Returns bytes read, or -1 on error.
int socket_recv(socket_t sock, char* buf, int len);

// Write `len` bytes. Returns bytes written, or -1 on error.
int socket_send(socket_t sock, const char* buf, int len);

// Close a socket.
void socket_close(socket_t sock);

// Set socket option SO_REUSEADDR.
void socket_set_reuseaddr(socket_t sock);

} // namespace platform
} // namespace localllm
