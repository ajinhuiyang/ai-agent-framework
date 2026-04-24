// Cross-platform implementations for mmap and sockets.
// Compiles on macOS, Linux, and Windows.

#include "platform.h"
#include <iostream>

#if LOCALLLM_POSIX
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <cerrno>
    #include <csignal>
#elif LOCALLLM_WINDOWS
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
#endif

namespace localllm {
namespace platform {

// ======================== mmap ========================

#if LOCALLLM_POSIX

bool mmap_file(const std::string& path, MappedFile& mf) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "[platform] Failed to open file: " << path << std::endl;
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return false;
    }
    mf.size = static_cast<size_t>(st.st_size);
    mf.addr = mmap(nullptr, mf.size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mf.addr == MAP_FAILED) {
        mf.addr = nullptr;
        mf.size = 0;
        return false;
    }
    return true;
}

void munmap_file(MappedFile& mf) {
    if (mf.addr && mf.size > 0) {
        munmap(mf.addr, mf.size);
        mf.addr = nullptr;
        mf.size = 0;
    }
}

void madvise_sequential(MappedFile& mf) {
    if (mf.addr && mf.size > 0) {
        madvise(mf.addr, mf.size, MADV_SEQUENTIAL);
    }
}

#elif LOCALLLM_WINDOWS

bool mmap_file(const std::string& path, MappedFile& mf) {
    HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        std::cerr << "[platform] Failed to open file: " << path << std::endl;
        return false;
    }

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        CloseHandle(hFile);
        return false;
    }
    mf.size = static_cast<size_t>(fileSize.QuadPart);

    HANDLE hMapping = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMapping) {
        CloseHandle(hFile);
        return false;
    }

    mf.addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!mf.addr) {
        CloseHandle(hMapping);
        CloseHandle(hFile);
        mf.size = 0;
        return false;
    }

    mf.file_handle = hFile;
    mf.mapping_handle = hMapping;
    return true;
}

void munmap_file(MappedFile& mf) {
    if (mf.addr) UnmapViewOfFile(mf.addr);
    if (mf.mapping_handle) CloseHandle(static_cast<HANDLE>(mf.mapping_handle));
    if (mf.file_handle) CloseHandle(static_cast<HANDLE>(mf.file_handle));
    mf.addr = nullptr;
    mf.size = 0;
    mf.file_handle = nullptr;
    mf.mapping_handle = nullptr;
}

void madvise_sequential(MappedFile& /*mf*/) {
    // No equivalent on Windows; the OS handles prefetching.
}

#endif

// ======================== sockets ========================

#if LOCALLLM_POSIX

bool socket_init() {
    // Ignore SIGPIPE so write() on closed socket returns error instead of killing process
    signal(SIGPIPE, SIG_IGN);
    return true;
}

void socket_cleanup() {
    // Nothing to do on POSIX
}

socket_t socket_listen(int port, int backlog) {
    socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCK) return INVALID_SOCK;

    socket_set_reuseaddr(sock);

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (bind(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(sock);
        return INVALID_SOCK;
    }

    if (listen(sock, backlog) < 0) {
        close(sock);
        return INVALID_SOCK;
    }

    return sock;
}

socket_t socket_accept(socket_t server_sock) {
    struct sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    return accept(server_sock, reinterpret_cast<struct sockaddr*>(&client_addr), &client_len);
}

int socket_recv(socket_t sock, char* buf, int len) {
    return static_cast<int>(read(sock, buf, len));
}

int socket_send(socket_t sock, const char* buf, int len) {
    return static_cast<int>(write(sock, buf, len));
}

void socket_close(socket_t sock) {
    close(sock);
}

void socket_set_reuseaddr(socket_t sock) {
    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
}

#elif LOCALLLM_WINDOWS

bool socket_init() {
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
}

void socket_cleanup() {
    WSACleanup();
}

socket_t socket_listen(int port, int backlog) {
    socket_t sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) return INVALID_SOCK;

    socket_set_reuseaddr(sock);

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<u_short>(port));

    if (bind(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        closesocket(sock);
        return INVALID_SOCK;
    }

    if (::listen(sock, backlog) == SOCKET_ERROR) {
        closesocket(sock);
        return INVALID_SOCK;
    }

    return sock;
}

socket_t socket_accept(socket_t server_sock) {
    struct sockaddr_in client_addr{};
    int client_len = sizeof(client_addr);
    return ::accept(server_sock, reinterpret_cast<struct sockaddr*>(&client_addr), &client_len);
}

int socket_recv(socket_t sock, char* buf, int len) {
    return ::recv(sock, buf, len, 0);
}

int socket_send(socket_t sock, const char* buf, int len) {
    return ::send(sock, buf, len, 0);
}

void socket_close(socket_t sock) {
    closesocket(sock);
}

void socket_set_reuseaddr(socket_t sock) {
    char opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
}

#endif

} // namespace platform
} // namespace localllm
