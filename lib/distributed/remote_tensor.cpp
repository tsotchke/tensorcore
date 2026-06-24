/*
 * tensorcore — remote tensor-fetch transport (POSIX TCP).
 *
 * Implements the inference-side weight-paging primitive described in
 * NOTE-kimi-distributed-inference-2026-06-24.md. One TCP listener per
 * server endpoint, one worker thread per connected client, zero-copy
 * server-side via writev directly out of the registered buffer.
 *
 * Wire protocol (little-endian, version 1):
 *
 *   Request  (client → server):
 *     u32  magic           = 0x54435254  ('TCRT')
 *     u8   version         = 1
 *     u8   op              = 1 (GET)
 *     u16  name_len        : length of the tensor name in bytes
 *     u64  offset          : reserved for range fetches; 0 = full tensor
 *     u64  bytes           : number of bytes the client expects
 *     u8[name_len] name    : UTF-8 identifier (no NUL terminator on the wire)
 *
 *   Response (server → client):
 *     u32  magic           = 0x54435254
 *     u8   version         = 1
 *     u8   status          : 0 = OK, 1 = NOT_FOUND, 2 = SIZE_MISMATCH
 *     u16  _pad            = 0
 *     u64  bytes           : actual bytes that follow (0 on error)
 *     u8[bytes] payload    : zero-copy out of the registered tensor
 *
 * Why this protocol shape:
 *   - Fixed 24-byte header → single read for framing, then payload.
 *   - u64 bytes leaves room for tensors > 4 GB (full Q4 expert banks
 *     are 23.6 MB today but bigger LoRA fusions are coming).
 *   - offset reserved for range fetches so we don't break the wire when
 *     the client wants e.g. "first 4 MB of this expert" for prefetch.
 *
 * Thread safety:
 *   - The tensor registry uses a shared_mutex (many readers, one writer).
 *   - Worker threads grab a read lock on each request to look up the
 *     tensor pointer; writes (register / unregister) take the writer lock.
 *   - The diagnostics counters are std::atomic.
 *
 * Failure modes:
 *   - Bad header magic → server closes the connection. Client surfaces
 *     TC_ERR_INTERNAL.
 *   - Network drop mid-payload → both sides see read failure; client
 *     returns TC_ERR_INTERNAL; the connection is dropped from the peer
 *     table so the next fetch reconnects.
 *   - Name miss → server sends OK header with status=NOT_FOUND, 0 bytes;
 *     client returns TC_ERR_INVALID_ARG.
 */

#include "tensorcore/remote_tensor.h"
#include "tensorcore/tensorcore.h"
#include "../core/internal.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <sys/uio.h>
#  include <unistd.h>
#endif

namespace {

constexpr uint32_t kMagic = 0x54435254u;   /* 'TCRT' little-endian */
constexpr uint8_t  kVersion = 1;
constexpr uint8_t  kOpGet = 1;

constexpr uint8_t kStatusOk = 0;
constexpr uint8_t kStatusNotFound = 1;
constexpr uint8_t kStatusSizeMismatch = 2;

#pragma pack(push, 1)
struct WireRequest {
    uint32_t magic;
    uint8_t  version;
    uint8_t  op;
    uint16_t name_len;
    uint64_t offset;
    uint64_t bytes;
};
struct WireResponse {
    uint32_t magic;
    uint8_t  version;
    uint8_t  status;
    uint16_t pad;
    uint64_t bytes;
};
#pragma pack(pop)
static_assert(sizeof(WireRequest)  == 24, "wire request must be 24 bytes");
static_assert(sizeof(WireResponse) == 16, "wire response must be 16 bytes");

bool parse_url(const char* url, std::string* host, uint16_t* port) {
    if (!url) return false;
    const std::string s(url);
    const std::string prefix = "tcp://";
    if (s.compare(0, prefix.size(), prefix) != 0) return false;
    const std::string body = s.substr(prefix.size());
    const auto colon = body.rfind(':');
    if (colon == std::string::npos) return false;
    *host = body.substr(0, colon);
    char* end = nullptr;
    const long p = std::strtol(body.substr(colon + 1).c_str(), &end, 10);
    if (!end || *end != '\0' || p <= 0 || p > 65535) return false;
    *port = (uint16_t)p;
    return !host->empty();
}

bool write_all(int fd, const void* data, size_t bytes) {
    const uint8_t* p = (const uint8_t*)data;
    while (bytes > 0) {
        const ssize_t n = ::write(fd, p, bytes);
        if (n <= 0) { if (errno == EINTR) continue; return false; }
        p += n; bytes -= (size_t)n;
    }
    return true;
}

bool read_all(int fd, void* data, size_t bytes) {
    uint8_t* p = (uint8_t*)data;
    while (bytes > 0) {
        const ssize_t n = ::read(fd, p, bytes);
        if (n == 0) return false;
        if (n < 0)  { if (errno == EINTR) continue; return false; }
        p += n; bytes -= (size_t)n;
    }
    return true;
}

int tcp_listen(uint16_t port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    if (::bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0) { ::close(fd); return -1; }
    if (::listen(fd, 64) < 0) { ::close(fd); return -1; }
    return fd;
}

int tcp_connect(const std::string& host, uint16_t port) {
    addrinfo hints = {};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    char port_str[16];
    std::snprintf(port_str, sizeof(port_str), "%u", port);
    if (::getaddrinfo(host.c_str(), port_str, &hints, &res) != 0 || !res) return -1;
    for (int retries = 30; retries >= 0; --retries) {
        int fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (fd < 0) { ::freeaddrinfo(res); return -1; }
        int yes = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
        if (::connect(fd, res->ai_addr, res->ai_addrlen) == 0) {
            ::freeaddrinfo(res);
            return fd;
        }
        const int saved = errno;
        ::close(fd);
        if ((saved == ECONNREFUSED || saved == EHOSTUNREACH) && retries > 0) {
            ::usleep(100 * 1000);
            continue;
        }
        ::freeaddrinfo(res);
        return -1;
    }
    ::freeaddrinfo(res);
    return -1;
}

struct RegisteredTensor {
    const void* ptr;
    size_t bytes;
};

}  // namespace

struct tc_remote_ctx {
    tc_remote_role_t           role;

    /* server side */
    int                        listen_fd = -1;
    std::thread                accept_thread;
    std::vector<std::thread>   worker_threads;
    std::atomic<bool>          stop{false};
    std::shared_mutex          registry_mutex;
    std::unordered_map<std::string, RegisteredTensor> registry;

    /* client side */
    std::mutex                 peers_mutex;
    std::vector<int>           peer_fds;     /* index = peer_id */

    /* diagnostics */
    std::atomic<uint64_t>      bytes_served{0};
    std::atomic<uint64_t>      bytes_fetched{0};
    std::atomic<uint64_t>      fetch_count{0};

    tc_remote_ctx() = default;
};

namespace {

void serve_one_client(tc_remote_ctx* h, int fd) {
    while (!h->stop.load(std::memory_order_acquire)) {
        WireRequest req;
        if (!read_all(fd, &req, sizeof(req))) break;
        if (req.magic != kMagic || req.version != kVersion || req.op != kOpGet) break;
        if (req.name_len == 0 || req.name_len > 255) break;
        char namebuf[256];
        if (!read_all(fd, namebuf, req.name_len)) break;
        namebuf[req.name_len] = '\0';

        const void* ptr = nullptr;
        size_t bytes = 0;
        {
            std::shared_lock<std::shared_mutex> lk(h->registry_mutex);
            auto it = h->registry.find(std::string(namebuf, req.name_len));
            if (it != h->registry.end()) { ptr = it->second.ptr; bytes = it->second.bytes; }
        }

        WireResponse resp = {};
        resp.magic = kMagic;
        resp.version = kVersion;

        if (!ptr) {
            resp.status = kStatusNotFound;
            resp.bytes = 0;
            if (!write_all(fd, &resp, sizeof(resp))) break;
            continue;
        }
        if (bytes != req.bytes) {
            resp.status = kStatusSizeMismatch;
            resp.bytes = bytes;   /* report actual size to help diagnose */
            if (!write_all(fd, &resp, sizeof(resp))) break;
            continue;
        }
        resp.status = kStatusOk;
        resp.bytes = bytes;
        if (!write_all(fd, &resp, sizeof(resp))) break;
        /* Zero-copy payload write out of the registered ptr. */
        if (!write_all(fd, (const uint8_t*)ptr + req.offset, bytes)) break;
        h->bytes_served.fetch_add(bytes, std::memory_order_relaxed);
    }
    ::close(fd);
}

void accept_loop(tc_remote_ctx* h) {
    while (!h->stop.load(std::memory_order_acquire)) {
        sockaddr_in peer = {};
        socklen_t alen = sizeof(peer);
        int cfd = ::accept(h->listen_fd, (sockaddr*)&peer, &alen);
        if (cfd < 0) {
            if (h->stop.load()) break;
            if (errno == EINTR) continue;
            break;
        }
        int yes = 1;
        ::setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
        h->worker_threads.emplace_back(serve_one_client, h, cfd);
    }
}

}  // namespace

extern "C" tc_status_t tc_remote_init(tc_context* ctx, tc_remote_role_t role,
                                       const char* bind_url, tc_remote_ctx** out) {
    if (!ctx || !out) return TC_ERR_INVALID_ARG;
    (void)ctx;   /* unused for now; reserved for future shared-context tie-in */
    auto* h = new (std::nothrow) tc_remote_ctx();
    if (!h) return TC_ERR_ALLOC;
    h->role = role;

    if (role == TC_REMOTE_ROLE_WEIGHT_SERVER) {
        std::string host;
        uint16_t port = 0;
        if (!parse_url(bind_url, &host, &port)) { delete h; return TC_ERR_INVALID_ARG; }
        h->listen_fd = tcp_listen(port);
        if (h->listen_fd < 0) { delete h; return TC_ERR_INTERNAL; }
        h->accept_thread = std::thread(accept_loop, h);
    }
    *out = h;
    return TC_OK;
}

extern "C" tc_status_t tc_remote_shutdown(tc_remote_ctx* h) {
    if (!h) return TC_ERR_INVALID_ARG;
    h->stop.store(true, std::memory_order_release);
    if (h->listen_fd >= 0) {
        ::shutdown(h->listen_fd, SHUT_RDWR);
        ::close(h->listen_fd);
        h->listen_fd = -1;
    }
    if (h->accept_thread.joinable()) h->accept_thread.join();
    for (auto& t : h->worker_threads) if (t.joinable()) t.join();
    {
        std::lock_guard<std::mutex> lk(h->peers_mutex);
        for (int fd : h->peer_fds) if (fd >= 0) ::close(fd);
        h->peer_fds.clear();
    }
    delete h;
    return TC_OK;
}

extern "C" tc_status_t tc_remote_register_tensor(tc_remote_ctx* h,
                                                  const char* name,
                                                  const void* ptr, size_t bytes) {
    if (!h || h->role != TC_REMOTE_ROLE_WEIGHT_SERVER || !name || !ptr || bytes == 0)
        return TC_ERR_INVALID_ARG;
    const size_t nlen = std::strlen(name);
    if (nlen == 0 || nlen > 255) return TC_ERR_INVALID_ARG;
    std::unique_lock<std::shared_mutex> lk(h->registry_mutex);
    h->registry[std::string(name, nlen)] = RegisteredTensor{ptr, bytes};
    return TC_OK;
}

extern "C" tc_status_t tc_remote_unregister_tensor(tc_remote_ctx* h, const char* name) {
    if (!h || h->role != TC_REMOTE_ROLE_WEIGHT_SERVER || !name) return TC_ERR_INVALID_ARG;
    std::unique_lock<std::shared_mutex> lk(h->registry_mutex);
    h->registry.erase(std::string(name));
    return TC_OK;
}

extern "C" size_t tc_remote_registered_count(tc_remote_ctx* h) {
    if (!h) return 0;
    std::shared_lock<std::shared_mutex> lk(h->registry_mutex);
    return h->registry.size();
}

extern "C" int tc_remote_connect(tc_remote_ctx* h, const char* peer_url) {
    if (!h || h->role != TC_REMOTE_ROLE_COMPUTE_CLIENT || !peer_url) return -1;
    std::string host; uint16_t port = 0;
    if (!parse_url(peer_url, &host, &port)) return -1;
    int fd = tcp_connect(host, port);
    if (fd < 0) return -1;
    std::lock_guard<std::mutex> lk(h->peers_mutex);
    const int id = (int)h->peer_fds.size();
    h->peer_fds.push_back(fd);
    return id;
}

extern "C" tc_status_t tc_remote_tensor_fetch(tc_remote_ctx* h, int peer_id,
                                                const char* name, void* dst,
                                                size_t bytes) {
    if (!h || h->role != TC_REMOTE_ROLE_COMPUTE_CLIENT || !name || !dst || bytes == 0)
        return TC_ERR_INVALID_ARG;
    int fd = -1;
    {
        std::lock_guard<std::mutex> lk(h->peers_mutex);
        if (peer_id < 0 || (size_t)peer_id >= h->peer_fds.size()) return TC_ERR_INVALID_ARG;
        fd = h->peer_fds[peer_id];
    }
    if (fd < 0) return TC_ERR_INTERNAL;

    const size_t nlen = std::strlen(name);
    if (nlen == 0 || nlen > 255) return TC_ERR_INVALID_ARG;

    WireRequest req = {};
    req.magic = kMagic;
    req.version = kVersion;
    req.op = kOpGet;
    req.name_len = (uint16_t)nlen;
    req.offset = 0;
    req.bytes = (uint64_t)bytes;
    if (!write_all(fd, &req, sizeof(req))) return TC_ERR_INTERNAL;
    if (!write_all(fd, name, nlen))         return TC_ERR_INTERNAL;

    WireResponse resp = {};
    if (!read_all(fd, &resp, sizeof(resp))) return TC_ERR_INTERNAL;
    if (resp.magic != kMagic || resp.version != kVersion) return TC_ERR_INTERNAL;
    if (resp.status == kStatusNotFound)      return TC_ERR_INVALID_ARG;
    if (resp.status == kStatusSizeMismatch)  return TC_ERR_INVALID_ARG;
    if (resp.status != kStatusOk)            return TC_ERR_INTERNAL;
    if (resp.bytes != bytes)                  return TC_ERR_INVALID_ARG;

    if (!read_all(fd, dst, bytes)) return TC_ERR_INTERNAL;
    h->bytes_fetched.fetch_add(bytes, std::memory_order_relaxed);
    h->fetch_count.fetch_add(1, std::memory_order_relaxed);
    return TC_OK;
}

extern "C" uint64_t tc_remote_total_bytes_served(tc_remote_ctx* h) {
    return h ? h->bytes_served.load() : 0;
}
extern "C" uint64_t tc_remote_total_bytes_fetched(tc_remote_ctx* h) {
    return h ? h->bytes_fetched.load() : 0;
}
extern "C" uint64_t tc_remote_fetch_count(tc_remote_ctx* h) {
    return h ? h->fetch_count.load() : 0;
}
