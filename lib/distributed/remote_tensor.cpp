/*
 * tensorcore — remote tensor-fetch transport (POSIX TCP).
 *
 * Implements the inference-side weight-paging primitive. One TCP listener
 * per server endpoint, one worker thread per connected client, zero-copy
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
 *     u8   status          : 0 = OK, 1 = NOT_FOUND, 2 = SIZE_MISMATCH,
 *                            3 = INVALID_REQUEST
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
 *   - Worker threads hold a read lock through the socket write. This makes
 *     unregister a lifetime barrier: after it returns, no worker can still
 *     be reading from the caller-owned tensor pointer.
 *   - Fetches to one peer are serialized so concurrent callers cannot
 *     interleave request and response frames on the same TCP stream.
 *   - The diagnostics counters are std::atomic.
 *
 * Failure modes:
 *   - Bad header magic → server closes the connection. Client surfaces
 *     TC_ERR_INTERNAL.
 *   - Network drop mid-payload → both sides see read failure; client
 *     returns TC_ERR_INTERNAL and invalidates that peer connection.
 *   - Name miss → server sends OK header with status=NOT_FOUND, 0 bytes;
 *     client returns TC_ERR_INVALID_ARG.
 */

#include "tensorcore/remote_tensor.h"
#include "tensorcore/tensorcore.h"
#include "../core/internal.h"
#include "transport_auth_internal.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <memory>
#include <new>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
using tc_socket_t = SOCKET;
using tc_socklen_t = int;
#else
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <sys/uio.h>
#  include <unistd.h>
using tc_socket_t = int;
using tc_socklen_t = socklen_t;
#endif

namespace {

constexpr uint32_t kMagic = 0x54435254u;   /* 'TCRT' little-endian */
constexpr uint8_t  kVersion = 1;
constexpr uint8_t  kOpGet = 1;

constexpr uint8_t kStatusOk = 0;
constexpr uint8_t kStatusNotFound = 1;
constexpr uint8_t kStatusSizeMismatch = 2;
constexpr uint8_t kStatusInvalidRequest = 3;

constexpr size_t kDefaultIoTimeoutMs = 30000;
constexpr size_t kDefaultMaxClients = 64;

#if defined(_WIN32)
constexpr tc_socket_t kInvalidSocket = INVALID_SOCKET;

struct WinsockRuntime {
    bool ready = false;
    WinsockRuntime() {
        WSADATA data;
        ready = (::WSAStartup(MAKEWORD(2, 2), &data) == 0);
    }
    ~WinsockRuntime() {
        if (ready) ::WSACleanup();
    }
};

bool socket_runtime_ready() {
    static WinsockRuntime runtime;
    return runtime.ready;
}
#else
constexpr tc_socket_t kInvalidSocket = -1;
bool socket_runtime_ready() { return true; }
#endif

bool socket_valid(tc_socket_t fd) {
    return fd != kInvalidSocket;
}

int socket_last_error() {
#if defined(_WIN32)
    return ::WSAGetLastError();
#else
    return errno;
#endif
}

bool socket_interrupted(int error) {
#if defined(_WIN32)
    return error == WSAEINTR;
#else
    return error == EINTR;
#endif
}

bool socket_connect_retryable(int error) {
#if defined(_WIN32)
    return error == WSAECONNREFUSED || error == WSAEHOSTUNREACH ||
           error == WSAENETUNREACH || error == WSAETIMEDOUT;
#else
    return error == ECONNREFUSED || error == EHOSTUNREACH ||
           error == ENETUNREACH || error == ETIMEDOUT;
#endif
}

void socket_close(tc_socket_t fd) {
    if (!socket_valid(fd)) return;
#if defined(_WIN32)
    (void)::closesocket(fd);
#else
    (void)::close(fd);
#endif
}

void socket_shutdown(tc_socket_t fd) {
    if (!socket_valid(fd)) return;
#if defined(_WIN32)
    (void)::shutdown(fd, SD_BOTH);
#else
    (void)::shutdown(fd, SHUT_RDWR);
#endif
}

void socket_sleep_ms(unsigned milliseconds) {
#if defined(_WIN32)
    ::Sleep(static_cast<DWORD>(milliseconds));
#else
    ::usleep(static_cast<useconds_t>(milliseconds) * 1000u);
#endif
}

void socket_set_int_option(tc_socket_t fd, int level, int option, int value) {
#if defined(_WIN32)
    (void)::setsockopt(fd, level, option,
                       reinterpret_cast<const char*>(&value), sizeof(value));
#else
    (void)::setsockopt(fd, level, option, &value, sizeof(value));
#endif
}

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
    std::string port_text;
    if (!body.empty() && body.front() == '[') {
        const auto close = body.find(']');
        if (close == std::string::npos || close + 1 >= body.size() ||
            body[close + 1] != ':') return false;
        *host = body.substr(1, close - 1);
        port_text = body.substr(close + 2);
    } else {
        const auto colon = body.rfind(':');
        if (colon == std::string::npos) return false;
        *host = body.substr(0, colon);
        port_text = body.substr(colon + 1);
    }
    char* end = nullptr;
    const long p = std::strtol(port_text.c_str(), &end, 10);
    if (!end || *end != '\0' || p <= 0 || p > 65535) return false;
    *port = (uint16_t)p;
    return !host->empty();
}

size_t bounded_env_size(const char* name, size_t fallback,
                        size_t minimum, size_t maximum) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || !end || *end != '\0' || parsed < minimum || parsed > maximum)
        return fallback;
    return static_cast<size_t>(parsed);
}

void configure_connected_socket(tc_socket_t fd, size_t timeout_ms,
                                bool timeout_receive = true) {
    socket_set_int_option(fd, IPPROTO_TCP, TCP_NODELAY, 1);
#if defined(SO_NOSIGPIPE)
    socket_set_int_option(fd, SOL_SOCKET, SO_NOSIGPIPE, 1);
#endif
#if defined(_WIN32)
    const DWORD timeout = static_cast<DWORD>(timeout_ms);
    if (timeout_receive) {
        (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                           reinterpret_cast<const char*>(&timeout), sizeof(timeout));
    }
    (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO,
                       reinterpret_cast<const char*>(&timeout), sizeof(timeout));
#else
    timeval timeout = {};
    timeout.tv_sec = static_cast<time_t>(timeout_ms / 1000);
    timeout.tv_usec = static_cast<suseconds_t>((timeout_ms % 1000) * 1000);
    if (timeout_receive)
        (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#endif
}

bool write_all(tc_socket_t fd, const void* data, size_t bytes) {
    const uint8_t* p = (const uint8_t*)data;
    while (bytes > 0) {
        const size_t chunk = std::min(bytes, static_cast<size_t>(0x3fffffff));
#if defined(_WIN32)
        const int n = ::send(fd, reinterpret_cast<const char*>(p),
                             static_cast<int>(chunk), 0);
#if defined(MSG_NOSIGNAL)
        (void)0;
#endif
#else
        const ssize_t n = ::send(fd, p, chunk,
#  if defined(MSG_NOSIGNAL)
                                 MSG_NOSIGNAL
#  else
                                 0
#  endif
        );
#endif
        if (n <= 0) {
            if (socket_interrupted(socket_last_error())) continue;
            return false;
        }
        p += n; bytes -= (size_t)n;
    }
    return true;
}

bool read_all(tc_socket_t fd, void* data, size_t bytes) {
    uint8_t* p = (uint8_t*)data;
    while (bytes > 0) {
        const size_t chunk = std::min(bytes, static_cast<size_t>(0x3fffffff));
#if defined(_WIN32)
        const int n = ::recv(fd, reinterpret_cast<char*>(p), static_cast<int>(chunk), 0);
#else
        const ssize_t n = ::recv(fd, p, chunk, 0);
#endif
        if (n == 0) return false;
        if (n < 0) {
            if (socket_interrupted(socket_last_error())) continue;
            return false;
        }
        p += n; bytes -= (size_t)n;
    }
    return true;
}

tc_socket_t tcp_listen(const std::string& host, uint16_t port, size_t backlog) {
    if (!socket_runtime_ready()) return kInvalidSocket;
    addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = (host == "*") ? AI_PASSIVE : 0;
    addrinfo* res = nullptr;
    char port_str[16];
    std::snprintf(port_str, sizeof(port_str), "%u", port);
    const char* node = (host == "*") ? nullptr : host.c_str();
    if (::getaddrinfo(node, port_str, &hints, &res) != 0 || !res)
        return kInvalidSocket;

    tc_socket_t listen_fd = kInvalidSocket;
    for (addrinfo* ai = res; ai; ai = ai->ai_next) {
        tc_socket_t fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (!socket_valid(fd)) continue;
        socket_set_int_option(fd, SOL_SOCKET, SO_REUSEADDR, 1);
        if (::bind(fd, ai->ai_addr, static_cast<tc_socklen_t>(ai->ai_addrlen)) == 0 &&
            ::listen(fd, static_cast<int>(backlog)) == 0) {
            listen_fd = fd;
            break;
        }
        socket_close(fd);
    }
    ::freeaddrinfo(res);
    return listen_fd;
}

tc_socket_t tcp_connect(const std::string& host, uint16_t port, size_t timeout_ms) {
    if (!socket_runtime_ready()) return kInvalidSocket;
    addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    char port_str[16];
    std::snprintf(port_str, sizeof(port_str), "%u", port);
    if (::getaddrinfo(host.c_str(), port_str, &hints, &res) != 0 || !res)
        return kInvalidSocket;
    for (int retries = 30; retries >= 0; --retries) {
        bool retryable = false;
        for (addrinfo* ai = res; ai; ai = ai->ai_next) {
            tc_socket_t fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
            if (!socket_valid(fd)) continue;
            configure_connected_socket(fd, timeout_ms);
            if (::connect(fd, ai->ai_addr, static_cast<tc_socklen_t>(ai->ai_addrlen)) == 0) {
                ::freeaddrinfo(res);
                return fd;
            }
            const int saved = socket_last_error();
            socket_close(fd);
            retryable = retryable || socket_connect_retryable(saved);
        }
        if (retryable && retries > 0) {
            socket_sleep_ms(100);
            continue;
        }
        break;
    }
    ::freeaddrinfo(res);
    return kInvalidSocket;
}

struct RegisteredTensor {
    const void* ptr;
    size_t bytes;
};

struct ClientPeer {
    explicit ClientPeer(tc_socket_t socket_fd,
                        std::string authenticated_identity = {},
                        uint64_t authenticated_key_id = 0)
        : fd(socket_fd), identity(std::move(authenticated_identity)),
          key_id(authenticated_key_id) {}
    tc_socket_t fd;
    std::string identity;
    uint64_t key_id;
    std::mutex io_mutex;
};

}  // namespace

struct tc_remote_ctx {
    tc_remote_role_t           role;

    /* server side */
    std::atomic<tc_socket_t>   listen_fd{kInvalidSocket};
    std::thread                accept_thread;
    std::atomic<bool>          stop{false};
    std::mutex                 server_clients_mutex;
    std::condition_variable    server_clients_cv;
    std::unordered_set<tc_socket_t> server_client_fds;
    size_t                     max_clients = kDefaultMaxClients;
    size_t                     io_timeout_ms = kDefaultIoTimeoutMs;
    bool                       auth_required = false;
    std::mutex                 auth_mutex;
    tc_transport_auth_internal::Keyring auth_keyring;
    std::shared_mutex          registry_mutex;
    std::unordered_map<std::string, RegisteredTensor> registry;

    /* client side */
    std::mutex                 peers_mutex;
    std::vector<std::shared_ptr<ClientPeer>> peers; /* index = peer_id */

    /* diagnostics */
    std::atomic<uint64_t>      bytes_served{0};
    std::atomic<uint64_t>      bytes_fetched{0};
    std::atomic<uint64_t>      fetch_count{0};
    std::atomic<uint64_t>      auth_failures{0};

    tc_remote_ctx() = default;
};

namespace {

bool auth_read(void* user, void* data, size_t bytes) {
    return read_all(*static_cast<tc_socket_t*>(user), data, bytes);
}

bool auth_write(void* user, const void* data, size_t bytes) {
    return write_all(*static_cast<tc_socket_t*>(user), data, bytes);
}

void clear_receive_timeout(tc_socket_t fd) {
#if defined(_WIN32)
    const DWORD timeout = 0;
    (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                       reinterpret_cast<const char*>(&timeout), sizeof(timeout));
#else
    timeval timeout = {};
    (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#endif
}

void serve_one_client(tc_remote_ctx* h, tc_socket_t fd) {
    if (h->auth_required) {
        tc_transport_auth_internal::Keyring keyring;
        {
            std::lock_guard<std::mutex> lk(h->auth_mutex);
            keyring = h->auth_keyring;
        }
        tc_transport_auth_internal::Io io{&fd, auth_read, auth_write};
        tc_transport_auth_internal::PeerIdentity peer;
        configure_connected_socket(fd, h->io_timeout_ms, true);
        const tc_status_t auth_status = tc_transport_auth_internal::server_handshake(
            io, keyring, nullptr, "tensorcore/remote-tensor", nullptr, 0, &peer);
        clear_receive_timeout(fd);
        if (auth_status != TC_OK) {
            h->auth_failures.fetch_add(1, std::memory_order_relaxed);
            goto close_client;
        }
    }
    while (!h->stop.load(std::memory_order_acquire)) {
        WireRequest req;
        if (!read_all(fd, &req, sizeof(req))) break;
        if (req.magic != kMagic || req.version != kVersion || req.op != kOpGet) break;
        if (req.name_len == 0 || req.name_len > 255) break;
        char namebuf[256];
        if (!read_all(fd, namebuf, req.name_len)) break;
        namebuf[req.name_len] = '\0';

        WireResponse resp = {};
        resp.magic = kMagic;
        resp.version = kVersion;

        if (req.offset != 0) {
            resp.status = kStatusInvalidRequest;
            if (!write_all(fd, &resp, sizeof(resp))) break;
            continue;
        }

        std::shared_lock<std::shared_mutex> lk(h->registry_mutex);
        auto it = h->registry.find(std::string(namebuf, req.name_len));
        if (it == h->registry.end()) {
            resp.status = kStatusNotFound;
            resp.bytes = 0;
            if (!write_all(fd, &resp, sizeof(resp))) break;
            continue;
        }
        const RegisteredTensor tensor = it->second;
        if (static_cast<uint64_t>(tensor.bytes) != req.bytes) {
            resp.status = kStatusSizeMismatch;
            resp.bytes = tensor.bytes; /* report actual size to help diagnose */
            if (!write_all(fd, &resp, sizeof(resp))) break;
            continue;
        }
        resp.status = kStatusOk;
        resp.bytes = tensor.bytes;
        if (!write_all(fd, &resp, sizeof(resp))) break;
        /* Keep the shared lock until the zero-copy read is complete. */
        if (!write_all(fd, tensor.ptr, tensor.bytes)) break;
        h->bytes_served.fetch_add(tensor.bytes, std::memory_order_relaxed);
    }
close_client:
    {
        std::lock_guard<std::mutex> lk(h->server_clients_mutex);
        h->server_client_fds.erase(fd);
        socket_close(fd);
    }
    h->server_clients_cv.notify_all();
}

void accept_loop(tc_remote_ctx* h) {
    while (!h->stop.load(std::memory_order_acquire)) {
        sockaddr_storage peer = {};
        tc_socklen_t alen = sizeof(peer);
        const tc_socket_t listen_fd = h->listen_fd.load(std::memory_order_acquire);
        if (!socket_valid(listen_fd)) break;
        tc_socket_t cfd = ::accept(listen_fd, (sockaddr*)&peer, &alen);
        if (!socket_valid(cfd)) {
            if (h->stop.load()) break;
            if (socket_interrupted(socket_last_error())) continue;
            break;
        }
        if (h->stop.load(std::memory_order_acquire)) {
            socket_close(cfd);
            break;
        }
        /* A server connection may be intentionally idle during a long compute
         * phase. Bound writes to a stalled receiver, but let shutdown(2)
         * interrupt the header read instead of expiring healthy mesh peers. */
        configure_connected_socket(cfd, h->io_timeout_ms, false);
        {
            std::lock_guard<std::mutex> lk(h->server_clients_mutex);
            if (h->server_client_fds.size() >= h->max_clients) {
                socket_close(cfd);
                continue;
            }
            h->server_client_fds.insert(cfd);
        }
        try {
            std::thread(serve_one_client, h, cfd).detach();
        } catch (...) {
            {
                std::lock_guard<std::mutex> lk(h->server_clients_mutex);
                h->server_client_fds.erase(cfd);
            }
            socket_close(cfd);
            h->server_clients_cv.notify_all();
        }
    }
}

}  // namespace

tc_status_t remote_init_impl(tc_context* ctx, int role_value,
                             const char* bind_url,
                             const tc_transport_auth_config* auth,
                             tc_remote_ctx** out) {
    if (!ctx || !out) return TC_ERR_INVALID_ARG;
    *out = nullptr;
    if (role_value != TC_REMOTE_ROLE_WEIGHT_SERVER &&
        role_value != TC_REMOTE_ROLE_COMPUTE_CLIENT) return TC_ERR_INVALID_ARG;
    const auto role = static_cast<tc_remote_role_t>(role_value);
    (void)ctx;   /* unused for now; reserved for future shared-context tie-in */
    auto* h = new (std::nothrow) tc_remote_ctx();
    if (!h) return TC_ERR_ALLOC;
    h->role = role;
    if (auth) {
        const tc_status_t auth_status =
            tc_transport_auth_internal::Keyring::copy_from(auth, &h->auth_keyring);
        if (auth_status != TC_OK) { delete h; return auth_status; }
        h->auth_required = true;
    }
    h->io_timeout_ms = bounded_env_size("TC_REMOTE_IO_TIMEOUT_MS",
                                        kDefaultIoTimeoutMs, 100, 600000);
    h->max_clients = bounded_env_size("TC_REMOTE_MAX_CLIENTS",
                                      kDefaultMaxClients, 1, 4096);

    if (role == TC_REMOTE_ROLE_WEIGHT_SERVER) {
        std::string host;
        uint16_t port = 0;
        if (!parse_url(bind_url, &host, &port)) { delete h; return TC_ERR_INVALID_ARG; }
        h->listen_fd.store(tcp_listen(host, port, h->max_clients),
                           std::memory_order_release);
        if (!socket_valid(h->listen_fd.load(std::memory_order_acquire))) {
            delete h;
            return TC_ERR_INTERNAL;
        }
        try {
            h->accept_thread = std::thread(accept_loop, h);
        } catch (...) {
            const tc_socket_t fd = h->listen_fd.exchange(kInvalidSocket,
                                                         std::memory_order_acq_rel);
            socket_close(fd);
            delete h;
            return TC_ERR_INTERNAL;
        }
    }
    *out = h;
    return TC_OK;
}

extern "C" tc_status_t tc_remote_init(tc_context* ctx, tc_remote_role_t role,
                                       const char* bind_url, tc_remote_ctx** out) {
    static_assert(sizeof(tc_remote_role_t) == sizeof(int));
    int role_value = 0;
    std::memcpy(&role_value, &role, sizeof(role_value));
    return remote_init_impl(ctx, role_value, bind_url, nullptr, out);
}

extern "C" tc_status_t tc_remote_init_authenticated(
    tc_context* ctx, tc_remote_role_t role, const char* bind_url,
    const tc_transport_auth_config* auth, tc_remote_ctx** out) {
    if (!auth) {
        if (out) *out = nullptr;
        return TC_ERR_INVALID_ARG;
    }
    static_assert(sizeof(tc_remote_role_t) == sizeof(int));
    int role_value = 0;
    std::memcpy(&role_value, &role, sizeof(role_value));
    return remote_init_impl(ctx, role_value, bind_url, auth, out);
}

extern "C" tc_status_t tc_remote_shutdown(tc_remote_ctx* h) {
    if (!h) return TC_ERR_INVALID_ARG;
    h->stop.store(true, std::memory_order_release);
    const tc_socket_t listen_fd = h->listen_fd.exchange(kInvalidSocket,
                                                        std::memory_order_acq_rel);
    if (socket_valid(listen_fd)) {
        socket_shutdown(listen_fd);
        socket_close(listen_fd);
    }
    if (h->accept_thread.joinable()) h->accept_thread.join();
    {
        std::lock_guard<std::mutex> lk(h->server_clients_mutex);
        for (tc_socket_t fd : h->server_client_fds) socket_shutdown(fd);
    }
    {
        std::unique_lock<std::mutex> lk(h->server_clients_mutex);
        h->server_clients_cv.wait(lk, [&] { return h->server_client_fds.empty(); });
    }
    std::vector<std::shared_ptr<ClientPeer>> peers;
    {
        std::lock_guard<std::mutex> lk(h->peers_mutex);
        peers.swap(h->peers);
    }
    for (const auto& peer : peers) {
        std::lock_guard<std::mutex> lk(peer->io_mutex);
        if (socket_valid(peer->fd)) {
            socket_shutdown(peer->fd);
            socket_close(peer->fd);
            peer->fd = kInvalidSocket;
        }
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
    if (h->auth_required) return -1;
    std::string host; uint16_t port = 0;
    if (!parse_url(peer_url, &host, &port)) return -1;
    tc_socket_t fd = tcp_connect(host, port, h->io_timeout_ms);
    if (!socket_valid(fd)) return -1;
    try {
        auto peer = std::make_shared<ClientPeer>(fd);
        std::lock_guard<std::mutex> lk(h->peers_mutex);
        const int id = static_cast<int>(h->peers.size());
        h->peers.push_back(std::move(peer));
        return id;
    } catch (...) {
        socket_close(fd);
        return -1;
    }
}

extern "C" tc_status_t tc_remote_connect_authenticated(
    tc_remote_ctx* h, const char* peer_url,
    const char* expected_peer_identity, int* out_peer_id) {
    if (out_peer_id) *out_peer_id = -1;
    if (!h || h->role != TC_REMOTE_ROLE_COMPUTE_CLIENT || !peer_url ||
        !expected_peer_identity || !out_peer_id || !h->auth_required)
        return TC_ERR_INVALID_ARG;
    std::string host;
    uint16_t port = 0;
    if (!parse_url(peer_url, &host, &port)) return TC_ERR_INVALID_ARG;
    tc_socket_t fd = tcp_connect(host, port, h->io_timeout_ms);
    if (!socket_valid(fd)) return TC_ERR_INTERNAL;

    tc_transport_auth_internal::Keyring keyring;
    {
        std::lock_guard<std::mutex> lk(h->auth_mutex);
        keyring = h->auth_keyring;
    }
    tc_transport_auth_internal::Io io{&fd, auth_read, auth_write};
    tc_transport_auth_internal::PeerIdentity authenticated_peer;
    const tc_status_t auth_status = tc_transport_auth_internal::client_handshake(
        io, keyring, expected_peer_identity, "tensorcore/remote-tensor",
        nullptr, 0, &authenticated_peer);
    if (auth_status != TC_OK) {
        h->auth_failures.fetch_add(1, std::memory_order_relaxed);
        socket_shutdown(fd);
        socket_close(fd);
        return auth_status;
    }
    try {
        auto peer = std::make_shared<ClientPeer>(
            fd, authenticated_peer.identity, authenticated_peer.key_id);
        std::lock_guard<std::mutex> lk(h->peers_mutex);
        const int id = static_cast<int>(h->peers.size());
        h->peers.push_back(std::move(peer));
        *out_peer_id = id;
        return TC_OK;
    } catch (const std::bad_alloc&) {
        socket_close(fd);
        return TC_ERR_ALLOC;
    } catch (...) {
        socket_close(fd);
        return TC_ERR_INTERNAL;
    }
}

tc_status_t tc_remote_internal_reconnect(
    tc_remote_ctx* h, int peer_id, const char* peer_url,
    const char* expected_peer_identity) {
    if (!h || h->role != TC_REMOTE_ROLE_COMPUTE_CLIENT || !peer_url)
        return TC_ERR_INVALID_ARG;
    std::shared_ptr<ClientPeer> peer;
    {
        std::lock_guard<std::mutex> lk(h->peers_mutex);
        if (peer_id < 0 || static_cast<size_t>(peer_id) >= h->peers.size())
            return TC_ERR_INVALID_ARG;
        peer = h->peers[peer_id];
    }
    if (h->auth_required != (expected_peer_identity != nullptr))
        return TC_ERR_INVALID_ARG;
    std::string host;
    uint16_t port = 0;
    if (!parse_url(peer_url, &host, &port)) return TC_ERR_INVALID_ARG;
    tc_socket_t replacement = tcp_connect(host, port, h->io_timeout_ms);
    if (!socket_valid(replacement)) return TC_ERR_INTERNAL;

    std::string identity;
    uint64_t key_id = 0;
    if (h->auth_required) {
        tc_transport_auth_internal::Keyring keyring;
        {
            std::lock_guard<std::mutex> lk(h->auth_mutex);
            keyring = h->auth_keyring;
        }
        tc_transport_auth_internal::Io io{&replacement, auth_read, auth_write};
        tc_transport_auth_internal::PeerIdentity authenticated_peer;
        const tc_status_t auth_status =
            tc_transport_auth_internal::client_handshake(
                io, keyring, expected_peer_identity,
                "tensorcore/remote-tensor", nullptr, 0,
                &authenticated_peer);
        if (auth_status != TC_OK) {
            h->auth_failures.fetch_add(1, std::memory_order_relaxed);
            socket_shutdown(replacement);
            socket_close(replacement);
            return auth_status;
        }
        identity = authenticated_peer.identity;
        key_id = authenticated_peer.key_id;
    }

    std::lock_guard<std::mutex> io_lk(peer->io_mutex);
    if (socket_valid(peer->fd)) {
        socket_shutdown(peer->fd);
        socket_close(peer->fd);
    }
    peer->fd = replacement;
    peer->identity = std::move(identity);
    peer->key_id = key_id;
    return TC_OK;
}

extern "C" tc_status_t tc_remote_auth_rotate(
    tc_remote_ctx* h, const tc_transport_auth_config* auth) {
    if (!h || !auth || !h->auth_required) return TC_ERR_INVALID_ARG;
    tc_transport_auth_internal::Keyring replacement;
    const tc_status_t status =
        tc_transport_auth_internal::Keyring::copy_from(auth, &replacement);
    if (status != TC_OK) return status;
    std::lock_guard<std::mutex> lk(h->auth_mutex);
    if (replacement.local_identity() != h->auth_keyring.local_identity())
        return TC_ERR_INVALID_ARG;
    h->auth_keyring.swap(replacement);
    return TC_OK;
}

extern "C" const char* tc_remote_peer_identity(tc_remote_ctx* h, int peer_id) {
    if (!h || h->role != TC_REMOTE_ROLE_COMPUTE_CLIENT) return nullptr;
    std::lock_guard<std::mutex> lk(h->peers_mutex);
    if (peer_id < 0 || static_cast<size_t>(peer_id) >= h->peers.size() ||
        h->peers[peer_id]->identity.empty()) return nullptr;
    return h->peers[peer_id]->identity.c_str();
}

extern "C" uint64_t tc_remote_peer_key_id(tc_remote_ctx* h, int peer_id) {
    if (!h || h->role != TC_REMOTE_ROLE_COMPUTE_CLIENT) return 0;
    std::lock_guard<std::mutex> lk(h->peers_mutex);
    if (peer_id < 0 || static_cast<size_t>(peer_id) >= h->peers.size()) return 0;
    return h->peers[peer_id]->key_id;
}

extern "C" uint64_t tc_remote_auth_failure_count(tc_remote_ctx* h) {
    return h ? h->auth_failures.load(std::memory_order_relaxed) : 0;
}

extern "C" tc_status_t tc_remote_tensor_fetch(tc_remote_ctx* h, int peer_id,
                                                const char* name, void* dst,
                                                size_t bytes) {
    if (!h || h->role != TC_REMOTE_ROLE_COMPUTE_CLIENT || !name || !dst || bytes == 0)
        return TC_ERR_INVALID_ARG;
    std::shared_ptr<ClientPeer> peer;
    {
        std::lock_guard<std::mutex> lk(h->peers_mutex);
        if (peer_id < 0 || static_cast<size_t>(peer_id) >= h->peers.size())
            return TC_ERR_INVALID_ARG;
        peer = h->peers[peer_id];
    }
    std::lock_guard<std::mutex> io_lk(peer->io_mutex);
    tc_socket_t fd = peer->fd;
    if (!socket_valid(fd)) return TC_ERR_INTERNAL;

    const auto fail_connection = [&]() {
        if (socket_valid(peer->fd)) {
            socket_shutdown(peer->fd);
            socket_close(peer->fd);
            peer->fd = kInvalidSocket;
        }
        return TC_ERR_INTERNAL;
    };

    const size_t nlen = std::strlen(name);
    if (nlen == 0 || nlen > 255) return TC_ERR_INVALID_ARG;

    WireRequest req = {};
    req.magic = kMagic;
    req.version = kVersion;
    req.op = kOpGet;
    req.name_len = (uint16_t)nlen;
    req.offset = 0;
    req.bytes = (uint64_t)bytes;
    if (!write_all(fd, &req, sizeof(req))) return fail_connection();
    if (!write_all(fd, name, nlen))         return fail_connection();

    WireResponse resp = {};
    if (!read_all(fd, &resp, sizeof(resp))) return fail_connection();
    if (resp.magic != kMagic || resp.version != kVersion) return fail_connection();
    if (resp.status == kStatusNotFound)      return TC_ERR_INVALID_ARG;
    if (resp.status == kStatusSizeMismatch)  return TC_ERR_INVALID_ARG;
    if (resp.status != kStatusOk)            return fail_connection();
    if (resp.bytes != bytes)                  return fail_connection();

    if (!read_all(fd, dst, bytes)) return fail_connection();
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

/* Private mesh lifecycle hook. After all ranks publish their shutdown-ready
 * token and close outbound clients, each server drains accepted connections
 * naturally before the endpoint is destroyed. */
tc_status_t tc_remote_internal_wait_for_clients_closed(
    tc_remote_ctx* h, uint32_t timeout_ms) {
    if (!h || h->role != TC_REMOTE_ROLE_WEIGHT_SERVER || timeout_ms == 0)
        return TC_ERR_INVALID_ARG;
    std::unique_lock<std::mutex> lk(h->server_clients_mutex);
    const bool drained = h->server_clients_cv.wait_for(
        lk, std::chrono::milliseconds(timeout_ms),
        [&] { return h->server_client_fds.empty(); });
    return drained ? TC_OK : TC_ERR_INTERNAL;
}
