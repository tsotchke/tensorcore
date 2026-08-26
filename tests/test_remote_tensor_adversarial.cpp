/* Adversarial lifecycle and framing coverage for tc_remote_* POSIX TCP. */

#include "tensorcore/remote_tensor.h"
#include "tensorcore/tensorcore.h"
#include "../lib/distributed/remote_tensor_internal.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t kMagic = 0x54435254u;
constexpr uint8_t kVersion = 1;
constexpr uint8_t kOpGet = 1;
constexpr uint8_t kStatusInvalidRequest = 3;
constexpr size_t kTensorBytes = 64 * 1024;
constexpr size_t kLifetimeBytes = 32 * 1024 * 1024;

#pragma pack(push, 1)
struct WireRequest {
    uint32_t magic;
    uint8_t version;
    uint8_t op;
    uint16_t name_len;
    uint64_t offset;
    uint64_t bytes;
};

struct WireResponse {
    uint32_t magic;
    uint8_t version;
    uint8_t status;
    uint16_t pad;
    uint64_t bytes;
};
#pragma pack(pop)

bool send_all(int fd, const void* data, size_t bytes) {
    auto* p = static_cast<const uint8_t*>(data);
    while (bytes != 0) {
        const ssize_t n = ::send(fd, p, bytes, 0);
        if (n <= 0) return false;
        p += n;
        bytes -= static_cast<size_t>(n);
    }
    return true;
}

bool recv_all(int fd, void* data, size_t bytes) {
    auto* p = static_cast<uint8_t*>(data);
    while (bytes != 0) {
        const ssize_t n = ::recv(fd, p, bytes, 0);
        if (n <= 0) return false;
        p += n;
        bytes -= static_cast<size_t>(n);
    }
    return true;
}

uint16_t unused_loopback_port() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 0;
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return 0;
    }
    socklen_t len = sizeof(addr);
    const bool ok = ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0;
    ::close(fd);
    return ok ? ntohs(addr.sin_port) : 0;
}

int connect_loopback(uint16_t port, int receive_buffer = 0) {
    for (int attempt = 0; attempt != 30; ++attempt) {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return -1;
        if (receive_buffer > 0)
            (void)::setsockopt(fd, SOL_SOCKET, SO_RCVBUF,
                               &receive_buffer, sizeof(receive_buffer));
        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(port);
        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0)
            return fd;
        ::close(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return -1;
}

bool check_invalid_offset(uint16_t port) {
    const int fd = connect_loopback(port);
    if (fd < 0) return false;
    constexpr char name[] = "adversarial/tensor";
    WireRequest req = {};
    req.magic = kMagic;
    req.version = kVersion;
    req.op = kOpGet;
    req.name_len = sizeof(name) - 1;
    req.offset = 1;  // v1 only permits full-tensor fetches at offset zero.
    req.bytes = kTensorBytes;
    WireResponse resp = {};
    const bool ok = send_all(fd, &req, sizeof(req)) &&
                    send_all(fd, name, sizeof(name) - 1) &&
                    recv_all(fd, &resp, sizeof(resp)) &&
                    resp.magic == kMagic && resp.version == kVersion &&
                    resp.status == kStatusInvalidRequest && resp.bytes == 0;
    ::close(fd);
    return ok;
}

bool check_unregister_barrier(tc_remote_ctx* server, uint16_t port) {
    constexpr char name[] = "lifetime/tensor";
    std::vector<uint8_t> source(kLifetimeBytes, 0xA5);
    if (tc_remote_register_tensor(server, name, source.data(), source.size()) != TC_OK)
        return false;

    const int fd = connect_loopback(port, 4096);
    if (fd < 0) return false;
    WireRequest req = {};
    req.magic = kMagic;
    req.version = kVersion;
    req.op = kOpGet;
    req.name_len = sizeof(name) - 1;
    req.bytes = source.size();
    WireResponse resp = {};
    if (!send_all(fd, &req, sizeof(req)) ||
        !send_all(fd, name, sizeof(name) - 1) ||
        !recv_all(fd, &resp, sizeof(resp)) || resp.status != 0 ||
        resp.bytes != source.size()) {
        ::close(fd);
        return false;
    }

    std::atomic<bool> unregister_done{false};
    tc_status_t unregister_status = TC_ERR_INTERNAL;
    std::thread unregister_thread([&] {
        unregister_status = tc_remote_unregister_tensor(server, name);
        unregister_done.store(true, std::memory_order_release);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const bool was_blocked = !unregister_done.load(std::memory_order_acquire);

    std::vector<uint8_t> sink(kLifetimeBytes);
    const bool payload_ok = recv_all(fd, sink.data(), sink.size()) &&
                            sink.front() == 0xA5 && sink.back() == 0xA5;
    unregister_thread.join();
    ::close(fd);
    return was_blocked && payload_ok && unregister_status == TC_OK;
}

}  // namespace

int main() {
    bool ok = true;
    tc_context* ctx = nullptr;
    if (tc_init(&ctx) != TC_OK) return 1;

    const uint16_t port = unused_loopback_port();
    char server_url[64];
    std::snprintf(server_url, sizeof(server_url), "tcp://127.0.0.1:%u", port);
    tc_remote_ctx* server = nullptr;
    if (port == 0 || tc_remote_init(ctx, TC_REMOTE_ROLE_WEIGHT_SERVER,
                                    server_url, &server) != TC_OK) {
        tc_shutdown(ctx);
        return 1;
    }

    tc_remote_ctx* invalid_role_ctx = nullptr;
    const bool invalid_role_ok =
        tc_remote_init(ctx, static_cast<tc_remote_role_t>(99), nullptr,
                       &invalid_role_ctx) == TC_ERR_INVALID_ARG &&
        invalid_role_ctx == nullptr;
    std::printf("  invalid role rejected:        %s\n",
                invalid_role_ok ? "OK" : "FAIL");
    ok &= invalid_role_ok;

    const uint16_t foreign_port = unused_loopback_port();
    char foreign_url[64];
    std::snprintf(foreign_url, sizeof(foreign_url),
                  "tcp://203.0.113.254:%u", foreign_port);
    tc_remote_ctx* foreign_server = nullptr;
    const bool exact_bind_ok = foreign_port != 0 &&
        tc_remote_init(ctx, TC_REMOTE_ROLE_WEIGHT_SERVER, foreign_url,
                       &foreign_server) == TC_ERR_INTERNAL &&
        foreign_server == nullptr;
    if (foreign_server) (void)tc_remote_shutdown(foreign_server);
    std::printf("  bind host is not widened:     %s\n",
                exact_bind_ok ? "OK" : "FAIL");
    ok &= exact_bind_ok;

    std::vector<uint8_t> source(kTensorBytes);
    for (size_t i = 0; i != source.size(); ++i)
        source[i] = static_cast<uint8_t>((i * 131u) ^ (i >> 3));
    ok &= tc_remote_register_tensor(server, "adversarial/tensor",
                                    source.data(), source.size()) == TC_OK;

    const bool invalid_offset_ok = check_invalid_offset(port);
    std::printf("  nonzero v1 offset rejected:   %s\n",
                invalid_offset_ok ? "OK" : "FAIL");
    ok &= invalid_offset_ok;

    tc_remote_ctx* client = nullptr;
    ok &= tc_remote_init(ctx, TC_REMOTE_ROLE_COMPUTE_CLIENT, nullptr, &client) == TC_OK;
    const int peer = client ? tc_remote_connect(client, server_url) : -1;
    ok &= peer >= 0;

    constexpr int kThreads = 8;
    constexpr int kFetchesPerThread = 20;
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    if (peer >= 0) {
        for (int thread_index = 0; thread_index != kThreads; ++thread_index) {
            threads.emplace_back([&] {
                std::vector<uint8_t> dst(kTensorBytes);
                for (int fetch = 0; fetch != kFetchesPerThread; ++fetch) {
                    if (tc_remote_tensor_fetch(client, peer, "adversarial/tensor",
                                               dst.data(), dst.size()) != TC_OK ||
                        std::memcmp(dst.data(), source.data(), source.size()) != 0) {
                        failures.fetch_add(1, std::memory_order_relaxed);
                        return;
                    }
                }
            });
        }
        for (auto& thread : threads) thread.join();
    }
    const bool concurrent_ok = peer >= 0 && failures.load() == 0 &&
        tc_remote_fetch_count(client) == static_cast<uint64_t>(kThreads * kFetchesPerThread);
    std::printf("  concurrent same-peer framing: %s\n",
                concurrent_ok ? "OK" : "FAIL");
    ok &= concurrent_ok;

    std::vector<uint8_t> reconnected(kTensorBytes);
    const bool reconnect_ok = peer >= 0 &&
        tc_remote_internal_reconnect(client, peer, server_url, nullptr) == TC_OK &&
        tc_remote_tensor_fetch(client, peer, "adversarial/tensor",
                               reconnected.data(), reconnected.size()) == TC_OK &&
        std::memcmp(reconnected.data(), source.data(), source.size()) == 0;
    std::printf("  in-place peer reconnect:      %s\n",
                reconnect_ok ? "OK" : "FAIL");
    ok &= reconnect_ok;

    if (client) ok &= tc_remote_shutdown(client) == TC_OK;

    const bool unregister_barrier_ok = check_unregister_barrier(server, port);
    std::printf("  unregister lifetime barrier:  %s\n",
                unregister_barrier_ok ? "OK" : "FAIL");
    ok &= unregister_barrier_ok;

    // Leave a client stalled halfway through the request header. Shutdown must
    // actively interrupt the accepted socket instead of waiting for its read.
    const int idle_fd = connect_loopback(port);
    uint8_t partial_header = 0;
    if (idle_fd >= 0) (void)send_all(idle_fd, &partial_header, sizeof(partial_header));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::atomic<bool> shutdown_done{false};
    tc_status_t shutdown_status = TC_ERR_INTERNAL;
    std::thread shutdown_thread([&] {
        shutdown_status = tc_remote_shutdown(server);
        shutdown_done.store(true, std::memory_order_release);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!shutdown_done.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    const bool timely_shutdown = idle_fd >= 0 &&
        shutdown_done.load(std::memory_order_acquire);
    if (!timely_shutdown && idle_fd >= 0) ::shutdown(idle_fd, SHUT_RDWR);
    if (idle_fd >= 0) ::close(idle_fd);
    shutdown_thread.join();
    std::printf("  idle-client shutdown bounded: %s\n",
                (timely_shutdown && shutdown_status == TC_OK) ? "OK" : "FAIL");
    ok &= timely_shutdown && shutdown_status == TC_OK;

    tc_shutdown(ctx);
    std::printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
