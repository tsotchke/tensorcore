/*
 * tensorcore - internal TC_DIST_GLOO TCP transport prototype.
 *
 * This file is compiled in CPU builds to keep the future Ethernet transport
 * honest, but distributed_cpu.cpp still returns TC_ERR_UNSUPPORTED_FAMILY
 * for public multi-rank TC_DIST_GLOO. The helper below is intentionally
 * hidden from the shared-library export surface until that public backend
 * is wired and tested.
 *
 * Current prototype algorithm:
 *
 *   Rendezvous: rendezvous_url is parsed as "tcp://HOST:PORT". Rank 0
 *   listens on HOST:PORT; ranks 1..N-1 connect to it.
 *
 *   Collectives:
 *
 *     allreduce  = rank-0 brokered sum + broadcast.
 *     broadcast  = rank-0 root only.
 *     barrier    = rank-0 brokered byte exchange.
 *
 * Failure modes: if any peer disappears mid-collective, the
 * read/write fails with EPIPE/ECONNRESET. We surface as TC_ERR_INTERNAL
 * and the caller can choose to retry or fall back to local-only training.
 *
 * Currently supports fp32 and fp16 (fp16 converts to fp32 for accumulation,
 * back to fp16 on the wire). int8/bf16 hooks reserved.
 *
 * Build gate: POSIX sockets. The file compiles to an unsupported stub on
 * Windows; production Windows build would use Winsock.
 */

#include "tensorcore/tensorcore.h"
#include "../core/internal.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#  define TC_GLOO_AVAILABLE 0
#else
#  define TC_GLOO_AVAILABLE 1
#  include <arpa/inet.h>
#  include <errno.h>
#  include <fcntl.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <unistd.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define TC_GLOO_HIDDEN __attribute__((visibility("hidden")))
#else
#  define TC_GLOO_HIDDEN
#endif

#if TC_GLOO_AVAILABLE

namespace {

struct GlooState {
    int                    rendez_listen_fd = -1;   /* rank 0 only */
    int                    rendez_conn_fd   = -1;   /* rank > 0: connection to rank 0 */
    std::vector<int>       peer_conns;              /* fd to each peer rank (size world_size, self is -1) */
    int                    next_fd          = -1;
    int                    prev_fd          = -1;
    std::string            self_host;
    uint16_t               self_port        = 0;
};

bool parse_rendezvous(const std::string& url, std::string* host, uint16_t* port) {
    /* tcp://host:port */
    const std::string prefix = "tcp://";
    if (url.size() < prefix.size() || url.compare(0, prefix.size(), prefix) != 0) return false;
    const std::string body = url.substr(prefix.size());
    const auto colon = body.rfind(':');
    if (colon == std::string::npos) return false;
    *host = body.substr(0, colon);
    *port = (uint16_t)std::atoi(body.substr(colon + 1).c_str());
    return *port != 0;
}

bool write_all(int fd, const void* data, size_t bytes) {
    const uint8_t* p = (const uint8_t*)data;
    while (bytes > 0) {
        ssize_t n = ::write(fd, p, bytes);
        if (n <= 0) { if (errno == EINTR) continue; return false; }
        p += n; bytes -= (size_t)n;
    }
    return true;
}

bool read_all(int fd, void* data, size_t bytes) {
    uint8_t* p = (uint8_t*)data;
    while (bytes > 0) {
        ssize_t n = ::read(fd, p, bytes);
        if (n == 0) return false;     /* EOF - peer disconnected */
        if (n < 0)  { if (errno == EINTR) continue; return false; }
        p += n; bytes -= (size_t)n;
    }
    return true;
}

int tcp_listen(uint16_t port, std::string* out_host) {
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
    if (out_host) *out_host = "0.0.0.0";
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

    int fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { ::freeaddrinfo(res); return -1; }
    /* Disable Nagle for low-latency collectives. */
    int yes = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
    /* Retry on transient failures (rank 0 may not be listening yet). */
    int retries = 30;
    while (::connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        if (errno == ECONNREFUSED && retries-- > 0) {
            usleep(100 * 1000);
            continue;
        }
        ::close(fd); ::freeaddrinfo(res); return -1;
    }
    ::freeaddrinfo(res);
    return fd;
}

float f16_to_f32_gloo(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { float r; std::memcpy(&r, &sign, 4); return r; }
        int e = -14;
        while ((mant & 0x0400u) == 0) { mant <<= 1; --e; }
        mant &= 0x03ffu;
        bits = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    float r; std::memcpy(&r, &bits, 4); return r;
}

uint16_t f32_to_f16_gloo(float v) {
    union { float f; uint32_t u; } x = {v};
    const uint32_t bits = x.u;
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    const uint32_t exp = (bits >> 23) & 0xffu;
    uint32_t mant = bits & 0x7fffffu;
    if (exp == 0xffu) return (uint16_t)(sign | (mant ? 0x7e00u : 0x7c00u));
    int half_exp = (int)exp - 127 + 15;
    if (half_exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (half_exp <= 0) {
        if (half_exp < -10) return sign;
        mant |= 0x800000u;
        const int shift = 14 - half_exp;
        const uint32_t rounded = mant + ((1u << (shift - 1)) - 1u) + ((mant >> shift) & 1u);
        return (uint16_t)(sign | (rounded >> shift));
    }
    uint32_t rounded = mant + 0x0fffu + ((mant >> 13) & 1u);
    if (rounded & 0x800000u) { rounded = 0; ++half_exp; if (half_exp >= 31) return (uint16_t)(sign | 0x7c00u); }
    return (uint16_t)(sign | ((uint32_t)half_exp << 10) | (rounded >> 13));
}

}  // namespace

/* Internal API reserved for distributed_cpu.cpp once TC_DIST_GLOO is wired. */

extern "C" TC_GLOO_HIDDEN GlooState* tc_gloo_init(int world_size, int rank, const char* rendezvous_url) {
    if (world_size < 1 || rank < 0 || rank >= world_size || !rendezvous_url) return nullptr;
    std::string host;
    uint16_t port = 0;
    if (!parse_rendezvous(rendezvous_url, &host, &port)) return nullptr;

    auto* s = new GlooState();
    s->peer_conns.assign((size_t)world_size, -1);

    if (rank == 0) {
        /* Listen + accept all others. */
        s->rendez_listen_fd = tcp_listen(port, &s->self_host);
        if (s->rendez_listen_fd < 0) { delete s; return nullptr; }
        for (int r = 1; r < world_size; ++r) {
            sockaddr_in peer_addr = {};
            socklen_t alen = sizeof(peer_addr);
            int fd = ::accept(s->rendez_listen_fd, (sockaddr*)&peer_addr, &alen);
            if (fd < 0) { delete s; return nullptr; }
            /* First message from peer: its rank (uint32 LE). */
            uint32_t peer_rank = 0;
            if (!read_all(fd, &peer_rank, 4) || peer_rank >= (uint32_t)world_size) {
                ::close(fd); delete s; return nullptr;
            }
            int yes = 1;
            ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
            s->peer_conns[peer_rank] = fd;
        }
    } else {
        /* Connect to rank 0. */
        int fd = tcp_connect(host, port);
        if (fd < 0) { delete s; return nullptr; }
        uint32_t self_rank = (uint32_t)rank;
        if (!write_all(fd, &self_rank, 4)) { ::close(fd); delete s; return nullptr; }
        s->rendez_conn_fd = fd;
        s->peer_conns[0] = fd;
    }

    /* Set up ring neighbors. For the ring, each rank communicates with
     * rank+1 (next) and rank-1 (prev). On rank 0, the connections to all
     * others are already set up. For ranks > 0, we currently route all
     * collectives through the rendezvous (rank 0). For a real ring with
     * direct rank-to-rank links, the topology exchange would happen here
     * - for v0 we use the rendezvous-as-broker pattern which is correct
     * but slower (2x bandwidth through rank 0). */
    s->next_fd = (rank + 1 < world_size) ? s->peer_conns[(rank + 1) % world_size] : -1;
    s->prev_fd = (rank > 0)              ? s->peer_conns[(rank - 1 + world_size) % world_size]
                                         : ((world_size > 1) ? s->peer_conns[world_size - 1] : -1);
    return s;
}

extern "C" TC_GLOO_HIDDEN void tc_gloo_destroy(GlooState* s) {
    if (!s) return;
    if (s->rendez_listen_fd >= 0) ::close(s->rendez_listen_fd);
    if (s->rendez_conn_fd >= 0 && s->rendez_conn_fd != s->peer_conns[0]) ::close(s->rendez_conn_fd);
    for (int fd : s->peer_conns) if (fd >= 0) ::close(fd);
    delete s;
}

/* All-reduce SUM over a host fp32 buffer using rank-0-as-broker. Simpler
 * than full ring but correct: each non-zero rank sends to rank 0, rank 0
 * sums + broadcasts back. O(N * count) bytes through rank 0.
 *
 * For large clusters this should be replaced with the ring reduce-scatter
 * + all-gather algorithm; for the Quebec <-> Alaska 2-site case (effectively
 * world_size=2 across the bridge), the broker pattern is identical to
 * ring in bandwidth and simpler in code. */
extern "C" TC_GLOO_HIDDEN int tc_gloo_allreduce_f32_sum(GlooState* s, int world_size, int rank,
                                                         float* data, size_t n) {
    if (world_size <= 1) return 0;
    const size_t bytes = n * sizeof(float);
    if (rank == 0) {
        std::vector<float> tmp(n);
        for (int r = 1; r < world_size; ++r) {
            if (!read_all(s->peer_conns[r], tmp.data(), bytes)) return -1;
            for (size_t i = 0; i < n; ++i) data[i] += tmp[i];
        }
        for (int r = 1; r < world_size; ++r) {
            if (!write_all(s->peer_conns[r], data, bytes)) return -1;
        }
    } else {
        if (!write_all(s->peer_conns[0], data, bytes)) return -1;
        if (!read_all(s->peer_conns[0], data, bytes)) return -1;
    }
    return 0;
}

extern "C" TC_GLOO_HIDDEN int tc_gloo_allreduce_f16_sum(GlooState* s, int world_size, int rank,
                                                         uint16_t* data, size_t n) {
    /* Convert to fp32, allreduce, convert back. */
    std::vector<float> f32(n);
    for (size_t i = 0; i < n; ++i) f32[i] = f16_to_f32_gloo(data[i]);
    const int rc = tc_gloo_allreduce_f32_sum(s, world_size, rank, f32.data(), n);
    if (rc != 0) return rc;
    for (size_t i = 0; i < n; ++i) data[i] = f32_to_f16_gloo(f32[i]);
    return 0;
}

extern "C" TC_GLOO_HIDDEN int tc_gloo_broadcast_f32(GlooState* s, int world_size, int rank, int root,
                                                    float* data, size_t n) {
    const size_t bytes = n * sizeof(float);
    if (rank == root) {
        for (int r = 0; r < world_size; ++r) {
            if (r == root) continue;
            int fd = (rank == 0) ? s->peer_conns[r] : s->peer_conns[0];
            (void)fd;
            /* For root != 0: we'd need a separate direct connection. v0
             * supports root==0 only; other roots return error. */
            if (root != 0) return -1;
            if (!write_all(s->peer_conns[r], data, bytes)) return -1;
        }
    } else {
        if (root != 0) return -1;
        if (!read_all(s->peer_conns[0], data, bytes)) return -1;
    }
    return 0;
}

extern "C" TC_GLOO_HIDDEN int tc_gloo_barrier(GlooState* s, int world_size, int rank) {
    /* Implement barrier via a single-byte allreduce-sum. */
    if (world_size <= 1) return 0;
    if (rank == 0) {
        char buf;
        for (int r = 1; r < world_size; ++r) {
            if (!read_all(s->peer_conns[r], &buf, 1)) return -1;
        }
        char ack = 1;
        for (int r = 1; r < world_size; ++r) {
            if (!write_all(s->peer_conns[r], &ack, 1)) return -1;
        }
    } else {
        char buf = 0;
        if (!write_all(s->peer_conns[0], &buf, 1)) return -1;
        if (!read_all(s->peer_conns[0], &buf, 1)) return -1;
    }
    return 0;
}

#else  /* !TC_GLOO_AVAILABLE */

struct GlooState {};

extern "C" TC_GLOO_HIDDEN GlooState* tc_gloo_init(int, int, const char*) { return nullptr; }
extern "C" TC_GLOO_HIDDEN void tc_gloo_destroy(GlooState*) {}
extern "C" TC_GLOO_HIDDEN int  tc_gloo_allreduce_f32_sum(GlooState*, int, int, float*, size_t) { return -1; }
extern "C" TC_GLOO_HIDDEN int  tc_gloo_allreduce_f16_sum(GlooState*, int, int, uint16_t*, size_t) { return -1; }
extern "C" TC_GLOO_HIDDEN int  tc_gloo_broadcast_f32(GlooState*, int, int, int, float*, size_t) { return -1; }
extern "C" TC_GLOO_HIDDEN int  tc_gloo_barrier(GlooState*, int, int) { return -1; }

#endif
