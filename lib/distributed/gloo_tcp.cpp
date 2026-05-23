/*
 * tensorcore - internal TC_DIST_GLOO TCP transport prototype.
 *
 * This file backs the portable CPU TC_DIST_GLOO path. The helper symbols
 * are intentionally hidden from the shared-library export surface; the
 * public ABI remains the tc_dist_* functions in distributed_cpu.cpp.
 *
 * Current prototype algorithm:
 *
 *   Rendezvous: rendezvous_url is parsed as "tcp://HOST:PORT" or
 *   "gloo+tcp://HOST:PORT". Rank 0 listens on HOST:PORT; ranks 1..N-1
 *   connect to it.
 *
 *   Collectives:
 *
 *     allreduce  = rank-0 brokered reduction + broadcast by default,
 *                  opt-in IPv4/IPv6 ring reduce-scatter + all-gather for
 *                  fp32 SUM at >=3 ranks when TC_GLOO_RING=1.
 *     broadcast  = rank-0 brokered bitwise replication from any root.
 *     allgather  = rank-0 brokered gather + broadcast.
 *     barrier    = rank-0 brokered byte exchange.
 *
 * Failure modes: if any peer disappears mid-collective, the
 * read/write fails with EPIPE/ECONNRESET. We surface as TC_ERR_INTERNAL
 * and the caller can choose to retry or fall back to local-only training.
 *
 * Currently supports fp32 SUM/AVG/MIN/MAX and fp16 SUM/AVG. fp16 converts
 * to fp32 for accumulation, then back to fp16 on the wire. int8/bf16 hooks
 * are reserved.
 *
 * Build gate: POSIX sockets. The file compiles to an unsupported stub on
 * Windows; production Windows build would use Winsock.
 */

#include "tensorcore/tensorcore.h"
#include "../core/internal.h"

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <thread>
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
#  include <sys/select.h>
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

struct RingPeerInfo {
    uint16_t port;
    uint16_t family;   /* AF_INET or AF_INET6 */
    uint8_t  addr[16]; /* network-order IPv4 in addr[0..3], IPv6 in addr[0..15] */
};

bool gloo_trace_enabled(void) {
    const char* env = std::getenv("TC_GLOO_TRACE");
    return env && env[0] == '1';
}

void gloo_trace(int rank, const char* fmt, ...) {
    if (!gloo_trace_enabled()) return;
    std::fprintf(stderr, "[tensorcore:gloo rank %d] ", rank);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

bool parse_rendezvous(const std::string& url, std::string* host, uint16_t* port) {
    size_t prefix_len = 0;
    const std::string tcp_prefix = "tcp://";
    const std::string gloo_tcp_prefix = "gloo+tcp://";
    if (url.compare(0, tcp_prefix.size(), tcp_prefix) == 0) {
        prefix_len = tcp_prefix.size();
    } else if (url.compare(0, gloo_tcp_prefix.size(), gloo_tcp_prefix) == 0) {
        prefix_len = gloo_tcp_prefix.size();
    } else {
        return false;
    }
    const std::string body = url.substr(prefix_len);
    std::string port_str;
    if (!body.empty() && body[0] == '[') {
        const auto close = body.find(']');
        if (close == std::string::npos || close + 1 >= body.size() || body[close + 1] != ':') {
            return false;
        }
        *host = body.substr(1, close - 1);
        port_str = body.substr(close + 2);
    } else {
        const auto colon = body.rfind(':');
        if (colon == std::string::npos) return false;
        *host = body.substr(0, colon);
        port_str = body.substr(colon + 1);
    }
    char* end = nullptr;
    const long parsed = std::strtol(port_str.c_str(), &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > 65535) return false;
    *port = (uint16_t)parsed;
    return !host->empty();
}

bool host_looks_ipv6(const std::string& host) {
    return host.find(':') != std::string::npos;
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

bool checked_mul_size(size_t a, size_t b, size_t* out) {
    if (!out) return false;
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    *out = a * b;
    return true;
}

bool checked_f32_bytes(size_t n, size_t* out) {
    return checked_mul_size(n, sizeof(float), out);
}

int tcp_listen(uint16_t port, const std::string& rendezvous_host, std::string* out_host) {
    const int family = host_looks_ipv6(rendezvous_host) ? AF_INET6 : AF_INET;
    int fd = ::socket(family, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    if (family == AF_INET6) {
        int v6only = 1;
        (void)::setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &v6only, sizeof(v6only));
        sockaddr_in6 addr6 = {};
        addr6.sin6_family = AF_INET6;
        addr6.sin6_port = htons(port);
        addr6.sin6_addr = in6addr_any;
        if (::bind(fd, (sockaddr*)&addr6, sizeof(addr6)) < 0) { ::close(fd); return -1; }
        if (out_host) *out_host = "::";
    } else {
        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        if (::bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0) { ::close(fd); return -1; }
        if (out_host) *out_host = "0.0.0.0";
    }
    if (::listen(fd, 64) < 0) { ::close(fd); return -1; }
    return fd;
}

int tcp_connect(const std::string& host, uint16_t port) {
    addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    char port_str[16];
    std::snprintf(port_str, sizeof(port_str), "%u", port);
    if (::getaddrinfo(host.c_str(), port_str, &hints, &res) != 0 || !res) return -1;

    /* Retry on transient failures (rank 0 may not be listening yet). */
    for (int retries = 30; retries >= 0; --retries) {
        int saved_errno = 0;
        for (addrinfo* ai = res; ai; ai = ai->ai_next) {
            int fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
            if (fd < 0) { saved_errno = errno; continue; }
            int yes = 1;
            ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
            if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
                ::freeaddrinfo(res);
                return fd;
            }
            saved_errno = errno;
            ::close(fd);
        }
        if ((saved_errno == ECONNREFUSED || saved_errno == EHOSTUNREACH ||
             saved_errno == ENETUNREACH || saved_errno == ETIMEDOUT) && retries > 0) {
            usleep(100 * 1000);
            continue;
        }
        ::freeaddrinfo(res);
        return -1;
    }
    ::freeaddrinfo(res);
    return -1;
}

int tcp_connect_timeout(const std::string& host, uint16_t port, int timeout_ms) {
    addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    char port_str[16];
    std::snprintf(port_str, sizeof(port_str), "%u", port);
    if (::getaddrinfo(host.c_str(), port_str, &hints, &res) != 0 || !res) return -1;

    for (addrinfo* ai = res; ai; ai = ai->ai_next) {
        int fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int yes = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));

        const int flags = ::fcntl(fd, F_GETFL, 0);
        if (flags >= 0) (void)::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        int rc = ::connect(fd, ai->ai_addr, ai->ai_addrlen);
        if (rc == 0) {
            if (flags >= 0) (void)::fcntl(fd, F_SETFL, flags);
            ::freeaddrinfo(res);
            return fd;
        }
        if (errno != EINPROGRESS) {
            ::close(fd);
            continue;
        }

        timeval tv = {};
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        fd_set wfds;
        FD_ZERO(&wfds);
        FD_SET(fd, &wfds);
        do {
            rc = ::select(fd + 1, nullptr, &wfds, nullptr, &tv);
        } while (rc < 0 && errno == EINTR);
        if (rc <= 0) {
            ::close(fd);
            continue;
        }

        int so_error = 0;
        socklen_t len = sizeof(so_error);
        if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &len) != 0 || so_error != 0) {
            ::close(fd);
            continue;
        }
        if (flags >= 0) (void)::fcntl(fd, F_SETFL, flags);
        ::freeaddrinfo(res);
        return fd;
    }
    ::freeaddrinfo(res);
    return -1;
}

bool peer_addr_valid(const RingPeerInfo& p) {
    if (p.family == AF_INET) {
        uint32_t v4 = 0;
        std::memcpy(&v4, p.addr, sizeof(v4));
        return v4 != 0 && v4 != htonl(INADDR_ANY);
    }
    if (p.family == AF_INET6) {
        in6_addr v6 = {};
        std::memcpy(&v6, p.addr, sizeof(v6));
        return !IN6_IS_ADDR_UNSPECIFIED(&v6);
    }
    return false;
}

bool store_sockaddr_peer(const sockaddr_storage& ss, RingPeerInfo* out) {
    if (!out) return false;
    if (ss.ss_family == AF_INET) {
        const sockaddr_in* addr = (const sockaddr_in*)&ss;
        out->family = AF_INET;
        std::memset(out->addr, 0, sizeof(out->addr));
        std::memcpy(out->addr, &addr->sin_addr.s_addr, sizeof(addr->sin_addr.s_addr));
        return peer_addr_valid(*out);
    }
    if (ss.ss_family == AF_INET6) {
        const sockaddr_in6* addr = (const sockaddr_in6*)&ss;
        out->family = AF_INET6;
        std::memcpy(out->addr, &addr->sin6_addr, sizeof(addr->sin6_addr));
        return peer_addr_valid(*out);
    }
    return false;
}

bool resolve_host_peer(const std::string& host, RingPeerInfo* out) {
    if (!out || host.empty()) return false;
    addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (::getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res) return false;
    bool ok = false;
    for (addrinfo* ai = res; ai; ai = ai->ai_next) {
        sockaddr_storage ss = {};
        if (ai->ai_addrlen > sizeof(ss)) continue;
        std::memcpy(&ss, ai->ai_addr, ai->ai_addrlen);
        if (store_sockaddr_peer(ss, out)) { ok = true; break; }
    }
    ::freeaddrinfo(res);
    return ok;
}

bool endpoint_from_fd(int fd, bool peer, RingPeerInfo* out) {
    if (fd < 0 || !out) return false;
    sockaddr_storage ss = {};
    socklen_t alen = sizeof(ss);
    const int rc = peer
        ? ::getpeername(fd, (sockaddr*)&ss, &alen)
        : ::getsockname(fd, (sockaddr*)&ss, &alen);
    return rc == 0 && store_sockaddr_peer(ss, out);
}

RingPeerInfo advertised_peer_info(int rank, int rendez_fd, const std::string& rendezvous_host) {
    const char* env = std::getenv("TC_GLOO_ADVERTISE_HOST");
    RingPeerInfo out = {};
    if (env && env[0] && resolve_host_peer(env, &out)) return out;
    if (rank == 0 && resolve_host_peer(rendezvous_host, &out)) return out;
    if (endpoint_from_fd(rendez_fd, false, &out)) return out;
    return out;
}

std::string peer_addr_string(const RingPeerInfo& peer) {
    char buf[INET6_ADDRSTRLEN] = {};
    if (peer.family == AF_INET) {
        if (::inet_ntop(AF_INET, peer.addr, buf, sizeof(buf))) return std::string(buf);
    } else if (peer.family == AF_INET6) {
        if (::inet_ntop(AF_INET6, peer.addr, buf, sizeof(buf))) return std::string(buf);
    }
    return std::string();
}

int ring_connect_timeout_ms(void) {
    const char* env = std::getenv("TC_GLOO_RING_CONNECT_TIMEOUT_MS");
    if (!env || !env[0]) return 3000;
    char* end = nullptr;
    long v = std::strtol(env, &end, 10);
    if (!end || *end != '\0' || v < 10 || v > 60000) return 3000;
    return (int)v;
}

int accept_with_timeout(int listen_fd, int timeout_ms) {
    timeval tv = {};
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    int rc = -1;
    do {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(listen_fd, &rfds);
        rc = ::select(listen_fd + 1, &rfds, nullptr, nullptr, &tv);
    } while (rc < 0 && errno == EINTR);
    if (rc <= 0) return -1;

    sockaddr_storage peer_addr = {};
    socklen_t alen = sizeof(peer_addr);
    return ::accept(listen_fd, (sockaddr*)&peer_addr, &alen);
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

bool exchange_ring_chunks(GlooState* s,
                          const void* send_data, size_t send_bytes,
                          void* recv_data, size_t recv_bytes) {
    bool write_ok = true;
    std::thread writer([&]() {
        if (send_bytes > 0) {
            write_ok = write_all(s->next_fd, send_data, send_bytes);
        }
    });
    const bool read_ok = (recv_bytes == 0) || read_all(s->prev_fd, recv_data, recv_bytes);
    writer.join();
    return write_ok && read_ok;
}

}  // namespace

void close_gloo_state(GlooState* s) {
    if (!s) return;
    auto is_peer_fd = [&](int fd) {
        for (int peer : s->peer_conns) {
            if (peer == fd) return true;
        }
        return false;
    };
    auto close_if_unique = [&](int fd) {
        if (fd < 0) return;
        if (fd == s->rendez_listen_fd || fd == s->rendez_conn_fd) return;
        if (is_peer_fd(fd)) return;
        ::close(fd);
    };
    close_if_unique(s->next_fd);
    if (s->prev_fd != s->next_fd) close_if_unique(s->prev_fd);
    if (s->rendez_listen_fd >= 0) ::close(s->rendez_listen_fd);
    if (s->rendez_conn_fd >= 0 && s->rendez_conn_fd != s->peer_conns[0]) {
        ::close(s->rendez_conn_fd);
    }
    for (int fd : s->peer_conns) if (fd >= 0) ::close(fd);
    delete s;
}

/* Internal API consumed by distributed_cpu.cpp. */

extern "C" TC_GLOO_HIDDEN GlooState* tc_gloo_init(int world_size, int rank, const char* rendezvous_url) {
    if (world_size < 1 || rank < 0 || rank >= world_size || !rendezvous_url) return nullptr;
    std::string host;
    uint16_t port = 0;
    if (!parse_rendezvous(rendezvous_url, &host, &port)) {
        gloo_trace(rank, "init=parse_failed url=%s", rendezvous_url);
        return nullptr;
    }

    auto* s = new (std::nothrow) GlooState();
    if (!s) return nullptr;
    s->peer_conns.assign((size_t)world_size, -1);

    if (rank == 0) {
        /* Listen + accept all others. */
        s->rendez_listen_fd = tcp_listen(port, host, &s->self_host);
        if (s->rendez_listen_fd < 0) {
            gloo_trace(rank, "init=listen_failed host=%s port=%u errno=%d", host.c_str(), port, errno);
            close_gloo_state(s);
            return nullptr;
        }
        gloo_trace(rank, "init=listening host=%s port=%u family=%s",
                   host.c_str(), port, host_looks_ipv6(host) ? "ipv6" : "ipv4");
        for (int r = 1; r < world_size; ++r) {
            sockaddr_storage peer_addr = {};
            socklen_t alen = sizeof(peer_addr);
            int fd = ::accept(s->rendez_listen_fd, (sockaddr*)&peer_addr, &alen);
            if (fd < 0) {
                gloo_trace(rank, "init=accept_failed errno=%d", errno);
                close_gloo_state(s);
                return nullptr;
            }
            /* First message from peer: its rank (uint32 LE). */
            uint32_t peer_rank = 0;
            if (!read_all(fd, &peer_rank, 4) ||
                peer_rank == 0 ||
                peer_rank >= (uint32_t)world_size ||
                s->peer_conns[peer_rank] >= 0) {
                gloo_trace(rank, "init=peer_rank_failed peer_rank=%u", peer_rank);
                ::close(fd); close_gloo_state(s); return nullptr;
            }
            int yes = 1;
            ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
            s->peer_conns[peer_rank] = fd;
        }
    } else {
        /* Connect to rank 0. */
        int fd = tcp_connect(host, port);
        if (fd < 0) {
            gloo_trace(rank, "init=connect_failed host=%s port=%u errno=%d", host.c_str(), port, errno);
            close_gloo_state(s);
            return nullptr;
        }
        uint32_t self_rank = (uint32_t)rank;
        if (!write_all(fd, &self_rank, 4)) {
            gloo_trace(rank, "init=write_rank_failed errno=%d", errno);
            ::close(fd); close_gloo_state(s); return nullptr;
        }
        s->rendez_conn_fd = fd;
        s->peer_conns[0] = fd;
    }

    /* ------------------------------------------------------------------
     * Ring topology exchange.
     *
     * After rendezvous, each rank > 0 only has a connection to rank 0.
     * For ring all-reduce we need rank r connected to rank (r+1)%N.
     *
     * Protocol (executed by all ranks, in lockstep):
     *
     *   Phase A: each rank picks a local port and opens a listening socket
     *            on it. Ranks > 0 send their port to rank 0.
     *   Phase B: rank 0 builds a (rank, host, port) table and broadcasts
     *            it to everyone via the existing peer_conns[r] sockets.
     *   Phase C: each rank r opens a NEW connection to rank (r+1)%N's
     *            listening port. Accepts a connection from (r-1+N)%N.
     *            These become next_fd and prev_fd.
     *
     * For N=2 the ring degenerates to the broker pattern but still works:
     * rank 0 <-> rank 1 directly via the rendezvous, no extra connections
     * needed. We keep that fast path.
     * ------------------------------------------------------------------ */
    s->next_fd = -1;
    s->prev_fd = -1;

    if (world_size <= 2) {
        if (world_size == 2) {
            s->next_fd = s->peer_conns[(rank + 1) % world_size];
            s->prev_fd = s->peer_conns[(rank + 1) % world_size];   /* same fd */
        }
        return s;
    }

    /* Ring topology setup is opt-in via TC_GLOO_RING=1 (default off until
     * we have a NAT-transparent way to discover ring neighbors). When off,
     * we leave next_fd/prev_fd = -1 and the allreduce dispatcher falls
     * through to the broker path automatically. Once ring setup starts,
     * failures are fatal for this context: a per-rank silent fallback can
     * deadlock peers waiting in the topology exchange. */
    const char* enable_ring = std::getenv("TC_GLOO_RING");
    if (!(enable_ring && enable_ring[0] == '1')) {
        return s;
    }

    /* --- Phase A: every rank listens on a local ring port. If one rank
     * cannot bind, the group falls back to broker collectives instead of
     * failing the distributed context. */
    const uint16_t ring_port_base = (uint16_t)(port + 1);
    int my_ring_listen = -1;
    uint16_t my_ring_port = 0;
    for (int attempt = 0; attempt < 64; ++attempt) {
        my_ring_port = (uint16_t)(ring_port_base + rank + attempt * world_size);
        std::string ignored;
        my_ring_listen = tcp_listen(my_ring_port, host, &ignored);
        if (my_ring_listen >= 0) break;
    }
    if (my_ring_listen < 0) my_ring_port = 0;
    s->self_port = my_ring_port;
    auto broker_fallback = [&]() -> GlooState* {
        gloo_trace(rank, "direct_ring=fallback");
        if (my_ring_listen >= 0) {
            ::close(my_ring_listen);
            my_ring_listen = -1;
        }
        if (s->next_fd >= 0) { ::close(s->next_fd); s->next_fd = -1; }
        if (s->prev_fd >= 0 && s->prev_fd != s->next_fd) {
            ::close(s->prev_fd);
            s->prev_fd = -1;
        }
        s->next_fd = -1;
        s->prev_fd = -1;
        return s;
    };

    /* --- Phase B: each rank > 0 reports (port) to rank 0; rank 0
     *              collects + broadcasts the table. ----- */
    std::vector<RingPeerInfo> peers((size_t)world_size);
    if (rank == 0) {
        peers[0].port = my_ring_port;
        RingPeerInfo self = advertised_peer_info(rank, s->rendez_listen_fd, host);
        peers[0].family = self.family;
        std::memcpy(peers[0].addr, self.addr, sizeof(peers[0].addr));
        /* Read each peer's chosen port. Resolve their address from the
         * peer_conns[r] socket using getpeername. */
        for (int r = 1; r < world_size; ++r) {
            RingPeerInfo pr = {};
            if (!read_all(s->peer_conns[r], &pr, sizeof(pr))) return broker_fallback();
            peers[r] = pr;
            RingPeerInfo observed = {};
            if (!endpoint_from_fd(s->peer_conns[r], true, &observed)) {
                return broker_fallback();
            }
            if (!peer_addr_valid(peers[r])) {
                peers[r].family = observed.family;
                std::memcpy(peers[r].addr, observed.addr, sizeof(peers[r].addr));
            }
        }
        uint8_t topology_ok = 1;
        for (int r = 0; r < world_size; ++r) {
            if (peers[r].port == 0 || !peer_addr_valid(peers[r])) topology_ok = 0;
        }
        for (int r = 1; r < world_size; ++r) {
            if (!write_all(s->peer_conns[r], &topology_ok, 1)) return broker_fallback();
        }
        if (!topology_ok) return broker_fallback();

        /* Broadcast the table. */
        for (int r = 1; r < world_size; ++r) {
            if (!write_all(s->peer_conns[r], peers.data(),
                           peers.size() * sizeof(RingPeerInfo))) {
                return broker_fallback();
            }
        }
    } else {
        /* Tell rank 0 my port and the local address selected for the
         * rendezvous path. On Tailscale this is typically the rank's
         * reachable 100.x address, avoiding rank-0's NAT-observed source. */
        RingPeerInfo me = {};
        me.port = my_ring_port;
        RingPeerInfo self = advertised_peer_info(rank, s->peer_conns[0], host);
        me.family = self.family;
        std::memcpy(me.addr, self.addr, sizeof(me.addr));
        if (!write_all(s->peer_conns[0], &me, sizeof(me))) {
            return broker_fallback();
        }
        uint8_t topology_ok = 0;
        if (!read_all(s->peer_conns[0], &topology_ok, 1)) {
            return broker_fallback();
        }
        if (!topology_ok) return broker_fallback();
        if (!read_all(s->peer_conns[0], peers.data(),
                      peers.size() * sizeof(RingPeerInfo))) {
            return broker_fallback();
        }
    }

    /* --- Phase C: open ring neighbor connections. ----------------------
     *
     * Each listener is already open and has backlog, so every rank can
     * connect to next first, then accept the prev connection. This avoids
     * parity corner cases for odd world sizes and keeps rank 0 from
     * reusing rendezvous connections as ring links.
     * ------------------------------------------------------------------ */
    const int next_rank = (rank + 1) % world_size;

    const int timeout_ms = ring_connect_timeout_ms();
    bool direct_ok = true;
    const std::string next_host = peer_addr_string(peers[next_rank]);
    s->next_fd = tcp_connect_timeout(next_host, peers[next_rank].port, timeout_ms);
    if (s->next_fd < 0) direct_ok = false;
    if (my_ring_listen >= 0) {
        s->prev_fd = accept_with_timeout(my_ring_listen, timeout_ms);
        if (s->prev_fd < 0) direct_ok = false;
    } else {
        direct_ok = false;
    }

    uint8_t group_direct_ok = direct_ok ? 1 : 0;
    if (rank == 0) {
        for (int r = 1; r < world_size; ++r) {
            uint8_t peer_ok = 0;
            if (!read_all(s->peer_conns[r], &peer_ok, 1)) group_direct_ok = 0;
            if (!peer_ok) group_direct_ok = 0;
        }
        for (int r = 1; r < world_size; ++r) {
            if (!write_all(s->peer_conns[r], &group_direct_ok, 1)) group_direct_ok = 0;
        }
    } else {
        uint8_t ok = direct_ok ? 1 : 0;
        if (!write_all(s->peer_conns[0], &ok, 1) ||
            !read_all(s->peer_conns[0], &group_direct_ok, 1)) {
            group_direct_ok = 0;
        }
    }
    if (!group_direct_ok) return broker_fallback();

    int yes = 1;
    if (s->next_fd >= 0) ::setsockopt(s->next_fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
    if (s->prev_fd >= 0) ::setsockopt(s->prev_fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
    if (my_ring_listen >= 0) ::close(my_ring_listen);

    gloo_trace(rank, "direct_ring=enabled next_rank=%d next=%s:%u timeout_ms=%d",
               next_rank, next_host.c_str(), peers[next_rank].port,
               timeout_ms);
    return s;
}

extern "C" TC_GLOO_HIDDEN void tc_gloo_destroy(GlooState* s) {
    close_gloo_state(s);
}

/* Ring all-reduce SUM over fp32 buffer.
 *
 * Bandwidth: 2*(N-1)/N * count bytes per rank, vs N*count for broker.
 * For N=4 this is ~1.5*count per rank instead of 4*count through rank 0,
 * so the rank-0 hot spot disappears and per-rank wall-time drops ~2-4x.
 *
 * Algorithm (standard ring):
 *   Phase 1 (reduce-scatter, N-1 steps): each rank cycles a chunk of the
 *     buffer to its next neighbor while receiving and accumulating a
 *     chunk from its prev neighbor. After N-1 steps, rank r owns the
 *     fully-reduced chunk at index (r+1)%N.
 *   Phase 2 (all-gather, N-1 steps): each rank passes its now-reduced
 *     chunk around the ring, overwriting (not summing) on receive.
 *
 * Chunks are equal-sized except the last, which may be smaller. We pad
 * up to ceil-divide so every rank sends/receives the same chunk size,
 * keeping the loop simple. */
extern "C" TC_GLOO_HIDDEN int tc_gloo_allreduce_f32_sum_ring(GlooState* s, int world_size, int rank,
                                                              float* data, size_t n) {
    if (world_size <= 1) return 0;
    if (s->next_fd < 0 || s->prev_fd < 0) return -1;
    if (n > std::numeric_limits<size_t>::max() - (size_t)(world_size - 1)) return -1;
    const size_t chunk_elems = (n + world_size - 1) / world_size;
    std::vector<float> recv_buf(chunk_elems, 0.0f);

    auto chunk_range = [&](int idx, size_t* start, size_t* len) {
        *start = (size_t)idx * chunk_elems;
        if (*start >= n) { *len = 0; return; }
        const size_t end = std::min(n, *start + chunk_elems);
        *len = end - *start;
    };

    /* Phase 1: reduce-scatter.
     * Initial state: each rank holds the full buffer; we'll cycle chunks.
     * At step k, rank r sends chunk[(r-k+N)%N] to next, recvs chunk from
     * prev and accumulates into chunk[(r-k-1+N)%N]. */
    int send_idx = rank;
    int recv_idx = (rank - 1 + world_size) % world_size;
    for (int step = 0; step < world_size - 1; ++step) {
        size_t send_off = 0, send_len = 0;
        size_t recv_off = 0, recv_len = 0;
        chunk_range(send_idx, &send_off, &send_len);
        chunk_range(recv_idx, &recv_off, &recv_len);

        size_t send_bytes = 0, recv_bytes = 0;
        if (!checked_f32_bytes(send_len, &send_bytes) ||
            !checked_f32_bytes(recv_len, &recv_bytes)) return -1;
        if (!exchange_ring_chunks(s, data + send_off, send_bytes,
                                  recv_buf.data(), recv_bytes)) return -1;
        if (recv_len > 0) {
            for (size_t i = 0; i < recv_len; ++i) data[recv_off + i] += recv_buf[i];
        }
        send_idx = recv_idx;
        recv_idx = (recv_idx - 1 + world_size) % world_size;
    }

    /* Phase 2: all-gather. Each rank now owns the fully-reduced chunk at
     * index (rank+1)%N. Cycle it around. */
    send_idx = (rank + 1) % world_size;
    recv_idx = rank;
    for (int step = 0; step < world_size - 1; ++step) {
        size_t send_off = 0, send_len = 0;
        size_t recv_off = 0, recv_len = 0;
        chunk_range(send_idx, &send_off, &send_len);
        chunk_range(recv_idx, &recv_off, &recv_len);

        size_t send_bytes = 0, recv_bytes = 0;
        if (!checked_f32_bytes(send_len, &send_bytes) ||
            !checked_f32_bytes(recv_len, &recv_bytes)) return -1;
        if (!exchange_ring_chunks(s, data + send_off, send_bytes,
                                  data + recv_off, recv_bytes)) return -1;
        send_idx = recv_idx;
        recv_idx = (recv_idx - 1 + world_size) % world_size;
    }
    return 0;
}

/* All-reduce SUM over a host fp32 buffer. For world_size <= 2 uses the
 * rank-0-as-broker pattern (which is identical to ring at N=2). For
 * world_size >= 3, explicit TC_GLOO_RING=1 uses the ring algorithm via
 * tc_gloo_allreduce_f32_sum_ring, eliminating the rank-0 hot spot.
 *
 * TC_GLOO_NO_RING=1 forces broker even if ring descriptors are present. */
extern "C" TC_GLOO_HIDDEN int tc_gloo_allreduce_f32_sum(GlooState* s, int world_size, int rank,
                                                         float* data, size_t n) {
    if (world_size <= 1) return 0;
    const char* no_ring = std::getenv("TC_GLOO_NO_RING");
    if (world_size >= 3 && !(no_ring && no_ring[0] == '1') &&
        s->next_fd >= 0 && s->prev_fd >= 0) {
        gloo_trace(rank, "allreduce_f32_sum route=ring elements=%zu", n);
        return tc_gloo_allreduce_f32_sum_ring(s, world_size, rank, data, n);
    }
    gloo_trace(rank, "allreduce_f32_sum route=broker elements=%zu", n);
    size_t bytes = 0;
    if (!checked_f32_bytes(n, &bytes)) return -1;
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
    size_t bytes = 0;
    if (!checked_f32_bytes(n, &bytes)) return -1;
    (void)bytes;
    std::vector<float> f32(n);
    for (size_t i = 0; i < n; ++i) f32[i] = f16_to_f32_gloo(data[i]);
    const int rc = tc_gloo_allreduce_f32_sum(s, world_size, rank, f32.data(), n);
    if (rc != 0) return rc;
    for (size_t i = 0; i < n; ++i) data[i] = f32_to_f16_gloo(f32[i]);
    return 0;
}

extern "C" TC_GLOO_HIDDEN int tc_gloo_broadcast_f32(GlooState* s, int world_size, int rank, int root,
                                                    float* data, size_t n) {
    size_t bytes = 0;
    if (!checked_f32_bytes(n, &bytes)) return -1;
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

/* ------------------------------------------------------------------------
 * MIN / MAX reductions (rank-0-brokered, fp32).
 * ------------------------------------------------------------------------ */

extern "C" TC_GLOO_HIDDEN int tc_gloo_allreduce_f32_min(GlooState* s, int world_size, int rank,
                                                         float* data, size_t n) {
    if (world_size <= 1) return 0;
    size_t bytes = 0;
    if (!checked_f32_bytes(n, &bytes)) return -1;
    if (rank == 0) {
        std::vector<float> tmp(n);
        for (int r = 1; r < world_size; ++r) {
            if (!read_all(s->peer_conns[r], tmp.data(), bytes)) return -1;
            for (size_t i = 0; i < n; ++i) if (tmp[i] < data[i]) data[i] = tmp[i];
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

extern "C" TC_GLOO_HIDDEN int tc_gloo_allreduce_f32_max(GlooState* s, int world_size, int rank,
                                                         float* data, size_t n) {
    if (world_size <= 1) return 0;
    size_t bytes = 0;
    if (!checked_f32_bytes(n, &bytes)) return -1;
    if (rank == 0) {
        std::vector<float> tmp(n);
        for (int r = 1; r < world_size; ++r) {
            if (!read_all(s->peer_conns[r], tmp.data(), bytes)) return -1;
            for (size_t i = 0; i < n; ++i) if (tmp[i] > data[i]) data[i] = tmp[i];
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

/* ------------------------------------------------------------------------
 * Allgather (rank-0-brokered): each rank contributes n elements; the
 * concatenated output is world_size * n elements in rank order.
 *
 * Algorithm: rank > 0 sends its slice to rank 0; rank 0 fills its own
 * slice in place; rank 0 broadcasts the full concatenated buffer back.
 *
 * The caller passes a buffer of (world_size * n) elements. On entry,
 * each rank's slice (out + rank * n) holds its contribution. On exit,
 * every rank's full buffer matches.
 * ------------------------------------------------------------------------ */

extern "C" TC_GLOO_HIDDEN int tc_gloo_allgather(GlooState* s, int world_size, int rank,
                                                  void* out, size_t bytes_per_rank) {
    if (world_size <= 1) return 0;
    uint8_t* full = (uint8_t*)out;
    size_t total = 0;
    if (!checked_mul_size((size_t)world_size, bytes_per_rank, &total)) return -1;
    if (rank == 0) {
        /* My slice is already at out + 0; receive each other rank's slice. */
        for (int r = 1; r < world_size; ++r) {
            size_t offset = 0;
            if (!checked_mul_size((size_t)r, bytes_per_rank, &offset)) return -1;
            if (!read_all(s->peer_conns[r], full + offset, bytes_per_rank)) {
                return -1;
            }
        }
        /* Broadcast the full concatenated buffer. */
        for (int r = 1; r < world_size; ++r) {
            if (!write_all(s->peer_conns[r], full, total)) return -1;
        }
    } else {
        /* Send my slice (currently at out + rank * bytes_per_rank). */
        size_t offset = 0;
        if (!checked_mul_size((size_t)rank, bytes_per_rank, &offset)) return -1;
        if (!write_all(s->peer_conns[0], full + offset, bytes_per_rank)) {
            return -1;
        }
        /* Receive the full concatenated buffer. */
        if (!read_all(s->peer_conns[0], full, total)) return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------------
 * Non-root broadcast: forwards root != 0 by routing through rank 0.
 *
 * If root == 0: write directly to all others.
 * If root != 0 && rank == root: send to rank 0 first; rank 0 then
 *   broadcasts to everyone else.
 * If rank == 0 && root != 0: receive from root, then broadcast to all others.
 * Otherwise: receive from rank 0 (the broker).
 *
 * v1: bandwidth-optimal would be direct send to peers, but the rank-0
 * brokered design keeps the peer-conn topology simple.
 * ------------------------------------------------------------------------ */

extern "C" TC_GLOO_HIDDEN int tc_gloo_broadcast_any_root(GlooState* s, int world_size, int rank,
                                                          int root, void* data, size_t bytes) {
    if (world_size <= 1) return 0;
    if (root < 0 || root >= world_size) return -1;
    if (root == 0) {
        if (rank == 0) {
            for (int r = 1; r < world_size; ++r) {
                if (!write_all(s->peer_conns[r], data, bytes)) return -1;
            }
        } else {
            if (!read_all(s->peer_conns[0], data, bytes)) return -1;
        }
        return 0;
    }
    /* root != 0 - route through rank 0. */
    if (rank == root) {
        if (!write_all(s->peer_conns[0], data, bytes)) return -1;
        /* Wait for rank 0 to acknowledge it'll forward (single byte). */
        char ack = 0;
        if (!read_all(s->peer_conns[0], &ack, 1)) return -1;
    } else if (rank == 0) {
        if (!read_all(s->peer_conns[root], data, bytes)) return -1;
        char ack = 1;
        if (!write_all(s->peer_conns[root], &ack, 1)) return -1;
        for (int r = 1; r < world_size; ++r) {
            if (r == root) continue;
            if (!write_all(s->peer_conns[r], data, bytes)) return -1;
        }
    } else {
        if (!read_all(s->peer_conns[0], data, bytes)) return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------------
 * Sparse compressed allreduce. Each rank pre-packs its delta-theta into the
 * sparse (idx, fp16-val) format; we ship the packed payloads to rank 0,
 * which scatter-merges them into a dense fp32 accumulator and broadcasts
 * the merged dense result back.
 *
 * In: payload_in (this rank's packed payload), payload_in_bytes.
 * Out: caller's full-size dense fp32 buffer; on success it contains the
 * sum across all ranks of each contributor's sparse delta-theta. AVG is the
 * caller's responsibility (divide by world_size).
 * ------------------------------------------------------------------------ */

extern "C" TC_GLOO_HIDDEN int tc_gloo_sparse_allreduce(GlooState* s, int world_size, int rank,
                                                        const void* payload_in,
                                                        size_t payload_in_bytes,
                                                        float* dense_out, size_t n_total) {
    if (world_size <= 1) {
        /* For single-rank, the caller hasn't gone through this path; nop. */
        return 0;
    }
    if (!s || !payload_in || !dense_out || n_total > 0xffffffffu) return -1;
    size_t dense_bytes = 0;
    if (!checked_f32_bytes(n_total, &dense_bytes)) return -1;
    /* Wire format on rank-0 inbound: uint32 payload_bytes, then payload. */
    if (rank == 0) {
        /* Start the dense accumulator zeroed. */
        std::memset(dense_out, 0, dense_bytes);
        /* Unpack rank 0's own payload first. */
        const uint8_t* in = (const uint8_t*)payload_in;
        uint32_t n_t = 0, n_kept = 0;
        if (payload_in_bytes < 8) return -1;
        std::memcpy(&n_t, in + 0, 4);
        std::memcpy(&n_kept, in + 4, 4);
        if (n_t != n_total) return -1;
        if ((size_t)n_kept > (payload_in_bytes - 8) / 8) return -1;
        const uint8_t* entries = in + 8;
        for (uint32_t k = 0; k < n_kept; ++k) {
            uint32_t idx = 0; uint16_t val = 0;
            std::memcpy(&idx, entries + (size_t)k * 8 + 0, 4);
            std::memcpy(&val, entries + (size_t)k * 8 + 4, 2);
            if (idx >= n_total) return -1;
            dense_out[idx] += f16_to_f32_gloo(val);
        }
        /* Receive each peer's payload, unpack onto dense_out. */
        for (int r = 1; r < world_size; ++r) {
            uint32_t peer_bytes = 0;
            if (!read_all(s->peer_conns[r], &peer_bytes, 4)) return -1;
            if (peer_bytes < 8) return -1;
            std::vector<uint8_t> peer(peer_bytes);
            if (!read_all(s->peer_conns[r], peer.data(), peer_bytes)) return -1;
            uint32_t p_n = 0, p_kept = 0;
            std::memcpy(&p_n, peer.data() + 0, 4);
            std::memcpy(&p_kept, peer.data() + 4, 4);
            if (p_n != n_total) return -1;
            if ((size_t)p_kept > ((size_t)peer_bytes - 8) / 8) return -1;
            const uint8_t* pe = peer.data() + 8;
            for (uint32_t k = 0; k < p_kept; ++k) {
                uint32_t idx = 0; uint16_t val = 0;
                std::memcpy(&idx, pe + (size_t)k * 8 + 0, 4);
                std::memcpy(&val, pe + (size_t)k * 8 + 4, 2);
                if (idx >= n_total) return -1;
                dense_out[idx] += f16_to_f32_gloo(val);
            }
        }
        /* Broadcast dense_out fp32 to all peers (simplest scheme; could
         * re-sparsify for the broadcast too, but at this point the merged
         * vector has ~world_size*0.001*N entries which is no longer
         * usefully sparse). */
        for (int r = 1; r < world_size; ++r) {
            if (!write_all(s->peer_conns[r], dense_out, dense_bytes)) return -1;
        }
    } else {
        /* Send my payload (length-prefixed). */
        if (payload_in_bytes > 0xffffffffu) return -1;
        uint32_t payload_bytes = (uint32_t)payload_in_bytes;
        if (!write_all(s->peer_conns[0], &payload_bytes, 4)) return -1;
        if (!write_all(s->peer_conns[0], payload_in, payload_in_bytes)) return -1;
        /* Receive merged dense vector from rank 0. */
        if (!read_all(s->peer_conns[0], dense_out, dense_bytes)) return -1;
    }
    return 0;
}

#else  /* !TC_GLOO_AVAILABLE */

struct GlooState {};

extern "C" TC_GLOO_HIDDEN GlooState* tc_gloo_init(int, int, const char*) { return nullptr; }
extern "C" TC_GLOO_HIDDEN void tc_gloo_destroy(GlooState*) {}
extern "C" TC_GLOO_HIDDEN int  tc_gloo_allreduce_f32_sum(GlooState*, int, int, float*, size_t) { return -1; }
extern "C" TC_GLOO_HIDDEN int  tc_gloo_allreduce_f16_sum(GlooState*, int, int, uint16_t*, size_t) { return -1; }
extern "C" TC_GLOO_HIDDEN int  tc_gloo_allreduce_f32_min(GlooState*, int, int, float*, size_t) { return -1; }
extern "C" TC_GLOO_HIDDEN int  tc_gloo_allreduce_f32_max(GlooState*, int, int, float*, size_t) { return -1; }
extern "C" TC_GLOO_HIDDEN int  tc_gloo_broadcast_f32(GlooState*, int, int, int, float*, size_t) { return -1; }
extern "C" TC_GLOO_HIDDEN int  tc_gloo_broadcast_any_root(GlooState*, int, int, int, void*, size_t) { return -1; }
extern "C" TC_GLOO_HIDDEN int  tc_gloo_allgather(GlooState*, int, int, void*, size_t) { return -1; }
extern "C" TC_GLOO_HIDDEN int  tc_gloo_sparse_allreduce(GlooState*, int, int, const void*, size_t, float*, size_t) { return -1; }
extern "C" TC_GLOO_HIDDEN int  tc_gloo_barrier(GlooState*, int, int) { return -1; }

#endif
