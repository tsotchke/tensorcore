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
 *     allreduce  = rank-0 brokered reduction + broadcast.
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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
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
    const auto colon = body.rfind(':');
    if (colon == std::string::npos) return false;
    *host = body.substr(0, colon);
    char* end = nullptr;
    const std::string port_str = body.substr(colon + 1);
    const long parsed = std::strtol(port_str.c_str(), &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > 65535) return false;
    *port = (uint16_t)parsed;
    return !host->empty();
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

    /* Retry on transient failures (rank 0 may not be listening yet). */
    for (int retries = 30; retries >= 0; --retries) {
        int fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (fd < 0) { ::freeaddrinfo(res); return -1; }
        int yes = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
        if (::connect(fd, res->ai_addr, res->ai_addrlen) == 0) {
            ::freeaddrinfo(res);
            return fd;
        }
        const int saved_errno = errno;
        ::close(fd);
        if ((saved_errno == ECONNREFUSED || saved_errno == EHOSTUNREACH) && retries > 0) {
            usleep(100 * 1000);
            continue;
        }
        ::freeaddrinfo(res);
        return -1;
    }
    ::freeaddrinfo(res);
    return -1;
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

void close_gloo_state(GlooState* s) {
    if (!s) return;
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
    if (!parse_rendezvous(rendezvous_url, &host, &port)) return nullptr;

    auto* s = new (std::nothrow) GlooState();
    if (!s) return nullptr;
    s->peer_conns.assign((size_t)world_size, -1);

    if (rank == 0) {
        /* Listen + accept all others. */
        s->rendez_listen_fd = tcp_listen(port, &s->self_host);
        if (s->rendez_listen_fd < 0) { close_gloo_state(s); return nullptr; }
        for (int r = 1; r < world_size; ++r) {
            sockaddr_in peer_addr = {};
            socklen_t alen = sizeof(peer_addr);
            int fd = ::accept(s->rendez_listen_fd, (sockaddr*)&peer_addr, &alen);
            if (fd < 0) { close_gloo_state(s); return nullptr; }
            /* First message from peer: its rank (uint32 LE). */
            uint32_t peer_rank = 0;
            if (!read_all(fd, &peer_rank, 4) ||
                peer_rank == 0 ||
                peer_rank >= (uint32_t)world_size ||
                s->peer_conns[peer_rank] >= 0) {
                ::close(fd); close_gloo_state(s); return nullptr;
            }
            int yes = 1;
            ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
            s->peer_conns[peer_rank] = fd;
        }
    } else {
        /* Connect to rank 0. */
        int fd = tcp_connect(host, port);
        if (fd < 0) { close_gloo_state(s); return nullptr; }
        uint32_t self_rank = (uint32_t)rank;
        if (!write_all(fd, &self_rank, 4)) { ::close(fd); close_gloo_state(s); return nullptr; }
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
    close_gloo_state(s);
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

/* ------------------------------------------------------------------------
 * MIN / MAX reductions (rank-0-brokered, fp32).
 * ------------------------------------------------------------------------ */

extern "C" TC_GLOO_HIDDEN int tc_gloo_allreduce_f32_min(GlooState* s, int world_size, int rank,
                                                         float* data, size_t n) {
    if (world_size <= 1) return 0;
    const size_t bytes = n * sizeof(float);
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
    const size_t bytes = n * sizeof(float);
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
    if (rank == 0) {
        /* My slice is already at out + 0; receive each other rank's slice. */
        for (int r = 1; r < world_size; ++r) {
            if (!read_all(s->peer_conns[r], full + (size_t)r * bytes_per_rank, bytes_per_rank)) {
                return -1;
            }
        }
        /* Broadcast the full concatenated buffer. */
        const size_t total = (size_t)world_size * bytes_per_rank;
        for (int r = 1; r < world_size; ++r) {
            if (!write_all(s->peer_conns[r], full, total)) return -1;
        }
    } else {
        /* Send my slice (currently at out + rank * bytes_per_rank). */
        if (!write_all(s->peer_conns[0], full + (size_t)rank * bytes_per_rank, bytes_per_rank)) {
            return -1;
        }
        /* Receive the full concatenated buffer. */
        const size_t total = (size_t)world_size * bytes_per_rank;
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
    /* Wire format on rank-0 inbound: uint32 payload_bytes, then payload. */
    if (rank == 0) {
        /* Start the dense accumulator zeroed. */
        std::memset(dense_out, 0, n_total * sizeof(float));
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
        const size_t total_bytes = n_total * sizeof(float);
        for (int r = 1; r < world_size; ++r) {
            if (!write_all(s->peer_conns[r], dense_out, total_bytes)) return -1;
        }
    } else {
        /* Send my payload (length-prefixed). */
        if (payload_in_bytes > 0xffffffffu) return -1;
        uint32_t payload_bytes = (uint32_t)payload_in_bytes;
        if (!write_all(s->peer_conns[0], &payload_bytes, 4)) return -1;
        if (!write_all(s->peer_conns[0], payload_in, payload_in_bytes)) return -1;
        /* Receive merged dense vector from rank 0. */
        if (!read_all(s->peer_conns[0], dense_out, n_total * sizeof(float))) return -1;
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
