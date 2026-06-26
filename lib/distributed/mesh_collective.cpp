/*
 * tensorcore — Mesh collectives implementation.
 *
 * Layered on top of tc_remote_tensor_fetch (lib/distributed/remote_tensor.cpp).
 * The strategy is centralised rank-0 coordination — every rank registers
 * its buffer under a per-collective name; rank 0 fetches all peers'
 * buffers, applies the reduction in fp32, registers the result under a
 * "result" name, and every rank fetches it.
 *
 * Correct for any n_peers; bandwidth = 2(n-1) * count * 4 bytes through
 * rank 0 (not optimal). A ring AllReduce would change the inner loop to
 * 2(n-1) rounds each shipping count/n bytes — same total volume but
 * spread across links. Public API stays the same.
 *
 * Naming convention for transient buffers:
 *   "_mesh/<op>/<round>/<rank>"   — per-rank input
 *   "_mesh/<op>/<round>/result"   — reduced output
 * The round counter is per-group and monotonic so concurrent collectives
 * don't collide. Counter increments inside the group lock.
 */

#include "tensorcore/mesh_collective.h"
#include "tensorcore/remote_tensor.h"

#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct tc_mesh_group {
    tc_context*       ctx;
    int32_t           n_peers;
    int32_t           my_rank;
    tc_remote_ctx*    server;        /* one bound server (peer_urls[my_rank]) */
    tc_remote_ctx*    client;        /* one client context for outbound fetches */
    std::vector<int>  peer_ids;      /* peer_ids[r] = client peer id for rank r;
                                        peer_ids[my_rank] = -1 (loopback unused) */
    std::vector<std::string> urls;
    std::mutex        round_mu;
    uint64_t          round_counter; /* per-group monotonic */
    uint64_t          bytes_shipped;

    /* Per-collective result snapshots. The remote_tensor registry is
     * pointer-based, so reusing the caller's buf across rounds would
     * have later registrations alias earlier ones (since they all point
     * at the same `buf`). We memcpy the result/broadcast/allgather
     * payload into one of these vectors and register THAT — the vector
     * lives until group shutdown, so late-arriving peers always read
     * the correct round's data. */
    std::mutex        cache_mu;
    std::vector<std::vector<uint8_t>> snapshot_cache;
};

namespace {

size_t dtype_size(tc_coll_dtype_t d) {
    switch (d) {
        case TC_COLL_DTYPE_F32: return 4;
        default: return 0;
    }
}

/* Lock around the round counter to allow concurrent users of one group. */
uint64_t next_round(tc_mesh_group_t* g) {
    std::lock_guard<std::mutex> lk(g->round_mu);
    return g->round_counter++;
}

std::string mk_name(const char* op, uint64_t round, int rank) {
    std::string s = "_mesh/";
    s += op;
    s += "/";
    s += std::to_string(round);
    s += "/";
    if (rank < 0) s += "result";
    else          s += std::to_string(rank);
    return s;
}

/* Retry-wrap tc_remote_tensor_fetch: the producer rank may not have
 * registered the named tensor yet (concurrent participants in a
 * centralised collective don't share a barrier). Retry with backoff
 * up to ~5 s before propagating the error. Idempotent — the underlying
 * fetch is a stateless TCP request that re-runs cleanly. */
tc_status_t retry_fetch(tc_remote_ctx* cli, int peer_id, const char* name,
                         void* dst, size_t bytes) {
    /* Cumulative budget ≈ 30 s across 40 attempts with capped backoff.
     * Adequate for tests + production where collective participants
     * launch within a few seconds of each other. */
    int delay_ms = 5;
    for (int attempt = 0; attempt < 40; ++attempt) {
        const tc_status_t s = tc_remote_tensor_fetch(cli, peer_id, name, dst, bytes);
        if (s == TC_OK) return TC_OK;
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        if (delay_ms < 1000) delay_ms *= 2;
    }
    return tc_remote_tensor_fetch(cli, peer_id, name, dst, bytes);
}

/* Snapshot `bytes` of data from `src` into the group's owned cache,
 * then register the cached copy under `name`. The cache lives until
 * group_shutdown, so the registration stays valid even after the
 * caller's source buffer changes (which it WILL, the next round). */
tc_status_t snapshot_and_register(tc_mesh_group_t* g, const char* name,
                                   const void* src, size_t bytes) {
    std::vector<uint8_t> copy(bytes);
    std::memcpy(copy.data(), src, bytes);
    void* stable_ptr;
    {
        std::lock_guard<std::mutex> lk(g->cache_mu);
        g->snapshot_cache.emplace_back(std::move(copy));
        stable_ptr = g->snapshot_cache.back().data();
    }
    return tc_remote_register_tensor(g->server, name, stable_ptr, bytes);
}

/* Apply reduction op into accum from src. fp32 only at the moment.
 * Enum values come from distributed.h: SUM=0, AVG=1, MAX=2, MIN=3.
 * AVG is the same as SUM here; the caller divides by n_peers after. */
void reduce_inplace_f32(float* accum, const float* src, size_t count,
                         tc_reduce_op_t op) {
    switch (op) {
        case TC_REDUCE_SUM:
        case TC_REDUCE_AVG:
            for (size_t i = 0; i < count; ++i) accum[i] += src[i];
            break;
        case TC_REDUCE_MAX:
            for (size_t i = 0; i < count; ++i) {
                if (src[i] > accum[i]) accum[i] = src[i];
            }
            break;
        case TC_REDUCE_MIN:
            for (size_t i = 0; i < count; ++i) {
                if (src[i] < accum[i]) accum[i] = src[i];
            }
            break;
    }
}

}  // namespace

extern "C" tc_status_t tc_mesh_group_init(tc_context* ctx,
                                          int32_t n_peers, int32_t my_rank,
                                          const char* const* peer_urls,
                                          tc_mesh_group_t** out) {
    if (!ctx || !peer_urls || !out) return TC_ERR_INVALID_ARG;
    if (n_peers <= 0 || my_rank < 0 || my_rank >= n_peers) return TC_ERR_INVALID_ARG;

    auto* g = new tc_mesh_group_t{};
    g->ctx = ctx;
    g->n_peers = n_peers;
    g->my_rank = my_rank;
    g->server = nullptr;
    g->client = nullptr;
    g->peer_ids.assign((size_t)n_peers, -1);
    g->urls.reserve((size_t)n_peers);
    g->round_counter = 0;
    g->bytes_shipped = 0;
    for (int r = 0; r < n_peers; ++r) g->urls.emplace_back(peer_urls[r]);

    /* Bind local server. */
    tc_status_t s = tc_remote_init(ctx, TC_REMOTE_ROLE_WEIGHT_SERVER,
                                    peer_urls[my_rank], &g->server);
    if (s != TC_OK) { delete g; return s; }

    /* Client context for outbound fetches. */
    s = tc_remote_init(ctx, TC_REMOTE_ROLE_COMPUTE_CLIENT, nullptr, &g->client);
    if (s != TC_OK) {
        tc_remote_shutdown(g->server);
        delete g; return s;
    }

    /* Connect to every other peer. Peers may not have bound their server
     * yet (rendezvous race); retry with exponential backoff up to ~10 s
     * before giving up. This lets startup proceed without an external
     * barrier — adequate for the test harness and for production
     * launches where peers come up within a few seconds of each other. */
    for (int r = 0; r < n_peers; ++r) {
        if (r == my_rank) continue;
        int pid = -1;
        int delay_ms = 50;
        for (int attempt = 0; attempt < 12; ++attempt) {
            pid = tc_remote_connect(g->client, peer_urls[r]);
            if (pid >= 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            if (delay_ms < 2000) delay_ms *= 2;
        }
        if (pid < 0) {
            tc_remote_shutdown(g->server);
            tc_remote_shutdown(g->client);
            delete g;
            return TC_ERR_INTERNAL;
        }
        g->peer_ids[r] = pid;
    }

    *out = g;
    return TC_OK;
}

extern "C" tc_status_t tc_mesh_group_shutdown(tc_mesh_group_t* g) {
    if (!g) return TC_ERR_INVALID_ARG;
    if (g->client) tc_remote_shutdown(g->client);
    if (g->server) tc_remote_shutdown(g->server);
    delete g;
    return TC_OK;
}

extern "C" uint64_t tc_mesh_total_bytes(const tc_mesh_group_t* g) {
    return g ? g->bytes_shipped : 0;
}

/* ---- Internal accessors for sibling .cpp files in this directory.
 *      Declared in mesh_internal.h, NOT exported via public headers. */

extern "C" tc_remote_ctx* tc_mesh_internal_server(tc_mesh_group_t* g) {
    return g ? g->server : nullptr;
}
extern "C" tc_remote_ctx* tc_mesh_internal_client(tc_mesh_group_t* g) {
    return g ? g->client : nullptr;
}
extern "C" int tc_mesh_internal_peer_id(tc_mesh_group_t* g, int32_t rank) {
    if (!g) return -1;
    if (rank < 0 || rank >= g->n_peers) return -1;
    if (rank == g->my_rank) return -1;
    return g->peer_ids[(size_t)rank];
}
extern "C" int32_t tc_mesh_internal_my_rank(tc_mesh_group_t* g) {
    return g ? g->my_rank : -1;
}
extern "C" int32_t tc_mesh_internal_n_peers(tc_mesh_group_t* g) {
    return g ? g->n_peers : 0;
}
extern "C" void tc_mesh_internal_add_bytes(tc_mesh_group_t* g, uint64_t bytes) {
    if (g) g->bytes_shipped += bytes;
}

extern "C" tc_status_t tc_mesh_allreduce(tc_mesh_group_t* g,
                                          void* buf, size_t count,
                                          tc_coll_dtype_t dtype,
                                          tc_reduce_op_t op) {
    if (!g || !buf || count == 0) return TC_ERR_INVALID_ARG;
    if (dtype != TC_COLL_DTYPE_F32) return TC_ERR_UNSUPPORTED_DTYPE;

    const size_t bytes = count * dtype_size(dtype);
    const uint64_t r = next_round(g);
    const std::string my_name = mk_name("ar", r, g->my_rank);
    const std::string out_name = mk_name("ar", r, -1);

    /* Every rank publishes its input — snapshot-and-register so that
     * reusing `buf` for later rounds doesn't change earlier rounds'
     * data behind the registry's back. */
    tc_status_t s = snapshot_and_register(g, my_name.c_str(), buf, bytes);
    if (s != TC_OK) return s;

    if (g->my_rank == 0) {
        /* Rank 0 fetches all peers and reduces into its own buf. */
        std::vector<float> tmp(count);
        for (int rk = 1; rk < g->n_peers; ++rk) {
            const std::string name = mk_name("ar", r, rk);
            const int pid = g->peer_ids[rk];
            const tc_status_t fs = retry_fetch(g->client, pid,
                                                          name.c_str(),
                                                          tmp.data(), bytes);
            if (fs != TC_OK) {
                tc_remote_unregister_tensor(g->server, my_name.c_str());
                return fs;
            }
            g->bytes_shipped += bytes;
            reduce_inplace_f32(static_cast<float*>(buf), tmp.data(), count, op);
        }
        /* AVG: divide by n_peers after the sum. */
        if (op == TC_REDUCE_AVG && g->n_peers > 1) {
            const float inv_n = 1.0f / static_cast<float>(g->n_peers);
            float* b = static_cast<float*>(buf);
            for (size_t i = 0; i < count; ++i) b[i] *= inv_n;
        }
        /* Publish the reduced result under the shared "result" name.
         * Snapshot so subsequent rounds' buf updates don't corrupt it. */
        s = snapshot_and_register(g, out_name.c_str(), buf, bytes);
        if (s != TC_OK) {
            tc_remote_unregister_tensor(g->server, my_name.c_str());
            return s;
        }
    }

    /* Non-root ranks fetch the reduced result from rank 0. */
    if (g->my_rank != 0) {
        const int root_pid = g->peer_ids[0];
        const tc_status_t fs = retry_fetch(g->client, root_pid,
                                            out_name.c_str(),
                                            buf, bytes);
        if (fs != TC_OK) {
            tc_remote_unregister_tensor(g->server, my_name.c_str());
            return fs;
        }
        g->bytes_shipped += bytes;
    }

    /* DO NOT unregister: peers that haven't fetched yet would see
     * "not found" and fail. The round counter makes names unique so
     * stale entries don't collide; cost is O(per-collective) memory
     * which a long-running daemon can sweep via the group shutdown. */
    return TC_OK;
}

extern "C" tc_status_t tc_mesh_broadcast(tc_mesh_group_t* g,
                                          void* buf, size_t count,
                                          tc_coll_dtype_t dtype,
                                          int32_t root_rank) {
    if (!g || !buf || count == 0) return TC_ERR_INVALID_ARG;
    if (dtype != TC_COLL_DTYPE_F32) return TC_ERR_UNSUPPORTED_DTYPE;
    if (root_rank < 0 || root_rank >= g->n_peers) return TC_ERR_INVALID_ARG;

    const size_t bytes = count * dtype_size(dtype);
    const uint64_t r = next_round(g);
    const std::string name = mk_name("bc", r, root_rank);

    if (g->my_rank == root_rank) {
        const tc_status_t s = snapshot_and_register(g, name.c_str(),
                                                     buf, bytes);
        if (s != TC_OK) return s;
        /* Wait for all non-root ranks to fetch — there's no explicit
         * sync, but unregister is bounded below by the slowest fetch.
         * Simple correctness move: rely on caller's barrier discipline
         * or leave registration for the lifetime of the next collective
         * (next_round bumps the name). We unregister at end. */
    } else {
        const int pid = g->peer_ids[root_rank];
        const tc_status_t fs = retry_fetch(g->client, pid,
                                                      name.c_str(),
                                                      buf, bytes);
        if (fs != TC_OK) return fs;
        g->bytes_shipped += bytes;
    }

    /* Leave the broadcast registration for late consumers to fetch;
     * round counter makes the name unique. */
    return TC_OK;
}

extern "C" tc_status_t tc_mesh_allgather(tc_mesh_group_t* g,
                                          const void* send_buf, size_t send_count,
                                          void* recv_buf,
                                          tc_coll_dtype_t dtype) {
    if (!g || !send_buf || !recv_buf || send_count == 0) return TC_ERR_INVALID_ARG;
    if (dtype != TC_COLL_DTYPE_F32) return TC_ERR_UNSUPPORTED_DTYPE;

    const size_t bytes_per = send_count * dtype_size(dtype);
    const uint64_t r = next_round(g);
    const std::string my_name = mk_name("ag", r, g->my_rank);

    /* Every rank publishes its slice — snapshot for the same reason as
     * allreduce (later rounds reusing send_buf would alias). */
    tc_status_t s = snapshot_and_register(g, my_name.c_str(),
                                            send_buf, bytes_per);
    if (s != TC_OK) return s;

    /* Copy our own slice into the recv buffer directly. */
    std::memcpy(static_cast<char*>(recv_buf) + (size_t)g->my_rank * bytes_per,
                send_buf, bytes_per);

    /* Fetch every other rank's slice. */
    for (int rk = 0; rk < g->n_peers; ++rk) {
        if (rk == g->my_rank) continue;
        const std::string name = mk_name("ag", r, rk);
        const int pid = g->peer_ids[rk];
        void* dst = static_cast<char*>(recv_buf) + (size_t)rk * bytes_per;
        const tc_status_t fs = retry_fetch(g->client, pid,
                                                      name.c_str(),
                                                      dst, bytes_per);
        if (fs != TC_OK) {
            tc_remote_unregister_tensor(g->server, my_name.c_str());
            return fs;
        }
        g->bytes_shipped += bytes_per;
    }
    /* Same rationale as allreduce: keep registrations alive past this call
     * so the slower-arriving peer's fetch can succeed. */
    return TC_OK;
}
