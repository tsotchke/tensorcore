/*
 * tensorcore — Mesh collectives implementation.
 *
 * Layered on top of tc_remote_tensor_fetch (lib/distributed/remote_tensor.cpp).
 * AllReduce uses centralised rank-0 reduction; Broadcast and AllGather use the
 * natural producer topology. Every operation finishes with a distributed
 * completion-token barrier. That barrier proves all payload fetches are done,
 * allowing the payload registration to be removed immediately. Only the
 * current round's tiny completion token survives until the next ordered call.
 *
 * Correct for any n_peers; bandwidth = 2(n-1) * count * 4 bytes through
 * rank 0 (not optimal). A ring AllReduce would change the inner loop to
 * 2(n-1) rounds each shipping count/n bytes — same total volume but
 * spread across links. Public API stays the same.
 *
 * Naming convention for transient buffers:
 *   "_mesh/<op>/<round>/<rank>"   — per-rank input
 *   "_mesh/<op>/<round>/result"   — reduced output
 * The round counter is per-group and monotonic. Public ordered calls are
 * serialized for thread safety and all ranks must invoke them in the same
 * order, as with MPI collectives.
 */

#include "tensorcore/mesh_collective.h"
#include "tensorcore/remote_tensor.h"
#include "remote_tensor_internal.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct tc_mesh_snapshot {
    uint64_t round;
    bool ordered;
    std::vector<uint8_t> bytes;
};

struct tc_mesh_group {
    tc_context*       ctx;
    int32_t           n_peers;
    int32_t           my_rank;
    tc_remote_ctx*    server;        /* one bound server (peer_urls[my_rank]) */
    tc_remote_ctx*    client;        /* one client context for outbound fetches */
    std::vector<int>  peer_ids;      /* peer_ids[r] = client peer id for rank r;
                                        peer_ids[my_rank] = -1 (loopback unused) */
    std::vector<std::string> urls;
    std::vector<std::string> peer_identities;
    bool              authenticated;
    bool              decentralized_allreduce;
    std::mutex        peer_mu;
    std::mutex        collective_mu; /* ordered API is one collective at a time */
    std::mutex        tagged_mu;
    std::unordered_set<uint64_t> active_tagged;
    std::mutex        round_mu;
    uint64_t          round_counter; /* per-group monotonic */
    std::atomic<uint64_t> bytes_shipped;
    uint32_t          shutdown_token;

    /* Per-collective snapshots. The remote_tensor registry is
     * pointer-based, so reusing the caller's buf across rounds would
     * have later registrations alias earlier ones (since they all point
     * at the same `buf`). We memcpy the result/broadcast/allgather
     * payload into owned storage and register that stable copy. Completion
     * barriers allow payloads to be reclaimed in the same call; one round of
     * tiny completion tokens remains until the next barrier completes. */
    std::mutex        cache_mu;
    std::unordered_map<std::string, tc_mesh_snapshot> snapshot_cache;
};

namespace {

bool mesh_debug_enabled() {
    static const bool enabled = std::getenv("TC_MESH_DEBUG") != nullptr;
    return enabled;
}

void mesh_debug(tc_mesh_group_t* g, const char* stage,
                const char* op, uint64_t round, int peer = -1) {
    if (!mesh_debug_enabled()) return;
    std::fprintf(stderr,
                 "mesh rank=%d round=%llu op=%s stage=%s peer=%d\n",
                 g ? g->my_rank : -1,
                 (unsigned long long)round, op, stage, peer);
    std::fflush(stderr);
}

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
tc_status_t retry_fetch(tc_mesh_group_t* g, int rank, const char* name,
                         void* dst, size_t bytes) {
    /* Cumulative budget ≈ 30 s across 40 attempts with capped backoff.
     * Adequate for tests + production where collective participants
     * launch within a few seconds of each other. */
    int delay_ms = 5;
    for (int attempt = 0; attempt < 40; ++attempt) {
        int peer_id = -1;
        {
            std::lock_guard<std::mutex> lk(g->peer_mu);
            peer_id = g->peer_ids[(size_t)rank];
        }
        const tc_status_t s = tc_remote_tensor_fetch(
            g->client, peer_id, name, dst, bytes);
        if (s == TC_OK) return TC_OK;
        if (s == TC_ERR_AUTH || s == TC_ERR_ABI_MISMATCH) return s;
        if (s == TC_ERR_INTERNAL) {
            const char* identity = g->authenticated
                ? g->peer_identities[(size_t)rank].c_str() : nullptr;
            const tc_status_t reconnect = tc_remote_internal_reconnect(
                g->client, peer_id, g->urls[(size_t)rank].c_str(), identity);
            if (reconnect == TC_ERR_AUTH || reconnect == TC_ERR_ABI_MISMATCH)
                return reconnect;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        if (delay_ms < 1000) delay_ms *= 2;
    }
    int peer_id = -1;
    {
        std::lock_guard<std::mutex> lk(g->peer_mu);
        peer_id = g->peer_ids[(size_t)rank];
    }
    return tc_remote_tensor_fetch(g->client, peer_id, name, dst, bytes);
}

/* Snapshot `bytes` of data from `src` into the group's owned cache,
 * then register the cached copy under `name`. The cache lives until
 * group_shutdown, so the registration stays valid even after the
 * caller's source buffer changes (which it WILL, the next round). */
tc_status_t snapshot_and_register(tc_mesh_group_t* g, const char* name,
                                   const void* src, size_t bytes,
                                   uint64_t round,
                                   bool ordered = true) {
    std::vector<uint8_t> copy(bytes);
    std::memcpy(copy.data(), src, bytes);
    void* stable_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lk(g->cache_mu);
        auto inserted = g->snapshot_cache.emplace(
            std::string(name), tc_mesh_snapshot{round, ordered, std::move(copy)});
        if (!inserted.second) return TC_ERR_BUSY;
        stable_ptr = inserted.first->second.bytes.data();
    }
    const tc_status_t status =
        tc_remote_register_tensor(g->server, name, stable_ptr, bytes);
    if (status != TC_OK) {
        std::lock_guard<std::mutex> lk(g->cache_mu);
        g->snapshot_cache.erase(name);
    }
    return status;
}

tc_status_t release_snapshot(tc_mesh_group_t* g, const std::string& name) {
    const tc_status_t status =
        tc_remote_unregister_tensor(g->server, name.c_str());
    if (status != TC_OK) return status;
    std::lock_guard<std::mutex> lk(g->cache_mu);
    g->snapshot_cache.erase(name);
    return TC_OK;
}

tc_status_t reclaim_before(tc_mesh_group_t* g, uint64_t round) {
    std::vector<std::string> names;
    {
        std::lock_guard<std::mutex> lk(g->cache_mu);
        for (const auto& item : g->snapshot_cache) {
            if (item.second.ordered && item.second.round < round)
                names.push_back(item.first);
        }
    }
    for (const auto& name : names) {
        const tc_status_t status = release_snapshot(g, name);
        if (status != TC_OK) return status;
    }
    return TC_OK;
}

bool begin_tagged(tc_mesh_group_t* g, uint64_t collective_id) {
    std::lock_guard<std::mutex> tag_lk(g->tagged_mu);
    if (g->active_tagged.count(collective_id)) return false;
    {
        std::lock_guard<std::mutex> cache_lk(g->cache_mu);
        for (const auto& item : g->snapshot_cache) {
            if (!item.second.ordered && item.second.round == collective_id)
                return false;
        }
    }
    g->active_tagged.insert(collective_id);
    return true;
}

void end_tagged(tc_mesh_group_t* g, uint64_t collective_id) {
    std::lock_guard<std::mutex> lk(g->tagged_mu);
    g->active_tagged.erase(collective_id);
}

tc_status_t release_tagged(tc_mesh_group_t* g, uint64_t collective_id) {
    {
        std::lock_guard<std::mutex> lk(g->tagged_mu);
        if (g->active_tagged.count(collective_id)) return TC_ERR_BUSY;
    }
    std::vector<std::string> names;
    {
        std::lock_guard<std::mutex> lk(g->cache_mu);
        for (const auto& item : g->snapshot_cache) {
            if (!item.second.ordered && item.second.round == collective_id)
                names.push_back(item.first);
        }
    }
    for (const auto& name : names) {
        const tc_status_t status = release_snapshot(g, name);
        if (status != TC_OK) return status;
    }
    return TC_OK;
}

tc_status_t collective_barrier(tc_mesh_group_t* g,
                               const char* op,
                               uint64_t round,
                               bool ordered = true) {
    const uint64_t token = round ^
        (UINT64_C(0x9e3779b97f4a7c15) * (uint64_t)(g->my_rank + 1));
    const std::string my_name = mk_name(op, round, g->my_rank);
    mesh_debug(g, "barrier_register", op, round);
    tc_status_t status = snapshot_and_register(
        g, my_name.c_str(), &token, sizeof(token), round, ordered);
    if (status != TC_OK) return status;
    for (int rank = 0; rank < g->n_peers; ++rank) {
        if (rank == g->my_rank) continue;
        const std::string peer_name = mk_name(op, round, rank);
        uint64_t peer_token = 0;
        mesh_debug(g, "barrier_fetch_begin", op, round, rank);
        status = retry_fetch(
            g, rank, peer_name.c_str(), &peer_token, sizeof(peer_token));
        if (status != TC_OK) return status;
        mesh_debug(g, "barrier_fetch_done", op, round, rank);
        const uint64_t expected = round ^
            (UINT64_C(0x9e3779b97f4a7c15) * (uint64_t)(rank + 1));
        if (peer_token != expected) return TC_ERR_INTERNAL;
        g->bytes_shipped.fetch_add(sizeof(peer_token), std::memory_order_relaxed);
    }
    return TC_OK;
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

static tc_status_t mesh_group_init_impl(
    tc_context* ctx, int32_t n_peers, int32_t my_rank,
    const char* const* peer_urls, const char* const* peer_identities,
    const tc_transport_auth_config* auth, tc_mesh_group_t** out) {
    if (!ctx || !peer_urls || !out) return TC_ERR_INVALID_ARG;
    *out = nullptr;
    if (n_peers <= 0 || my_rank < 0 || my_rank >= n_peers) return TC_ERR_INVALID_ARG;
    if ((auth == nullptr) != (peer_identities == nullptr)) return TC_ERR_INVALID_ARG;
    if (auth) {
        if (!auth->local_identity || !peer_identities[my_rank] ||
            std::strcmp(auth->local_identity, peer_identities[my_rank]) != 0)
            return TC_ERR_INVALID_ARG;
        for (int32_t r = 0; r < n_peers; ++r) {
            if (!peer_urls[r] || !peer_identities[r] || !peer_identities[r][0])
                return TC_ERR_INVALID_ARG;
        }
    }

    auto* g = new tc_mesh_group_t{};
    g->ctx = ctx;
    g->n_peers = n_peers;
    g->my_rank = my_rank;
    g->server = nullptr;
    g->client = nullptr;
    g->peer_ids.assign((size_t)n_peers, -1);
    g->urls.reserve((size_t)n_peers);
    g->peer_identities.reserve((size_t)n_peers);
    g->authenticated = auth != nullptr;
    const char* algorithm = std::getenv("TC_MESH_ALLREDUCE_ALGORITHM");
    if (algorithm && std::strcmp(algorithm, "centralized") == 0) {
        g->decentralized_allreduce = false;
    } else if (algorithm && std::strcmp(algorithm, "all_to_all") == 0) {
        g->decentralized_allreduce = true;
    } else if (algorithm && std::strcmp(algorithm, "auto") != 0) {
        delete g;
        return TC_ERR_INVALID_ARG;
    } else {
        g->decentralized_allreduce = n_peers >= 3;
    }
    g->round_counter = 0;
    g->bytes_shipped = 0;
    g->shutdown_token = static_cast<uint32_t>(my_rank);
    for (int r = 0; r < n_peers; ++r) {
        g->urls.emplace_back(peer_urls[r]);
        g->peer_identities.emplace_back(
            peer_identities ? peer_identities[r] : "");
    }

    /* Bind local server. */
    tc_status_t s = auth
        ? tc_remote_init_authenticated(ctx, TC_REMOTE_ROLE_WEIGHT_SERVER,
                                       peer_urls[my_rank], auth, &g->server)
        : tc_remote_init(ctx, TC_REMOTE_ROLE_WEIGHT_SERVER,
                         peer_urls[my_rank], &g->server);
    if (s != TC_OK) { delete g; return s; }

    /* Client context for outbound fetches. */
    s = auth
        ? tc_remote_init_authenticated(ctx, TC_REMOTE_ROLE_COMPUTE_CLIENT,
                                       nullptr, auth, &g->client)
        : tc_remote_init(ctx, TC_REMOTE_ROLE_COMPUTE_CLIENT, nullptr, &g->client);
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
            if (auth) {
                const tc_status_t connect_status =
                    tc_remote_connect_authenticated(g->client, peer_urls[r],
                                                    peer_identities[r], &pid);
                if (connect_status == TC_ERR_AUTH ||
                    connect_status == TC_ERR_ABI_MISMATCH) {
                    break;
                }
            } else {
                pid = tc_remote_connect(g->client, peer_urls[r]);
            }
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

extern "C" tc_status_t tc_mesh_group_init(tc_context* ctx,
                                          int32_t n_peers, int32_t my_rank,
                                          const char* const* peer_urls,
                                          tc_mesh_group_t** out) {
    return mesh_group_init_impl(ctx, n_peers, my_rank, peer_urls,
                                nullptr, nullptr, out);
}

extern "C" tc_status_t tc_mesh_group_init_authenticated(
    tc_context* ctx, int32_t n_peers, int32_t my_rank,
    const char* const* peer_urls, const char* const* peer_identities,
    const tc_transport_auth_config* auth, tc_mesh_group_t** out) {
    if (!auth || !peer_identities) {
        if (out) *out = nullptr;
        return TC_ERR_INVALID_ARG;
    }
    return mesh_group_init_impl(ctx, n_peers, my_rank, peer_urls,
                                peer_identities, auth, out);
}

extern "C" tc_status_t tc_mesh_group_shutdown(tc_mesh_group_t* g) {
    if (!g) return TC_ERR_INVALID_ARG;
    tc_status_t result = TC_OK;

    /* A collective may return on one rank just before another rank performs
     * its final fetch. Publish a readiness token and observe every peer's
     * token before closing outbound connections. The servers then wait for
     * those outbound clients to close naturally, so no rank tears down an
     * accepted socket while a peer is still crossing the barrier. */
    if (g->server && g->client && g->n_peers > 1) {
        const std::string my_name = mk_name("shutdown", 0, g->my_rank);
        result = tc_remote_register_tensor(g->server, my_name.c_str(),
                                           &g->shutdown_token,
                                           sizeof(g->shutdown_token));
        for (int rk = 0; result == TC_OK && rk < g->n_peers; ++rk) {
            if (rk == g->my_rank) continue;
            const std::string peer_name = mk_name("shutdown", 0, rk);
            uint32_t peer_token = 0;
            result = retry_fetch(g, rk, peer_name.c_str(), &peer_token,
                                 sizeof(peer_token));
            if (result == TC_OK && peer_token != static_cast<uint32_t>(rk))
                result = TC_ERR_INTERNAL;
        }
    }

    if (g->client) {
        const tc_status_t s = tc_remote_shutdown(g->client);
        if (result == TC_OK && s != TC_OK) result = s;
        g->client = nullptr;
    }
    if (g->server) {
        const tc_status_t drain =
            tc_remote_internal_wait_for_clients_closed(g->server, 5000);
        if (result == TC_OK && drain != TC_OK) result = drain;
        const tc_status_t s = tc_remote_shutdown(g->server);
        if (result == TC_OK && s != TC_OK) result = s;
        g->server = nullptr;
    }
    delete g;
    return result;
}

extern "C" uint64_t tc_mesh_total_bytes(const tc_mesh_group_t* g) {
    return g ? g->bytes_shipped.load(std::memory_order_relaxed) : 0;
}

extern "C" size_t tc_mesh_retained_snapshot_count(const tc_mesh_group_t* g) {
    if (!g) return 0;
    auto* mutable_group = const_cast<tc_mesh_group_t*>(g);
    std::lock_guard<std::mutex> lk(mutable_group->cache_mu);
    return mutable_group->snapshot_cache.size();
}

extern "C" const char* tc_mesh_allreduce_algorithm(const tc_mesh_group_t* g) {
    if (!g) return "none";
    return g->decentralized_allreduce ? "all_to_all" : "centralized";
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
    if (g) g->bytes_shipped.fetch_add(bytes, std::memory_order_relaxed);
}

static tc_status_t mesh_allreduce_round(tc_mesh_group_t* g,
                                        void* buf, size_t count,
                                        tc_coll_dtype_t dtype,
                                        tc_reduce_op_t op,
                                        uint64_t r,
                                        const char* name_prefix,
                                        bool ordered) {
    if (!g || !buf || count == 0) return TC_ERR_INVALID_ARG;
    if (dtype != TC_COLL_DTYPE_F32) return TC_ERR_UNSUPPORTED_DTYPE;

    const size_t bytes = count * dtype_size(dtype);
    tc_status_t s = TC_OK;
    const std::string my_name = mk_name(name_prefix, r, g->my_rank);
    const std::string out_name = mk_name(name_prefix, r, -1);
    mesh_debug(g, "input_register", name_prefix, r);

    /* Every rank publishes its input — snapshot-and-register so that
     * reusing `buf` for later rounds doesn't change earlier rounds'
     * data behind the registry's back. */
    s = snapshot_and_register(g, my_name.c_str(), buf, bytes, r, ordered);
    if (s != TC_OK) return s;

    if (g->decentralized_allreduce) {
        std::vector<float> tmp(count);
        for (int rank = 0; rank < g->n_peers; ++rank) {
            if (rank == g->my_rank) continue;
            const std::string name = mk_name(name_prefix, r, rank);
            mesh_debug(g, "input_fetch_begin", name_prefix, r, rank);
            const tc_status_t fetch = retry_fetch(
                g, rank, name.c_str(), tmp.data(), bytes);
            if (fetch != TC_OK) {
                release_snapshot(g, my_name);
                return fetch;
            }
            mesh_debug(g, "input_fetch_done", name_prefix, r, rank);
            g->bytes_shipped.fetch_add(bytes, std::memory_order_relaxed);
            reduce_inplace_f32(static_cast<float*>(buf), tmp.data(), count, op);
        }
        if (op == TC_REDUCE_AVG && g->n_peers > 1) {
            const float inv_n = 1.0f / static_cast<float>(g->n_peers);
            float* values = static_cast<float*>(buf);
            for (size_t index = 0; index < count; ++index)
                values[index] *= inv_n;
        }
    } else if (g->my_rank == 0) {
        /* Rank 0 fetches all peers and reduces into its own buf. */
        std::vector<float> tmp(count);
        for (int rk = 1; rk < g->n_peers; ++rk) {
            const std::string name = mk_name(name_prefix, r, rk);
            mesh_debug(g, "input_fetch_begin", name_prefix, r, rk);
            const tc_status_t fs = retry_fetch(
                g, rk, name.c_str(), tmp.data(), bytes);
            if (fs != TC_OK) {
                release_snapshot(g, my_name);
                return fs;
            }
            mesh_debug(g, "input_fetch_done", name_prefix, r, rk);
            g->bytes_shipped.fetch_add(bytes, std::memory_order_relaxed);
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
        mesh_debug(g, "result_register", name_prefix, r);
        s = snapshot_and_register(g, out_name.c_str(), buf, bytes, r, ordered);
        if (s != TC_OK) {
            release_snapshot(g, my_name);
            return s;
        }
    }

    /* Non-root ranks fetch the reduced result from rank 0. */
    if (!g->decentralized_allreduce && g->my_rank != 0) {
        mesh_debug(g, "result_fetch_begin", name_prefix, r, 0);
        const tc_status_t fs = retry_fetch(
            g, 0, out_name.c_str(), buf, bytes);
        if (fs != TC_OK) {
            release_snapshot(g, my_name);
            return fs;
        }
        mesh_debug(g, "result_fetch_done", name_prefix, r, 0);
        g->bytes_shipped.fetch_add(bytes, std::memory_order_relaxed);
    }

    const std::string done_prefix = std::string(name_prefix) + "-done";
    s = collective_barrier(g, done_prefix.c_str(), r, ordered);
    if (s != TC_OK) return s;
    s = release_snapshot(g, my_name);
    if (s != TC_OK) return s;
    if (!g->decentralized_allreduce && g->my_rank == 0) {
        s = release_snapshot(g, out_name);
        if (s != TC_OK) return s;
    }
    return ordered ? reclaim_before(g, r) : TC_OK;
}

extern "C" tc_status_t tc_mesh_allreduce(tc_mesh_group_t* g,
                                          void* buf, size_t count,
                                          tc_coll_dtype_t dtype,
                                          tc_reduce_op_t op) {
    if (!g) return TC_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> collective_lock(g->collective_mu);
    return mesh_allreduce_round(
        g, buf, count, dtype, op, next_round(g), "ar", true);
}

extern "C" tc_status_t tc_mesh_allreduce_tagged(
    tc_mesh_group_t* g, uint64_t collective_id,
    void* buf, size_t count, tc_coll_dtype_t dtype, tc_reduce_op_t op) {
    if (!g) return TC_ERR_INVALID_ARG;
    if (!begin_tagged(g, collective_id)) return TC_ERR_BUSY;
    const tc_status_t status = mesh_allreduce_round(
        g, buf, count, dtype, op, collective_id, "tar", false);
    end_tagged(g, collective_id);
    return status;
}

static tc_status_t mesh_broadcast_round(tc_mesh_group_t* g,
                                        void* buf, size_t count,
                                        tc_coll_dtype_t dtype,
                                        int32_t root_rank,
                                        uint64_t r,
                                        const char* name_prefix,
                                        bool ordered) {
    if (!g || !buf || count == 0) return TC_ERR_INVALID_ARG;
    if (dtype != TC_COLL_DTYPE_F32) return TC_ERR_UNSUPPORTED_DTYPE;
    if (root_rank < 0 || root_rank >= g->n_peers) return TC_ERR_INVALID_ARG;

    const size_t bytes = count * dtype_size(dtype);
    tc_status_t s = TC_OK;
    const std::string name = mk_name(name_prefix, r, root_rank);

    if (g->my_rank == root_rank) {
        s = snapshot_and_register(g, name.c_str(), buf, bytes, r, ordered);
        if (s != TC_OK) return s;
    } else {
        const tc_status_t fs = retry_fetch(
            g, root_rank, name.c_str(), buf, bytes);
        if (fs != TC_OK) return fs;
        g->bytes_shipped.fetch_add(bytes, std::memory_order_relaxed);
    }

    const std::string done_prefix = std::string(name_prefix) + "-done";
    s = collective_barrier(g, done_prefix.c_str(), r, ordered);
    if (s != TC_OK) return s;
    if (g->my_rank == root_rank) {
        s = release_snapshot(g, name);
        if (s != TC_OK) return s;
    }
    return ordered ? reclaim_before(g, r) : TC_OK;
}

extern "C" tc_status_t tc_mesh_broadcast(tc_mesh_group_t* g,
                                          void* buf, size_t count,
                                          tc_coll_dtype_t dtype,
                                          int32_t root_rank) {
    if (!g) return TC_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> collective_lock(g->collective_mu);
    return mesh_broadcast_round(
        g, buf, count, dtype, root_rank, next_round(g), "bc", true);
}

extern "C" tc_status_t tc_mesh_broadcast_tagged(
    tc_mesh_group_t* g, uint64_t collective_id,
    void* buf, size_t count, tc_coll_dtype_t dtype, int32_t root_rank) {
    if (!g) return TC_ERR_INVALID_ARG;
    if (!begin_tagged(g, collective_id)) return TC_ERR_BUSY;
    const tc_status_t status = mesh_broadcast_round(
        g, buf, count, dtype, root_rank, collective_id, "tbc", false);
    end_tagged(g, collective_id);
    return status;
}

static tc_status_t mesh_allgather_round(tc_mesh_group_t* g,
                                        const void* send_buf,
                                        size_t send_count,
                                        void* recv_buf,
                                        tc_coll_dtype_t dtype,
                                        uint64_t r,
                                        const char* name_prefix,
                                        bool ordered) {
    if (!g || !send_buf || !recv_buf || send_count == 0) return TC_ERR_INVALID_ARG;
    if (dtype != TC_COLL_DTYPE_F32) return TC_ERR_UNSUPPORTED_DTYPE;

    const size_t bytes_per = send_count * dtype_size(dtype);
    tc_status_t s = TC_OK;
    const std::string my_name = mk_name(name_prefix, r, g->my_rank);

    /* Every rank publishes its slice — snapshot for the same reason as
     * allreduce (later rounds reusing send_buf would alias). */
    s = snapshot_and_register(
        g, my_name.c_str(), send_buf, bytes_per, r, ordered);
    if (s != TC_OK) return s;

    /* Copy our own slice into the recv buffer directly. */
    std::memcpy(static_cast<char*>(recv_buf) + (size_t)g->my_rank * bytes_per,
                send_buf, bytes_per);

    /* Fetch every other rank's slice. */
    for (int rk = 0; rk < g->n_peers; ++rk) {
        if (rk == g->my_rank) continue;
        const std::string name = mk_name(name_prefix, r, rk);
        void* dst = static_cast<char*>(recv_buf) + (size_t)rk * bytes_per;
        const tc_status_t fs = retry_fetch(
            g, rk, name.c_str(), dst, bytes_per);
        if (fs != TC_OK) {
            release_snapshot(g, my_name);
            return fs;
        }
        g->bytes_shipped.fetch_add(bytes_per, std::memory_order_relaxed);
    }
    const std::string done_prefix = std::string(name_prefix) + "-done";
    s = collective_barrier(g, done_prefix.c_str(), r, ordered);
    if (s != TC_OK) return s;
    s = release_snapshot(g, my_name);
    if (s != TC_OK) return s;
    return ordered ? reclaim_before(g, r) : TC_OK;
}

extern "C" tc_status_t tc_mesh_allgather(tc_mesh_group_t* g,
                                          const void* send_buf, size_t send_count,
                                          void* recv_buf,
                                          tc_coll_dtype_t dtype) {
    if (!g) return TC_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> collective_lock(g->collective_mu);
    return mesh_allgather_round(
        g, send_buf, send_count, recv_buf, dtype,
        next_round(g), "ag", true);
}

extern "C" tc_status_t tc_mesh_allgather_tagged(
    tc_mesh_group_t* g, uint64_t collective_id,
    const void* send_buf, size_t send_count,
    void* recv_buf, tc_coll_dtype_t dtype) {
    if (!g) return TC_ERR_INVALID_ARG;
    if (!begin_tagged(g, collective_id)) return TC_ERR_BUSY;
    const tc_status_t status = mesh_allgather_round(
        g, send_buf, send_count, recv_buf, dtype,
        collective_id, "tag", false);
    end_tagged(g, collective_id);
    return status;
}

extern "C" tc_status_t tc_mesh_collective_release_tag(
    tc_mesh_group_t* g, uint64_t collective_id) {
    if (!g) return TC_ERR_INVALID_ARG;
    return release_tagged(g, collective_id);
}
