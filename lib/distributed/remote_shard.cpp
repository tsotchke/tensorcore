/*
 * tensorcore — Row-sharded tensor across N peers (owner-based routing).
 *
 * Math + ABI: include/tensorcore/remote_shard.h.
 *
 * Reuses the mesh_group's TCP transport (lib/distributed/remote_tensor.cpp)
 * via the private accessors in mesh_internal.h. Shards are first-class
 * named tensors on the transport ("_shard/<name>/<rank>") — the lifetime
 * is the caller's, so reusing the same name across rounds is fine: the
 * shard plan is fixed, only the bytes update. We use a per-(name, rank)
 * snapshot cache here too so the same buffer pointer can be re-registered
 * round after round without leaking state.
 *
 * Ownership is balanced block-row: rank r owns rows
 *   [r * rows / N,  (r+1) * rows / N).
 * That keeps the row count per rank within 1 of every other rank and
 * keeps a row's owner a one-line integer computation (no per-rank tables).
 */

#include "tensorcore/remote_shard.h"
#include "tensorcore/remote_tensor.h"
#include "tensorcore/status.h"
#include "mesh_internal.h"

#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

size_t shard_dtype_size(tc_coll_dtype_t d) {
    switch (d) {
        case TC_COLL_DTYPE_F32: return 4;
        default: return 0;
    }
}

std::string mk_shard_name(const char* name, int32_t rank) {
    std::string s = "_shard/";
    s += name;
    s += "/";
    s += std::to_string(rank);
    return s;
}

/* Backoff fetch — peers may not have registered the shard yet. Same
 * shape as mesh_collective::retry_fetch (deliberately not shared to
 * keep the dependency arrow one-way: shard.cpp → mesh_internal). */
tc_status_t retry_fetch_shard(tc_remote_ctx* cli, int peer_id,
                               const char* name, void* dst, size_t bytes) {
    int delay_ms = 5;
    for (int attempt = 0; attempt < 40; ++attempt) {
        const tc_status_t s = tc_remote_tensor_fetch(cli, peer_id, name,
                                                     dst, bytes);
        if (s == TC_OK) return TC_OK;
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        if (delay_ms < 1000) delay_ms *= 2;
    }
    return tc_remote_tensor_fetch(cli, peer_id, name, dst, bytes);
}

}  // namespace

extern "C" int32_t tc_remote_shard_owner(const tc_shard_plan_t* plan,
                                          int32_t row) {
    if (!plan || plan->rows <= 0 || plan->n_peers <= 0) return -1;
    if (row < 0 || row >= plan->rows) return -1;
    /* owner(r) = r * N / rows  (integer division, balanced block-row). */
    int64_t owner = (int64_t)row * (int64_t)plan->n_peers / (int64_t)plan->rows;
    if (owner >= plan->n_peers) owner = plan->n_peers - 1;
    return (int32_t)owner;
}

extern "C" void tc_remote_shard_local_range(const tc_shard_plan_t* plan,
                                             int32_t my_rank,
                                             int32_t* out_row_start,
                                             int32_t* out_row_end) {
    if (!plan || !out_row_start || !out_row_end) return;
    if (my_rank < 0 || my_rank >= plan->n_peers || plan->rows <= 0) {
        *out_row_start = 0; *out_row_end = 0; return;
    }
    /* Inverse of the balanced block-row owner formula:
     *   row_start(r) = ⌈r * rows / N⌉, row_end(r) = ⌈(r+1) * rows / N⌉.
     * Using ceiling division so the partition is a clean partition (no
     * gaps, no overlaps) AND consistent with tc_remote_shard_owner. */
    const int64_t N = plan->n_peers;
    const int64_t R = plan->rows;
    int64_t start = (((int64_t)my_rank) * R + (N - 1)) / N;
    int64_t end   = (((int64_t)my_rank + 1) * R + (N - 1)) / N;
    if (start > R) start = R;
    if (end   > R) end   = R;
    *out_row_start = (int32_t)start;
    *out_row_end   = (int32_t)end;
}

/* The transport registers by pointer (zero-copy on the server). To allow
 * callers to reuse the same name across rounds with different buffers
 * we snapshot the bytes into a per-(name) cache that we own for the
 * group's lifetime. */
struct shard_registry_t {
    std::mutex                                    mu;
    std::map<std::string, std::vector<uint8_t>>   snapshots;
};
static shard_registry_t* registry_for(tc_mesh_group_t* /*g*/) {
    /* One process-global registry keyed by full shard name; that's
     * sufficient because the name embeds the group's rank and the
     * caller's tensor name, which together are unique per (group,
     * tensor). Group teardown drops the entries we created (NOT
     * tracked here — they're freed at process exit; in practice the
     * mesh group lives for the whole training run). */
    static shard_registry_t g_reg;
    return &g_reg;
}

extern "C" tc_status_t tc_remote_shard_register(tc_mesh_group_t* g,
                                                 const tc_shard_plan_t* plan,
                                                 const char* name,
                                                 const void* my_buf) {
    if (!g || !plan || !name || !my_buf) return TC_ERR_INVALID_ARG;
    if (plan->n_peers != tc_mesh_internal_n_peers(g)) return TC_ERR_INVALID_ARG;
    if (plan->dtype != TC_COLL_DTYPE_F32) return TC_ERR_UNSUPPORTED_DTYPE;

    const int32_t my_rank = tc_mesh_internal_my_rank(g);
    int32_t lo, hi;
    tc_remote_shard_local_range(plan, my_rank, &lo, &hi);
    const size_t local_rows = (size_t)(hi - lo);
    const size_t local_bytes = local_rows * (size_t)plan->cols * shard_dtype_size(plan->dtype);

    const std::string full = mk_shard_name(name, my_rank);
    auto* reg = registry_for(g);
    std::vector<uint8_t> snap(local_bytes);
    if (local_bytes > 0) std::memcpy(snap.data(), my_buf, local_bytes);
    void* stable_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lk(reg->mu);
        auto& slot = reg->snapshots[full];
        slot = std::move(snap);
        stable_ptr = slot.data();
    }
    /* Re-register: unregister-then-register is the safe idempotent path
     * since tc_remote_register_tensor would reject a duplicate name. */
    tc_remote_unregister_tensor(tc_mesh_internal_server(g), full.c_str());
    return tc_remote_register_tensor(tc_mesh_internal_server(g), full.c_str(),
                                     stable_ptr, local_bytes);
}

extern "C" tc_status_t tc_remote_shard_get(tc_mesh_group_t* g,
                                            const tc_shard_plan_t* plan,
                                            const char* name,
                                            int32_t row_start, int32_t row_end,
                                            const void* local_buf,
                                            void* dst) {
    if (!g || !plan || !name || !dst) return TC_ERR_INVALID_ARG;
    if (plan->n_peers != tc_mesh_internal_n_peers(g)) return TC_ERR_INVALID_ARG;
    if (plan->dtype != TC_COLL_DTYPE_F32) return TC_ERR_UNSUPPORTED_DTYPE;
    if (row_start < 0 || row_end > plan->rows || row_start >= row_end) {
        return TC_ERR_INVALID_ARG;
    }

    const size_t row_bytes = (size_t)plan->cols * shard_dtype_size(plan->dtype);
    const int32_t my_rank = tc_mesh_internal_my_rank(g);
    uint8_t* dst_bytes = static_cast<uint8_t*>(dst);

    /* Walk the requested range one owner-block at a time: each step
     * advances to the end of the current owner's locally-held rows
     * (clamped to row_end). */
    int32_t r = row_start;
    while (r < row_end) {
        const int32_t owner = tc_remote_shard_owner(plan, r);
        if (owner < 0) return TC_ERR_INTERNAL;
        int32_t owner_lo, owner_hi;
        tc_remote_shard_local_range(plan, owner, &owner_lo, &owner_hi);
        const int32_t chunk_end = (owner_hi < row_end) ? owner_hi : row_end;
        const size_t chunk_rows = (size_t)(chunk_end - r);
        const size_t chunk_bytes = chunk_rows * row_bytes;
        const size_t owner_local_offset = (size_t)(r - owner_lo) * row_bytes;

        if (owner == my_rank) {
            /* Local fast-path: copy from caller's local buf. */
            if (local_buf) {
                std::memcpy(dst_bytes,
                            static_cast<const uint8_t*>(local_buf) + owner_local_offset,
                            chunk_bytes);
            } else {
                /* Caller chose not to pass a local buf: fetch our OWN
                 * registered shard via the loopback path. The transport
                 * doesn't loopback (peer_id == -1 for self), so synthesise
                 * the result by reading from the registry cache. */
                const std::string full = mk_shard_name(name, owner);
                auto* reg = registry_for(g);
                std::lock_guard<std::mutex> lk(reg->mu);
                auto it = reg->snapshots.find(full);
                if (it == reg->snapshots.end()) return TC_ERR_INTERNAL;
                std::memcpy(dst_bytes, it->second.data() + owner_local_offset,
                            chunk_bytes);
            }
        } else {
            /* Remote: fetch entire owner shard if the chunk is the
             * whole shard, else fetch the whole shard and slice. The
             * tc_remote_tensor_fetch contract is whole-tensor, so we
             * always fetch the whole owner shard into a temporary,
             * then memcpy our slice out. */
            const size_t shard_bytes = (size_t)(owner_hi - owner_lo) * row_bytes;
            std::vector<uint8_t> tmp(shard_bytes);
            const std::string full = mk_shard_name(name, owner);
            const int pid = tc_mesh_internal_peer_id(g, owner);
            if (pid < 0) return TC_ERR_INTERNAL;
            tc_status_t s = retry_fetch_shard(tc_mesh_internal_client(g),
                                              pid, full.c_str(),
                                              tmp.data(), shard_bytes);
            if (s != TC_OK) return s;
            std::memcpy(dst_bytes, tmp.data() + owner_local_offset, chunk_bytes);
            tc_mesh_internal_add_bytes(g, (uint64_t)chunk_bytes);
        }

        dst_bytes += chunk_bytes;
        r = chunk_end;
    }
    return TC_OK;
}

