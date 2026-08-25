#ifndef TENSORCORE_REMOTE_SHARD_H
#define TENSORCORE_REMOTE_SHARD_H

/*
 * tensorcore — Row-sharded tensor across N peers (owner-based routing).
 *
 * Logical tensor of shape [rows × cols] is partitioned into N row-shards
 * across an existing tc_mesh_group. Each shard is owned by exactly one
 * peer (the "owner"); ownership follows balanced block-row assignment:
 *
 *   owner(r) = r * N / rows
 *
 * which gives every peer ⌈rows/N⌉ rows except possibly the last
 * (catch-the-tail). The shard plan is deterministic from (rows, N), so
 * every peer in the group computes the same plan without communication.
 *
 * Public surface:
 *
 *   tc_remote_shard_owner(plan, row)                — owner rank for a row
 *   tc_remote_shard_local_range(plan, my_rank, ...)  — [row_start, row_end)
 *                                                      owned locally
 *   tc_remote_shard_register(group, plan, my_buf)    — owner publishes its
 *                                                      row-block on the
 *                                                      mesh transport
 *   tc_remote_shard_get(group, plan, row_start,      — any rank fetches a
 *                       row_end, dst)                  contiguous row range
 *                                                      from whichever
 *                                                      owners hold it
 *
 * The (rows, cols, dtype) tuple is the SHARD PLAN; same on every peer.
 * The plan is plain values, not a handle — there's no global state to
 * synchronize.
 *
 * Naming on the mesh transport:
 *   "_shard/<name>/<rank>"   — owner rank's row-block
 *
 * `<name>` is caller-supplied (e.g. "weights/L7", "grads/L7"). This lets
 * multiple sharded tensors coexist on one group without collision.
 *
 * Push semantics (owner ← non-owner gradient apply) live as a
 * publish/drain protocol on top of the pull-only TCP transport:
 *
 *   - Non-owner calls tc_remote_shard_publish_put(group, plan, name,
 *     row_start, row_end, src). It snapshots `src` into the local
 *     put cache and registers it on the local server under a
 *     deterministic put-name "_shard_put/<name>/<source_rank>"
 *     prefixed with the row range. The publisher returns immediately;
 *     no synchronous handshake with the owner.
 *
 *   - Owner calls tc_remote_shard_drain_puts(group, plan, name,
 *     owner_mut_buf, out_applied). The owner iterates every
 *     non-self peer, attempts to fetch any pending put named
 *     "_shard_put/<name>/<peer>". If a put exists, it's applied
 *     into the owner's mutable shard at the row range encoded in
 *     the put header, then the put is unregistered on the
 *     publisher's side (via a clear-side fetch). out_applied
 *     reports how many puts were applied this round.
 *
 *   - Workers + owners cooperate at the application layer: e.g. a
 *     parameter-server flow calls publish_put at the end of every
 *     gradient round and the owner calls drain_puts at the start
 *     of every weight-update round.
 *
 *  Per-put header layout in the publish cache (first 16 bytes of
 *  the registered blob): { row_start: int32, row_end: int32,
 *  source_rank: int32, magic: 0x53504755 }. Followed by the
 *  row-block payload in row-major fp32. The owner reads the header
 *  first, then the payload.
 *
 * Replaces: ad-hoc per-tensor MPI scatter+gather in tsotchke-chan's
 * parameter-server, the manual rank-route table in DiLoCo's gradient
 * exchange, and qLLM's bespoke row-blocked weight loader.
 */

#include <stddef.h>
#include <stdint.h>
#include "tensorcore/status.h"
#include "tensorcore/mesh_collective.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t           n_peers;     /* MUST equal group->n_peers */
    int32_t           rows;        /* total rows in the logical tensor */
    int32_t           cols;        /* row width in elements */
    tc_coll_dtype_t   dtype;       /* element dtype (currently F32) */
} tc_shard_plan_t;

/* Owner rank for a given logical row. Plan-derivable, no comm. */
int32_t tc_remote_shard_owner(const tc_shard_plan_t* plan, int32_t row);

/* Local row range [out_row_start, out_row_end) owned by `my_rank` under
 * the balanced block-row assignment. Both outputs are clamped to
 * [0, plan->rows]. */
void tc_remote_shard_local_range(const tc_shard_plan_t* plan, int32_t my_rank,
                                  int32_t* out_row_start,
                                  int32_t* out_row_end);

/* Owner-side publish: register my row-block (rows
 * [local_start, local_end) × cols) on the mesh transport under
 * "_shard/<name>/<my_rank>". `my_buf` must point at the OWNER's local
 * row-block in row-major layout; it MUST remain valid until the
 * caller next reshapes / unpublishes (the transport reads zero-copy). */
tc_status_t tc_remote_shard_register(tc_mesh_group_t* group,
                                      const tc_shard_plan_t* plan,
                                      const char* name,
                                      const void* my_buf);

/* Fetch the contiguous row range [row_start, row_end) into `dst`
 * (row-major, (row_end - row_start) × cols × dtype-size). Routes each
 * sub-range to its owner via the mesh transport. Rows the caller
 * already owns are copied locally from `local_buf` (pass NULL to skip
 * the local fast-path; the call will still complete by fetching
 * over the loopback, which is correct but slower). */
tc_status_t tc_remote_shard_get(tc_mesh_group_t* group,
                                 const tc_shard_plan_t* plan,
                                 const char* name,
                                 int32_t row_start, int32_t row_end,
                                 const void* local_buf,
                                 void* dst);

/* Publisher side of the push protocol. Snapshots `src` (length
 * (row_end - row_start) * cols * dtype-size) into a put cache and
 * registers it on this rank's server. row_start/row_end MUST lie
 * within the owner's local range; the publisher rank MUST NOT be
 * the owner of any of those rows (use a local memcpy for that). */
tc_status_t tc_remote_shard_publish_put(tc_mesh_group_t* group,
                                         const tc_shard_plan_t* plan,
                                         const char* name,
                                         int32_t row_start, int32_t row_end,
                                         const void* src);

/* Owner side of the push protocol. Polls every non-self peer for a
 * pending put under "_shard_put/<name>/<peer>"; for each one that
 * exists, applies the row range encoded in its header into
 * `owner_mut_buf` (which MUST point at the owner's local row-block
 * for this shard, same buffer originally registered via
 * tc_remote_shard_register). Returns TC_OK on success;
 * `out_applied` (may be NULL) receives the number of puts applied
 * this round. Drain is non-blocking — peers without a pending put
 * are skipped silently. */
tc_status_t tc_remote_shard_drain_puts(tc_mesh_group_t* group,
                                        const tc_shard_plan_t* plan,
                                        const char* name,
                                        void* owner_mut_buf,
                                        int32_t* out_applied);

#ifdef __cplusplus
}
#endif
#endif
