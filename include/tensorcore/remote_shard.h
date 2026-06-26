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
 * Push semantics (owner ← non-owner gradient apply) deliberately not in
 * this version: the pull-only TCP transport in remote_tensor.cpp makes
 * a true push awkward — would need either a server-side message queue
 * or a pull-based "drain pending puts" loop on the owner. Future work.
 * For now, parameter-server flows that need push can have workers
 * re-register their local shard slice each step (the same path the
 * owner uses) and have the owner pull-merge via tc_remote_shard_get.
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

#ifdef __cplusplus
}
#endif
#endif
