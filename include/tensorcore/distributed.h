#ifndef TENSORCORE_DISTRIBUTED_H
#define TENSORCORE_DISTRIBUTED_H

#include "tensorcore/status.h"
#include "tensorcore/dtype.h"
#include "tensorcore/device.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Distributed primitives for multi-Mac training.
 *
 * Backends:
 *   - TC_DIST_SINGLE: one-process emulation (no-op all-reduce; useful for
 *                     unit tests and correctness validation of the API).
 *   - TC_DIST_RING:   Thunderbolt-5 ring (real multi-Mac, requires TB5 daisy
 *                     chain + macOS 26.2+ JACCL substrate). Implementation
 *                     lives in lib/distributed/ring_tb5.mm (phase v0.5 once
 *                     we can validate against MLX's JACCL).
 *   - TC_DIST_GLOO:   CPU-backed Gloo fallback over Ethernet (phase v0.5).
 *
 * v0.1 ships TC_DIST_SINGLE working end-to-end so the API + Eshkol bindings
 * + correctness tests + the user-side gradient sync code is exercised.
 */

typedef enum {
    TC_DIST_SINGLE = 0,
    TC_DIST_RING   = 1,
    TC_DIST_GLOO   = 2,
} tc_dist_backend_t;

typedef enum {
    TC_REDUCE_SUM = 0,
    TC_REDUCE_AVG = 1,
    TC_REDUCE_MAX = 2,
    TC_REDUCE_MIN = 3,
} tc_reduce_op_t;

typedef struct tc_dist_ctx tc_dist_ctx;

/* Initialize a distributed context. world_size=1 always succeeds and produces
 * a no-op collective layer. For RING/GLOO, the caller supplies a rendezvous
 * URL (e.g. "tb5://192.168.42.0/cluster"). v0.1 RING returns
 * TC_ERR_UNSUPPORTED_FAMILY until the JACCL substrate ships. */
tc_status_t tc_dist_init(tc_context*        tc,
                         tc_dist_backend_t   backend,
                         int                 world_size,
                         int                 rank,
                         const char*         rendezvous_url,
                         tc_dist_ctx**       out);

tc_status_t tc_dist_finalize(tc_dist_ctx* d);

int tc_dist_world_size(const tc_dist_ctx* d);
int tc_dist_rank(const tc_dist_ctx* d);

/* All-reduce in-place on a tc_buffer.  Element type passed explicitly so the
 * kernel can pick fp32 vs fp16 accumulation paths. */
tc_status_t tc_allreduce(tc_dist_ctx*    d,
                         tc_buffer*       buf,
                         size_t           num_elements,
                         tc_dtype_t       dtype,
                         tc_reduce_op_t   op);

/* Broadcast from `root` rank to all others. */
tc_status_t tc_broadcast(tc_dist_ctx*    d,
                         tc_buffer*       buf,
                         size_t           num_elements,
                         tc_dtype_t       dtype,
                         int              root);

/* All-gather: each rank contributes `num_elements`, output holds
 * `world_size * num_elements`. */
tc_status_t tc_allgather(tc_dist_ctx*       d,
                         const tc_buffer*    in,
                         tc_buffer*          out,
                         size_t              num_elements_per_rank,
                         tc_dtype_t          dtype);

/* Barrier — all ranks meet before any continues. */
tc_status_t tc_barrier(tc_dist_ctx* d);

#ifdef __cplusplus
}
#endif
#endif
