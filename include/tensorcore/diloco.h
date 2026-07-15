#ifndef TENSORCORE_DILOCO_H
#define TENSORCORE_DILOCO_H

/*
 * tensorcore — DiLoCo cross-site distributed training primitives.
 *
 * Standard distributed training assumes a low-latency low-loss network
 * (NCCL on InfiniBand, JACCL on Thunderbolt-5). When workers live on
 * different continents, RTT is 100-200 ms and bandwidth is 1-10 MB/s.
 * Per-step gradient sync becomes physically impossible at those numbers.
 *
 * DiLoCo (Distributed Low-Communication training, Douillard et al. 2024,
 * OpenDiLoCo / Prime Intellect 2024) solves this by running K=100-1000
 * "inner" SGD steps locally on each worker before syncing parameter
 * deltas — not gradients — at the "outer" loop. Communication volume
 * drops by ~K× without accuracy loss, validated at 10B model scale
 * across the public internet (INTELLECT-1).
 *
 * Algorithm:
 *
 *   Each worker holds θ_local (its current parameters). At an outer
 *   step boundary:
 *
 *     1. Each worker computes Δθ = θ_local − θ_global_anchor
 *     2. (Optional) Compress Δθ via fp16 / top-k sparsification
 *     3. all-reduce Δθ across all outer-step workers (AVG)
 *     4. Decompress the averaged Δ̄θ
 *     5. θ_global_anchor += outer_lr × Δ̄θ      (outer optimizer step)
 *     6. θ_local = θ_global_anchor             (every worker resyncs)
 *     7. Run K inner SGD steps locally, then loop.
 *
 *   The outer optimizer is typically Nesterov momentum on the parameter
 *   delta. The inner optimizer is the model's normal optimizer (Adam,
 *   etc.) operating on θ_local.
 *
 * This primitive is the cross-site bridge in the tensorcore distributed
 * stack. Within a site, use tight TC_DIST_RING / TC_DIST_GLOO collectives
 * for traditional per-step gradient sync; *between* sites, DiLoCo carries
 * the parameter state at a much lower duty cycle.
 *
 * Layered above tc_dist_*: a DiLoCo context is parameterized by an
 * existing tc_dist_ctx that handles the actual cross-site communication.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "tensorcore/status.h"
#include "tensorcore/dtype.h"
#include "tensorcore/device.h"
#include "tensorcore/distributed.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tc_diloco_ctx tc_diloco_ctx;

/* Requested compression scheme for Δθ. Runtime support is transport-specific:
 * TOPK uses sparse payloads on Gloo, FP16 currently remains dense fp32 on the
 * wire, and reserved modes are rejected by tc_diloco_init. */
typedef enum {
    TC_DILOCO_COMPRESS_NONE      = 0,   /* full-precision Δθ; 1:1 with model */
    TC_DILOCO_COMPRESS_FP16      = 1,   /* accepted; currently dense fp32 on wire */
    TC_DILOCO_COMPRESS_FP8       = 2,   /* reserved; init currently rejects */
    TC_DILOCO_COMPRESS_TOPK_1PCT = 3,   /* sparse on Gloo; dense masked fallback elsewhere */
    TC_DILOCO_COMPRESS_TOPK_01PCT = 4,  /* sparse on Gloo; dense masked fallback elsewhere */
    TC_DILOCO_COMPRESS_LOWRANK   = 5,   /* reserved; init currently rejects */
    TC_DILOCO_COMPRESS_SIGNSGD   = 6,   /* reserved; init currently rejects */
} tc_diloco_compress_t;

typedef enum {
    TC_DILOCO_OUTER_SGD          = 0,   /* θ_global += lr × Δ̄θ */
    TC_DILOCO_OUTER_NESTEROV     = 1,   /* DiLoCo default: SGD+Nesterov momentum */
    TC_DILOCO_OUTER_ADAM         = 2,   /* Adam on the parameter delta */
} tc_diloco_outer_optimizer_t;

typedef struct {
    int                          inner_steps;       /* K: inner SGD steps between outer syncs */
    float                        outer_lr;          /* outer-optimizer learning rate */
    float                        outer_momentum;    /* for Nesterov / Adam β1 */
    float                        outer_beta2;       /* for Adam */
    float                        outer_eps;         /* for Adam */
    tc_diloco_outer_optimizer_t  outer_optimizer;
    tc_diloco_compress_t         compress;
    bool                         async_overlap;     /* snapshot worker + explicit caller-thread commit */
    bool                         tolerate_dropouts; /* reserved; recovery is not implemented yet */
} tc_diloco_config;

/* Initialize a DiLoCo context layered on an existing distributed context.
 * The dist_ctx must be cross-site (TC_DIST_GLOO over WAN typically); the
 * within-site tight sync is a separate tc_dist_ctx used by the inner loop. */
tc_status_t tc_diloco_init(tc_dist_ctx*               dist_ctx,
                           const tc_diloco_config*    cfg,
                           tc_diloco_ctx**            out);

/* Destroy an idle context. If an async result is READY or FAILED, finalize
 * returns TC_ERR_BUSY or the worker error without destroying the context;
 * acknowledge it with tc_diloco_async_commit, then finalize again. */
tc_status_t tc_diloco_finalize(tc_diloco_ctx* d);

/* Register a parameter tensor with the DiLoCo context. Synchronous outer
 * steps read this buffer directly on the caller thread. Async outer steps
 * snapshot it before returning; the worker never reads or writes it. */
tc_status_t tc_diloco_add_parameter(tc_diloco_ctx*   d,
                                    const char*      name,
                                    tc_buffer*       theta_local,
                                    size_t           num_elements,
                                    tc_dtype_t       dtype);

/* Advance the inner loop. The caller has just completed an inner SGD step.
 * tc_diloco_step returns whether an outer step boundary was just crossed
 * (every cfg->inner_steps calls). When *out_outer_step_pending is true,
 * the next tc_diloco_apply_outer call will perform the cross-site sync. */
tc_status_t tc_diloco_step(tc_diloco_ctx* d, bool* out_outer_step_pending);

/* Execute the outer step:
 *   1. Δθ = θ_local − θ_global_anchor (per parameter)
 *   2. Optional compression
 *   3. all-reduce across the dist_ctx
 *   4. Outer-optimizer update of θ_global_anchor
 *   5. θ_local := θ_global_anchor
 *
 * If cfg.async_overlap is true, this captures an immutable fp32 snapshot on
 * the caller thread, launches communication/optimizer work against private
 * state, and returns. The worker never touches registered buffers. Call
 * tc_diloco_async_wait/poll and tc_diloco_async_commit at a caller-owned
 * parameter boundary. Commit rebases local updates made after the snapshot
 * onto the new anchor before writing registered buffers. Only one async round
 * may be outstanding; another apply returns TC_ERR_BUSY until commit. */
tc_status_t tc_diloco_apply_outer(tc_diloco_ctx* d);

typedef enum {
    TC_DILOCO_ASYNC_IDLE    = 0,
    TC_DILOCO_ASYNC_RUNNING = 1,
    TC_DILOCO_ASYNC_READY   = 2,
    TC_DILOCO_ASYNC_FAILED  = 3,
} tc_diloco_async_state_t;

/* Non-blocking status snapshot. out_worker_status is TC_OK for IDLE/RUNNING/
 * READY and the background failure for FAILED. out_round_id is zero before
 * the first async round. */
tc_status_t tc_diloco_async_poll(const tc_diloco_ctx* d,
                                 tc_diloco_async_state_t* out_state,
                                 tc_status_t* out_worker_status,
                                 uint64_t* out_round_id);

/* Wait for the outstanding worker without touching registered parameters.
 * Returns the background failure, or TC_OK when the result is READY/IDLE. */
tc_status_t tc_diloco_async_wait(tc_diloco_ctx* d);

/* Caller-thread commit/acknowledgement boundary. READY results are rebased and
 * applied with rollback on a buffer-write failure. FAILED results are
 * acknowledged and returned without changing parameters. RUNNING returns
 * TC_ERR_BUSY; call wait first when a blocking boundary is desired. */
tc_status_t tc_diloco_async_commit(tc_diloco_ctx* d);

/* Versioned, little-endian checkpoint blob for DiLoCo-owned state. The model's
 * live parameter bytes remain caller-owned and are intentionally not included.
 * Version 1 preserves anchors, optimizer moments, top-k error feedback,
 * counters/metrics, topology and membership epochs, and a READY/FAILED async
 * result. A RUNNING worker must first reach READY/FAILED; serialization returns
 * TC_ERR_BUSY while it is still executing. */
#define TC_DILOCO_STATE_ABI_VERSION_1 UINT32_C(1)
#define TC_DILOCO_STATE_ABI_VERSION_CURRENT TC_DILOCO_STATE_ABI_VERSION_1

/* Bind checkpoint state to the caller's authoritative topology epochs. Epochs
 * may be set only while the context is IDLE. A non-zero local epoch must match
 * the serialized epoch during restore; zero adopts the checkpoint value. */
tc_status_t tc_diloco_state_set_epochs(tc_diloco_ctx* d,
                                       uint64_t topology_epoch,
                                       uint64_t membership_epoch);
tc_status_t tc_diloco_state_get_epochs(const tc_diloco_ctx* d,
                                       uint64_t* out_topology_epoch,
                                       uint64_t* out_membership_epoch);

/* Query, serialize, and restore the deterministic v1 blob. Restore is atomic
 * with respect to DiLoCo-owned state: the blob is fully parsed and validated
 * against config, distributed rank/world, registered parameter names, dtypes,
 * and sizes before the live context changes. The caller must separately
 * restore each registered theta_local buffer from its model checkpoint. */
tc_status_t tc_diloco_state_size(const tc_diloco_ctx* d,
                                 uint32_t requested_abi_version,
                                 size_t* out_size);
tc_status_t tc_diloco_state_serialize(const tc_diloco_ctx* d,
                                      uint32_t requested_abi_version,
                                      void* out_data,
                                      size_t out_size,
                                      size_t* out_written);
tc_status_t tc_diloco_state_deserialize(tc_diloco_ctx* d,
                                        uint32_t requested_abi_version,
                                        const void* data,
                                        size_t data_size);

/* For benchmarking + ops introspection. */
uint64_t tc_diloco_outer_steps_completed(const tc_diloco_ctx* d);
uint64_t tc_diloco_inner_steps_completed(const tc_diloco_ctx* d);
double   tc_diloco_last_outer_step_seconds(const tc_diloco_ctx* d);
double   tc_diloco_last_outer_bytes_sent(const tc_diloco_ctx* d);

#ifdef __cplusplus
}
#endif
#endif
