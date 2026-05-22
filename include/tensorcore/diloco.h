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
#include <stdbool.h>
#include "tensorcore/status.h"
#include "tensorcore/dtype.h"
#include "tensorcore/device.h"
#include "tensorcore/distributed.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tc_diloco_ctx tc_diloco_ctx;

/* Compression scheme applied to Δθ before the outer all-reduce.
 * fp16 / top-k / low-rank land on top of the same tc_buffer surface;
 * the all-reduce sees compressed payloads, decompressed at each rank. */
typedef enum {
    TC_DILOCO_COMPRESS_NONE      = 0,   /* full-precision Δθ; 1:1 with model */
    TC_DILOCO_COMPRESS_FP16      = 1,   /* 2× over fp32 master, free */
    TC_DILOCO_COMPRESS_FP8       = 2,   /* 4× over fp32, per-tensor scale */
    TC_DILOCO_COMPRESS_TOPK_1PCT = 3,   /* keep top 1% magnitudes, ~100× */
    TC_DILOCO_COMPRESS_TOPK_01PCT = 4,  /* keep top 0.1%, ~1000× — INTELLECT-1 ran ~here */
    TC_DILOCO_COMPRESS_LOWRANK   = 5,   /* PowerSGD-style low-rank Δθ */
    TC_DILOCO_COMPRESS_SIGNSGD   = 6,   /* 1-bit per element, 32×; needs error-feedback */
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
    bool                         async_overlap;     /* true: outer comm in background while inner steps run */
    bool                         tolerate_dropouts; /* true: ranks that disappear are skipped, not deadlocked */
} tc_diloco_config;

/* Initialize a DiLoCo context layered on an existing distributed context.
 * The dist_ctx must be cross-site (TC_DIST_GLOO over WAN typically); the
 * within-site tight sync is a separate tc_dist_ctx used by the inner loop. */
tc_status_t tc_diloco_init(tc_dist_ctx*               dist_ctx,
                           const tc_diloco_config*    cfg,
                           tc_diloco_ctx**            out);

tc_status_t tc_diloco_finalize(tc_diloco_ctx* d);

/* Register a parameter tensor with the DiLoCo context. The same buffer is
 * used as θ_local by the inner loop and as input to the outer-step Δθ
 * computation. */
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
 * If cfg.async_overlap is true, the all-reduce runs on a worker thread and
 * tc_diloco_apply_outer returns immediately; the inner loop continues
 * against the *previous* θ_global_anchor until the new one is ready, then
 * the swap happens at the next outer-step boundary. This is what makes a
 * 100-200 ms transcontinental RTT invisible to per-token throughput. */
tc_status_t tc_diloco_apply_outer(tc_diloco_ctx* d);

/* For benchmarking + ops introspection. */
uint64_t tc_diloco_outer_steps_completed(const tc_diloco_ctx* d);
uint64_t tc_diloco_inner_steps_completed(const tc_diloco_ctx* d);
double   tc_diloco_last_outer_step_seconds(const tc_diloco_ctx* d);
double   tc_diloco_last_outer_bytes_sent(const tc_diloco_ctx* d);

#ifdef __cplusplus
}
#endif
#endif
