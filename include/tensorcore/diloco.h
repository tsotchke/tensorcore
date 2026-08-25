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
 * TC_ERR_BUSY while it is still executing.
 *
 * Version 2 carries the identical payload bytes behind a wider header that adds
 * an explicit payload size and a SHA-256 of the payload. The payload layout is
 * byte-for-byte the same in both versions, so a v1 and a v2 blob written from
 * the same state differ only in their headers.
 *
 * Blob layout, little-endian throughout:
 *
 *   v1 header (32 bytes)          v2 header (64 bytes)
 *     0  magic "TCDLSTA1"           0  magic "TCDLSTA2"
 *     8  u32 abi_version = 1        8  u32 abi_version = 2
 *    12  u32 header_size = 32      12  u32 header_size = 64
 *    16  u64 total_size            16  u64 total_size
 *    24  u64 FNV-1a payload hash   24  u64 payload_size
 *                                  32  32-byte SHA-256 of the payload
 *
 * Serialization is deterministic: the same DiLoCo state always produces the
 * same bytes for a given requested version. */
#define TC_DILOCO_STATE_ABI_VERSION_1 UINT32_C(1)
#define TC_DILOCO_STATE_ABI_VERSION_2 UINT32_C(2)
#define TC_DILOCO_STATE_ABI_VERSION_MIN TC_DILOCO_STATE_ABI_VERSION_1
#define TC_DILOCO_STATE_ABI_VERSION_CURRENT TC_DILOCO_STATE_ABI_VERSION_2

#define TC_DILOCO_STATE_V1_HEADER_SIZE UINT32_C(32)
#define TC_DILOCO_STATE_V2_HEADER_SIZE UINT32_C(64)
#define TC_DILOCO_STATE_V2_DIGEST_BYTES UINT32_C(32)

/* Bind checkpoint state to the caller's authoritative topology epochs. Epochs
 * may be set only while the context is IDLE. A non-zero local epoch must match
 * the serialized epoch during restore; zero adopts the checkpoint value. */
tc_status_t tc_diloco_state_set_epochs(tc_diloco_ctx* d,
                                       uint64_t topology_epoch,
                                       uint64_t membership_epoch);
tc_status_t tc_diloco_state_get_epochs(const tc_diloco_ctx* d,
                                       uint64_t* out_topology_epoch,
                                       uint64_t* out_membership_epoch);

/* Query, serialize, and restore the deterministic blob. Restore is atomic with
 * respect to DiLoCo-owned state: the blob is fully parsed, its header and
 * payload digest verified, and its contents validated against config,
 * distributed rank/world, registered parameter names, dtypes, and sizes before
 * the live context changes. A rejected blob leaves the context byte-identical
 * to what it was. The caller must separately restore each registered
 * theta_local buffer from its model checkpoint.
 *
 * For tc_diloco_state_size and tc_diloco_state_serialize, requested_abi_version
 * selects the exact wire version to emit; an unimplemented version returns
 * TC_ERR_ABI_MISMATCH and leaves every output untouched. Requesting
 * TC_DILOCO_STATE_ABI_VERSION_1 keeps producing byte-identical v1 blobs.
 *
 * For tc_diloco_state_deserialize, requested_abi_version is the highest version
 * the caller understands. Blobs from TC_DILOCO_STATE_ABI_VERSION_MIN through
 * that version are accepted; a newer blob returns TC_ERR_ABI_MISMATCH rather
 * than being partially interpreted. */
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

/* ------------------------------------------------------------------------
 * Versioned DiLoCo capability query
 *
 * Consumers must not infer DiLoCo behavior from the package version or from
 * the presence of a symbol. This is the authoritative, size-versioned answer
 * to "which outer optimizers can I resume exactly", "is the top-k
 * error-feedback residual part of the checkpoint", and "what does this
 * compression enum actually put on the wire".
 *
 * struct_size comes first so the struct can grow by appending. The runtime
 * reports the size it wrote; a caller compiled against an older header passes
 * its own smaller out_size and receives only the prefix that fits.
 * ------------------------------------------------------------------------ */

#define TC_DILOCO_CAPABILITIES_ABI_VERSION_1 UINT32_C(1)
#define TC_DILOCO_CAPABILITIES_ABI_VERSION_CURRENT \
    TC_DILOCO_CAPABILITIES_ABI_VERSION_1

/* Optimizer and compression masks use the enum value as the bit position, so
 * a mask stays meaningful as new enumerators are appended. */
#define TC_DILOCO_OUTER_OPTIMIZER_BIT(optimizer) \
    (UINT64_C(1) << (unsigned)(optimizer))
#define TC_DILOCO_COMPRESS_BIT(compress) \
    (UINT64_C(1) << (unsigned)(compress))

/* Which pieces of DiLoCo-owned state the serialization surface covers. */
#define TC_DILOCO_STATE_FEATURE_OUTER_ANCHOR        (UINT64_C(1) << 0)
#define TC_DILOCO_STATE_FEATURE_OUTER_MOMENTS       (UINT64_C(1) << 1)
#define TC_DILOCO_STATE_FEATURE_ERROR_FEEDBACK      (UINT64_C(1) << 2)
#define TC_DILOCO_STATE_FEATURE_COUNTERS            (UINT64_C(1) << 3)
#define TC_DILOCO_STATE_FEATURE_PENDING_ROUND       (UINT64_C(1) << 4)
#define TC_DILOCO_STATE_FEATURE_TOPOLOGY_EPOCH      (UINT64_C(1) << 5)
#define TC_DILOCO_STATE_FEATURE_MEMBERSHIP_EPOCH    (UINT64_C(1) << 6)
#define TC_DILOCO_STATE_FEATURE_PAYLOAD_SHA256      (UINT64_C(1) << 7)
#define TC_DILOCO_STATE_FEATURE_DETERMINISTIC_LAYOUT (UINT64_C(1) << 8)
#define TC_DILOCO_STATE_FEATURE_ATOMIC_RESTORE      (UINT64_C(1) << 9)

typedef struct {
    /* Size of the structure the runtime wrote, not the caller's buffer. */
    uint32_t struct_size;
    /* Capability ABI version actually written. */
    uint32_t abi_version;

    uint32_t runtime_version_major;
    uint32_t runtime_version_minor;
    uint32_t runtime_version_patch;
    uint32_t reserved0;

    /* Serialization wire versions this runtime implements, inclusive. */
    uint32_t state_abi_version_min;
    uint32_t state_abi_version_max;
    /* Version emitted for TC_DILOCO_STATE_ABI_VERSION_CURRENT. */
    uint32_t state_abi_version_current;
    /* Header bytes and payload-digest bytes of state_abi_version_current. */
    uint32_t state_header_size;
    uint32_t state_payload_digest_bytes;
    uint32_t reserved1;

    /* Bit positions are tc_diloco_outer_optimizer_t values. "Serializable"
     * means every moment the optimizer owns survives an export/import
     * round-trip exactly, so resume is bit-exact. */
    uint64_t outer_optimizer_supported_mask;
    uint64_t outer_optimizer_serializable_mask;

    /* Bit positions are tc_diloco_compress_t values. */
    uint64_t compress_supported_mask;
    /* Compression modes whose full residual state survives a round-trip. */
    uint64_t compress_serializable_mask;
    /* Modes that genuinely send a sparse payload on this context's transport.
     * Empty when the query is not bound to a context. */
    uint64_t compress_sparse_wire_mask;
    /* Modes accepted by tc_diloco_init that nevertheless place dense fp32 on
     * the wire. A consumer must not describe these as compressed transport. */
    uint64_t compress_dense_fp32_wire_mask;

    uint64_t state_feature_mask;

    uint32_t async_overlap_supported;
    uint32_t async_state_serializable;
    uint32_t tolerate_dropouts_supported;
    uint32_t reserved2;

    /* Must be zero. Reserved for append-only ABI growth. */
    uint64_t reserved[4];
} tc_diloco_capabilities;

/* The v1 prefix through state_feature_mask. Older v1 callers may pass exactly
 * this size; newer runtimes copy only the prefix that fits. */
#define TC_DILOCO_CAPABILITIES_V1_MIN_SIZE \
    (offsetof(tc_diloco_capabilities, state_feature_mask) + \
     sizeof(((tc_diloco_capabilities*)0)->state_feature_mask))

/* Query DiLoCo capabilities.
 *
 * d may be NULL to obtain build-level capabilities before any context exists,
 * which is what a consumer needs in order to reject an unsupported
 * configuration before calling tc_diloco_init. When d is non-NULL the answer
 * is refined by the context's bound transport, so compress_sparse_wire_mask
 * reflects what that transport actually does.
 *
 * requested_abi_version must be a version implemented by the runtime; a
 * newer/unknown version returns TC_ERR_ABI_MISMATCH without modifying out.
 * out_size may be between TC_DILOCO_CAPABILITIES_V1_MIN_SIZE and any larger
 * value. The runtime copies at most sizeof(tc_diloco_capabilities), leaving a
 * newer caller's tail untouched. */
tc_status_t tc_diloco_capability_query(const tc_diloco_ctx* d,
                                       uint32_t requested_abi_version,
                                       tc_diloco_capabilities* out,
                                       size_t out_size);

/* Header-only fail-closed helpers; they add no exported ABI symbol. A caller
 * that received only the v1 minimum prefix still gets correct answers, because
 * every field these read lives inside that prefix. */
static inline bool tc_diloco_outer_optimizer_is_serializable(
    const tc_diloco_capabilities* capabilities,
    tc_diloco_outer_optimizer_t optimizer) {
    if (capabilities == NULL || (unsigned)optimizer >= 64u) return false;
    const uint64_t bit = TC_DILOCO_OUTER_OPTIMIZER_BIT(optimizer);
    return (capabilities->outer_optimizer_supported_mask & bit) == bit &&
           (capabilities->outer_optimizer_serializable_mask & bit) == bit;
}

static inline bool tc_diloco_compress_is_serializable(
    const tc_diloco_capabilities* capabilities,
    tc_diloco_compress_t compress) {
    if (capabilities == NULL || (unsigned)compress >= 64u) return false;
    const uint64_t bit = TC_DILOCO_COMPRESS_BIT(compress);
    return (capabilities->compress_supported_mask & bit) == bit &&
           (capabilities->compress_serializable_mask & bit) == bit;
}

static inline bool tc_diloco_state_feature_available(
    const tc_diloco_capabilities* capabilities,
    uint64_t feature) {
    return capabilities != NULL && feature != 0 &&
           (capabilities->state_feature_mask & feature) == feature;
}

/* For benchmarking + ops introspection. */
uint64_t tc_diloco_outer_steps_completed(const tc_diloco_ctx* d);
uint64_t tc_diloco_inner_steps_completed(const tc_diloco_ctx* d);
double   tc_diloco_last_outer_step_seconds(const tc_diloco_ctx* d);
double   tc_diloco_last_outer_bytes_sent(const tc_diloco_ctx* d);

#ifdef __cplusplus
}
#endif
#endif
