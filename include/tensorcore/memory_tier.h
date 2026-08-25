#ifndef TENSORCORE_MEMORY_TIER_H
#define TENSORCORE_MEMORY_TIER_H

/*
 * tensorcore — memory tier hints + async promotion / demotion.
 *
 * Modern training mixes memory of very different speeds:
 *
 *   L0  local GPU / Apple unified RAM           100-3000 GB/s
 *   L1  NUMA-attached system RAM (PCIe / CXL)   30-100 GB/s
 *   L2  RDMA-attached peer RAM (IB / 100GbE)    10-25 GB/s
 *   L3  local NVMe                              7-14 GB/s
 *   L4  NVMe-over-Fabrics                       5-10 GB/s
 *
 * For a 70B model training run, only the active layer's weights and
 * activations need to be in L0; the optimizer state, activation
 * checkpoints, and inactive layer weights can tier down to L1-L4 without
 * affecting per-step throughput, as long as the runtime overlaps the
 * promotion with compute on the active layer.
 *
 * This header declares the C-ABI surface for that tiering. The actual
 * tier hosting (which physical memory backs which tier) is configured at
 * tc_context init or via cluster topology; the user code just hints
 * which tier a buffer should live in and when to promote/demote.
 *
 * All operations are non-destructive: a promoted buffer keeps its old
 * tier copy until demote is explicitly called; a demoted buffer's
 * contents are flushed to the lower tier and the L0 copy is freed.
 */

#include <stdint.h>
#include "tensorcore/status.h"
#include "tensorcore/device.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Tier identifiers. Lower = faster. */
typedef enum {
    TC_TIER_L0_DEVICE        = 0,   /* GPU VRAM or Apple unified RAM */
    TC_TIER_L1_HOST_RAM      = 1,   /* system RAM, NUMA-local         */
    TC_TIER_L2_REMOTE_RAM    = 2,   /* peer RAM via RDMA / TCP        */
    TC_TIER_L3_LOCAL_NVME    = 3,   /* local NVMe                     */
    TC_TIER_L4_REMOTE_NVME   = 4,   /* NVMe-oF / shared FS            */
} tc_memory_tier_t;

/* Hint how the buffer is expected to be used. The runtime uses this to
 * decide when to evict it to a lower tier:
 *
 *   HOT  : actively touched every step; do not demote
 *   WARM : touched a few times per epoch (e.g. optimizer state); OK to L1
 *   COLD : touched rarely (checkpoints, inactive shards); OK to L2-L4
 *   ICE  : explicitly archived; will be promoted only on explicit request
 */
typedef enum {
    TC_TIER_HINT_HOT   = 0,
    TC_TIER_HINT_WARM  = 1,
    TC_TIER_HINT_COLD  = 2,
    TC_TIER_HINT_ICE   = 3,
} tc_tier_hint_t;

/* Set a usage hint on a buffer. The runtime is free to keep the buffer
 * at its current tier — this is advisory. */
tc_status_t tc_buffer_set_tier_hint(tc_buffer* b, tc_tier_hint_t hint);

/* Query the current physical tier a buffer is living in. */
tc_status_t tc_buffer_get_tier(const tc_buffer* b, tc_memory_tier_t* out_tier);

/* Asynchronously promote a buffer to a faster tier. If the buffer is
 * already at or above target_tier, no-op. The buffer is usable
 * synchronously after this call but copy-on-write semantics apply:
 * reads will block until the promotion completes; writes after this
 * call will land in the new tier.
 *
 * stream may be NULL for synchronous behavior. */
tc_status_t tc_buffer_promote_async(tc_buffer* b,
                                     tc_memory_tier_t target_tier,
                                     tc_stream* stream);

/* Asynchronously demote a buffer to a slower tier. The L0 copy is freed
 * after the demote completes; subsequent reads will fault into a promote
 * on the original tier the user requested. */
tc_status_t tc_buffer_demote_async(tc_buffer* b,
                                    tc_memory_tier_t target_tier,
                                    tc_stream* stream);

/* Synchronous fence: block until all outstanding tier transitions on
 * this buffer have completed. */
tc_status_t tc_buffer_tier_sync(tc_buffer* b);

/* For the activation-checkpointing use case: the runtime exposes the
 * total bytes resident at each tier so the user can budget. */
tc_status_t tc_memory_tier_usage(tc_context* ctx,
                                  tc_memory_tier_t tier,
                                  uint64_t* out_bytes_resident,
                                  uint64_t* out_bytes_capacity);

#ifdef __cplusplus
}
#endif
#endif
