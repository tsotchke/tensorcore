#ifndef TENSORCORE_CHECKPOINT_H
#define TENSORCORE_CHECKPOINT_H

/*
 * tensorcore — activation checkpointing primitives.
 *
 * Training a transformer at depth L stores ~O(L) activations for the
 * backward pass. For a 70B model with sequence length 4K and batch 1,
 * that's hundreds of GB just in fp16 activations — exceeds the 192 GB
 * unified-memory budget of an M2 Ultra by 2-3×.
 *
 * Activation checkpointing trades compute for memory: discard most
 * activations after the forward pass, then recompute them on the
 * backward pass from a smaller set of saved "anchor" activations.
 * Memory drops from O(L) to O(√L). Compute increases by ~33% (one
 * extra forward pass per anchor block).
 *
 * tensorcore exposes this as a buffer-level primitive: a checkpoint
 * handle wraps a tc_buffer plus a "recompute" callback. Marking the
 * buffer as "discardable" releases its underlying memory; calling
 * "realize" invokes the callback to repopulate the buffer from a saved
 * subset of inputs.
 *
 * The framework above tensorcore (tensorcore-train) coordinates which
 * buffers are checkpointed and in what blocks. This header just provides
 * the primitive.
 *
 * Typical use, conceptually:
 *
 *     // Forward pass:
 *     tc_buffer* act_layer_N = ...;                    // big activation
 *     tc_checkpoint_id ck = 0;
 *     tc_checkpoint_register(act_layer_N, recompute_fn, user_data, &ck);
 *     // ... finish forward pass ...
 *     tc_checkpoint_discard(ck);                       // free act_layer_N's memory
 *
 *     // Backward pass for layer N:
 *     tc_checkpoint_realize(ck);                       // recompute act_layer_N
 *     // ... use act_layer_N for gradient ...
 *     tc_checkpoint_discard(ck);                       // free again
 *
 * The recompute callback signature is just (void* user_data) → tc_status_t,
 * and the user is responsible for ensuring it writes into the registered
 * tc_buffer. tensorcore tracks the buffer/callback pair and validates the
 * realize/discard lifecycle.
 */

#include <stdint.h>
#include "tensorcore/status.h"
#include "tensorcore/device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t tc_checkpoint_id;

/* User-supplied recompute function. Returns TC_OK on success; the runtime
 * propagates the error from tc_checkpoint_realize. user_data is owned
 * by the caller — tensorcore neither copies nor frees it. */
typedef tc_status_t checkpoint_recompute_status_t;
typedef checkpoint_recompute_status_t (*tc_checkpoint_recompute_fn)(void* user_data);

/* Register a buffer for checkpointing. After this call, the buffer's
 * memory may be freed and re-allocated via discard/realize.
 *
 * The recompute_fn is called from inside tc_checkpoint_realize; it must
 * not call checkpoint APIs on the same id (no same-id reentrancy). It
 * MAY call realize on other ids (nested checkpoints are fine for
 * ascending dependency order).
 *
 * Returns TC_ERR_INVALID_ARG if buf or recompute_fn is null, or if buf
 * is already registered. */
tc_status_t tc_checkpoint_register(tc_buffer* buf,
                                    tc_checkpoint_recompute_fn recompute_fn,
                                    void* user_data,
                                    tc_checkpoint_id* out_id);

/* Discard the buffer's contents — frees the underlying memory while
 * keeping the buffer handle valid. After discard, the buffer cannot be
 * mapped or read until tc_checkpoint_realize is called. */
tc_status_t tc_checkpoint_discard(tc_checkpoint_id id);

/* Re-create the buffer's contents by calling the registered recompute_fn.
 * After this, the buffer is readable again. The runtime serializes
 * realize/discard calls per id; concurrent calls block. */
tc_status_t tc_checkpoint_realize(tc_checkpoint_id id);

/* True if the checkpoint's underlying buffer currently holds data; false
 * if it has been discarded and not yet realized. */
int tc_checkpoint_is_resident(tc_checkpoint_id id);

/* Unregister the checkpoint — the buffer remains owned by the caller,
 * but no longer participates in checkpointing. Safe to call after the
 * caller has finished training. */
tc_status_t tc_checkpoint_unregister(tc_checkpoint_id id);

/* Observability: total bytes currently saved by all discarded checkpoints. */
uint64_t tc_checkpoint_total_bytes_discarded(void);
uint64_t tc_checkpoint_count_resident(void);
uint64_t tc_checkpoint_count_discarded(void);

#ifdef __cplusplus
}
#endif
#endif
