#ifndef TENSORCORE_ATTENTION_H
#define TENSORCORE_ATTENTION_H

#include <stdint.h>
#include <stdbool.h>
#include "tensorcore/status.h"
#include "tensorcore/dtype.h"
#include "tensorcore/device.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Fused scaled-dot-product attention (FlashAttention-2 style forward).
 *
 *   S = (Q @ K^T) * softmax_scale  (+ optional causal mask + bias)
 *   P = softmax(S, dim=-1)
 *   O = P @ V
 *
 * Shapes are explicit (batch, heads, seq, head_dim). Q/K/V/O live in fp16
 * or bf16; accumulators are fp32. For training, return_lse=true writes the
 * log-sum-exp vector needed by the backward pass.
 */
typedef struct {
    int32_t batch;
    int32_t heads;
    int32_t seq_q;
    int32_t seq_kv;
    int32_t head_dim;       /* must be ≤ 128 for the on-chip tile layout */

    tc_dtype_t io_dtype;    /* F16 or BF16                                */
    tc_dtype_t accum_dtype; /* F32                                        */

    float    softmax_scale; /* commonly 1 / sqrt(head_dim)                */
    bool     causal;
    bool     return_lse;    /* write LSE for backward                     */

    /* GQA / MQA: kv_heads divides heads. If 0, defaults to `heads`. */
    int32_t  kv_heads;
} tc_attention_desc;

tc_status_t tc_attention_forward(tc_context* ctx,
                                 const tc_attention_desc* desc,
                                 const tc_buffer* Q,
                                 const tc_buffer* K,
                                 const tc_buffer* V,
                                 tc_buffer*       O,
                                 tc_buffer*       LSE  /* nullable        */);

tc_status_t tc_attention_forward_async(tc_context* ctx,
                                       const tc_attention_desc* desc,
                                       const tc_buffer* Q,
                                       const tc_buffer* K,
                                       const tc_buffer* V,
                                       tc_buffer*       O,
                                       tc_buffer*       LSE,
                                       tc_stream*       stream);

/* Backward pass.  Given the forward inputs (Q, K, V), forward outputs (O, LSE),
 * and the output gradient (dO), computes dQ, dK, dV.
 *
 * Requirements:
 *   - LSE must be the log-sum-exp written by tc_attention_forward with
 *     return_lse=true (or recomputed equivalently).
 *   - All tensors are fp16; LSE is fp32.
 *   - v0.1 of the backward: head_dim = 64 only. */
tc_status_t tc_attention_backward(tc_context* ctx,
                                  const tc_attention_desc* desc,
                                  const tc_buffer* Q,
                                  const tc_buffer* K,
                                  const tc_buffer* V,
                                  const tc_buffer* O,
                                  const tc_buffer* dO,
                                  const tc_buffer* LSE,
                                  tc_buffer*       dQ,
                                  tc_buffer*       dK,
                                  tc_buffer*       dV);

#ifdef __cplusplus
}
#endif
#endif
