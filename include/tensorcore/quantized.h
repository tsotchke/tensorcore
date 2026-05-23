#ifndef TENSORCORE_QUANTIZED_H
#define TENSORCORE_QUANTIZED_H

#include "tensorcore/status.h"
#include "tensorcore/dtype.h"
#include "tensorcore/device.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Quantized matrix-multiply (block-wise weight quantization, fp16 activations,
 * fp16 output). Format mirrors ggml's Q4_0 and Q8_0:
 *
 *   Q4_0: 32 weights per block, one fp16 scale, 4-bit weights packed in GGML
 *         order: byte i stores weight i in the low nibble and weight i+16 in
 *         the high nibble.
 *         Bytes/block = 2 + 16 = 18. Bits/weight = 4.5.
 *   Q8_0: 32 weights per block, one fp16 scale, signed int8 weights.
 *         Bytes/block = 2 + 32 = 34. Bits/weight = 8.5.
 *
 * Used for LLM inference: weights pre-quantized once; activations stay fp16.
 * Memory bandwidth dominates, so these win 4-8x over fp16 GEMV on inference.
 */

typedef enum {
    TC_QUANT_Q4_0 = 0,
    TC_QUANT_Q8_0 = 1,
} tc_quant_t;

/* Quantize an [N, K] fp16 weight tensor into a block-quantized buffer.
 * Output buffer size: N * (K/32) * (18 for Q4_0, 34 for Q8_0) bytes. */
tc_status_t tc_quantize_weights(tc_context* ctx,
                                const tc_buffer* W_fp16,
                                tc_buffer*       W_quant,
                                tc_quant_t       fmt,
                                int N, int K);

/* GEMV: Y[M, N] = X[M, K] @ W^T  where W is quantized [N, K].
 * Currently optimized for M small (<= 4), the LLM-inference path. Larger M
 * routes through dequant + tc_gemm in a future pass. */
tc_status_t tc_gemv_quantized(tc_context* ctx,
                              const tc_buffer* X,
                              const tc_buffer* W_quant,
                              tc_buffer*       Y,
                              tc_quant_t       fmt,
                              int M, int N, int K);

/* Fused RMSNorm + quantized GEMV:
 *
 *   X_norm = RMSNorm(X, gamma)
 *   Y[M, N] = X_norm[M, K] @ W_quant[N, K]^T
 *
 * X/gamma/Y are fp16; W_quant is Q4_0 or Q8_0. This is the token-decode
 * primitive for GGUF/qLLM/Kimi paths that project a normalized hidden state
 * through quantized weights without each runtime hand-rolling its own norm
 * and dequant loop.
 */
tc_status_t tc_fused_rmsnorm_gemv_quantized(tc_context* ctx,
                                            const tc_buffer* X,
                                            const tc_buffer* gamma,
                                            const tc_buffer* W_quant,
                                            tc_buffer*       Y,
                                            tc_quant_t       fmt,
                                            int M, int N, int K,
                                            float eps);

/* Async variant: encodes into the provided stream without sync. Caller must
 * call tc_stream_sync afterwards. The async path keeps a single command buffer
 * open across calls, avoiding a per-GEMV command-buffer round trip. */
tc_status_t tc_gemv_quantized_async(tc_context* ctx,
                                    const tc_buffer* X,
                                    const tc_buffer* W_quant,
                                    tc_buffer*       Y,
                                    tc_quant_t       fmt,
                                    int M, int N, int K,
                                    tc_stream*       stream);

/* Compute the storage size (bytes) for an [N, K] quantized weight buffer. */
size_t tc_quantized_size(tc_quant_t fmt, int N, int K);

#ifdef __cplusplus
}
#endif
#endif
