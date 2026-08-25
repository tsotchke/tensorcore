#ifndef TENSORCORE_TRAINING_H
#define TENSORCORE_TRAINING_H

#include "tensorcore/status.h"
#include "tensorcore/dtype.h"
#include "tensorcore/device.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------------------------- *
 * RMSnorm — Llama-style normalization without mean subtraction.            *
 *   y = (x / rms(x)) * gamma                                               *
 *   rms(x) = sqrt(mean(x^2) + eps)                                         *
 * Input X is [N, D] in fp16. gamma is [D] in fp16. rstd_out is [N] fp32    *
 * (saved for the backward pass).                                           *
 * ----------------------------------------------------------------------- */
tc_status_t tc_rmsnorm_forward(tc_context* ctx,
                               const tc_buffer* X,
                               const tc_buffer* gamma,
                               tc_buffer*       Y,
                               tc_buffer*       rstd_out,
                               int N, int D, float eps);

tc_status_t tc_rmsnorm_backward(tc_context* ctx,
                                const tc_buffer* X,
                                const tc_buffer* gamma,
                                const tc_buffer* dY,
                                const tc_buffer* rstd,
                                tc_buffer*       dX,
                                tc_buffer*       dgamma,
                                int N, int D);

/* ----------------------------------------------------------------------- *
 * LayerNorm — standard mean+std normalization.                             *
 * ----------------------------------------------------------------------- */
tc_status_t tc_layernorm_forward(tc_context* ctx,
                                 const tc_buffer* X,
                                 const tc_buffer* gamma,
                                 const tc_buffer* beta,
                                 tc_buffer*       Y,
                                 tc_buffer*       mean_out,
                                 tc_buffer*       rstd_out,
                                 int N, int D, float eps);

tc_status_t tc_layernorm_backward(tc_context* ctx,
                                  const tc_buffer* X,
                                  const tc_buffer* gamma,
                                  const tc_buffer* dY,
                                  const tc_buffer* mean,
                                  const tc_buffer* rstd,
                                  tc_buffer*       dX,
                                  int N, int D);

/* ----------------------------------------------------------------------- *
 * Rotary Position Embedding (RoPE) — in-place on X = [B, H, S, D].          *
 * cos_t, sin_t precomputed [S, D/2] fp32.                                  *
 * ----------------------------------------------------------------------- */
tc_status_t tc_rope_forward(tc_context* ctx,
                            tc_buffer*       X,
                            const tc_buffer* cos_t,
                            const tc_buffer* sin_t,
                            int batch, int heads, int seq, int head_dim);

/* Backward pass through RoPE, in-place on dX = [B, H, S, D].
 * Since RoPE is an orthonormal rotation, this applies the inverse rotation
 * to the incoming gradient:
 *   dx0 = dy0 * cos + dy1 * sin
 *   dx1 = -dy0 * sin + dy1 * cos
 */
tc_status_t tc_rope_backward(tc_context* ctx,
                             tc_buffer*       dX,
                             const tc_buffer* cos_t,
                             const tc_buffer* sin_t,
                             int batch, int heads, int seq, int head_dim);

/* ----------------------------------------------------------------------- *
 * SwiGLU: y = silu(gate) * up                                              *
 * ----------------------------------------------------------------------- */
tc_status_t tc_swiglu_forward(tc_context* ctx,
                              const tc_buffer* gate,
                              const tc_buffer* up,
                              tc_buffer*       out,
                              int n);

tc_status_t tc_swiglu_backward(tc_context* ctx,
                               const tc_buffer* gate,
                               const tc_buffer* up,
                               const tc_buffer* dout,
                               tc_buffer*       dgate,
                               tc_buffer*       dup,
                               int n);

/* ----------------------------------------------------------------------- *
 * Standalone softmax (row-wise, fp16).                                     *
 * ----------------------------------------------------------------------- */
tc_status_t tc_softmax_forward (tc_context* ctx,
                                const tc_buffer* X,
                                tc_buffer*       Y,
                                int N, int D);

tc_status_t tc_softmax_backward(tc_context* ctx,
                                const tc_buffer* Y,
                                const tc_buffer* dY,
                                tc_buffer*       dX,
                                int N, int D);

/* ----------------------------------------------------------------------- *
 * Fused AdamW step.                                                        *
 *   params: fp32 master copy (in/out)                                      *
 *   m, v:   fp32 moments (in/out)                                          *
 *   grads:  gradient (fp32 or fp16)                                        *
 *   bc1, bc2: bias correction = 1 - beta^t  (precomputed by host).         *
 * ----------------------------------------------------------------------- */
tc_status_t tc_adamw_step(tc_context* ctx,
                          tc_buffer*       params_fp32,
                          tc_buffer*       m_fp32,
                          tc_buffer*       v_fp32,
                          const tc_buffer* grads,
                          tc_dtype_t       grad_dtype,
                          int n,
                          float lr, float beta1, float beta2, float eps,
                          float wd, float bc1, float bc2);

/* Fused RMSnorm + GEMV for inference: Y = RMSnorm(X, gamma) @ W.
 *
 *   X     : [M, K]  fp16     (typically M ≤ 4 for inference)
 *   gamma : [K]     fp16
 *   W     : [K, N]  fp16
 *   Y     : [M, N]  fp16
 *
 * Eliminates the round-trip of the normalized intermediate, which dominates
 * latency for the Q/K/V and MLP projections during LLM decode.
 *
 * For training (M > 4), callers should use tc_rmsnorm_forward + tc_gemm
 * separately — the per-row rstd recomputation overhead dominates at larger M.
 */
tc_status_t tc_fused_rmsnorm_gemv(tc_context* ctx,
                                  const tc_buffer* X,
                                  const tc_buffer* gamma,
                                  const tc_buffer* W,
                                  tc_buffer*       Y,
                                  int M, int N, int K, float eps);

/* Fused LayerNorm + GEMV for inference: Y = LayerNorm(X, gamma, beta) @ W.
 *
 *   X     : [M, K]  fp16     (typically M <= 4 for inference)
 *   gamma : [K]     fp16
 *   beta  : [K]     fp16
 *   W     : [K, N]  fp16
 *   Y     : [M, N]  fp16
 *
 * Like tc_fused_rmsnorm_gemv, this skips materializing the normalized
 * intermediate. Use the separate tc_layernorm_forward + tc_gemm path when
 * callers need saved mean/rstd for backward or large training batches.
 */
tc_status_t tc_fused_layernorm_gemv(tc_context* ctx,
                                    const tc_buffer* X,
                                    const tc_buffer* gamma,
                                    const tc_buffer* beta,
                                    const tc_buffer* W,
                                    tc_buffer*       Y,
                                    int M, int N, int K, float eps);

#ifdef __cplusplus
}
#endif
#endif
