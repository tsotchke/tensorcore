#ifndef TENSORCORE_POINCARE_H
#define TENSORCORE_POINCARE_H

/*
 * tensorcore — Poincaré ball ops (hyperbolic geometric ML primitives).
 *
 * These are the GeoRefine / GeometricLM hot paths: Möbius addition,
 * exp/log maps, geodesic distance, parallel transport, conformal factor.
 * All ops operate on batches of D-dimensional row-major fp32 vectors.
 *
 * Curvature contract:
 *   c = 0   → Euclidean limit. Every op reduces to its standard
 *              Euclidean equivalent. Distance: d_0(x,y) = 2‖x−y‖
 *              (Ganea et al. 2018 convention; the factor 2 falls out
 *              of the (2/√c)·atanh formula's c→0 limit).
 *   c > 0   → hyperbolic Poincaré ball of radius 1/√c.
 *
 * Numerical safety (compile-time constants, env-overridable in Python):
 *   - `||x||` floored at `TC_POINCARE_EPS_NORM` (1e-15) before division
 *   - atanh(z) argument clamped to (−1+TC_POINCARE_EPS_ATANH, 1−...) (1e-7)
 *   - points whose norm would exceed (1/√c − TC_POINCARE_MARGIN)
 *     are scaled back inside the ball.
 *
 * Eshkol bindings live in eshkol/poincare.esk; Python torch.autograd.Function
 * wrappers live in bindings/pytorch/tensorcore_torch/poincare.py.
 *
 * Build-time gate: always compiled (CPU-side fp32 only for v0.1).
 * Native Metal/CUDA kernels for the bandwidth-bound elementwise ops
 * land in v0.2 once the autograd path is validated end-to-end.
 */

#include "tensorcore/status.h"
#include "tensorcore/tensorcore.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Single-batch primitives (operate on one row of D fp32 elements) --
 * For batched ops over [N, D], the inner loop is over the N rows.
 * Outputs are written into `out` of the same shape unless noted. */

/* Möbius addition X ⊕_c Y, elementwise per row.
 *   X, Y, out : [N, D] fp32 (row-major, contiguous)
 *   c         : scalar curvature
 * Returns TC_OK on success. */
tc_status_t tc_poincare_mobius_add(tc_context* ctx,
                                    const tc_buffer* X,
                                    const tc_buffer* Y,
                                    tc_buffer* out,
                                    float c, int N, int D);

/* Conformal factor λ_x = 2 / (1 − c‖x‖²). Output shape: [N] fp32. */
tc_status_t tc_poincare_conformal_factor(tc_context* ctx,
                                          const tc_buffer* X,
                                          tc_buffer* lam_out,
                                          float c, int N, int D);

/* exp_0^c(v) = tanh(√c ‖v‖) · v / (√c ‖v‖).
 *   V, out : [N, D] fp32 */
tc_status_t tc_poincare_exp_map_zero(tc_context* ctx,
                                      const tc_buffer* V,
                                      tc_buffer* out,
                                      float c, int N, int D);

/* log_0^c(x) = atanh(√c ‖x‖) · x / (√c ‖x‖).
 *   X, out : [N, D] fp32 */
tc_status_t tc_poincare_log_map_zero(tc_context* ctx,
                                      const tc_buffer* X,
                                      tc_buffer* out,
                                      float c, int N, int D);

/* General exp map: exp_x^c(v) = x ⊕_c (tanh(√c λ_x ‖v‖ / 2) · v/(√c‖v‖)).
 *   X, V, out : [N, D] fp32 */
tc_status_t tc_poincare_exp_map(tc_context* ctx,
                                 const tc_buffer* X,
                                 const tc_buffer* V,
                                 tc_buffer* out,
                                 float c, int N, int D);

/* General log map: log_x^c(y) = (2/(√c λ_x)) · atanh(√c‖−x ⊕_c y‖) · δ/‖δ‖.
 *   X, Y, out : [N, D] fp32 */
tc_status_t tc_poincare_log_map(tc_context* ctx,
                                 const tc_buffer* X,
                                 const tc_buffer* Y,
                                 tc_buffer* out,
                                 float c, int N, int D);

/* Geodesic distance d_c(x, y) = (2/√c) · atanh(√c ‖−x ⊕_c y‖).
 *   X, Y     : [N, D] fp32
 *   dist_out : [N]    fp32. At c=0 returns 2‖x−y‖ (Ganea convention). */
tc_status_t tc_poincare_distance(tc_context* ctx,
                                  const tc_buffer* X,
                                  const tc_buffer* Y,
                                  tc_buffer* dist_out,
                                  float c, int N, int D);

/* Parallel transport PT^c_{x→y}(v) = (λ_x / λ_y) · v (conformal-factor form
 * — the gyration rotation is absorbed in the conformal scaling for the
 * tangent-space identification used by hyperbolic transformers).
 *   X, Y, V, out : [N, D] fp32 */
tc_status_t tc_poincare_parallel_transport(tc_context* ctx,
                                            const tc_buffer* V,
                                            const tc_buffer* X,
                                            const tc_buffer* Y,
                                            tc_buffer* out,
                                            float c, int N, int D);

#ifdef __cplusplus
}
#endif
#endif
