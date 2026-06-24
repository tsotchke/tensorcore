#ifndef TENSORCORE_RIEMANNIAN_ADAM_H
#define TENSORCORE_RIEMANNIAN_ADAM_H

/*
 * tensorcore — RiemannianAdam optimizer step (Bécigneul & Ganea 2018).
 *
 * Extends the AdamW step (tc_adamw_step) with manifold-aware gradient
 * projection and exp-map retraction. The substrate ships one fused step
 * per supported manifold:
 *
 *   tc_riemannian_adam_step_poincare  — Poincaré ball with curvature c
 *   tc_riemannian_adam_step_sphere    — unit sphere (great-circle exp)
 *   tc_riemannian_adam_step_euclidean — c→0 limit = plain AdamW (provided
 *                                        for parity; same semantics as
 *                                        tc_adamw_step on fp32 grads).
 *
 * The product-manifold optimizer composes these per slice (H component
 * → poincare, S component → sphere, R component → euclidean), exactly
 * mirroring GeoRefine's `geometric_lm.py` RiemannianAdam.
 *
 * Algorithm (per parameter row x ∈ R^D):
 *   g_t  = proj_{T_x M}(grad)                    # tangent-space projection
 *   m    = β1·m + (1-β1)·g_t
 *   v    = β2·v + (1-β2)·g_t²
 *   m̂    = m / bias_correction1
 *   v̂    = v / bias_correction2
 *   step = m̂ / (√v̂ + ε)
 *   step = step + weight_decay·x                 # AdamW-style decoupled decay
 *   x_new = retract_x(-lr · step)                # exp_x^M for the manifold
 *   m    = parallel_transport_{x→x_new}(m)       # keep moments in T_{x_new}
 *
 * Tangent-space projection per manifold:
 *   - Poincaré: g_t = g / λ_x²    with  λ_x = 2/(1 − c‖x‖²)
 *   - Sphere:   g_t = g − ⟨g,x⟩·x
 *   - Euclidean: g_t = g
 *
 * Retraction (exp map):
 *   - Poincaré: x_new = x ⊕_c (tanh(√c·λ_x·‖v‖/2)·v/(√c·‖v‖))
 *   - Sphere:   x_new = cos(‖v‖)·x + sin(‖v‖)·v/‖v‖
 *   - Euclidean: x_new = x + v
 *
 * Parallel transport of the first moment:
 *   - Poincaré: PT^c_{x→y}(m) = (λ_x/λ_y)·m
 *   - Sphere:   PT_{x→y}(m) = m − ⟨m,y⟩/(1+⟨x,y⟩)·(x+y)
 *   - Euclidean: identity (m unchanged)
 *
 * fp32 IO + fp32 accumulators throughout. v0.2 will add native Metal/CUDA
 * kernels; CPU reference unblocks the GeoRefine optimizer end-to-end now.
 */

#include "tensorcore/status.h"
#include "tensorcore/tensorcore.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Fused RiemannianAdam step on the Poincaré ball.
 *   params, grads, m, v : [N, D] fp32 (all manifold points / tangents)
 *   N : number of parameter rows on the manifold
 *   D : manifold dimension
 *   c : curvature (≥ 0; c=0 → euclidean AdamW)
 *   lr, beta1, beta2, eps, weight_decay : standard Adam hyperparameters
 *   bias_correction1 / 2 : caller computes (1 − βᵗ); matches tc_adamw_step
 */
tc_status_t tc_riemannian_adam_step_poincare(tc_context* ctx,
                                              tc_buffer* params,
                                              const tc_buffer* grads,
                                              tc_buffer* m, tc_buffer* v,
                                              int N, int D, float c,
                                              float lr, float beta1, float beta2,
                                              float eps, float weight_decay,
                                              float bias_correction1,
                                              float bias_correction2);

/* RiemannianAdam step on the unit sphere. Inputs/outputs same shape. */
tc_status_t tc_riemannian_adam_step_sphere(tc_context* ctx,
                                            tc_buffer* params,
                                            const tc_buffer* grads,
                                            tc_buffer* m, tc_buffer* v,
                                            int N, int D,
                                            float lr, float beta1, float beta2,
                                            float eps, float weight_decay,
                                            float bias_correction1,
                                            float bias_correction2);

/* Euclidean fallback. Identical semantics to tc_adamw_step with fp32 grads;
 * provided here for callers that want a single dispatch surface across the
 * product manifold (H → poincare, S → sphere, R → euclidean). */
tc_status_t tc_riemannian_adam_step_euclidean(tc_context* ctx,
                                               tc_buffer* params,
                                               const tc_buffer* grads,
                                               tc_buffer* m, tc_buffer* v,
                                               int N, int D,
                                               float lr, float beta1, float beta2,
                                               float eps, float weight_decay,
                                               float bias_correction1,
                                               float bias_correction2);

#ifdef __cplusplus
}
#endif
#endif
