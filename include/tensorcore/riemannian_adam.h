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
 * fp32 IO + fp32 accumulators throughout.
 *
 * Backends: the three tc_riemannian_adam_step_* entry points dispatch to
 * fused CUDA kernels (lib/cuda/riemannian_adam.cu) when CUDA is active and
 * the buffers are CUDA-managed, and otherwise run the OpenMP CPU reference
 * (lib/ops/riemannian_adam_cpu.cpp). Metal is still CPU-only.
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

/* ----------------------------------------------------------------------- *
 * Off-manifold telemetry.
 *
 * The Poincaré step clamps the conformal denominator 1 − c‖x‖² up to
 * kEpsNorm whenever a row sits on or outside the ball boundary. The clamp
 * keeps the arithmetic finite, but it silently converts
 *
 *     "this point is not on the manifold"
 *
 * into
 *
 *     "λ_x ≈ 2e15, so the tangent rescale 1/λ_x² ≈ 2.5e-31 annihilates this
 *      row's gradient"
 *
 * — a total loss of learning signal on exactly the rows that are most broken,
 * visible only as a loss curve that quietly stops moving. Rescuing the number
 * without reporting the condition is the wrong trade: a caller cannot
 * distinguish "the geometry did nothing" from "the geometry was never
 * evaluated in its domain".
 *
 * Every clamp is therefore counted. Two counters, because the two sites mean
 * different things:
 *
 *   entry     the parameter was already outside the ball when the step began
 *             — bad initialisation, or a previous step that escaped
 *   retract   the exp-map landed outside despite the boundary projection
 *             — the step was too large for the local geometry
 *
 * Counters are process-global, monotonic, and thread-safe; both the OpenMP
 * reference and the CUDA kernels feed them. Read them after a step, or zero
 * them first to scope the reading to one step:
 *
 *     tc_riemannian_offmanifold_reset();
 *     tc_riemannian_adam_step_poincare(...);
 *     if (tc_riemannian_offmanifold_count() > 0) { ... report, do not ignore ... }
 * ----------------------------------------------------------------------- */

/* Total clamps since the last reset (entry + retract). */
unsigned long long tc_riemannian_offmanifold_count(void);

/* Split counts. Either pointer may be NULL. */
void tc_riemannian_offmanifold_counts(unsigned long long* entry,
                                      unsigned long long* retract);

/* Zero both counters. */
void tc_riemannian_offmanifold_reset(void);

/* ----------------------------------------------------------------------- *
 * CPU reference paths, exposed by name.
 *
 * These are the implementations the dispatching entry points above fall back
 * to. They are public for one reason: tests/test_riemannian_adam_parity.c
 * needs to run the same inputs through CPU and CUDA in one process and
 * compare. Toggling a backend through the environment would only prove the
 * two runs *were configured* differently; calling both directly proves which
 * code actually ran. Production callers should use the dispatching names.
 * ----------------------------------------------------------------------- */
tc_status_t tc_riemannian_adam_step_poincare_reference(
        tc_context* ctx, tc_buffer* params, const tc_buffer* grads,
        tc_buffer* m, tc_buffer* v, int N, int D, float c,
        float lr, float beta1, float beta2, float eps, float weight_decay,
        float bias_correction1, float bias_correction2);

tc_status_t tc_riemannian_adam_step_sphere_reference(
        tc_context* ctx, tc_buffer* params, const tc_buffer* grads,
        tc_buffer* m, tc_buffer* v, int N, int D,
        float lr, float beta1, float beta2, float eps, float weight_decay,
        float bias_correction1, float bias_correction2);

tc_status_t tc_riemannian_adam_step_euclidean_reference(
        tc_context* ctx, tc_buffer* params, const tc_buffer* grads,
        tc_buffer* m, tc_buffer* v, int N, int D,
        float lr, float beta1, float beta2, float eps, float weight_decay,
        float bias_correction1, float bias_correction2);

#ifdef __cplusplus
}
#endif
#endif
