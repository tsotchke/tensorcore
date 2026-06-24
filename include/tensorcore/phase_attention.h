#ifndef TENSORCORE_PHASE_ATTENTION_H
#define TENSORCORE_PHASE_ATTENTION_H

/*
 * tensorcore — phase attention + Born-rule output (qLLM-specific kernels).
 *
 * These are GeoRefine's GeometricLM hot paths for the qLLM scoring +
 * output projection. Both are fused elementwise reductions that take
 * pre-computed per-manifold quantities (inner products, geodesic
 * distances, phase differences, amplitudes) and combine them per the
 * GeometricLM formulas.
 *
 * Phase-attention score (per Q-K pair, summed over M manifolds):
 *
 *   score(q, k) = Σ_m w_m · ⟨Q_m, K_m⟩_m · cos(Δφ_m + γ_m) · exp(−λ_m · d_m)
 *
 *   where m ∈ {H, S, R} indexes the product-manifold components,
 *   ⟨·,·⟩_m is the manifold inner product (callers pre-compute via GEMM),
 *   Δφ_m is the per-pair phase difference (caller-computed),
 *   d_m is the per-pair geodesic distance (caller-computed via
 *   tc_poincare_distance / great_circle_distance / euclidean norm),
 *   w_m, γ_m, λ_m are per-manifold learnable scalars.
 *
 * Born-rule output projection over a vocab of size V:
 *
 *   P(w) = |h_amp[w] + s_amp[w] + e_amp[w]|² / Z
 *   Z    = Σ_w |h_amp[w] + s_amp[w] + e_amp[w]|²
 *
 * Real-amplitude form is implemented now (|·|² → sum²); complex form
 * lands when GeometricLM moves to complex-valued amplitudes (v0.2).
 *
 * Backward kernels are scaffolded in the Python autograd wrappers using
 * forward composition until the dedicated C backward primitives ship.
 *
 * fp32 IO + accumulators. Native Metal/CUDA kernels are a v0.2 follow-up;
 * the CPU reference unblocks the GeometricLM forward/backward today.
 */

#include "tensorcore/status.h"
#include "tensorcore/tensorcore.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Combine pre-computed per-manifold inner products, phase diffs and
 * distances into one score per Q-K pair.
 *
 *   inner_products  : [N_pairs, M] fp32 — caller's GEMM outputs
 *   phase_diffs     : [N_pairs, M] fp32 — Δφ_m per pair, per manifold
 *   distances       : [N_pairs, M] fp32 — d_m per pair (positive)
 *   weights, gammas, lambdas : [M] fp32 each — per-manifold scalars
 *   scores_out      : [N_pairs] fp32
 */
tc_status_t tc_phase_attention_combine(tc_context* ctx,
                                        const tc_buffer* inner_products,
                                        const tc_buffer* phase_diffs,
                                        const tc_buffer* distances,
                                        const tc_buffer* weights,
                                        const tc_buffer* gammas,
                                        const tc_buffer* lambdas,
                                        tc_buffer* scores_out,
                                        int N_pairs, int M);

/* Born-rule output projection (real amplitudes form):
 *
 *   probs[n, w] = (h_amp[n, w] + s_amp[n, w] + e_amp[n, w])²  /  Z[n]
 *   Z[n]        = Σ_w (h_amp[n, w] + s_amp[n, w] + e_amp[n, w])²
 *
 * Output is a proper probability distribution (rows sum to 1).
 *
 *   h_amp, s_amp, e_amp, probs_out : [N, V] fp32
 *   N : batch / token dim, V : vocab dim.
 *
 * Pass e_amp = NULL to skip the third amplitude; pass s_amp = NULL to
 * skip both s and e (callers controlling which manifolds contribute). */
tc_status_t tc_born_rule_output(tc_context* ctx,
                                 const tc_buffer* h_amp,
                                 const tc_buffer* s_amp,
                                 const tc_buffer* e_amp,
                                 tc_buffer* probs_out,
                                 int N, int V);

#ifdef __cplusplus
}
#endif
#endif
