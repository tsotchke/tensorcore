#ifndef TENSORCORE_GEODESIC_H
#define TENSORCORE_GEODESIC_H

/*
 * tensorcore — Geodesic ODE solver (RK4 on arbitrary metric).
 *
 * Solves the geodesic equation
 *
 *   γ̈^k + Γ^k_{ij}(γ) γ̇^i γ̇^j = 0
 *
 * on a d-dimensional Riemannian manifold whose metric g_{ij}(p) is
 * exposed via a tc_metric_fn callback (see include/tensorcore/metric.h).
 * Christoffel symbols are evaluated numerically at every substep, so
 * any metric callback works without an analytic Γ.
 *
 * Public surface:
 *   - tc_geodesic_step       : one RK4 step of length dt
 *   - tc_geodesic_integrate  : N steps; writes final state to outputs
 *
 * State is the phase-space pair (position γ ∈ R^d, velocity γ̇ ∈ R^d),
 * both stored as flat fp32 buffers of length d. Integration runs in
 * double internally per-step then casts back to fp32 outputs to keep
 * the ODE from drifting under fp32 accumulation of Γ.
 *
 * Replaces qLLM's geodesic_solver.c + fast_geodesic.c with a substrate
 * primitive that any manifold can drive.
 */

#include <stddef.h>
#include "tensorcore/metric.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One RK4 step:
 *   (γ, γ̇)  -->  (γ', γ̇')   after time dt.
 * `h_christoffel` is the finite-difference step for the metric callback
 * passed to tc_metric_christoffel; pass h_christoffel ≤ 0 for default.
 * Returns 0 on success, nonzero if the metric is singular at any
 * RK4 substep. */
int tc_geodesic_step(tc_metric_fn fn, void* user,
                     int dim, float dt, float h_christoffel,
                     const float* pos_in,  const float* vel_in,
                     float* pos_out,       float* vel_out);

/* Integrate `n_steps` of length `dt` starting from (pos0, vel0); writes
 * the final state into (pos_out, vel_out). pos_out / vel_out may alias
 * pos0 / vel0 — the solver double-buffers internally. */
int tc_geodesic_integrate(tc_metric_fn fn, void* user,
                          int dim, float dt, int n_steps,
                          float h_christoffel,
                          const float* pos0, const float* vel0,
                          float* pos_out,    float* vel_out);

#ifdef __cplusplus
}
#endif
#endif
