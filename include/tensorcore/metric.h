#ifndef TENSORCORE_METRIC_H
#define TENSORCORE_METRIC_H

/*
 * tensorcore — Riemannian metric tensor (general manifold).
 *
 * Encapsulates a metric g_{ij}(p) on a d-dimensional manifold via a
 * user-supplied callback. Provides:
 *
 *   - tc_metric_apply         : g_{ij}(p) v^i w^j   (raise/lower-friendly inner product)
 *   - tc_metric_inverse_apply : g^{ij}(p) α_i β_j   (cotangent inner product)
 *   - tc_metric_inverse       : g^{ij} = (g_{ij})⁻¹ via Gauss-Jordan
 *   - tc_metric_christoffel   : Γ^k_{ij}(p) via central-difference + g⁻¹
 *
 * The metric callback writes the d×d symmetric matrix g_{ij}(p) into a
 * row-major float buffer. This means any manifold whose metric can be
 * evaluated point-wise — Euclidean, Poincaré, Sphere, Lorentz, arbitrary
 * Riemannian — plugs into the geodesic ODE solver in geodesic.h.
 *
 * Christoffel symbols of the second kind:
 *   Γ^k_{ij} = ½ g^{kℓ} (∂_i g_{jℓ} + ∂_j g_{iℓ} - ∂_ℓ g_{ij})
 *
 * The partials are evaluated by symmetric central difference with a
 * caller-supplied step h (default 1e-3 if h ≤ 0). Output layout is
 * row-major: christoffel[k * d * d + i * d + j] = Γ^k_{ij}.
 *
 * Replaces qLLM's metric_tensor.c / riemannian_metrics.c surface
 * with a substrate-level primitive every consumer (qLLM, QGTL,
 * Noesis) can call.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Metric callback: write the d×d symmetric matrix g_{ij}(p) into
 * `out` (row-major, d² floats). `user` is opaque caller state. */
typedef void (*tc_metric_fn)(const float* point, int dim,
                             float* out_g, void* user);

/* g_{ij}(p) v^i w^j = vᵀ g(p) w. */
float tc_metric_apply(tc_metric_fn fn, void* user,
                      const float* point, int dim,
                      const float* v, const float* w);

/* Invert a d×d symmetric positive-definite matrix g into g_inv via
 * Gauss-Jordan with partial pivoting. Returns 0 on success, nonzero
 * if g is singular within fp32 tolerance. */
int tc_metric_inverse(const float* g, int dim, float* g_inv);

/* g^{ij}(p) α_i β_j — convenience: builds g(p), inverts, then applies. */
float tc_metric_inverse_apply(tc_metric_fn fn, void* user,
                              const float* point, int dim,
                              const float* alpha, const float* beta);

/* Christoffel symbols Γ^k_{ij}(p) via central-difference partials.
 * `h` is the finite-difference step; pass h ≤ 0 for the default (1e-3).
 * `out_christoffel` must hold d³ floats; layout
 *   out[k * d * d + i * d + j] = Γ^k_{ij}.
 * Returns 0 on success, nonzero if the metric is singular at `point`. */
int tc_metric_christoffel(tc_metric_fn fn, void* user,
                          const float* point, int dim, float h,
                          float* out_christoffel);

/* ---- Stock metric callbacks (bind these to tc_metric_fn) ---- */

/* Euclidean: g_{ij} = δ_{ij}. `user` is ignored. */
void tc_metric_euclidean(const float* point, int dim,
                         float* out_g, void* user);

/* Poincaré ball with curvature c > 0: g_{ij}(x) = λ²(x) δ_{ij},
 * λ(x) = 2 / (1 - c ‖x‖²). Pass &c as `user` (float*). */
void tc_metric_poincare(const float* point, int dim,
                        float* out_g, void* user);

/* Sphere stereographic chart for S^d of radius r, x ∈ R^d:
 *   g_{ij}(x) = (2r² / (r² + ‖x‖²))² δ_{ij}.
 * Pass &r as `user` (float*). Lives on R^d directly (intrinsic dim),
 * not on the ambient (d+1)-sphere coordinates used by sphere.h. */
void tc_metric_sphere_stereographic(const float* point, int dim,
                                    float* out_g, void* user);

#ifdef __cplusplus
}
#endif
#endif
