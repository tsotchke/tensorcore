#ifndef TENSORCORE_SPHERE_H
#define TENSORCORE_SPHERE_H

/*
 * tensorcore — Sphere manifold (round n-sphere) primitives.
 *
 * S^n_r = { x ∈ R^{n+1} | <x, x> = r² }, with the standard Euclidean
 * inner product. Tangent space T_p S = { v | <p, v> = 0 }.
 *
 * Closed-form maps:
 *   exp_p(v) = cos(‖v‖/r) p + sinc(‖v‖/r) v
 *   log_p(q) = (θ / sin θ) (q - cos θ · p),   θ = arccos(<p, q> / r²)
 *   d(p, q)  = r · θ
 *
 * Parallel transport (Riemannian connection on the round sphere):
 *   PT_{p→q}(v) = v - <q, v> / (r² + <p, q>) · (p + q)
 *
 * Numerics:
 *   - Small-θ branch uses Taylor (1 - θ²/6 for sinc, 1 - θ²/2 for cos)
 *     to avoid catastrophic cancellation near identity geodesics.
 *   - cos θ is clamped to [-1, 1] before acos.
 *   - Antipodal handling (θ ≈ π) returns 0 tangent (log is multi-valued
 *     there; behaviour matches qLLM spherical_fast.c).
 *   - exp_map projects the result back onto the sphere to absorb FP32
 *     drift; project() rescales to length r exactly.
 *
 * Sphere is the H × S × R "S" factor in GeoRefine's GeometricLM
 * product manifold (see include/tensorcore/product_manifold.h).
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Euclidean inner product on R^{n+1}. */
float tc_sphere_inner(const float* a, const float* b, size_t n);

/* Project an arbitrary R^{n+1} vector onto S^n_r by rescaling to ‖·‖ = r. */
void tc_sphere_project(float* x, size_t n, float radius);

/* Project a tangent v at base p onto T_p S: subtract the radial component. */
void tc_sphere_tangent_project(const float* base, float* v, size_t n);

/* Tangent inner product (Euclidean restricted to T_p S, positive-definite). */
float tc_sphere_inner_product(const float* base,
                               const float* u, const float* v, size_t n);

/* Tangent addition (with tangent_project to absorb FP32 drift). */
void tc_sphere_add(const float* base,
                   const float* tangent_x, const float* tangent_y,
                   float* out_tangent, size_t n);

/* Scalar multiplication on the tangent space. */
void tc_sphere_scale(const float* tangent, float s, float* out, size_t n);

/* exp_p(v) — returns a point on S^n_r. */
void tc_sphere_exp(const float* base, const float* tangent,
                   float* point, size_t n, float radius);

/* log_p(q) — returns a vector in T_p S. Antipodal q returns 0. */
void tc_sphere_log(const float* base, const float* point,
                   float* tangent, size_t n, float radius);

/* Geodesic distance d(p, q) = r · arccos(<p, q> / r²). */
float tc_sphere_distance(const float* p, const float* q, size_t n,
                          float radius);

/* Parallel transport of `tangent` ∈ T_from S along the geodesic from
 * `from` to `to`. Near-antipodal: returns tangent unchanged. */
void tc_sphere_parallel_transport(const float* from, const float* to,
                                   const float* tangent, float* out,
                                   size_t n, float radius);

/* Spherical linear interpolation:
 *   slerp(p, q, t) = sin((1-t)θ)/sin θ · p + sin(tθ)/sin θ · q,
 *   θ = arccos(<p, q> / r²).  t ∈ [0, 1]. */
void tc_sphere_slerp(const float* p, const float* q, float t,
                     float* out, size_t n, float radius);

#ifdef __cplusplus
}
#endif
#endif
