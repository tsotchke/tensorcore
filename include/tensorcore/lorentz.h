#ifndef TENSORCORE_LORENTZ_H
#define TENSORCORE_LORENTZ_H

/*
 * tensorcore — Lorentz / hyperboloid manifold primitives.
 *
 * Hyperboloid model of n-dimensional hyperbolic space with curvature κ < 0:
 *   H^n_κ = { x ∈ R^{n+1} | <x, x>_L = -1/|κ|,  x_0 > 0 }
 *   <a, b>_L = -a_0 b_0 + Σ_{i≥1} a_i b_i.
 *
 * Tangent space at p:  T_p H = { v ∈ R^{n+1} | <p, v>_L = 0 }.
 *
 * Closed-form maps (let R = 1/sqrt(|κ|)):
 *   exp_p(v) = cosh(‖v‖_L / R) p + R sinh(‖v‖_L / R) v / ‖v‖_L
 *   log_p(q) = (R · θ) · w / ‖w‖_L,  θ = arcosh(-κ <p, q>_L),
 *              w = q - cosh(θ) · p
 *   d(p, q)  = R · arcosh(-κ <p, q>_L)
 *
 * Parallel transport (Nickel & Kiela 2018):
 *   PT_{p→q}(v) = v - <q, v>_L / (R² + <p, q>_L) · (p + q)
 *
 * Vectors are passed as flat float arrays of length `n` (ambient dim,
 * i.e. one MORE than the intrinsic hyperbolic dimension). Curvature is
 * a single float; |κ| is used internally so callers can pass negative
 * values without changing meaning.
 *
 * Numerical safety: ε = 1e-7 on curvature, sheet-projection after every
 * exp_map (absorbs FP32 drift off the hyperboloid).
 *
 * Companion bridges (Poincaré ball ↔ hyperboloid) at the bottom let
 * callers convert between the two equivalent models when needed.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Minkowski inner product <a, b>_L on R^{n+1}. */
float tc_lorentz_minkowski(const float* a, const float* b, size_t n);

/* Project an arbitrary (n+1)-vector onto the upper sheet
 *   { x | <x, x>_L = -1/|κ|, x_0 > 0 }
 * by keeping spatial coords intact and recomputing x_0. */
void tc_lorentz_project(float* x, size_t n, float curvature);

/* Project a vector v at the implicit base point p onto T_p H. */
void tc_lorentz_tangent_project(const float* base, float* v, size_t n);

/* Tangent inner product (Minkowski restricted to T_p H — positive-definite there). */
float tc_lorentz_inner_product(const float* base,
                                const float* u, const float* v, size_t n);

/* Tangent addition (with tangent-project to absorb FP32 drift). */
void tc_lorentz_add(const float* base,
                    const float* tangent_x, const float* tangent_y,
                    float* out_tangent, size_t n);

/* Scalar multiplication on the tangent space. */
void tc_lorentz_scale(const float* tangent, float s, float* out, size_t n);

/* Exponential map exp_p(v) — returns a point on the upper sheet. */
void tc_lorentz_exp(const float* base, const float* tangent,
                    float* point, size_t n, float curvature);

/* Logarithm map log_p(q) — returns a vector in T_p H. */
void tc_lorentz_log(const float* base, const float* point,
                    float* tangent, size_t n, float curvature);

/* Geodesic distance d(p, q) = R · arcosh(-κ <p, q>_L). */
float tc_lorentz_distance(const float* p, const float* q, size_t n,
                          float curvature);

/* Parallel transport of `tangent` ∈ T_from H along the geodesic from
 * `from` to `to`. */
void tc_lorentz_parallel_transport(const float* from, const float* to,
                                    const float* tangent, float* out,
                                    size_t n, float curvature);

/* Hyperboloid → Poincaré ball conversion.
 *   h has length n (ambient), p has length n-1 (intrinsic). */
void tc_lorentz_to_poincare(const float* h, float* p, size_t n,
                             float curvature);

/* Poincaré ball → hyperboloid conversion.
 *   p has length n-1 (intrinsic), h has length n (ambient).
 *   Caller passes n = intrinsic + 1 (ambient dim). */
void tc_poincare_to_lorentz(const float* p, float* h, size_t n,
                             float curvature);

#ifdef __cplusplus
}
#endif
#endif
