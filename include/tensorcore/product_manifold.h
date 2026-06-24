#ifndef TENSORCORE_PRODUCT_MANIFOLD_H
#define TENSORCORE_PRODUCT_MANIFOLD_H

/*
 * tensorcore — Product manifold (mixed curvature) primitives.
 *
 * A point on a product manifold M = M_1 × M_2 × ... × M_K is the
 * concatenation of points on each factor. Geodesic structure factorises:
 *   exp_{(p1,...,pK)}((v1,...,vK)) = (exp_{p1}(v1), ..., exp_{pK}(vK))
 *   log_{(p1,...,pK)}((q1,...,qK)) = (log_{p1}(q1), ..., log_{pK}(qK))
 *   d² = Σ_k d_k²  (Riemannian product metric)
 *   PT factor-wise.
 *
 * Factor kinds:
 *   TC_FACTOR_EUCLIDEAN — R^d, additive (no curvature)
 *   TC_FACTOR_POINCARE  — Poincaré ball of curvature c > 0 (d intrinsic)
 *   TC_FACTOR_SPHERE    — n-sphere of radius r (d+1 ambient = d intrinsic + 1)
 *   TC_FACTOR_LORENTZ   — Hyperboloid of curvature c (d+1 ambient = d intrinsic + 1)
 *
 * Layout: a point lives in a single flat fp32 buffer of length
 *   sum_k tc_factor_ambient_dim(factors[k]).
 * Concatenation order matches factor[] order. Helpers below compute
 * the layout, project, exp/log/distance/parallel transport, all
 * factor-by-factor under the hood. Replaces qLLM's mixed_curvature.c
 * for the H × S × R product manifold used by GeoRefine's
 * GeometricLM (semiclassical_qllm).
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TC_FACTOR_EUCLIDEAN = 0,
    TC_FACTOR_POINCARE  = 1,
    TC_FACTOR_SPHERE    = 2,
    TC_FACTOR_LORENTZ   = 3,
    TC_FACTOR_TORUS     = 4,
} tc_factor_kind_t;

typedef struct {
    tc_factor_kind_t kind;
    int32_t intrinsic_dim;    /* manifold's intrinsic dimension */
    float   curvature;        /* poincare c; sphere radius; lorentz |k|; euclidean ignored */
} tc_factor_t;

/* Ambient dim per factor (Sphere/Lorentz add 1 for the ambient coordinate). */
int32_t tc_factor_ambient_dim(const tc_factor_t* factor);

/* Sum of ambient dims = total flat buffer length for one product point. */
int32_t tc_product_ambient_dim(const tc_factor_t* factors, int32_t n_factors);

/* Project an arbitrary flat vector onto the product manifold, factor by
 * factor. For Euclidean: identity. For Poincaré: clip into ball. For
 * Sphere/Lorentz: project onto the respective sheet. */
void tc_product_project(const tc_factor_t* factors, int32_t n_factors,
                         float* x);

/* exp_p(v) on the product: factor-by-factor. */
void tc_product_exp(const tc_factor_t* factors, int32_t n_factors,
                     const float* base, const float* tangent, float* point);

/* log_p(q) on the product: factor-by-factor. */
void tc_product_log(const tc_factor_t* factors, int32_t n_factors,
                     const float* base, const float* point, float* tangent);

/* Riemannian product distance: d² = Σ_k d_k(p_k, q_k)². */
float tc_product_distance(const tc_factor_t* factors, int32_t n_factors,
                           const float* p, const float* q);

/* Parallel transport, factor-by-factor (Sasaki product connection). */
void tc_product_parallel_transport(const tc_factor_t* factors, int32_t n_factors,
                                    const float* from, const float* to,
                                    const float* tangent, float* out);

#ifdef __cplusplus
}
#endif
#endif
