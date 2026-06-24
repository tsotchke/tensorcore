/*
 * tensorcore — Product manifold (CPU implementation).
 *
 * Math reference: see include/tensorcore/product_manifold.h. The
 * algorithm is straight Sasaki/Riemannian product: every map
 * factorises across factors, distance is the L² combination of
 * factor distances. Replaces qLLM's mixed_curvature.c for the
 * H × S × R product used by GeometricLM.
 *
 * The Poincaré factor here uses the raw fp32 single-vector API
 * implemented inline below (the public tc_poincare_* ops are batched
 * over tc_buffer; we need single-vector here). Sphere and Lorentz
 * delegate to their single-vector kernels.
 */

#include "tensorcore/product_manifold.h"
#include "tensorcore/sphere.h"
#include "tensorcore/lorentz.h"
#include <cmath>
#include <cstring>

namespace {

constexpr float kBoundaryMargin = 1e-5f;
constexpr float kNormFloor     = 1e-15f;

/* --- Poincaré ball (single-vector fp32) ----------------------------- */

float poincare_norm(const float* x, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += x[i] * x[i];
    return std::sqrt(s + kNormFloor * kNormFloor);
}

void poincare_project_inplace(float* x, int n, float c) {
    /* Clip to ball of radius (1 - margin)/sqrt(c). */
    if (c <= 0.0f) return;
    const float max_norm = (1.0f - kBoundaryMargin) / std::sqrt(c);
    const float nrm = poincare_norm(x, n);
    if (nrm > max_norm) {
        const float scale = max_norm / nrm;
        for (int i = 0; i < n; ++i) x[i] *= scale;
    }
}

float poincare_distance(const float* x, const float* y, int n, float c) {
    /* d_c(x, y) = (2/√c) · atanh(√c · ‖-x ⊕ y‖); approximated via
     * the equivalent (2/√c) atanh(√c (‖x-y‖ ‖x-y‖) / (‖x‖² ‖y‖² formula).
     * For simplicity we use the safe form via log_x. */
    if (c <= 0.0f) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) { float d = x[i] - y[i]; s += d * d; }
        return 2.0f * std::sqrt(s);
    }
    /* Möbius difference -x ⊕ y has well-defined norm; use:
     *   d_c(x, y) = (2/√c) · atanh(√c · ‖m‖)
     * with m = (1 - 2c<x,y> + c‖y‖²)x + (1 - c‖x‖²)y, normalized.
     * Closed-form Ganea identity:
     *   d_c(x, y) = acosh(1 + 2 c ‖x-y‖² / ((1 - c ‖x‖²)(1 - c ‖y‖²)))
     *               / √c. */
    const float sc = std::sqrt(c);
    float dxy_sq = 0.0f, x_sq = 0.0f, y_sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = x[i] - y[i];
        dxy_sq += d * d;
        x_sq += x[i] * x[i];
        y_sq += y[i] * y[i];
    }
    const float denom = (1.0f - c * x_sq) * (1.0f - c * y_sq);
    if (denom < kNormFloor) return 0.0f;
    const float arg = 1.0f + 2.0f * c * dxy_sq / denom;
    return std::acosh(arg < 1.0f ? 1.0f : arg) / sc;
}

/* For PT on Poincaré at non-zero base — gyro PT. Here we use the
 * standard formula PT_{x→y}(v) = λ_x / λ_y · gyr[y, -x] v. Approximate
 * with the small-distance identity when ‖log_x y‖ is small. For now,
 * implement the conformal-factor scaling only — sufficient for the
 * GeoRefine product-manifold use case where transport is dominated by
 * the conformal factor. */
void poincare_parallel_transport(const float* from, const float* to,
                                  const float* tangent, float* out,
                                  int n, float c) {
    if (c <= 0.0f) {
        for (int i = 0; i < n; ++i) out[i] = tangent[i];
        return;
    }
    float fn2 = 0.0f, tn2 = 0.0f;
    for (int i = 0; i < n; ++i) { fn2 += from[i] * from[i]; tn2 += to[i] * to[i]; }
    const float lambda_from = 2.0f / (1.0f - c * fn2 + kNormFloor);
    const float lambda_to   = 2.0f / (1.0f - c * tn2 + kNormFloor);
    const float scale = lambda_from / lambda_to;
    for (int i = 0; i < n; ++i) out[i] = scale * tangent[i];
}

}  // namespace

extern "C" int32_t tc_factor_ambient_dim(const tc_factor_t* factor) {
    switch (factor->kind) {
        case TC_FACTOR_SPHERE:
        case TC_FACTOR_LORENTZ:
            return factor->intrinsic_dim + 1;
        case TC_FACTOR_EUCLIDEAN:
        case TC_FACTOR_POINCARE:
        default:
            return factor->intrinsic_dim;
    }
}

extern "C" int32_t tc_product_ambient_dim(const tc_factor_t* factors, int32_t n_factors) {
    int32_t total = 0;
    for (int32_t k = 0; k < n_factors; ++k) total += tc_factor_ambient_dim(&factors[k]);
    return total;
}

extern "C" void tc_product_project(const tc_factor_t* factors, int32_t n_factors,
                                    float* x) {
    int32_t off = 0;
    for (int32_t k = 0; k < n_factors; ++k) {
        const tc_factor_t* f = &factors[k];
        const int32_t adim = tc_factor_ambient_dim(f);
        switch (f->kind) {
            case TC_FACTOR_EUCLIDEAN: break;  /* identity */
            case TC_FACTOR_POINCARE:
                poincare_project_inplace(x + off, adim, f->curvature);
                break;
            case TC_FACTOR_SPHERE:
                tc_sphere_project(x + off, (size_t)adim, f->curvature);
                break;
            case TC_FACTOR_LORENTZ:
                tc_lorentz_project(x + off, (size_t)adim, f->curvature);
                break;
        }
        off += adim;
    }
}

extern "C" void tc_product_exp(const tc_factor_t* factors, int32_t n_factors,
                                const float* base, const float* tangent,
                                float* point) {
    int32_t off = 0;
    for (int32_t k = 0; k < n_factors; ++k) {
        const tc_factor_t* f = &factors[k];
        const int32_t adim = tc_factor_ambient_dim(f);
        switch (f->kind) {
            case TC_FACTOR_EUCLIDEAN:
                for (int32_t i = 0; i < adim; ++i)
                    point[off + i] = base[off + i] + tangent[off + i];
                break;
            case TC_FACTOR_POINCARE: {
                /* Tangent at base, exp via the base-shifted formula.
                 * For simplicity we use the small-tangent approximation
                 * exp_p(v) ≈ p + v then project; full Möbius exp_p is in
                 * lib/ops/poincare_cpu.cpp (batched). The product-manifold
                 * use case keeps tangents small (one RiemannianAdam step). */
                for (int32_t i = 0; i < adim; ++i)
                    point[off + i] = base[off + i] + tangent[off + i];
                poincare_project_inplace(point + off, adim, f->curvature);
                break;
            }
            case TC_FACTOR_SPHERE:
                tc_sphere_exp(base + off, tangent + off, point + off,
                              (size_t)adim, f->curvature);
                break;
            case TC_FACTOR_LORENTZ:
                tc_lorentz_exp(base + off, tangent + off, point + off,
                               (size_t)adim, f->curvature);
                break;
        }
        off += adim;
    }
}

extern "C" void tc_product_log(const tc_factor_t* factors, int32_t n_factors,
                                const float* base, const float* point,
                                float* tangent) {
    int32_t off = 0;
    for (int32_t k = 0; k < n_factors; ++k) {
        const tc_factor_t* f = &factors[k];
        const int32_t adim = tc_factor_ambient_dim(f);
        switch (f->kind) {
            case TC_FACTOR_EUCLIDEAN:
                for (int32_t i = 0; i < adim; ++i)
                    tangent[off + i] = point[off + i] - base[off + i];
                break;
            case TC_FACTOR_POINCARE:
                for (int32_t i = 0; i < adim; ++i)
                    tangent[off + i] = point[off + i] - base[off + i];
                break;
            case TC_FACTOR_SPHERE:
                tc_sphere_log(base + off, point + off, tangent + off,
                              (size_t)adim, f->curvature);
                break;
            case TC_FACTOR_LORENTZ:
                tc_lorentz_log(base + off, point + off, tangent + off,
                               (size_t)adim, f->curvature);
                break;
        }
        off += adim;
    }
}

extern "C" float tc_product_distance(const tc_factor_t* factors, int32_t n_factors,
                                      const float* p, const float* q) {
    float d2 = 0.0f;
    int32_t off = 0;
    for (int32_t k = 0; k < n_factors; ++k) {
        const tc_factor_t* f = &factors[k];
        const int32_t adim = tc_factor_ambient_dim(f);
        float d_k = 0.0f;
        switch (f->kind) {
            case TC_FACTOR_EUCLIDEAN: {
                float s = 0.0f;
                for (int32_t i = 0; i < adim; ++i) {
                    float d = p[off + i] - q[off + i];
                    s += d * d;
                }
                d_k = std::sqrt(s);
                break;
            }
            case TC_FACTOR_POINCARE:
                d_k = poincare_distance(p + off, q + off, adim, f->curvature);
                break;
            case TC_FACTOR_SPHERE:
                d_k = tc_sphere_distance(p + off, q + off,
                                          (size_t)adim, f->curvature);
                break;
            case TC_FACTOR_LORENTZ:
                d_k = tc_lorentz_distance(p + off, q + off,
                                           (size_t)adim, f->curvature);
                break;
        }
        d2 += d_k * d_k;
        off += adim;
    }
    return std::sqrt(d2);
}

extern "C" void tc_product_parallel_transport(const tc_factor_t* factors,
                                               int32_t n_factors,
                                               const float* from, const float* to,
                                               const float* tangent, float* out) {
    int32_t off = 0;
    for (int32_t k = 0; k < n_factors; ++k) {
        const tc_factor_t* f = &factors[k];
        const int32_t adim = tc_factor_ambient_dim(f);
        switch (f->kind) {
            case TC_FACTOR_EUCLIDEAN:
                for (int32_t i = 0; i < adim; ++i) out[off + i] = tangent[off + i];
                break;
            case TC_FACTOR_POINCARE:
                poincare_parallel_transport(from + off, to + off, tangent + off,
                                             out + off, adim, f->curvature);
                break;
            case TC_FACTOR_SPHERE:
                tc_sphere_parallel_transport(from + off, to + off, tangent + off,
                                              out + off, (size_t)adim, f->curvature);
                break;
            case TC_FACTOR_LORENTZ:
                tc_lorentz_parallel_transport(from + off, to + off, tangent + off,
                                               out + off, (size_t)adim, f->curvature);
                break;
        }
        off += adim;
    }
}
