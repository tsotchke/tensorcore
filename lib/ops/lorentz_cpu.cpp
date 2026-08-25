/*
 * tensorcore — Lorentz / hyperboloid manifold (CPU implementation).
 *
 * Math reference: see include/tensorcore/lorentz.h. fp32 closed-form
 * implementations; algorithms match qLLM's lorentz_fast.c so that any
 * caller can swap qllm_lorentz_* for tc_lorentz_* without behavior change.
 *
 * Numerical safety:
 *   - LORENTZ_EPS = 1e-7f for curvature clamps
 *   - arcosh(α≈1) uses the sqrt(2(α-1)) series expansion to avoid
 *     catastrophic cancellation
 *   - exp_map post-projects onto the sheet to absorb FP32 drift
 *   - parallel_transport post-projects onto T_to H
 */

#include "tensorcore/lorentz.h"
#include <cmath>
#include <cstring>

namespace {
constexpr float kLorentzEps = 1e-7f;
}

extern "C" float tc_lorentz_minkowski(const float* a, const float* b, size_t n) {
    float s = -a[0] * b[0];
    for (size_t i = 1; i < n; ++i) s += a[i] * b[i];
    return s;
}

extern "C" void tc_lorentz_project(float* x, size_t n, float curvature) {
    float c = std::fabs(curvature);
    if (c < kLorentzEps) c = kLorentzEps;
    const float R2 = 1.0f / c;
    float spatial_sq = 0.0f;
    for (size_t i = 1; i < n; ++i) spatial_sq += x[i] * x[i];
    /* Always upper sheet: x_0 = +sqrt(R² + |spatial|²). */
    x[0] = std::sqrt(R2 + spatial_sq);
}

extern "C" void tc_lorentz_tangent_project(const float* base, float* v, size_t n) {
    /* α = <base, v>_L / <base, base>_L; subtract α·base from v. */
    const float pp = tc_lorentz_minkowski(base, base, n);
    if (std::fabs(pp) < kLorentzEps) return;
    const float pv = tc_lorentz_minkowski(base, v, n);
    const float alpha = pv / pp;
    for (size_t i = 0; i < n; ++i) v[i] -= alpha * base[i];
}

extern "C" float tc_lorentz_inner_product(const float* base,
                                          const float* u, const float* v,
                                          size_t n) {
    /* Minkowski form on T_p H is positive-definite; identical to the
     * ambient Minkowski form on tangent vectors. `base` is required so
     * the API matches Riemannian inner-product signatures elsewhere. */
    (void)base;
    return tc_lorentz_minkowski(u, v, n);
}

extern "C" void tc_lorentz_add(const float* base,
                                const float* tangent_x, const float* tangent_y,
                                float* out_tangent, size_t n) {
    for (size_t i = 0; i < n; ++i) out_tangent[i] = tangent_x[i] + tangent_y[i];
    tc_lorentz_tangent_project(base, out_tangent, n);
}

extern "C" void tc_lorentz_scale(const float* tangent, float s, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = s * tangent[i];
}

extern "C" void tc_lorentz_exp(const float* base, const float* tangent,
                                float* point, size_t n, float curvature) {
    float c = std::fabs(curvature);
    if (c < kLorentzEps) c = kLorentzEps;
    const float R2 = 1.0f / c;
    const float R  = std::sqrt(R2);

    float vv = tc_lorentz_minkowski(tangent, tangent, n);
    if (vv < 0.0f) vv = 0.0f;
    const float v_norm = std::sqrt(vv);
    const float theta = v_norm / R;

    if (theta < 1e-7f) {
        for (size_t i = 0; i < n; ++i) point[i] = base[i] + tangent[i];
    } else {
        const float ch = std::cosh(theta);
        const float sh_over_norm = std::sinh(theta) / v_norm;
        for (size_t i = 0; i < n; ++i) {
            point[i] = ch * base[i] + sh_over_norm * tangent[i];
        }
    }
    /* Absorb FP32 drift off the sheet. */
    tc_lorentz_project(point, n, curvature);
}

extern "C" void tc_lorentz_log(const float* base, const float* point,
                                float* tangent, size_t n, float curvature) {
    float c = std::fabs(curvature);
    if (c < kLorentzEps) c = kLorentzEps;
    const float R2 = 1.0f / c;

    const float pq = tc_lorentz_minkowski(base, point, n);
    float alpha = -pq / R2;
    if (alpha < 1.0f) alpha = 1.0f;

    float w_norm_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        tangent[i] = point[i] - alpha * base[i];
        if (i == 0) w_norm_sq += -tangent[i] * tangent[i];
        else        w_norm_sq +=  tangent[i] * tangent[i];
    }
    if (w_norm_sq <= 0.0f) {
        std::memset(tangent, 0, n * sizeof(float));
        return;
    }
    const float w_norm = std::sqrt(w_norm_sq);

    /* arcosh(α) with stable α≈1 branch. */
    float theta;
    if (alpha < 1.0f + 1e-6f) {
        theta = std::sqrt(2.0f * (alpha - 1.0f));
        if (theta < 0.0f) theta = 0.0f;
    } else {
        theta = std::acosh(alpha);
    }

    const float R = std::sqrt(R2);
    const float scale = R * theta / w_norm;
    for (size_t i = 0; i < n; ++i) tangent[i] *= scale;
    tc_lorentz_tangent_project(base, tangent, n);
}

extern "C" float tc_lorentz_distance(const float* p, const float* q, size_t n,
                                      float curvature) {
    float c = std::fabs(curvature);
    if (c < kLorentzEps) c = kLorentzEps;
    const float R2 = 1.0f / c;
    const float R  = std::sqrt(R2);

    const float pq = tc_lorentz_minkowski(p, q, n);
    float alpha = -pq / R2;
    if (alpha < 1.0f) alpha = 1.0f;
    return R * std::acosh(alpha);
}

extern "C" void tc_lorentz_parallel_transport(const float* from, const float* to,
                                               const float* tangent, float* out,
                                               size_t n, float curvature) {
    float c = std::fabs(curvature);
    if (c < kLorentzEps) c = kLorentzEps;
    const float R2 = 1.0f / c;

    const float pq = tc_lorentz_minkowski(from, to, n);
    const float diff = pq + R2;   /* ≤ 0; equals 0 only when from == to */
    if (std::fabs(diff) < 1e-8f) {
        if (out != tangent) std::memcpy(out, tangent, n * sizeof(float));
        return;
    }
    const float qv = tc_lorentz_minkowski(to, tangent, n);
    const float coef = qv / diff;
    for (size_t i = 0; i < n; ++i) {
        out[i] = tangent[i] - coef * (from[i] + to[i]);
    }
    tc_lorentz_tangent_project(to, out, n);
}

extern "C" void tc_lorentz_to_poincare(const float* h, float* p, size_t n,
                                        float curvature) {
    float c = std::fabs(curvature);
    if (c < kLorentzEps) c = kLorentzEps;
    const float R = 1.0f / std::sqrt(c);
    float denom = h[0] + R;
    if (std::fabs(denom) < 1e-12f) denom = (denom < 0.0f) ? -1e-12f : 1e-12f;
    const float scale = 1.0f / denom;
    for (size_t i = 1; i < n; ++i) {
        p[i - 1] = h[i] * scale;
    }
}

extern "C" void tc_poincare_to_lorentz(const float* p, float* h, size_t n,
                                        float curvature) {
    float c = std::fabs(curvature);
    if (c < kLorentzEps) c = kLorentzEps;
    const float R = 1.0f / std::sqrt(c);

    float p_norm_sq = 0.0f;
    for (size_t i = 1; i < n; ++i) p_norm_sq += p[i - 1] * p[i - 1];
    float u = 1.0f - p_norm_sq * c;
    if (u < 1e-7f) u = 1e-7f;
    const float inv_u = 1.0f / u;

    h[0] = R * (1.0f + p_norm_sq * c) * inv_u;
    for (size_t i = 1; i < n; ++i) {
        h[i] = 2.0f * R * p[i - 1] * inv_u;
    }
}
