/*
 * tensorcore — Sphere manifold (CPU implementation).
 *
 * Math reference: see include/tensorcore/sphere.h. fp32 closed-form
 * implementations match qLLM's fast_sphere_* APIs in spherical_fast.c
 * (substituting tc_sphere for fast_sphere) so consumers can swap one
 * for the other with no behavioural change.
 *
 * Numerical safety:
 *   - Norm floor 1e-30 before sqrt/divide (matches qLLM)
 *   - Small-θ Taylor branch under 1e-6
 *   - cos θ clamped to [-1, 1] before acos
 *   - Antipodal log returns 0 tangent
 *   - exp post-projects onto S^n_r
 */

#include "tensorcore/sphere.h"
#include <cmath>
#include <cstring>

namespace {
constexpr float kSphereEps = 1e-30f;
constexpr float kSmallTheta = 1e-6f;
constexpr float kPi = 3.14159265358979323846f;
}

extern "C" float tc_sphere_inner(const float* a, const float* b, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

extern "C" void tc_sphere_project(float* x, size_t n, float radius) {
    float norm_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) norm_sq += x[i] * x[i];
    const float norm = std::sqrt(norm_sq + kSphereEps);
    const float scale = radius / norm;
    for (size_t i = 0; i < n; ++i) x[i] *= scale;
}

extern "C" void tc_sphere_tangent_project(const float* base, float* v, size_t n) {
    /* Subtract <base, v> / <base, base> · base. <base, base> = r² for a
     * point on S^n_r; we compute it from base to be robust to drift. */
    float bb = 0.0f, bv = 0.0f;
    for (size_t i = 0; i < n; ++i) { bb += base[i] * base[i]; bv += base[i] * v[i]; }
    if (bb < kSphereEps) return;
    const float alpha = bv / bb;
    for (size_t i = 0; i < n; ++i) v[i] -= alpha * base[i];
}

extern "C" float tc_sphere_inner_product(const float* base,
                                          const float* u, const float* v,
                                          size_t n) {
    /* Tangent inner product is just Euclidean dot on T_p S. */
    (void)base;
    return tc_sphere_inner(u, v, n);
}

extern "C" void tc_sphere_add(const float* base,
                               const float* tangent_x, const float* tangent_y,
                               float* out_tangent, size_t n) {
    for (size_t i = 0; i < n; ++i) out_tangent[i] = tangent_x[i] + tangent_y[i];
    tc_sphere_tangent_project(base, out_tangent, n);
}

extern "C" void tc_sphere_scale(const float* tangent, float s, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = s * tangent[i];
}

extern "C" void tc_sphere_exp(const float* base, const float* tangent,
                               float* point, size_t n, float radius) {
    float norm_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) norm_sq += tangent[i] * tangent[i];
    const float norm = std::sqrt(norm_sq + kSphereEps);
    const float theta = norm / radius;

    float sinc_theta, cos_theta;
    if (theta < kSmallTheta) {
        sinc_theta = 1.0f - theta * theta / 6.0f;
        cos_theta  = 1.0f - theta * theta / 2.0f;
    } else {
        sinc_theta = std::sin(theta) / theta;
        cos_theta  = std::cos(theta);
    }
    for (size_t i = 0; i < n; ++i) {
        point[i] = cos_theta * base[i] + sinc_theta * tangent[i];
    }
    tc_sphere_project(point, n, radius);
}

extern "C" void tc_sphere_log(const float* base, const float* point,
                               float* tangent, size_t n, float radius) {
    float dot = 0.0f;
    for (size_t i = 0; i < n; ++i) dot += base[i] * point[i];
    const float r2 = radius * radius;
    float cos_theta = dot / r2;
    if (cos_theta < -1.0f) cos_theta = -1.0f;
    if (cos_theta >  1.0f) cos_theta =  1.0f;
    const float theta = std::acos(cos_theta);

    if (theta < kSmallTheta || std::fabs(theta - kPi) < kSmallTheta) {
        std::memset(tangent, 0, n * sizeof(float));
        return;
    }
    const float scale = theta / std::sin(theta);
    for (size_t i = 0; i < n; ++i) {
        tangent[i] = scale * (point[i] - cos_theta * base[i]);
    }
}

extern "C" float tc_sphere_distance(const float* p, const float* q, size_t n,
                                     float radius) {
    float dot = 0.0f;
    for (size_t i = 0; i < n; ++i) dot += p[i] * q[i];
    const float r2 = radius * radius;
    float cos_theta = dot / r2;
    if (cos_theta < -1.0f) cos_theta = -1.0f;
    if (cos_theta >  1.0f) cos_theta =  1.0f;
    return radius * std::acos(cos_theta);
}

extern "C" void tc_sphere_parallel_transport(const float* from, const float* to,
                                              const float* tangent, float* out,
                                              size_t n, float radius) {
    const float r2 = radius * radius;
    float pq = 0.0f, qv = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        pq += from[i] * to[i];
        qv += to[i] * tangent[i];
    }
    const float denom = r2 + pq;
    /* Near-antipodal: transport undefined, return unchanged. */
    if (std::fabs(denom) < 1e-10f) {
        if (out != tangent) std::memcpy(out, tangent, n * sizeof(float));
        return;
    }
    const float coef = qv / denom;
    for (size_t i = 0; i < n; ++i) {
        out[i] = tangent[i] - coef * (from[i] + to[i]);
    }
    tc_sphere_tangent_project(to, out, n);
}

extern "C" void tc_sphere_slerp(const float* p, const float* q, float t,
                                 float* out, size_t n, float radius) {
    float dot = 0.0f;
    for (size_t i = 0; i < n; ++i) dot += p[i] * q[i];
    const float r2 = radius * radius;
    float cos_theta = dot / r2;
    if (cos_theta < -1.0f) cos_theta = -1.0f;
    if (cos_theta >  1.0f) cos_theta =  1.0f;
    const float theta = std::acos(cos_theta);
    if (theta < kSmallTheta) {
        for (size_t i = 0; i < n; ++i) out[i] = (1.0f - t) * p[i] + t * q[i];
        tc_sphere_project(out, n, radius);
        return;
    }
    const float sin_theta = std::sin(theta);
    const float a = std::sin((1.0f - t) * theta) / sin_theta;
    const float b = std::sin(t * theta) / sin_theta;
    for (size_t i = 0; i < n; ++i) {
        out[i] = a * p[i] + b * q[i];
    }
    tc_sphere_project(out, n, radius);
}
