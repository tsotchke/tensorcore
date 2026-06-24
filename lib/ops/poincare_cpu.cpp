/*
 * tensorcore — Poincaré ball ops (CPU reference, fp32).
 *
 * Canonical implementations of mobius_add / exp_map / log_map /
 * distance / parallel_transport / conformal_factor on the Poincaré ball
 * (hyperbolic curvature −c, c ≥ 0).
 *
 * These are the math fragments GeoRefine's GeometricLM (`geometric_lm.py`)
 * needs to dispatch onto tensorcore. The Eshkol bridge in
 * `eshkol/poincare.esk` exposes them as Eshkol functions; the Python
 * torch.autograd.Function wrappers in tensorcore_torch/poincare.py call
 * them via ctypes.
 *
 * Numerics: fp32 IO + fp32 accumulators throughout, OpenMP per-row
 * parallelism. Numerical safety (matching geometric_lm.py defaults):
 *   - norm floor:       1e-15
 *   - atanh clip:       (−1+1e-7, 1−1e-7)
 *   - boundary margin:  1e-5  (scale back inside ball if ‖x‖ ≥ (1/√c − margin))
 *
 * Native Metal/CUDA kernels are a v0.2 follow-up; the CPU reference is
 * shipped first so the autograd wrappers and Eshkol bindings can validate
 * end-to-end against the GeoRefine torch reference today.
 */

#include "tensorcore/poincare.h"
#include "tensorcore/tensorcore.h"
#include "../core/internal.h"

#include <cmath>
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

constexpr float kEpsNorm   = 1e-15f;
constexpr float kEpsAtanh  = 1e-7f;
constexpr float kMargin    = 1e-5f;

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline float safe_norm_squared(const float* x, int D) {
    float s = 0.0f;
    for (int d = 0; d < D; ++d) s += x[d] * x[d];
    return s;
}

inline float safe_norm(const float* x, int D) {
    const float ns = safe_norm_squared(x, D);
    return ns > kEpsNorm * kEpsNorm ? std::sqrt(ns) : kEpsNorm;
}

inline float safe_atanh(float z) {
    z = clampf(z, -1.0f + kEpsAtanh, 1.0f - kEpsAtanh);
    return std::atanh(z);
}

/* Scale a single row back inside the Poincaré ball if needed. */
inline void project_row(float* x, int D, float sqrt_c) {
    if (sqrt_c == 0.0f) return;
    const float n = safe_norm(x, D);
    const float max_n = (1.0f - kMargin) / sqrt_c;
    if (n > max_n) {
        const float s = max_n / n;
        for (int d = 0; d < D; ++d) x[d] *= s;
    }
}

inline void mobius_add_row(const float* x, const float* y, float* out,
                            int D, float c) {
    float x2 = 0.0f, y2 = 0.0f, xy = 0.0f;
    for (int d = 0; d < D; ++d) {
        x2 += x[d] * x[d];
        y2 += y[d] * y[d];
        xy += x[d] * y[d];
    }
    const float a = 1.0f + 2.0f * c * xy + c * y2;
    const float b = 1.0f - c * x2;
    float denom = 1.0f + 2.0f * c * xy + c * c * x2 * y2;
    if (denom < kEpsNorm) denom = kEpsNorm;
    const float inv = 1.0f / denom;
    for (int d = 0; d < D; ++d) {
        out[d] = (a * x[d] + b * y[d]) * inv;
    }
    project_row(out, D, std::sqrt(c));
}

tc_status_t validate_pair(tc_context* ctx, const tc_buffer* A,
                          const tc_buffer* B, tc_buffer* C,
                          int N, int D, size_t bytes_per_row) {
    if (!ctx || !A || !B || !C || N <= 0 || D <= 0) return TC_ERR_INVALID_ARG;
    const size_t bytes = (size_t)N * bytes_per_row;
    tc_status_t s;
    if ((s = tc_buffer_validate(ctx, A, bytes)) != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, B, bytes)) != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, C, bytes)) != TC_OK) return s;
    return TC_OK;
}

}  // namespace

extern "C" tc_status_t tc_poincare_mobius_add(tc_context* ctx,
                                               const tc_buffer* X,
                                               const tc_buffer* Y,
                                               tc_buffer* out,
                                               float c, int N, int D) {
    tc_status_t s = validate_pair(ctx, X, Y, out, N, D, (size_t)D * sizeof(float));
    if (s != TC_OK) return s;
    void *Xp = nullptr, *Yp = nullptr, *Op = nullptr;
    tc_buffer_map((tc_buffer*)X, &Xp);
    tc_buffer_map((tc_buffer*)Y, &Yp);
    tc_buffer_map(out, &Op);
    const float* Xd = (const float*)Xp;
    const float* Yd = (const float*)Yp;
    float* Od = (float*)Op;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (N > 1)
#endif
    for (int n = 0; n < N; ++n) {
        mobius_add_row(Xd + (size_t)n * D, Yd + (size_t)n * D,
                       Od + (size_t)n * D, D, c);
    }
    return TC_OK;
}

extern "C" tc_status_t tc_poincare_conformal_factor(tc_context* ctx,
                                                     const tc_buffer* X,
                                                     tc_buffer* lam_out,
                                                     float c, int N, int D) {
    if (!ctx || !X || !lam_out || N <= 0 || D <= 0) return TC_ERR_INVALID_ARG;
    tc_status_t s;
    if ((s = tc_buffer_validate(ctx, X, (size_t)N * D * sizeof(float))) != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, lam_out, (size_t)N * sizeof(float))) != TC_OK) return s;
    void *Xp = nullptr, *Lp = nullptr;
    tc_buffer_map((tc_buffer*)X, &Xp);
    tc_buffer_map(lam_out, &Lp);
    const float* Xd = (const float*)Xp;
    float* Ld = (float*)Lp;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (N > 1)
#endif
    for (int n = 0; n < N; ++n) {
        const float* x = Xd + (size_t)n * D;
        const float x2 = safe_norm_squared(x, D);
        float denom = 1.0f - c * x2;
        if (denom < kEpsNorm) denom = kEpsNorm;
        Ld[n] = 2.0f / denom;
    }
    return TC_OK;
}

extern "C" tc_status_t tc_poincare_exp_map_zero(tc_context* ctx,
                                                 const tc_buffer* V,
                                                 tc_buffer* out,
                                                 float c, int N, int D) {
    if (!ctx || !V || !out || N <= 0 || D <= 0) return TC_ERR_INVALID_ARG;
    const size_t bytes = (size_t)N * D * sizeof(float);
    tc_status_t s;
    if ((s = tc_buffer_validate(ctx, V, bytes)) != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, out, bytes)) != TC_OK) return s;
    void *Vp = nullptr, *Op = nullptr;
    tc_buffer_map((tc_buffer*)V, &Vp);
    tc_buffer_map(out, &Op);
    const float* Vd = (const float*)Vp;
    float* Od = (float*)Op;
    const float sqrt_c = std::sqrt(c);

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (N > 1)
#endif
    for (int n = 0; n < N; ++n) {
        const float* v = Vd + (size_t)n * D;
        float* o = Od + (size_t)n * D;
        const float nv = safe_norm(v, D);
        const float sc_n = sqrt_c * nv;
        const float coef = (sc_n > kEpsNorm) ? (std::tanh(sc_n) / sc_n) : 1.0f;
        for (int d = 0; d < D; ++d) o[d] = coef * v[d];
        project_row(o, D, sqrt_c);
    }
    return TC_OK;
}

extern "C" tc_status_t tc_poincare_log_map_zero(tc_context* ctx,
                                                 const tc_buffer* X,
                                                 tc_buffer* out,
                                                 float c, int N, int D) {
    if (!ctx || !X || !out || N <= 0 || D <= 0) return TC_ERR_INVALID_ARG;
    const size_t bytes = (size_t)N * D * sizeof(float);
    tc_status_t s;
    if ((s = tc_buffer_validate(ctx, X, bytes)) != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, out, bytes)) != TC_OK) return s;
    void *Xp = nullptr, *Op = nullptr;
    tc_buffer_map((tc_buffer*)X, &Xp);
    tc_buffer_map(out, &Op);
    const float* Xd = (const float*)Xp;
    float* Od = (float*)Op;
    const float sqrt_c = std::sqrt(c);

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (N > 1)
#endif
    for (int n = 0; n < N; ++n) {
        const float* x = Xd + (size_t)n * D;
        float* o = Od + (size_t)n * D;
        const float nx = safe_norm(x, D);
        const float sc_n = sqrt_c * nx;
        const float coef = (sc_n > kEpsNorm) ? (safe_atanh(sc_n) / sc_n) : 1.0f;
        for (int d = 0; d < D; ++d) o[d] = coef * x[d];
    }
    return TC_OK;
}

extern "C" tc_status_t tc_poincare_exp_map(tc_context* ctx,
                                            const tc_buffer* X,
                                            const tc_buffer* V,
                                            tc_buffer* out,
                                            float c, int N, int D) {
    tc_status_t s = validate_pair(ctx, X, V, out, N, D, (size_t)D * sizeof(float));
    if (s != TC_OK) return s;
    void *Xp = nullptr, *Vp = nullptr, *Op = nullptr;
    tc_buffer_map((tc_buffer*)X, &Xp);
    tc_buffer_map((tc_buffer*)V, &Vp);
    tc_buffer_map(out, &Op);
    const float* Xd = (const float*)Xp;
    const float* Vd = (const float*)Vp;
    float* Od = (float*)Op;
    const float sqrt_c = std::sqrt(c);

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (N > 1)
#endif
    for (int n = 0; n < N; ++n) {
        const float* x = Xd + (size_t)n * D;
        const float* v = Vd + (size_t)n * D;
        float* o = Od + (size_t)n * D;
        const float x2 = safe_norm_squared(x, D);
        float lam_denom = 1.0f - c * x2;
        if (lam_denom < kEpsNorm) lam_denom = kEpsNorm;
        const float lam = 2.0f / lam_denom;
        const float nv = safe_norm(v, D);
        const float sc_n = sqrt_c * nv;
        const float second = (sc_n > kEpsNorm)
            ? (std::tanh(sc_n * lam * 0.5f) / sc_n)
            : (lam * 0.5f);
        float scaled[64];   /* small per-row stack scratch; D should be <= 64 typical */
        const bool use_stack = D <= 64;
        float* tmp = use_stack ? scaled : (float*)alloca((size_t)D * sizeof(float));
        for (int d = 0; d < D; ++d) tmp[d] = second * v[d];
        mobius_add_row(x, tmp, o, D, c);
    }
    return TC_OK;
}

extern "C" tc_status_t tc_poincare_log_map(tc_context* ctx,
                                            const tc_buffer* X,
                                            const tc_buffer* Y,
                                            tc_buffer* out,
                                            float c, int N, int D) {
    tc_status_t s = validate_pair(ctx, X, Y, out, N, D, (size_t)D * sizeof(float));
    if (s != TC_OK) return s;
    void *Xp = nullptr, *Yp = nullptr, *Op = nullptr;
    tc_buffer_map((tc_buffer*)X, &Xp);
    tc_buffer_map((tc_buffer*)Y, &Yp);
    tc_buffer_map(out, &Op);
    const float* Xd = (const float*)Xp;
    const float* Yd = (const float*)Yp;
    float* Od = (float*)Op;
    const float sqrt_c = std::sqrt(c);

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (N > 1)
#endif
    for (int n = 0; n < N; ++n) {
        const float* x = Xd + (size_t)n * D;
        const float* y = Yd + (size_t)n * D;
        float* o = Od + (size_t)n * D;
        /* delta = −x ⊕_c y, into o (temporary). */
        float neg_x[64];
        const bool use_stack = D <= 64;
        float* neg_x_buf = use_stack ? neg_x : (float*)alloca((size_t)D * sizeof(float));
        for (int d = 0; d < D; ++d) neg_x_buf[d] = -x[d];
        mobius_add_row(neg_x_buf, y, o, D, c);

        const float nd = safe_norm(o, D);
        const float x2 = safe_norm_squared(x, D);
        float lam_denom = 1.0f - c * x2;
        if (lam_denom < kEpsNorm) lam_denom = kEpsNorm;
        const float lam = 2.0f / lam_denom;
        const float sc_n = sqrt_c * nd;
        float coef;
        if (sc_n > kEpsNorm) {
            coef = (2.0f / (sqrt_c < kEpsNorm ? kEpsNorm : sqrt_c * lam))
                   * safe_atanh(sc_n) / nd;
        } else {
            coef = 2.0f / lam;
        }
        for (int d = 0; d < D; ++d) o[d] *= coef;
    }
    return TC_OK;
}

extern "C" tc_status_t tc_poincare_distance(tc_context* ctx,
                                             const tc_buffer* X,
                                             const tc_buffer* Y,
                                             tc_buffer* dist_out,
                                             float c, int N, int D) {
    if (!ctx || !X || !Y || !dist_out || N <= 0 || D <= 0) return TC_ERR_INVALID_ARG;
    const size_t mat_bytes = (size_t)N * D * sizeof(float);
    const size_t vec_bytes = (size_t)N * sizeof(float);
    tc_status_t s;
    if ((s = tc_buffer_validate(ctx, X, mat_bytes)) != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, Y, mat_bytes)) != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, dist_out, vec_bytes)) != TC_OK) return s;
    void *Xp = nullptr, *Yp = nullptr, *Dp = nullptr;
    tc_buffer_map((tc_buffer*)X, &Xp);
    tc_buffer_map((tc_buffer*)Y, &Yp);
    tc_buffer_map(dist_out, &Dp);
    const float* Xd = (const float*)Xp;
    const float* Yd = (const float*)Yp;
    float* Dd = (float*)Dp;
    const float sqrt_c = std::sqrt(c);

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (N > 1)
#endif
    for (int n = 0; n < N; ++n) {
        const float* x = Xd + (size_t)n * D;
        const float* y = Yd + (size_t)n * D;
        float delta[64];
        const bool use_stack = D <= 64;
        float* d_buf = use_stack ? delta : (float*)alloca((size_t)D * sizeof(float));
        float neg_x[64];
        float* neg_x_buf = use_stack ? neg_x : (float*)alloca((size_t)D * sizeof(float));
        for (int d = 0; d < D; ++d) neg_x_buf[d] = -x[d];
        mobius_add_row(neg_x_buf, y, d_buf, D, c);
        const float nd = safe_norm(d_buf, D);
        const float sc_n = sqrt_c * nd;
        if (sc_n > kEpsNorm) {
            Dd[n] = 2.0f * safe_atanh(sc_n) / sqrt_c;
        } else {
            Dd[n] = 2.0f * nd;
        }
    }
    return TC_OK;
}

extern "C" tc_status_t tc_poincare_parallel_transport(tc_context* ctx,
                                                       const tc_buffer* V,
                                                       const tc_buffer* X,
                                                       const tc_buffer* Y,
                                                       tc_buffer* out,
                                                       float c, int N, int D) {
    if (!ctx || !V || !X || !Y || !out || N <= 0 || D <= 0) return TC_ERR_INVALID_ARG;
    const size_t bytes = (size_t)N * D * sizeof(float);
    tc_status_t s;
    if ((s = tc_buffer_validate(ctx, V, bytes)) != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, X, bytes)) != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, Y, bytes)) != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, out, bytes)) != TC_OK) return s;
    void *Vp = nullptr, *Xp = nullptr, *Yp = nullptr, *Op = nullptr;
    tc_buffer_map((tc_buffer*)V, &Vp);
    tc_buffer_map((tc_buffer*)X, &Xp);
    tc_buffer_map((tc_buffer*)Y, &Yp);
    tc_buffer_map(out, &Op);
    const float* Vd = (const float*)Vp;
    const float* Xd = (const float*)Xp;
    const float* Yd = (const float*)Yp;
    float* Od = (float*)Op;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (N > 1)
#endif
    for (int n = 0; n < N; ++n) {
        const float* x = Xd + (size_t)n * D;
        const float* y = Yd + (size_t)n * D;
        const float* v = Vd + (size_t)n * D;
        float* o = Od + (size_t)n * D;
        const float x2 = safe_norm_squared(x, D);
        const float y2 = safe_norm_squared(y, D);
        float lx_denom = 1.0f - c * x2;
        float ly_denom = 1.0f - c * y2;
        if (lx_denom < kEpsNorm) lx_denom = kEpsNorm;
        if (ly_denom < kEpsNorm) ly_denom = kEpsNorm;
        const float lam_x = 2.0f / lx_denom;
        const float lam_y = 2.0f / ly_denom;
        const float ratio = lam_x / lam_y;
        for (int d = 0; d < D; ++d) o[d] = ratio * v[d];
    }
    return TC_OK;
}
