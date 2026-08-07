/*
 * tensorcore — RiemannianAdam (CPU reference, fp32).
 *
 * Fused per-row updates for the Poincaré ball, unit sphere, and Euclidean
 * components. OpenMP outer parallelism over N parameter rows; each row's
 * inner loop is a small D-dim pointwise pass + a Möbius / sphere closed
 * form for the retraction + transport.
 *
 * The C reference matches Bécigneul & Ganea 2018 ("Riemannian Adaptive
 * Optimization Methods") and the implementation in GeoRefine's
 * `geometric_lm.py` (which is the diff-test target). Numerical safety
 * matches the Poincaré primitives: norm floor 1e-15, atanh clip 1e-7,
 * ball margin 1e-5 (boundary projection enforced after the retraction).
 */

#include "tensorcore/riemannian_adam.h"
#include "tensorcore/tensorcore.h"
#include "../core/internal.h"

#include <cmath>
#include <cstring>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(TC_ENABLE_CUDA)
/* Fused manifold kernels in lib/cuda/riemannian_adam.cu. Hidden visibility;
 * called from the public entry points below when CUDA is active and the
 * buffers are CUDA-managed. Return: 0 done, 1 unsupported, -1 failed. */
extern "C" int tc_cuda_is_active(void);
extern "C" int tc_cuda_riemannian_adam_step_poincare(
        void* params, const void* grads, void* m, void* v,
        int N, int D, float c, float lr, float beta1, float beta2,
        float eps, float weight_decay, float bc1, float bc2);
extern "C" int tc_cuda_riemannian_adam_step_sphere(
        void* params, const void* grads, void* m, void* v,
        int N, int D, float lr, float beta1, float beta2,
        float eps, float weight_decay, float bc1, float bc2);
extern "C" int tc_cuda_riemannian_adam_step_euclidean(
        void* params, const void* grads, void* m, void* v,
        int N, int D, float lr, float beta1, float beta2,
        float eps, float weight_decay, float bc1, float bc2);
#endif

namespace {
constexpr float kEpsNorm  = 1e-15f;
constexpr float kEpsAtanh = 1e-7f;
constexpr float kMargin   = 1e-5f;

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline float sq_norm(const float* x, int D) {
    float s = 0.0f;
    for (int d = 0; d < D; ++d) s += x[d] * x[d];
    return s;
}

inline float safe_norm(const float* x, int D) {
    const float ns = sq_norm(x, D);
    return ns > kEpsNorm * kEpsNorm ? std::sqrt(ns) : kEpsNorm;
}

inline void mobius_add_row(const float* x, const float* y, float* out,
                            int D, float c) {
    float x2 = 0.0f, y2 = 0.0f, xy = 0.0f;
    for (int d = 0; d < D; ++d) {
        x2 += x[d] * x[d]; y2 += y[d] * y[d]; xy += x[d] * y[d];
    }
    const float a = 1.0f + 2.0f * c * xy + c * y2;
    const float b = 1.0f - c * x2;
    float denom = 1.0f + 2.0f * c * xy + c * c * x2 * y2;
    if (denom < kEpsNorm) denom = kEpsNorm;
    const float inv = 1.0f / denom;
    for (int d = 0; d < D; ++d) out[d] = (a * x[d] + b * y[d]) * inv;
}

inline void project_ball_row(float* x, int D, float sqrt_c) {
    if (sqrt_c == 0.0f) return;
    const float n = safe_norm(x, D);
    const float max_n = (1.0f - kMargin) / sqrt_c;
    if (n > max_n) {
        const float s = max_n / n;
        for (int d = 0; d < D; ++d) x[d] *= s;
    }
}

tc_status_t validate_quad(tc_context* ctx, tc_buffer* params,
                          const tc_buffer* grads, tc_buffer* m, tc_buffer* v,
                          int N, int D) {
    if (!ctx || !params || !grads || !m || !v || N <= 0 || D <= 0)
        return TC_ERR_INVALID_ARG;
    const size_t bytes = (size_t)N * D * sizeof(float);
    tc_status_t s;
    if ((s = tc_buffer_validate(ctx, params, bytes)) != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, grads,  bytes)) != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, m,      bytes)) != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, v,      bytes)) != TC_OK) return s;
    return TC_OK;
}
}  // namespace

extern "C" tc_status_t tc_riemannian_adam_step_poincare_reference(
        tc_context* ctx,
        tc_buffer* params, const tc_buffer* grads,
        tc_buffer* m, tc_buffer* v,
        int N, int D, float c,
        float lr, float beta1, float beta2,
        float eps, float weight_decay,
        float bias_correction1, float bias_correction2) {
    tc_status_t s = validate_quad(ctx, params, grads, m, v, N, D);
    if (s != TC_OK) return s;
    void *Pp = nullptr, *Gp = nullptr, *Mp = nullptr, *Vp = nullptr;
    tc_buffer_map(params, &Pp); tc_buffer_map((tc_buffer*)grads, &Gp);
    tc_buffer_map(m, &Mp);      tc_buffer_map(v, &Vp);
    float* P = (float*)Pp; const float* G = (const float*)Gp;
    float* M = (float*)Mp; float* V = (float*)Vp;
    const float sqrt_c = std::sqrt(c);
    std::vector<float> dynamic_scratch;
    if (D > 64) {
        try {
            dynamic_scratch.resize((size_t)N * D * 3);
        } catch (...) {
            return TC_ERR_ALLOC;
        }
    }

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (N > 1)
#endif
    for (int n = 0; n < N; ++n) {
        float* x = P + (size_t)n * D;
        const float* g = G + (size_t)n * D;
        float* mr = M + (size_t)n * D;
        float* vr = V + (size_t)n * D;

        /* Conformal factor at x. */
        const float x2 = sq_norm(x, D);
        float lam_denom = 1.0f - c * x2;
        if (lam_denom < kEpsNorm) lam_denom = kEpsNorm;
        const float lam_x = 2.0f / lam_denom;
        /* Tangent-space projection of Euclidean gradient: g_t = g / λ_x². */
        const float inv_lam_sq = 1.0f / (lam_x * lam_x);

        /* Per-component Adam moments + step direction. Use small stack
         * scratch for typical transformer head dimensions and one bounded
         * heap allocation for the full operation when D is larger. */
        float step_stack[64];
        const bool use_stack = D <= 64;
        float* row_scratch = use_stack ? nullptr
                                       : dynamic_scratch.data() + (size_t)n * D * 3;
        float* step = use_stack ? step_stack : row_scratch;

        for (int d = 0; d < D; ++d) {
            const float g_t = g[d] * inv_lam_sq;
            const float new_m = beta1 * mr[d] + (1.0f - beta1) * g_t;
            const float new_v = beta2 * vr[d] + (1.0f - beta2) * g_t * g_t;
            mr[d] = new_m;
            vr[d] = new_v;
            const float m_hat = new_m / bias_correction1;
            const float v_hat = new_v / bias_correction2;
            float update = m_hat / (std::sqrt(v_hat) + eps);
            /* AdamW-style decoupled decay (acts on params in tangent
             * frame; matches geometric_lm.py). */
            update += weight_decay * x[d];
            step[d] = -lr * update;
        }

        /* Retract: x_new = x ⊕_c (tanh(√c·λ_x·‖step‖/2)·step/(√c·‖step‖)). */
        const float nv = safe_norm(step, D);
        const float sc_n = sqrt_c * nv;
        const float second = (sc_n > kEpsNorm)
            ? (std::tanh(sc_n * lam_x * 0.5f) / sc_n)
            : (lam_x * 0.5f);
        float scaled_stack[64];
        float* scaled = use_stack ? scaled_stack : row_scratch + D;
        for (int d = 0; d < D; ++d) scaled[d] = second * step[d];

        float x_new_stack[64];
        float* x_new = use_stack ? x_new_stack : row_scratch + 2 * D;
        mobius_add_row(x, scaled, x_new, D, c);
        project_ball_row(x_new, D, sqrt_c);

        /* Parallel transport m: PT^c_{x→x_new}(m) = (λ_x/λ_{x_new})·m. */
        const float xn2 = sq_norm(x_new, D);
        float lam_new_denom = 1.0f - c * xn2;
        if (lam_new_denom < kEpsNorm) lam_new_denom = kEpsNorm;
        const float lam_new = 2.0f / lam_new_denom;
        const float ratio = lam_x / lam_new;
        for (int d = 0; d < D; ++d) mr[d] *= ratio;

        /* Commit the new point. */
        for (int d = 0; d < D; ++d) x[d] = x_new[d];
    }
    return TC_OK;
}

extern "C" tc_status_t tc_riemannian_adam_step_sphere_reference(
        tc_context* ctx,
        tc_buffer* params, const tc_buffer* grads,
        tc_buffer* m, tc_buffer* v,
        int N, int D,
        float lr, float beta1, float beta2,
        float eps, float weight_decay,
        float bias_correction1, float bias_correction2) {
    tc_status_t s = validate_quad(ctx, params, grads, m, v, N, D);
    if (s != TC_OK) return s;
    void *Pp = nullptr, *Gp = nullptr, *Mp = nullptr, *Vp = nullptr;
    tc_buffer_map(params, &Pp); tc_buffer_map((tc_buffer*)grads, &Gp);
    tc_buffer_map(m, &Mp);      tc_buffer_map(v, &Vp);
    float* P = (float*)Pp; const float* G = (const float*)Gp;
    float* M = (float*)Mp; float* V = (float*)Vp;
    std::vector<float> dynamic_scratch;
    if (D > 64) {
        try {
            dynamic_scratch.resize((size_t)N * D * 2);
        } catch (...) {
            return TC_ERR_ALLOC;
        }
    }

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (N > 1)
#endif
    for (int n = 0; n < N; ++n) {
        float* x = P + (size_t)n * D;
        const float* g = G + (size_t)n * D;
        float* mr = M + (size_t)n * D;
        float* vr = V + (size_t)n * D;

        /* Tangent projection: g_t = g − ⟨g, x⟩·x. */
        float gx = 0.0f;
        for (int d = 0; d < D; ++d) gx += g[d] * x[d];

        float step_stack[64];
        const bool use_stack = D <= 64;
        float* row_scratch = use_stack ? nullptr
                                       : dynamic_scratch.data() + (size_t)n * D * 2;
        float* step = use_stack ? step_stack : row_scratch;

        for (int d = 0; d < D; ++d) {
            const float g_t = g[d] - gx * x[d];
            const float new_m = beta1 * mr[d] + (1.0f - beta1) * g_t;
            const float new_v = beta2 * vr[d] + (1.0f - beta2) * g_t * g_t;
            mr[d] = new_m;
            vr[d] = new_v;
            const float m_hat = new_m / bias_correction1;
            const float v_hat = new_v / bias_correction2;
            float update = m_hat / (std::sqrt(v_hat) + eps);
            update += weight_decay * x[d];
            step[d] = -lr * update;
        }
        /* Re-project step onto T_x (decoupled decay introduces a normal
         * component; subtract it). */
        float step_x = 0.0f;
        for (int d = 0; d < D; ++d) step_x += step[d] * x[d];
        for (int d = 0; d < D; ++d) step[d] -= step_x * x[d];

        /* Sphere exp: x_new = cos(‖step‖)·x + sin(‖step‖)·step/‖step‖. */
        const float n_s = safe_norm(step, D);
        const float c_v = std::cos(n_s);
        const float s_v = std::sin(n_s);
        float x_new_stack[64];
        float* x_new = use_stack ? x_new_stack : row_scratch + D;
        for (int d = 0; d < D; ++d) x_new[d] = c_v * x[d] + s_v * step[d] / n_s;

        /* Re-normalize x_new to the sphere (floating-point drift safety). */
        const float nx = safe_norm(x_new, D);
        for (int d = 0; d < D; ++d) x_new[d] /= nx;

        /* PT_{x→x_new}(m): m − ⟨m, x_new⟩/(1 + ⟨x, x_new⟩)·(x + x_new). */
        float m_xn = 0.0f, x_xn = 0.0f;
        for (int d = 0; d < D; ++d) {
            m_xn += mr[d] * x_new[d];
            x_xn += x[d] * x_new[d];
        }
        x_xn = clampf(x_xn, -1.0f + kEpsAtanh, 1.0f - kEpsAtanh);
        const float coef = m_xn / (1.0f + x_xn);
        for (int d = 0; d < D; ++d) mr[d] -= coef * (x[d] + x_new[d]);

        /* Commit new point. */
        for (int d = 0; d < D; ++d) x[d] = x_new[d];
    }
    return TC_OK;
}

extern "C" tc_status_t tc_riemannian_adam_step_euclidean_reference(
        tc_context* ctx,
        tc_buffer* params, const tc_buffer* grads,
        tc_buffer* m, tc_buffer* v,
        int N, int D,
        float lr, float beta1, float beta2,
        float eps, float weight_decay,
        float bias_correction1, float bias_correction2) {
    tc_status_t s = validate_quad(ctx, params, grads, m, v, N, D);
    if (s != TC_OK) return s;
    void *Pp = nullptr, *Gp = nullptr, *Mp = nullptr, *Vp = nullptr;
    tc_buffer_map(params, &Pp); tc_buffer_map((tc_buffer*)grads, &Gp);
    tc_buffer_map(m, &Mp);      tc_buffer_map(v, &Vp);
    float* P = (float*)Pp; const float* G = (const float*)Gp;
    float* M = (float*)Mp; float* V = (float*)Vp;
    const int total = N * D;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (total > 1024)
#endif
    for (int i = 0; i < total; ++i) {
        const float g = G[i];
        const float new_m = beta1 * M[i] + (1.0f - beta1) * g;
        const float new_v = beta2 * V[i] + (1.0f - beta2) * g * g;
        M[i] = new_m; V[i] = new_v;
        const float m_hat = new_m / bias_correction1;
        const float v_hat = new_v / bias_correction2;
        const float update = m_hat / (std::sqrt(v_hat) + eps);
        P[i] = (P[i] - lr * weight_decay * P[i]) - lr * update;
    }
    return TC_OK;
}

/* ----------------------------------------------------------------------- *
 * Public entry points: CUDA when the buffers are managed, otherwise the
 * reference above.
 *
 * The CUDA kernels take raw pointers, and a tc_buffer on a CUDA build is
 * cudaMallocManaged memory (lib/cuda/buffer.cpp), so tc_buffer_map hands
 * back a pointer the device can dereference directly — no staging copy, and
 * the same pointer the CPU reference would have used. Validation happens
 * once, here, so both paths reject the same bad arguments.
 * ----------------------------------------------------------------------- */

#if defined(TC_ENABLE_CUDA)
namespace {
/* Map all four buffers and report whether the CUDA path should be tried. */
bool cuda_ready(tc_buffer* params, const tc_buffer* grads,
                tc_buffer* m, tc_buffer* v,
                void** pp, void** gp, void** mp, void** vp) {
    tc_buffer_map(params, pp);
    tc_buffer_map((tc_buffer*)grads, gp);
    tc_buffer_map(m, mp);
    tc_buffer_map(v, vp);
    return tc_cuda_is_active() != 0;
}
}  // namespace
#endif

extern "C" tc_status_t tc_riemannian_adam_step_poincare(
        tc_context* ctx,
        tc_buffer* params, const tc_buffer* grads,
        tc_buffer* m, tc_buffer* v,
        int N, int D, float c,
        float lr, float beta1, float beta2,
        float eps, float weight_decay,
        float bias_correction1, float bias_correction2) {
    tc_status_t s = validate_quad(ctx, params, grads, m, v, N, D);
    if (s != TC_OK) return s;
#if defined(TC_ENABLE_CUDA)
    void *pp = nullptr, *gp = nullptr, *mp = nullptr, *vp = nullptr;
    if (cuda_ready(params, grads, m, v, &pp, &gp, &mp, &vp)) {
        const int rc = tc_cuda_riemannian_adam_step_poincare(
            pp, gp, mp, vp, N, D, c, lr, beta1, beta2, eps, weight_decay,
            bias_correction1, bias_correction2);
        if (rc == 0) {
            return tc_record_dispatch("tc_riemannian_adam_step_poincare",
                                      TC_BACKEND_CUDA, TC_OK);
        }
        if (rc < 0) return TC_ERR_INTERNAL;
    }
#endif
    s = tc_riemannian_adam_step_poincare_reference(
        ctx, params, grads, m, v, N, D, c, lr, beta1, beta2, eps,
        weight_decay, bias_correction1, bias_correction2);
    if (s != TC_OK) return s;
    return tc_record_dispatch("tc_riemannian_adam_step_poincare",
                              TC_BACKEND_PORTABLE_CPU, TC_OK);
}

extern "C" tc_status_t tc_riemannian_adam_step_sphere(
        tc_context* ctx,
        tc_buffer* params, const tc_buffer* grads,
        tc_buffer* m, tc_buffer* v,
        int N, int D,
        float lr, float beta1, float beta2,
        float eps, float weight_decay,
        float bias_correction1, float bias_correction2) {
    tc_status_t s = validate_quad(ctx, params, grads, m, v, N, D);
    if (s != TC_OK) return s;
#if defined(TC_ENABLE_CUDA)
    void *pp = nullptr, *gp = nullptr, *mp = nullptr, *vp = nullptr;
    if (cuda_ready(params, grads, m, v, &pp, &gp, &mp, &vp)) {
        const int rc = tc_cuda_riemannian_adam_step_sphere(
            pp, gp, mp, vp, N, D, lr, beta1, beta2, eps, weight_decay,
            bias_correction1, bias_correction2);
        if (rc == 0) {
            return tc_record_dispatch("tc_riemannian_adam_step_sphere",
                                      TC_BACKEND_CUDA, TC_OK);
        }
        if (rc < 0) return TC_ERR_INTERNAL;
    }
#endif
    s = tc_riemannian_adam_step_sphere_reference(
        ctx, params, grads, m, v, N, D, lr, beta1, beta2, eps,
        weight_decay, bias_correction1, bias_correction2);
    if (s != TC_OK) return s;
    return tc_record_dispatch("tc_riemannian_adam_step_sphere",
                              TC_BACKEND_PORTABLE_CPU, TC_OK);
}

extern "C" tc_status_t tc_riemannian_adam_step_euclidean(
        tc_context* ctx,
        tc_buffer* params, const tc_buffer* grads,
        tc_buffer* m, tc_buffer* v,
        int N, int D,
        float lr, float beta1, float beta2,
        float eps, float weight_decay,
        float bias_correction1, float bias_correction2) {
    tc_status_t s = validate_quad(ctx, params, grads, m, v, N, D);
    if (s != TC_OK) return s;
#if defined(TC_ENABLE_CUDA)
    void *pp = nullptr, *gp = nullptr, *mp = nullptr, *vp = nullptr;
    if (cuda_ready(params, grads, m, v, &pp, &gp, &mp, &vp)) {
        const int rc = tc_cuda_riemannian_adam_step_euclidean(
            pp, gp, mp, vp, N, D, lr, beta1, beta2, eps, weight_decay,
            bias_correction1, bias_correction2);
        if (rc == 0) {
            return tc_record_dispatch("tc_riemannian_adam_step_euclidean",
                                      TC_BACKEND_CUDA, TC_OK);
        }
        if (rc < 0) return TC_ERR_INTERNAL;
    }
#endif
    s = tc_riemannian_adam_step_euclidean_reference(
        ctx, params, grads, m, v, N, D, lr, beta1, beta2, eps,
        weight_decay, bias_correction1, bias_correction2);
    if (s != TC_OK) return s;
    return tc_record_dispatch("tc_riemannian_adam_step_euclidean",
                              TC_BACKEND_PORTABLE_CPU, TC_OK);
}
