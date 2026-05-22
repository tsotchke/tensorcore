/*
 * tensorcore — CPU implementations of the transformer training kernels.
 *
 * Pointwise / row-reduction ops with OpenMP outer parallelism. Designed for
 * the portable CPU backend (TC_ENABLE_METAL=OFF) so a Linux box without a
 * GPU can still run inference + training loops at honest CPU rates.
 *
 * Numerical contract matches the Metal kernels in
 * kernels/metal/training_kernels.metal:
 *   - fp16 IO, fp32 internal accumulation
 *   - rms_scaled error ≤ 5e-3 vs fp64 reference (see docs/numerics.md)
 *   - dgamma is fp32 (matches the GPU accumulator dtype)
 *
 * Performance targets on old-donkey (88-core Xeon E5-2699 v4):
 *   - RMSnorm forward, [1024 × 4096] fp16: ~5-15 GB/s (memory-bandwidth bound)
 *   - RoPE forward: in-place, ~10 GB/s of activation bandwidth
 *   - SwiGLU: elementwise, ~10-20 GB/s
 *   - softmax: row-reduction + scaling, bandwidth bound
 *   - AdamW: 4-tensor read-modify-write, ~2-4 GB/s effective
 *
 * These are not Apple-GPU-equivalent throughputs (the M2 Ultra hits ~700 GB/s
 * for the same kernels). They ARE enough to make a CPU-only worker a useful
 * participant in a heterogeneous mesh — old-donkey holds 500 GB of optimizer
 * state and runs its own slice of training at speeds limited by memory bw,
 * not by lack of kernel coverage.
 */

#include "tensorcore/tensorcore.h"
#include "../core/internal.h"
#include "../core/cpu_float.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

constexpr float kSiluClampLo = -50.0f;
constexpr float kSiluClampHi =  50.0f;

inline float silu_scalar(float x) {
    /* x * sigmoid(x), clamped to avoid expf overflow at the tails. */
    if (x < kSiluClampLo) return 0.0f;
    if (x > kSiluClampHi) return x;
    return x / (1.0f + std::exp(-x));
}

inline float dsilu_scalar(float x) {
    /* d/dx (x sigmoid(x)) = sigmoid(x) + x sigmoid(x)(1 - sigmoid(x)). */
    const float s = 1.0f / (1.0f + std::exp(-x));
    return s + x * s * (1.0f - s);
}

inline tc_status_t validate_pointwise_buf(tc_context* ctx, const tc_buffer* b, size_t bytes) {
    return tc_buffer_validate(ctx, b, bytes);
}

}  // namespace

/* ----------------------------------------------------------------------- *
 * RMSnorm forward
 *   y[n, d] = x[n, d] * rsqrt(mean(x[n]^2) + eps) * gamma[d]
 *   rstd_out[n] = rsqrt(...)         (saved for backward)
 * ----------------------------------------------------------------------- */
extern "C" tc_status_t tc_rmsnorm_forward(tc_context* ctx,
                                           const tc_buffer* X,
                                           const tc_buffer* gamma,
                                           tc_buffer* Y,
                                           tc_buffer* rstd_out,
                                           int N, int D, float eps) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!X || !gamma || !Y || !rstd_out || N <= 0 || D <= 0) return TC_ERR_INVALID_ARG;
    const size_t nd_bytes = (size_t)N * D * sizeof(uint16_t);
    const size_t d_bytes = (size_t)D * sizeof(uint16_t);
    const size_t n_bytes_fp32 = (size_t)N * sizeof(float);
    tc_status_t s;
    if ((s = validate_pointwise_buf(ctx, X, nd_bytes)) != TC_OK) return s;
    if ((s = validate_pointwise_buf(ctx, gamma, d_bytes)) != TC_OK) return s;
    if ((s = validate_pointwise_buf(ctx, Y, nd_bytes)) != TC_OK) return s;
    if ((s = validate_pointwise_buf(ctx, rstd_out, n_bytes_fp32)) != TC_OK) return s;

    void *Xp = nullptr, *gp = nullptr, *Yp = nullptr, *rp = nullptr;
    tc_buffer_map((tc_buffer*)X, &Xp);
    tc_buffer_map((tc_buffer*)gamma, &gp);
    tc_buffer_map(Y, &Yp);
    tc_buffer_map(rstd_out, &rp);

    const uint16_t* x_data = (const uint16_t*)Xp;
    const uint16_t* g_data = (const uint16_t*)gp;
    uint16_t* y_data = (uint16_t*)Yp;
    float* rstd_data = (float*)rp;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (N > 1)
#endif
    for (int n = 0; n < N; ++n) {
        const uint16_t* xr = x_data + (size_t)n * D;
        uint16_t* yr = y_data + (size_t)n * D;
        double ss = 0.0;
        for (int d = 0; d < D; ++d) {
            const float xv = tc_cpu_f16_to_f32(xr[d]);
            ss += (double)xv * xv;
        }
        const float rstd = 1.0f / std::sqrt((float)(ss / D) + eps);
        rstd_data[n] = rstd;
        for (int d = 0; d < D; ++d) {
            const float xv = tc_cpu_f16_to_f32(xr[d]);
            const float gv = tc_cpu_f16_to_f32(g_data[d]);
            yr[d] = tc_cpu_f32_to_f16(xv * rstd * gv);
        }
    }
    return TC_OK;
}

/* ----------------------------------------------------------------------- *
 * RMSnorm backward
 *   dX, dgamma (fp32 accumulator) given X, gamma, dY, rstd
 * ----------------------------------------------------------------------- */
extern "C" tc_status_t tc_rmsnorm_backward(tc_context* ctx,
                                            const tc_buffer* X,
                                            const tc_buffer* gamma,
                                            const tc_buffer* dY,
                                            const tc_buffer* rstd,
                                            tc_buffer* dX,
                                            tc_buffer* dgamma,
                                            int N, int D) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!X || !gamma || !dY || !rstd || !dX || !dgamma || N <= 0 || D <= 0) return TC_ERR_INVALID_ARG;

    void *Xp, *gp, *dYp, *rp, *dXp, *dgp;
    tc_buffer_map((tc_buffer*)X, &Xp);
    tc_buffer_map((tc_buffer*)gamma, &gp);
    tc_buffer_map((tc_buffer*)dY, &dYp);
    tc_buffer_map((tc_buffer*)rstd, &rp);
    tc_buffer_map(dX, &dXp);
    tc_buffer_map(dgamma, &dgp);

    const uint16_t* x_data = (const uint16_t*)Xp;
    const uint16_t* g_data = (const uint16_t*)gp;
    const uint16_t* dy_data = (const uint16_t*)dYp;
    const float*    rstd_data = (const float*)rp;
    uint16_t* dx_data = (uint16_t*)dXp;
    float*    dg_data = (float*)dgp;

    /* Zero dgamma (fp32 accumulator); we'll OpenMP-reduce into it. */
    std::memset(dg_data, 0, (size_t)D * sizeof(float));

#if defined(_OPENMP)
    const int n_threads = std::max(1, omp_get_max_threads());
#else
    const int n_threads = 1;
#endif
    std::vector<float> dg_partials((size_t)n_threads * D, 0.0f);

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < N; ++n) {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        float* dg_local = dg_partials.data() + (size_t)tid * D;

        const uint16_t* xr = x_data + (size_t)n * D;
        const uint16_t* dyr = dy_data + (size_t)n * D;
        uint16_t* dxr = dx_data + (size_t)n * D;
        const float rs = rstd_data[n];

        /* dY_dot_xhat = sum_d dY[d] * gamma[d] * (X[d] * rstd) */
        double s = 0.0;
        for (int d = 0; d < D; ++d) {
            const float xv  = tc_cpu_f16_to_f32(xr[d]);
            const float dyv = tc_cpu_f16_to_f32(dyr[d]);
            const float gv  = tc_cpu_f16_to_f32(g_data[d]);
            s += (double)dyv * gv * xv * rs;
        }
        const float dot = (float)(s / D);

        for (int d = 0; d < D; ++d) {
            const float xv  = tc_cpu_f16_to_f32(xr[d]);
            const float dyv = tc_cpu_f16_to_f32(dyr[d]);
            const float gv  = tc_cpu_f16_to_f32(g_data[d]);
            const float xhat = xv * rs;
            dxr[d] = tc_cpu_f32_to_f16(rs * (gv * dyv - xhat * dot));
            dg_local[d] += dyv * xhat;   /* fp32 accumulator */
        }
    }

    /* Reduce thread-local dgamma partials. */
    for (int t = 0; t < n_threads; ++t) {
        const float* row = dg_partials.data() + (size_t)t * D;
        for (int d = 0; d < D; ++d) dg_data[d] += row[d];
    }
    return TC_OK;
}

/* ----------------------------------------------------------------------- *
 * LayerNorm forward
 *   mean[n] = (1/D) Σ x[n, d]
 *   var[n]  = (1/D) Σ (x[n, d] - mean[n])^2
 *   y[n, d] = (x[n, d] - mean[n]) / sqrt(var[n] + eps) * gamma[d] + beta[d]
 * ----------------------------------------------------------------------- */
extern "C" tc_status_t tc_layernorm_forward(tc_context* ctx,
                                             const tc_buffer* X,
                                             const tc_buffer* gamma,
                                             const tc_buffer* beta,
                                             tc_buffer* Y,
                                             tc_buffer* mean_out,
                                             tc_buffer* rstd_out,
                                             int N, int D, float eps) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!X || !gamma || !beta || !Y || !mean_out || !rstd_out || N <= 0 || D <= 0)
        return TC_ERR_INVALID_ARG;

    void *Xp, *gp, *bp, *Yp, *mp, *rp;
    tc_buffer_map((tc_buffer*)X, &Xp);
    tc_buffer_map((tc_buffer*)gamma, &gp);
    tc_buffer_map((tc_buffer*)beta, &bp);
    tc_buffer_map(Y, &Yp);
    tc_buffer_map(mean_out, &mp);
    tc_buffer_map(rstd_out, &rp);

    const uint16_t* x_data = (const uint16_t*)Xp;
    const uint16_t* g_data = (const uint16_t*)gp;
    const uint16_t* b_data = (const uint16_t*)bp;
    uint16_t* y_data = (uint16_t*)Yp;
    float* mean_data = (float*)mp;
    float* rstd_data = (float*)rp;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (N > 1)
#endif
    for (int n = 0; n < N; ++n) {
        const uint16_t* xr = x_data + (size_t)n * D;
        uint16_t* yr = y_data + (size_t)n * D;
        double s = 0.0;
        for (int d = 0; d < D; ++d) s += tc_cpu_f16_to_f32(xr[d]);
        const float mean = (float)(s / D);
        double sq = 0.0;
        for (int d = 0; d < D; ++d) {
            const float c = tc_cpu_f16_to_f32(xr[d]) - mean;
            sq += (double)c * c;
        }
        const float rstd = 1.0f / std::sqrt((float)(sq / D) + eps);
        mean_data[n] = mean;
        rstd_data[n] = rstd;
        for (int d = 0; d < D; ++d) {
            const float xv = tc_cpu_f16_to_f32(xr[d]);
            const float gv = tc_cpu_f16_to_f32(g_data[d]);
            const float bv = tc_cpu_f16_to_f32(b_data[d]);
            yr[d] = tc_cpu_f32_to_f16((xv - mean) * rstd * gv + bv);
        }
    }
    return TC_OK;
}

extern "C" tc_status_t tc_layernorm_backward(tc_context* ctx,
                                              const tc_buffer* X,
                                              const tc_buffer* gamma,
                                              const tc_buffer* dY,
                                              const tc_buffer* mean,
                                              const tc_buffer* rstd,
                                              tc_buffer* dX,
                                              int N, int D) {
    /* Standard LayerNorm backward: dX = (1/D) * rstd * (D dY' - sum(dY') - xhat sum(dY' xhat))
     * where dY' = dY * gamma; xhat = (x - mean) * rstd. */
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!X || !gamma || !dY || !mean || !rstd || !dX || N <= 0 || D <= 0) return TC_ERR_INVALID_ARG;

    void *Xp, *gp, *dYp, *mp, *rp, *dXp;
    tc_buffer_map((tc_buffer*)X, &Xp);
    tc_buffer_map((tc_buffer*)gamma, &gp);
    tc_buffer_map((tc_buffer*)dY, &dYp);
    tc_buffer_map((tc_buffer*)mean, &mp);
    tc_buffer_map((tc_buffer*)rstd, &rp);
    tc_buffer_map(dX, &dXp);

    const uint16_t* x_data = (const uint16_t*)Xp;
    const uint16_t* g_data = (const uint16_t*)gp;
    const uint16_t* dy_data = (const uint16_t*)dYp;
    const float* mean_data = (const float*)mp;
    const float* rstd_data = (const float*)rp;
    uint16_t* dx_data = (uint16_t*)dXp;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < N; ++n) {
        const uint16_t* xr = x_data + (size_t)n * D;
        const uint16_t* dyr = dy_data + (size_t)n * D;
        uint16_t* dxr = dx_data + (size_t)n * D;
        const float rs = rstd_data[n];
        const float me = mean_data[n];

        double s1 = 0.0, s2 = 0.0;
        for (int d = 0; d < D; ++d) {
            const float dyp = tc_cpu_f16_to_f32(dyr[d]) * tc_cpu_f16_to_f32(g_data[d]);
            const float xhat = (tc_cpu_f16_to_f32(xr[d]) - me) * rs;
            s1 += dyp;
            s2 += (double)dyp * xhat;
        }
        const float inv_d = 1.0f / D;
        for (int d = 0; d < D; ++d) {
            const float dyp = tc_cpu_f16_to_f32(dyr[d]) * tc_cpu_f16_to_f32(g_data[d]);
            const float xhat = (tc_cpu_f16_to_f32(xr[d]) - me) * rs;
            dxr[d] = tc_cpu_f32_to_f16(rs * (dyp - (float)s1 * inv_d - xhat * (float)s2 * inv_d));
        }
    }
    return TC_OK;
}

/* ----------------------------------------------------------------------- *
 * RoPE forward (in place on X = [B, H, S, D])
 *   For each pair (X[..., 2k], X[..., 2k+1])  k in 0..D/2
 *     X[..., 2k]   = x0 * cos - x1 * sin
 *     X[..., 2k+1] = x0 * sin + x1 * cos
 *   cos_t / sin_t are precomputed [S, D/2] fp32 tables.
 *
 *   Convention: "half-rotation grouping" (Llama / Mistral); pair index k
 *   uses indices k and k + D/2, not 2k and 2k+1. The Metal kernel does
 *   half-rotation too; we match.
 * ----------------------------------------------------------------------- */
extern "C" tc_status_t tc_rope_forward(tc_context* ctx,
                                        tc_buffer* X,
                                        const tc_buffer* cos_t,
                                        const tc_buffer* sin_t,
                                        int batch, int heads, int seq, int head_dim) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!X || !cos_t || !sin_t || batch <= 0 || heads <= 0 || seq <= 0 || head_dim <= 0
        || head_dim % 2 != 0) return TC_ERR_INVALID_ARG;

    void *Xp, *cp, *sp;
    tc_buffer_map(X, &Xp);
    tc_buffer_map((tc_buffer*)cos_t, &cp);
    tc_buffer_map((tc_buffer*)sin_t, &sp);

    uint16_t* x_data = (uint16_t*)Xp;
    const float* cos_data = (const float*)cp;
    const float* sin_data = (const float*)sp;
    const int half = head_dim / 2;

    const long total = (long)batch * heads * seq;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (total > 1)
#endif
    for (long bhs = 0; bhs < total; ++bhs) {
        const int s_idx = (int)(bhs % seq);
        uint16_t* xr = x_data + (size_t)bhs * head_dim;
        const float* crow = cos_data + (size_t)s_idx * half;
        const float* srow = sin_data + (size_t)s_idx * half;
        for (int k = 0; k < half; ++k) {
            const float x0 = tc_cpu_f16_to_f32(xr[k]);
            const float x1 = tc_cpu_f16_to_f32(xr[k + half]);
            const float c = crow[k], si = srow[k];
            xr[k]        = tc_cpu_f32_to_f16(x0 * c - x1 * si);
            xr[k + half] = tc_cpu_f32_to_f16(x0 * si + x1 * c);
        }
    }
    return TC_OK;
}

/* ----------------------------------------------------------------------- *
 * SwiGLU forward
 *   out[i] = silu(gate[i]) * up[i]
 * ----------------------------------------------------------------------- */
extern "C" tc_status_t tc_swiglu_forward(tc_context* ctx,
                                          const tc_buffer* gate,
                                          const tc_buffer* up,
                                          tc_buffer* out,
                                          int n) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!gate || !up || !out || n <= 0) return TC_ERR_INVALID_ARG;
    void *gp, *up_p, *op;
    tc_buffer_map((tc_buffer*)gate, &gp);
    tc_buffer_map((tc_buffer*)up, &up_p);
    tc_buffer_map(out, &op);
    const uint16_t* gd = (const uint16_t*)gp;
    const uint16_t* ud = (const uint16_t*)up_p;
    uint16_t* od = (uint16_t*)op;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        const float gv = tc_cpu_f16_to_f32(gd[i]);
        const float uv = tc_cpu_f16_to_f32(ud[i]);
        od[i] = tc_cpu_f32_to_f16(silu_scalar(gv) * uv);
    }
    return TC_OK;
}

extern "C" tc_status_t tc_swiglu_backward(tc_context* ctx,
                                           const tc_buffer* gate,
                                           const tc_buffer* up,
                                           const tc_buffer* dout,
                                           tc_buffer* dgate,
                                           tc_buffer* dup,
                                           int n) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!gate || !up || !dout || !dgate || !dup || n <= 0) return TC_ERR_INVALID_ARG;
    void *gp, *up_p, *dop, *dgp, *dup_p;
    tc_buffer_map((tc_buffer*)gate, &gp);
    tc_buffer_map((tc_buffer*)up, &up_p);
    tc_buffer_map((tc_buffer*)dout, &dop);
    tc_buffer_map(dgate, &dgp);
    tc_buffer_map(dup, &dup_p);
    const uint16_t* gd = (const uint16_t*)gp;
    const uint16_t* ud = (const uint16_t*)up_p;
    const uint16_t* dod = (const uint16_t*)dop;
    uint16_t* dgd = (uint16_t*)dgp;
    uint16_t* dud = (uint16_t*)dup_p;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        const float gv = tc_cpu_f16_to_f32(gd[i]);
        const float uv = tc_cpu_f16_to_f32(ud[i]);
        const float dv = tc_cpu_f16_to_f32(dod[i]);
        dgd[i] = tc_cpu_f32_to_f16(dv * uv * dsilu_scalar(gv));
        dud[i] = tc_cpu_f32_to_f16(dv * silu_scalar(gv));
    }
    return TC_OK;
}

/* ----------------------------------------------------------------------- *
 * Standalone softmax (row-wise, fp16, numerically stable)
 * ----------------------------------------------------------------------- */
extern "C" tc_status_t tc_softmax_forward(tc_context* ctx,
                                           const tc_buffer* X,
                                           tc_buffer* Y,
                                           int N, int D) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!X || !Y || N <= 0 || D <= 0) return TC_ERR_INVALID_ARG;
    void *Xp, *Yp;
    tc_buffer_map((tc_buffer*)X, &Xp);
    tc_buffer_map(Y, &Yp);
    const uint16_t* xd = (const uint16_t*)Xp;
    uint16_t* yd = (uint16_t*)Yp;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < N; ++n) {
        const uint16_t* xr = xd + (size_t)n * D;
        uint16_t* yr = yd + (size_t)n * D;
        float mx = -INFINITY;
        for (int d = 0; d < D; ++d) {
            const float v = tc_cpu_f16_to_f32(xr[d]);
            if (v > mx) mx = v;
        }
        double s = 0.0;
        std::vector<float> tmp(D);
        for (int d = 0; d < D; ++d) {
            const float e = std::exp(tc_cpu_f16_to_f32(xr[d]) - mx);
            tmp[d] = e;
            s += e;
        }
        const float inv = 1.0f / (float)s;
        for (int d = 0; d < D; ++d) yr[d] = tc_cpu_f32_to_f16(tmp[d] * inv);
    }
    return TC_OK;
}

extern "C" tc_status_t tc_softmax_backward(tc_context* ctx,
                                            const tc_buffer* Y,
                                            const tc_buffer* dY,
                                            tc_buffer* dX,
                                            int N, int D) {
    /* dx_d = y_d * (dy_d - sum_k(y_k dy_k)) */
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!Y || !dY || !dX || N <= 0 || D <= 0) return TC_ERR_INVALID_ARG;
    void *Yp, *dYp, *dXp;
    tc_buffer_map((tc_buffer*)Y, &Yp);
    tc_buffer_map((tc_buffer*)dY, &dYp);
    tc_buffer_map(dX, &dXp);
    const uint16_t* yd = (const uint16_t*)Yp;
    const uint16_t* dyd = (const uint16_t*)dYp;
    uint16_t* dxd = (uint16_t*)dXp;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < N; ++n) {
        const uint16_t* yr = yd + (size_t)n * D;
        const uint16_t* dyr = dyd + (size_t)n * D;
        uint16_t* dxr = dxd + (size_t)n * D;
        double s = 0.0;
        for (int d = 0; d < D; ++d) {
            const float yv = tc_cpu_f16_to_f32(yr[d]);
            const float dv = tc_cpu_f16_to_f32(dyr[d]);
            s += (double)yv * dv;
        }
        for (int d = 0; d < D; ++d) {
            const float yv = tc_cpu_f16_to_f32(yr[d]);
            const float dv = tc_cpu_f16_to_f32(dyr[d]);
            dxr[d] = tc_cpu_f32_to_f16(yv * (dv - (float)s));
        }
    }
    return TC_OK;
}

/* ----------------------------------------------------------------------- *
 * Fused AdamW step
 *   m_t = β1 m_{t-1} + (1 - β1) g
 *   v_t = β2 v_{t-1} + (1 - β2) g^2
 *   m̂   = m / bc1                            (host pre-computes bc1 = 1 - β1^t)
 *   v̂   = v / bc2
 *   θ   = θ - lr * (m̂ / (sqrt(v̂) + eps) + wd * θ)
 * ----------------------------------------------------------------------- */
extern "C" tc_status_t tc_adamw_step(tc_context* ctx,
                                      tc_buffer* params_fp32,
                                      tc_buffer* m_fp32,
                                      tc_buffer* v_fp32,
                                      const tc_buffer* grads,
                                      tc_dtype_t grad_dtype,
                                      int n,
                                      float lr, float beta1, float beta2, float eps,
                                      float wd, float bc1, float bc2) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!params_fp32 || !m_fp32 || !v_fp32 || !grads || n <= 0) return TC_ERR_INVALID_ARG;
    void *pp, *mp, *vp, *gp;
    tc_buffer_map(params_fp32, &pp);
    tc_buffer_map(m_fp32, &mp);
    tc_buffer_map(v_fp32, &vp);
    tc_buffer_map((tc_buffer*)grads, &gp);
    float* p = (float*)pp;
    float* m = (float*)mp;
    float* v = (float*)vp;

    /* Two grad-dtype paths: fp16 (the mixed-precision case) and fp32. */
    if (grad_dtype == TC_DTYPE_F32) {
        const float* g = (const float*)gp;
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n; ++i) {
            const float gi = g[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * gi;
            v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;
            const float mhat = m[i] / bc1;
            const float vhat = v[i] / bc2;
            p[i] = p[i] - lr * (mhat / (std::sqrt(vhat) + eps) + wd * p[i]);
        }
    } else if (grad_dtype == TC_DTYPE_F16) {
        const uint16_t* g = (const uint16_t*)gp;
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n; ++i) {
            const float gi = tc_cpu_f16_to_f32(g[i]);
            m[i] = beta1 * m[i] + (1.0f - beta1) * gi;
            v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;
            const float mhat = m[i] / bc1;
            const float vhat = v[i] / bc2;
            p[i] = p[i] - lr * (mhat / (std::sqrt(vhat) + eps) + wd * p[i]);
        }
    } else {
        return TC_ERR_UNSUPPORTED_DTYPE;
    }
    return TC_OK;
}

/* ----------------------------------------------------------------------- *
 * Fused RMSnorm + GEMV (inference primitive)
 *   Y[m, n] = RMSnorm(X[m], γ) @ W[k, n]      M ≤ 4
 * For the CPU path we just call tc_rmsnorm_forward + tc_gemm separately;
 * the perf win of the fused kernel comes from avoiding the intermediate
 * write on the GPU, which is irrelevant on CPU where everything is in RAM.
 * ----------------------------------------------------------------------- */
extern "C" tc_status_t tc_fused_rmsnorm_gemv(tc_context* ctx,
                                              const tc_buffer* X,
                                              const tc_buffer* gamma,
                                              const tc_buffer* W,
                                              tc_buffer* Y,
                                              int M, int N, int K, float eps) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!X || !gamma || !W || !Y || M <= 0 || N <= 0 || K <= 0) return TC_ERR_INVALID_ARG;

    /* Allocate scratch for X_norm and rstd; reuse via thread-local pool. */
    static thread_local std::vector<uint16_t> tls_xnorm;
    static thread_local std::vector<float> tls_rstd;
    if ((int)tls_xnorm.size() < M * K) tls_xnorm.resize((size_t)M * K);
    if ((int)tls_rstd.size() < M) tls_rstd.resize((size_t)M);

    /* Run the norm directly on host pointers to avoid an extra buffer alloc. */
    void *Xp, *gp, *Wp, *Yp;
    tc_buffer_map((tc_buffer*)X, &Xp);
    tc_buffer_map((tc_buffer*)gamma, &gp);
    tc_buffer_map((tc_buffer*)W, &Wp);
    tc_buffer_map(Y, &Yp);

    const uint16_t* x_data = (const uint16_t*)Xp;
    const uint16_t* g_data = (const uint16_t*)gp;
    uint16_t* xn = tls_xnorm.data();

    for (int m = 0; m < M; ++m) {
        const uint16_t* xr = x_data + (size_t)m * K;
        uint16_t* nr = xn + (size_t)m * K;
        double ss = 0.0;
        for (int k = 0; k < K; ++k) {
            const float xv = tc_cpu_f16_to_f32(xr[k]);
            ss += (double)xv * xv;
        }
        const float rstd = 1.0f / std::sqrt((float)(ss / K) + eps);
        for (int k = 0; k < K; ++k) {
            nr[k] = tc_cpu_f32_to_f16(tc_cpu_f16_to_f32(xr[k]) * rstd * tc_cpu_f16_to_f32(g_data[k]));
        }
    }

    /* Now do the GEMV by calling our GEMM with M small. */
    tc_gemm_desc d = {};
    d.M = M; d.N = N; d.K = K;
    d.a_dtype = TC_DTYPE_F16; d.b_dtype = TC_DTYPE_F16; d.c_dtype = TC_DTYPE_F16;
    d.accum_dtype = TC_DTYPE_F32;
    d.alpha = 1.0f; d.beta = 0.0f;
    /* We need to call into tc_gemm but our X is local; create a tc_buffer
     * around it via a temporary alloc. Simplest path: alloc, memcpy, call,
     * memcpy out. Acceptable for the inference path. */
    tc_buffer* xn_buf = nullptr;
    tc_buffer_alloc(ctx, (size_t)M * K * sizeof(uint16_t), &xn_buf);
    void* xn_buf_p = nullptr;
    tc_buffer_map(xn_buf, &xn_buf_p);
    std::memcpy(xn_buf_p, xn, (size_t)M * K * sizeof(uint16_t));
    tc_status_t s = tc_gemm(ctx, &d, xn_buf, W, Y);
    tc_buffer_free(ctx, xn_buf);
    return s;
}
