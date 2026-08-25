/*
 * tensorcore — phase attention + Born-rule output (CPU reference, fp32).
 *
 * Both ops are bandwidth-bound elementwise reductions; OpenMP per-row
 * parallelism with fp32 accumulators is the right structure on CPU. Native
 * Metal/CUDA kernels are a v0.2 follow-up — the CPU reference exists so
 * the autograd wrappers + Eshkol bindings can validate against the
 * GeoRefine torch reference today.
 *
 * Phase-attention combine:
 *   score(pair) = Σ_m w_m · ip_m · cos(Δφ_m + γ_m) · exp(−λ_m · d_m)
 *
 * Born-rule output (real amplitudes form):
 *   sum[n,w] = h[n,w] + s[n,w] + e[n,w]
 *   probs[n,w] = sum[n,w]² / Σ_w sum[n,w]²
 *
 * NULL e_amp / s_amp gates the corresponding term — callers can plug
 * in a subset of manifolds during ablations.
 */

#include "tensorcore/phase_attention.h"
#include "tensorcore/tensorcore.h"
#include "../core/internal.h"

#include <cmath>
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {
constexpr float kBornEps = 1e-30f;   /* Z floor to avoid /0 on a row of zeros */
}  // namespace

extern "C" tc_status_t tc_phase_attention_combine(tc_context* ctx,
                                                   const tc_buffer* inner_products,
                                                   const tc_buffer* phase_diffs,
                                                   const tc_buffer* distances,
                                                   const tc_buffer* weights,
                                                   const tc_buffer* gammas,
                                                   const tc_buffer* lambdas,
                                                   tc_buffer* scores_out,
                                                   int N_pairs, int M) {
    if (!ctx || !inner_products || !phase_diffs || !distances ||
        !weights || !gammas || !lambdas || !scores_out ||
        N_pairs <= 0 || M <= 0) return TC_ERR_INVALID_ARG;
    const size_t pair_bytes = (size_t)N_pairs * M * sizeof(float);
    const size_t scalar_bytes = (size_t)M * sizeof(float);
    const size_t score_bytes = (size_t)N_pairs * sizeof(float);
    tc_status_t s;
    if ((s = tc_buffer_validate(ctx, inner_products, pair_bytes)) != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, phase_diffs, pair_bytes))    != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, distances, pair_bytes))      != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, weights, scalar_bytes))      != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, gammas, scalar_bytes))       != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, lambdas, scalar_bytes))      != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, scores_out, score_bytes))    != TC_OK) return s;

    void *Ip = nullptr, *Pp = nullptr, *Dp = nullptr;
    void *Wp = nullptr, *Gp = nullptr, *Lp = nullptr, *Sp = nullptr;
    tc_buffer_map((tc_buffer*)inner_products, &Ip);
    tc_buffer_map((tc_buffer*)phase_diffs, &Pp);
    tc_buffer_map((tc_buffer*)distances, &Dp);
    tc_buffer_map((tc_buffer*)weights, &Wp);
    tc_buffer_map((tc_buffer*)gammas, &Gp);
    tc_buffer_map((tc_buffer*)lambdas, &Lp);
    tc_buffer_map(scores_out, &Sp);

    const float* ip = (const float*)Ip;
    const float* pd = (const float*)Pp;
    const float* dd = (const float*)Dp;
    const float* w  = (const float*)Wp;
    const float* g  = (const float*)Gp;
    const float* l  = (const float*)Lp;
    float* out      = (float*)Sp;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (N_pairs > 1)
#endif
    for (int n = 0; n < N_pairs; ++n) {
        float acc = 0.0f;
        const float* ip_row = ip + (size_t)n * M;
        const float* pd_row = pd + (size_t)n * M;
        const float* dd_row = dd + (size_t)n * M;
        for (int m = 0; m < M; ++m) {
            const float cos_term = std::cos(pd_row[m] + g[m]);
            const float exp_term = std::exp(-l[m] * dd_row[m]);
            acc += w[m] * ip_row[m] * cos_term * exp_term;
        }
        out[n] = acc;
    }
    return TC_OK;
}

extern "C" tc_status_t tc_born_rule_output(tc_context* ctx,
                                            const tc_buffer* h_amp,
                                            const tc_buffer* s_amp,
                                            const tc_buffer* e_amp,
                                            tc_buffer* probs_out,
                                            int N, int V) {
    if (!ctx || !h_amp || !probs_out || N <= 0 || V <= 0) return TC_ERR_INVALID_ARG;
    const size_t bytes = (size_t)N * V * sizeof(float);
    tc_status_t s;
    if ((s = tc_buffer_validate(ctx, h_amp, bytes)) != TC_OK) return s;
    if (s_amp && (s = tc_buffer_validate(ctx, s_amp, bytes)) != TC_OK) return s;
    if (e_amp && (s = tc_buffer_validate(ctx, e_amp, bytes)) != TC_OK) return s;
    if ((s = tc_buffer_validate(ctx, probs_out, bytes)) != TC_OK) return s;

    void *Hp = nullptr, *Sp = nullptr, *Ep = nullptr, *Pp = nullptr;
    tc_buffer_map((tc_buffer*)h_amp, &Hp);
    if (s_amp) tc_buffer_map((tc_buffer*)s_amp, &Sp);
    if (e_amp) tc_buffer_map((tc_buffer*)e_amp, &Ep);
    tc_buffer_map(probs_out, &Pp);

    const float* H = (const float*)Hp;
    const float* S = s_amp ? (const float*)Sp : nullptr;
    const float* E = e_amp ? (const float*)Ep : nullptr;
    float* P = (float*)Pp;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (N > 1)
#endif
    for (int n = 0; n < N; ++n) {
        const float* Hr = H + (size_t)n * V;
        const float* Sr = S ? S + (size_t)n * V : nullptr;
        const float* Er = E ? E + (size_t)n * V : nullptr;
        float* Pr = P + (size_t)n * V;
        /* Pass 1: square the summed amplitudes, accumulate Z. */
        double Z = 0.0;
        for (int v = 0; v < V; ++v) {
            float a = Hr[v];
            if (Sr) a += Sr[v];
            if (Er) a += Er[v];
            const float sq = a * a;
            Pr[v] = sq;
            Z += sq;
        }
        /* Pass 2: normalize. */
        const float inv = (Z > (double)kBornEps) ? (float)(1.0 / Z) : 0.0f;
        for (int v = 0; v < V; ++v) Pr[v] *= inv;
    }
    return TC_OK;
}
