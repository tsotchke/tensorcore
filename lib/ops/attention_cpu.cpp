/*
 * tensorcore — CPU FlashAttention-2 forward + backward.
 *
 * Memory-efficient attention on the portable CPU backend. The standard
 * attention algorithm materializes the [Sq × Sk] S = QK^T / sqrt(d) matrix
 * before softmax, which costs O(Sq * Sk) memory and bandwidth. For a
 * 4096-context attention, S alone is 64 MB per head per batch — fp16
 * inputs notwithstanding, that's prohibitive on cache-sized memory.
 *
 * FlashAttention-2 streams the attention computation in tiles of
 * (Br × Bc) output rows × key columns, never materializing S, and uses
 * the online-softmax trick for numerically-stable running max + sum
 * tracking. Memory is O(Sq * d), bandwidth is asymptotically the same
 * as materialized attention but with much better cache behavior.
 *
 * This CPU port:
 *   - Per-(B, H) outer parallelism via OpenMP.
 *   - Br=Bq=32, Bc=Bk=32 inner tile (fits cache footprint per thread).
 *   - GQA via head index mapping (Q heads → KV heads).
 *   - Causal masking via Sk_row > Sq_row + offset.
 *   - Sliding-window and ALiBi support (matches Metal kernel).
 *
 * Performance target on old-donkey (88-core Xeon, 32MB L2 per socket):
 *   - 4K context, 32 heads, dim 128: ~20-50 GFLOPS per core
 *   - Bandwidth-bound on QK^T loads, not compute-bound
 *
 * The Metal version of this kernel hits ~5 TFLOPS on M2 Ultra. The CPU
 * version is ~100× slower — that's expected and useful: old-donkey can
 * run inference + training on small models locally without needing a GPU.
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

/* Runtime-gated x86 fp16->fp32 dot product. Keep the generic CPU backend
 * portable: only the helper below is compiled for AVX2/F16C/FMA, and it is
 * called only when the host CPU reports those features. */
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#  define TC_ATTN_CAN_BUILD_X86_DOT 1
#  include <immintrin.h>
#endif

namespace {

constexpr int kBr = 32;   /* query rows per tile */
constexpr int kBc = 32;   /* key   cols per tile */

#if defined(TC_ATTN_CAN_BUILD_X86_DOT)
__attribute__((target("avx2,f16c,fma")))
float dot_fp16_x86_avx2(const uint16_t* a, const uint16_t* b, int n) {
    int i = 0;
    __m256 acc = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        const __m128i ah = _mm_loadu_si128((const __m128i*)(a + i));
        const __m128i bh = _mm_loadu_si128((const __m128i*)(b + i));
        const __m256 af = _mm256_cvtph_ps(ah);
        const __m256 bf = _mm256_cvtph_ps(bh);
        acc = _mm256_fmadd_ps(af, bf, acc);
    }
    /* Horizontal sum of the 8 lanes. */
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    float result = _mm_cvtss_f32(s1);
    for (; i < n; ++i) {
        result += tc_cpu_f16_to_f32(a[i]) * tc_cpu_f16_to_f32(b[i]);
    }
    return result;
}

bool x86_supports_avx2_dot(void) {
    static int cached = -1;
    if (cached < 0) {
#  if defined(__GNUC__) && !defined(__clang__)
        __builtin_cpu_init();
#  endif
        cached = (__builtin_cpu_supports("avx2") &&
                  __builtin_cpu_supports("f16c") &&
                  __builtin_cpu_supports("fma")) ? 1 : 0;
    }
    return cached != 0;
}
#endif

inline float dot_fp16_scalar(const uint16_t* a, const uint16_t* b, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += tc_cpu_f16_to_f32(a[i]) * tc_cpu_f16_to_f32(b[i]);
    }
    return acc;
}

inline float dot_fp16(const uint16_t* a, const uint16_t* b, int n) {
#if defined(TC_ATTN_CAN_BUILD_X86_DOT)
    if (x86_supports_avx2_dot()) {
        return dot_fp16_x86_avx2(a, b, n);
    }
#endif
    return dot_fp16_scalar(a, b, n);
}

/* Tile of attention output accumulator. Online-softmax state:
 *   m_i : running max across all Bc tiles processed so far
 *   l_i : running denominator (sum of softmax exponents)
 *   O_i : running output accumulator before final divide by l_i
 *
 * At the end of the loop over Sk, divide O_i by l_i to get the final
 * normalized output. */
struct OnlineSoftmaxState {
    float m;            /* running max */
    float l;            /* running denom */
    std::vector<float> O;  /* head_dim long, fp32 accumulator */
};

}  // namespace

extern "C" tc_status_t tc_attention_forward(tc_context* ctx,
                                             const tc_attention_desc* desc,
                                             const tc_buffer* Q,
                                             const tc_buffer* K,
                                             const tc_buffer* V,
                                             tc_buffer* O,
                                             tc_buffer* LSE) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!desc || !Q || !K || !V || !O) return TC_ERR_INVALID_ARG;
    if (desc->batch <= 0 || desc->heads <= 0 ||
        desc->seq_q <= 0 || desc->seq_kv <= 0 || desc->head_dim <= 0) return TC_ERR_INVALID_ARG;
    if (desc->io_dtype != TC_DTYPE_F16 || desc->accum_dtype != TC_DTYPE_F32) {
        return TC_ERR_UNSUPPORTED_DTYPE;
    }
    if (desc->return_lse && !LSE) return TC_ERR_INVALID_ARG;

    const int B = desc->batch;
    const int Hq = desc->heads;
    const int Hkv = desc->kv_heads > 0 ? desc->kv_heads : desc->heads;
    if (Hkv <= 0 || Hq % Hkv != 0) return TC_ERR_INVALID_ARG;
    const int Sq = desc->seq_q;
    const int Sk = desc->seq_kv;
    const int D = desc->head_dim;
    const int Hq_per_Hkv = Hq / Hkv;
    const float scale = desc->softmax_scale > 0 ? desc->softmax_scale : (1.0f / std::sqrt((float)D));
    const bool causal = desc->causal != 0;
    const int swin = desc->window_size;   /* 0 = disabled */

    void *Qp, *Kp, *Vp, *Op, *Lp = nullptr;
    tc_buffer_map((tc_buffer*)Q, &Qp);
    tc_buffer_map((tc_buffer*)K, &Kp);
    tc_buffer_map((tc_buffer*)V, &Vp);
    tc_buffer_map(O, &Op);
    if (LSE) tc_buffer_map(LSE, &Lp);

    const uint16_t* Qd = (const uint16_t*)Qp;
    const uint16_t* Kd = (const uint16_t*)Kp;
    const uint16_t* Vd = (const uint16_t*)Vp;
    uint16_t* Od = (uint16_t*)Op;
    float* Ld = (float*)Lp;

    /* Layout convention: row-major contiguous over [B, H, S, D].
     * Q index: ((b * Hq + h) * Sq + s) * D + d
     * K/V index: ((b * Hkv + hkv) * Sk + s) * D + d
     * O index: ((b * Hq + h) * Sq + s) * D + d
     * LSE index (optional): (b * Hq + h) * Sq + s
     */

    const long total = (long)B * Hq;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic) if (total > 1)
#endif
    for (long bh = 0; bh < total; ++bh) {
        const int b = (int)(bh / Hq);
        const int h = (int)(bh % Hq);
        const int hkv = h / Hq_per_Hkv;

        const uint16_t* Qbh = Qd + ((size_t)b * Hq + h) * Sq * D;
        const uint16_t* Kbh = Kd + ((size_t)b * Hkv + hkv) * Sk * D;
        const uint16_t* Vbh = Vd + ((size_t)b * Hkv + hkv) * Sk * D;
        uint16_t* Obh = Od + ((size_t)b * Hq + h) * Sq * D;
        float* Lbh = Ld ? Ld + ((size_t)b * Hq + h) * Sq : nullptr;

        /* Scratch for one Q-row tile: m, l, O (fp32). */
        std::vector<float> m_state(kBr, -INFINITY);
        std::vector<float> l_state(kBr, 0.0f);
        std::vector<float> O_state((size_t)kBr * D, 0.0f);
        std::vector<float> S_tile((size_t)kBr * kBc, 0.0f);

        for (int qb = 0; qb < Sq; qb += kBr) {
            const int br_h = std::min(kBr, Sq - qb);
            /* Reset state for this Q tile. */
            for (int r = 0; r < br_h; ++r) {
                m_state[r] = -INFINITY;
                l_state[r] = 0.0f;
            }
            std::memset(O_state.data(), 0, (size_t)br_h * D * sizeof(float));

            for (int kb = 0; kb < Sk; kb += kBc) {
                const int bc_w = std::min(kBc, Sk - kb);

                /* Compute S[r][c] = scale * dot(Q[qb+r], K[kb+c]) for the tile. */
                for (int r = 0; r < br_h; ++r) {
                    const uint16_t* Qrow = Qbh + (size_t)(qb + r) * D;
                    for (int c = 0; c < bc_w; ++c) {
                        const uint16_t* Krow = Kbh + (size_t)(kb + c) * D;
                        float s = dot_fp16(Qrow, Krow, D) * scale;

                        /* Causal mask: future tokens get -inf. */
                        if (causal && (kb + c) > (qb + r)) s = -INFINITY;

                        /* Sliding window: too-old tokens get -inf. */
                        if (swin > 0 && (qb + r) - (kb + c) > swin) s = -INFINITY;

                        /* ALiBi: linear bias (slope * relative distance). */
                        if (desc->alibi_slopes) {
                            const float slope = desc->alibi_slopes[h];
                            s -= slope * (float)((qb + r) - (kb + c));
                        }
                        S_tile[r * kBc + c] = s;
                    }
                }

                /* Online softmax: update m, l, O for each Q row. */
                for (int r = 0; r < br_h; ++r) {
                    float m_new = m_state[r];
                    for (int c = 0; c < bc_w; ++c) {
                        if (S_tile[r * kBc + c] > m_new) m_new = S_tile[r * kBc + c];
                    }
                    if (!std::isfinite(m_new)) m_new = m_state[r];

                    const float scale_old = (m_state[r] == -INFINITY) ? 0.0f
                                          : std::exp(m_state[r] - m_new);
                    float l_new = l_state[r] * scale_old;
                    /* Rescale O. */
                    for (int d = 0; d < D; ++d) O_state[r * D + d] *= scale_old;

                    /* Add contributions from this tile's keys/values. */
                    for (int c = 0; c < bc_w; ++c) {
                        const float p = std::exp(S_tile[r * kBc + c] - m_new);
                        if (!std::isfinite(p)) continue;
                        l_new += p;
                        const uint16_t* Vrow = Vbh + (size_t)(kb + c) * D;
                        for (int d = 0; d < D; ++d) {
                            O_state[r * D + d] += p * tc_cpu_f16_to_f32(Vrow[d]);
                        }
                    }
                    m_state[r] = m_new;
                    l_state[r] = l_new;
                }
            }

            /* Finalize: O[r][d] /= l[r], write back fp16. Save LSE = m + log(l)
             * for use in attention backward. */
            for (int r = 0; r < br_h; ++r) {
                const float inv_l = l_state[r] != 0.0f ? 1.0f / l_state[r] : 0.0f;
                uint16_t* Orow = Obh + (size_t)(qb + r) * D;
                for (int d = 0; d < D; ++d) {
                    Orow[d] = tc_cpu_f32_to_f16(O_state[r * D + d] * inv_l);
                }
                if (Lbh) {
                    Lbh[qb + r] = m_state[r] + std::log(std::max(l_state[r], 1e-30f));
                }
            }
        }
    }
    return tc_record_dispatch("tc_attention_forward", TC_BACKEND_PORTABLE_CPU, TC_OK);
}

extern "C" tc_status_t tc_attention_forward_async(tc_context* ctx,
                                                   const tc_attention_desc* desc,
                                                   const tc_buffer* Q,
                                                   const tc_buffer* K,
                                                   const tc_buffer* V,
                                                   tc_buffer* O,
                                                   tc_buffer* LSE,
                                                   tc_stream* stream) {
    (void)stream;
    /* CPU backend has no real GPU stream; just run synchronously. */
    return tc_attention_forward(ctx, desc, Q, K, V, O, LSE);
}

/* ----------------------------------------------------------------------- *
 * Attention backward
 *
 *   Inputs: Q, K, V, O (forward output), dO (gradient at output), LSE
 *   Outputs: dQ, dK, dV
 *
 *   Algorithm (FlashAttention-2 backward):
 *     Recompute S, P from Q, K, scale, LSE (no need to store P).
 *     dV  += P^T @ dO
 *     dP   = dO @ V^T
 *     dS   = P * (dP - rowsum(dO * O))
 *     dQ  += dS @ K * scale
 *     dK  += dS^T @ Q * scale
 *
 *   Each (b, h, q_tile) tile recomputes its row of S/P and updates
 *   dQ for that row, accumulating dK/dV across all q_tiles for fixed
 *   (b, h, k_tile). To avoid contention on dK/dV we hold per-thread
 *   partials and reduce at the end.
 * ----------------------------------------------------------------------- */
extern "C" tc_status_t tc_attention_backward(tc_context* ctx,
                                              const tc_attention_desc* desc,
                                              const tc_buffer* Q,
                                              const tc_buffer* K,
                                              const tc_buffer* V,
                                              const tc_buffer* O,
                                              const tc_buffer* dO,
                                              const tc_buffer* LSE,
                                              tc_buffer* dQ,
                                              tc_buffer* dK,
                                              tc_buffer* dV) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!desc || !Q || !K || !V || !O || !dO || !LSE || !dQ || !dK || !dV)
        return TC_ERR_INVALID_ARG;
    if (desc->batch <= 0 || desc->heads <= 0 ||
        desc->seq_q <= 0 || desc->seq_kv <= 0 || desc->head_dim <= 0) return TC_ERR_INVALID_ARG;
    if (desc->io_dtype != TC_DTYPE_F16 || desc->accum_dtype != TC_DTYPE_F32) {
        return TC_ERR_UNSUPPORTED_DTYPE;
    }

    const int B = desc->batch;
    const int Hq = desc->heads;
    const int Hkv = desc->kv_heads > 0 ? desc->kv_heads : desc->heads;
    if (Hkv <= 0 || Hq % Hkv != 0) return TC_ERR_INVALID_ARG;
    const int Sq = desc->seq_q;
    const int Sk = desc->seq_kv;
    const int D = desc->head_dim;
    const int Hq_per_Hkv = Hq / Hkv;
    const float scale = desc->softmax_scale > 0 ? desc->softmax_scale : (1.0f / std::sqrt((float)D));
    const bool causal = desc->causal != 0;
    const int swin = desc->window_size;

    void *Qp, *Kp, *Vp, *Op, *dOp, *Lp, *dQp, *dKp, *dVp;
    tc_buffer_map((tc_buffer*)Q, &Qp);
    tc_buffer_map((tc_buffer*)K, &Kp);
    tc_buffer_map((tc_buffer*)V, &Vp);
    tc_buffer_map((tc_buffer*)O, &Op);
    tc_buffer_map((tc_buffer*)dO, &dOp);
    tc_buffer_map((tc_buffer*)LSE, &Lp);
    tc_buffer_map(dQ, &dQp);
    tc_buffer_map(dK, &dKp);
    tc_buffer_map(dV, &dVp);

    const uint16_t* Qd = (const uint16_t*)Qp;
    const uint16_t* Kd = (const uint16_t*)Kp;
    const uint16_t* Vd = (const uint16_t*)Vp;
    const uint16_t* Od = (const uint16_t*)Op;
    const uint16_t* dOd = (const uint16_t*)dOp;
    const float* Ld = (const float*)Lp;
    uint16_t* dQd = (uint16_t*)dQp;
    uint16_t* dKd = (uint16_t*)dKp;
    uint16_t* dVd = (uint16_t*)dVp;

    /* Zero outputs (fp16 outputs are initialized by us). */
    std::memset(dQd, 0, (size_t)B * Hq * Sq * D * sizeof(uint16_t));
    std::memset(dKd, 0, (size_t)B * Hkv * Sk * D * sizeof(uint16_t));
    std::memset(dVd, 0, (size_t)B * Hkv * Sk * D * sizeof(uint16_t));

    /* Per-thread fp32 partials for dK and dV (avoids fp16 atomics). */
#if defined(_OPENMP)
    const int n_threads = std::max(1, omp_get_max_threads());
#else
    const int n_threads = 1;
#endif
    const size_t kv_floats = (size_t)B * Hkv * Sk * D;
    std::vector<float> dK_partials((size_t)n_threads * kv_floats, 0.0f);
    std::vector<float> dV_partials((size_t)n_threads * kv_floats, 0.0f);
    /* dQ doesn't have the cross-head accumulation issue (each q index is
     * written by exactly one (b, h, q) thread), so we use fp32 for it too
     * and convert at the end. */
    std::vector<float> dQ_fp32((size_t)B * Hq * Sq * D, 0.0f);

    const long total = (long)B * Hq;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic) if (total > 1)
#endif
    for (long bh = 0; bh < total; ++bh) {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        const int b = (int)(bh / Hq);
        const int h = (int)(bh % Hq);
        const int hkv = h / Hq_per_Hkv;

        const uint16_t* Qbh = Qd + ((size_t)b * Hq + h) * Sq * D;
        const uint16_t* Kbh = Kd + ((size_t)b * Hkv + hkv) * Sk * D;
        const uint16_t* Vbh = Vd + ((size_t)b * Hkv + hkv) * Sk * D;
        const uint16_t* Obh = Od + ((size_t)b * Hq + h) * Sq * D;
        const uint16_t* dObh = dOd + ((size_t)b * Hq + h) * Sq * D;
        const float*    Lbh = Ld + ((size_t)b * Hq + h) * Sq;
        float* dQbh = dQ_fp32.data() + ((size_t)b * Hq + h) * Sq * D;
        float* dKbh = dK_partials.data() + (size_t)tid * kv_floats
                      + ((size_t)b * Hkv + hkv) * Sk * D;
        float* dVbh = dV_partials.data() + (size_t)tid * kv_floats
                      + ((size_t)b * Hkv + hkv) * Sk * D;

        /* Precompute rowsum(dO * O) per Q row — needed in dS computation. */
        std::vector<float> Di(Sq, 0.0f);
        for (int q = 0; q < Sq; ++q) {
            float s = 0.0f;
            const uint16_t* dOq = dObh + (size_t)q * D;
            const uint16_t* Oq  = Obh  + (size_t)q * D;
            for (int d = 0; d < D; ++d) s += tc_cpu_f16_to_f32(dOq[d]) * tc_cpu_f16_to_f32(Oq[d]);
            Di[q] = s;
        }

        /* Recompute P[q][k] = exp(s[q][k] - lse[q]) and accumulate. */
        for (int q = 0; q < Sq; ++q) {
            const float lse_q = Lbh[q];
            const float Dq = Di[q];
            const uint16_t* Qq = Qbh + (size_t)q * D;
            const uint16_t* dOq = dObh + (size_t)q * D;
            float* dQq = dQbh + (size_t)q * D;
            for (int k = 0; k < Sk; ++k) {
                /* causal / sliding mask same as forward */
                if (causal && k > q) continue;
                if (swin > 0 && q - k > swin) continue;

                const uint16_t* Kk = Kbh + (size_t)k * D;
                const uint16_t* Vk = Vbh + (size_t)k * D;
                float s = dot_fp16(Qq, Kk, D) * scale;
                if (desc->alibi_slopes) {
                    const float slope = desc->alibi_slopes[h];
                    s -= slope * (float)(q - k);
                }
                const float P = std::exp(s - lse_q);

                /* dP_qk = sum_d dO[q,d] * V[k,d]    (== <dO_q, V_k>) */
                float dP = 0.0f;
                for (int d = 0; d < D; ++d) dP += tc_cpu_f16_to_f32(dOq[d]) * tc_cpu_f16_to_f32(Vk[d]);
                const float dS = P * (dP - Dq);

                /* dV[k] += P * dO[q] */
                for (int d = 0; d < D; ++d) dVbh[(size_t)k * D + d] += P * tc_cpu_f16_to_f32(dOq[d]);
                /* dK[k] += dS * Q[q] * scale */
                for (int d = 0; d < D; ++d) dKbh[(size_t)k * D + d] += dS * tc_cpu_f16_to_f32(Qq[d]) * scale;
                /* dQ[q] += dS * K[k] * scale */
                for (int d = 0; d < D; ++d) dQq[d] += dS * tc_cpu_f16_to_f32(Kk[d]) * scale;
            }
        }
    }

    /* Reduce per-thread dK/dV partials and convert to fp16. */
    const size_t total_kv = kv_floats;
    std::vector<float> dK_acc(total_kv, 0.0f);
    std::vector<float> dV_acc(total_kv, 0.0f);
    for (int t = 0; t < n_threads; ++t) {
        const float* dKp_t = dK_partials.data() + (size_t)t * total_kv;
        const float* dVp_t = dV_partials.data() + (size_t)t * total_kv;
        for (size_t i = 0; i < total_kv; ++i) {
            dK_acc[i] += dKp_t[i];
            dV_acc[i] += dVp_t[i];
        }
    }
    for (size_t i = 0; i < total_kv; ++i) {
        dKd[i] = tc_cpu_f32_to_f16(dK_acc[i]);
        dVd[i] = tc_cpu_f32_to_f16(dV_acc[i]);
    }

    /* Convert dQ partials (we never sharded across threads since each q index
     * belonged to exactly one thread). */
    const size_t total_q = (size_t)B * Hq * Sq * D;
    for (size_t i = 0; i < total_q; ++i) dQd[i] = tc_cpu_f32_to_f16(dQ_fp32[i]);

    return tc_record_dispatch("tc_attention_backward", TC_BACKEND_PORTABLE_CPU, TC_OK);
}
