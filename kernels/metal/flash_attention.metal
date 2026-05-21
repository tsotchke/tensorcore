/*
 * tensorcore — fused FlashAttention-2 forward, Apple Silicon.
 *
 *   S = (Q @ K^T) * softmax_scale     (+ optional causal mask)
 *   P = softmax(S, dim=-1)
 *   O = P @ V
 *
 * v0.1 layout (fits 32 KB threadgroup memory on Apple7+):
 *   - Br = Bc = 32 (query / kv block size)
 *   - WM = WN = 2 → 4 simdgroups = 128 threads per threadgroup
 *   - Each simdgroup owns 16x16 of S (2x2 of 8x8 fragments)
 *                       and 16x32 of O (2x4 of 8x8 fragments) when D=64.
 *   - All accumulators fp32. Inputs/outputs fp16.
 *
 * Threadgroup memory (≈ 22 KB):
 *   sQ : Br × D  fp16  = 32 × 64 × 2 = 4096
 *   sK : Bc × D  fp16  = 32 × 64 × 2 = 4096   (reused for sP in phase C)
 *   sV : Bc × D  fp16  = 32 × 64 × 2 = 4096
 *   sS : Br × Bc fp32  = 32 × 32 × 4 = 4096   (lives in sV region; sV not yet loaded
 *                                              when we need sS)
 *   sP : Br × Bc fp16  = 32 × 32 × 2 = 2048   (overlaps with sK after K is consumed)
 *   sO_scratch : 64 fp32 per simdgroup × 4 sg = 1024
 *   m_row, l_row, alpha_row : 3 × Br × fp32   = 384
 *
 * Dispatch grid: (num_q_blocks, heads, batch).
 *
 * Online softmax (FlashAttention-2): for each query row r and kv block j,
 *   m_new = max(m_prev, rowmax(S_j[r]))
 *   alpha = exp(m_prev - m_new)
 *   l_new = alpha * l_prev + rowsum(exp(S_j[r] - m_new))
 *   O_new = alpha * O_prev + exp(S_j[r] - m_new) @ V_j
 * Final: O <- O / l ;  optionally LSE = m + log(l).
 */

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

constant constexpr uint FA_BR      = 32;
constant constexpr uint FA_BC      = 32;
constant constexpr uint FA_D       = 64;
constant constexpr uint FA_WM      = 2;
constant constexpr uint FA_WN      = 2;
constant constexpr uint FA_THREADS = FA_WM * FA_WN * 32;   /* 128 */

/* Per-simdgroup output tile dims (in 8x8 fragments). */
constant constexpr uint FA_TM_S = FA_BR / FA_WM / 8;   /* 32/2/8 = 2 */
constant constexpr uint FA_TN_S = FA_BC / FA_WN / 8;   /* 32/2/8 = 2 */
constant constexpr uint FA_TM_O = FA_BR / FA_WM / 8;   /* 2 */
constant constexpr uint FA_TN_O = FA_D  / FA_WN / 8;   /* 64/2/8 = 4 */

constant bool g_causal     [[function_constant(0)]];
constant bool g_return_lse [[function_constant(1)]];
constant bool g_use_window [[function_constant(2)]];
constant bool g_use_alibi  [[function_constant(3)]];

/* Cooperative tile load. */
template <typename T>
inline void coop_load(threadgroup T*       dst, uint dst_stride,
                      device   const T*   src, uint src_stride,
                      uint                 row0, uint col0,
                      uint                 rows, uint cols,
                      uint                 row_limit, uint col_limit,
                      uint                 tid)
{
    const uint n = rows * cols;
    for (uint idx = tid; idx < n; idx += FA_THREADS) {
        const uint r  = idx / cols;
        const uint c  = idx % cols;
        const uint gr = row0 + r;
        const uint gc = col0 + c;
        dst[r * dst_stride + c] =
            (gr < row_limit && gc < col_limit) ? src[gr * src_stride + gc] : T(0);
    }
}

kernel void tc_flash_attention_f16_d64(
    device const half*  Q         [[buffer(0)]],
    device const half*  K         [[buffer(1)]],
    device const half*  V         [[buffer(2)]],
    device       half*  O         [[buffer(3)]],
    device       float* LSE       [[buffer(4), function_constant(g_return_lse)]],
    constant uint& batch          [[buffer(5)]],
    constant uint& heads          [[buffer(6)]],
    constant uint& kv_heads       [[buffer(7)]],
    constant uint& seq_q          [[buffer(8)]],
    constant uint& seq_kv         [[buffer(9)]],
    constant float& softmax_scale [[buffer(10)]],
    constant uint& window_size    [[buffer(11), function_constant(g_use_window)]],
    constant float* alibi_slopes  [[buffer(12), function_constant(g_use_alibi)]],
    uint3 group_id                [[threadgroup_position_in_grid]],
    uint  sgid                    [[simdgroup_index_in_threadgroup]],
    uint  slid                    [[thread_index_in_simdgroup]])
{
    constexpr uint D = FA_D;

    const uint q_block_idx = group_id.x;
    const uint head_idx    = group_id.y;
    const uint batch_idx   = group_id.z;
    const uint kv_head_idx = (kv_heads > 0 && kv_heads != heads)
                             ? (head_idx * kv_heads / heads) : head_idx;

    const uint row0 = q_block_idx * FA_BR;
    if (row0 >= seq_q) return;

    const uint q_base   = ((batch_idx * heads    + head_idx)    * seq_q  + 0) * D;
    const uint k_base   = ((batch_idx * kv_heads + kv_head_idx) * seq_kv + 0) * D;
    const uint v_base   = ((batch_idx * kv_heads + kv_head_idx) * seq_kv + 0) * D;
    const uint o_base   = ((batch_idx * heads    + head_idx)    * seq_q  + 0) * D;
    const uint lse_base = ((batch_idx * heads    + head_idx)    * seq_q  + 0);

    const uint tid    = sgid * 32 + slid;
    const uint sg_row = sgid / FA_WN;     /* 0..FA_WM-1 */
    const uint sg_col = sgid % FA_WN;     /* 0..FA_WN-1 */

    /* ------- Threadgroup memory: single buffer, manually partitioned ------- */
    threadgroup half  sQ[FA_BR * D];
    threadgroup half  sK[FA_BC * D];
    threadgroup half  sV[FA_BC * D];
    threadgroup float sS[FA_BR * FA_BC];
    threadgroup half  sP[FA_BR * FA_BC];
    threadgroup float m_row[FA_BR];
    threadgroup float l_row[FA_BR];
    threadgroup float alpha_row[FA_BR];
    threadgroup float sg_scratch[FA_WM * FA_WN * 64];  /* per-sg 8x8 spill */

    /* ------- Load Q once ------- */
    coop_load<half>(sQ, D, Q + q_base, D,
                    row0, 0, FA_BR, D, seq_q, D, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* ------- Init O_acc, m_row, l_row ------- */
    simdgroup_matrix<float, 8, 8> O_acc[FA_TM_O][FA_TN_O];
    for (uint i = 0; i < FA_TM_O; ++i)
        for (uint j = 0; j < FA_TN_O; ++j)
            O_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint idx = tid; idx < FA_BR; idx += FA_THREADS) {
        m_row[idx] = -INFINITY;
        l_row[idx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* ============================================================ */
    /* Outer loop over KV blocks                                    */
    /* ============================================================ */
    const uint Tc = (seq_kv + FA_BC - 1) / FA_BC;
    for (uint jblk = 0; jblk < Tc; ++jblk) {
        const uint kv_col0 = jblk * FA_BC;

        /* Causal early-exit: every query in this row tile <= row0+Br-1; if
         * kv_col0 strictly greater, all entries are masked. */
        if (g_causal && (kv_col0 > row0 + FA_BR - 1)) break;

        /* ----- load K and V ----- */
        coop_load<half>(sK, D, K + k_base, D,
                        kv_col0, 0, FA_BC, D, seq_kv, D, tid);
        coop_load<half>(sV, D, V + v_base, D,
                        kv_col0, 0, FA_BC, D, seq_kv, D, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* ----- compute S = Q @ K^T, fp32 accum ----- */
        simdgroup_matrix<float, 8, 8> S_acc[FA_TM_S][FA_TN_S];
        for (uint i = 0; i < FA_TM_S; ++i)
            for (uint j = 0; j < FA_TN_S; ++j)
                S_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint kk = 0; kk < D; kk += 8) {
            simdgroup_matrix<half, 8, 8> q_frag[FA_TM_S];
            simdgroup_matrix<half, 8, 8> k_frag[FA_TN_S];

            for (uint i = 0; i < FA_TM_S; ++i) {
                const uint row = sg_row * (FA_TM_S * 8) + i * 8;
                simdgroup_load(q_frag[i], sQ + row * D + kk, D);
            }
            /* Load K transposed: K is [Bc × D] row-major; we want K^T [D × Bc]
             * fragments for QK^T. Use the 5-arg transposed simdgroup_load. */
            for (uint j = 0; j < FA_TN_S; ++j) {
                const uint col = sg_col * (FA_TN_S * 8) + j * 8;
                simdgroup_load(k_frag[j], sK + col * D + kk, D, ulong2(0, 0), true);
            }
            for (uint i = 0; i < FA_TM_S; ++i)
                for (uint j = 0; j < FA_TN_S; ++j)
                    simdgroup_multiply_accumulate(S_acc[i][j],
                                                  q_frag[i], k_frag[j],
                                                  S_acc[i][j]);
        }

        /* ----- spill S to TG memory ----- */
        for (uint i = 0; i < FA_TM_S; ++i) {
            for (uint j = 0; j < FA_TN_S; ++j) {
                const uint sr = sg_row * (FA_TM_S * 8) + i * 8;
                const uint sc = sg_col * (FA_TN_S * 8) + j * 8;
                simdgroup_store(S_acc[i][j], sS + sr * FA_BC + sc, FA_BC);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* ----- scale + causal/window/alibi ----- */
        if (g_causal || g_use_window || g_use_alibi || softmax_scale != 1.0f) {
            for (uint idx = tid; idx < FA_BR * FA_BC; idx += FA_THREADS) {
                const uint r = idx / FA_BC;
                const uint c = idx % FA_BC;
                float v = sS[r * FA_BC + c] * softmax_scale;
                const uint gq = row0 + r;
                const uint gk = kv_col0 + c;
                if (g_causal && gk > gq) v = -INFINITY;
                if (g_use_window && gq > gk + window_size) v = -INFINITY;
                if (g_use_alibi && v > -1e30f) {
                    /* ALiBi linear bias: subtract slope * (i - j). */
                    const float alibi_slope = alibi_slopes[head_idx];
                    const float bias = alibi_slope * (float)((int)gk - (int)gq);
                    v += bias;
                }
                sS[r * FA_BC + c] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        /* ----- per-row online softmax (one thread per Br row) ----- */
        if (tid < FA_BR) {
            const uint r = tid;
            const float m_prev = m_row[r];
            const float l_prev = l_row[r];

            float m_new = m_prev;
            for (uint c = 0; c < FA_BC; ++c) {
                const float v = sS[r * FA_BC + c];
                m_new = max(m_new, v);
            }

            float l_partial = 0.0f;
            for (uint c = 0; c < FA_BC; ++c) {
                float v = sS[r * FA_BC + c];
                v = (v > -1e30f) ? exp(v - m_new) : 0.0f;
                sS[r * FA_BC + c] = v;
                l_partial += v;
            }
            const float alpha = (m_prev > -1e30f) ? exp(m_prev - m_new) : 0.0f;
            m_row[r]     = m_new;
            l_row[r]     = alpha * l_prev + l_partial;
            alpha_row[r] = alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* ----- rescale O_acc by alpha[r] ----- */
        threadgroup float* my_scratch = sg_scratch + sgid * 64;
        for (uint i = 0; i < FA_TM_O; ++i) {
            for (uint j = 0; j < FA_TN_O; ++j) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_store(O_acc[i][j], my_scratch, 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const uint sr = sg_row * (FA_TM_O * 8) + i * 8;
                /* Scale this 8x8 fragment by alpha for its rows. */
                for (uint k = 0; k < 64; k += 32) {
                    const uint idx = k + slid;
                    if (idx < 64) {
                        const uint lr = idx >> 3;
                        const float a = alpha_row[sr + lr];
                        my_scratch[idx] *= a;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(O_acc[i][j], my_scratch, 8);
            }
        }

        /* ----- convert S (now == exp(S-m)) into sP (fp16) ----- */
        for (uint idx = tid; idx < FA_BR * FA_BC; idx += FA_THREADS) {
            sP[idx] = (half)sS[idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* ----- O_acc += P @ V ----- */
        for (uint kk = 0; kk < FA_BC; kk += 8) {
            simdgroup_matrix<half, 8, 8> p_frag[FA_TM_O];
            simdgroup_matrix<half, 8, 8> v_frag[FA_TN_O];

            for (uint i = 0; i < FA_TM_O; ++i) {
                const uint row = sg_row * (FA_TM_O * 8) + i * 8;
                simdgroup_load(p_frag[i], sP + row * FA_BC + kk, FA_BC);
            }
            for (uint j = 0; j < FA_TN_O; ++j) {
                const uint col = sg_col * (FA_TN_O * 8) + j * 8;
                simdgroup_load(v_frag[j], sV + kk * D + col, D);
            }
            for (uint i = 0; i < FA_TM_O; ++i)
                for (uint j = 0; j < FA_TN_O; ++j)
                    simdgroup_multiply_accumulate(O_acc[i][j],
                                                  p_frag[i], v_frag[j],
                                                  O_acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* ============================================================ */
    /* Final: O <- O / l ; store as half; (optional) LSE             */
    /* ============================================================ */
    threadgroup float* my_scratch = sg_scratch + sgid * 64;
    for (uint i = 0; i < FA_TM_O; ++i) {
        for (uint j = 0; j < FA_TN_O; ++j) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(O_acc[i][j], my_scratch, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint sr = sg_row * (FA_TM_O * 8) + i * 8;
            const uint sc = sg_col * (FA_TN_O * 8) + j * 8;

            for (uint k = 0; k < 64; k += 32) {
                const uint idx = k + slid;
                if (idx < 64) {
                    const uint lr  = idx >> 3;
                    const uint lc  = idx & 7;
                    const uint gr  = row0 + sr + lr;
                    const uint gc  = sc + lc;
                    if (gr < seq_q) {
                        const float l = l_row[sr + lr];
                        const float v = my_scratch[idx] / (l + 1e-30f);
                        O[o_base + gr * D + gc] = (half)v;
                    }
                }
            }
        }
    }

    if (g_return_lse) {
        for (uint idx = tid; idx < FA_BR; idx += FA_THREADS) {
            const uint gr = row0 + idx;
            if (gr >= seq_q) continue;
            LSE[lse_base + gr] = m_row[idx] + log(max(l_row[idx], 1e-30f));
        }
    }
}
