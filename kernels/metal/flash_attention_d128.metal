/*
 * tensorcore — fused FlashAttention-2 forward, head_dim = 128.
 *
 * Shape: this is the llama / GPT-J / mistral standard head_dim.
 *
 * To keep within the 32 KB threadgroup-memory budget while loading Q/K/V at
 * D=128, we drop Br = Bc to 16. That reduces parallelism per-threadgroup but
 * is what fits today. Larger Br for D=128 needs the Apple9+ TG memory bump
 * to 64 KB (M3+) or aliased buffer regions; phase-2.
 *
 * Layout:
 *   - Br = Bc = 16
 *   - WM = 2, WN = 2 → 4 simdgroups = 128 threads
 *   - S (16×16): each simdgroup owns 8×8 (single 8×8 fragment, TM_S=1, TN_S=1)
 *   - O (16×128): each sg owns 8×64 (1×8 fragments) — TM_O=1, TN_O=8
 *   - All accumulators fp32, IO fp16.
 *
 * TG memory ≈ 15 KB.
 */

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

constant constexpr uint FA128_BR      = 16;
constant constexpr uint FA128_BC      = 16;
constant constexpr uint FA128_D       = 128;
constant constexpr uint FA128_WM      = 2;
constant constexpr uint FA128_WN      = 2;
constant constexpr uint FA128_THREADS = FA128_WM * FA128_WN * 32;   /* 128 */

constant constexpr uint FA128_TM_S = FA128_BR / FA128_WM / 8;   /* 1 */
constant constexpr uint FA128_TN_S = FA128_BC / FA128_WN / 8;   /* 1 */
constant constexpr uint FA128_TM_O = FA128_BR / FA128_WM / 8;   /* 1 */
constant constexpr uint FA128_TN_O = FA128_D  / FA128_WN / 8;   /* 8 */

constant bool g128_causal     [[function_constant(0)]];
constant bool g128_return_lse [[function_constant(1)]];
constant bool g128_use_window [[function_constant(2)]];
constant bool g128_use_alibi  [[function_constant(3)]];

template <typename T>
inline void coop128(threadgroup T*       dst, uint dst_stride,
                    device   const T*   src, uint src_stride,
                    uint                 row0, uint col0,
                    uint                 rows, uint cols,
                    uint                 row_limit, uint col_limit,
                    uint                 tid)
{
    const uint n = rows * cols;
    for (uint idx = tid; idx < n; idx += FA128_THREADS) {
        const uint r  = idx / cols;
        const uint c  = idx % cols;
        const uint gr = row0 + r;
        const uint gc = col0 + c;
        dst[r * dst_stride + c] =
            (gr < row_limit && gc < col_limit) ? src[gr * src_stride + gc] : T(0);
    }
}

kernel void tc_flash_attention_f16_d128(
    device const half*  Q         [[buffer(0)]],
    device const half*  K         [[buffer(1)]],
    device const half*  V         [[buffer(2)]],
    device       half*  O         [[buffer(3)]],
    device       float* LSE       [[buffer(4), function_constant(g128_return_lse)]],
    constant uint& batch          [[buffer(5)]],
    constant uint& heads          [[buffer(6)]],
    constant uint& kv_heads       [[buffer(7)]],
    constant uint& seq_q          [[buffer(8)]],
    constant uint& seq_kv         [[buffer(9)]],
    constant float& softmax_scale [[buffer(10)]],
    constant uint& window_size    [[buffer(11), function_constant(g128_use_window)]],
    constant float& alibi_slope   [[buffer(12), function_constant(g128_use_alibi)]],
    uint3 group_id                [[threadgroup_position_in_grid]],
    uint  sgid                    [[simdgroup_index_in_threadgroup]],
    uint  slid                    [[thread_index_in_simdgroup]])
{
    constexpr uint D = FA128_D;

    const uint q_block_idx = group_id.x;
    const uint head_idx    = group_id.y;
    const uint batch_idx   = group_id.z;
    const uint kv_head_idx = (kv_heads > 0 && kv_heads != heads)
                             ? (head_idx * kv_heads / heads) : head_idx;

    const uint row0 = q_block_idx * FA128_BR;
    if (row0 >= seq_q) return;

    const uint q_base   = ((batch_idx * heads    + head_idx)    * seq_q  + 0) * D;
    const uint k_base   = ((batch_idx * kv_heads + kv_head_idx) * seq_kv + 0) * D;
    const uint v_base   = ((batch_idx * kv_heads + kv_head_idx) * seq_kv + 0) * D;
    const uint o_base   = ((batch_idx * heads    + head_idx)    * seq_q  + 0) * D;
    const uint lse_base = ((batch_idx * heads    + head_idx)    * seq_q  + 0);

    const uint tid    = sgid * 32 + slid;
    const uint sg_row = sgid / FA128_WN;
    const uint sg_col = sgid % FA128_WN;

    threadgroup half  sQ[FA128_BR * D];
    threadgroup half  sK[FA128_BC * D];
    threadgroup half  sV[FA128_BC * D];
    threadgroup float sS[FA128_BR * FA128_BC];
    threadgroup half  sP[FA128_BR * FA128_BC];
    threadgroup float m_row[FA128_BR];
    threadgroup float l_row[FA128_BR];
    threadgroup float alpha_row[FA128_BR];
    threadgroup float sg_scratch[FA128_WM * FA128_WN * 64];

    coop128<half>(sQ, D, Q + q_base, D,
                  row0, 0, FA128_BR, D, seq_q, D, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> O_acc[FA128_TM_O][FA128_TN_O];
    for (uint i = 0; i < FA128_TM_O; ++i)
        for (uint j = 0; j < FA128_TN_O; ++j)
            O_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint idx = tid; idx < FA128_BR; idx += FA128_THREADS) {
        m_row[idx] = -INFINITY;
        l_row[idx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint Tc = (seq_kv + FA128_BC - 1) / FA128_BC;
    for (uint jblk = 0; jblk < Tc; ++jblk) {
        const uint kv_col0 = jblk * FA128_BC;
        if (g128_causal && (kv_col0 > row0 + FA128_BR - 1)) break;

        coop128<half>(sK, D, K + k_base, D,
                      kv_col0, 0, FA128_BC, D, seq_kv, D, tid);
        coop128<half>(sV, D, V + v_base, D,
                      kv_col0, 0, FA128_BC, D, seq_kv, D, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> S_acc[FA128_TM_S][FA128_TN_S];
        for (uint i = 0; i < FA128_TM_S; ++i)
            for (uint j = 0; j < FA128_TN_S; ++j)
                S_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint kk = 0; kk < D; kk += 8) {
            simdgroup_matrix<half, 8, 8> q_frag[FA128_TM_S];
            simdgroup_matrix<half, 8, 8> k_frag[FA128_TN_S];
            for (uint i = 0; i < FA128_TM_S; ++i) {
                const uint row = sg_row * (FA128_TM_S * 8) + i * 8;
                simdgroup_load(q_frag[i], sQ + row * D + kk, D);
            }
            for (uint j = 0; j < FA128_TN_S; ++j) {
                const uint col = sg_col * (FA128_TN_S * 8) + j * 8;
                simdgroup_load(k_frag[j], sK + col * D + kk, D, ulong2(0, 0), true);
            }
            for (uint i = 0; i < FA128_TM_S; ++i)
                for (uint j = 0; j < FA128_TN_S; ++j)
                    simdgroup_multiply_accumulate(S_acc[i][j],
                                                  q_frag[i], k_frag[j],
                                                  S_acc[i][j]);
        }

        for (uint i = 0; i < FA128_TM_S; ++i) {
            for (uint j = 0; j < FA128_TN_S; ++j) {
                const uint sr = sg_row * (FA128_TM_S * 8) + i * 8;
                const uint sc = sg_col * (FA128_TN_S * 8) + j * 8;
                simdgroup_store(S_acc[i][j], sS + sr * FA128_BC + sc, FA128_BC);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (g128_causal || g128_use_window || g128_use_alibi || softmax_scale != 1.0f) {
            for (uint idx = tid; idx < FA128_BR * FA128_BC; idx += FA128_THREADS) {
                const uint r = idx / FA128_BC;
                const uint c = idx % FA128_BC;
                float v = sS[r * FA128_BC + c] * softmax_scale;
                const uint gq = row0 + r;
                const uint gk = kv_col0 + c;
                if (g128_causal) {
                    if (gk > gq) v = -INFINITY;
                }
                if (g128_use_window && gq > gk + window_size) v = -INFINITY;
                if (g128_use_alibi && v > -1e30f) {
                    const float bias = alibi_slope * (float)((int)gk - (int)gq);
                    v += bias;
                }
                sS[r * FA128_BC + c] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid < FA128_BR) {
            const uint r = tid;
            const float m_prev = m_row[r];
            const float l_prev = l_row[r];

            float m_new = m_prev;
            for (uint c = 0; c < FA128_BC; ++c) {
                const float v = sS[r * FA128_BC + c];
                m_new = max(m_new, v);
            }

            float l_partial = 0.0f;
            for (uint c = 0; c < FA128_BC; ++c) {
                float v = sS[r * FA128_BC + c];
                v = (v > -1e30f) ? exp(v - m_new) : 0.0f;
                sS[r * FA128_BC + c] = v;
                l_partial += v;
            }
            const float alpha = (m_prev > -1e30f) ? exp(m_prev - m_new) : 0.0f;
            m_row[r]     = m_new;
            l_row[r]     = alpha * l_prev + l_partial;
            alpha_row[r] = alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float* my_scratch = sg_scratch + sgid * 64;
        for (uint i = 0; i < FA128_TM_O; ++i) {
            for (uint j = 0; j < FA128_TN_O; ++j) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_store(O_acc[i][j], my_scratch, 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const uint sr = sg_row * (FA128_TM_O * 8) + i * 8;
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

        for (uint idx = tid; idx < FA128_BR * FA128_BC; idx += FA128_THREADS) {
            sP[idx] = (half)sS[idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < FA128_BC; kk += 8) {
            simdgroup_matrix<half, 8, 8> p_frag[FA128_TM_O];
            simdgroup_matrix<half, 8, 8> v_frag[FA128_TN_O];
            for (uint i = 0; i < FA128_TM_O; ++i) {
                const uint row = sg_row * (FA128_TM_O * 8) + i * 8;
                simdgroup_load(p_frag[i], sP + row * FA128_BC + kk, FA128_BC);
            }
            for (uint j = 0; j < FA128_TN_O; ++j) {
                const uint col = sg_col * (FA128_TN_O * 8) + j * 8;
                simdgroup_load(v_frag[j], sV + kk * D + col, D);
            }
            for (uint i = 0; i < FA128_TM_O; ++i)
                for (uint j = 0; j < FA128_TN_O; ++j)
                    simdgroup_multiply_accumulate(O_acc[i][j],
                                                  p_frag[i], v_frag[j],
                                                  O_acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* my_scratch = sg_scratch + sgid * 64;
    for (uint i = 0; i < FA128_TM_O; ++i) {
        for (uint j = 0; j < FA128_TN_O; ++j) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(O_acc[i][j], my_scratch, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint sr = sg_row * (FA128_TM_O * 8) + i * 8;
            const uint sc = sg_col * (FA128_TN_O * 8) + j * 8;

            for (uint k = 0; k < 64; k += 32) {
                const uint idx = k + slid;
                if (idx < 64) {
                    const uint lr = idx >> 3;
                    const uint lc = idx & 7;
                    const uint gr = row0 + sr + lr;
                    const uint gc = sc + lc;
                    if (gr < seq_q) {
                        const float l = l_row[sr + lr];
                        const float v = my_scratch[idx] / (l + 1e-30f);
                        O[o_base + gr * D + gc] = (half)v;
                    }
                }
            }
        }
    }

    if (g128_return_lse) {
        for (uint idx = tid; idx < FA128_BR; idx += FA128_THREADS) {
            const uint gr = row0 + idx;
            if (gr >= seq_q) continue;
            LSE[lse_base + gr] = m_row[idx] + log(max(l_row[idx], 1e-30f));
        }
    }
}
