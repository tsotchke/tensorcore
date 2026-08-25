/*
 * tensorcore — FlashAttention-2 backward at head_dim = 128.
 *
 * Same split-kernel design as flash_attention_backward.metal (D=64), with
 * Br=Bc=16 to keep within 32 KB threadgroup memory at D=128.
 *
 * Each simdgroup owns 8 of the 8x8 fragments of the 16x128 output (TM_O=1,
 * TN_O=8 for dQ; same for dK/dV). 4 simdgroups × 32 threads = 128 threads/TG.
 */

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

constant constexpr uint BR128 = 16;
constant constexpr uint BC128 = 16;
constant constexpr uint D128  = 128;
constant constexpr uint WM128 = 2;
constant constexpr uint WN128 = 2;
constant constexpr uint THREADS128 = WM128 * WN128 * 32;

constant constexpr uint TM_S128 = BR128 / WM128 / 8;        /* 1 */
constant constexpr uint TN_S128 = BC128 / WN128 / 8;        /* 1 */
constant constexpr uint TM_O128 = BR128 / WM128 / 8;        /* 1 */
constant constexpr uint TN_O128 = D128  / WN128 / 8;        /* 8 */
constant constexpr uint TM_K128 = BC128 / WM128 / 8;        /* 1 */
constant constexpr uint TN_K128 = D128  / WN128 / 8;        /* 8 */

constant bool g_bw128_causal [[function_constant(0)]];

template <typename T>
inline void coop128bw(threadgroup T*       dst, uint dst_stride,
                      device   const T*   src, uint src_stride,
                      uint row0, uint col0,
                      uint rows, uint cols,
                      uint row_limit, uint col_limit,
                      uint tid)
{
    const uint n = rows * cols;
    for (uint idx = tid; idx < n; idx += THREADS128) {
        const uint r  = idx / cols;
        const uint c  = idx % cols;
        const uint gr = row0 + r;
        const uint gc = col0 + c;
        dst[r * dst_stride + c] =
            (gr < row_limit && gc < col_limit) ? src[gr * src_stride + gc] : T(0);
    }
}

kernel void tc_flash_attention_backward_dq_d128(
    device const half*  Q              [[buffer(0)]],
    device const half*  K              [[buffer(1)]],
    device const half*  V              [[buffer(2)]],
    device const half*  O              [[buffer(3)]],
    device const half*  dO             [[buffer(4)]],
    device const float* LSE            [[buffer(5)]],
    device       half*  dQ             [[buffer(6)]],
    constant uint& batch               [[buffer(7)]],
    constant uint& heads               [[buffer(8)]],
    constant uint& kv_heads            [[buffer(9)]],
    constant uint& seq_q               [[buffer(10)]],
    constant uint& seq_kv              [[buffer(11)]],
    constant float& softmax_scale      [[buffer(12)]],
    uint3 group_id                     [[threadgroup_position_in_grid]],
    uint  sgid                         [[simdgroup_index_in_threadgroup]],
    uint  slid                         [[thread_index_in_simdgroup]])
{
    constexpr uint D = D128;
    const uint q_block_idx = group_id.x;
    const uint head_idx    = group_id.y;
    const uint batch_idx   = group_id.z;
    const uint kv_head_idx = (kv_heads > 0 && kv_heads != heads)
                             ? (head_idx * kv_heads / heads) : head_idx;

    const uint row0 = q_block_idx * BR128;
    if (row0 >= seq_q) return;

    const uint q_off   = ((batch_idx * heads    + head_idx)    * seq_q  + 0) * D;
    const uint kv_kbase= ((batch_idx * kv_heads + kv_head_idx) * seq_kv + 0) * D;
    const uint o_off   = q_off, do_off = q_off, dq_off = q_off;
    const uint lse_off = ((batch_idx * heads    + head_idx)    * seq_q  + 0);

    const uint tid = sgid * 32 + slid;
    const uint sg_row = sgid / WN128;
    const uint sg_col = sgid % WN128;

    threadgroup half  sQ [BR128 * D];
    threadgroup half  sO [BR128 * D];
    threadgroup half  sdO[BR128 * D];
    threadgroup float sLSE[BR128];
    threadgroup float sD_i[BR128];
    threadgroup half  sK [BC128 * D];
    threadgroup half  sV [BC128 * D];
    threadgroup float sS [BR128 * BC128];
    threadgroup half  sP [BR128 * BC128];
    threadgroup float sdP[BR128 * BC128];

    coop128bw<half>(sQ,  D, Q  + q_off,  D, row0, 0, BR128, D, seq_q, D, tid);
    coop128bw<half>(sO,  D, O  + o_off,  D, row0, 0, BR128, D, seq_q, D, tid);
    coop128bw<half>(sdO, D, dO + do_off, D, row0, 0, BR128, D, seq_q, D, tid);
    for (uint idx = tid; idx < BR128; idx += THREADS128) {
        const uint gr = row0 + idx;
        sLSE[idx] = (gr < seq_q) ? LSE[lse_off + gr] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < BR128) {
        float acc = 0.0f;
        for (uint d = 0; d < D; ++d)
            acc += (float)sdO[tid * D + d] * (float)sO[tid * D + d];
        sD_i[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> dQ_acc[TM_O128][TN_O128];
    for (uint i = 0; i < TM_O128; ++i)
        for (uint j = 0; j < TN_O128; ++j)
            dQ_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    const uint Tc = (seq_kv + BC128 - 1) / BC128;
    for (uint j = 0; j < Tc; ++j) {
        const uint kv_col0 = j * BC128;
        if (g_bw128_causal && (kv_col0 > row0 + BR128 - 1)) break;

        coop128bw<half>(sK, D, K + kv_kbase, D, kv_col0, 0, BC128, D, seq_kv, D, tid);
        coop128bw<half>(sV, D, V + kv_kbase, D, kv_col0, 0, BC128, D, seq_kv, D, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> S_acc[TM_S128][TN_S128];
        S_acc[0][0] = simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint kk = 0; kk < D; kk += 8) {
            simdgroup_matrix<half, 8, 8> q_frag;
            simdgroup_matrix<half, 8, 8> k_frag;
            const uint row = sg_row * 8;
            const uint col = sg_col * 8;
            simdgroup_load(q_frag, sQ + row * D + kk, D);
            simdgroup_load(k_frag, sK + col * D + kk, D, ulong2(0,0), true);
            simdgroup_multiply_accumulate(S_acc[0][0], q_frag, k_frag, S_acc[0][0]);
        }
        {
            const uint sr = sg_row * 8, sc = sg_col * 8;
            simdgroup_store(S_acc[0][0], sS + sr * BC128 + sc, BC128);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = tid; idx < BR128 * BC128; idx += THREADS128) {
            const uint r = idx / BC128;
            const uint c = idx % BC128;
            float v = sS[r * BC128 + c] * softmax_scale;
            if (g_bw128_causal) {
                const uint gq = row0 + r;
                const uint gk = kv_col0 + c;
                if (gk > gq) v = -INFINITY;
            }
            const float p = (v > -1e30f) ? exp(v - sLSE[r]) : 0.0f;
            sP[r * BC128 + c] = (half)p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> dP_acc[TM_S128][TN_S128];
        dP_acc[0][0] = simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint kk = 0; kk < D; kk += 8) {
            simdgroup_matrix<half, 8, 8> dO_frag, v_frag;
            const uint row = sg_row * 8, col = sg_col * 8;
            simdgroup_load(dO_frag, sdO + row * D + kk, D);
            simdgroup_load(v_frag, sV + col * D + kk, D, ulong2(0,0), true);
            simdgroup_multiply_accumulate(dP_acc[0][0], dO_frag, v_frag, dP_acc[0][0]);
        }
        {
            const uint sr = sg_row * 8, sc = sg_col * 8;
            simdgroup_store(dP_acc[0][0], sdP + sr * BC128 + sc, BC128);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = tid; idx < BR128 * BC128; idx += THREADS128) {
            const uint r = idx / BC128;
            const uint c = idx % BC128;
            const float p = (float)sP[r * BC128 + c];
            const float dp = sdP[r * BC128 + c];
            sP[r * BC128 + c] = (half)(p * (dp - sD_i[r]) * softmax_scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* dQ_acc += dS @ K  (Br × D += Br × Bc · Bc × D) */
        for (uint kk = 0; kk < BC128; kk += 8) {
            simdgroup_matrix<half, 8, 8> ds_frag[TM_O128];
            simdgroup_matrix<half, 8, 8> k_frag [TN_O128];
            for (uint i = 0; i < TM_O128; ++i) {
                const uint row = sg_row * 8 + i * 8;
                simdgroup_load(ds_frag[i], sP + row * BC128 + kk, BC128);
            }
            for (uint jj = 0; jj < TN_O128; ++jj) {
                const uint col = sg_col * (TN_O128 * 8) + jj * 8;
                simdgroup_load(k_frag[jj], sK + kk * D + col, D);
            }
            for (uint i = 0; i < TM_O128; ++i)
                for (uint jj = 0; jj < TN_O128; ++jj)
                    simdgroup_multiply_accumulate(dQ_acc[i][jj], ds_frag[i], k_frag[jj], dQ_acc[i][jj]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float scratch[WM128 * WN128 * 64];
    threadgroup float* my = scratch + sgid * 64;
    for (uint i = 0; i < TM_O128; ++i) {
        for (uint jj = 0; jj < TN_O128; ++jj) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(dQ_acc[i][jj], my, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint sr = sg_row * 8 + i * 8;
            const uint sc = sg_col * (TN_O128 * 8) + jj * 8;
            for (uint k = 0; k < 64; k += 32) {
                const uint idx = k + slid;
                if (idx < 64) {
                    const uint lr = idx >> 3;
                    const uint lc = idx & 7;
                    const uint gr = row0 + sr + lr;
                    const uint gc = sc + lc;
                    if (gr < seq_q) {
                        dQ[dq_off + gr * D + gc] = (half)my[idx];
                    }
                }
            }
        }
    }
}

kernel void tc_flash_attention_backward_dk_dv_d128(
    device const half*  Q              [[buffer(0)]],
    device const half*  K              [[buffer(1)]],
    device const half*  V              [[buffer(2)]],
    device const half*  O              [[buffer(3)]],
    device const half*  dO             [[buffer(4)]],
    device const float* LSE            [[buffer(5)]],
    device       half*  dK             [[buffer(6)]],
    device       half*  dV             [[buffer(7)]],
    constant uint& batch               [[buffer(8)]],
    constant uint& heads               [[buffer(9)]],
    constant uint& kv_heads            [[buffer(10)]],
    constant uint& seq_q               [[buffer(11)]],
    constant uint& seq_kv              [[buffer(12)]],
    constant float& softmax_scale      [[buffer(13)]],
    uint3 group_id                     [[threadgroup_position_in_grid]],
    uint  sgid                         [[simdgroup_index_in_threadgroup]],
    uint  slid                         [[thread_index_in_simdgroup]])
{
    constexpr uint D = D128;
    const uint kv_block_idx = group_id.x;
    const uint head_idx     = group_id.y;
    const uint batch_idx    = group_id.z;
    const uint kv_head_idx  = (kv_heads > 0 && kv_heads != heads)
                              ? (head_idx * kv_heads / heads) : head_idx;

    const uint col0 = kv_block_idx * BC128;
    if (col0 >= seq_kv) return;

    const uint q_off    = ((batch_idx * heads    + head_idx)    * seq_q  + 0) * D;
    const uint kv_kbase = ((batch_idx * kv_heads + kv_head_idx) * seq_kv + 0) * D;
    const uint dkv_off  = kv_kbase;
    const uint o_off    = q_off, do_off = q_off;
    const uint lse_off  = ((batch_idx * heads    + head_idx)    * seq_q  + 0);

    const uint tid = sgid * 32 + slid;
    const uint sg_row = sgid / WN128;
    const uint sg_col = sgid % WN128;

    threadgroup half  sK [BC128 * D];
    threadgroup half  sV [BC128 * D];
    threadgroup half  sQ [BR128 * D];
    threadgroup half  sO [BR128 * D];
    threadgroup half  sdO[BR128 * D];
    threadgroup float sLSE[BR128];
    threadgroup float sD_i[BR128];
    threadgroup float sS [BR128 * BC128];
    threadgroup half  sP [BR128 * BC128];
    threadgroup float sdP[BR128 * BC128];

    coop128bw<half>(sK, D, K + kv_kbase, D, col0, 0, BC128, D, seq_kv, D, tid);
    coop128bw<half>(sV, D, V + kv_kbase, D, col0, 0, BC128, D, seq_kv, D, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> dK_acc[TM_K128][TN_K128];
    simdgroup_matrix<float, 8, 8> dV_acc[TM_K128][TN_K128];
    for (uint i = 0; i < TM_K128; ++i)
        for (uint j = 0; j < TN_K128; ++j) {
            dK_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
            dV_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
        }

    const uint Tr = (seq_q + BR128 - 1) / BR128;
    for (uint i_blk = 0; i_blk < Tr; ++i_blk) {
        const uint row0 = i_blk * BR128;
        if (g_bw128_causal && (row0 + BR128 - 1 < col0)) continue;

        coop128bw<half>(sQ,  D, Q  + q_off,  D, row0, 0, BR128, D, seq_q, D, tid);
        coop128bw<half>(sO,  D, O  + o_off,  D, row0, 0, BR128, D, seq_q, D, tid);
        coop128bw<half>(sdO, D, dO + do_off, D, row0, 0, BR128, D, seq_q, D, tid);
        for (uint idx = tid; idx < BR128; idx += THREADS128) {
            const uint gr = row0 + idx;
            sLSE[idx] = (gr < seq_q) ? LSE[lse_off + gr] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < BR128) {
            float acc = 0.0f;
            for (uint d = 0; d < D; ++d)
                acc += (float)sdO[tid * D + d] * (float)sO[tid * D + d];
            sD_i[tid] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> S_acc;
        S_acc = simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint kk = 0; kk < D; kk += 8) {
            simdgroup_matrix<half, 8, 8> q_frag, k_frag;
            const uint row = sg_row * 8, col = sg_col * 8;
            simdgroup_load(q_frag, sQ + row * D + kk, D);
            simdgroup_load(k_frag, sK + col * D + kk, D, ulong2(0,0), true);
            simdgroup_multiply_accumulate(S_acc, q_frag, k_frag, S_acc);
        }
        {
            const uint sr = sg_row * 8, sc = sg_col * 8;
            simdgroup_store(S_acc, sS + sr * BC128 + sc, BC128);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = tid; idx < BR128 * BC128; idx += THREADS128) {
            const uint r = idx / BC128;
            const uint c = idx % BC128;
            float v = sS[r * BC128 + c] * softmax_scale;
            if (g_bw128_causal) {
                const uint gq = row0 + r;
                const uint gk = col0 + c;
                if (gk > gq) v = -INFINITY;
            }
            const float p = (v > -1e30f) ? exp(v - sLSE[r]) : 0.0f;
            sP[r * BC128 + c] = (half)p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* dV += P^T @ dO  (Bc × D += Bc × Br · Br × D) */
        for (uint kk = 0; kk < BR128; kk += 8) {
            simdgroup_matrix<half, 8, 8> p_t_frag[TM_K128];
            simdgroup_matrix<half, 8, 8> do_frag [TN_K128];
            for (uint i = 0; i < TM_K128; ++i) {
                const uint row = sg_row * 8 + i * 8;
                simdgroup_load(p_t_frag[i], sP + kk * BC128 + row, BC128, ulong2(0,0), true);
            }
            for (uint jj = 0; jj < TN_K128; ++jj) {
                const uint col = sg_col * (TN_K128 * 8) + jj * 8;
                simdgroup_load(do_frag[jj], sdO + kk * D + col, D);
            }
            for (uint i = 0; i < TM_K128; ++i)
                for (uint jj = 0; jj < TN_K128; ++jj)
                    simdgroup_multiply_accumulate(dV_acc[i][jj], p_t_frag[i], do_frag[jj], dV_acc[i][jj]);
        }

        simdgroup_matrix<float, 8, 8> dP_acc;
        dP_acc = simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint kk = 0; kk < D; kk += 8) {
            simdgroup_matrix<half, 8, 8> do_frag, v_frag;
            const uint row = sg_row * 8, col = sg_col * 8;
            simdgroup_load(do_frag, sdO + row * D + kk, D);
            simdgroup_load(v_frag, sV + col * D + kk, D, ulong2(0,0), true);
            simdgroup_multiply_accumulate(dP_acc, do_frag, v_frag, dP_acc);
        }
        {
            const uint sr = sg_row * 8, sc = sg_col * 8;
            simdgroup_store(dP_acc, sdP + sr * BC128 + sc, BC128);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = tid; idx < BR128 * BC128; idx += THREADS128) {
            const uint r = idx / BC128;
            const uint c = idx % BC128;
            const float p = (float)sP[r * BC128 + c];
            const float dp = sdP[r * BC128 + c];
            sP[r * BC128 + c] = (half)(p * (dp - sD_i[r]) * softmax_scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* dK += dS^T @ Q (Bc × D) */
        for (uint kk = 0; kk < BR128; kk += 8) {
            simdgroup_matrix<half, 8, 8> ds_t_frag[TM_K128];
            simdgroup_matrix<half, 8, 8> q_frag   [TN_K128];
            for (uint i = 0; i < TM_K128; ++i) {
                const uint row = sg_row * 8 + i * 8;
                simdgroup_load(ds_t_frag[i], sP + kk * BC128 + row, BC128, ulong2(0,0), true);
            }
            for (uint jj = 0; jj < TN_K128; ++jj) {
                const uint col = sg_col * (TN_K128 * 8) + jj * 8;
                simdgroup_load(q_frag[jj], sQ + kk * D + col, D);
            }
            for (uint i = 0; i < TM_K128; ++i)
                for (uint jj = 0; jj < TN_K128; ++jj)
                    simdgroup_multiply_accumulate(dK_acc[i][jj], ds_t_frag[i], q_frag[jj], dK_acc[i][jj]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float scratch[WM128 * WN128 * 64];
    threadgroup float* my = scratch + sgid * 64;

    for (uint i = 0; i < TM_K128; ++i) {
        for (uint j = 0; j < TN_K128; ++j) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(dK_acc[i][j], my, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint sr = sg_row * 8 + i * 8;
            const uint sc = sg_col * (TN_K128 * 8) + j * 8;
            for (uint k = 0; k < 64; k += 32) {
                const uint idx = k + slid;
                if (idx < 64) {
                    const uint lr = idx >> 3, lc = idx & 7;
                    const uint gr = col0 + sr + lr, gc = sc + lc;
                    if (gr < seq_kv) dK[dkv_off + gr * D + gc] = (half)my[idx];
                }
            }
        }
    }
    for (uint i = 0; i < TM_K128; ++i) {
        for (uint j = 0; j < TN_K128; ++j) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(dV_acc[i][j], my, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint sr = sg_row * 8 + i * 8;
            const uint sc = sg_col * (TN_K128 * 8) + j * 8;
            for (uint k = 0; k < 64; k += 32) {
                const uint idx = k + slid;
                if (idx < 64) {
                    const uint lr = idx >> 3, lc = idx & 7;
                    const uint gr = col0 + sr + lr, gc = sc + lc;
                    if (gr < seq_kv) dV[dkv_off + gr * D + gc] = (half)my[idx];
                }
            }
        }
    }
}
