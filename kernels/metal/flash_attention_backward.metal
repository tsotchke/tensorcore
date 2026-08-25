/*
 * tensorcore — FlashAttention-2 backward pass.
 *
 * Given (Q, K, V, O, LSE, dO), compute (dQ, dK, dV).
 *
 * Split into two kernels to avoid cross-block atomic accumulation:
 *   - tc_flash_attention_backward_dq      : one TG per query block; iterates
 *                                            all KV blocks; writes dQ block.
 *   - tc_flash_attention_backward_dk_dv   : one TG per KV block; iterates
 *                                            all Q blocks; writes dK, dV block.
 *
 * v0.1 of the backward: Br=Bc=32, D=64, fp16 IO, fp32 accumulators. Larger
 * tile sizes and D=128 land in v0.2 alongside the forward's Br=64.
 *
 * Algorithm (Dao 2023 FA-2 backward):
 *   S_ij  = Q_i @ K_j^T * scale
 *   P_ij  = exp(S_ij - LSE_i)
 *   D_i   = sum(dO_i * O_i, dim=-1)
 *   dV_j += P_ij^T @ dO_i
 *   dP_ij = dO_i @ V_j^T
 *   dS_ij = P_ij * (dP_ij - D_i)
 *   dQ_i += dS_ij @ K_j * scale
 *   dK_j += dS_ij^T @ Q_i * scale
 */

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

constant constexpr uint BR = 32;
constant constexpr uint BC = 32;
constant constexpr uint D  = 64;
constant constexpr uint WM = 2;
constant constexpr uint WN = 2;
constant constexpr uint THREADS = WM * WN * 32;     /* 128 */

constant constexpr uint TM_S = BR / WM / 8;          /* 2  */
constant constexpr uint TN_S = BC / WN / 8;          /* 2  */
constant constexpr uint TM_O = BR / WM / 8;          /* 2  */
constant constexpr uint TN_O = D  / WN / 8;          /* 4  */
constant constexpr uint TM_K = BC / WM / 8;          /* 2  */
constant constexpr uint TN_K = D  / WN / 8;          /* 4  */

constant bool g_bw_causal [[function_constant(0)]];

template <typename T>
inline void coop_load(threadgroup T*       dst, uint dst_stride,
                      device   const T*   src, uint src_stride,
                      uint row0, uint col0,
                      uint rows, uint cols,
                      uint row_limit, uint col_limit,
                      uint tid)
{
    const uint n = rows * cols;
    for (uint idx = tid; idx < n; idx += THREADS) {
        const uint r  = idx / cols;
        const uint c  = idx % cols;
        const uint gr = row0 + r;
        const uint gc = col0 + c;
        dst[r * dst_stride + c] =
            (gr < row_limit && gc < col_limit) ? src[gr * src_stride + gc] : T(0);
    }
}

/* ========================================================================== *
 *  Kernel A: dQ                                                               *
 * ========================================================================== */
kernel void tc_flash_attention_backward_dq(
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
    const uint q_block_idx = group_id.x;
    const uint head_idx    = group_id.y;
    const uint batch_idx   = group_id.z;
    const uint kv_head_idx = (kv_heads > 0 && kv_heads != heads)
                             ? (head_idx * kv_heads / heads) : head_idx;

    const uint row0 = q_block_idx * BR;
    if (row0 >= seq_q) return;

    const uint q_off   = ((batch_idx * heads    + head_idx)    * seq_q  + 0) * D;
    const uint kv_kbase= ((batch_idx * kv_heads + kv_head_idx) * seq_kv + 0) * D;
    const uint o_off   = q_off;
    const uint do_off  = q_off;
    const uint dq_off  = q_off;
    const uint lse_off = ((batch_idx * heads    + head_idx)    * seq_q  + 0);

    const uint tid = sgid * 32 + slid;
    const uint sg_row = sgid / WN;
    const uint sg_col = sgid % WN;

    /* TG memory layout — fits ~28 KB. */
    threadgroup half  sQ [BR * D];
    threadgroup half  sO [BR * D];
    threadgroup half  sdO[BR * D];
    threadgroup float sLSE[BR];
    threadgroup float sD_i[BR];
    threadgroup half  sK [BC * D];
    threadgroup half  sV [BC * D];
    threadgroup float sS [BR * BC];
    threadgroup half  sP [BR * BC];
    threadgroup float sdP[BR * BC];

    /* Load Q, O, dO, LSE once. */
    coop_load<half>(sQ,  D, Q  + q_off,  D, row0, 0, BR, D, seq_q, D, tid);
    coop_load<half>(sO,  D, O  + o_off,  D, row0, 0, BR, D, seq_q, D, tid);
    coop_load<half>(sdO, D, dO + do_off, D, row0, 0, BR, D, seq_q, D, tid);
    for (uint idx = tid; idx < BR; idx += THREADS) {
        const uint gr = row0 + idx;
        sLSE[idx] = (gr < seq_q) ? LSE[lse_off + gr] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* D_i = rowsum(dO * O). */
    if (tid < BR) {
        float acc = 0.0f;
        for (uint d = 0; d < D; ++d) {
            acc += (float)sdO[tid * D + d] * (float)sO[tid * D + d];
        }
        sD_i[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* dQ_acc fp32 in registers via simdgroup_matrix. */
    simdgroup_matrix<float, 8, 8> dQ_acc[TM_O][TN_O];
    for (uint i = 0; i < TM_O; ++i)
        for (uint j = 0; j < TN_O; ++j)
            dQ_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    const uint Tc = (seq_kv + BC - 1) / BC;
    for (uint j = 0; j < Tc; ++j) {
        const uint kv_col0 = j * BC;
        if (g_bw_causal && (kv_col0 > row0 + BR - 1)) break;

        coop_load<half>(sK, D, K + kv_kbase, D, kv_col0, 0, BC, D, seq_kv, D, tid);
        coop_load<half>(sV, D, V + kv_kbase, D, kv_col0, 0, BC, D, seq_kv, D, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* S = Q @ K^T (Br × Bc) — fp32 accum. */
        simdgroup_matrix<float, 8, 8> S_acc[TM_S][TN_S];
        for (uint i = 0; i < TM_S; ++i)
            for (uint jj = 0; jj < TN_S; ++jj)
                S_acc[i][jj] = simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint kk = 0; kk < D; kk += 8) {
            simdgroup_matrix<half, 8, 8> q_frag[TM_S];
            simdgroup_matrix<half, 8, 8> k_frag[TN_S];
            for (uint i = 0; i < TM_S; ++i) {
                const uint row = sg_row * (TM_S * 8) + i * 8;
                simdgroup_load(q_frag[i], sQ + row * D + kk, D);
            }
            for (uint jj = 0; jj < TN_S; ++jj) {
                const uint col = sg_col * (TN_S * 8) + jj * 8;
                simdgroup_load(k_frag[jj], sK + col * D + kk, D, ulong2(0,0), true);
            }
            for (uint i = 0; i < TM_S; ++i)
                for (uint jj = 0; jj < TN_S; ++jj)
                    simdgroup_multiply_accumulate(S_acc[i][jj], q_frag[i], k_frag[jj], S_acc[i][jj]);
        }
        /* Spill S to TG memory. */
        for (uint i = 0; i < TM_S; ++i) {
            for (uint jj = 0; jj < TN_S; ++jj) {
                const uint sr = sg_row * (TM_S * 8) + i * 8;
                const uint sc = sg_col * (TN_S * 8) + jj * 8;
                simdgroup_store(S_acc[i][jj], sS + sr * BC + sc, BC);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* P = exp(S * scale - LSE), with causal mask. */
        for (uint idx = tid; idx < BR * BC; idx += THREADS) {
            const uint r = idx / BC;
            const uint c = idx % BC;
            float v = sS[r * BC + c] * softmax_scale;
            if (g_bw_causal) {
                const uint gq = row0 + r;
                const uint gk = kv_col0 + c;
                if (gk > gq) v = -INFINITY;
            }
            const float p = (v > -1e30f) ? exp(v - sLSE[r]) : 0.0f;
            sP[r * BC + c] = (half)p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* dP = dO @ V^T (Br × Bc). */
        simdgroup_matrix<float, 8, 8> dP_acc[TM_S][TN_S];
        for (uint i = 0; i < TM_S; ++i)
            for (uint jj = 0; jj < TN_S; ++jj)
                dP_acc[i][jj] = simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint kk = 0; kk < D; kk += 8) {
            simdgroup_matrix<half, 8, 8> dO_frag[TM_S];
            simdgroup_matrix<half, 8, 8> v_frag[TN_S];
            for (uint i = 0; i < TM_S; ++i) {
                const uint row = sg_row * (TM_S * 8) + i * 8;
                simdgroup_load(dO_frag[i], sdO + row * D + kk, D);
            }
            for (uint jj = 0; jj < TN_S; ++jj) {
                const uint col = sg_col * (TN_S * 8) + jj * 8;
                simdgroup_load(v_frag[jj], sV + col * D + kk, D, ulong2(0,0), true);
            }
            for (uint i = 0; i < TM_S; ++i)
                for (uint jj = 0; jj < TN_S; ++jj)
                    simdgroup_multiply_accumulate(dP_acc[i][jj], dO_frag[i], v_frag[jj], dP_acc[i][jj]);
        }
        for (uint i = 0; i < TM_S; ++i) {
            for (uint jj = 0; jj < TN_S; ++jj) {
                const uint sr = sg_row * (TM_S * 8) + i * 8;
                const uint sc = sg_col * (TN_S * 8) + jj * 8;
                simdgroup_store(dP_acc[i][jj], sdP + sr * BC + sc, BC);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* dS = P * (dP - D_i) — store back into sP as fp16 since we'll
         * matmul dS @ K below. Scaled by softmax_scale at use. */
        for (uint idx = tid; idx < BR * BC; idx += THREADS) {
            const uint r = idx / BC;
            const uint c = idx % BC;
            const float p = (float)sP[r * BC + c];
            const float dp = sdP[r * BC + c];
            const float ds = p * (dp - sD_i[r]) * softmax_scale;
            sP[r * BC + c] = (half)ds;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* dQ += dS @ K. */
        for (uint kk = 0; kk < BC; kk += 8) {
            simdgroup_matrix<half, 8, 8> ds_frag[TM_O];
            simdgroup_matrix<half, 8, 8> k_frag [TN_O];
            for (uint i = 0; i < TM_O; ++i) {
                const uint row = sg_row * (TM_O * 8) + i * 8;
                simdgroup_load(ds_frag[i], sP + row * BC + kk, BC);
            }
            for (uint jj = 0; jj < TN_O; ++jj) {
                const uint col = sg_col * (TN_O * 8) + jj * 8;
                simdgroup_load(k_frag[jj], sK + kk * D + col, D);
            }
            for (uint i = 0; i < TM_O; ++i)
                for (uint jj = 0; jj < TN_O; ++jj)
                    simdgroup_multiply_accumulate(dQ_acc[i][jj], ds_frag[i], k_frag[jj], dQ_acc[i][jj]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* Write dQ block. */
    threadgroup float scratch[WM * WN * 64];
    threadgroup float* my = scratch + sgid * 64;
    for (uint i = 0; i < TM_O; ++i) {
        for (uint jj = 0; jj < TN_O; ++jj) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(dQ_acc[i][jj], my, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint sr = sg_row * (TM_O * 8) + i * 8;
            const uint sc = sg_col * (TN_O * 8) + jj * 8;
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

/* ========================================================================== *
 *  Kernel B: dK, dV                                                           *
 * ========================================================================== */
kernel void tc_flash_attention_backward_dk_dv(
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
    const uint kv_block_idx = group_id.x;
    const uint head_idx     = group_id.y;
    const uint batch_idx    = group_id.z;
    const uint kv_head_idx  = (kv_heads > 0 && kv_heads != heads)
                              ? (head_idx * kv_heads / heads) : head_idx;

    const uint col0 = kv_block_idx * BC;
    if (col0 >= seq_kv) return;

    const uint q_off    = ((batch_idx * heads    + head_idx)    * seq_q  + 0) * D;
    const uint kv_kbase = ((batch_idx * kv_heads + kv_head_idx) * seq_kv + 0) * D;
    const uint dkv_off  = kv_kbase;
    const uint o_off    = q_off;
    const uint do_off   = q_off;
    const uint lse_off  = ((batch_idx * heads    + head_idx)    * seq_q  + 0);

    const uint tid = sgid * 32 + slid;
    const uint sg_row = sgid / WN;
    const uint sg_col = sgid % WN;

    threadgroup half  sK [BC * D];
    threadgroup half  sV [BC * D];
    threadgroup half  sQ [BR * D];
    threadgroup half  sO [BR * D];
    threadgroup half  sdO[BR * D];
    threadgroup float sLSE[BR];
    threadgroup float sD_i[BR];
    threadgroup float sS [BR * BC];
    threadgroup half  sP [BR * BC];
    threadgroup float sdP[BR * BC];

    coop_load<half>(sK, D, K + kv_kbase, D, col0, 0, BC, D, seq_kv, D, tid);
    coop_load<half>(sV, D, V + kv_kbase, D, col0, 0, BC, D, seq_kv, D, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* dK_acc and dV_acc — both (Bc × D), fp32 accumulators in registers. */
    simdgroup_matrix<float, 8, 8> dK_acc[TM_K][TN_K];
    simdgroup_matrix<float, 8, 8> dV_acc[TM_K][TN_K];
    for (uint i = 0; i < TM_K; ++i)
        for (uint j = 0; j < TN_K; ++j) {
            dK_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
            dV_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
        }

    const uint Tr = (seq_q + BR - 1) / BR;
    for (uint i_blk = 0; i_blk < Tr; ++i_blk) {
        const uint row0 = i_blk * BR;
        if (g_bw_causal && (row0 + BR - 1 < col0)) continue;   /* fully masked */

        coop_load<half>(sQ,  D, Q  + q_off,  D, row0, 0, BR, D, seq_q, D, tid);
        coop_load<half>(sO,  D, O  + o_off,  D, row0, 0, BR, D, seq_q, D, tid);
        coop_load<half>(sdO, D, dO + do_off, D, row0, 0, BR, D, seq_q, D, tid);
        for (uint idx = tid; idx < BR; idx += THREADS) {
            const uint gr = row0 + idx;
            sLSE[idx] = (gr < seq_q) ? LSE[lse_off + gr] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < BR) {
            float acc = 0.0f;
            for (uint d = 0; d < D; ++d)
                acc += (float)sdO[tid * D + d] * (float)sO[tid * D + d];
            sD_i[tid] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* S = Q @ K^T. */
        simdgroup_matrix<float, 8, 8> S_acc[TM_S][TN_S];
        for (uint a = 0; a < TM_S; ++a)
            for (uint b = 0; b < TN_S; ++b)
                S_acc[a][b] = simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint kk = 0; kk < D; kk += 8) {
            simdgroup_matrix<half, 8, 8> q_frag[TM_S];
            simdgroup_matrix<half, 8, 8> k_frag[TN_S];
            for (uint a = 0; a < TM_S; ++a) {
                const uint row = sg_row * (TM_S * 8) + a * 8;
                simdgroup_load(q_frag[a], sQ + row * D + kk, D);
            }
            for (uint b = 0; b < TN_S; ++b) {
                const uint col = sg_col * (TN_S * 8) + b * 8;
                simdgroup_load(k_frag[b], sK + col * D + kk, D, ulong2(0,0), true);
            }
            for (uint a = 0; a < TM_S; ++a)
                for (uint b = 0; b < TN_S; ++b)
                    simdgroup_multiply_accumulate(S_acc[a][b], q_frag[a], k_frag[b], S_acc[a][b]);
        }
        for (uint a = 0; a < TM_S; ++a) {
            for (uint b = 0; b < TN_S; ++b) {
                const uint sr = sg_row * (TM_S * 8) + a * 8;
                const uint sc = sg_col * (TN_S * 8) + b * 8;
                simdgroup_store(S_acc[a][b], sS + sr * BC + sc, BC);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* P = exp(S*scale - LSE), causal. */
        for (uint idx = tid; idx < BR * BC; idx += THREADS) {
            const uint r = idx / BC;
            const uint c = idx % BC;
            float v = sS[r * BC + c] * softmax_scale;
            if (g_bw_causal) {
                const uint gq = row0 + r;
                const uint gk = col0 + c;
                if (gk > gq) v = -INFINITY;
            }
            const float p = (v > -1e30f) ? exp(v - sLSE[r]) : 0.0f;
            sP[r * BC + c] = (half)p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* dV_j += P^T @ dO. (Bc × D) += (Bc × Br) @ (Br × D). */
        for (uint kk = 0; kk < BR; kk += 8) {
            simdgroup_matrix<half, 8, 8> p_t_frag[TM_K];
            simdgroup_matrix<half, 8, 8> do_frag [TN_K];
            for (uint a = 0; a < TM_K; ++a) {
                const uint row = sg_row * (TM_K * 8) + a * 8;
                /* P^T: read P with transpose=true; row of P^T = col of P. */
                simdgroup_load(p_t_frag[a], sP + kk * BC + row, BC, ulong2(0,0), true);
            }
            for (uint b = 0; b < TN_K; ++b) {
                const uint col = sg_col * (TN_K * 8) + b * 8;
                simdgroup_load(do_frag[b], sdO + kk * D + col, D);
            }
            for (uint a = 0; a < TM_K; ++a)
                for (uint b = 0; b < TN_K; ++b)
                    simdgroup_multiply_accumulate(dV_acc[a][b], p_t_frag[a], do_frag[b], dV_acc[a][b]);
        }

        /* dP = dO @ V^T. */
        simdgroup_matrix<float, 8, 8> dP_acc[TM_S][TN_S];
        for (uint a = 0; a < TM_S; ++a)
            for (uint b = 0; b < TN_S; ++b)
                dP_acc[a][b] = simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint kk = 0; kk < D; kk += 8) {
            simdgroup_matrix<half, 8, 8> do_frag[TM_S];
            simdgroup_matrix<half, 8, 8> v_frag [TN_S];
            for (uint a = 0; a < TM_S; ++a) {
                const uint row = sg_row * (TM_S * 8) + a * 8;
                simdgroup_load(do_frag[a], sdO + row * D + kk, D);
            }
            for (uint b = 0; b < TN_S; ++b) {
                const uint col = sg_col * (TN_S * 8) + b * 8;
                simdgroup_load(v_frag[b], sV + col * D + kk, D, ulong2(0,0), true);
            }
            for (uint a = 0; a < TM_S; ++a)
                for (uint b = 0; b < TN_S; ++b)
                    simdgroup_multiply_accumulate(dP_acc[a][b], do_frag[a], v_frag[b], dP_acc[a][b]);
        }
        for (uint a = 0; a < TM_S; ++a) {
            for (uint b = 0; b < TN_S; ++b) {
                const uint sr = sg_row * (TM_S * 8) + a * 8;
                const uint sc = sg_col * (TN_S * 8) + b * 8;
                simdgroup_store(dP_acc[a][b], sdP + sr * BC + sc, BC);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* dS = P * (dP - D_i) * scale, stored as half in sP. */
        for (uint idx = tid; idx < BR * BC; idx += THREADS) {
            const uint r = idx / BC;
            const uint c = idx % BC;
            const float p = (float)sP[r * BC + c];
            const float dp = sdP[r * BC + c];
            sP[r * BC + c] = (half)(p * (dp - sD_i[r]) * softmax_scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* dK_j += dS^T @ Q. (Bc × D) += (Bc × Br) @ (Br × D). */
        for (uint kk = 0; kk < BR; kk += 8) {
            simdgroup_matrix<half, 8, 8> ds_t_frag[TM_K];
            simdgroup_matrix<half, 8, 8> q_frag   [TN_K];
            for (uint a = 0; a < TM_K; ++a) {
                const uint row = sg_row * (TM_K * 8) + a * 8;
                simdgroup_load(ds_t_frag[a], sP + kk * BC + row, BC, ulong2(0,0), true);
            }
            for (uint b = 0; b < TN_K; ++b) {
                const uint col = sg_col * (TN_K * 8) + b * 8;
                simdgroup_load(q_frag[b], sQ + kk * D + col, D);
            }
            for (uint a = 0; a < TM_K; ++a)
                for (uint b = 0; b < TN_K; ++b)
                    simdgroup_multiply_accumulate(dK_acc[a][b], ds_t_frag[a], q_frag[b], dK_acc[a][b]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* Write dK and dV. */
    threadgroup float scratch[WM * WN * 64];
    threadgroup float* my = scratch + sgid * 64;

    /* dK */
    for (uint i = 0; i < TM_K; ++i) {
        for (uint j = 0; j < TN_K; ++j) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(dK_acc[i][j], my, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint sr = sg_row * (TM_K * 8) + i * 8;
            const uint sc = sg_col * (TN_K * 8) + j * 8;
            for (uint k = 0; k < 64; k += 32) {
                const uint idx = k + slid;
                if (idx < 64) {
                    const uint lr = idx >> 3;
                    const uint lc = idx & 7;
                    const uint gr = col0 + sr + lr;
                    const uint gc = sc + lc;
                    if (gr < seq_kv) {
                        dK[dkv_off + gr * D + gc] = (half)my[idx];
                    }
                }
            }
        }
    }
    /* dV */
    for (uint i = 0; i < TM_K; ++i) {
        for (uint j = 0; j < TN_K; ++j) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(dV_acc[i][j], my, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint sr = sg_row * (TM_K * 8) + i * 8;
            const uint sc = sg_col * (TN_K * 8) + j * 8;
            for (uint k = 0; k < 64; k += 32) {
                const uint idx = k + slid;
                if (idx < 64) {
                    const uint lr = idx >> 3;
                    const uint lc = idx & 7;
                    const uint gr = col0 + sr + lr;
                    const uint gc = sc + lc;
                    if (gr < seq_kv) {
                        dV[dkv_off + gr * D + gc] = (half)my[idx];
                    }
                }
            }
        }
    }
}
