/*
 * tensorcore — 128×128 GEMM with simdgroup_async_copy.
 *
 * Register-pressure tuned: 16 simdgroups (4×4 grid), 512 threads, each
 * simdgroup owns 32×32 of the output = TM=4, TN=4 of 8×8 fragments. That's
 * 16 frags/sg — same as the v0.1 attempt, but with async DMA instead of sync
 * vec4 loads the compiler has more room to schedule arithmetic between async
 * issue and wait.
 *
 * TG memory at BK=8:
 *   sA: 128×8  = 1024 half = 2 KB
 *   sB: 8×128  = 1024 half = 2 KB
 *   Spill: 16 sg × 64 fp32 = 4 KB
 *   Total: ~8 KB, well within 32 KB budget.
 *
 * Math: 128² × 8 = 131072 MAC per K-block / 4 KB loaded = 32 MAC/B.
 *       Same arithmetic intensity as 64x64 BK=32, but each loaded element is
 *       reused across 128 rows/cols instead of 64. Memory traffic per FLOP
 *       halves for the steady-state large-shape case.
 */

#include "metal_simdgroup_event.h"
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

constant constexpr uint AS128_BM = 128;
constant constexpr uint AS128_BN = 128;
constant constexpr uint AS128_BK = 8;
constant constexpr uint AS128_WM = 4;
constant constexpr uint AS128_WN = 4;
constant constexpr uint AS128_TM = 4;
constant constexpr uint AS128_TN = 4;
constant constexpr uint AS128_THREADS = AS128_WM * AS128_WN * 32;   /* 512 */

kernel void tc_gemm_f16_f32_async_128(
    device const half*  A     [[buffer(0)]],
    device const half*  B     [[buffer(1)]],
    device       half*  C     [[buffer(2)]],
    constant uint& M          [[buffer(3)]],
    constant uint& N          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    constant float& alpha     [[buffer(6)]],
    constant float& beta      [[buffer(7)]],
    uint2 group_id            [[threadgroup_position_in_grid]],
    uint  sgid                [[simdgroup_index_in_threadgroup]],
    uint  slid                [[thread_index_in_simdgroup]])
{
    const uint baseRow = group_id.y * AS128_BM;
    const uint baseCol = group_id.x * AS128_BN;
    const uint sg_row = sgid / AS128_WN;
    const uint sg_col = sgid % AS128_WN;

    threadgroup half sA[AS128_BM * AS128_BK];
    threadgroup half sB[AS128_BK * AS128_BN];

    simdgroup_matrix<float, 8, 8> acc[AS128_TM][AS128_TN];
    for (uint i = 0; i < AS128_TM; ++i)
        for (uint j = 0; j < AS128_TN; ++j)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    const bool full_m = (baseRow + AS128_BM <= M);
    const bool full_n = (baseCol + AS128_BN <= N);

    for (uint kBlock = 0; kBlock < K; kBlock += AS128_BK) {
        const bool full_k = (kBlock + AS128_BK <= K);
        const bool full_tile = full_m && full_n && full_k;

        if (full_tile) {
            /* Async DMA issue from sgid==0 only. */
            if (sgid == 0) {
                tc::simdgroup_event ev[2];
                ev[0].async_copy<half>(
                    sA, /*dst_ld=*/AS128_BK, ushort2(AS128_BK, AS128_BM),
                    A + baseRow * K + kBlock, /*src_ld=*/K, ushort2(AS128_BK, AS128_BM));
                ev[1].async_copy<half>(
                    sB, /*dst_ld=*/AS128_BN, ushort2(AS128_BN, AS128_BK),
                    B + kBlock * N + baseCol, /*src_ld=*/N, ushort2(AS128_BN, AS128_BK));
                tc::simdgroup_event::wait(2, ev);
            }
        } else {
            /* Boundary fallback: sync scalar load by all threads. */
            const uint tid = sgid * 32 + slid;
            for (uint idx = tid; idx < AS128_BM * AS128_BK; idx += AS128_THREADS) {
                const uint row = idx / AS128_BK;
                const uint col = idx % AS128_BK;
                const uint gr = baseRow + row;
                const uint gc = kBlock + col;
                sA[row * AS128_BK + col] = (gr < M && gc < K) ? A[gr * K + gc] : half(0);
            }
            for (uint idx = tid; idx < AS128_BK * AS128_BN; idx += AS128_THREADS) {
                const uint row = idx / AS128_BN;
                const uint col = idx % AS128_BN;
                const uint gr = kBlock + row;
                const uint gc = baseCol + col;
                sB[row * AS128_BN + col] = (gr < K && gc < N) ? B[gr * N + gc] : half(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Inner K-loop: BK=8, single MMA pass per K-block. */
        {
            simdgroup_matrix<half, 8, 8> a_frag[AS128_TM];
            simdgroup_matrix<half, 8, 8> b_frag[AS128_TN];
            for (uint i = 0; i < AS128_TM; ++i) {
                const uint row = sg_row * (AS128_TM * 8) + i * 8;
                simdgroup_load(a_frag[i], sA + row * AS128_BK, AS128_BK);
            }
            for (uint j = 0; j < AS128_TN; ++j) {
                const uint col = sg_col * (AS128_TN * 8) + j * 8;
                simdgroup_load(b_frag[j], sB + col, AS128_BN);
            }
            for (uint i = 0; i < AS128_TM; ++i)
                for (uint j = 0; j < AS128_TN; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* Epilogue: spill + cooperative write, alpha/beta scaling. */
    threadgroup float scratch[AS128_WM * AS128_WN * 64];
    threadgroup float* slot = scratch + sgid * 64;
    for (uint i = 0; i < AS128_TM; ++i) {
        for (uint j = 0; j < AS128_TN; ++j) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(acc[i][j], slot, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint gRow = baseRow + sg_row * (AS128_TM * 8) + i * 8;
            const uint gCol = baseCol + sg_col * (AS128_TN * 8) + j * 8;
            for (uint k = 0; k < 64; k += 32) {
                const uint idx = k + slid;
                if (idx < 64) {
                    const uint r = idx >> 3;
                    const uint c = idx & 7;
                    const uint Gr = gRow + r;
                    const uint Gc = gCol + c;
                    if (Gr < M && Gc < N) {
                        float s = slot[r * 8 + c] * alpha;
                        if (beta != 0.0f) s += (float)C[Gr * N + Gc] * beta;
                        C[Gr * N + Gc] = (half)s;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
