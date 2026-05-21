/*
 * tensorcore — GEMM with simdgroup_async_copy (MFA-style).
 *
 * MFA reports 10-30% end-to-end win on attention; raw GEMM win is in the
 * 8-15% range vs sync vec4 loads. Pattern: ONE simdgroup (sidx==0) issues
 * async DMAs for A and B blocks, all simdgroups wait on a threadgroup_barrier,
 * compute proceeds. Compiler reorders compute instructions ahead of wait when
 * register pressure allows, giving the latency hiding.
 *
 * Single-buffered (matches MFA's actual production kernel). Threadgroup mem:
 *   BM=64, BN=64, BK=32, fp16 → 64*32*2 + 32*64*2 = 8 KB total. Comfortable.
 *
 * Compatibility: __asm("air.simdgroup_async_copy_2d.…") is supported by
 * Xcode 16.x. Newer Xcode rejects the __asm form; the kernel is gated by
 * CMake to skip on macOS 26+ until we ship the AIR-IR fallback.
 */

#include "metal_simdgroup_event.h"
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

constant constexpr uint ABM = 64;
constant constexpr uint ABN = 64;
constant constexpr uint ABK = 32;
constant constexpr uint AWM = 2;
constant constexpr uint AWN = 2;
constant constexpr uint ATM = 4;
constant constexpr uint ATN = 4;
constant constexpr uint ATHREADS = AWM * AWN * 32;   /* 128 */

kernel void tc_gemm_f16_f32_async(
    device const half*  A     [[buffer(0)]],
    device const half*  B     [[buffer(1)]],
    device       half*  C     [[buffer(2)]],
    constant uint& M          [[buffer(3)]],
    constant uint& N          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    constant float& alpha     [[buffer(6)]],
    constant float& beta      [[buffer(7)]],
    constant uint& lda        [[buffer(8)]],
    constant uint& ldb        [[buffer(9)]],
    constant uint& ldc        [[buffer(10)]],
    uint2 group_id            [[threadgroup_position_in_grid]],
    uint  sgid                [[simdgroup_index_in_threadgroup]],
    uint  slid                [[thread_index_in_simdgroup]])
{
    const uint baseRow = group_id.y * ABM;
    const uint baseCol = group_id.x * ABN;
    const uint sg_row = sgid / AWN;
    const uint sg_col = sgid % AWN;
    const uint tid = sgid * 32 + slid;
    (void)tid;

    threadgroup half sA[ABM * ABK];
    threadgroup half sB[ABK * ABN];

    simdgroup_matrix<float, 8, 8> acc[ATM][ATN];
    for (uint i = 0; i < ATM; ++i)
        for (uint j = 0; j < ATN; ++j)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    const bool full_tile_m = (baseRow + ABM <= M);
    const bool full_tile_n = (baseCol + ABN <= N);

    for (uint kBlock = 0; kBlock < K; kBlock += ABK) {
        const bool full_k = (kBlock + ABK <= K);
        const bool full_a = (full_tile_m && full_k);
        const bool full_b = (full_tile_n && full_k);

        /* Issue async DMAs from ONE simdgroup. */
        if (sgid == 0) {
            tc::simdgroup_event ev[2];
            if (full_a) {
                ev[0].async_copy<half>(
                    sA, /*dst_ld=*/ABK, ushort2(ABK, ABM),
                    A + baseRow * lda + kBlock, /*src_ld=*/lda, ushort2(ABK, ABM));
            }
            if (full_b) {
                ev[1].async_copy<half>(
                    sB, /*dst_ld=*/ABN, ushort2(ABN, ABK),
                    B + kBlock * ldb + baseCol, /*src_ld=*/ldb, ushort2(ABN, ABK));
            }
            tc::simdgroup_event::wait((full_a && full_b) ? 2 : (full_a || full_b) ? 1 : 0,
                                       ev);
        }

        /* Boundary tiles: fall back to sync scalar load. */
        if (!full_a || !full_b) {
            for (uint idx = tid; idx < ABM * ABK; idx += ATHREADS) {
                const uint row = idx / ABK;
                const uint col = idx % ABK;
                const uint gr = baseRow + row;
                const uint gc = kBlock + col;
                sA[row * ABK + col] = (gr < M && gc < K) ? A[gr * lda + gc] : half(0);
            }
            for (uint idx = tid; idx < ABK * ABN; idx += ATHREADS) {
                const uint row = idx / ABN;
                const uint col = idx % ABN;
                const uint gr = kBlock + row;
                const uint gc = baseCol + col;
                sB[row * ABN + col] = (gr < K && gc < N) ? B[gr * ldb + gc] : half(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Compute. */
        for (uint kk = 0; kk < ABK; kk += 8) {
            simdgroup_matrix<half, 8, 8> a_frag[ATM];
            simdgroup_matrix<half, 8, 8> b_frag[ATN];
            for (uint i = 0; i < ATM; ++i) {
                const uint row = sg_row * (ATM * 8) + i * 8;
                simdgroup_load(a_frag[i], sA + row * ABK + kk, ABK);
            }
            for (uint j = 0; j < ATN; ++j) {
                const uint col = sg_col * (ATN * 8) + j * 8;
                simdgroup_load(b_frag[j], sB + kk * ABN + col, ABN);
            }
            for (uint i = 0; i < ATM; ++i)
                for (uint j = 0; j < ATN; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* Store. */
    threadgroup float scratch[AWM * AWN * 64];
    threadgroup float* slot = scratch + sgid * 64;
    for (uint i = 0; i < ATM; ++i) {
        for (uint j = 0; j < ATN; ++j) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(acc[i][j], slot, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint gRow = baseRow + sg_row * (ATM * 8) + i * 8;
            const uint gCol = baseCol + sg_col * (ATN * 8) + j * 8;
            for (uint k = 0; k < 64; k += 32) {
                const uint idx = k + slid;
                if (idx < 64) {
                    const uint r = idx >> 3;
                    const uint c = idx & 7;
                    const uint Gr = gRow + r;
                    const uint Gc = gCol + c;
                    if (Gr < M && Gc < N) {
                        float s = slot[r * 8 + c] * alpha;
                        if (beta != 0.0f) s += (float)C[Gr * ldc + Gc] * beta;
                        C[Gr * ldc + Gc] = (half)s;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
