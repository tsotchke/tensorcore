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
constant constexpr uint ABK = 32;    /* ABK=64 was tried (2× arithmetic intensity)
                                       but reduced occupancy via larger threadgroup
                                       memory: 16.91 vs 19.37 TFLOPS. Kept at 32. */
constant constexpr uint AWM = 2;
constant constexpr uint AWN = 2;
constant constexpr uint ATM = 4;
constant constexpr uint ATN = 4;
constant constexpr uint ATHREADS = AWM * AWN * 32;   /* 128 */

/* Double-buffered async variant: pipelines memory load with compute. While
 * one simdgroup is loading K[next] into buffer 1, the other simdgroups
 * compute K[curr] from buffer 0. Should overlap memory latency with
 * arithmetic, closing the ~30 TFLOPS gap to the 54 TFLOPS M2 Ultra peak. */
constant constexpr uint DBABM = 64;
constant constexpr uint DBABN = 64;
constant constexpr uint DBABK = 32;
constant constexpr uint DBAWM = 2;
constant constexpr uint DBAWN = 2;
constant constexpr uint DBATM = 4;
constant constexpr uint DBATN = 4;
constant constexpr uint DBATHREADS = DBAWM * DBAWN * 32;   /* 128 */

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

/* ---------------------------------------------------------------------- *
 * Double-buffered async kernel.
 *
 * Pipelines memory load with compute by maintaining two threadgroup-mem
 * buffer banks (sA0/sA1 and sB0/sB1). While compute consumes bank `r`,
 * the async DMA loads K[k+1] into bank `1-r`. On the next iteration we
 * swap banks. This overlaps memory latency (~hundreds of cycles) with
 * arithmetic, closing the gap to the M2 Ultra fp16 peak.
 *
 * Threadgroup memory: 2 × (64 × 32 × 2 bytes) = 8 KB; well under the
 * 32 KB per-threadgroup limit (occupancy preserved).
 * ---------------------------------------------------------------------- */
kernel void tc_gemm_f16_f32_async_db(
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
    const uint baseRow = group_id.y * DBABM;
    const uint baseCol = group_id.x * DBABN;
    const uint sg_row = sgid / DBAWN;
    const uint sg_col = sgid % DBAWN;
    const uint tid = sgid * 32 + slid;

    /* Two buffer banks for ping-pong. */
    threadgroup half sA[2][DBABM * DBABK];
    threadgroup half sB[2][DBABK * DBABN];

    simdgroup_matrix<float, 8, 8> acc[DBATM][DBATN];
    for (uint i = 0; i < DBATM; ++i)
        for (uint j = 0; j < DBATN; ++j)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    const bool full_tile_m = (baseRow + DBABM <= M);
    const bool full_tile_n = (baseCol + DBABN <= N);
    const uint n_k_blocks = (K + DBABK - 1) / DBABK;

    /* Macro inlines for Metal (no lambdas). */
#define DB_LOAD_BLOCK(kBlock, bank) do {                                          \
        const bool _full_k = ((kBlock) + DBABK <= K);                            \
        const bool _full_a = (full_tile_m && _full_k);                           \
        const bool _full_b = (full_tile_n && _full_k);                           \
        if (sgid == 0) {                                                          \
            tc::simdgroup_event _ev[2];                                           \
            if (_full_a) {                                                        \
                _ev[0].async_copy<half>(                                          \
                    sA[bank], DBABK, ushort2(DBABK, DBABM),                       \
                    A + baseRow * lda + (kBlock), lda, ushort2(DBABK, DBABM));    \
            }                                                                     \
            if (_full_b) {                                                        \
                _ev[1].async_copy<half>(                                          \
                    sB[bank], DBABN, ushort2(DBABN, DBABK),                       \
                    B + (kBlock) * ldb + baseCol, ldb, ushort2(DBABN, DBABK));    \
            }                                                                     \
            tc::simdgroup_event::wait(                                            \
                (_full_a && _full_b) ? 2 : (_full_a || _full_b) ? 1 : 0, _ev);   \
        }                                                                         \
        if (!_full_a || !_full_b) {                                               \
            for (uint _idx = tid; _idx < DBABM * DBABK; _idx += DBATHREADS) {    \
                const uint _row = _idx / DBABK;                                   \
                const uint _col = _idx % DBABK;                                   \
                const uint _gr = baseRow + _row;                                  \
                const uint _gc = (kBlock) + _col;                                 \
                sA[bank][_row * DBABK + _col] = (_gr < M && _gc < K) ? A[_gr * lda + _gc] : half(0); \
            }                                                                     \
            for (uint _idx = tid; _idx < DBABK * DBABN; _idx += DBATHREADS) {    \
                const uint _row = _idx / DBABN;                                   \
                const uint _col = _idx % DBABN;                                   \
                const uint _gr = (kBlock) + _row;                                 \
                const uint _gc = baseCol + _col;                                  \
                sB[bank][_row * DBABN + _col] = (_gr < K && _gc < N) ? B[_gr * ldb + _gc] : half(0); \
            }                                                                     \
        }                                                                         \
    } while (0)

#define DB_COMPUTE_BLOCK(bank) do {                                               \
        for (uint _kk = 0; _kk < DBABK; _kk += 8) {                              \
            simdgroup_matrix<half, 8, 8> _a_frag[DBATM];                          \
            simdgroup_matrix<half, 8, 8> _b_frag[DBATN];                          \
            for (uint _i = 0; _i < DBATM; ++_i) {                                 \
                const uint _row = sg_row * (DBATM * 8) + _i * 8;                  \
                simdgroup_load(_a_frag[_i], sA[bank] + _row * DBABK + _kk, DBABK);\
            }                                                                     \
            for (uint _j = 0; _j < DBATN; ++_j) {                                 \
                const uint _col = sg_col * (DBATN * 8) + _j * 8;                  \
                simdgroup_load(_b_frag[_j], sB[bank] + _kk * DBABN + _col, DBABN);\
            }                                                                     \
            for (uint _i = 0; _i < DBATM; ++_i)                                   \
                for (uint _j = 0; _j < DBATN; ++_j)                               \
                    simdgroup_multiply_accumulate(acc[_i][_j], _a_frag[_i], _b_frag[_j], acc[_i][_j]); \
        }                                                                         \
    } while (0)

    /* Prime: load K[0] into bank 0. */
    DB_LOAD_BLOCK(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Main loop: while compute uses bank `bank`, async-load next iter into 1-bank. */
    uint bank = 0;
    for (uint k = 1; k < n_k_blocks; ++k) {
        const uint next_bank = 1 - bank;
        DB_LOAD_BLOCK(k * DBABK, next_bank);
        DB_COMPUTE_BLOCK(bank);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bank = next_bank;
    }
    DB_COMPUTE_BLOCK(bank);
    threadgroup_barrier(mem_flags::mem_threadgroup);

#undef DB_LOAD_BLOCK
#undef DB_COMPUTE_BLOCK

    /* Store (same as the single-buffer variant). */
    threadgroup float scratch[DBAWM * DBAWN * 64];
    threadgroup float* slot = scratch + sgid * 64;
    for (uint i = 0; i < DBATM; ++i) {
        for (uint j = 0; j < DBATN; ++j) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(acc[i][j], slot, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint gRow = baseRow + sg_row * (DBATM * 8) + i * 8;
            const uint gCol = baseCol + sg_col * (DBATN * 8) + j * 8;
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
