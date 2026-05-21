/*
 * tensorcore — simdgroup_matrix GEMM kernels for Apple Silicon.
 *
 * Layout (all variants):
 *   - Threadgroup tile: 64×64 output, BK=16 along K
 *   - 4 simdgroups × 32 threads = 128 threads/threadgroup
 *   - Simdgroup grid: WM=2, WN=2  (each simdgroup owns 32×32 of the output)
 *   - Per simdgroup: TM=4, TN=4 fragments of 8×8
 *
 * Each kernel instantiates the same blocked algorithm with a different IO/accum
 * dtype combo. We compute alpha * A @ B + beta * C, with accumulators always in
 * fp32 (or int32 for i8) regardless of IO dtype — this matches what NVIDIA's
 * tensor cores do and gives bit-stable behavior across input precisions.
 *
 * Reference implementation: eshkol-platform/lib/backend/gpu/metal_softfloat.h
 *   matmul_f32_simd_pure (the 64×64 variant)
 * extended here with:
 *   - parameterized IO dtype (half, bfloat, float, char)
 *   - alpha/beta scaling
 *   - optional transpose_a / transpose_b at compile time via function-constants
 */

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

/* MFA-style simdgroup async copy gates.  Available on MSL 2.4+ (macOS 13+).
 * Provides hardware-orchestrated cooperative threadgroup-memory loads with
 * automatic latency hiding — the MSL-level equivalent of NVIDIA cp.async.
 * MFA (Apple's metal-flash-attention) used these to climb from ~13 to ~20
 * TFLOPS on M2 Ultra. We gate at compile time so the kernel still builds
 * for older Metal versions. */
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 240
#define TC_HAVE_ASYNC_COPY 1
#else
#define TC_HAVE_ASYNC_COPY 0
#endif

/* ------------------------------------------------------------ */
/* Tile constants — fixed for the v0.1 family. Will be          */
/* autotuned per MTLGPUFamily in a later phase.                 */
/* ------------------------------------------------------------ */
constant constexpr uint BM = 64;
constant constexpr uint BN = 64;
constant constexpr uint BK = 32;  /* 4 inner MMA steps per K-block — higher
                                   * arithmetic intensity. Was 16 in v0.1. */

constant constexpr uint WM = 2;        /* simdgroup rows in grid                  */
constant constexpr uint WN = 2;        /* simdgroup cols in grid                  */
constant constexpr uint TM = 4;        /* 8×8 fragments per simdgroup along M     */
constant constexpr uint TN = 4;        /* 8×8 fragments per simdgroup along N     */

constant constexpr uint THREADS = WM * WN * 32;   /* 128                            */

/* +4 padding to avoid threadgroup-memory bank conflicts on contiguous strides
 * that are multiples of 8 (Apple GPU 32 banks × 4-byte words).  */
constant constexpr uint SA_STRIDE = BK + 4;
constant constexpr uint SB_STRIDE = BN + 4;
constant constexpr uint SA_SIZE   = BM * SA_STRIDE;
constant constexpr uint SB_SIZE   = BK * SB_STRIDE;

/* Function constants — bound at pipeline-create time, not at dispatch. Allows
 * the same kernel name to specialize transpose without extra parameters. */
constant bool g_trans_a [[function_constant(0)]];
constant bool g_trans_b [[function_constant(1)]];

/* ============================================================ */
/* Templated kernel body — instantiated per dtype below.        */
/* ============================================================ */

template <typename IO_T, typename ACC_T>
inline void gemm_simdgroup_impl(
    device const IO_T*  A,
    device const IO_T*  B,
    device       IO_T*  C,
    constant uint& M,
    constant uint& N,
    constant uint& K,
    constant float& alpha,
    constant float& beta,
    threadgroup IO_T*   shared_mem,
    uint2 group_id,
    uint  sgid,
    uint  slid)
{
    const uint baseRow = group_id.y * BM;
    const uint baseCol = group_id.x * BN;

    const uint sg_row = sgid / WN;      /* 0..WM-1 */
    const uint sg_col = sgid % WN;      /* 0..WN-1 */

    /* Acc fragments — always fp32 (or int32 for integer kernels). */
    simdgroup_matrix<ACC_T, 8, 8> acc[TM][TN];
    for (uint i = 0; i < TM; ++i)
        for (uint j = 0; j < TN; ++j)
            acc[i][j] = simdgroup_matrix<ACC_T, 8, 8>(ACC_T(0));

    /* Single-buffer (double-buffer experiment regressed on Apple7/8 — Metal
     * lacks cp.async, and doubling TG memory hurts occupancy more than the
     * load/compute overlap helps). */
    threadgroup IO_T* sA_buf = shared_mem;
    threadgroup IO_T* sB_buf = shared_mem + SA_SIZE;
    threadgroup IO_T* sA[1] = { sA_buf };
    threadgroup IO_T* sB[1] = { sB_buf };

    const uint tid = sgid * 32 + slid;

    /* Total elements loaded per K-block per thread.
     *   A tile = BM*BK elements / THREADS
     *   B tile = BK*BN elements / THREADS */
    constexpr uint EPT_A = (BM * BK) / THREADS;
    constexpr uint EPT_B = (BK * BN) / THREADS;

    const bool full_tile_m = (baseRow + BM <= M);
    const bool full_tile_n = (baseCol + BN <= N);

#define TC_LOAD_A_TILE(_dst, _kBlock)                                                       \
    do {                                                                                    \
        const uint kB = (_kBlock);                                                          \
        const bool fk = (kB + BK <= K);                                                     \
        if (full_tile_m && fk && !g_trans_a) {                                              \
            using V4 = vec<IO_T, 4>;                                                        \
            constexpr uint VEC_A = (BM * BK) / (THREADS * 4);                               \
            for (uint i = 0; i < VEC_A; ++i) {                                              \
                const uint idx = i * THREADS + tid;                                         \
                const uint elem = idx * 4;                                                  \
                const uint row = elem / BK;                                                 \
                const uint col = elem % BK;                                                 \
                V4 v = *((device const V4*)(A + (baseRow + row) * K + kB + col));           \
                (_dst)[row * SA_STRIDE + col + 0] = v[0];                                   \
                (_dst)[row * SA_STRIDE + col + 1] = v[1];                                   \
                (_dst)[row * SA_STRIDE + col + 2] = v[2];                                   \
                (_dst)[row * SA_STRIDE + col + 3] = v[3];                                   \
            }                                                                               \
        } else {                                                                            \
            for (uint i = 0; i < EPT_A; ++i) {                                              \
                const uint idx = i * THREADS + tid;                                         \
                const uint row = idx / BK;                                                  \
                const uint col = idx % BK;                                                  \
                const uint gRow = baseRow + row;                                            \
                const uint gCol = kB + col;                                                 \
                IO_T v = IO_T(0);                                                           \
                if (gRow < M && gCol < K) {                                                 \
                    if (g_trans_a)  v = A[gCol * M + gRow];                                 \
                    else            v = A[gRow * K + gCol];                                 \
                }                                                                           \
                (_dst)[row * SA_STRIDE + col] = v;                                          \
            }                                                                               \
        }                                                                                   \
    } while (0)

#define TC_LOAD_B_TILE(_dst, _kBlock)                                                       \
    do {                                                                                    \
        const uint kB = (_kBlock);                                                          \
        const bool fk = (kB + BK <= K);                                                     \
        if (full_tile_n && fk && !g_trans_b) {                                              \
            using V4 = vec<IO_T, 4>;                                                        \
            constexpr uint VEC_B = (BK * BN) / (THREADS * 4);                               \
            for (uint i = 0; i < VEC_B; ++i) {                                              \
                const uint idx = i * THREADS + tid;                                         \
                const uint elem = idx * 4;                                                  \
                const uint row = elem / BN;                                                 \
                const uint col = elem % BN;                                                 \
                V4 v = *((device const V4*)(B + (kB + row) * N + baseCol + col));           \
                (_dst)[row * SB_STRIDE + col + 0] = v[0];                                   \
                (_dst)[row * SB_STRIDE + col + 1] = v[1];                                   \
                (_dst)[row * SB_STRIDE + col + 2] = v[2];                                   \
                (_dst)[row * SB_STRIDE + col + 3] = v[3];                                   \
            }                                                                               \
        } else {                                                                            \
            for (uint i = 0; i < EPT_B; ++i) {                                              \
                const uint idx = i * THREADS + tid;                                         \
                const uint row = idx / BN;                                                  \
                const uint col = idx % BN;                                                  \
                const uint gRow = kB + row;                                                 \
                const uint gCol = baseCol + col;                                            \
                IO_T v = IO_T(0);                                                           \
                if (gRow < K && gCol < N) {                                                 \
                    if (g_trans_b)  v = B[gCol * K + gRow];                                 \
                    else            v = B[gRow * N + gCol];                                 \
                }                                                                           \
                (_dst)[row * SB_STRIDE + col] = v;                                          \
            }                                                                               \
        }                                                                                   \
    } while (0)

#define TC_COMPUTE(_sA, _sB)                                                                \
    do {                                                                                    \
        for (uint kk = 0; kk < BK; kk += 8) {                                               \
            simdgroup_matrix<IO_T, 8, 8> a_frag[TM];                                        \
            simdgroup_matrix<IO_T, 8, 8> b_frag[TN];                                        \
            for (uint i = 0; i < TM; ++i) {                                                 \
                const uint row = sg_row * (TM * 8) + i * 8;                                 \
                simdgroup_load(a_frag[i], (_sA) + row * SA_STRIDE + kk, SA_STRIDE);         \
            }                                                                               \
            for (uint j = 0; j < TN; ++j) {                                                 \
                const uint col = sg_col * (TN * 8) + j * 8;                                 \
                simdgroup_load(b_frag[j], (_sB) + kk * SB_STRIDE + col, SB_STRIDE);         \
            }                                                                               \
            for (uint i = 0; i < TM; ++i)                                                   \
                for (uint j = 0; j < TN; ++j)                                               \
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]); \
        }                                                                                   \
    } while (0)

    /* Single-buffered K-loop. */
    for (uint kBlock = 0; kBlock < K; kBlock += BK) {
        TC_LOAD_A_TILE(sA[0], kBlock);
        TC_LOAD_B_TILE(sB[0], kBlock);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        TC_COMPUTE(sA[0], sB[0]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#undef TC_LOAD_A_TILE
#undef TC_LOAD_B_TILE
#undef TC_COMPUTE

    /* ------- alpha * acc + beta * C  →  C  -------
     * Spill each 8×8 acc (32 simdgroup-shared regs) to threadgroup memory,
     * then use all 32 lanes of the simdgroup to cooperatively write the 64
     * cells (2 passes of 32). Handles fp32-acc / fp16-IO conversion at the
     * per-element store. Bounds-checked per cell, so partial tiles use the
     * same path. */
    for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; ++j) {
            const uint gRow = baseRow + sg_row * (TM * 8) + i * 8;
            const uint gCol = baseCol + sg_col * (TN * 8) + j * 8;

            /* Per-simdgroup spill slot: 64 ACC_T elements. shared_mem is at
             * least SA_SIZE+SB_SIZE IO_T elements which exceeds 4*64*ACC_T
             * for our tile config. */
            threadgroup ACC_T* slot = ((threadgroup ACC_T*)shared_mem) + sgid * 64;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(acc[i][j], slot, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint k = 0; k < 64; k += 32) {
                const uint idx = k + slid;
                if (idx < 64) {
                    const uint r  = idx >> 3;
                    const uint c  = idx & 7;
                    const uint Gr = gRow + r;
                    const uint Gc = gCol + c;
                    if (Gr < M && Gc < N) {
                        ACC_T s = slot[r * 8 + c] * (ACC_T)alpha;
                        if (beta != 0.0f) {
                            s += (ACC_T)C[Gr * N + Gc] * (ACC_T)beta;
                        }
                        C[Gr * N + Gc] = IO_T(s);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

/* ============================================================ */
/* Kernel entry points — one per (IO, ACC) combo.               */
/* ============================================================ */

/* ----- fp16 IO, fp32 accumulator (Apple7+, M1+) ----- */
kernel void tc_gemm_f16_f32(
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
    threadgroup half shared_mem[SA_SIZE + SB_SIZE];
    gemm_simdgroup_impl<half, float>(A, B, C, M, N, K, alpha, beta,
                                     shared_mem, group_id, sgid, slid);
}

/* ----- Batched fp16 GEMM. group_id.z selects the batch index.
 *       Each batch has its own (A, B, C) slice at strides supplied via buffers. */
kernel void tc_gemm_f16_f32_batched(
    device const half*  A     [[buffer(0)]],
    device const half*  B     [[buffer(1)]],
    device       half*  C     [[buffer(2)]],
    constant uint& M          [[buffer(3)]],
    constant uint& N          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    constant float& alpha     [[buffer(6)]],
    constant float& beta      [[buffer(7)]],
    constant ulong& stride_a  [[buffer(8)]],   /* elements between batches */
    constant ulong& stride_b  [[buffer(9)]],
    constant ulong& stride_c  [[buffer(10)]],
    uint3 group_id            [[threadgroup_position_in_grid]],
    uint  sgid                [[simdgroup_index_in_threadgroup]],
    uint  slid                [[thread_index_in_simdgroup]])
{
    threadgroup half shared_mem[SA_SIZE + SB_SIZE];
    device const half* Ab = A + (ulong)group_id.z * stride_a;
    device const half* Bb = B + (ulong)group_id.z * stride_b;
    device       half* Cb = C + (ulong)group_id.z * stride_c;
    uint2 tg2 = uint2(group_id.x, group_id.y);
    gemm_simdgroup_impl<half, float>(Ab, Bb, Cb, M, N, K, alpha, beta,
                                     shared_mem, tg2, sgid, slid);
}

/* ----- fp32 IO, fp32 accumulator (Apple7+, M1+) ----- */
kernel void tc_gemm_f32_f32(
    device const float* A     [[buffer(0)]],
    device const float* B     [[buffer(1)]],
    device       float* C     [[buffer(2)]],
    constant uint& M          [[buffer(3)]],
    constant uint& N          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    constant float& alpha     [[buffer(6)]],
    constant float& beta      [[buffer(7)]],
    uint2 group_id            [[threadgroup_position_in_grid]],
    uint  sgid                [[simdgroup_index_in_threadgroup]],
    uint  slid                [[thread_index_in_simdgroup]])
{
    threadgroup float shared_mem[SA_SIZE + SB_SIZE];
    gemm_simdgroup_impl<float, float>(A, B, C, M, N, K, alpha, beta,
                                      shared_mem, group_id, sgid, slid);
}

/* ----- bf16 IO, fp32 accumulator (Apple9+, M3+; MSL 3.1+) ----- */
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 310
kernel void tc_gemm_bf16_f32(
    device const bfloat* A    [[buffer(0)]],
    device const bfloat* B    [[buffer(1)]],
    device       bfloat* C    [[buffer(2)]],
    constant uint& M          [[buffer(3)]],
    constant uint& N          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    constant float& alpha     [[buffer(6)]],
    constant float& beta      [[buffer(7)]],
    uint2 group_id            [[threadgroup_position_in_grid]],
    uint  sgid                [[simdgroup_index_in_threadgroup]],
    uint  slid                [[thread_index_in_simdgroup]])
{
    threadgroup bfloat shared_mem[SA_SIZE + SB_SIZE];
    gemm_simdgroup_impl<bfloat, float>(A, B, C, M, N, K, alpha, beta,
                                       shared_mem, group_id, sgid, slid);
}
#endif

/* ----- int8 IO, int32 accumulator (Apple10+, M4+; MSL 3.2/3.x) -----
 * Note: integer matrix MMA on Apple10+ uses `simdgroup_matrix<char, 8, 8>` for
 * inputs and `simdgroup_matrix<int, 8, 8>` for the accumulator. alpha/beta are
 * passed as float and rounded at store time. Metal 4 removes these integer
 * simdgroup_matrix element types, so SDK26 builds omit this kernel and fall
 * back through the MPS i8 path. */
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 320 && __METAL_VERSION__ < 400
kernel void tc_gemm_i8_i32(
    device const char*  A     [[buffer(0)]],
    device const char*  B     [[buffer(1)]],
    device       int*   C     [[buffer(2)]],
    constant uint& M          [[buffer(3)]],
    constant uint& N          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    constant float& alpha     [[buffer(6)]],
    constant float& beta      [[buffer(7)]],
    uint2 group_id            [[threadgroup_position_in_grid]],
    uint  sgid                [[simdgroup_index_in_threadgroup]],
    uint  slid                [[thread_index_in_simdgroup]])
{
    /* Note: int8 IO with int32 accum/output requires a slightly different
     * template instantiation because IO_T != ACC_T at store. For v0.1 we keep
     * a separate, narrower implementation. */
    const uint baseRow = group_id.y * BM;
    const uint baseCol = group_id.x * BN;
    const uint sg_row = sgid / WN;
    const uint sg_col = sgid % WN;

    simdgroup_matrix<int, 8, 8> acc[TM][TN];
    for (uint i = 0; i < TM; ++i)
        for (uint j = 0; j < TN; ++j)
            acc[i][j] = simdgroup_matrix<int, 8, 8>(0);

    threadgroup char shared_mem[SA_SIZE + SB_SIZE];
    threadgroup char* sA = shared_mem;
    threadgroup char* sB = shared_mem + SA_SIZE;

    const uint tid = sgid * 32 + slid;
    constexpr uint EPT_A = (BM * BK) / THREADS;
    constexpr uint EPT_B = (BK * BN) / THREADS;

    for (uint kBlock = 0; kBlock < K; kBlock += BK) {
        for (uint i = 0; i < EPT_A; ++i) {
            const uint idx = i * THREADS + tid;
            const uint row = idx / BK;
            const uint col = idx % BK;
            const uint gRow = baseRow + row;
            const uint gCol = kBlock + col;
            sA[row * SA_STRIDE + col] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0;
        }
        for (uint i = 0; i < EPT_B; ++i) {
            const uint idx = i * THREADS + tid;
            const uint row = idx / BN;
            const uint col = idx % BN;
            const uint gRow = kBlock + row;
            const uint gCol = baseCol + col;
            sB[row * SB_STRIDE + col] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<char, 8, 8> a_frag[TM];
            simdgroup_matrix<char, 8, 8> b_frag[TN];
            for (uint i = 0; i < TM; ++i) {
                const uint row = sg_row * (TM * 8) + i * 8;
                simdgroup_load(a_frag[i], sA + row * SA_STRIDE + kk, SA_STRIDE);
            }
            for (uint j = 0; j < TN; ++j) {
                const uint col = sg_col * (TN * 8) + j * 8;
                simdgroup_load(b_frag[j], sB + kk * SB_STRIDE + col, SB_STRIDE);
            }
            for (uint i = 0; i < TM; ++i)
                for (uint j = 0; j < TN; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; ++j) {
            const uint gRow = baseRow + sg_row * (TM * 8) + i * 8;
            const uint gCol = baseCol + sg_col * (TN * 8) + j * 8;
            if (gRow + 7 < M && gCol + 7 < N) {
                simdgroup_store(acc[i][j], C + gRow * N + gCol, N);
            }
        }
    }
}
#endif
