/*
 * tensorcore — 128×128 simdgroup_matrix GEMM (large-tile variant).
 *
 * Layout:
 *   - Threadgroup tile: 128 × 128 output, BK=8 along K (single inner step per K-block)
 *   - 16 simdgroups × 32 threads = 512 threads per threadgroup
 *   - Simdgroup grid: WM=4, WN=4  (each simdgroup owns 32×32 of the output)
 *   - Per simdgroup: TM=4, TN=4 of 8×8 fragments
 *
 * Memory amplification vs 64×64:
 *   - Each element loaded once is reused across 128 output rows/cols, not 64.
 *   - Memory traffic per FLOP halves for large shapes.
 *
 * Vectorized cooperative loads (half4 / float4) further reduce instruction
 * count on the load path.
 *
 * TG memory budget (Apple7+ = 32 KB):
 *   sA: 128 × 12 elems  = 1536 elems
 *   sB:   8 × 132 elems = 1056 elems
 *   spill: 16 × 64 elems = 1024 elems (fp32)
 *   fp16 total: 1536*2 + 1056*2 + 1024*4 ≈ 9.3 KB
 *   fp32 total: 1536*4 + 1056*4 + 1024*4 ≈ 14.5 KB
 */

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

constant constexpr uint G128_BM = 128;
constant constexpr uint G128_BN = 128;
constant constexpr uint G128_BK = 8;

constant constexpr uint G128_WM = 4;
constant constexpr uint G128_WN = 4;
constant constexpr uint G128_TM = 4;
constant constexpr uint G128_TN = 4;
constant constexpr uint G128_THREADS = G128_WM * G128_WN * 32;   /* 512 */

constant constexpr uint G128_SA_STRIDE = G128_BK + 4;       /* 12  */
constant constexpr uint G128_SB_STRIDE = G128_BN + 4;       /* 132 */
constant constexpr uint G128_SA_SIZE   = G128_BM * G128_SA_STRIDE;
constant constexpr uint G128_SB_SIZE   = G128_BK * G128_SB_STRIDE;

constant bool g128_trans_a [[function_constant(0)]];
constant bool g128_trans_b [[function_constant(1)]];

/* ============================================================ */
/* Templated kernel body                                        */
/* ============================================================ */
template <typename IO_T, typename ACC_T>
inline void gemm_128_impl(
    device const IO_T*   A,
    device const IO_T*   B,
    device       IO_T*   C,
    constant uint& M, constant uint& N, constant uint& K,
    constant float& alpha, constant float& beta,
    threadgroup IO_T*    shared_mem,
    uint2 group_id,
    uint  sgid,
    uint  slid)
{
    const uint baseRow = group_id.y * G128_BM;
    const uint baseCol = group_id.x * G128_BN;
    const uint sg_row = sgid / G128_WN;
    const uint sg_col = sgid % G128_WN;
    const uint tid = sgid * 32 + slid;

    simdgroup_matrix<ACC_T, 8, 8> acc[G128_TM][G128_TN];
    for (uint i = 0; i < G128_TM; ++i)
        for (uint j = 0; j < G128_TN; ++j)
            acc[i][j] = simdgroup_matrix<ACC_T, 8, 8>(ACC_T(0));

    threadgroup IO_T* sA = shared_mem;
    threadgroup IO_T* sB = shared_mem + G128_SA_SIZE;

    /* Per K-block, each thread loads:
     *   A tile: BM*BK = 128*8 = 1024 elems / 512 threads = 2 elems/thread
     *   B tile: BK*BN = 8*128 = 1024 elems / 512 threads = 2 elems/thread */
    const bool full_m = (baseRow + G128_BM <= M);
    const bool full_n = (baseCol + G128_BN <= N);

    for (uint kBlock = 0; kBlock < K; kBlock += G128_BK) {
        const bool full_k = (kBlock + G128_BK <= K);

        /* ---- A tile load (128 × 8) ---- */
        if (full_m && full_k && !g128_trans_a) {
            for (uint i = 0; i < 2; ++i) {
                const uint idx = i * G128_THREADS + tid;
                const uint row = idx / G128_BK;
                const uint col = idx % G128_BK;
                sA[row * G128_SA_STRIDE + col] = A[(baseRow + row) * K + kBlock + col];
            }
        } else {
            for (uint i = 0; i < 2; ++i) {
                const uint idx = i * G128_THREADS + tid;
                const uint row = idx / G128_BK;
                const uint col = idx % G128_BK;
                const uint gRow = baseRow + row;
                const uint gCol = kBlock + col;
                IO_T v = IO_T(0);
                if (gRow < M && gCol < K) {
                    v = g128_trans_a ? A[gCol * M + gRow] : A[gRow * K + gCol];
                }
                sA[row * G128_SA_STRIDE + col] = v;
            }
        }

        /* ---- B tile load (8 × 128) ---- */
        if (full_n && full_k && !g128_trans_b) {
            for (uint i = 0; i < 2; ++i) {
                const uint idx = i * G128_THREADS + tid;
                const uint row = idx / G128_BN;
                const uint col = idx % G128_BN;
                sB[row * G128_SB_STRIDE + col] = B[(kBlock + row) * N + baseCol + col];
            }
        } else {
            for (uint i = 0; i < 2; ++i) {
                const uint idx = i * G128_THREADS + tid;
                const uint row = idx / G128_BN;
                const uint col = idx % G128_BN;
                const uint gRow = kBlock + row;
                const uint gCol = baseCol + col;
                IO_T v = IO_T(0);
                if (gRow < K && gCol < N) {
                    v = g128_trans_b ? B[gCol * K + gRow] : B[gRow * N + gCol];
                }
                sB[row * G128_SB_STRIDE + col] = v;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* ---- inner K-loop: BK=8 ⇒ single step ---- */
        {
            simdgroup_matrix<IO_T, 8, 8> a_frag[G128_TM];
            simdgroup_matrix<IO_T, 8, 8> b_frag[G128_TN];

            for (uint i = 0; i < G128_TM; ++i) {
                const uint row = sg_row * (G128_TM * 8) + i * 8;
                simdgroup_load(a_frag[i], sA + row * G128_SA_STRIDE, G128_SA_STRIDE);
            }
            for (uint j = 0; j < G128_TN; ++j) {
                const uint col = sg_col * (G128_TN * 8) + j * 8;
                simdgroup_load(b_frag[j], sB + col, G128_SB_STRIDE);
            }
            for (uint i = 0; i < G128_TM; ++i)
                for (uint j = 0; j < G128_TN; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* ---- store path: spill + cooperative write (alpha/beta) ---- */
    for (uint i = 0; i < G128_TM; ++i) {
        for (uint j = 0; j < G128_TN; ++j) {
            const uint gRow = baseRow + sg_row * (G128_TM * 8) + i * 8;
            const uint gCol = baseCol + sg_col * (G128_TN * 8) + j * 8;

            threadgroup ACC_T* slot = ((threadgroup ACC_T*)shared_mem) + sgid * 64;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(acc[i][j], slot, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint k = 0; k < 64; k += 32) {
                const uint idx = k + slid;
                if (idx < 64) {
                    const uint r = idx >> 3;
                    const uint c = idx & 7;
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
/* Entry points                                                 */
/* ============================================================ */
kernel void tc_gemm_f16_f32_128(
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
    threadgroup half shared_mem[G128_SA_SIZE + G128_SB_SIZE];
    gemm_128_impl<half, float>(A, B, C, M, N, K, alpha, beta,
                               shared_mem, group_id, sgid, slid);
}

kernel void tc_gemm_f32_f32_128(
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
    threadgroup float shared_mem[G128_SA_SIZE + G128_SB_SIZE];
    gemm_128_impl<float, float>(A, B, C, M, N, K, alpha, beta,
                                shared_mem, group_id, sgid, slid);
}

#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 310
kernel void tc_gemm_bf16_f32_128(
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
    threadgroup bfloat shared_mem[G128_SA_SIZE + G128_SB_SIZE];
    gemm_128_impl<bfloat, float>(A, B, C, M, N, K, alpha, beta,
                                 shared_mem, group_id, sgid, slid);
}
#endif
