#ifndef TENSORCORE_SPARSE_GEMM_H
#define TENSORCORE_SPARSE_GEMM_H

/*
 * tensorcore — N:M (2:4) structured-sparse tensor-core GEMM.
 *
 * Ampere (sm_80+) tensor cores expose a 2:4 sparse GEMM mode: when B
 * is pruned so every 4-element block along the K axis contains at most
 * 2 non-zero values, the tensor cores skip the zeros and run at 2×
 * the dense fp16/bf16 throughput. cuSPARSELt is the cuBLAS-aligned
 * library that exposes this for arbitrary problem shapes.
 *
 * GeoRefine's T8 (geometry-guided structured-sparsity) track:
 *   - GeometricLM identifies low-importance weights via the geometric
 *     prior, prunes to 2:4 pattern, and gets measured GPU speedup on
 *     Ampere/Ada/Hopper. tensorcore exposes the kernel so the loss
 *     of speedup over dense isn't an issue for inference paths.
 *
 * API design:
 *   - Pruning step:    tc_sparse_24_prune  → applies the 2:4 mask in-place
 *                        (selects the top-2 magnitudes per 4-block).
 *   - Compute step:    tc_sparse_24_gemm   → fp16 / bf16 GEMM with the
 *                        pruned B; compute_type fp32 accumulator.
 *
 * Build gate: TC_ENABLE_CUDA=ON AND cusparseLt available. On hosts
 * without cuSPARSELt (Mac, CPU-only Linux, CUDA without cuSPARSELt),
 * tc_sparse_24_gemm falls back to a dense reference that ignores the
 * sparsity and produces the bit-correct dense result — useful for
 * portability tests of caller code. tc_sparse_24_prune always runs
 * (pure host arithmetic).
 *
 * Caller responsibility:
 *   - K must be a multiple of 4 (2:4 pattern requires 4-blocks along K).
 *   - For best performance on cusparseLt, M and N should be multiples
 *     of 16 (tensor-core tile alignment). Smaller shapes may fall
 *     through to dense in v0.1.
 */

#include "tensorcore/status.h"
#include "tensorcore/tensorcore.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Apply the 2:4 sparsity pattern to a dense fp16 / fp32 weight matrix
 * in-place. Each 4-element block along the K axis keeps the top-2
 * absolute-value entries and zeros the other two.
 *
 *   W : [rows, cols] in row-major. The K axis ("inner" dim of the
 *       subsequent GEMM) is `cols` when W is used as B (the standard
 *       PyTorch nn.Linear weight layout).
 *   dtype : TC_DTYPE_F16 or TC_DTYPE_F32 (fp16/fp32 weights both supported)
 *
 * Returns TC_ERR_INVALID_SHAPE if cols % 4 != 0. */
tc_status_t tc_sparse_24_prune(tc_context* ctx,
                                tc_buffer* W,
                                tc_dtype_t dtype,
                                int rows, int cols);

/* Verify a matrix is in 2:4 pattern (each 4-block has ≤2 non-zeros).
 * Returns TC_OK if pattern holds, TC_ERR_INVALID_ARG if it doesn't.
 * Useful for diff-tests after caller-side pruning to confirm the
 * cusparseLt path will accept the input. */
tc_status_t tc_sparse_24_check(tc_context* ctx,
                                const tc_buffer* W,
                                tc_dtype_t dtype,
                                int rows, int cols);

/* 2:4 structured-sparse GEMM:  C = alpha * A @ B + beta * C
 * where B must already be in 2:4 pattern (call tc_sparse_24_prune first).
 *
 *   A : [M, K]  fp16/bf16/fp32, row-major dense
 *   B : [K, N]  fp16/bf16/fp32, row-major, 2:4 pattern along K
 *   C : [M, N]  fp16/bf16/fp32, row-major dense
 *
 * On CUDA hosts with cusparseLt, dispatches to cusparseLtMatmul (tensor
 * cores). On other hosts, falls back to dense GEMM (correct but no
 * speedup — useful for portability tests). */
tc_status_t tc_sparse_24_gemm(tc_context* ctx,
                               const tc_buffer* A,
                               const tc_buffer* B,
                               tc_buffer* C,
                               int M, int N, int K,
                               tc_dtype_t a_dtype,
                               tc_dtype_t b_dtype,
                               tc_dtype_t c_dtype,
                               float alpha, float beta);

/* Returns 1 if the runtime has a hardware 2:4 sparse path (Ampere+ with
 * cusparseLt linked), 0 otherwise. Callers use this to choose between
 * the sparse and dense compute paths at model-construction time. */
int tc_sparse_24_available(void);

#ifdef __cplusplus
}
#endif
#endif
