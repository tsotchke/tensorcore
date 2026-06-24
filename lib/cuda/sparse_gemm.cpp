/*
 * tensorcore — 2:4 sparse-tensor-core GEMM via cuSPARSELt (NVIDIA Ampere+).
 *
 * Gated on TC_ENABLE_CUSPARSELT=ON in CMake (auto-set when cmake detects
 * `cusparseLt.h` + `libcusparseLt.so` in the CUDA toolkit). When enabled,
 * this TU's strong tc_sparse_24_gemm + tc_sparse_24_available override
 * the weak defaults in lib/ops/sparse_gemm_cpu.cpp.
 *
 * Algorithm:
 *   1. Initialize a cusparseLt handle (per process).
 *   2. Describe dense A (matrix descriptor), structured B (sparse descriptor
 *      with CUSPARSELT_SPARSITY_50_PERCENT), dense C.
 *   3. Initialize a matmul descriptor with the appropriate compute type.
 *   4. Select an algorithm (CUSPARSELT_MATMUL_ALG_DEFAULT) and create a plan.
 *   5. cusparseLtSpMMACompress(B) — compresses B from dense 2:4 layout to
 *      cusparseLt's internal packed format (4 indices + 2 values per block).
 *   6. cusparseLtMatmul — runs on tensor cores at 2× dense fp16/bf16 rate.
 *
 * Caller responsibility:
 *   - B must already be pruned to 2:4 (tc_sparse_24_prune does this).
 *   - K % 4 == 0 (enforced by tc_sparse_24_check).
 *   - For best perf, M and N should be multiples of 16. cusparseLt accepts
 *     unaligned shapes but pads internally with measurable overhead.
 *
 * Validation pending: this TU compiles unconditionally on CUDA hosts; the
 * actual cusparseLt call path is exercised by the integration tests on
 * cosbox (RTX 3090) + GCP A100/H100 once those benches land.
 */

#include "tensorcore/sparse_gemm.h"
#include "tensorcore/tensorcore.h"

#include <cstdint>
#include <cstring>

#if defined(TC_ENABLE_CUDA) && defined(TC_ENABLE_CUSPARSELT)
#  include <cuda_runtime.h>
#  include <cusparseLt.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define TC_CUSPARSELT_INTERNAL __attribute__((visibility("hidden")))
#else
#  define TC_CUSPARSELT_INTERNAL
#endif

#if defined(TC_ENABLE_CUDA) && defined(TC_ENABLE_CUSPARSELT)

namespace {

struct CuSparseLtState {
    cusparseLtHandle_t handle;
    bool initialized = false;
};

CuSparseLtState& state() {
    static CuSparseLtState s;
    if (!s.initialized) {
        if (cusparseLtInit(&s.handle) == CUSPARSE_STATUS_SUCCESS) {
            s.initialized = true;
        }
    }
    return s;
}

cudaDataType_t to_cuda_dtype(tc_dtype_t d) {
    switch (d) {
        case TC_DTYPE_F32:  return CUDA_R_32F;
        case TC_DTYPE_F16:  return CUDA_R_16F;
        case TC_DTYPE_BF16: return CUDA_R_16BF;
        default:            return CUDA_R_32F;
    }
}

cusparseComputeType compute_type_for(tc_dtype_t c) {
    /* Compute in fp32 for fp16/bf16 inputs; fp32 IO keeps fp32 compute. */
    if (c == TC_DTYPE_F32) return CUSPARSE_COMPUTE_TF32;
    return CUSPARSE_COMPUTE_32F;
}

}  // namespace

extern "C" tc_status_t tc_sparse_24_gemm(tc_context* ctx,
                                          const tc_buffer* A,
                                          const tc_buffer* B,
                                          tc_buffer* C,
                                          int M, int N, int K,
                                          tc_dtype_t a_dtype,
                                          tc_dtype_t b_dtype,
                                          tc_dtype_t c_dtype,
                                          float alpha, float beta) {
    if (!ctx || !A || !B || !C || M <= 0 || N <= 0 || K <= 0)
        return TC_ERR_INVALID_ARG;
    if ((K & 3) != 0) return TC_ERR_INVALID_SHAPE;
    auto& s = state();
    if (!s.initialized) return TC_ERR_INTERNAL;

    void *Ap = nullptr, *Bp = nullptr, *Cp = nullptr;
    if (tc_buffer_map((tc_buffer*)A, &Ap) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map((tc_buffer*)B, &Bp) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map(C, &Cp) != TC_OK) return TC_ERR_INTERNAL;

    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t algSel;
    cusparseLtMatmulPlan_t plan;

    const cudaDataType_t at = to_cuda_dtype(a_dtype);
    const cudaDataType_t bt = to_cuda_dtype(b_dtype);
    const cudaDataType_t ct = to_cuda_dtype(c_dtype);
    const cusparseComputeType compute = compute_type_for(c_dtype);

    /* Row-major; tensorcore's GEMM convention. */
    if (cusparseLtDenseDescriptorInit(&s.handle, &matA, M, K, K, 16, at,
                                       CUSPARSE_ORDER_ROW) != CUSPARSE_STATUS_SUCCESS)
        return TC_ERR_INTERNAL;
    if (cusparseLtStructuredDescriptorInit(&s.handle, &matB, K, N, N, 16, bt,
                                            CUSPARSE_ORDER_ROW,
                                            CUSPARSELT_SPARSITY_50_PERCENT)
            != CUSPARSE_STATUS_SUCCESS)
        return TC_ERR_INTERNAL;
    if (cusparseLtDenseDescriptorInit(&s.handle, &matC, M, N, N, 16, ct,
                                       CUSPARSE_ORDER_ROW) != CUSPARSE_STATUS_SUCCESS)
        return TC_ERR_INTERNAL;

    if (cusparseLtMatmulDescriptorInit(&s.handle, &matmul, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &matA, &matB, &matC, &matC, compute)
            != CUSPARSE_STATUS_SUCCESS)
        return TC_ERR_INTERNAL;
    if (cusparseLtMatmulAlgSelectionInit(&s.handle, &algSel, &matmul,
                                          CUSPARSELT_MATMUL_ALG_DEFAULT)
            != CUSPARSE_STATUS_SUCCESS)
        return TC_ERR_INTERNAL;
    if (cusparseLtMatmulPlanInit(&s.handle, &plan, &matmul, &algSel)
            != CUSPARSE_STATUS_SUCCESS)
        return TC_ERR_INTERNAL;

    /* Compress B (in-place compression buffer = caller's B if size matches). */
    size_t compressed_size = 0;
    if (cusparseLtSpMMACompressedSize(&s.handle, &plan, &compressed_size)
            != CUSPARSE_STATUS_SUCCESS)
        return TC_ERR_INTERNAL;
    void* B_compressed = nullptr;
    if (cudaMalloc(&B_compressed, compressed_size) != cudaSuccess)
        return TC_ERR_ALLOC;
    if (cusparseLtSpMMACompress(&s.handle, &plan, Bp, B_compressed, /*stream=*/0)
            != CUSPARSE_STATUS_SUCCESS) {
        cudaFree(B_compressed);
        return TC_ERR_INTERNAL;
    }

    /* Get workspace. */
    size_t ws_size = 0;
    if (cusparseLtMatmulGetWorkspace(&s.handle, &plan, &ws_size)
            != CUSPARSE_STATUS_SUCCESS) {
        cudaFree(B_compressed);
        return TC_ERR_INTERNAL;
    }
    void* ws = nullptr;
    if (ws_size > 0 && cudaMalloc(&ws, ws_size) != cudaSuccess) {
        cudaFree(B_compressed);
        return TC_ERR_ALLOC;
    }

    const cusparseStatus_t mm_status = cusparseLtMatmul(
        &s.handle, &plan, &alpha, Ap, B_compressed, &beta, Cp, Cp, ws,
        /*streams=*/nullptr, /*numStreams=*/0);

    cudaFree(B_compressed);
    if (ws) cudaFree(ws);
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matB);
    cusparseLtMatDescriptorDestroy(&matC);

    return (mm_status == CUSPARSE_STATUS_SUCCESS) ? TC_OK : TC_ERR_INTERNAL;
}

extern "C" int tc_sparse_24_available(void) {
    return state().initialized ? 1 : 0;
}

#endif  /* TC_ENABLE_CUDA && TC_ENABLE_CUSPARSELT */
