#ifndef TENSORCORE_GEMM_H
#define TENSORCORE_GEMM_H

#include <stdint.h>
#include <stdbool.h>
#include "tensorcore/status.h"
#include "tensorcore/dtype.h"
#include "tensorcore/device.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Layout matches BLAS: matrices are row-major M×K, K×N, M×N. */

typedef struct {
    /* Required */
    int32_t   M, N, K;

    /* Compute dtypes. accum_dtype may differ from a_dtype/b_dtype/c_dtype.
     * Standard combos:
     *   {a:F16, b:F16, c:F16, accum:F32}            (Apple7+)
     *   {a:BF16, b:BF16, c:BF16, accum:F32}         (Apple9+)
     *   {a:F32, b:F32, c:F32, accum:F32}            (Apple7+)
     *   {a:I8, b:I8, c:I32, accum:I32}              (Apple10+)
     */
    tc_dtype_t a_dtype;
    tc_dtype_t b_dtype;
    tc_dtype_t c_dtype;
    tc_dtype_t accum_dtype;

    bool transpose_a;
    bool transpose_b;

    /* alpha * A @ B + beta * C   (stored in c_dtype). */
    float alpha;
    float beta;

    /* Leading dimensions; if 0 a sensible default for row-major contiguous
     * is used (K for A non-trans, N for B non-trans, N for C). */
    int32_t lda, ldb, ldc;
} tc_gemm_desc;

/* Synchronous GEMM. Returns once C is fully written. */
tc_status_t tc_gemm(tc_context* ctx,
                    const tc_gemm_desc* desc,
                    const tc_buffer* A,
                    const tc_buffer* B,
                    tc_buffer*       C);

/* Async GEMM. Caller owns `stream`; sync with tc_stream_sync. */
tc_status_t tc_gemm_async(tc_context* ctx,
                          const tc_gemm_desc* desc,
                          const tc_buffer* A,
                          const tc_buffer* B,
                          tc_buffer*       C,
                          tc_stream*       stream);

/* Batched GEMM: same shape per batch, stride between batches in elements. */
typedef struct {
    tc_gemm_desc base;
    int32_t      batch;
    int64_t      stride_a;
    int64_t      stride_b;
    int64_t      stride_c;
} tc_gemm_batched_desc;

tc_status_t tc_gemm_batched(tc_context* ctx,
                            const tc_gemm_batched_desc* desc,
                            const tc_buffer* A,
                            const tc_buffer* B,
                            tc_buffer*       C);

/* For diagnostics: which backend served the last call (per-thread). */
typedef enum {
    TC_BACKEND_NONE             = 0,
    TC_BACKEND_SIMDGROUP_MATRIX = 1,   /* MSL simdgroup_matrix kernels    */
    TC_BACKEND_TENSOROPS_M5     = 2,   /* Metal 4 mpp::tensor_ops kernels   */
    TC_BACKEND_MPS              = 3,   /* MPSMatrix fallback              */
    TC_BACKEND_ACCELERATE_CPU   = 4,   /* cblas_*gemm fallback            */
    TC_BACKEND_SF64_EMULATED    = 5,   /* SoftFloat-64 emulation          */
    TC_BACKEND_OZAKI_II         = 6,   /* CRT-based exact GEMM            */
    TC_BACKEND_PORTABLE_CPU     = 7,   /* portable C CPU backend          */
} tc_backend_t;

tc_backend_t tc_last_backend(void);
const char*  tc_backend_name(tc_backend_t b);

/* Diagnostic selector for the Metal 4 TensorOps GEMM path. Returns NULL and
 * writes TC_ERR_UNSUPPORTED_DTYPE when the dtype combo has no TensorOps kernel. */
const char*  tc_tensorops_gemm_kernel_name(const tc_gemm_desc* desc,
                                           tc_status_t* err);

#ifdef __cplusplus
}
#endif
#endif
