/*
 * tensorcore — HIP GEMM dispatch.
 *
 * Routes tc_gemm calls into chipStar's hipBLAS port when the HIP backend
 * is selected. hipBLAS provides vendor-tuned GEMM across Intel/NVIDIA/AMD
 * via the chipStar SPIR-V dispatch layer (or native rocBLAS on AMD when
 * available).
 *
 * Compile gate: TC_ENABLE_HIPBLAS=ON in CMake. Without hipBLAS, returns
 * TC_ERR_UNSUPPORTED_FAMILY so the public dispatch falls back to CBLAS or
 * the reference loop while tc_hip_init/device diagnostics can still work.
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/hip.h"

#include <cstdint>
#include <cstdlib>

#if defined(__GNUC__) || defined(__clang__)
#  define TC_HIP_INTERNAL __attribute__((visibility("hidden")))
#else
#  define TC_HIP_INTERNAL
#endif

#if defined(TC_ENABLE_HIPBLAS)
#  include <hip/hip_runtime.h>
#  include <hipblas/hipblas.h>
#endif

extern "C" TC_HIP_INTERNAL void tc_hip_set_last_kernel(const char* name);

namespace {

#if defined(TC_ENABLE_HIPBLAS)

struct HipBlasHandle {
    hipblasHandle_t h = nullptr;
    bool            initialized = false;
};

HipBlasHandle& handle() {
    static HipBlasHandle hh;
    if (!hh.initialized) {
        if (hipblasCreate(&hh.h) == HIPBLAS_STATUS_SUCCESS) {
            hh.initialized = true;
        }
    }
    return hh;
}

/* Translate tc_gemm_desc transpose flags into hipBLAS HIPBLAS_OP_*. */
hipblasOperation_t trans_of(bool t) { return t ? HIPBLAS_OP_T : HIPBLAS_OP_N; }

struct DeviceTriple {
    void* A = nullptr;
    void* B = nullptr;
    void* C = nullptr;
    ~DeviceTriple() {
        if (A) hipFree(A);
        if (B) hipFree(B);
        if (C) hipFree(C);
    }
};

bool matrix_bytes(const tc_gemm_desc* d,
                  size_t elem_size,
                  size_t* a_bytes,
                  size_t* b_bytes,
                  size_t* c_bytes) {
    if (!d || !a_bytes || !b_bytes || !c_bytes || elem_size == 0) return false;
    const int a_rows = d->transpose_a ? d->K : d->M;
    const int a_cols = d->transpose_a ? d->M : d->K;
    const int b_rows = d->transpose_b ? d->N : d->K;
    const int b_cols = d->transpose_b ? d->K : d->N;
    const int lda_h = d->lda ? d->lda : a_cols;
    const int ldb_h = d->ldb ? d->ldb : b_cols;
    const int ldc_h = d->ldc ? d->ldc : d->N;
    if (a_rows <= 0 || b_rows <= 0 || d->M <= 0 ||
        lda_h < a_cols || ldb_h < b_cols || ldc_h < d->N) {
        return false;
    }
    *a_bytes = (size_t)a_rows * (size_t)lda_h * elem_size;
    *b_bytes = (size_t)b_rows * (size_t)ldb_h * elem_size;
    *c_bytes = (size_t)d->M * (size_t)ldc_h * elem_size;
    return true;
}

tc_status_t stage_buffers(const tc_gemm_desc* d,
                          size_t elem_size,
                          const void* Ap,
                          const void* Bp,
                          const void* Cp,
                          DeviceTriple* dev) {
    if (!d || !Ap || !Bp || !Cp || !dev) return TC_ERR_INVALID_ARG;
    size_t a_bytes = 0, b_bytes = 0, c_bytes = 0;
    if (!matrix_bytes(d, elem_size, &a_bytes, &b_bytes, &c_bytes)) {
        return TC_ERR_INVALID_SHAPE;
    }
    if (hipMalloc(&dev->A, a_bytes) != hipSuccess) return TC_ERR_ALLOC;
    if (hipMalloc(&dev->B, b_bytes) != hipSuccess) return TC_ERR_ALLOC;
    if (hipMalloc(&dev->C, c_bytes) != hipSuccess) return TC_ERR_ALLOC;
    if (hipMemcpy(dev->A, Ap, a_bytes, hipMemcpyHostToDevice) != hipSuccess) return TC_ERR_INTERNAL;
    if (hipMemcpy(dev->B, Bp, b_bytes, hipMemcpyHostToDevice) != hipSuccess) return TC_ERR_INTERNAL;
    if (d->beta != 0.0f) {
        if (hipMemcpy(dev->C, Cp, c_bytes, hipMemcpyHostToDevice) != hipSuccess) {
            return TC_ERR_INTERNAL;
        }
    }
    return TC_OK;
}

tc_status_t fetch_c(const tc_gemm_desc* d, size_t elem_size, void* Cp, const DeviceTriple& dev) {
    size_t a_bytes = 0, b_bytes = 0, c_bytes = 0;
    if (!matrix_bytes(d, elem_size, &a_bytes, &b_bytes, &c_bytes)) {
        return TC_ERR_INVALID_SHAPE;
    }
    (void)a_bytes; (void)b_bytes;
    if (hipMemcpy(Cp, dev.C, c_bytes, hipMemcpyDeviceToHost) != hipSuccess) return TC_ERR_INTERNAL;
    if (hipDeviceSynchronize() != hipSuccess) return TC_ERR_INTERNAL;
    return TC_OK;
}

tc_status_t hip_gemm_sgemm(const tc_gemm_desc* d,
                           const tc_buffer* A, const tc_buffer* B,
                           tc_buffer* C) {
    auto& hh = handle();
    if (!hh.initialized) return TC_ERR_INTERNAL;

    /* tc_buffer's storage is host-mapped on integrated GPUs (Apple is Metal
     * here; HIP integrated is xavier-class), but discrete GPUs need explicit
     * device pointers. chipStar's hipMalloc returns device pointers; the
     * tc_buffer's map() returns a host pointer that's transparently mirrored.
     * For v0, expect tc_buffer to expose a device pointer via a new helper. */
    void *Ap = nullptr, *Bp = nullptr, *Cp = nullptr;
    if (tc_buffer_map((tc_buffer*)A, &Ap) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map((tc_buffer*)B, &Bp) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map(C, &Cp) != TC_OK) return TC_ERR_INTERNAL;
    DeviceTriple dev;
    tc_status_t stage = stage_buffers(d, sizeof(float), Ap, Bp, Cp, &dev);
    if (stage != TC_OK) return stage;

    /* hipBLAS uses column-major. Our C ABI is row-major. The standard trick:
     *   C^T = B^T * A^T   (row-major -> column-major).
     * So call hipBLAS with M/N swapped and transposes flipped. */
    const int M = d->N, N = d->M, K = d->K;
    const int lda = d->ldb ? d->ldb : (d->transpose_b ? d->K : d->N);
    const int ldb = d->lda ? d->lda : (d->transpose_a ? d->M : d->K);
    const int ldc = d->ldc ? d->ldc : d->N;
    const hipblasOperation_t op_a = trans_of(d->transpose_b);
    const hipblasOperation_t op_b = trans_of(d->transpose_a);

    hipblasStatus_t s = hipblasSgemm(
        hh.h, op_a, op_b,
        M, N, K,
        &d->alpha,
        (const float*)dev.B, lda,
        (const float*)dev.A, ldb,
        &d->beta,
        (float*)dev.C, ldc);
    if (s != HIPBLAS_STATUS_SUCCESS) return TC_ERR_INTERNAL;
    tc_status_t fetch = fetch_c(d, sizeof(float), Cp, dev);
    if (fetch != TC_OK) return fetch;
    tc_hip_set_last_kernel("hipblas_sgemm_staged");
    return TC_OK;
}

tc_status_t hip_gemm_hgemm(const tc_gemm_desc* d,
                           const tc_buffer* A, const tc_buffer* B,
                           tc_buffer* C) {
    /* hipBLAS hgemm available on NVIDIA + AMD; Intel route through
     * sgemm with fp16 → fp32 dequant. v0 just routes everything through
     * hgemm and lets chipStar's hipBLAS port pick the right impl. */
    auto& hh = handle();
    if (!hh.initialized) return TC_ERR_INTERNAL;

    void *Ap = nullptr, *Bp = nullptr, *Cp = nullptr;
    if (tc_buffer_map((tc_buffer*)A, &Ap) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map((tc_buffer*)B, &Bp) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map(C, &Cp) != TC_OK) return TC_ERR_INTERNAL;
    DeviceTriple dev;
    tc_status_t stage = stage_buffers(d, sizeof(hipblasHalf), Ap, Bp, Cp, &dev);
    if (stage != TC_OK) return stage;

    const hipblasHalf alpha = hipblasHalf((__half)d->alpha);
    const hipblasHalf beta = hipblasHalf((__half)d->beta);
    const int M = d->N, N = d->M, K = d->K;
    const int lda = d->ldb ? d->ldb : (d->transpose_b ? d->K : d->N);
    const int ldb = d->lda ? d->lda : (d->transpose_a ? d->M : d->K);
    const int ldc = d->ldc ? d->ldc : d->N;

    hipblasStatus_t s = hipblasHgemm(
        hh.h, trans_of(d->transpose_b), trans_of(d->transpose_a),
        M, N, K,
        &alpha,
        (const hipblasHalf*)dev.B, lda,
        (const hipblasHalf*)dev.A, ldb,
        &beta,
        (hipblasHalf*)dev.C, ldc);
    if (s != HIPBLAS_STATUS_SUCCESS) return TC_ERR_INTERNAL;
    tc_status_t fetch = fetch_c(d, sizeof(hipblasHalf), Cp, dev);
    if (fetch != TC_OK) return fetch;
    tc_hip_set_last_kernel("hipblas_hgemm_staged");
    return TC_OK;
}

#endif  /* TC_ENABLE_HIPBLAS */

}  // namespace

/* Public-ish entry point — called from the main tc_gemm dispatcher when
 * the active backend is HIP. Returns TC_ERR_UNSUPPORTED_FAMILY when
 * TC_ENABLE_HIPBLAS is OFF so the dispatcher falls through to the next
 * backend (CBLAS / reference). */
extern "C" TC_HIP_INTERNAL tc_status_t tc_hip_gemm(tc_context* ctx,
                                                    const tc_gemm_desc* desc,
                                                    const tc_buffer* A,
                                                    const tc_buffer* B,
                                                    tc_buffer* C) {
#if !defined(TC_ENABLE_HIPBLAS)
    (void)ctx; (void)desc; (void)A; (void)B; (void)C;
    return TC_ERR_UNSUPPORTED_FAMILY;
#else
    if (!ctx || !desc) return TC_ERR_INVALID_ARG;
    if (!A || !B || !C) return TC_ERR_INVALID_ARG;
    if (desc->a_dtype == TC_DTYPE_F32 && desc->b_dtype == TC_DTYPE_F32 &&
        desc->c_dtype == TC_DTYPE_F32) {
        return hip_gemm_sgemm(desc, A, B, C);
    }
    if (desc->a_dtype == TC_DTYPE_F16 && desc->b_dtype == TC_DTYPE_F16 &&
        desc->c_dtype == TC_DTYPE_F16) {
        return hip_gemm_hgemm(desc, A, B, C);
    }
    return TC_ERR_UNSUPPORTED_DTYPE;
#endif
}
