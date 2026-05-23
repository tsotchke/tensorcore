/*
 * tensorcore - CUDA GEMM via cuBLAS.
 *
 * Gated on TC_ENABLE_CUDA=ON. Wires tc_gemm into cuBLAS sgemm / hgemm /
 * gemmEx for tensor-core paths. Returns TC_ERR_UNSUPPORTED_FAMILY when
 * CUDA is not compiled in, so the public dispatch falls through.
 *
 * Row-major C ABI to column-major cuBLAS: the standard transpose trick
 * C^T = B^T * A^T applies (swap M/N, swap A/B, flip transposes).
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/cuda.h"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>

#if defined(__GNUC__) || defined(__clang__)
#  define TC_CUDA_INTERNAL __attribute__((visibility("hidden")))
#else
#  define TC_CUDA_INTERNAL
#endif

#if defined(TC_ENABLE_CUDA)
#  include <cuda_runtime.h>
#  include <cublas_v2.h>
#  include <cuda_fp16.h>
#endif

extern "C" TC_CUDA_INTERNAL void tc_cuda_set_last_kernel(const char* name);

namespace {

#if defined(TC_ENABLE_CUDA)

struct CublasHandle {
    cublasHandle_t h = nullptr;
    bool initialized = false;
};

CublasHandle& handle() {
    static CublasHandle hh;
    if (!hh.initialized) {
        if (cublasCreate(&hh.h) == CUBLAS_STATUS_SUCCESS) {
            hh.initialized = true;
            /* Enable tensor-core paths whenever shape allows. */
            cublasSetMathMode(hh.h, CUBLAS_TENSOR_OP_MATH);
        }
    }
    return hh;
}

cublasOperation_t trans_of(bool t) { return t ? CUBLAS_OP_T : CUBLAS_OP_N; }

bool current_device_props(cudaDeviceProp* prop) {
    if (!prop) return false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    if (cudaGetDeviceProperties(prop, dev) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return true;
}

bool current_device_supports_bf16(void) {
    cudaDeviceProp prop = {};
    return current_device_props(&prop) && prop.major >= 8;
}

bool current_device_supports_int8_tensor_core(void) {
    cudaDeviceProp prop = {};
    return current_device_props(&prop) &&
           (prop.major > 7 || (prop.major == 7 && prop.minor >= 2));
}

bool float_is_i32(float v) {
    return std::isfinite(v) &&
           v >= (float)std::numeric_limits<int32_t>::min() &&
           v <= (float)std::numeric_limits<int32_t>::max() &&
           std::floor(v) == v;
}

/* Fallback staging for host-only pointers, including tc_buffer_from_ptr
 * wrappers around caller-owned memory. Buffers allocated while CUDA GEMM is
 * active use managed memory and bypass this path. */
struct DeviceTriple {
    void* A = nullptr;
    void* B = nullptr;
    void* C = nullptr;
    ~DeviceTriple() {
        if (A) cudaFree(A);
        if (B) cudaFree(B);
        if (C) cudaFree(C);
    }
    bool ok() const { return A && B && C; }
};

/* Detect whether a pointer is CUDA-managed/device memory. When it is,
 * cuBLAS dereferences directly without host/device staging. */
bool ptr_is_cuda_managed(const void* p) {
    if (!p) return false;
    cudaPointerAttributes attr = {};
    if (cudaPointerGetAttributes(&attr, p) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return attr.type == cudaMemoryTypeManaged || attr.type == cudaMemoryTypeDevice;
}

tc_status_t cuda_gemm_sgemm(const tc_gemm_desc* d,
                            const tc_buffer* A, const tc_buffer* B,
                            tc_buffer* C) {
    auto& hh = handle();
    if (!hh.initialized) return TC_ERR_INTERNAL;

    void *Ap = nullptr, *Bp = nullptr, *Cp = nullptr;
    if (tc_buffer_map((tc_buffer*)A, &Ap) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map((tc_buffer*)B, &Bp) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map(C, &Cp) != TC_OK) return TC_ERR_INTERNAL;

    /* Sizes for upload: use the dimensions the descriptor declares. */
    const int a_rows = d->transpose_a ? d->K : d->M;
    const int a_cols = d->transpose_a ? d->M : d->K;
    const int b_rows = d->transpose_b ? d->N : d->K;
    const int b_cols = d->transpose_b ? d->K : d->N;
    const int lda_h = d->lda ? d->lda : a_cols;
    const int ldb_h = d->ldb ? d->ldb : b_cols;
    const int ldc_h = d->ldc ? d->ldc : d->N;
    const size_t a_bytes = (size_t)a_rows * lda_h * sizeof(float);
    const size_t b_bytes = (size_t)b_rows * ldb_h * sizeof(float);
    const size_t c_bytes = (size_t)d->M * ldc_h * sizeof(float);

    /* Fast path: managed memory; cuBLAS dereferences user pointers directly. */
    const bool managed = ptr_is_cuda_managed(Ap) && ptr_is_cuda_managed(Bp) &&
                          ptr_is_cuda_managed(Cp);
    if (managed) {
        const int M = d->N, N = d->M, K = d->K;
        cublasStatus_t cs = cublasSgemm(
            hh.h, trans_of(d->transpose_b), trans_of(d->transpose_a),
            M, N, K,
            &d->alpha,
            (const float*)Bp, ldb_h,
            (const float*)Ap, lda_h,
            &d->beta,
            (float*)Cp, ldc_h);
        if (cs != CUBLAS_STATUS_SUCCESS) return TC_ERR_INTERNAL;
        if (cudaDeviceSynchronize() != cudaSuccess) return TC_ERR_INTERNAL;
        tc_cuda_set_last_kernel("cublas_sgemm_managed");
        return TC_OK;
    }

    /* Slow path: host buffers; upload to device. */
    DeviceTriple dev;
    if (cudaMalloc(&dev.A, a_bytes) != cudaSuccess) return TC_ERR_ALLOC;
    if (cudaMalloc(&dev.B, b_bytes) != cudaSuccess) return TC_ERR_ALLOC;
    if (cudaMalloc(&dev.C, c_bytes) != cudaSuccess) return TC_ERR_ALLOC;
    if (cudaMemcpy(dev.A, Ap, a_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return TC_ERR_INTERNAL;
    if (cudaMemcpy(dev.B, Bp, b_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return TC_ERR_INTERNAL;
    if (d->beta != 0.0f) {
        if (cudaMemcpy(dev.C, Cp, c_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return TC_ERR_INTERNAL;
    }

    /* Row-major to column-major: compute (B^T x A^T) instead. */
    const int M = d->N, N = d->M, K = d->K;

    cublasStatus_t s = cublasSgemm(
        hh.h, trans_of(d->transpose_b), trans_of(d->transpose_a),
        M, N, K,
        &d->alpha,
        (const float*)dev.B, ldb_h,
        (const float*)dev.A, lda_h,
        &d->beta,
        (float*)dev.C, ldc_h);
    if (s != CUBLAS_STATUS_SUCCESS) return TC_ERR_INTERNAL;

    /* Download C back to host. */
    if (cudaMemcpy(Cp, dev.C, c_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return TC_ERR_INTERNAL;
    if (cudaDeviceSynchronize() != cudaSuccess) return TC_ERR_INTERNAL;

    tc_cuda_set_last_kernel("cublas_sgemm");
    return TC_OK;
}

tc_status_t cuda_gemm_hgemm(const tc_gemm_desc* d,
                            const tc_buffer* A, const tc_buffer* B,
                            tc_buffer* C) {
    /* fp16 via cublasGemmEx with CUBLAS_GEMM_DEFAULT_TENSOR_OP; picks
     * tensor-core path on sm_70+ (Volta/Turing/Ampere/Ada/Hopper). */
    auto& hh = handle();
    if (!hh.initialized) return TC_ERR_INTERNAL;

    void *Ap = nullptr, *Bp = nullptr, *Cp = nullptr;
    if (tc_buffer_map((tc_buffer*)A, &Ap) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map((tc_buffer*)B, &Bp) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map(C, &Cp) != TC_OK) return TC_ERR_INTERNAL;

    const int a_rows = d->transpose_a ? d->K : d->M;
    const int a_cols = d->transpose_a ? d->M : d->K;
    const int b_rows = d->transpose_b ? d->N : d->K;
    const int b_cols = d->transpose_b ? d->K : d->N;
    const int lda_h = d->lda ? d->lda : a_cols;
    const int ldb_h = d->ldb ? d->ldb : b_cols;
    const int ldc_h = d->ldc ? d->ldc : d->N;
    const size_t a_bytes = (size_t)a_rows * lda_h * sizeof(uint16_t);
    const size_t b_bytes = (size_t)b_rows * ldb_h * sizeof(uint16_t);
    const size_t c_bytes = (size_t)d->M * ldc_h * sizeof(uint16_t);

    /* Fast path: managed memory. */
    const bool managed_h = ptr_is_cuda_managed(Ap) && ptr_is_cuda_managed(Bp) &&
                            ptr_is_cuda_managed(Cp);
    const float alpha_f = d->alpha;
    const float beta_f = d->beta;
    const int M = d->N, N = d->M, K = d->K;

    /* fp16 accumulation opt-in via TC_CUDA_FP16_ACCUM=1. Uses fp16 as the
     * cuBLAS compute type — ~2× the tensor-core throughput on Ampere
     * (~142 TFLOPS vs ~71 TFLOPS at fp32-accum) at the cost of numerical
     * range. Default stays fp32-accum for correctness. */
    const char* fp16_accum_env = std::getenv("TC_CUDA_FP16_ACCUM");
    const bool fp16_accum = fp16_accum_env && fp16_accum_env[0] == '1';
    const cudaDataType_t compute_type = fp16_accum ? CUDA_R_16F : CUDA_R_32F;
    const __half alpha_h = __float2half(d->alpha);
    const __half beta_h  = __float2half(d->beta);
    const void* alpha_p = fp16_accum ? (const void*)&alpha_h : (const void*)&alpha_f;
    const void* beta_p  = fp16_accum ? (const void*)&beta_h  : (const void*)&beta_f;

    if (managed_h) {
        cublasStatus_t cs = cublasGemmEx(
            hh.h, trans_of(d->transpose_b), trans_of(d->transpose_a),
            M, N, K,
            alpha_p,
            Bp, CUDA_R_16F, ldb_h,
            Ap, CUDA_R_16F, lda_h,
            beta_p,
            Cp, CUDA_R_16F, ldc_h,
            compute_type,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (cs != CUBLAS_STATUS_SUCCESS) return TC_ERR_INTERNAL;
        if (cudaDeviceSynchronize() != cudaSuccess) return TC_ERR_INTERNAL;
        tc_cuda_set_last_kernel(fp16_accum
            ? "cublas_gemmex_fp16_tensorop_managed_fp16accum"
            : "cublas_gemmex_fp16_tensorop_managed");
        return TC_OK;
    }

    /* Slow path: host buffers. */
    DeviceTriple dev;
    if (cudaMalloc(&dev.A, a_bytes) != cudaSuccess) return TC_ERR_ALLOC;
    if (cudaMalloc(&dev.B, b_bytes) != cudaSuccess) return TC_ERR_ALLOC;
    if (cudaMalloc(&dev.C, c_bytes) != cudaSuccess) return TC_ERR_ALLOC;
    if (cudaMemcpy(dev.A, Ap, a_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return TC_ERR_INTERNAL;
    if (cudaMemcpy(dev.B, Bp, b_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return TC_ERR_INTERNAL;
    if (d->beta != 0.0f) {
        if (cudaMemcpy(dev.C, Cp, c_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return TC_ERR_INTERNAL;
    }

    cublasStatus_t s = cublasGemmEx(
        hh.h, trans_of(d->transpose_b), trans_of(d->transpose_a),
        M, N, K,
        alpha_p,
        dev.B, CUDA_R_16F, ldb_h,
        dev.A, CUDA_R_16F, lda_h,
        beta_p,
        dev.C, CUDA_R_16F, ldc_h,
        compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) return TC_ERR_INTERNAL;

    if (cudaMemcpy(Cp, dev.C, c_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return TC_ERR_INTERNAL;
    if (cudaDeviceSynchronize() != cudaSuccess) return TC_ERR_INTERNAL;

    tc_cuda_set_last_kernel(fp16_accum
        ? "cublas_gemmex_fp16_tensorop_staged_fp16accum"
        : "cublas_gemmex_fp16_tensorop_staged");
    return TC_OK;
}

/* bf16 GEMM via cublasGemmEx, fp32 compute accumulator.
 *
 * bf16 has the same dynamic range as fp32 (8-bit exponent) but lower
 * precision (7-bit mantissa vs 23). Standard choice for training large
 * models: Ampere/Hopper tensor cores accelerate bf16 input + fp32 accum
 * at ~141 TFLOPS on RTX 3090. */
tc_status_t cuda_gemm_bf16(const tc_gemm_desc* d,
                           const tc_buffer* A, const tc_buffer* B,
                           tc_buffer* C) {
#if !defined(CUDART_VERSION) || CUDART_VERSION < 11000
    (void)d; (void)A; (void)B; (void)C;
    return TC_ERR_UNSUPPORTED_FAMILY;
#else
    if (d->accum_dtype != TC_DTYPE_F32) return TC_ERR_UNSUPPORTED_DTYPE;
    if (!current_device_supports_bf16()) return TC_ERR_UNSUPPORTED_FAMILY;

    auto& hh = handle();
    if (!hh.initialized) return TC_ERR_INTERNAL;

    void *Ap = nullptr, *Bp = nullptr, *Cp = nullptr;
    if (tc_buffer_map((tc_buffer*)A, &Ap) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map((tc_buffer*)B, &Bp) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map(C, &Cp) != TC_OK) return TC_ERR_INTERNAL;

    const int a_rows = d->transpose_a ? d->K : d->M;
    const int a_cols = d->transpose_a ? d->M : d->K;
    const int b_rows = d->transpose_b ? d->N : d->K;
    const int b_cols = d->transpose_b ? d->K : d->N;
    const int lda_h = d->lda ? d->lda : a_cols;
    const int ldb_h = d->ldb ? d->ldb : b_cols;
    const int ldc_h = d->ldc ? d->ldc : d->N;
    const size_t a_bytes = (size_t)a_rows * lda_h * sizeof(uint16_t);
    const size_t b_bytes = (size_t)b_rows * ldb_h * sizeof(uint16_t);
    const size_t c_bytes = (size_t)d->M * ldc_h * sizeof(uint16_t);

    const bool managed = ptr_is_cuda_managed(Ap) && ptr_is_cuda_managed(Bp) &&
                          ptr_is_cuda_managed(Cp);
    const float alpha_f = d->alpha;
    const float beta_f = d->beta;
    const int M = d->N, N = d->M, K = d->K;

    auto run = [&](const void* dA, const void* dB, void* dC) -> cublasStatus_t {
        /* CUBLAS_GEMM_DEFAULT lets the driver pick the best path including
         * tensor cores; CUBLAS_GEMM_DEFAULT_TENSOR_OP is deprecated in CUDA
         * 11+ and may reject bf16 on some cublas versions. */
        return cublasGemmEx(
            hh.h, trans_of(d->transpose_b), trans_of(d->transpose_a),
            M, N, K,
            &alpha_f,
            dB, CUDA_R_16BF, ldb_h,
            dA, CUDA_R_16BF, lda_h,
            &beta_f,
            dC, CUDA_R_16BF, ldc_h,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT);
    };

    if (managed) {
        if (run(Ap, Bp, Cp) != CUBLAS_STATUS_SUCCESS) return TC_ERR_INTERNAL;
        if (cudaDeviceSynchronize() != cudaSuccess) return TC_ERR_INTERNAL;
        tc_cuda_set_last_kernel("cublas_gemmex_bf16_tensorop_managed");
        return TC_OK;
    }

    DeviceTriple dev;
    if (cudaMalloc(&dev.A, a_bytes) != cudaSuccess) return TC_ERR_ALLOC;
    if (cudaMalloc(&dev.B, b_bytes) != cudaSuccess) return TC_ERR_ALLOC;
    if (cudaMalloc(&dev.C, c_bytes) != cudaSuccess) return TC_ERR_ALLOC;
    if (cudaMemcpy(dev.A, Ap, a_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return TC_ERR_INTERNAL;
    if (cudaMemcpy(dev.B, Bp, b_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return TC_ERR_INTERNAL;
    if (d->beta != 0.0f) {
        if (cudaMemcpy(dev.C, Cp, c_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return TC_ERR_INTERNAL;
    }
    if (run(dev.A, dev.B, dev.C) != CUBLAS_STATUS_SUCCESS) return TC_ERR_INTERNAL;
    if (cudaMemcpy(Cp, dev.C, c_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return TC_ERR_INTERNAL;
    if (cudaDeviceSynchronize() != cudaSuccess) return TC_ERR_INTERNAL;
    tc_cuda_set_last_kernel("cublas_gemmex_bf16_tensorop_staged");
    return TC_OK;
#endif
}

/* int8 GEMM via cublasGemmEx with int32 accumulator. Used for quantized
 * inference: input/output int8, accumulate in int32 to avoid overflow.
 *
 * Caller-side responsibility: scale factors. The tensorcore int8 GEMM is
 * the inner-loop primitive; quantization scales live one layer up.
 *
 * Tensor-core path requires K % 16 == 0 on most cuBLAS versions (alignment
 * requirement). For shapes that don't meet this, cuBLAS may return error;
 * caller should fall back to fp16 or fp32 path. */
tc_status_t cuda_gemm_i8(const tc_gemm_desc* d,
                         const tc_buffer* A, const tc_buffer* B,
                         tc_buffer* C) {
    if (d->accum_dtype != TC_DTYPE_I32) return TC_ERR_UNSUPPORTED_DTYPE;
    if (!current_device_supports_int8_tensor_core()) return TC_ERR_UNSUPPORTED_FAMILY;
    if (!float_is_i32(d->alpha) || !float_is_i32(d->beta)) return TC_ERR_INVALID_ARG;

    auto& hh = handle();
    if (!hh.initialized) return TC_ERR_INTERNAL;
    if (d->K % 16 != 0) return TC_ERR_INVALID_SHAPE;

    void *Ap = nullptr, *Bp = nullptr, *Cp = nullptr;
    if (tc_buffer_map((tc_buffer*)A, &Ap) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map((tc_buffer*)B, &Bp) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map(C, &Cp) != TC_OK) return TC_ERR_INTERNAL;

    const int a_rows = d->transpose_a ? d->K : d->M;
    const int b_rows = d->transpose_b ? d->N : d->K;
    const int a_cols = d->transpose_a ? d->M : d->K;
    const int b_cols = d->transpose_b ? d->K : d->N;
    const int lda_h = d->lda ? d->lda : a_cols;
    const int ldb_h = d->ldb ? d->ldb : b_cols;
    const int ldc_h = d->ldc ? d->ldc : d->N;
    const size_t a_bytes = (size_t)a_rows * lda_h * sizeof(int8_t);
    const size_t b_bytes = (size_t)b_rows * ldb_h * sizeof(int8_t);
    const size_t c_bytes = (size_t)d->M * ldc_h * sizeof(int32_t);

    const bool managed = ptr_is_cuda_managed(Ap) && ptr_is_cuda_managed(Bp) &&
                          ptr_is_cuda_managed(Cp);
    const int32_t alpha_i = (int32_t)d->alpha;
    const int32_t beta_i = (int32_t)d->beta;
    const int M = d->N, N = d->M, K = d->K;

    auto run = [&](const void* dA, const void* dB, void* dC) -> cublasStatus_t {
        return cublasGemmEx(
            hh.h, trans_of(d->transpose_b), trans_of(d->transpose_a),
            M, N, K,
            &alpha_i,
            dB, CUDA_R_8I, ldb_h,
            dA, CUDA_R_8I, lda_h,
            &beta_i,
            dC, CUDA_R_32I, ldc_h,
            CUDA_R_32I,
            CUBLAS_GEMM_DEFAULT);
    };

    if (managed) {
        if (run(Ap, Bp, Cp) != CUBLAS_STATUS_SUCCESS) return TC_ERR_INTERNAL;
        if (cudaDeviceSynchronize() != cudaSuccess) return TC_ERR_INTERNAL;
        tc_cuda_set_last_kernel("cublas_gemmex_i8_tensorop_managed");
        return TC_OK;
    }

    DeviceTriple dev;
    if (cudaMalloc(&dev.A, a_bytes) != cudaSuccess) return TC_ERR_ALLOC;
    if (cudaMalloc(&dev.B, b_bytes) != cudaSuccess) return TC_ERR_ALLOC;
    if (cudaMalloc(&dev.C, c_bytes) != cudaSuccess) return TC_ERR_ALLOC;
    if (cudaMemcpy(dev.A, Ap, a_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return TC_ERR_INTERNAL;
    if (cudaMemcpy(dev.B, Bp, b_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return TC_ERR_INTERNAL;
    if (beta_i != 0) {
        if (cudaMemcpy(dev.C, Cp, c_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return TC_ERR_INTERNAL;
    }
    if (run(dev.A, dev.B, dev.C) != CUBLAS_STATUS_SUCCESS) return TC_ERR_INTERNAL;
    if (cudaMemcpy(Cp, dev.C, c_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return TC_ERR_INTERNAL;
    if (cudaDeviceSynchronize() != cudaSuccess) return TC_ERR_INTERNAL;
    tc_cuda_set_last_kernel("cublas_gemmex_i8_tensorop_staged");
    return TC_OK;
}
#endif  /* TC_ENABLE_CUDA */

}  // namespace

extern "C" TC_CUDA_INTERNAL tc_status_t tc_cuda_gemm(tc_context* ctx,
                                                     const tc_gemm_desc* desc,
                                                     const tc_buffer* A,
                                                     const tc_buffer* B,
                                                     tc_buffer* C) {
#if !defined(TC_ENABLE_CUDA)
    (void)ctx; (void)desc; (void)A; (void)B; (void)C;
    return TC_ERR_UNSUPPORTED_FAMILY;
#else
    if (!ctx || !desc || !A || !B || !C) return TC_ERR_INVALID_ARG;
    if (desc->a_dtype == TC_DTYPE_F32 && desc->b_dtype == TC_DTYPE_F32 &&
        desc->c_dtype == TC_DTYPE_F32) {
        return cuda_gemm_sgemm(desc, A, B, C);
    }
    if (desc->a_dtype == TC_DTYPE_F16 && desc->b_dtype == TC_DTYPE_F16 &&
        desc->c_dtype == TC_DTYPE_F16) {
        return cuda_gemm_hgemm(desc, A, B, C);
    }
    if (desc->a_dtype == TC_DTYPE_BF16 && desc->b_dtype == TC_DTYPE_BF16 &&
        desc->c_dtype == TC_DTYPE_BF16) {
        return cuda_gemm_bf16(desc, A, B, C);
    }
    if (desc->a_dtype == TC_DTYPE_I8 && desc->b_dtype == TC_DTYPE_I8 &&
        desc->c_dtype == TC_DTYPE_I32) {
        return cuda_gemm_i8(desc, A, B, C);
    }
    return TC_ERR_UNSUPPORTED_DTYPE;
#endif
}
