/*
 * tensorcore — 2:4 structured-sparse GEMM (CPU pruning + fallback compute).
 *
 * The pruning helper (`tc_sparse_24_prune`) is CPU-native — it's pure host
 * arithmetic (top-2 magnitudes per 4-element block) and runs identically on
 * every platform. The compute helper (`tc_sparse_24_gemm`) here is the
 * dense fallback used when the host lacks cusparseLt: bit-correct dense
 * GEMM that ignores the sparsity pattern (still produces the right answer
 * because the zeroed entries of B contribute zero). On CUDA hosts with
 * cusparseLt, lib/cuda/sparse_gemm_cuda.cpp overrides this with the real
 * tensor-core 2:4 path.
 *
 * Pattern check (`tc_sparse_24_check`) is also CPU-native; useful for
 * portability tests that need to confirm a caller's pruning is valid
 * regardless of whether cusparseLt is available.
 */

#include "tensorcore/sparse_gemm.h"
#include "tensorcore/tensorcore.h"
#include "../core/internal.h"
#include "../core/cpu_float.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define TC_SPARSE_WEAK __attribute__((weak))
#else
#  define TC_SPARSE_WEAK
#endif

namespace {

inline float read_elem(const void* base, int idx, tc_dtype_t dtype) {
    if (dtype == TC_DTYPE_F32) {
        return ((const float*)base)[idx];
    }
    /* fp16 / bf16 — convert to fp32 for magnitude compare. fp16 uses the
     * shared CPU helper; bf16 is a simple bit shift. */
    if (dtype == TC_DTYPE_F16) {
        return tc_cpu_f16_to_f32(((const uint16_t*)base)[idx]);
    }
    if (dtype == TC_DTYPE_BF16) {
        uint32_t u = ((uint32_t)((const uint16_t*)base)[idx]) << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    }
    return 0.0f;
}

inline void write_elem(void* base, int idx, tc_dtype_t dtype, float v) {
    if (dtype == TC_DTYPE_F32) {
        ((float*)base)[idx] = v;
    } else if (dtype == TC_DTYPE_F16) {
        ((uint16_t*)base)[idx] = tc_cpu_f32_to_f16(v);
    } else if (dtype == TC_DTYPE_BF16) {
        uint32_t u; std::memcpy(&u, &v, 4);
        ((uint16_t*)base)[idx] = (uint16_t)(u >> 16);
    }
}

bool supported_dtype(tc_dtype_t d) {
    return d == TC_DTYPE_F32 || d == TC_DTYPE_F16 || d == TC_DTYPE_BF16;
}

}  // namespace

extern "C" tc_status_t tc_sparse_24_prune(tc_context* ctx,
                                           tc_buffer* W,
                                           tc_dtype_t dtype,
                                           int rows, int cols) {
    if (!ctx || !W || rows <= 0 || cols <= 0) return TC_ERR_INVALID_ARG;
    if ((cols & 3) != 0) return TC_ERR_INVALID_SHAPE;
    if (!supported_dtype(dtype)) return TC_ERR_UNSUPPORTED_DTYPE;
    const size_t elem_size = tc_dtype_size(dtype);
    const size_t bytes = (size_t)rows * cols * elem_size;
    tc_status_t s = tc_buffer_validate(ctx, W, bytes);
    if (s != TC_OK) return s;
    void* p = nullptr;
    if (tc_buffer_map(W, &p) != TC_OK) return TC_ERR_INTERNAL;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (rows > 1)
#endif
    for (int r = 0; r < rows; ++r) {
        for (int c0 = 0; c0 < cols; c0 += 4) {
            const int base = r * cols + c0;
            /* Read 4 magnitudes, find indices of the two smallest, zero them. */
            float vals[4];
            for (int k = 0; k < 4; ++k) vals[k] = read_elem(p, base + k, dtype);
            float abs_v[4] = {std::fabs(vals[0]), std::fabs(vals[1]),
                              std::fabs(vals[2]), std::fabs(vals[3])};
            /* Find two smallest indices. */
            int min1 = 0, min2 = 1;
            if (abs_v[min2] < abs_v[min1]) std::swap(min1, min2);
            for (int k = 2; k < 4; ++k) {
                if (abs_v[k] < abs_v[min1]) {
                    min2 = min1;
                    min1 = k;
                } else if (abs_v[k] < abs_v[min2]) {
                    min2 = k;
                }
            }
            /* Zero the two smallest-magnitude entries (keep top 2). */
            write_elem(p, base + min1, dtype, 0.0f);
            write_elem(p, base + min2, dtype, 0.0f);
        }
    }
    return TC_OK;
}

extern "C" tc_status_t tc_sparse_24_check(tc_context* ctx,
                                           const tc_buffer* W,
                                           tc_dtype_t dtype,
                                           int rows, int cols) {
    if (!ctx || !W || rows <= 0 || cols <= 0) return TC_ERR_INVALID_ARG;
    if ((cols & 3) != 0) return TC_ERR_INVALID_SHAPE;
    if (!supported_dtype(dtype)) return TC_ERR_UNSUPPORTED_DTYPE;
    const size_t elem_size = tc_dtype_size(dtype);
    tc_status_t s = tc_buffer_validate(ctx, W, (size_t)rows * cols * elem_size);
    if (s != TC_OK) return s;
    void* p = nullptr;
    if (tc_buffer_map((tc_buffer*)W, &p) != TC_OK) return TC_ERR_INTERNAL;

    for (int r = 0; r < rows; ++r) {
        for (int c0 = 0; c0 < cols; c0 += 4) {
            int nz = 0;
            for (int k = 0; k < 4; ++k) {
                if (read_elem(p, r * cols + c0 + k, dtype) != 0.0f) ++nz;
            }
            if (nz > 2) return TC_ERR_INVALID_ARG;
        }
    }
    return TC_OK;
}

/* Weak default — the CUDA backend will override this when cusparseLt is
 * available. For now (and on any non-CUDA host) we do a correct dense
 * GEMM through tc_gemm; the zeros in B contribute zero to the output. */
extern "C" TC_SPARSE_WEAK tc_status_t tc_sparse_24_gemm(
        tc_context* ctx,
        const tc_buffer* A,
        const tc_buffer* B,
        tc_buffer* C,
        int M, int N, int K,
        tc_dtype_t a_dtype, tc_dtype_t b_dtype, tc_dtype_t c_dtype,
        float alpha, float beta) {
    if (!ctx || !A || !B || !C || M <= 0 || N <= 0 || K <= 0)
        return TC_ERR_INVALID_ARG;
    if ((K & 3) != 0) return TC_ERR_INVALID_SHAPE;

    tc_gemm_desc d;
    std::memset(&d, 0, sizeof(d));
    d.M = M; d.N = N; d.K = K;
    d.alpha = alpha; d.beta = beta;
    d.a_dtype = a_dtype; d.b_dtype = b_dtype; d.c_dtype = c_dtype;
    /* Accum follows dtype: fp32 accum for fp16/bf16, fp32 for fp32. */
    d.accum_dtype = TC_DTYPE_F32;
    return tc_gemm(ctx, &d, A, B, C);
}

/* Default to 0 — CUDA + cusparseLt override sets to 1. */
extern "C" TC_SPARSE_WEAK int tc_sparse_24_available(void) {
    return 0;
}
