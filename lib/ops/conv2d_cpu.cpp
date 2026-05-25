/*
 * tensorcore — CPU Conv2D forward + backward (im2col + GEMM).
 *
 * Standard im2col-then-GEMM. Memory-inefficient for large feature maps
 * but correctness-first and dispatch-clean: each Conv2D becomes a
 * call into the same BLAS-backed fp32 GEMM path used by tc_gemm.
 *
 * Layout: NCHW for input X, weight W is [out_C × in_C × kH × kW] (PyTorch
 * convention). The im2col buffer is the standard [in_C * kH * kW] × [out_H * out_W]
 * matrix, written into scratch_col which the caller is required to provide
 * (matches the Metal kernel ABI).
 *
 * Perf is bounded by the underlying GEMM. On old-donkey with MKL,
 * a typical [B=1, C=64, H=W=224, kH=kW=3] Conv at 56×56 output is
 * ~600 MFLOPS (im2col is the bottleneck, not GEMM).
 */

#include "tensorcore/tensorcore.h"
#include "../core/internal.h"
#include "../core/cpu_float.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(TC_HAS_CBLAS)
#  if defined(__APPLE__)
#    include <Accelerate/Accelerate.h>
#  else
#    include <cblas.h>
#  endif
#endif

namespace {

/* Direct fp32 GEMM helper for the backward conv path. Avoids the
 * tc_buffer_alloc/free/memcpy dance that runs through tensorcore's buffer
 * pool — keeps memory ownership entirely on the stack/std::vector side. */
void direct_sgemm_f32(bool transpose_a, bool transpose_b,
                      int M, int N, int K,
                      float alpha,
                      const float* A, int lda,
                      const float* B, int ldb,
                      float beta,
                      float* C, int ldc) {
#if defined(TC_HAS_CBLAS)
#  if defined(__APPLE__) && defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdeprecated-declarations"
#  endif
    const CBLAS_TRANSPOSE ta = transpose_a ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE tb = transpose_b ? CblasTrans : CblasNoTrans;
    cblas_sgemm(CblasRowMajor, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#  if defined(__APPLE__) && defined(__clang__)
#    pragma clang diagnostic pop
#  endif
#else
    /* Reference triple loop. Conv backward shapes are small (≤ tens of MB);
     * the BLAS path is the fast one, this is the correctness floor. */
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const size_t idx = (size_t)m * ldc + n;
            float acc = (beta == 0.0f) ? 0.0f : beta * C[idx];
            for (int k = 0; k < K; ++k) {
                const float av = transpose_a ? A[(size_t)k * lda + m] : A[(size_t)m * lda + k];
                const float bv = transpose_b ? B[(size_t)n * ldb + k] : B[(size_t)k * ldb + n];
                acc += alpha * av * bv;
            }
            C[idx] = acc;
        }
    }
#endif
}

/* Convert an fp16 [n] array into a packed fp32 host vector. */
void f16_to_f32_block(const uint16_t* src, size_t n, float* dst) {
    for (size_t i = 0; i < n; ++i) dst[i] = tc_cpu_f16_to_f32(src[i]);
}

bool checked_mul(size_t a, size_t b, size_t* out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    *out = a * b;
    return true;
}

bool conv_dims_valid(int batch, int in_channels, int out_channels,
                     int H, int W_in, int kH, int kW,
                     int pad_h, int pad_w,
                     int stride_h, int stride_w,
                     int out_H, int out_W) {
    if (batch <= 0 || in_channels <= 0 || out_channels <= 0 ||
        H <= 0 || W_in <= 0 || kH <= 0 || kW <= 0 ||
        pad_h < 0 || pad_w < 0 || stride_h <= 0 || stride_w <= 0 ||
        out_H <= 0 || out_W <= 0) {
        return false;
    }
    const int64_t expect_H_num = (int64_t)H + 2 * (int64_t)pad_h - (int64_t)kH;
    const int64_t expect_W_num = (int64_t)W_in + 2 * (int64_t)pad_w - (int64_t)kW;
    if (expect_H_num < 0 || expect_W_num < 0) return false;
    if (expect_H_num / stride_h + 1 != (int64_t)out_H) return false;
    if (expect_W_num / stride_w + 1 != (int64_t)out_W) return false;

    const int64_t out_hw = (int64_t)out_H * out_W;
    const int64_t k_elems = (int64_t)in_channels * kH * kW;
    const int64_t in_elems = (int64_t)in_channels * H * W_in;
    const int64_t out_elems = (int64_t)out_channels * out_hw;
    const int64_t max_int = std::numeric_limits<int>::max();
    return out_hw <= max_int && k_elems <= max_int &&
           in_elems <= max_int && out_elems <= max_int;
}

bool conv_bytes(int batch, int in_channels, int out_channels,
                int H, int W_in, int kH, int kW,
                int out_H, int out_W,
                size_t* x_bytes, size_t* weight_bytes,
                size_t* y_bytes, size_t* scratch_col_bytes,
                size_t* dx_f32_bytes) {
    size_t x_elems = 0, w_elems = 0, y_elems = 0, k_elems = 0;
    size_t out_hw = 0, col_elems = 0, tmp = 0;
    if (!checked_mul((size_t)H, (size_t)W_in, &tmp) ||
        !checked_mul((size_t)batch, (size_t)in_channels, &x_elems) ||
        !checked_mul(x_elems, tmp, &x_elems) ||
        !checked_mul((size_t)out_channels, (size_t)in_channels, &w_elems) ||
        !checked_mul(w_elems, (size_t)kH, &w_elems) ||
        !checked_mul(w_elems, (size_t)kW, &w_elems) ||
        !checked_mul((size_t)out_H, (size_t)out_W, &out_hw) ||
        !checked_mul((size_t)batch, (size_t)out_channels, &y_elems) ||
        !checked_mul(y_elems, out_hw, &y_elems) ||
        !checked_mul((size_t)in_channels, (size_t)kH, &k_elems) ||
        !checked_mul(k_elems, (size_t)kW, &k_elems) ||
        !checked_mul((size_t)batch, k_elems, &col_elems) ||
        !checked_mul(col_elems, out_hw, &col_elems)) {
        return false;
    }
    return checked_mul(x_elems, sizeof(uint16_t), x_bytes) &&
           checked_mul(w_elems, sizeof(uint16_t), weight_bytes) &&
           checked_mul(y_elems, sizeof(uint16_t), y_bytes) &&
           checked_mul(col_elems, sizeof(uint16_t), scratch_col_bytes) &&
           checked_mul(x_elems, sizeof(float), dx_f32_bytes);
}

tc_status_t validate_conv_common(int batch, int in_channels, int out_channels,
                                 int H, int W_in, int kH, int kW,
                                 int pad_h, int pad_w,
                                 int stride_h, int stride_w,
                                 int out_H, int out_W,
                                 size_t* x_bytes, size_t* weight_bytes,
                                 size_t* y_bytes, size_t* scratch_col_bytes,
                                 size_t* dx_f32_bytes) {
    if (!conv_dims_valid(batch, in_channels, out_channels, H, W_in, kH, kW,
                         pad_h, pad_w, stride_h, stride_w, out_H, out_W)) {
        return TC_ERR_INVALID_SHAPE;
    }
    if (!conv_bytes(batch, in_channels, out_channels, H, W_in, kH, kW,
                    out_H, out_W, x_bytes, weight_bytes, y_bytes,
                    scratch_col_bytes, dx_f32_bytes)) {
        return TC_ERR_INVALID_SHAPE;
    }
    return TC_OK;
}

tc_status_t alloc_mapped_buffer(tc_context* ctx, size_t bytes,
                                tc_buffer** out, void** out_ptr) {
    *out = nullptr;
    *out_ptr = nullptr;
    tc_status_t s = tc_buffer_alloc(ctx, bytes, out);
    if (s != TC_OK) return s;
    s = tc_buffer_map(*out, out_ptr);
    if (s != TC_OK) {
        tc_buffer_free(ctx, *out);
        *out = nullptr;
        *out_ptr = nullptr;
    }
    return s;
}

/* im2col into scratch_col (fp16). Layout:
 *   scratch_col[(ic * kH * kW + kh * kW + kw) * (out_H * out_W) + oh * out_W + ow]
 *     = X[batch_idx, ic, oh*stride_h + kh - pad_h, ow*stride_w + kw - pad_w]
 *     (zero-padded for out-of-bounds reads). */
void im2col_fp16(const uint16_t* X_b, int in_C, int H, int W_in,
                 int kH, int kW, int pad_h, int pad_w,
                 int stride_h, int stride_w,
                 int out_H, int out_W,
                 uint16_t* col) {
    const int out_HW = out_H * out_W;
    const int chans = in_C * kH * kW;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int c = 0; c < chans; ++c) {
        for (int oh = 0; oh < out_H; ++oh) {
            const int ic = c / (kH * kW);
            const int kh = (c % (kH * kW)) / kW;
            const int kw = c % kW;
            const int ih_base = oh * stride_h + kh - pad_h;
            uint16_t* col_row = col + (size_t)c * out_HW + (size_t)oh * out_W;
            for (int ow = 0; ow < out_W; ++ow) {
                const int iw = ow * stride_w + kw - pad_w;
                const bool in_bounds = (ih_base >= 0 && ih_base < H && iw >= 0 && iw < W_in);
                col_row[ow] = in_bounds
                    ? X_b[((size_t)ic * H + ih_base) * W_in + iw]
                    : (uint16_t)0;  /* fp16 zero */
            }
        }
    }
}

}  // namespace

extern "C" tc_status_t tc_conv2d_forward(tc_context* ctx,
                                          const tc_buffer* X,
                                          const tc_buffer* weight,
                                          const tc_buffer* bias,
                                          tc_buffer* Y,
                                          tc_buffer* scratch_col,
                                          int batch, int in_channels, int out_channels,
                                          int H, int W_in, int kH, int kW,
                                          int pad_h, int pad_w,
                                          int stride_h, int stride_w,
                                          int out_H, int out_W) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!X || !weight || !Y || !scratch_col)
        return TC_ERR_INVALID_ARG;

    size_t x_bytes = 0, weight_bytes = 0, y_bytes = 0;
    size_t scratch_col_bytes = 0, dx_f32_bytes = 0;
    tc_status_t s = validate_conv_common(batch, in_channels, out_channels,
                                         H, W_in, kH, kW, pad_h, pad_w,
                                         stride_h, stride_w, out_H, out_W,
                                         &x_bytes, &weight_bytes, &y_bytes,
                                         &scratch_col_bytes, &dx_f32_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, X, x_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, weight, weight_bytes);
    if (s != TC_OK) return s;
    if (bias) {
        s = tc_buffer_validate(ctx, bias, (size_t)out_channels * sizeof(uint16_t));
        if (s != TC_OK) return s;
    }
    s = tc_buffer_validate(ctx, Y, y_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, scratch_col, scratch_col_bytes);
    if (s != TC_OK) return s;

    void *Xp, *Wp, *Yp, *colp, *bp = nullptr;
    s = tc_buffer_map((tc_buffer*)X, &Xp);
    if (s != TC_OK) return s;
    s = tc_buffer_map((tc_buffer*)weight, &Wp);
    if (s != TC_OK) return s;
    s = tc_buffer_map(Y, &Yp);
    if (s != TC_OK) return s;
    s = tc_buffer_map(scratch_col, &colp);
    if (s != TC_OK) return s;
    if (bias) {
        s = tc_buffer_map((tc_buffer*)bias, &bp);
        if (s != TC_OK) return s;
    }

    const uint16_t* Xd = (const uint16_t*)Xp;
    const uint16_t* Wd = (const uint16_t*)Wp;
    uint16_t* Yd = (uint16_t*)Yp;
    uint16_t* cold = (uint16_t*)colp;
    const uint16_t* bd = (const uint16_t*)bp;

    const int in_size = in_channels * H * W_in;
    const int out_size = out_channels * out_H * out_W;
    const int out_HW = out_H * out_W;
    const int K = in_channels * kH * kW;

    for (int b = 0; b < batch; ++b) {
        /* im2col into the scratch buffer. */
        im2col_fp16(Xd + (size_t)b * in_size, in_channels, H, W_in,
                    kH, kW, pad_h, pad_w, stride_h, stride_w, out_H, out_W, cold);

        /* GEMM: Y[b, oc, *] = W[oc, *] @ col[*, *]
         * Shape: [out_channels x K] @ [K x out_HW] = [out_channels x out_HW]. */
        tc_gemm_desc d = {};
        d.M = out_channels; d.N = out_HW; d.K = K;
        d.a_dtype = TC_DTYPE_F16; d.b_dtype = TC_DTYPE_F16; d.c_dtype = TC_DTYPE_F16;
        d.accum_dtype = TC_DTYPE_F32;
        d.alpha = 1.0f; d.beta = 0.0f;
        d.lda = K; d.ldb = out_HW; d.ldc = out_HW;

        /* Need to wrap our pointers as tc_buffer*. The cleanest path is to
         * alloc per-call buffers, copy in, run, copy out. Acceptable here
         * because Conv2D on CPU is not the perf bottleneck anyway. */
        tc_buffer *Wb = nullptr, *colb = nullptr, *Yb = nullptr;
        void *Wbp, *colbp, *Ybp;
        s = alloc_mapped_buffer(ctx, (size_t)out_channels * K * sizeof(uint16_t), &Wb, &Wbp);
        if (s != TC_OK) return s;
        s = alloc_mapped_buffer(ctx, (size_t)K * out_HW * sizeof(uint16_t), &colb, &colbp);
        if (s != TC_OK) { tc_buffer_free(ctx, Wb); return s; }
        s = alloc_mapped_buffer(ctx, (size_t)out_channels * out_HW * sizeof(uint16_t), &Yb, &Ybp);
        if (s != TC_OK) { tc_buffer_free(ctx, Wb); tc_buffer_free(ctx, colb); return s; }
        std::memcpy(Wbp, Wd, (size_t)out_channels * K * sizeof(uint16_t));
        std::memcpy(colbp, cold, (size_t)K * out_HW * sizeof(uint16_t));
        s = tc_gemm(ctx, &d, Wb, colb, Yb);
        if (s == TC_OK) {
            std::memcpy(Yd + (size_t)b * out_size, Ybp,
                        (size_t)out_channels * out_HW * sizeof(uint16_t));
        }
        tc_buffer_free(ctx, Wb);
        tc_buffer_free(ctx, colb);
        tc_buffer_free(ctx, Yb);
        if (s != TC_OK) return s;

        /* Add bias[oc] to each output position. */
        if (bd) {
            uint16_t* Yb_ptr = Yd + (size_t)b * out_size;
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static)
#endif
            for (int oc = 0; oc < out_channels; ++oc) {
                const float bv = tc_cpu_f16_to_f32(bd[oc]);
                uint16_t* row = Yb_ptr + (size_t)oc * out_HW;
                for (int i = 0; i < out_HW; ++i) {
                    row[i] = tc_cpu_f32_to_f16(tc_cpu_f16_to_f32(row[i]) + bv);
                }
            }
        }
    }
    return tc_record_dispatch("tc_conv2d_forward", TC_BACKEND_PORTABLE_CPU, TC_OK);
}

/* ----------------------------------------------------------------------- *
 * Conv2D backward (col2im + GEMM).
 *
 * Conv2D forward computes Y = W ∘ X via im2col → GEMM. The two backward
 * passes are:
 *
 *   dX = W^T @ dY → col2im                  (gradient w.r.t. input)
 *   dW = sum_b dY @ im2col(X[b])^T          (gradient w.r.t. weight)
 *
 * Both delegate the dense linear algebra to direct fp32 GEMM / CBLAS;
 * only the im2col / col2im transformations live here.
 * ----------------------------------------------------------------------- */

namespace {

/* fp32-input variant of col2im — used when the upstream gradient is already
 * in fp32 (e.g. from a direct cblas_sgemm result). */
void col2im_acc_fp32(const float* col, int in_C, int H, int W_in,
                     int kH, int kW, int pad_h, int pad_w,
                     int stride_h, int stride_w,
                     int out_H, int out_W,
                     float* dX_acc) {
    const int out_HW = out_H * out_W;
    for (int c = 0; c < in_C * kH * kW; ++c) {
        const int ic = c / (kH * kW);
        const int kh = (c % (kH * kW)) / kW;
        const int kw = c % kW;
        for (int oh = 0; oh < out_H; ++oh) {
            const int ih = oh * stride_h + kh - pad_h;
            if (ih < 0 || ih >= H) continue;
            const float* col_row = col + (size_t)c * out_HW + (size_t)oh * out_W;
            float* dst_row = dX_acc + ((size_t)ic * H + ih) * W_in;
            for (int ow = 0; ow < out_W; ++ow) {
                const int iw = ow * stride_w + kw - pad_w;
                if (iw < 0 || iw >= W_in) continue;
                dst_row[iw] += col_row[ow];
            }
        }
    }
}

}  // namespace

extern "C" tc_status_t tc_conv2d_backward_input(tc_context* ctx,
                                                 const tc_buffer* dY,
                                                 const tc_buffer* weight,
                                                 tc_buffer* dX,
                                                 tc_buffer* scratch_col,
                                                 tc_buffer* scratch_dX_f32,
                                                 int batch, int in_channels, int out_channels,
                                                 int H, int W_in, int kH, int kW,
                                                 int pad_h, int pad_w,
                                                 int stride_h, int stride_w,
                                                 int out_H, int out_W) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!dY || !weight || !dX || !scratch_col)
        return TC_ERR_INVALID_ARG;

    size_t x_bytes = 0, weight_bytes = 0, y_bytes = 0;
    size_t scratch_col_bytes = 0, dx_f32_bytes = 0;
    tc_status_t s = validate_conv_common(batch, in_channels, out_channels,
                                         H, W_in, kH, kW, pad_h, pad_w,
                                         stride_h, stride_w, out_H, out_W,
                                         &x_bytes, &weight_bytes, &y_bytes,
                                         &scratch_col_bytes, &dx_f32_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, dY, y_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, weight, weight_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, dX, x_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, scratch_col, scratch_col_bytes);
    if (s != TC_OK) return s;
    if (scratch_dX_f32) {
        s = tc_buffer_validate(ctx, scratch_dX_f32, dx_f32_bytes);
        if (s != TC_OK) return s;
    }

    void *dYp, *Wp, *dXp, *dX_f32p = nullptr;
    s = tc_buffer_map((tc_buffer*)dY, &dYp);
    if (s != TC_OK) return s;
    s = tc_buffer_map((tc_buffer*)weight, &Wp);
    if (s != TC_OK) return s;
    s = tc_buffer_map(dX, &dXp);
    if (s != TC_OK) return s;
    if (scratch_dX_f32) {
        s = tc_buffer_map(scratch_dX_f32, &dX_f32p);
        if (s != TC_OK) return s;
    }

    const uint16_t* dYd = (const uint16_t*)dYp;
    const uint16_t* Wd = (const uint16_t*)Wp;
    uint16_t* dXd = (uint16_t*)dXp;

    const int out_HW = out_H * out_W;
    const int K = in_channels * kH * kW;
    const int in_size = in_channels * H * W_in;
    const int out_size = out_channels * out_H * out_W;

    std::vector<float> dX_acc;
    float* dX_acc_data = nullptr;
    if (dX_f32p) {
        dX_acc_data = (float*)dX_f32p;
    } else {
        dX_acc.resize((size_t)in_size, 0.0f);
        dX_acc_data = dX_acc.data();
    }

    /* Pre-convert W to fp32 host buffer once (reused across batches). */
    std::vector<float> W_f32((size_t)out_channels * K);
    f16_to_f32_block(Wd, (size_t)out_channels * K, W_f32.data());

    /* Per-batch fp32 dY and col-gradient scratch. */
    std::vector<float> dY_f32((size_t)out_channels * out_HW);
    std::vector<float> col_grad_f32((size_t)K * out_HW);

    for (int b = 0; b < batch; ++b) {
        /* dY[b]: fp16 → fp32 */
        f16_to_f32_block(dYd + (size_t)b * out_size,
                         (size_t)out_channels * out_HW, dY_f32.data());

        /* col_grad = W^T @ dY[b]
         *   W is [out_C × K]; W^T is [K × out_C]
         *   dY[b] is [out_C × out_HW]
         *   col_grad is [K × out_HW]
         * Direct sgemm with transpose_a=true on W. */
        direct_sgemm_f32(true, false,
                         K, out_HW, out_channels,
                         1.0f,
                         W_f32.data(), K,           /* W [out_C × K], lda = K (row-major) */
                         dY_f32.data(), out_HW,     /* dY [out_C × out_HW] */
                         0.0f,
                         col_grad_f32.data(), out_HW);

        /* col2im accumulates col_grad → dX_acc. */
        std::memset(dX_acc_data, 0, (size_t)in_size * sizeof(float));
        col2im_acc_fp32(col_grad_f32.data(), in_channels, H, W_in,
                        kH, kW, pad_h, pad_w, stride_h, stride_w,
                        out_H, out_W, dX_acc_data);

        /* Convert to fp16 output. */
        uint16_t* dX_b = dXd + (size_t)b * in_size;
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < in_size; ++i) {
            dX_b[i] = tc_cpu_f32_to_f16(dX_acc_data[i]);
        }
    }
    return tc_record_dispatch("tc_conv2d_backward_input", TC_BACKEND_PORTABLE_CPU, TC_OK);
}

extern "C" tc_status_t tc_conv2d_backward_weight(tc_context* ctx,
                                                  const tc_buffer* X,
                                                  const tc_buffer* dY,
                                                  tc_buffer* dW,
                                                  tc_buffer* scratch_col,
                                                  int batch, int in_channels, int out_channels,
                                                  int H, int W_in, int kH, int kW,
                                                  int pad_h, int pad_w,
                                                  int stride_h, int stride_w,
                                                  int out_H, int out_W) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!X || !dY || !dW || !scratch_col)
        return TC_ERR_INVALID_ARG;

    size_t x_bytes = 0, weight_bytes = 0, y_bytes = 0;
    size_t scratch_col_bytes = 0, dx_f32_bytes = 0;
    tc_status_t s = validate_conv_common(batch, in_channels, out_channels,
                                         H, W_in, kH, kW, pad_h, pad_w,
                                         stride_h, stride_w, out_H, out_W,
                                         &x_bytes, &weight_bytes, &y_bytes,
                                         &scratch_col_bytes, &dx_f32_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, X, x_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, dY, y_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, dW, weight_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, scratch_col, scratch_col_bytes);
    if (s != TC_OK) return s;

    void *Xp, *dYp, *dWp, *colp;
    s = tc_buffer_map((tc_buffer*)X, &Xp);
    if (s != TC_OK) return s;
    s = tc_buffer_map((tc_buffer*)dY, &dYp);
    if (s != TC_OK) return s;
    s = tc_buffer_map(dW, &dWp);
    if (s != TC_OK) return s;
    s = tc_buffer_map(scratch_col, &colp);
    if (s != TC_OK) return s;

    const uint16_t* Xd = (const uint16_t*)Xp;
    const uint16_t* dYd = (const uint16_t*)dYp;
    uint16_t* dWd = (uint16_t*)dWp;
    uint16_t* cold = (uint16_t*)colp;

    const int out_HW = out_H * out_W;
    const int K = in_channels * kH * kW;
    const int in_size = in_channels * H * W_in;
    const int out_size = out_channels * out_H * out_W;

    /* fp32 accumulator for dW: shape [out_C × K]. */
    std::vector<float> dW_acc((size_t)out_channels * K, 0.0f);

    /* fp32 scratch reused across batches. */
    std::vector<float> col_f32((size_t)K * out_HW);
    std::vector<float> dY_f32((size_t)out_channels * out_HW);
    std::vector<float> dW_b_f32((size_t)out_channels * K);

    for (int b = 0; b < batch; ++b) {
        /* col = im2col(X[b]); convert to fp32 host scratch. */
        im2col_fp16(Xd + (size_t)b * in_size, in_channels, H, W_in,
                    kH, kW, pad_h, pad_w, stride_h, stride_w, out_H, out_W, cold);
        f16_to_f32_block(cold, (size_t)K * out_HW, col_f32.data());
        f16_to_f32_block(dYd + (size_t)b * out_size,
                         (size_t)out_channels * out_HW, dY_f32.data());

        /* dW_b = dY[b] @ col^T :  [out_C × K] = [out_C × out_HW] @ [out_HW × K]
         * col is laid out [K × out_HW]; we want col^T, hence transpose_b=true. */
        direct_sgemm_f32(false, true,
                         out_channels, K, out_HW,
                         1.0f,
                         dY_f32.data(), out_HW,
                         col_f32.data(), out_HW,
                         0.0f,
                         dW_b_f32.data(), K);

        /* Accumulate into the running dW_acc. */
        for (size_t i = 0; i < (size_t)out_channels * K; ++i) {
            dW_acc[i] += dW_b_f32[i];
        }
    }

    /* fp32 accumulator → fp16 output */
    for (size_t i = 0; i < (size_t)out_channels * K; ++i) {
        dWd[i] = tc_cpu_f32_to_f16(dW_acc[i]);
    }
    return tc_record_dispatch("tc_conv2d_backward_weight", TC_BACKEND_PORTABLE_CPU, TC_OK);
}
