/*
 * tensorcore — CPU Conv2D forward + backward (im2col + GEMM).
 *
 * Standard im2col-then-GEMM. Memory-inefficient for large feature maps
 * but correctness-first and dispatch-clean: each Conv2D becomes a
 * call into tc_gemm, which already has the AVX2 / OpenBLAS / MKL fast
 * paths wired in.
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
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

inline int output_dim(int in_dim, int kern, int pad, int stride) {
    return (in_dim + 2 * pad - kern) / stride + 1;
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
    if (!X || !weight || !Y || !scratch_col || batch <= 0 || in_channels <= 0
        || out_channels <= 0 || H <= 0 || W_in <= 0 || kH <= 0 || kW <= 0
        || stride_h <= 0 || stride_w <= 0 || out_H <= 0 || out_W <= 0)
        return TC_ERR_INVALID_ARG;
    if (out_H != output_dim(H, kH, pad_h, stride_h)) return TC_ERR_INVALID_ARG;
    if (out_W != output_dim(W_in, kW, pad_w, stride_w)) return TC_ERR_INVALID_ARG;

    void *Xp, *Wp, *Yp, *colp, *bp = nullptr;
    tc_buffer_map((tc_buffer*)X, &Xp);
    tc_buffer_map((tc_buffer*)weight, &Wp);
    tc_buffer_map(Y, &Yp);
    tc_buffer_map(scratch_col, &colp);
    if (bias) tc_buffer_map((tc_buffer*)bias, &bp);

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
        tc_buffer_alloc(ctx, (size_t)out_channels * K * sizeof(uint16_t), &Wb);
        tc_buffer_alloc(ctx, (size_t)K * out_HW * sizeof(uint16_t), &colb);
        tc_buffer_alloc(ctx, (size_t)out_channels * out_HW * sizeof(uint16_t), &Yb);
        void *Wbp, *colbp, *Ybp;
        tc_buffer_map(Wb, &Wbp);
        tc_buffer_map(colb, &colbp);
        tc_buffer_map(Yb, &Ybp);
        std::memcpy(Wbp, Wd, (size_t)out_channels * K * sizeof(uint16_t));
        std::memcpy(colbp, cold, (size_t)K * out_HW * sizeof(uint16_t));
        tc_status_t s = tc_gemm(ctx, &d, Wb, colb, Yb);
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
    return TC_OK;
}

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
    /* col2im backward: dX = col2im(W^T @ dY).
     * For correctness-first, deferred to a follow-up implementation;
     * returns UNSUPPORTED until then so models that use it fail loud. */
    (void)dY; (void)weight; (void)dX; (void)scratch_col; (void)scratch_dX_f32;
    (void)batch; (void)in_channels; (void)out_channels; (void)H; (void)W_in;
    (void)kH; (void)kW; (void)pad_h; (void)pad_w; (void)stride_h; (void)stride_w;
    (void)out_H; (void)out_W;
    return ctx ? TC_ERR_UNSUPPORTED_FAMILY : TC_ERR_NOT_INITIALIZED;
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
    /* dW = sum_b im2col(X[b]) @ dY[b]^T. Deferred to follow-up; returns
     * UNSUPPORTED for now. */
    (void)X; (void)dY; (void)dW; (void)scratch_col;
    (void)batch; (void)in_channels; (void)out_channels; (void)H; (void)W_in;
    (void)kH; (void)kW; (void)pad_h; (void)pad_w; (void)stride_h; (void)stride_w;
    (void)out_H; (void)out_W;
    return ctx ? TC_ERR_UNSUPPORTED_FAMILY : TC_ERR_NOT_INITIALIZED;
}
