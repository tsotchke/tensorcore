#ifndef TENSORCORE_CONV_H
#define TENSORCORE_CONV_H

#include "tensorcore/status.h"
#include "tensorcore/dtype.h"
#include "tensorcore/device.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Conv2D forward (fp16).
 *
 *   Y = conv2d(X, W) + bias
 *
 *   X    : [batch, in_channels, H, W]                          (fp16)
 *   W    : [out_channels, in_channels, kH, kW]                 (fp16)
 *   bias : [out_channels] or NULL                              (fp16)
 *   Y    : [batch, out_channels, out_H, out_W]                 (fp16)
 *   scratch_col: workspace buffer of size batch * (in_channels*kH*kW) * out_H * out_W * 2 (fp16)
 *
 * Implementation: im2col + tc_gemm. v0.1 supports stride, padding; dilation = 1.
 * out_H = floor((H + 2*pad_h - kH) / stride_h) + 1
 * out_W = floor((W + 2*pad_w - kW) / stride_w) + 1  (caller supplies).
 */
tc_status_t tc_conv2d_forward(tc_context* ctx,
                              const tc_buffer* X,
                              const tc_buffer* weight,
                              const tc_buffer* bias,   /* nullable */
                              tc_buffer*       Y,
                              tc_buffer*       scratch_col,
                              int batch, int in_channels, int out_channels,
                              int H, int W_in, int kH, int kW,
                              int pad_h, int pad_w,
                              int stride_h, int stride_w,
                              int out_H, int out_W);

/* Conv2D backward — input gradient.
 *   dX = col2im(W^T @ dY).
 *
 * Caller supplies:
 *   scratch_col   : [batch, K, out_hw] half  workspace (same shape as fwd)
 *   scratch_dX_f32: [batch, in_channels, H, W_in] float, atomic accumulation
 *                   (must be zero-initialized before each call).
 *
 * Implementation: dCol = W^T @ dY via tc_gemm with transpose_a=true, then
 * tc_col2im_atomic_f32 scatters-with-add into scratch_dX_f32, finalize copies
 * to fp16 dX. */
tc_status_t tc_conv2d_backward_input(tc_context* ctx,
                                     const tc_buffer* dY,
                                     const tc_buffer* weight,
                                     tc_buffer*       dX,
                                     tc_buffer*       scratch_col,
                                     tc_buffer*       scratch_dX_f32,
                                     int batch, int in_channels, int out_channels,
                                     int H, int W_in, int kH, int kW,
                                     int pad_h, int pad_w,
                                     int stride_h, int stride_w,
                                     int out_H, int out_W);

/* Conv2D backward — weight gradient.
 *   dW[oc, K] = sum_n dY[n, oc, out_hw] @ col[n, K, out_hw]^T
 *
 * Caller supplies the im2col scratch from the forward (reusable). */
tc_status_t tc_conv2d_backward_weight(tc_context* ctx,
                                      const tc_buffer* X,
                                      const tc_buffer* dY,
                                      tc_buffer*       dW,
                                      tc_buffer*       scratch_col,
                                      int batch, int in_channels, int out_channels,
                                      int H, int W_in, int kH, int kW,
                                      int pad_h, int pad_w,
                                      int stride_h, int stride_w,
                                      int out_H, int out_W);

#ifdef __cplusplus
}
#endif
#endif
