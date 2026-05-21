/*
 * tensorcore - portable CPU unsupported-op stubs.
 *
 * The CPU backend intentionally starts with buffers, streams, GGUF loading,
 * distributed-single, and GEMM. Other public ABI entry points return a stable
 * unsupported status so Python/ctypes and downstream FFI imports can bind the
 * full surface without requiring Metal symbols.
 */

#include "tensorcore/tensorcore.h"

extern "C" tc_status_t tc_attention_forward(tc_context* ctx,
                                             const tc_attention_desc* desc,
                                             const tc_buffer* Q,
                                             const tc_buffer* K,
                                             const tc_buffer* V,
                                             tc_buffer* O,
                                             tc_buffer* LSE) {
    (void)desc; (void)Q; (void)K; (void)V; (void)O; (void)LSE;
    return ctx ? TC_ERR_UNSUPPORTED_FAMILY : TC_ERR_NOT_INITIALIZED;
}

extern "C" tc_status_t tc_attention_forward_async(tc_context* ctx,
                                                   const tc_attention_desc* desc,
                                                   const tc_buffer* Q,
                                                   const tc_buffer* K,
                                                   const tc_buffer* V,
                                                   tc_buffer* O,
                                                   tc_buffer* LSE,
                                                   tc_stream* stream) {
    (void)desc; (void)Q; (void)K; (void)V; (void)O; (void)LSE; (void)stream;
    return ctx ? TC_ERR_UNSUPPORTED_FAMILY : TC_ERR_NOT_INITIALIZED;
}

extern "C" tc_status_t tc_attention_backward(tc_context* ctx,
                                              const tc_attention_desc* desc,
                                              const tc_buffer* Q,
                                              const tc_buffer* K,
                                              const tc_buffer* V,
                                              const tc_buffer* O,
                                              const tc_buffer* dO,
                                              const tc_buffer* LSE,
                                              tc_buffer* dQ,
                                              tc_buffer* dK,
                                              tc_buffer* dV) {
    (void)desc; (void)Q; (void)K; (void)V; (void)O; (void)dO;
    (void)LSE; (void)dQ; (void)dK; (void)dV;
    return ctx ? TC_ERR_UNSUPPORTED_FAMILY : TC_ERR_NOT_INITIALIZED;
}

extern "C" tc_status_t tc_rmsnorm_forward(tc_context* ctx,
                                           const tc_buffer* X,
                                           const tc_buffer* gamma,
                                           tc_buffer* Y,
                                           tc_buffer* rstd_out,
                                           int N, int D, float eps) {
    (void)X; (void)gamma; (void)Y; (void)rstd_out; (void)N; (void)D; (void)eps;
    return ctx ? TC_ERR_UNSUPPORTED_FAMILY : TC_ERR_NOT_INITIALIZED;
}

extern "C" tc_status_t tc_rmsnorm_backward(tc_context* ctx,
                                            const tc_buffer* X,
                                            const tc_buffer* gamma,
                                            const tc_buffer* dY,
                                            const tc_buffer* rstd,
                                            tc_buffer* dX,
                                            tc_buffer* dgamma,
                                            int N, int D) {
    (void)X; (void)gamma; (void)dY; (void)rstd; (void)dX; (void)dgamma; (void)N; (void)D;
    return ctx ? TC_ERR_UNSUPPORTED_FAMILY : TC_ERR_NOT_INITIALIZED;
}

extern "C" tc_status_t tc_layernorm_forward(tc_context* ctx,
                                             const tc_buffer* X,
                                             const tc_buffer* gamma,
                                             const tc_buffer* beta,
                                             tc_buffer* Y,
                                             tc_buffer* mean_out,
                                             tc_buffer* rstd_out,
                                             int N, int D, float eps) {
    (void)X; (void)gamma; (void)beta; (void)Y; (void)mean_out; (void)rstd_out;
    (void)N; (void)D; (void)eps;
    return ctx ? TC_ERR_UNSUPPORTED_FAMILY : TC_ERR_NOT_INITIALIZED;
}

extern "C" tc_status_t tc_layernorm_backward(tc_context* ctx,
                                              const tc_buffer* X,
                                              const tc_buffer* gamma,
                                              const tc_buffer* dY,
                                              const tc_buffer* mean,
                                              const tc_buffer* rstd,
                                              tc_buffer* dX,
                                              int N, int D) {
    (void)X; (void)gamma; (void)dY; (void)mean; (void)rstd; (void)dX; (void)N; (void)D;
    return ctx ? TC_ERR_UNSUPPORTED_FAMILY : TC_ERR_NOT_INITIALIZED;
}

extern "C" tc_status_t tc_rope_forward(tc_context* ctx,
                                        tc_buffer* X,
                                        const tc_buffer* cos_t,
                                        const tc_buffer* sin_t,
                                        int batch, int heads, int seq, int head_dim) {
    (void)X; (void)cos_t; (void)sin_t; (void)batch; (void)heads; (void)seq; (void)head_dim;
    return ctx ? TC_ERR_UNSUPPORTED_FAMILY : TC_ERR_NOT_INITIALIZED;
}

extern "C" tc_status_t tc_swiglu_forward(tc_context* ctx,
                                          const tc_buffer* gate,
                                          const tc_buffer* up,
                                          tc_buffer* out,
                                          int n) {
    (void)gate; (void)up; (void)out; (void)n;
    return ctx ? TC_ERR_UNSUPPORTED_FAMILY : TC_ERR_NOT_INITIALIZED;
}

extern "C" tc_status_t tc_swiglu_backward(tc_context* ctx,
                                           const tc_buffer* gate,
                                           const tc_buffer* up,
                                           const tc_buffer* dout,
                                           tc_buffer* dgate,
                                           tc_buffer* dup,
                                           int n) {
    (void)gate; (void)up; (void)dout; (void)dgate; (void)dup; (void)n;
    return ctx ? TC_ERR_UNSUPPORTED_FAMILY : TC_ERR_NOT_INITIALIZED;
}

extern "C" tc_status_t tc_softmax_forward(tc_context* ctx,
                                           const tc_buffer* X,
                                           tc_buffer* Y,
                                           int N, int D) {
    (void)X; (void)Y; (void)N; (void)D;
    return ctx ? TC_ERR_UNSUPPORTED_FAMILY : TC_ERR_NOT_INITIALIZED;
}

extern "C" tc_status_t tc_softmax_backward(tc_context* ctx,
                                            const tc_buffer* Y,
                                            const tc_buffer* dY,
                                            tc_buffer* dX,
                                            int N, int D) {
    (void)Y; (void)dY; (void)dX; (void)N; (void)D;
    return ctx ? TC_ERR_UNSUPPORTED_FAMILY : TC_ERR_NOT_INITIALIZED;
}

extern "C" tc_status_t tc_adamw_step(tc_context* ctx,
                                      tc_buffer* params_fp32,
                                      tc_buffer* m_fp32,
                                      tc_buffer* v_fp32,
                                      const tc_buffer* grads,
                                      tc_dtype_t grad_dtype,
                                      int n,
                                      float lr, float beta1, float beta2, float eps,
                                      float wd, float bc1, float bc2) {
    (void)params_fp32; (void)m_fp32; (void)v_fp32; (void)grads; (void)grad_dtype;
    (void)n; (void)lr; (void)beta1; (void)beta2; (void)eps; (void)wd; (void)bc1; (void)bc2;
    return ctx ? TC_ERR_UNSUPPORTED_FAMILY : TC_ERR_NOT_INITIALIZED;
}

extern "C" tc_status_t tc_fused_rmsnorm_gemv(tc_context* ctx,
                                              const tc_buffer* X,
                                              const tc_buffer* gamma,
                                              const tc_buffer* W,
                                              tc_buffer* Y,
                                              int M, int N, int K, float eps) {
    (void)X; (void)gamma; (void)W; (void)Y; (void)M; (void)N; (void)K; (void)eps;
    return ctx ? TC_ERR_UNSUPPORTED_FAMILY : TC_ERR_NOT_INITIALIZED;
}

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
    (void)X; (void)weight; (void)bias; (void)Y; (void)scratch_col;
    (void)batch; (void)in_channels; (void)out_channels; (void)H; (void)W_in;
    (void)kH; (void)kW; (void)pad_h; (void)pad_w; (void)stride_h; (void)stride_w;
    (void)out_H; (void)out_W;
    return ctx ? TC_ERR_UNSUPPORTED_FAMILY : TC_ERR_NOT_INITIALIZED;
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
    (void)X; (void)dY; (void)dW; (void)scratch_col;
    (void)batch; (void)in_channels; (void)out_channels; (void)H; (void)W_in;
    (void)kH; (void)kW; (void)pad_h; (void)pad_w; (void)stride_h; (void)stride_w;
    (void)out_H; (void)out_W;
    return ctx ? TC_ERR_UNSUPPORTED_FAMILY : TC_ERR_NOT_INITIALIZED;
}
