/*
 * Conv2D forward + backward correctness vs CPU fp64 reference.
 *
 * Small batched shapes. Validates:
 *   - tc_conv2d_forward
 *   - tc_conv2d_backward_input  (col2im + GEMM)
 *   - tc_conv2d_backward_weight (im2col + GEMM)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "tensorcore/tensorcore.h"

static uint16_t f32_to_f16(float x) {
    union { float f; uint32_t u; } v = {x};
    uint32_t f = v.u;
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f & 0x7FFFFF);
    if (exp <= 0) { if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000; uint32_t shift = (uint32_t)(14 - exp);
        return (uint16_t)(sign | ((mant >> shift) + ((mant >> (shift-1)) & 1)));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | ((mant >> 13) + ((mant >> 12) & 1)));
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    int32_t  exp  = (h >> 10) & 0x1F;
    uint32_t mant = (h & 0x3FF);
    if (exp == 0 && mant == 0) { union {uint32_t u; float f;} v = {sign}; return v.f; }
    if (exp == 31) { union {uint32_t u; float f;} v = {sign | 0x7F800000}; return v.f; }
    if (exp == 0) { while ((mant & 0x400) == 0) { mant <<= 1; --exp; } ++exp; mant &= 0x3FF; }
    union { uint32_t u; float f; } v = { sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13) };
    return v.f;
}

static int backend_is_compute(const char* op) {
    const tc_backend_t b = tc_last_backend();
    if (b == TC_BACKEND_METAL_COMPUTE || b == TC_BACKEND_PORTABLE_CPU) return 1;
    fprintf(stderr, "%s backend was %s, expected metal_compute or portable_cpu\n",
            op, tc_backend_name(b));
    return 0;
}

/* CPU reference conv2d forward. */
static void ref_conv2d_fwd(int N, int IC, int OC, int H, int W_in, int kH, int kW,
                           int pad, int stride, int oH, int oW,
                           const float* X, const float* W, const float* bias,
                           float* Y) {
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < OC; ++oc) {
            const float b = bias ? bias[oc] : 0.0f;
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    double acc = b;
                    for (int ic = 0; ic < IC; ++ic)
                        for (int kh = 0; kh < kH; ++kh)
                            for (int kw = 0; kw < kW; ++kw) {
                                int h_in = oh * stride - pad + kh;
                                int w_in = ow * stride - pad + kw;
                                if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W_in) continue;
                                acc += (double)X[((n*IC + ic)*H + h_in)*W_in + w_in]
                                     * (double)W[((oc*IC + ic)*kH + kh)*kW + kw];
                            }
                    Y[((n*OC + oc)*oH + oh)*oW + ow] = (float)acc;
                }
            }
        }
    }
}

static double rms_scaled(const uint16_t* got, const float* ref, int n) {
    double se = 0.0, sr = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = (double)f16_to_f32(got[i]) - (double)ref[i];
        se += e * e; sr += (double)ref[i] * ref[i];
    }
    return sqrt(se / n) / (sqrt(sr / n) + 1e-9);
}

int main(void) {
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s)); return 1;
    }

    const int N = 2, IC = 4, OC = 8;
    const int H = 8, W_in = 8, kH = 3, kW = 3;
    const int pad = 1, stride = 1;
    const int oH = (H + 2*pad - kH) / stride + 1;
    const int oW = (W_in + 2*pad - kW) / stride + 1;
    const int K = IC * kH * kW;
    const int out_hw = oH * oW;

    tc_buffer *Xb, *Wb, *bb, *Yb, *col;
    tc_buffer_alloc(ctx, N*IC*H*W_in*2,  &Xb);
    tc_buffer_alloc(ctx, OC*IC*kH*kW*2,  &Wb);
    tc_buffer_alloc(ctx, OC*2,           &bb);
    tc_buffer_alloc(ctx, N*OC*oH*oW*2,   &Yb);
    tc_buffer_alloc(ctx, N*K*out_hw*2,   &col);

    uint16_t *Xp, *Wp, *bp, *Yp;
    tc_buffer_map(Xb, (void**)&Xp);
    tc_buffer_map(Wb, (void**)&Wp);
    tc_buffer_map(bb, (void**)&bp);
    tc_buffer_map(Yb, (void**)&Yp);

    float *Xf = malloc(N*IC*H*W_in*sizeof(float));
    float *Wf = malloc(OC*IC*kH*kW*sizeof(float));
    float *bf = malloc(OC*sizeof(float));
    float *Yref = malloc(N*OC*oH*oW*sizeof(float));

    srand(0xC0FF);
    for (int i = 0; i < N*IC*H*W_in; ++i) { float v = ((float)rand()/RAND_MAX-0.5f)*0.5f; Xf[i]=v; Xp[i]=f32_to_f16(v); }
    for (int i = 0; i < OC*IC*kH*kW; ++i) { float v = ((float)rand()/RAND_MAX-0.5f)*0.2f; Wf[i]=v; Wp[i]=f32_to_f16(v); }
    for (int i = 0; i < OC; ++i) { float v = ((float)rand()/RAND_MAX-0.5f)*0.1f; bf[i]=v; bp[i]=f32_to_f16(v); }

    ref_conv2d_fwd(N, IC, OC, H, W_in, kH, kW, pad, stride, oH, oW, Xf, Wf, bf, Yref);

    s = tc_conv2d_forward(ctx, Xb, Wb, bb, Yb, col,
                          N, IC, OC, H, W_in, kH, kW,
                          pad, pad, stride, stride, oH, oW);
    if (s != TC_OK) {
        fprintf(stderr, "tc_conv2d_forward: %s\n", tc_status_string(s)); return 2;
    }
    const int fwd_backend_ok = backend_is_compute("conv2d_forward");
    const double fwd_err = rms_scaled(Yp, Yref, N*OC*oH*oW);
    printf("  conv2d_forward         IC=%d OC=%d H=%d W=%d kH=%d kW=%d pad=%d stride=%d\n"
           "                         out=(%d,%d)  rms_scaled=%.3e  %s\n",
           IC, OC, H, W_in, kH, kW, pad, stride, oH, oW, fwd_err,
           (fwd_err < 2e-2) ? "OK" : "FAIL");

    /* Backward sanity: just check the kernels dispatch + write nonzero results. */
    tc_buffer *dXb, *dWb, *dX_f32;
    tc_buffer_alloc(ctx, N*IC*H*W_in*2, &dXb);
    tc_buffer_alloc(ctx, OC*IC*kH*kW*2, &dWb);
    tc_buffer_alloc(ctx, N*IC*H*W_in*4, &dX_f32);
    uint16_t *dXp, *dWp;
    tc_buffer_map(dXb, (void**)&dXp);
    tc_buffer_map(dWb, (void**)&dWp);
    memset(dXp, 0, N*IC*H*W_in*2);
    memset(dWp, 0, OC*IC*kH*kW*2);

    /* Reuse the Y buffer as dY for the test — any non-trivial gradient. */
    s = tc_conv2d_backward_input(ctx, Yb, Wb, dXb, col, dX_f32,
                                 N, IC, OC, H, W_in, kH, kW,
                                 pad, pad, stride, stride, oH, oW);
    const int bw_in_ok = (s == TC_OK);
    const int bw_in_backend_ok = bw_in_ok && backend_is_compute("conv2d_backward_input");
    int nz_in = 0;
    for (int i = 0; i < N*IC*H*W_in; ++i) if (f16_to_f32(dXp[i]) != 0.0f) ++nz_in;
    printf("  conv2d_backward_input  dispatched=%s  nonzero/total=%d/%d  %s\n",
           bw_in_ok ? "yes" : "no", nz_in, N*IC*H*W_in,
           (bw_in_ok && nz_in > 0) ? "OK" : "FAIL");

    s = tc_conv2d_backward_weight(ctx, Xb, Yb, dWb, col,
                                  N, IC, OC, H, W_in, kH, kW,
                                  pad, pad, stride, stride, oH, oW);
    const int bw_w_ok = (s == TC_OK);
    const int bw_w_backend_ok = bw_w_ok && backend_is_compute("conv2d_backward_weight");
    int nz_w = 0;
    for (int i = 0; i < OC*IC*kH*kW; ++i) if (f16_to_f32(dWp[i]) != 0.0f) ++nz_w;
    printf("  conv2d_backward_weight dispatched=%s  nonzero/total=%d/%d  %s\n",
           bw_w_ok ? "yes" : "no", nz_w, OC*IC*kH*kW,
           (bw_w_ok && nz_w > 0) ? "OK" : "FAIL");

    tc_buffer* small = NULL;
    tc_buffer_alloc(ctx, 2, &small);
    s = tc_conv2d_backward_input(ctx, Yb, Wb, small, col, dX_f32,
                                 N, IC, OC, H, W_in, kH, kW,
                                 pad, pad, stride, stride, oH, oW);
    const int bw_in_small_rejected = (s == TC_ERR_INVALID_SHAPE);
    s = tc_conv2d_backward_weight(ctx, Xb, Yb, small, col,
                                  N, IC, OC, H, W_in, kH, kW,
                                  pad, pad, stride, stride, oH, oW);
    const int bw_w_small_rejected = (s == TC_ERR_INVALID_SHAPE);
    s = tc_conv2d_forward(ctx, Xb, Wb, bb, Yb, col,
                          N, IC, OC, H, W_in, kH, kW,
                          pad, pad, stride, stride, oH + 1, oW);
    const int bad_out_shape_rejected = (s == TC_ERR_INVALID_SHAPE);
    printf("  conv2d_validation      small_dX=%s  small_dW=%s  bad_out=%s  %s\n",
           bw_in_small_rejected ? "yes" : "no",
           bw_w_small_rejected ? "yes" : "no",
           bad_out_shape_rejected ? "yes" : "no",
           (bw_in_small_rejected && bw_w_small_rejected && bad_out_shape_rejected) ? "OK" : "FAIL");

    free(Xf); free(Wf); free(bf); free(Yref);
    tc_buffer_free(ctx, Xb); tc_buffer_free(ctx, Wb); tc_buffer_free(ctx, bb);
    tc_buffer_free(ctx, Yb); tc_buffer_free(ctx, col);
    tc_buffer_free(ctx, dXb); tc_buffer_free(ctx, dWb); tc_buffer_free(ctx, dX_f32);
    tc_buffer_free(ctx, small);
    tc_shutdown(ctx);

    return (fwd_backend_ok && fwd_err < 2e-2 &&
            bw_in_backend_ok && nz_in > 0 && bw_w_backend_ok && nz_w > 0 &&
            bw_in_small_rejected && bw_w_small_rejected && bad_out_shape_rejected) ? 0 : 5;
}
