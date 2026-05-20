/*
 * tensorcore — Conv2D via im2col + GEMM.
 *
 * Forward: y[n, oc, h_out, w_out] = sum_{ic, kh, kw} x[n, ic, h_in, w_in] *
 *                                                    w[oc, ic, kh, kw] + b[oc]
 *
 * Implementation: im2col transforms the input into a matrix shape that lets
 * us call tc_gemm. The im2col kernel here writes
 *   col[k * out_hw + n_oh_ow] = x[n, ic, h_in, w_in]
 * with k = ic*kH*kW + kh*kW + kw, suitable as the A operand of a GEMM where
 * B is the weight matrix [oc, ic*kH*kW] and C is [oc, out_hw].
 */

#include <metal_stdlib>
using namespace metal;

kernel void tc_im2col_f16(
    device const half*  X         [[buffer(0)]],
    device       half*  col       [[buffer(1)]],
    constant uint& batch          [[buffer(2)]],
    constant uint& in_channels    [[buffer(3)]],
    constant uint& H              [[buffer(4)]],
    constant uint& W              [[buffer(5)]],
    constant uint& kH             [[buffer(6)]],
    constant uint& kW             [[buffer(7)]],
    constant int&  pad_h          [[buffer(8)]],
    constant int&  pad_w          [[buffer(9)]],
    constant uint& stride_h       [[buffer(10)]],
    constant uint& stride_w       [[buffer(11)]],
    constant uint& out_H          [[buffer(12)]],
    constant uint& out_W          [[buffer(13)]],
    uint3 gid                     [[thread_position_in_grid]])
{
    /* gid.x = (oh, ow) packed; gid.y = (kh, kw, ic) packed; gid.z = batch */
    const uint n  = gid.z;
    const uint k_idx = gid.y;       /* k = ic*kH*kW + kh*kW + kw */
    const uint hw = gid.x;
    if (n >= batch || hw >= out_H * out_W || k_idx >= in_channels * kH * kW) return;

    const uint kw_idx = k_idx % kW;
    const uint k_div  = k_idx / kW;
    const uint kh_idx = k_div % kH;
    const uint ic     = k_div / kH;

    const uint oh = hw / out_W;
    const uint ow = hw % out_W;

    const int h_in = (int)oh * (int)stride_h - pad_h + (int)kh_idx;
    const int w_in = (int)ow * (int)stride_w - pad_w + (int)kw_idx;

    half v = half(0);
    if (h_in >= 0 && h_in < (int)H && w_in >= 0 && w_in < (int)W) {
        const uint x_off = ((n * in_channels + ic) * H + (uint)h_in) * W + (uint)w_in;
        v = X[x_off];
    }
    /* col layout: [batch, k_total, out_hw]
     *   col[ ( n*K + k_idx ) * out_hw + hw ] = v
     */
    const uint K_total = in_channels * kH * kW;
    const uint out_hw = out_H * out_W;
    col[(n * K_total + k_idx) * out_hw + hw] = v;
}

/* Add bias + reshape: takes the GEMM result [batch, oc, out_hw] and adds a
 * per-channel bias. y[n, oc, hw] += bias[oc]. */
kernel void tc_conv2d_bias_add_f16(
    device       half* Y          [[buffer(0)]],
    device const half* bias       [[buffer(1)]],
    constant uint& batch          [[buffer(2)]],
    constant uint& out_channels   [[buffer(3)]],
    constant uint& out_hw         [[buffer(4)]],
    uint3 gid                     [[thread_position_in_grid]])
{
    const uint n  = gid.z;
    const uint oc = gid.y;
    const uint hw = gid.x;
    if (n >= batch || oc >= out_channels || hw >= out_hw) return;
    Y[(n * out_channels + oc) * out_hw + hw] += bias[oc];
}
