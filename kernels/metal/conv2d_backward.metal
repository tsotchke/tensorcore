/*
 * tensorcore — Conv2D backward.
 *
 *   Forward:
 *     col[n, K=ic*kH*kW, out_hw] = im2col(X)
 *     Y[n, oc, out_hw]           = W_flat[oc, K] @ col[n, K, out_hw]
 *
 *   Backward:
 *     dW_flat[oc, K]  = sum_n dY[n, oc, out_hw] @ col[n, K, out_hw]^T
 *     dCol[n, K, out_hw] = W_flat[oc, K]^T @ dY[n, oc, out_hw]
 *     dX[n, ic, H, W]    = col2im(dCol)    (scatter-add)
 *
 * This file ships two kernels:
 *   tc_col2im_f16   — scatter dCol back into dX with overlap accumulation.
 *   tc_conv2d_dY_to_dCol_setup_f16 — no-op (we feed dY directly into GEMM).
 *
 * The actual GEMM calls for dW and dCol are done by the host via tc_gemm,
 * mirroring how the forward composes im2col + tc_gemm.
 */

#include <metal_stdlib>
using namespace metal;

/* col2im — inverse of im2col. Scatter dcol back into dX, accumulating where
 * multiple (kh, kw) positions overlap the same input pixel. Uses atomic
 * add (fp32 accum buffer separately from fp16 final result) to handle the
 * overlap. */
kernel void tc_col2im_atomic_f32(
    device const half*  dCol             [[buffer(0)]],
    device atomic_uint* dX_atomic_u32    [[buffer(1)]],  /* fp32 reinterpret */
    constant uint& batch                 [[buffer(2)]],
    constant uint& in_channels           [[buffer(3)]],
    constant uint& H                     [[buffer(4)]],
    constant uint& W                     [[buffer(5)]],
    constant uint& kH                    [[buffer(6)]],
    constant uint& kW                    [[buffer(7)]],
    constant int&  pad_h                 [[buffer(8)]],
    constant int&  pad_w                 [[buffer(9)]],
    constant uint& stride_h              [[buffer(10)]],
    constant uint& stride_w              [[buffer(11)]],
    constant uint& out_H                 [[buffer(12)]],
    constant uint& out_W                 [[buffer(13)]],
    uint3 gid                            [[thread_position_in_grid]])
{
    /* gid mirrors im2col's grid: (oh*ow, k=ic*kH*kW, batch). */
    const uint n  = gid.z;
    const uint k_idx = gid.y;
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
    if (h_in < 0 || h_in >= (int)H || w_in < 0 || w_in >= (int)W) return;

    const uint K_total = in_channels * kH * kW;
    const uint out_hw = out_H * out_W;
    const half v = dCol[(n * K_total + k_idx) * out_hw + hw];
    const uint x_off = ((n * in_channels + ic) * H + (uint)h_in) * W + (uint)w_in;

    /* Atomic fp32 add via uint reinterpret (compare-exchange loop). */
    uint old_u = atomic_load_explicit(&dX_atomic_u32[x_off], memory_order_relaxed);
    while (true) {
        const float old_f = as_type<float>(old_u);
        const float new_f = old_f + (float)v;
        const uint new_u = as_type<uint>(new_f);
        if (atomic_compare_exchange_weak_explicit(
                &dX_atomic_u32[x_off], &old_u, new_u,
                memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

/* Convert the fp32 accumulation buffer into a fp16 dX result. */
kernel void tc_col2im_finalize_f16(
    device const float* dX_fp32          [[buffer(0)]],
    device       half*  dX_fp16          [[buffer(1)]],
    constant uint& n_elements            [[buffer(2)]],
    uint i                               [[thread_position_in_grid]])
{
    if (i >= n_elements) return;
    dX_fp16[i] = (half)dX_fp32[i];
}
