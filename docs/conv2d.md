# Conv2D

`tc_conv2d_*` implements the standard 2D convolution: forward, dInput
backward, and dWeight backward. The strategy is **im2col + GEMM** — same
as cuDNN's `IMPLICIT_GEMM` strategy. The im2col-and-gemm path is straight,
predictable, and reuses the well-tuned `tc_gemm` kernels.

This is v0.1 territory: not the fastest possible conv (no Winograd, no
specialized 1×1 path, no NHWC), but correct, fp16, with backward.

## Surface

```c
tc_status_t tc_conv2d_forward(ctx,
                              X,        /* [N, IC, H, W]    fp16    */
                              weight,   /* [OC, IC, kH, kW] fp16    */
                              bias,     /* [OC]    fp16  or NULL    */
                              Y,        /* [N, OC, oH, oW]  fp16    */
                              scratch_col,
                              batch, in_channels, out_channels,
                              H, W, kH, kW,
                              pad_h, pad_w, stride_h, stride_w,
                              out_H, out_W);

tc_status_t tc_conv2d_backward_input (ctx,
                                      dY, weight,
                                      dX,
                                      scratch_col, scratch_dX_f32,
                                      ...);

tc_status_t tc_conv2d_backward_weight(ctx,
                                      X, dY,
                                      dW,
                                      scratch_col,
                                      ...);
```

Output spatial dims (caller computes and passes):

```
out_H = floor((H + 2 * pad_h - kH) / stride_h) + 1
out_W = floor((W + 2 * pad_w - kW) / stride_w) + 1
```

## Strategy

Forward:

```
1.  col   = im2col(X)        [N, IC*kH*kW, oH*oW]    via conv2d.metal
2.  Y     = weight @ col      via tc_gemm
            weight is reshaped [OC, IC*kH*kW]
            Y is reshaped     [N, OC, oH*oW] then [N, OC, oH, oW]
3.  Y    += bias              (optional)
```

dInput:

```
1.  dCol  = weight^T @ dY     via tc_gemm with transpose_a=true
2.  dX    = col2im_atomic(dCol) into scratch_dX_f32  (atomic, fp32)
3.  dX_fp16 = cast(scratch_dX_f32)
```

dWeight:

```
1.  col   = im2col(X)         (reuse forward scratch)
2.  dW    = sum_n (dY[n] @ col[n]^T)  per-batch tc_gemm with offset
```

`scratch_col` is allocated by the caller and reused across forward and
backward; `scratch_dX_f32` is needed only by dInput because col2im writes
overlap and we need atomic accumulation in fp32.

## Scratch sizing

The Python binding has helpers that match these formulas; the C API trusts
you to compute them.

```
scratch_col bytes      = batch * (in_channels * kH * kW) * out_H * out_W * 2 (fp16)
scratch_dX_f32 bytes   = batch * in_channels * H * W * 4               (fp32 atomic accumulator)
```

`scratch_dX_f32` must be zero-initialized before each dInput call (it's
the accumulator).

## Kernels

| Kernel source | Purpose |
|---|---|
| `conv2d.metal` | im2col gather |
| `conv2d_backward.metal` | col2im with fp32 atomic-add scatter, plus cast-back-to-fp16 finalize |

The GEMM step is just `tc_gemm` (or `tc_gemm_batched` for multi-batch
dWeight in v0.1.3+).

## Validation

`tests/test_conv2d.c` validates forward + dInput + dWeight against a
fp64 CPU reference. The dInput uses fp32 atomic accumulation, so the
result is bit-exact for small shapes and rms_scaled ≤ 1e-3 for larger ones.

Multi-batch dInput was added in v0.1.3 (per-batch GEMM with MTLBuffer
offset binding); the kernel itself is shape-agnostic, but multi-batch was
previously broken by a bug in the offset binding.

## What's not done in v0.1

- **No Winograd / FFT.** The 3×3 stride-1 case would benefit from Winograd
  (~2-2.25× speedup on most chips); not worth the complexity for v0.1.
- **No NHWC layout.** PyTorch prefers NCHW; we follow.
- **No depthwise / separable conv specialization.** Depthwise works (set
  `groups = in_channels`) via the general path, but the optimal kernel is
  different.
- **No mixed-stride / mixed-dilation tuning.** Dilation = 1 only in v0.1.

Realistic v0.2/v0.3 priorities are the 3×3 stride-1 Winograd path (covers
ResNet, ConvNeXt) and the depthwise specialization (covers MobileNet,
EfficientNet).

## Conv2D vs GEMM tradeoffs

- The im2col scratch is large: `batch * IC * kH * kW * oH * oW * 2` bytes.
  For a typical ResNet-50 first layer (224×224 input, 7×7 kernel, 64
  output channels), the col buffer is ~7 MB per batch element. Allocate
  once outside the training loop and reuse.
- For 1×1 convolutions, the im2col is a no-op (`oH × oW = H × W`,
  `IC * 1 * 1 = IC`), and the whole op degenerates to a single `tc_gemm`.
- For very large kernels, the col buffer dominates memory; consider
  Winograd if v0.2 ships it before you need it.

## Example

Pure C, one fp16 conv2d:

```c
const int N = 1, IC = 3, OC = 64, H = 224, W = 224;
const int kH = 7, kW = 7, pad = 3, stride = 2;
const int oH = (H + 2*pad - kH) / stride + 1;
const int oW = (W + 2*pad - kW) / stride + 1;

tc_buffer *X, *Wt, *Y, *col;
tc_buffer_alloc(ctx, N*IC*H*W*2, &X);
tc_buffer_alloc(ctx, OC*IC*kH*kW*2, &Wt);
tc_buffer_alloc(ctx, N*OC*oH*oW*2, &Y);
tc_buffer_alloc(ctx, N*IC*kH*kW*oH*oW*2, &col);

/* fill X, Wt ... */

tc_conv2d_forward(ctx, X, Wt, /*bias=*/NULL, Y, col,
                  N, IC, OC, H, W, kH, kW, pad, pad, stride, stride, oH, oW);
```

Backward dInput needs the additional fp32 scratch:

```c
tc_buffer *dY, *dX, *dX_scratch;
tc_buffer_alloc(ctx, N*OC*oH*oW*2, &dY);
tc_buffer_alloc(ctx, N*IC*H*W*2, &dX);
tc_buffer_alloc(ctx, N*IC*H*W*4, &dX_scratch);

/* zero dX_scratch before each call */
void* p; tc_buffer_map(dX_scratch, &p);
memset(p, 0, N*IC*H*W*4);

tc_conv2d_backward_input(ctx, dY, Wt, dX, col, dX_scratch,
                          N, IC, OC, H, W, kH, kW, pad, pad, stride, stride, oH, oW);
```
