# Training kernels

`training.h` is the transformer training kit: RMSnorm, LayerNorm, RoPE,
SwiGLU, softmax, AdamW, and the fused-RMSnorm-GEMV inference primitive.
Apple builds use `kernels/metal/training_kernels.metal` (and
`fused_norm_gemv.metal` for the fusion). Portable CPU builds use
`lib/ops/training_cpu.cpp`. CUDA builds with `TC_ENABLE_CUDA=ON` route
managed-memory RMSNorm forward/backward, LayerNorm forward/backward, RoPE
forward/backward, SwiGLU forward/backward, softmax forward/backward, and
fp32/fp16-gradient AdamW through `lib/cuda/training.cu` after CUDA
initialization; host-only buffers fall back to the portable CPU path. Set
`TC_DISABLE_CUDA_GEMM=1`,
`TC_CUDA_GEMM=0`, or `TC_USE_CUDA_GEMM=0` to force CPU fallback.

This page describes shapes, the kernel design, and the conventions you
need to plug them into a training step.

## RMSnorm — Llama-style

```c
tc_status_t tc_rmsnorm_forward (ctx, X, gamma, Y, rstd_out, N, D, eps);
tc_status_t tc_rmsnorm_backward(ctx, X, gamma, dY, rstd, dX, dgamma, N, D);
```

| Buffer | Shape | dtype |
|---|---|---|
| `X` | `[N, D]` | fp16 |
| `gamma` | `[D]` | fp16 |
| `Y` | `[N, D]` | fp16 |
| `rstd_out` | `[N]` | fp32 (saved for backward) |
| `dY` | `[N, D]` | fp16 |
| `dX` | `[N, D]` | fp16 |
| `dgamma` | `[D]` | **fp32** (accumulator dtype — feeds directly into `tc_adamw_step` with `grad_dtype=TC_DTYPE_F32`) |

Math:

```
rms(x) = sqrt(mean(x^2) + eps)
y      = (x / rms) * gamma
```

Compared with LayerNorm, RMSnorm drops the mean-subtraction and the bias.
This is the normalization used by Llama, Mistral, Qwen, and basically
every modern open LLM since 2023.

Kernel design: one threadgroup per row. Per row we compute `mean(x^2)`
via a simdgroup-level sum reduction, then broadcast `rstd` back, then
apply `y[i] = x[i] * rstd * gamma[i]` in fp32 internally, cast to fp16 on
write.

## LayerNorm — standard

```c
tc_status_t tc_layernorm_forward (ctx, X, gamma, beta, Y, mean_out, rstd_out,
                                  N, D, eps);
tc_status_t tc_layernorm_backward(ctx, X, gamma, dY, mean, rstd, dX, N, D);
```

| Buffer | Shape | dtype |
|---|---|---|
| `X` | `[N, D]` | fp16 |
| `gamma`, `beta` | `[D]` | fp16 |
| `Y` | `[N, D]` | fp16 |
| `mean_out`, `rstd_out` | `[N]` | fp32 |

Same per-row threadgroup layout as RMSnorm, with both the mean and the
inverse standard deviation saved for backward.

## RoPE — Rotary Position Embedding

```c
tc_status_t tc_rope_forward(ctx, X, cos_t, sin_t, batch, heads, seq, head_dim);
tc_status_t tc_rope_backward(ctx, dX, cos_t, sin_t, batch, heads, seq, head_dim);
```

| Buffer | Shape | dtype | Notes |
|---|---|---|---|
| `X` / `dX` | `[B, H, S, D]` | fp16 | **in-place** |
| `cos_t` | `[S, D/2]` | fp32 | precomputed by host |
| `sin_t` | `[S, D/2]` | fp32 | precomputed by host |

For each `(b, h, s, k)` where `k < D/2`, the kernel rotates the pair
`(X[b, h, s, k], X[b, h, s, k + D/2])` by `(cos_t[s, k], sin_t[s, k])`.
The backward pass applies the inverse rotation to `dX`:
`dx0 = dy0 * cos + dy1 * sin`, `dx1 = -dy0 * sin + dy1 * cos`.
This matches Llama / Mistral's RoPE convention (half-rotation grouping,
not the interleaved variant some PyTorch implementations use).

Compute `cos_t` and `sin_t` on the host once per sequence length:

```c
for (int s = 0; s < S; ++s) {
    for (int k = 0; k < D/2; ++k) {
        float freq  = powf(rope_base, -2.f * k / D);  /* rope_base = 10000 */
        float angle = (float)s * freq;
        cos_t[s][k] = cosf(angle);
        sin_t[s][k] = sinf(angle);
    }
}
```

`tc_gguf_get_llama_config` returns `rope_freq_base` and `rope_freq_scale`
in case you're loading a model whose RoPE was scaled.

## SwiGLU

```c
tc_status_t tc_swiglu_forward (ctx, gate, up, out, n);
tc_status_t tc_swiglu_backward(ctx, gate, up, dout, dgate, dup, n);
```

| Buffer | Shape | dtype |
|---|---|---|
| `gate`, `up`, `out`, `dout`, `dgate`, `dup` | `[n]` | fp16 |

Math:

```
silu(x) = x / (1 + exp(-x))
out     = silu(gate) * up
```

Pointwise; one thread per element. Used in the MLP of every modern LLM
(`o = down(silu(gate(x)) * up(x))`). The corresponding GEMMs are
`tc_gemm` calls; SwiGLU is the elementwise glue.

## Softmax — standalone, row-wise

```c
tc_status_t tc_softmax_forward (ctx, X, Y, N, D);
tc_status_t tc_softmax_backward(ctx, Y, dY, dX, N, D);
```

| Buffer | Shape | dtype |
|---|---|---|
| `X`, `Y`, `dY`, `dX` | `[N, D]` | fp16 |

Numerically stable row-wise softmax: per-row max subtraction, exp, sum,
divide. Used outside attention (e.g. final logits, mixture-of-experts
routing).

The attention kernel has its own softmax inlined (the FlashAttention
online-softmax scheme); this standalone variant is for everything else.

## AdamW — fused step

```c
tc_status_t tc_adamw_step(ctx,
                          params_fp32, m_fp32, v_fp32,
                          grads, grad_dtype,
                          n,
                          lr, beta1, beta2, eps, wd, bc1, bc2);
```

| Buffer | Shape | dtype | Read/Write |
|---|---|---|---|
| `params_fp32` | `[n]` | fp32 | RW (master weights) |
| `m_fp32` | `[n]` | fp32 | RW (1st moment) |
| `v_fp32` | `[n]` | fp32 | RW (2nd moment) |
| `grads` | `[n]` | fp16 or fp32 | R |

Math:

```
m  = β1 * m  + (1 - β1) * g
v  = β2 * v  + (1 - β2) * g²
m̂  = m / bc1     (bc1 = 1 - β1^t passed by host)
v̂  = v / bc2     (bc2 = 1 - β2^t passed by host)
θ  = θ - lr * (m̂ / (√v̂ + eps) + wd * θ)
```

Two notes:

- **fp32 master weights, fp16 grads.** This is the standard mixed-precision
  training recipe; you keep the fp32 copy of the parameters, and the
  forward/backward produce fp16 grads which the optimizer reads.
- **Bias corrections passed by host.** The kernel doesn't know what step
  it is. Compute `bc1 = 1 - powf(beta1, t)`, `bc2 = 1 - powf(beta2, t)`
  on the host and pass them in. This is one fp32 division saved per
  element per step.

One thread per element; pointwise.

## Fused Norm + GEMV

```c
tc_status_t tc_fused_rmsnorm_gemv(ctx, X, gamma, W, Y, M, N, K, eps);
tc_status_t tc_fused_layernorm_gemv(ctx, X, gamma, beta, W, Y, M, N, K, eps);
```

| Buffer | Shape | dtype | Notes |
|---|---|---|---|
| `X` | `[M, K]` | fp16 | typically `M ≤ 4` (inference batch) |
| `gamma` | `[K]` | fp16 | norm scale |
| `beta` | `[K]` | fp16 | LayerNorm shift; only used by `tc_fused_layernorm_gemv` |
| `W` | `[K, N]` | fp16 | projection weight |
| `Y` | `[M, N]` | fp16 | `Y = Norm(X, γ[, β]) @ W` |

The hot path inside an LLM decode step is:

```
x_norm = RMSnorm(x)          # write [hidden]
q      = x_norm @ Wq         # read  [hidden], write [hidden]
k      = x_norm @ Wk
v      = x_norm @ Wv
```

That `x_norm` write/read round trip is pure memory traffic -- the normalized
vector is consumed three times immediately after being produced. The
fused kernels compute `rstd` inline, apply the normalization in-register
during the matmul accumulation, and skip the write-back entirely. The
LayerNorm variant also computes row mean and applies `beta[k]`.

Two-pass intra-threadgroup design:
- Pass 1: each simdgroup computes `sum(x_i²)` for a slice of `K`; warp-reduce
  to `mean(x²)`, then `rstd = rsqrt(mean + eps)`.
- Pass 2: same threadgroup iterates over `K`, multiplying `x[k] * rstd *
  gamma[k]` *while* accumulating against `W[k, :]` for the row of `Y` it
  owns.

For `tc_fused_layernorm_gemv`, pass 1 computes `sum(x_i)` and
`sum(x_i²)` so the kernel can derive `mean` and `variance`; pass 2 uses
`((x[k] - mean) * rstd * gamma[k] + beta[k])` before multiplying by
`W[k, :]`.

Caveats:
- Tuned for M ≤ 4. For training (M ≥ 32), use the separate norm-forward +
  `tc_gemm` path -- the per-row statistic recompute would dominate at
  larger M.
- The fused kernels don't expose `rstd_out` or `mean_out`. If you need them
  for backward, use the separate path. The fused path is inference-only.

Validated by `tests/test_fused_norm_gemv.c` against the separate
`tc_rmsnorm_forward + tc_gemm` and `tc_layernorm_forward + tc_gemm`
paths at rms_scaled ≤ 5e-3.

## A typical training step

Putting it together, one forward step of a Llama-style transformer block:

```c
/* RMSnorm */
tc_rmsnorm_forward(ctx, X, gamma_attn, X_norm, rstd_attn, N, D, eps);

/* QKV projection */
tc_gemm(ctx, &qkv_desc, X_norm, W_qkv, QKV);

/* RoPE on Q and K, in-place */
tc_rope_forward(ctx, Q, cos_t, sin_t, B, H, S, head_dim);
tc_rope_forward(ctx, K, cos_t, sin_t, B, H_kv, S, head_dim);

/* Attention */
tc_attention_desc adesc = { ... return_lse=true ... };
tc_attention_forward(ctx, &adesc, Q, K, V, O, LSE);

/* O projection */
tc_gemm(ctx, &o_desc, O, W_o, X_after_attn);

/* Residual */
/* y = x + x_after_attn done with a small elementwise kernel or
   tc_gemm with alpha=1, beta=1 trick */

/* Second RMSnorm */
tc_rmsnorm_forward(ctx, Y, gamma_mlp, Y_norm, rstd_mlp, N, D, eps);

/* MLP gate, up */
tc_gemm(ctx, &gate_desc, Y_norm, W_gate, GATE);
tc_gemm(ctx, &up_desc,   Y_norm, W_up,   UP);

/* SwiGLU */
tc_swiglu_forward(ctx, GATE, UP, GU, B * S * mlp_dim);

/* MLP down */
tc_gemm(ctx, &down_desc, GU, W_down, MLP_OUT);

/* Final residual */
```

The backward mirrors with `_backward` variants; the optimizer step is one
`tc_adamw_step` per parameter group.

`tests/test_transformer_block.c` and `tests/test_e2e_training.c` build
exactly this pattern at small sizes; they're worth reading as a reference
for shape and buffer lifecycle.

## v0.2 closes

- Fused-adamw for fp16 grads + bf16 master weight option
- Bias-add fused into `tc_gemm` for FFN
