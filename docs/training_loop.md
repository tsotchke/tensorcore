# Training — assembling a transformer step

The mirror image of [inference.md](inference.md): one full forward +
backward + optimizer step of a Llama-style transformer block, with every
tensorcore call called out.

This is what `tests/test_transformer_block.c` and
`tests/test_e2e_training.c` actually exercise — they're the executable
form of this doc.

## The shape

A single training step on one batch:

```
Forward:
    activations = []
    for each layer:
        x_norm, rstd_attn = RMSnorm(x, γ_attn)               # save rstd
        Q, K, V          = (W_q, W_k, W_v) @ x_norm           # GEMM × 3
        Q, K             = RoPE(Q, K)
        O, LSE           = FlashAttention(Q, K, V, causal)    # save LSE
        x_attn           = W_o @ O
        x                = x + x_attn
        x_norm2, rstd_mlp = RMSnorm(x, γ_mlp)
        gate, up         = (W_gate, W_up) @ x_norm2
        gu               = SwiGLU(gate, up)
        x_mlp            = W_down @ gu
        x                = x + x_mlp
        activations.append(...all of the above for backward...)

    loss             = loss_fn(lm_head(RMSnorm(x)), targets)
    dLoss/dlogits    = grad_loss
    dLoss/dx_final   = lm_head_grad(dLoss/dlogits)

Backward (reverse order):
    for each layer in reverse:
        dx_norm2 = ...                          # from MLP residual
        dgu      = dx_mlp through W_down^T
        dgate, dup = SwiGLU_backward(dgu, gate, up)
        dx_norm2 += (W_gate^T @ dgate) + (W_up^T @ dup)
        dx       += RMSnorm_backward(dx_norm2, rstd_mlp, γ_mlp)
        ... attention block backward symmetrically ...

Optimizer:
    for each parameter:
        AdamW(param_fp32, m, v, grad, lr, β1, β2, ε, wd, bc1, bc2)
```

Every primitive in that block has a tensorcore call.

## Per-call mapping

### Forward

| Step | tensorcore call | Notes |
|---|---|---|
| `x_norm = RMSnorm(x, γ_attn)` | `tc_rmsnorm_forward(ctx, X, gamma, Y, rstd_out, N, D, eps)` | `rstd_out` saved for backward |
| `Q = W_q @ x_norm`  (and K, V) | `tc_gemm(ctx, &qkv_desc, x_norm, W_qkv, QKV)` | Fuse Wq/Wk/Wv into one Wqkv if you can |
| `Q, K = RoPE(Q, K)` | `tc_rope_forward(ctx, X, cos_t, sin_t, B, H, S, head_dim)` | in-place; one call for Q, one for K |
| `O, LSE = FlashAttention(...)` | `tc_attention_forward(ctx, &adesc, Q, K, V, O, LSE)` | **set `return_lse=true`**; LSE saved for backward |
| `x_attn = W_o @ O` | `tc_gemm` | regular fp16 GEMM |
| `x += x_attn` | small elementwise; not in tensorcore today | one custom kernel or fold into next RMSnorm |
| `x_norm2 = RMSnorm(x, γ_mlp)` | `tc_rmsnorm_forward` | another rstd saved |
| `gate, up = (W_gate, W_up) @ x_norm2` | `tc_gemm × 2` | or one fused `Wgate_up` |
| `gu = SwiGLU(gate, up)` | `tc_swiglu_forward(ctx, gate, up, out, n)` | n = B × S × mlp_dim |
| `x_mlp = W_down @ gu` | `tc_gemm` | |
| `x += x_mlp` | elementwise | |

### Backward (one layer)

| Step | tensorcore call |
|---|---|
| `dx_mlp = dx` (from residual; no kernel) | — |
| `dgu = W_down^T @ dx_mlp` | `tc_gemm` with `transpose_a=true` |
| `dgate, dup = SwiGLU_backward(dgu, gate, up)` | `tc_swiglu_backward(ctx, gate, up, dout, dgate, dup, n)` |
| `dx_norm2 = (W_gate^T @ dgate) + (W_up^T @ dup)` | `tc_gemm × 2` with `transpose_a=true`, beta=1 on the second |
| `dx, dγ_mlp = RMSnorm_backward(dx_norm2, X, γ, rstd_mlp)` | `tc_rmsnorm_backward(ctx, X, gamma, dY, rstd, dX, dgamma, N, D)` |
| `dx_attn = dx` (from residual) | — |
| `dO = W_o^T @ dx_attn` | `tc_gemm` with `transpose_a=true` |
| `dQ, dK, dV = FlashAttention_backward(Q, K, V, O, LSE, dO)` | `tc_attention_backward(ctx, &adesc, Q, K, V, O, dO, LSE, dQ, dK, dV)` |
| `dQ, dK = RoPE_backward(dQ, dK)` | `tc_rope_backward(ctx, dX, cos_t, sin_t, B, H, S, head_dim)` |
| `dx_norm = (W_q^T @ dQ) + (W_k^T @ dK) + (W_v^T @ dV)` | `tc_gemm × 3` with `transpose_a=true`, accumulating |
| `dx, dγ_attn = RMSnorm_backward(...)` | `tc_rmsnorm_backward` |
| Weight gradients: `dW_q = dQ @ x_norm^T`, etc. | `tc_gemm` with `transpose_b=true` per param |

### Optimizer

One call per parameter group:

```c
tc_adamw_step(ctx,
              params_fp32, m_fp32, v_fp32,    /* in/out: fp32 master */
              grads, TC_DTYPE_F16,             /* fp16 grads (mixed precision) */
              n,
              lr, beta1, beta2, eps, wd,
              bc1, bc2);                       /* bias corrections precomputed */
```

`bc1 = 1 - powf(beta1, t)`, `bc2 = 1 - powf(beta2, t)` — host computes per step.

## What gets saved between forward and backward

| Saved | Shape | Reused in backward |
|---|---|---|
| `rstd_attn` per token | `[B*S]` fp32 | `tc_rmsnorm_backward` |
| `rstd_mlp` per token | `[B*S]` fp32 | `tc_rmsnorm_backward` |
| `LSE` from FlashAttention | `[B, H, S]` fp32 | `tc_attention_backward` |
| `Q, K, V, O` from attention | `[B, H, S, D]` fp16 each | `tc_attention_backward` |
| `gate, up` from MLP | `[B, S, mlp_dim]` fp16 each | `tc_swiglu_backward` |
| `x_norm` after each RMSnorm | `[B, S, D]` fp16 | `tc_rmsnorm_backward` |
| layer inputs `x` | `[B, S, D]` fp16 | residual through layers |

On a 7B-class model at batch 4, sequence 2048, this is ~6 GB of
activation memory before checkpointing. tensorcore now exposes the
buffer-level discard/realize primitive on CPU and Metal; the remaining
higher-level training work is deciding which layer inputs to save and
which intermediates to recompute for a given mesh memory budget.

## Mixed-precision recipe

The default training pattern matches NVIDIA's Apex AMP:

1. **Parameters:** fp32 master weights (1 copy per parameter), held by
   the optimizer.
2. **Forward / backward compute:** fp16 (or bf16 on Apple9+).
3. **Accumulators inside kernels:** fp32. Non-negotiable.
4. **Gradients:** fp16, with optional dynamic loss scaling. tensorcore
   doesn't manage the loss scaler — your code does (it's just
   `loss *= scale; grads /= scale`).
5. **Optimizer update:** runs on the fp32 masters, reads fp16 grads via
   `grad_dtype=TC_DTYPE_F16`.

The fp32 master copy is what makes the recipe stable across long training
runs; the fp16 forward/backward is what makes it fast.

## Tests that exercise this

| Test | What it covers |
|---|---|
| `tests/test_transformer_block.c` | One full forward + backward of a single Llama-style block at small shapes |
| `tests/test_e2e_training.c` | A few iterations of the full forward + backward + AdamW; checks parameter convergence |
| `tests/test_attention_backward.c` | `tc_attention_backward` at D=64 and D=128 against a numerical-differences reference |
| `tests/test_training_kernels.c` | RMSnorm / LayerNorm / RoPE / SwiGLU / softmax / AdamW kernels |
| `tests/test_fused_norm_gemv.c` | fused RMSNorm/LayerNorm GEMV vs the separate norm+gemm paths |

Read those if you want the smallest-shape concrete example of any
specific piece. They're ~150-300 lines each, pure C, link only against
`libtensorcore.dylib`.

## Distributed training

The training loop should treat distribution as a backend choice:

```c
tc_dist_ctx* d = NULL;
tc_dist_init(ctx, TC_DIST_RING, /*world_size=*/4, /*rank=*/r,
             "tb5://192.168.42.0/cluster", &d);

/* ... per-step training body unchanged ... */

/* After backward, before optimizer: all-reduce gradients */
tc_allreduce(d, grad_buffer, n_elements, TC_DTYPE_F16, TC_REDUCE_AVG);

/* Optimizer step runs on every rank with the averaged grads */
tc_adamw_step(...);
```

For one process, `TC_DIST_SINGLE` with `world_size=1` makes the
`tc_allreduce` a no-op. Default Apple and portable CPU builds also support
`TC_DIST_GLOO` over `gloo+tcp://host:port` for multi-rank TCP collectives
today. The multi-Mac TB5/JACCL ring remains the v0.5 backend swap.

See [distributed.md](distributed.md) for the ZeRO-1/2/3 plan.

## What's missing in v0.1 / v0.2

Honest list:

- **Activation checkpointing.** Save inputs only, recompute everything
  else. v0.6 ships the integrated kernel-level support.
- **Sequence parallelism / tensor parallelism.** v0.6 — for now, data
  parallelism + ZeRO-2 is the path.
- **bf16 native training kernels.** Today, training kernels are fp16 IO;
  bf16 lands in v0.2.

## See also

- [training_kernels.md](training_kernels.md) — per-kernel reference.
- [attention.md](attention.md) — FlashAttention forward + backward
  semantics.
- [gemm.md](gemm.md) — GEMM with `transpose_a` / `transpose_b` and
  beta-accumulation patterns.
- [numerics.md](numerics.md) — what to compare against; how to detect
  precision regressions.
