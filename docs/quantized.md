# Quantized inference

`quantized.h` is the LLM-inference path: block-quantized weights with
fp16 activations and fp16 output. The format mirrors `ggml`'s Q4_0 and
Q8_0 so we can load any GGUF model that uses those encodings without a
re-quantization step.

This is where the per-watt advantage of Apple Silicon really shows up:
weight-only quantization, unified memory, no PCIe to mask, and a tight
inner loop with cooperative simd-sum reduction.

## Formats

### Q4_0 — 4.5 bits / weight

```
block = struct {
    fp16  scale;            // 2 bytes
    uint8 packed[16];       // 32 4-bit weights, packed
}                           // total 18 bytes per 32 weights
```

The 16 nibble pairs follow the GGML / GGUF convention exactly:

```
packed[i] low nibble  = weight i        (signed 4-bit, range [-8, 7])
packed[i] high nibble = weight i + 16   (signed 4-bit)
```

This was previously the wrong way around in v0.1.2 — fixed in v0.1.4.

### Q8_0 — 8.5 bits / weight

```
block = struct {
    fp16  scale;            // 2 bytes
    int8  weights[32];      // 32 signed 8-bit
}                           // total 34 bytes per 32 weights
```

Higher fidelity than Q4_0; smaller memory savings.

## Storage cost

```
tc_quantized_size(TC_QUANT_Q4_0, N, K) = N * (K / 32) * 18  = N * K * 0.5625 bytes
tc_quantized_size(TC_QUANT_Q8_0, N, K) = N * (K / 32) * 34  = N * K * 1.0625 bytes
```

For 7B Llama (32 layers × ~200M weights touched per token), that's:

- Q4_0: 3.6 GB / token of weight traffic
- Q8_0: 6.8 GB / token

On M2 Ultra's ~800 GB/s LPDDR5 bandwidth, the theoretical Q4_0 ceiling
is ~220 tok/s. llama.cpp lands ~55-65 tok/s. The current
`bench_inference_7b` async-batched harness gets **186 tok/s @ 632 GB/s**
on the same shape — ~85% of theoretical, **3-3.5× ahead of llama.cpp's
GEMV core** on M2 Ultra. End-to-end inference (attention, softmax, RoPE,
RMSnorm on top) is a v0.2 integration target; the GEMV bottleneck is no
longer a blocker.

## API

### Quantize fp16 weights on the GPU

```c
tc_buffer* W_fp16 = ...;   /* [N, K] fp16 weights */
tc_buffer* W_q4   = NULL;
tc_buffer_alloc(ctx, tc_quantized_size(TC_QUANT_Q4_0, N, K), &W_q4);

tc_quantize_weights(ctx, W_fp16, W_q4, TC_QUANT_Q4_0, N, K);
```

The kernel reads 32-weight blocks, computes the per-block scale
(`scale = max_abs(block) / 7`), divides, rounds, and packs nibbles in
GGML order. Q8_0 quantization works the same way.

### GEMV

```c
tc_buffer* X = ...;        /* [M, K] fp16 activations          */
tc_buffer* Y = ...;        /* [M, N] fp16 output               */

tc_gemv_quantized(ctx, X, W_q4, Y, TC_QUANT_Q4_0, M, N, K);
```

The shape convention is `Y[M, N] = X[M, K] @ W^T` where `W` is `[N, K]`
quantized. The kernel is optimized for `M ≤ 4`; larger `M` routes through
dequant + `tc_gemm` in a future pass (today returns `TC_ERR_INVALID_SHAPE`
for `M > 4`).

### Fused RMSNorm + quantized GEMV

```c
tc_buffer* X = ...;        /* [M, K] fp16 hidden state         */
tc_buffer* gamma = ...;    /* [K] fp16 RMSNorm scale           */
tc_buffer* Y = ...;        /* [M, N] fp16 output               */

tc_fused_rmsnorm_gemv_quantized(ctx, X, gamma, W_q4, Y,
                                TC_QUANT_Q4_0, M, N, K, 1e-5f);
```

This is the reusable decode primitive for final token heads and any
projection that consumes a normalized hidden state with GGUF/qLLM-style
quantized weights. Runtimes should call this instead of hand-rolling
RMSNorm, temporary normalized buffers, and per-format dequant loops.

### Async

```c
tc_stream* s; tc_stream_create(ctx, &s);
tc_gemv_quantized_async(ctx, X, W_q,  Y,  TC_QUANT_Q4_0, M, N, K, s);
tc_gemv_quantized_async(ctx, X, W_q2, Y2, TC_QUANT_Q4_0, M, N, K, s);
/* ... batch many GEMV calls into one CB ... */
tc_stream_sync(s);
```

This is the bench-winning path. The stream keeps a single
`MTLCommandBuffer` open across calls so the per-GEMV command-buffer
round trip is amortized; on the 7B-decode harness, that's worth the
difference between memory-bound (632 GB/s ≈ 79% of LPDDR5 peak) and
dispatch-bound (small fraction of bandwidth, dozens of CB round trips
per token).

## Kernel design

Two `.metal` files:

- `gemm_quantized.metal` — original Q4_0 / Q8_0 GEMV path.
- `gemm_quantized_v2.metal` — faster Q4_0 path, default since v0.1.6.
  Reachable by env: `TC_Q4_USE_V1=1` reverts to the original.

Per-cell pattern (v1):
- One simdgroup per output cell (`Y[m, n]`).
- The simdgroup walks `k` in blocks of 32.
- Each thread reads 1-2 nibbles from `W_q[n, k_block]`, unpacks, multiplies
  by `X[m, k]`, accumulates.
- Final simd_sum gives the dot product; write `Y[m, n]`.

v2 changes:
- Multiple output cells per simdgroup (better register reuse).
- Cooperative load of the activation row into TG memory.

The "1 simdgroup per output cell" pattern is what's eating the
bandwidth gap with llama.cpp. llama.cpp uses 4 output cells per simdgroup
with inter-block pipelining; that's the v0.2 retune.

## Numerical accuracy

| Test | What it validates | Tolerance |
|---|---|---|
| `test_quantized` Q4_0 sync | dequant-reference vs GEMV | rms_scaled ≤ 2e-4 |
| `test_quantized` Q4_0 async | same, via `_async` path | rms_scaled ≤ 2e-4 |
| `test_quantized` Q4_0 tail N | shapes where N isn't a multiple of the simdgroup count | rms_scaled ≤ 2e-4 |
| `test_quantized` Q8_0 GPU quant + GEMV | round-trip through `tc_quantize_weights` | rms_scaled ≤ 1e-4 |
| `test_quantized` invalid quant enum | `tc_quantized_size` returns 0 | — |

The "dequant-reference" is a CPU implementation that dequantizes Q4 blocks
back to fp16, then does an fp32 GEMV. The kernel result must agree to ≤
2e-4 RMS-scaled — quantization error is the dominant term, and we get it
right to that precision.

For ICC/runtime readiness, use:

```sh
python3 scripts/run_quantized_gguf_runtime_evidence.py --require-pass
python3 scripts/check_quantized_gguf_runtime_evidence.py \
  build/quantized_gguf_runtime_evidence.json --require-pass
```

That evidence path wraps `test_quantized` and `test_gguf`, so it proves the
Metal `gemv_quant_encode` helper and the GGUF-loaded quantized GEMV path from
one deterministic artifact. It reports `metal_device_unavailable` as blocked
when the host sandbox hides the Metal device.

## Patterns

### Per-layer decode

```c
/* Q, K, V projections via fused RMSnorm+GEMV */
tc_fused_rmsnorm_gemv(ctx, x, gamma_attn, W_qkv, qkv, 1, qkv_dim, hidden, eps);

/* Or, if Wq Wk Wv are separate (and Q4_0 quantized): */
tc_gemv_quantized(ctx, x_norm, W_q4_q, q, TC_QUANT_Q4_0, 1, hidden, hidden);
tc_gemv_quantized(ctx, x_norm, W_q4_k, k, TC_QUANT_Q4_0, 1, kv_dim, hidden);
tc_gemv_quantized(ctx, x_norm, W_q4_v, v, TC_QUANT_Q4_0, 1, kv_dim, hidden);

/* Attention (fp16 activations even when weights are quantized) */
tc_attention_forward(ctx, &adesc, q, k_cache, v_cache, o, NULL);

/* Output projection */
tc_gemv_quantized(ctx, o, W_q4_o, out, TC_QUANT_Q4_0, 1, hidden, hidden);

/* MLP gate + up + down */
tc_gemv_quantized(ctx, out_norm, W_q4_gate, gate, TC_QUANT_Q4_0, 1, mlp_dim, hidden);
tc_gemv_quantized(ctx, out_norm, W_q4_up,   up,   TC_QUANT_Q4_0, 1, mlp_dim, hidden);
tc_swiglu_forward(ctx, gate, up, gu, mlp_dim);
tc_gemv_quantized(ctx, gu, W_q4_down, mlp_out, TC_QUANT_Q4_0, 1, hidden, mlp_dim);
```

All GEMV calls can be batched into a single stream — that's where the
async speedup lives.

### Loading from GGUF

The natural input source is a GGUF model. Use the
`tc_gguf_quantized_matrix_info` helper to translate GGUF's `[K, N]`
matrix layout to the `[N, K]` that `tc_gemv_quantized` expects:

```c
tc_gguf_loaded_tensor_info proj;
tc_gguf_loaded_get_tensor(model, "blk.0.attn_q.weight", &proj);

tc_gguf_quantized_matrix_info q;
tc_gguf_loaded_tensor_quantized_matrix_info(&proj, &q);

tc_gemv_quantized(ctx, x, q.buffer, y, q.quant_type, 1, q.N, q.K);
```

See [gguf.md](gguf.md) for the full loading pattern.

## What v0.2 closes

- Match llama.cpp's 55-65 tok/s on M2 Ultra at 7B Q4_0:
  - Multiple output cells per simdgroup (4 is the magic number on M-series).
  - Inter-block pipelining (prefetch next K-block while computing prev).
  - vec4-aligned activation loads.
- Add `M ≥ 4` GEMV variant (dequant + tc_gemm) so prefill works at scale.
- Add Q4_1, Q5_0, Q5_1 (the ggml lineup) — Q4_K_M and Q5_K_M are the
  community's preferred formats and rebuild on top of the Q-block design.
- Add `tc_quantize_q4_0_async` so prepare-then-GEMV can run in one stream.
