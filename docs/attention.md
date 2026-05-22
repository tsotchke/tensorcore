# Attention

`tc_attention_forward` is FlashAttention-2 forward, fused into a single
Metal kernel: Q @ K^T, scaled softmax with optional bias, P @ V, all
accumulated in fp32 on-chip, output written once. `tc_attention_backward`
is the LSE-saved backward pass.

The descriptor exposes the modern transformer feature set: causal mask,
GQA / MQA, sliding window, ALiBi positional bias, LSE save for training.

## Surface

```c
typedef struct {
    int32_t batch, heads, seq_q, seq_kv, head_dim;     /* head_dim ≤ 128 */
    tc_dtype_t io_dtype;                               /* F16 or BF16   */
    tc_dtype_t accum_dtype;                            /* F32           */
    float    softmax_scale;                            /* 1/sqrt(D)     */
    bool     causal;
    bool     return_lse;
    int32_t  kv_heads;                                 /* 0 → heads      */
    int32_t  window_size;                              /* 0 → no window  */
    const float* alibi_slopes;                         /* NULL → no bias */
} tc_attention_desc;

tc_status_t tc_attention_forward      (ctx, desc, Q, K, V, O, LSE);
tc_status_t tc_attention_forward_async(ctx, desc, Q, K, V, O, LSE, stream);
tc_status_t tc_attention_backward     (ctx, desc, Q, K, V, O, dO, LSE,
                                       dQ, dK, dV);
```

Buffer shapes (row-major):

```
Q   : [batch, heads,    seq_q,  head_dim]  fp16
K   : [batch, kv_heads, seq_kv, head_dim]  fp16
V   : [batch, kv_heads, seq_kv, head_dim]  fp16
O   : [batch, heads,    seq_q,  head_dim]  fp16
LSE : [batch, heads,    seq_q]              fp32   (nullable; set return_lse)
dO  : same shape as O                      fp16
dQ/dK/dV: same shapes as Q/K/V             fp16
```

## Math

The forward computes:

```
S[i, j] = (Q[i] · K[j]) * softmax_scale
        + (causal     ? -inf if j > i else 0)
        + (window     ? -inf if j < i - W + 1 else 0)
        - (alibi      ? alibi_slope[h] * (i - j) else 0)

P[i, :] = softmax(S[i, :])
O[i, :] = sum_j P[i, j] * V[j]
LSE[i]  = log(sum_j exp(S[i, j]))      (if return_lse)
```

The backward uses the LSE-saved scheme: recomputes P from S = Q·K^T - LSE
(numerically safe since LSE is the row-max-shifted), then computes dQ,
dK, dV in a single fused pass.

## Kernels

| Kernel source | head_dim | Pass | Tile (Br × Bc) | Family |
|---|---|---|---|---|
| `flash_attention.metal` | 64 | forward | 32 × 32 | Apple7+ |
| `flash_attention_d128.metal` | 128 | forward | 16 × 16 (v0.1) | Apple7+ |
| `flash_attention_backward.metal` | 64 | backward | 32 × 32 | Apple7+ |
| `flash_attention_backward_d128.metal` | 128 | backward | 16 × 16 (v0.1) | Apple7+ |
| `tensorops_flash_attention.metal` | 64 / 128 | forward | 64 × 64 (M5 tensor_ops) | Apple11 + SDK 26+ |

### Why Br × Bc = 32 × 32 at D=64, and 16 × 16 at D=128?

Threadgroup memory budget: ~32 KB on M-series. The on-chip working set per
CTA is `Br × Bc + Br × D + Bc × D` fp16 elements plus fp32 accumulators
and reductions.

At D=64, Br=Bc=32 (`flash_attention.metal`):
- `sQ`:  32 × 64 × 2 =  4 KB
- `sK`:  32 × 64 × 2 =  4 KB (reused for sP after K is consumed)
- `sV`:  32 × 64 × 2 =  4 KB
- `sS`:  32 × 32 × 4 =  4 KB (fp32, in sV region pre-V-load)
- `sP`:  32 × 32 × 2 =  2 KB (overlaps with sK)
- per-row scratch (m, l, alpha) + sO spill ≈ 1.5 KB
- total ≈ 22 KB ✓ — 4 simdgroups, 128 threads, TM_S=TN_S=2

At D=128, Br=Bc=16 (`flash_attention_d128.metal`):
- `S_tile`: 16 × 16 × 4 fp32 = 1 KB
- `Q_tile`: 16 × 128 × 2 = 4 KB
- `KV_tile`: 16 × 128 × 2 = 4 KB
- total ≈ 15 KB ✓ — Br/Bc smaller because the D dimension blows up
  per-element memory cost. Br=Bc=64 at D=128 would need ~64 KB.

The v0.2 plan ([ROADMAP.md](../ROADMAP.md)) lifts D=64 to Br=Bc=64 and
D=128 to Br=Bc=32 on Apple9+ via aliased threadgroup memory regions —
the 32 KB cap stays, but reusing the K-tile region for P after K is
consumed buys back the headroom.

## Function constants

`flash_attention.metal` exposes:

```metal
constant bool g_causal      [[function_constant(0)]];
constant bool g_use_lse     [[function_constant(1)]];
constant bool g_use_window  [[function_constant(2)]];
constant bool g_use_alibi   [[function_constant(3)]];
```

The pipeline cache compiles one specialized variant per `(causal,
return_lse, has_window, has_alibi)` combination. There are 16 possible
combinations; in practice the working set is 4-6.

## Causal masking

`causal = true` clamps each query to attend only to keys at positions
`j ≤ i`. The kernel skips entire K-blocks that lie strictly above the
diagonal (no work) and partially masks the boundary K-block. v0.2 adds
early-exit pruning at K-block granularity for very long sequences.

When `seq_q != seq_kv` (cross-attention or decode-step inference), the
mask uses absolute positions: query `i` attends to keys at `0..(seq_kv -
seq_q + i)`.

## GQA / MQA — Grouped-Query Attention

Set `kv_heads < heads`. The constraint is `heads % kv_heads == 0`. Each
KV head is shared by `heads / kv_heads` query heads. Common configurations:

| Model | heads | kv_heads | group |
|---|---:|---:|---:|
| Llama-2 7B | 32 | 32 | 1 (no GQA) |
| Llama-2 13B | 40 | 40 | 1 |
| Llama-2 70B | 64 | 8 | 8:1 GQA |
| Llama-3 8B | 32 | 8 | 4:1 GQA |
| Llama-3 70B | 64 | 8 | 8:1 GQA |
| MQA (e.g. PaLM) | H | 1 | H:1 |

The kernel reads K and V with the head index `h_kv = h / (heads/kv_heads)`.
No replication, no extra memory.

Validated by `tests/test_attention_correctness.c` with three GQA cases:
MQA (1 KV head), GQA H/2, and GQA H=8/KV=2 at D=128.

## Sliding-window attention

Set `window_size > 0`. Each query at position `i` attends only to keys
within `[i - window_size + 1, i]`. Mistral 7B uses `window_size = 4096`.
Combine freely with causal.

Inside the kernel the window is applied as another `-inf` offset in the
score-modification step. K-blocks entirely outside the window are still
loaded today; the K-block early-exit lives on the v0.2 list.

## ALiBi — Attention with Linear Biases

Set `alibi_slopes` to a host fp32 array of length `heads`. The kernel
applies:

```
S[i, j] -= alibi_slope[h] * (i - j)
```

`alibi_slopes` is read by the host at dispatch time and pushed into the
encoder via `setBytes:`. No extra GPU memory needed.

BLOOM and several open models use ALiBi instead of RoPE. Combine freely
with causal / window / GQA.

## Backward

`tc_attention_backward` is the FlashAttention-2 LSE-saved scheme. The
forward must have been called with `return_lse = true` (or LSE recomputed
equivalently — same row-max-shifted convention).

The Metal path covers head_dim = 64 and head_dim = 128. The D=128 kernels
use smaller attention tiles to stay within threadgroup-memory limits.

Tested by `tests/test_attention_backward.c` against a numerical-differences
reference at small shapes, validating dQ/dK/dV to ~3e-3 RMS-scaled error.

## Performance

Current-checkpoint measurements, M2 Ultra:

| Shape | dtype | causal | TFLOPS |
|---|---|:---:|---:|
| B=1, H=8,  S=2048, D=64 | F16 | yes | 3.82 |
| B=1, H=16, S=2048, D=64 | F16 | yes | 5.26 |
| B=1, H=16, S=4096, D=64 | F16 | yes | 6.51 |
| **B=1, H=32, S=4096, D=64** | **F16** | yes | **7.07** |
| B=1, H=32, S=4096, D=128 | F16 | yes | (Br=16, throughput-limited; v0.2) |

The fp32 accumulator path means we don't lose accuracy at long sequences;
RMS-scaled error stays ≤ 1e-3 vs an fp64 reference at S=4096.

`bench/bench_attention.c` sweeps common transformer shapes and reports
median TFLOPS plus tokens/sec equivalent.

## What v0.2 brings

- D=128 forward with Br=Bc=64 on Apple9+ via aliased TG memory regions.
- D=128 backward.
- K-block early-exit pruning for causal + sliding-window.
- Split-K for the short-seq → long-context generation case (each query is
  one tile; long key-cache is parallelized over a third axis).
- Match MFA's (Apple's open metal-flash-attention) numbers at the
  config points where it's tuned.

## Adding a new attention variant

If you want to add a new score modifier (per-query bias matrix, relative
position embedding, etc.):

1. Add a function constant to `flash_attention.metal` (e.g.
   `g_use_my_bias [[function_constant(4)]]`).
2. Add the score adjustment inside the inner loop, branched on the
   constant.
3. Add the corresponding field to `tc_attention_desc` (with `kv_heads=0`
   meaning "not set" semantics — don't break the ABI).
4. Update `lib/ops/attention.mm` to wire the constant into the pipeline
   lookup.
5. Add a correctness test.

GQA, window, and ALiBi all followed this pattern and added zero overhead
on the legacy paths.
