# Inference — assembling a decode loop

`tensorcore` is a kernel library, not a model runtime. It ships the
primitives — `tc_attention_forward`, `tc_fused_rmsnorm_gemv`,
`tc_gemv_quantized`, `tc_rope_forward`, `tc_swiglu_forward`,
`tc_softmax_forward`, plus the GGUF loader — and you compose them. This
page is the assembly guide.

For real-model integration there are still missing pieces (tokenization,
sampling, KV-cache eviction) that live above tensorcore. This page covers
the matrix layer: what to call, in what order, with what shapes, to
produce a single decode token's logits from a real GGUF model.

## The shape

A Llama-architecture decode step is:

```
for each layer in 0..L:
    x_norm = RMSnorm(x)
    Q, K, V = projections(x_norm)             # tc_gemv_quantized × 3
    Q, K = rotate(Q, K)                       # tc_rope_forward × 2
    append (K, V) to KV-cache
    O = attention(Q, KV-cache)                # tc_attention_forward
    x = x + projection(O)                     # tc_gemv_quantized + residual
    x_norm2 = RMSnorm(x)
    gate, up = mlp_projections(x_norm2)       # tc_gemv_quantized × 2
    gu = swiglu(gate, up)                     # tc_swiglu_forward
    x = x + mlp_down(gu)                      # tc_gemv_quantized + residual
logits = lm_head(RMSnorm(x))                  # tc_gemv_quantized
next_token = sample(logits)                   # your code
```

Every step except the last `sample` is a tensorcore call. The trick is
batching them into one stream.

## End-to-end skeleton (C)

```c
#include "tensorcore/tensorcore.h"
#include <stdio.h>

int main(int argc, char** argv) {
    if (argc < 2) return 1;

    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) return 1;

    /* --- 1. Load the model --- */
    tc_gguf_file* gguf = NULL;
    tc_gguf_open(argv[1], &gguf);

    tc_gguf_llama_config cfg;
    tc_gguf_get_llama_config(gguf, &cfg);
    const int hidden     = (int)cfg.embedding_length;          /* 4096 for 7B  */
    const int mlp_dim    = (int)cfg.feed_forward_length;       /* 11008 for 7B */
    const int n_layers   = (int)cfg.block_count;               /* 32 for 7B    */
    const int n_heads    = (int)cfg.attention_head_count;
    const int n_kv_heads = (int)cfg.attention_head_count_kv;   /* may differ (GQA) */
    const int head_dim   = (int)cfg.rope_dimension_count;
    const float rms_eps  = (float)cfg.rms_norm_epsilon;
    const float rope_base = (float)cfg.rope_freq_base;
    (void)rope_base; /* used in cos/sin table construction */

    tc_gguf_loaded_model* model = NULL;
    tc_gguf_load_supported_tensors(ctx, gguf, &model);

    /* --- 2. One-time scratch + RoPE cos/sin --- */
    tc_buffer *x, *x_norm, *q, *k_step, *v_step, *o, *attn_out;
    tc_buffer_alloc(ctx, hidden * 2, &x);          /* fp16; M=1 decode */
    tc_buffer_alloc(ctx, hidden * 2, &x_norm);
    tc_buffer_alloc(ctx, hidden * 2, &q);
    tc_buffer_alloc(ctx, n_kv_heads * head_dim * 2, &k_step);
    tc_buffer_alloc(ctx, n_kv_heads * head_dim * 2, &v_step);
    tc_buffer_alloc(ctx, hidden * 2, &o);
    tc_buffer_alloc(ctx, hidden * 2, &attn_out);

    /* MLP scratch */
    tc_buffer *gate, *up, *gu, *mlp_out;
    tc_buffer_alloc(ctx, mlp_dim * 2, &gate);
    tc_buffer_alloc(ctx, mlp_dim * 2, &up);
    tc_buffer_alloc(ctx, mlp_dim * 2, &gu);
    tc_buffer_alloc(ctx, hidden * 2, &mlp_out);

    /* KV-cache buffers (per layer × per kv-head × max_seq × head_dim) — your
     * code allocates these once at startup. */

    /* RoPE cos/sin tables — precomputed for the max seq length you'll handle. */
    /* tc_buffer* cos_t, *sin_t; ... populate per the rope_freq_base from cfg */

    /* --- 3. Embed the first token --- */
    /* Reuse the embedding table tensor: model->get_tensor("token_embd.weight").
     * Pull token_id-th row into x. (Embedding is row-lookup, not a GEMV.) */

    /* --- 4. Run the decode --- */
    tc_stream* s = NULL;
    tc_stream_create(ctx, &s);

    for (int layer = 0; layer < n_layers; ++layer) {
        /* fetch this layer's projection matrices via the loaded-model lookup */
        tc_gguf_loaded_tensor_info wq, wk, wv, wo, w_gate, w_up, w_down;
        tc_gguf_loaded_tensor_info attn_norm, mlp_norm;
        char name[128];
        snprintf(name, sizeof name, "blk.%d.attn_q.weight", layer);
        tc_gguf_loaded_get_tensor(model, name, &wq);
        snprintf(name, sizeof name, "blk.%d.attn_k.weight", layer);
        tc_gguf_loaded_get_tensor(model, name, &wk);
        /* ... same for wv, wo, w_gate, w_up, w_down, attn_norm, mlp_norm */

        /* fused RMSnorm + Q projection (the GEMV layout requires N=hidden,
         * K=hidden for self-projection; Wq is Q4_0). */
        tc_gguf_quantized_matrix_info q_info;
        tc_gguf_loaded_tensor_quantized_matrix_info(&wq, &q_info);
        /* For inference: fused norm+gemv, then separate gemv for K/V */
        tc_fused_rmsnorm_gemv(ctx, x, /*attn_norm.buffer*/NULL, q_info.buffer,
                              q, 1, q_info.N, q_info.K, rms_eps);

        tc_gguf_quantized_matrix_info k_info, v_info;
        tc_gguf_loaded_tensor_quantized_matrix_info(&wk, &k_info);
        tc_gguf_loaded_tensor_quantized_matrix_info(&wv, &v_info);
        /* x_norm is still in flight from the fused step above; for K/V we'd
         * re-norm. In a real runtime you'd either: emit a tc_rmsnorm_forward
         * first and use tc_gemv_quantized × 3, or fold all three projections
         * into Wqkv and use one fused call. */
        tc_gemv_quantized_async(ctx, x_norm, k_info.buffer, k_step,
                                k_info.quant_type, 1, k_info.N, k_info.K, s);
        tc_gemv_quantized_async(ctx, x_norm, v_info.buffer, v_step,
                                v_info.quant_type, 1, v_info.N, v_info.K, s);

        /* RoPE on Q and K at the current sequence position */
        /* tc_rope_forward(ctx, q, cos_t_at_pos, sin_t_at_pos,
                          1, n_heads, 1, head_dim);
           tc_rope_forward(ctx, k_step, cos_t_at_pos, sin_t_at_pos,
                          1, n_kv_heads, 1, head_dim); */

        /* Append k_step / v_step into the KV-cache at the current position.
         * (Your code; tensorcore doesn't manage this for you.) */

        /* Attention over the KV-cache so far */
        tc_attention_desc adesc = {0};
        adesc.batch       = 1;
        adesc.heads       = n_heads;
        adesc.kv_heads    = n_kv_heads;        /* GQA-aware */
        adesc.seq_q       = 1;                 /* one query token */
        adesc.seq_kv      = /* current cache length */ 1;
        adesc.head_dim    = head_dim;
        adesc.io_dtype    = TC_DTYPE_F16;
        adesc.accum_dtype = TC_DTYPE_F32;
        adesc.softmax_scale = 1.0f / sqrtf((float)head_dim);
        adesc.causal      = true;
        /* tc_attention_forward_async(ctx, &adesc, q, KV_cache_k, KV_cache_v,
                                      o, NULL, s); */

        /* Output projection */
        tc_gguf_quantized_matrix_info o_info;
        tc_gguf_loaded_tensor_quantized_matrix_info(&wo, &o_info);
        tc_gemv_quantized_async(ctx, o, o_info.buffer, attn_out,
                                o_info.quant_type, 1, o_info.N, o_info.K, s);

        /* Residual: x += attn_out — small elementwise kernel (or fused into
         * the next norm). Your code or a small custom kernel. */

        /* MLP block: fused norm + gate, then up, then SwiGLU, then down */
        tc_gguf_quantized_matrix_info gate_info, up_info, down_info;
        tc_gguf_loaded_tensor_quantized_matrix_info(&w_gate, &gate_info);
        tc_gguf_loaded_tensor_quantized_matrix_info(&w_up, &up_info);
        tc_gguf_loaded_tensor_quantized_matrix_info(&w_down, &down_info);

        tc_fused_rmsnorm_gemv(ctx, x, /*mlp_norm.buffer*/NULL, gate_info.buffer,
                              gate, 1, gate_info.N, gate_info.K, rms_eps);
        tc_gemv_quantized_async(ctx, x_norm, up_info.buffer, up,
                                up_info.quant_type, 1, up_info.N, up_info.K, s);
        tc_swiglu_forward(ctx, gate, up, gu, mlp_dim);

        tc_gemv_quantized_async(ctx, gu, down_info.buffer, mlp_out,
                                down_info.quant_type, 1, down_info.N, down_info.K, s);

        /* Residual: x += mlp_out */
    }

    /* Final norm + lm_head projection over vocab */
    /* tc_rmsnorm_forward(ctx, x, output_norm, x_norm, rstd, 1, hidden, rms_eps);
       tc_gemv_quantized_async(ctx, x_norm, lm_head_q, logits,
                               TC_QUANT_Q4_0, 1, vocab_size, hidden, s); */

    tc_stream_sync(s);
    tc_stream_destroy(ctx, s);

    /* logits buffer now holds the next-token distribution — your sample() */

    tc_gguf_loaded_model_free(ctx, model);
    tc_gguf_close(gguf);
    tc_shutdown(ctx);
    return 0;
}
```

This isn't compilable as-is — the KV-cache, RoPE table construction,
the residual elementwise add, the embedding lookup, and sampling are
left as user code. But every load-bearing matrix call goes through one
of the documented tensorcore entry points.

## Shape table for a 7B Llama

```
hidden       = 4096
mlp_dim      = 11008
n_heads      = 32
n_kv_heads   = 32       (no GQA in Llama-2 7B; Llama-3 8B has kv=8)
head_dim     = 128
n_layers     = 32
vocab        = 32000
```

Per token per layer:

| Op | Inputs / Outputs | Q4_0 weight read |
|---|---|---:|
| RMSnorm | `x[hidden]` → `x_norm[hidden]` | — |
| Wq projection | `x_norm[hidden]` × `Wq[hidden,hidden]` → `q[hidden]` | ~9.4 MB |
| Wk projection | `x_norm[hidden]` × `Wk[hidden,hidden]` → `k[hidden]` | ~9.4 MB |
| Wv projection | `x_norm[hidden]` × `Wv[hidden,hidden]` → `v[hidden]` | ~9.4 MB |
| RoPE | `q[H,D]` and `k[Hkv,D]` rotated in place | — |
| Attention | `q[H,D]` + `KV_cache[Hkv,S,D]` → `o[hidden]` | — |
| Wo projection | `o[hidden]` × `Wo[hidden,hidden]` → `attn_out[hidden]` | ~9.4 MB |
| Residual | `x += attn_out` | — |
| RMSnorm | `x[hidden]` → `x_norm[hidden]` | — |
| W_gate, W_up | `x_norm[hidden]` × `W*[mlp_dim,hidden]` → `gate, up[mlp_dim]` | ~50.6 MB |
| SwiGLU | `gate * silu(gate) * up` (in place) | — |
| W_down | `gu[mlp_dim]` × `W_down[hidden,mlp_dim]` → `mlp_out[hidden]` | ~25.3 MB |
| Residual | `x += mlp_out` | — |

Per layer: ~113 MB of weight traffic. × 32 layers = ~3.6 GB per token.

At 800 GB/s LPDDR5 peak on M2 Ultra, the theoretical decode ceiling is
~220 tok/s. The current async-batched harness lands at **186 tok/s @
632 GB/s effective** — 79% of peak; ~3× llama.cpp on the same chip. See
[benchmarks.md](benchmarks.md).

## The async-stream pattern

Every GEMV between `tc_stream_create` and `tc_stream_sync` shares a
single pending `MTLCommandBuffer`. The command-buffer commit cost is
**~50µs** on M2 Ultra; per-token you'd have ~200 GEMVs across 32 layers,
so 10ms+ of pure CB overhead if you don't batch.

With `_async` everywhere and one `sync` at the end of the decode step,
that overhead collapses to a single commit. This is the difference
between "memory-bound" (632 GB/s on this harness) and "dispatch-bound"
(would be 10-20× slower).

**Don't sync between layers.** Sync at the end of the token, not the end
of the layer. The kernel for layer N+1 can be encoded before layer N's
GEMVs have finished.

## What tensorcore doesn't handle for you

- **Tokenization.** GGUF carries the tokenizer vocabulary
  (`tokenizer.ggml.tokens`), but tensorcore doesn't tokenize. Use a
  separate library (e.g. tokenizers.cpp).
- **Sampling.** The `lm_head` output is a `vocab`-wide logit buffer.
  You read it via `tc_buffer_map`, apply top-k/top-p/temperature, sample,
  and feed the next token in. No GPU sampling kernel yet.
- **KV-cache eviction.** Allocate enough KV-cache for your max sequence
  length up-front. The `tc_buffer` is yours.
- **Embedding lookup.** It's a row-gather from `token_embd.weight`, not a
  matmul. No tensorcore kernel; copy the row from `tc_buffer_map`'d
  memory.
- **Cross-token continuity.** Each decode step is its own sequence of
  calls. State (KV-cache, position) lives in your code.

## See the assembly in action

The skeleton above is the matrix-layer plumbing. Two concrete
exercises ship with the repo:

- **[`examples/decode_step.c`](../examples/decode_step.c)** — compilable
  end-to-end. Two synthetic Llama layers driving the exact primitive
  sequence above: `tc_fused_rmsnorm_gemv → tc_gemv_quantized_async ×
  2 → tc_rope_forward × 2 → tc_attention_forward → tc_gemv_quantized
  → tc_fused_rmsnorm_gemv → tc_gemv_quantized_async → tc_swiglu_forward
  → tc_gemv_quantized`. Built automatically by `cmake --build build`
  and runnable as `./build/examples/decode_step`.
- **`bench/bench_inference_7b.c`** — the synthetic 7B Q4_0 GEMV
  throughput bench that produces the 186 tok/s number.

A complete `llama_decode.c` that wraps tokenization + sampling + KV-cache
+ the GEMV loop into one runnable binary lands as part of the v0.2
"end-to-end inference" milestone. See [ROADMAP.md](../ROADMAP.md).
Real-model integration changes the weights the existing kernel calls
read, not the calls themselves.
