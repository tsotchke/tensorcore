# GGUF — loading real models

`gguf.h` is a memory-mapped GGUF v3 reader. It parses the header, metadata
table, and tensor info table; gives you direct mmap pointers into tensor
data; and provides a bulk-copy path that materializes every supported
tensor as a `tc_buffer*` ready for kernels.

This is the bridge between "a 4 GB Q4_0 llama.gguf on disk" and "a working
inference pipeline running on `tc_attention_forward` and
`tc_gemv_quantized`."

Spec reference: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

## What we support

| GGUF type | Code | Storage |
|---|---|---|
| F32 | `TC_GGUF_TYPE_F32` | float32 |
| F16 | `TC_GGUF_TYPE_F16` | float16 |
| BF16 | `TC_GGUF_TYPE_BF16` | bfloat16 |
| Q4_0 | `TC_GGUF_TYPE_Q4_0` | the format described in [quantized.md](quantized.md) |
| Q4_1 | `TC_GGUF_TYPE_Q4_1` | parsed; no kernel coverage in v0.1 |
| Q8_0 | `TC_GGUF_TYPE_Q8_0` | the format described in [quantized.md](quantized.md) |
| other | `TC_GGUF_TYPE_UNSUPPORTED` | skipped on bulk load |

Q4_K_M / Q5_K_M and the GGUF "k-quant" family are not parsed in v0.1.
They're v0.2 priorities.

## File layout

```
tc_gguf_open("model.gguf", &gguf);
   ↓
   mmap the file
   parse the header (magic GGUF, version 3)
   read N metadata entries (key, value-type, value)
   read M tensor info entries (name, dims, type, offset)
   ↓
gguf  →  tc_gguf_file*  (opaque)
```

`tc_gguf_close(gguf)` munmaps. String values inside metadata are
pointer+length pairs *into the mmap* — valid only until close.

## Three usage patterns

### Pattern 1 — inspect only

```c
tc_gguf_file* gguf = NULL;
tc_gguf_open("model.gguf", &gguf);

uint64_t nt = tc_gguf_tensor_count(gguf);
uint64_t nm = tc_gguf_metadata_count(gguf);
printf("%llu tensors, %llu metadata entries\n", nt, nm);

for (uint64_t i = 0; i < nt; ++i) {
    tc_gguf_tensor_info info;
    tc_gguf_tensor_at(gguf, i, &info);
    printf("  %s  type=%d  dims=[%llu,%llu,%llu,%llu]  bytes=%zu\n",
           info.name, info.type,
           info.dims[0], info.dims[1], info.dims[2], info.dims[3],
           info.n_bytes);
}

tc_gguf_close(gguf);
```

`examples/gguf_inspect.c` is exactly this pattern, plus pretty-printed
metadata and an optional tensor-to-buffer copy.

### Pattern 2 — copy individual tensors

```c
tc_gguf_file* gguf;  tc_gguf_open("model.gguf", &gguf);

tc_buffer* q_weight = NULL;
tc_gguf_tensor_to_buffer(ctx, gguf, "blk.0.attn_q.weight", &q_weight);

/* q_weight is now an owned tc_buffer; use it in kernels */
/* tc_buffer_free when done */

tc_gguf_close(gguf);
```

Allocates a new `tc_buffer`, copies the named tensor bytes into it (via
`tc_buffer_map` + `memcpy`). The mmap stays alive until `tc_gguf_close`;
the buffer is yours.

### Pattern 3 — bulk load

```c
tc_gguf_file*         gguf  = NULL;
tc_gguf_loaded_model* model = NULL;

tc_gguf_open("model.gguf", &gguf);
tc_gguf_load_supported_tensors(ctx, gguf, &model);

uint64_t loaded  = tc_gguf_loaded_tensor_count(model);
uint64_t skipped = tc_gguf_loaded_skipped_tensor_count(model);
printf("loaded %llu, skipped %llu\n", loaded, skipped);

tc_gguf_loaded_tensor_info q;
tc_gguf_loaded_get_tensor(model, "blk.0.attn_q.weight", &q);

/* q.buffer is owned by the loaded_model; do NOT tc_buffer_free it directly */

tc_gguf_loaded_model_free(ctx, model);    /* releases all owned buffers */
tc_gguf_close(gguf);
```

`tc_gguf_load_supported_tensors` allocates one `tc_buffer` per supported
tensor and bulk-copies. Unsupported encodings are skipped (not errors);
inspect `tc_gguf_loaded_skipped_tensor_count` to detect partial loads
before you build a runtime that depends on tensors that didn't load.

For a 7B Q4_0 llama.gguf (~4 GB on disk), bulk load takes a few seconds
and allocates roughly the same amount of memory (Apple's `Shared` storage
means the mmap and the buffer can share pages, but the current
implementation copies — this is on the v0.2 list to make zero-copy where
the on-disk layout permits).

## Metadata access

GGUF metadata is a typed key/value store. tensorcore exposes scalar and
array getters:

```c
const char* arch = tc_gguf_meta_get_str(gguf, "general.architecture");

int64_t  ctx_len = tc_gguf_meta_get_i64(gguf, "llama.context_length",
                                        2048 /* default */);
double   eps     = tc_gguf_meta_get_f64(gguf, "llama.attention.layer_norm_rms_epsilon",
                                        1e-5);

uint64_t vocab   = tc_gguf_meta_array_count(gguf, "tokenizer.ggml.tokens");
const char* tok0 = NULL;
size_t      tok0_len = 0;
tc_gguf_meta_array_get_str(gguf, "tokenizer.ggml.tokens", 0,
                            &tok0, &tok0_len);
/* tok0 is NOT NUL-terminated; use tok0_len */
```

For LLaMA-family models, `tc_gguf_get_llama_config` extracts the common
fields in one call:

```c
tc_gguf_llama_config cfg;
tc_gguf_get_llama_config(gguf, &cfg);
/* cfg.context_length, embedding_length, feed_forward_length, block_count,
 * attention_head_count, attention_head_count_kv, rope_dimension_count,
 * vocab_size, rms_norm_epsilon, rope_freq_base, rope_freq_scale */
```

This covers Llama, Mistral, Qwen, Gemma, Phi — anything that uses the
`llama.*` namespace in GGUF metadata.

## The `[K, N]` ↔ `[N, K]` translation

GGUF stores matrix tensors as `dim[0] = K`, `dim[1] = N`. But the kernel
APIs (`tc_gemv_quantized`, `tc_gemm`) take `N, K` parameters in that
order, with `W` interpreted as `[N, K]`. To avoid every consumer hand-coding
this, the GGUF reader provides a descriptor helper:

```c
tc_gguf_loaded_tensor_info proj;
tc_gguf_loaded_get_tensor(model, "blk.0.attn_q.weight", &proj);

tc_gguf_quantized_matrix_info q;
tc_gguf_loaded_tensor_quantized_matrix_info(&proj, &q);

tc_gemv_quantized(ctx, x, q.buffer, y, q.quant_type, 1, q.N, q.K);
```

`q.N` and `q.K` are derived from `proj.dims` with the GGUF convention
applied (`N = dims[1]`, `K = dims[0]`). `q.quant_type` is the matching
`tc_quant_t` enum (`TC_QUANT_Q4_0` or `TC_QUANT_Q8_0`).

The mmap-side variant is `tc_gguf_tensor_quantized_matrix_info(&tensor,
&q)`, which leaves `q.buffer` set to NULL (you didn't materialize the
tensor; you're working from the mmap pointer).

## Memory model

- The mmap is alive between `tc_gguf_open` and `tc_gguf_close`.
- `tc_gguf_tensor_info.data` points into the mmap; valid until close.
- `tc_gguf_loaded_tensor_info.buffer` is owned by the loaded model;
  released by `tc_gguf_loaded_model_free`.
- `tc_buffer*` returned by `tc_gguf_tensor_to_buffer` is owned by you;
  release with `tc_buffer_free`.
- String values from metadata are pointer+length into the mmap; copy if
  you need them past `tc_gguf_close`.

## What's deliberately not supported in v0.1

- **Writing GGUF.** Read-only. Use llama.cpp's `convert.py` to author files.
- **Lazy / partial load.** Bulk load is all-supported-tensors; for a 70B
  model you'll allocate ~40 GB at once. Future versions can lazy-load
  per-layer.
- **K-quants (Q*_K_M).** Parsed enough to skip; no GEMV kernel coverage.
- **Mixed-precision models.** A model with Q4_K_M on some layers and Q5_K_M
  on others will load the supported subset and report the rest as skipped.

## Testing

`tests/test_gguf.c` builds a synthetic GGUF file in memory (header + a
handful of metadata fields + several tensors of mixed types), parses it
back, and validates:

- metadata getters return expected values
- tensor info matches what was written
- `tc_gguf_tensor_to_buffer` round-trips a Q4_0 tensor and a Q4 GEMV
  result equals the dequantized reference
- bulk load picks up all supported tensors and counts the skipped one
- the matrix descriptor maps GGUF dims to N/K correctly

`examples/gguf_inspect.c` plus a real `tinyllama-1.1b-q4_0.gguf` is the
manual smoke. The bulk-load mode (`--load-supported`) is the path most
downstream runtimes will actually take.

## Where the GGUF reader lives

`lib/io/gguf.c` — compact pure C. No Metal dependency beyond the
`tc_context` + `tc_buffer` ABI it uses for the loaded-model variant. You
can read this end-to-end in an afternoon; it's the only "format parser"
the library has.
