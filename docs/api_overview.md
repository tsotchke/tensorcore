# API overview — flat surface map

Every public C symbol in `include/tensorcore/*.h`, one line each. Use this
when you want to scan the whole surface in one pass. For full signatures
and shape semantics, see [api_reference.md](api_reference.md).

## Lifecycle

| Symbol | One-liner |
|---|---|
| `tc_init(out_ctx)` | Initialize the global context. Idempotent. |
| `tc_shutdown(ctx)` | Drain pending work, release the context. |
| `tc_device_info_get(ctx, out_info)` | Fill `tc_device_info` (family, name, capability flags). |
| `tc_version()` | Version string such as `"tensorcore 0.1.22"`. |

## Buffers

| Symbol | One-liner |
|---|---|
| `tc_buffer_alloc(ctx, bytes, out_buf)` | Allocate from the power-of-2 LIFO pool. |
| `tc_buffer_free(ctx, buf)` | Return to the pool. |
| `tc_buffer_map(buf, out_ptr)` | Get the CPU-addressable pointer (no copy on UMA). |
| `tc_buffer_size(buf)` | Byte size of the buffer. |

## Streams

| Symbol | One-liner |
|---|---|
| `tc_stream_create(ctx, out)` | New `MTLCommandBuffer` lane. |
| `tc_stream_destroy(ctx, s)` | Release the stream. |
| `tc_stream_sync(s)` | Commit pending CB; block until complete. |

## GEMM

| Symbol | One-liner |
|---|---|
| `tc_gemm(ctx, &desc, A, B, C)` | `C = α A·B + β C`. Sync; serves fp16/bf16/fp32/int8. |
| `tc_gemm_async(ctx, &desc, A, B, C, stream)` | Same, encoded into a stream. |
| `tc_gemm_batched(ctx, &batch_desc, A, B, C)` | Per-batch GEMM with element strides. |
| `tc_last_backend()` | Thread-local: which backend served the last GEMM/attention/tensorops call. |
| `tc_backend_name(b)` | Render the enum as a lowercase string. |
| `tc_tensorops_gemm_kernel_name(&desc, out_err)` | Diagnostic selector for the Metal 4 TensorOps GEMM kernel name. |

`tc_gemm_desc`: `M, N, K`, `a_dtype/b_dtype/c_dtype/accum_dtype`,
`transpose_a/b`, `alpha/beta`, `lda/ldb/ldc`.

## Attention

| Symbol | One-liner |
|---|---|
| `tc_attention_forward(ctx, &desc, Q, K, V, O, LSE)` | FlashAttention-2 forward, fused. |
| `tc_attention_forward_async(..., stream)` | Same, encoded into a stream. |
| `tc_attention_backward(ctx, &desc, Q, K, V, O, dO, LSE, dQ, dK, dV)` | LSE-saved backward. |

`tc_attention_desc`: `batch, heads, seq_q, seq_kv, head_dim`,
`io_dtype/accum_dtype`, `softmax_scale`, `causal`, `return_lse`,
`kv_heads` (GQA), `window_size` (sliding), `alibi_slopes` (host fp32 array).

## Training kernels

| Symbol | One-liner |
|---|---|
| `tc_rmsnorm_forward(ctx, X, gamma, Y, rstd_out, N, D, eps)` | Llama-style RMSnorm. `rstd_out` saved for backward. |
| `tc_rmsnorm_backward(ctx, X, gamma, dY, rstd, dX, dgamma, N, D)` | `dgamma` is **fp32** (accumulator). |
| `tc_layernorm_forward(ctx, X, γ, β, Y, mean_out, rstd_out, N, D, eps)` | Standard LayerNorm. |
| `tc_layernorm_backward(ctx, X, γ, dY, mean, rstd, dX, N, D)` | LayerNorm backward. |
| `tc_rope_forward(ctx, X, cos_t, sin_t, B, H, S, D)` | Rotary position embedding in-place. |
| `tc_swiglu_forward(ctx, gate, up, out, n)` | `silu(gate) * up`. |
| `tc_swiglu_backward(ctx, gate, up, dout, dgate, dup, n)` | SwiGLU backward. |
| `tc_softmax_forward(ctx, X, Y, N, D)` | Standalone row-wise softmax. |
| `tc_softmax_backward(ctx, Y, dY, dX, N, D)` | softmax backward. |
| `tc_adamw_step(ctx, p_fp32, m_fp32, v_fp32, grads, grad_dtype, n, lr, β1, β2, eps, wd, bc1, bc2)` | Fused AdamW step. |
| `tc_fused_rmsnorm_gemv(ctx, X, gamma, W, Y, M, N, K, eps)` | Inference primitive: norm + GEMV in one kernel (M ≤ 4). |

## Conv2D

| Symbol | One-liner |
|---|---|
| `tc_conv2d_forward(ctx, X, W, bias, Y, scratch_col, ...)` | im2col + `tc_gemm`. |
| `tc_conv2d_backward_input(ctx, dY, W, dX, scratch_col, scratch_dX_f32, ...)` | col2im with atomic fp32 accumulation. |
| `tc_conv2d_backward_weight(ctx, X, dY, dW, scratch_col, ...)` | Per-batch GEMM with buffer offset. |

## Quantized (Q4_0 / Q8_0)

| Symbol | One-liner |
|---|---|
| `tc_quantize_weights(ctx, W_fp16, W_quant, fmt, N, K)` | GPU quantize fp16 → Q4_0 or Q8_0. |
| `tc_gemv_quantized(ctx, X, W_quant, Y, fmt, M, N, K)` | Quantized GEMV (M ≤ 4). |
| `tc_gemv_quantized_async(..., stream)` | Same, encoded into a stream. |
| `tc_quantized_size(fmt, N, K)` | Byte size for a quantized `[N, K]` buffer. |

## GGUF (v3 reader)

| Symbol | One-liner |
|---|---|
| `tc_gguf_open(path, out)` | mmap + parse a GGUF v3 file. |
| `tc_gguf_close(f)` | munmap, release. |
| `tc_gguf_tensor_count(f)` / `tc_gguf_metadata_count(f)` | Stats. |
| `tc_gguf_get_tensor(f, name, out)` | Lookup tensor info by name. |
| `tc_gguf_tensor_at(f, i, out)` | Iterate tensors. |
| `tc_gguf_meta_get_str(f, key)` | String scalar metadata. |
| `tc_gguf_meta_get_i64(f, key, default)` | Signed-integer scalar metadata. |
| `tc_gguf_meta_get_f64(f, key, default)` | Floating-point scalar metadata. |
| `tc_gguf_meta_array_count(f, key)` | Array length. |
| `tc_gguf_meta_array_get_str(f, key, i, out_ptr, out_len)` | String array element. |
| `tc_gguf_meta_array_get_i64(f, key, i, default)` | Signed-integer array element. |
| `tc_gguf_meta_array_get_f64(f, key, i, default)` | Floating-point array element. |
| `tc_gguf_get_llama_config(f, out)` | Extract Llama-family config in one call. |
| `tc_gguf_tensor_to_buffer(ctx, f, name, out_buffer)` | Copy named tensor → owned `tc_buffer`. |
| `tc_gguf_tensor_quantized_matrix_info(tensor, out)` | mmap-side quantized matrix descriptor. |
| `tc_gguf_loaded_tensor_quantized_matrix_info(tensor, out)` | Loaded-model quantized matrix descriptor. |
| `tc_gguf_load_supported_tensors(ctx, f, out_model)` | Bulk-copy every supported tensor. |
| `tc_gguf_loaded_model_free(ctx, model)` | Release the bulk-loaded model. |
| `tc_gguf_loaded_tensor_count(model)` | Number of supported tensors copied into buffers. |
| `tc_gguf_loaded_skipped_tensor_count(model)` | Number of unsupported tensors skipped during bulk load. |
| `tc_gguf_loaded_tensor_at(model, i, out)` | Iterate loaded tensors. |
| `tc_gguf_loaded_get_tensor(model, name, out)` | Lookup a loaded tensor by name. |

## Distributed

| Symbol | One-liner |
|---|---|
| `tc_dist_init(ctx, backend, world_size, rank, rendezvous_url, out)` | New distributed context. |
| `tc_dist_finalize(d)` | Release the distributed context. |
| `tc_dist_world_size(d)` / `tc_dist_rank(d)` | Topology query. |
| `tc_allreduce(d, buf, n, dtype, op)` | In-place all-reduce (sum / avg / max / min). |
| `tc_broadcast(d, buf, n, dtype, root)` | From `root` to all. |
| `tc_allgather(d, in, out, n_per_rank, dtype)` | Concatenate `world_size × n_per_rank` elements. |
| `tc_barrier(d)` | All ranks meet before any continues. |

`tc_dist_backend_t`: `TC_DIST_SINGLE` (no-op), `TC_DIST_RING` (TB5, v0.5), `TC_DIST_GLOO` (portable CPU TCP collectives).

## Dtype + status

| Symbol | One-liner |
|---|---|
| `tc_dtype_size(d)` | Bytes / element (inline). |
| `tc_dtype_name(d)` | Render dtype as a lowercase string. |
| `tc_status_string(s)` | Render status as a human-readable string. |

## Enums and structs

### `tc_status_t`
`TC_OK`, `TC_ERR_NOT_INITIALIZED`, `TC_ERR_ALREADY_INITIALIZED`,
`TC_ERR_NO_DEVICE`, `TC_ERR_UNSUPPORTED_FAMILY`, `TC_ERR_UNSUPPORTED_DTYPE`,
`TC_ERR_INVALID_SHAPE`, `TC_ERR_INVALID_ARG`, `TC_ERR_ALLOC`,
`TC_ERR_KERNEL_NOT_FOUND`, `TC_ERR_PIPELINE`, `TC_ERR_DISPATCH`,
`TC_ERR_INTERNAL`.

### `tc_dtype_t`
`TC_DTYPE_F16`, `_BF16`, `_F32`, `_I8`, `_I32`, `_F64`, `_SF64`, `_DF64`,
`_FP24`, `_FP53`.

### `tc_family_t`
`TC_FAMILY_UNKNOWN`=0, `_APPLE7`=7 (M1), `_APPLE8`=8 (M2), `_APPLE9`=9
(M3 / A17 Pro), `_APPLE10`=10 (M4), `_APPLE11`=11 (M5).

### `tc_backend_t`
`TC_BACKEND_NONE`, `_SIMDGROUP_MATRIX`, `_TENSOROPS_M5`, `_MPS`,
`_ACCELERATE_CPU`, `_SF64_EMULATED`, `_OZAKI_II`, `_PORTABLE_CPU`.

### `tc_quant_t`
`TC_QUANT_Q4_0`, `TC_QUANT_Q8_0`.

### `tc_gguf_type_t`
`TC_GGUF_TYPE_F32`, `_F16`, `_Q4_0`, `_Q4_1`, `_Q8_0`, `_BF16`,
`_UNSUPPORTED`.

### `tc_dist_backend_t`
`TC_DIST_SINGLE`, `_RING`, `_GLOO`.

### `tc_reduce_op_t`
`TC_REDUCE_SUM`, `_AVG`, `_MAX`, `_MIN`.

## Headers, by file

| Header | What's in it |
|---|---|
| `tensorcore.h` | Umbrella; `TENSORCORE_VERSION_{MAJOR,MINOR,PATCH}`, `tc_version()`. |
| `status.h` | `tc_status_t` enum, `tc_status_string`. |
| `dtype.h` | `tc_dtype_t` enum, `tc_dtype_size`, `tc_dtype_name`. |
| `device.h` | `tc_context/buffer/stream` opaque types, `tc_family_t`, `tc_device_info`, lifecycle + buffer/stream entries. |
| `gemm.h` | `tc_gemm_desc`, `tc_gemm_batched_desc`, GEMM entries, `tc_backend_t`. |
| `attention.h` | `tc_attention_desc`, attention forward/backward. |
| `training.h` | RMSnorm, LayerNorm, RoPE, SwiGLU, softmax, AdamW, fused RMSnorm+GEMV. |
| `conv.h` | Conv2D forward + backward (dInput + dWeight). |
| `quantized.h` | `tc_quant_t`, Q4_0 / Q8_0 quantize + GEMV. |
| `gguf.h` | `tc_gguf_type_t`, file + tensor handles, metadata getters, bulk load. |
| `distributed.h` | `tc_dist_backend_t`, `tc_reduce_op_t`, distributed primitives. |
| `diloco.h` | DiLoCo config, parameter registration, local outer-step runtime. |
| `hip.h` | HIP/chipStar device discovery and backend selection diagnostics. |
| `cuda.h` | CUDA device discovery and backend selection diagnostics. |
| `memory_tier.h` | Buffer tier hints, promote/demote hooks, tier usage counters. |
| `checkpoint.h` | Activation-checkpoint lifecycle and resident/discarded counters. |
| `tensorcore.h` | Umbrella include for the public ABI. |

16 headers, 105 exported symbols (`cmake/tensorcore.exports`), full
Python wrapper parity in `python/tensorcore/__init__.py`.

## Per-backend coverage matrix

| Op family | simdgroup_matrix | tensorops_m5 | mps | accelerate_cpu | portable_cpu |
|---|:---:|:---:|:---:|:---:|:---:|
| `tc_gemm` (fp16/fp32) | ✓ | ✓ (M5 + SDK 26) | ✓ (fallback) | ✓ (fallback) | ✓ |
| `tc_gemm` (bf16) | ✓ (Apple9+) | ✓ (M5 + SDK 26) | ✓ (fallback) | ✓ (cast) | ✓ |
| `tc_gemm` (int8) | ✓ (Apple10+) | ✓ (M5 + SDK 26) | ✓ (fallback) | ✓ (widen) | ✓ |
| `tc_attention_forward` | ✓ | (v0.2) | — | — | ✓ |
| `tc_attention_backward` | ✓ (D=64) | — | — | — | ✓ |
| `tc_conv2d_*` | ✓ (im2col + gemm) | (inherits) | (inherits) | (inherits) | ✓ |
| `tc_rmsnorm_*` / training kernels | ✓ | — | — | — | ✓ |
| `tc_fused_rmsnorm_gemv` | ✓ | — | — | — | ✓ |
| `tc_gemv_quantized` | ✓ | — | — | — | ✓ |
| `tc_quantize_weights` | ✓ | — | — | — | ✓ |
| `tc_gguf_*` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `tc_allreduce` / `_broadcast` / `_allgather` / `_barrier` | ✓ (single, ring, fork) | ✓ | ✓ | ✓ | ✓ (single + GLOO TCP) |

`—` means returns `TC_ERR_UNSUPPORTED_FAMILY` on that build. The
portable-CPU backend now covers the main math and GLOO TCP collective
surface for non-Apple mesh workers while HIP/CUDA execution remains an
explicit unsupported path.

## See also

- [api_reference.md](api_reference.md) — full signatures + shape rules.
- [dtypes.md](dtypes.md) — every dtype's storage and accumulation rules.
- [family_gating.md](family_gating.md) — when each backend dispatches.
- [extending.md](extending.md) — adding new symbols to this surface.
