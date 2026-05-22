# Python binding

`python/tensorcore/__init__.py` is a pure-Python `ctypes` wrapper around the
C ABI. No build step beyond installing the package -- the heavy lifting is
the native `libtensorcore.dylib` / `libtensorcore.so` you built with CMake.

The binding mirrors the C surface almost line-for-line and adds owned
object wrappers (`Context`, `Buffer`, `Stream`, `DistContext`,
`DiLoCoContext`, `GgufFile`, `LoadedModel`, `LoadedTensor`,
`QuantizedMatrix`) for ergonomic context-manager usage.

## Install

```sh
# Build and install the native library
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix /opt/tensorcore

# Install the Python package
python3 -m pip install -e . --no-build-isolation

# Point the binding at the native library if it's not on the default path
export TENSORCORE_LIB=/opt/tensorcore/lib/libtensorcore.dylib
export TC_METALLIB=/opt/tensorcore/lib/tensorcore.metallib
```

`pyproject.toml` declares the package; there are no native build hooks.
The binding finds the dylib via:

1. `TENSORCORE_LIB` environment variable (if set).
2. A package-local native library next to the loaded Python file
   (`libtensorcore.dylib` on macOS, `libtensorcore.so` on Linux,
   `tensorcore.dll` on Windows). The macOS release wheel also ships the
   metallib inside the package.
3. A few standard `/opt`, `/usr/local`, and CMake install prefixes.
4. The build tree (`build/libtensorcore.dylib`) — for editable installs
   without `cmake --install`.

If none of those work, you get a `TensorcoreError` at import with the
searched paths in the message.

## Smoke test

```python
import tensorcore as tc
print(tc.version())          # "tensorcore 0.1.22 (metallib_path)"

ctx = tc.init()
info = tc.device_info(ctx)
print(info.name, info.family)
tc.shutdown(ctx)
```

If you got a version string and a device name, the binding works.

## The big picture

The binding has three layers:

1. **Raw ctypes functions** — `tc.init`, `tc.gemm`, `tc.attention_forward`,
   ... — return status codes or raise `TensorcoreError`.
2. **Structures** — `TCDeviceInfo`, `TCGemmDesc`, `TCAttentionDesc`,
   `TCGGufLlamaConfig`, ... — mirror the C structs exactly.
3. **Owned object wrappers** — `Context`, `Buffer`, `Stream`,
   `DistContext`, `DiLoCoContext`, `GgufFile`, `LoadedModel`,
   `LoadedTensor`, `QuantizedMatrix` -- context-manager-friendly wrappers
   that own the underlying handle and release it on `close()` / scope exit.

All three styles are exposed. The object wrappers are the ergonomic path;
the raw functions match the C ABI 1:1 when you need it.

## NumPy interop

Buffers wrap the unified-memory pointer; `Buffer.write(arr)` and
`Buffer.read(arr)` do an in-place memcpy. `Buffer.to_numpy(shape, dtype)`
returns a NumPy array view that copies out.

```python
import numpy as np
import tensorcore as tc

with tc.Context() as ctx:
    host_a = np.random.randn(256, 256).astype(np.float16)
    host_b = np.random.randn(256, 256).astype(np.float16)

    A = ctx.buffer_from_array(host_a)      # convenience: alloc + write
    B = ctx.buffer_from_array(host_b)
    C = ctx.buffer(256 * 256 * 2)          # alloc only

    ctx.gemm(A, B, C, M=256, N=256, K=256, dtype="f16", accum="f32")

    out = C.to_numpy((256, 256), np.float16)
    print(out[:4, :4])
```

`buffer_from_array` accepts any C-contiguous NumPy array; `buffer(nbytes)`
allocates raw bytes. The Buffer class exposes `.write(arr)`, `.read(arr)`,
`.to_numpy(shape, dtype)`, `.map()` (raw pointer), `.size()` / `.nbytes()`.

## Raw API

Function names follow the C ABI with `tc_` dropped. Status codes raise
`TensorcoreError`.

### Lifecycle
`init`, `shutdown`, `device_info`, `buffer_alloc`, `buffer_free`,
`buffer_map`, `buffer_size`, `buffer_write`, `buffer_read`,
`stream_create`, `stream_sync`, `stream_destroy`.

### Diagnostics
`version`, `status_string`, `dtype_name`, `backend_name`, `last_backend`,
`last_backend_name`, `tensorops_gemm_kernel_name`.

### Memory tiering
`buffer_set_tier_hint`, `buffer_get_tier`, `buffer_promote_async`,
`buffer_demote_async`, `buffer_tier_sync`, `memory_tier_usage`.
The current runtime exposes the L0-only stub baseline; L1-L4 hosting lands
with the heterogeneous mesh runtime.

### Activation checkpointing
`checkpoint_register`, `checkpoint_discard`, `checkpoint_realize`,
`checkpoint_is_resident`, `checkpoint_unregister`,
`checkpoint_total_bytes_discarded`, `checkpoint_count_resident`,
`checkpoint_count_discarded`. The current runtime keeps buffers resident and
uses the calls for lifecycle validation plus counters until real discard /
reallocate support lands.

### HIP
`hip_init`, `hip_device_info_get`, `hip_device_count`, `hip_device_at`,
`hip_select_device`, `hip_last_kernel_name`. The in-tree HIP/chipStar
backend currently exposes deterministic unsupported diagnostics when no
runtime is built in.

### CUDA
`cuda_init`, `cuda_device_count`, `cuda_device_at`, `cuda_select_device`,
`cuda_last_kernel_name`. The in-tree CUDA backend currently exposes
deterministic unsupported diagnostics when no CUDA runtime is built in.

### GEMM
`gemm`, `gemm_async`, `gemm_batched`.

### Attention
`attention_forward`, `attention_forward_async`, `attention_backward`.
Descriptor fields: `causal`, `return_lse`, `kv_heads`, `window_size`,
`alibi_slopes` (pass a NumPy fp32 array; the binding pushes it via
`setBytes`).

### Training kernels
`rmsnorm_forward`, `rmsnorm_backward`, `layernorm_forward`,
`layernorm_backward`, `rope_forward`, `rope_backward`, `swiglu_forward`, `swiglu_backward`,
`softmax_forward`, `softmax_backward`, `adamw_step`, `fused_rmsnorm_gemv`.

### Conv2D
`conv2d_forward`, `conv2d_backward_input`, `conv2d_backward_weight`, plus
helpers `conv2d_output_shape`, `conv2d_scratch_bytes`,
`conv2d_backward_input_scratch_bytes`.

### Quantized
`quantize_weights`, `gemv_quantized`, `gemv_quantized_async`,
`quantized_size`.

### GGUF
`gguf_open`, `gguf_close`, `gguf_tensor_count`, `gguf_metadata_count`,
`gguf_meta_get_str`, `gguf_meta_get_i64`, `gguf_meta_get_f64`,
`gguf_meta_array_count`, `gguf_meta_array_get_str`,
`gguf_meta_array_get_i64`, `gguf_meta_array_get_f64`,
`gguf_get_llama_config`, `gguf_get_tensor`, `gguf_tensor_at`,
`gguf_tensor_to_buffer`, `gguf_tensor_quantized_matrix_info`,
`gguf_loaded_tensor_quantized_matrix_info`, `gguf_load_supported_tensors`,
`gguf_loaded_model_free`, `gguf_loaded_tensor_count`,
`gguf_loaded_skipped_tensor_count`, `gguf_loaded_tensor_at`,
`gguf_loaded_get_tensor`.

### DiLoCo
`diloco_config`, `diloco_init`, `diloco_finalize`,
`diloco_add_parameter`, `diloco_step`, `diloco_apply_outer`,
`diloco_outer_steps_completed`, `diloco_inner_steps_completed`,
`diloco_last_outer_step_seconds`, `diloco_last_outer_bytes_sent`.
`DiLoCoContext` wraps these functions for owned lifetime management.

## Object wrappers

### `Context`

```python
with tc.Context() as ctx:
    info = ctx.device_info()
    print(info.name, info.family)
    print(ctx.last_backend_name())     # "none" before any kernel runs
```

Methods (excerpt):

| Method | What it does |
|---|---|
| `buffer(nbytes)` | Allocate a raw `Buffer` of `nbytes` |
| `buffer_from_array(arr)` | Allocate + `write(arr)` in one call |
| `memory_tier_usage(tier)` | Return `(resident_bytes, capacity_bytes)` for a tier |
| `stream()` | Create a `Stream` |
| `dist(backend, world_size, rank, rendezvous_url=...)` | Create a `DistContext` |
| `hip_init()`, `hip_device_info()`, `hip_select_device(index)` | HIP/chipStar diagnostics and device selection |
| `gemm(A, B, C, M, N, K, **kwargs)` | sync GEMM (dtype/accum/transpose flags via kwargs) |
| `gemm_async(A, B, C, M, N, K, stream, **kwargs)` | async GEMM |
| `gemm_batched(A, B, C, batch, M, N, K, **kwargs)` | batched GEMM |
| `attention_forward(Q, K, V, O, batch, heads, seq_q, seq_kv, head_dim, ...)` | forward |
| `attention_forward_async(...)` | forward async |
| `attention_backward(Q, K, V, O, dO, LSE, dQ, dK, dV, ...)` | backward |
| `conv2d_forward(...)`, `conv2d_backward_input(...)`, `conv2d_backward_weight(...)` | conv2d |
| `quantize_weights(...)`, `gemv_quantized(...)`, `gemv_quantized_async(...)` | quantized |
| `rmsnorm_*`, `layernorm_*`, `rope_*`, `swiglu_*`, `softmax_*`, `adamw_step`, `fused_rmsnorm_gemv` | training kernels |
| `open_gguf(path)` | Open a GGUF file, return a `GgufFile` |
| `load_supported_tensors(gguf)` | Bulk-load supported tensors, return a `LoadedModel` |
| `last_backend()` / `last_backend_name()` | Read the diagnostic backend enum |
| `close()` | Tear down (also runs at `__exit__` / `__del__`) |

Lifetimes: the Context tracks weak references to every Buffer, Stream,
LoadedModel, and DistContext it creates and closes them in dependency
order on shutdown — you can't accidentally release the Context out from
under a live buffer.

### `Buffer`

```python
A = ctx.buffer_from_array(np.array([1.0, 2.0], dtype=np.float32))
print(A.nbytes(), A.size())     # 8, 8
arr = A.to_numpy((2,), np.float32)
A.write(np.array([3.0, 4.0], dtype=np.float32))
A.read(arr)                     # in-place read into existing array
```

Methods: `map()` (raw pointer), `size()`, `nbytes`, `write(arr)`,
`read(arr)`, `to_numpy(shape, dtype)`, `set_tier_hint(hint)`,
`get_tier()`, `promote_async(target_tier, stream=None)`,
`demote_async(target_tier, stream=None)`, `tier_sync()`, `close()`.

### `Stream`

```python
with ctx.stream() as s:
    ctx.gemm_async(A, B, C, M=256, N=256, K=256, stream=s)
    ctx.gemm_async(A, B, D, M=256, N=256, K=256, stream=s)
    s.sync()
```

Methods: `sync()`, `close()`. This is the path that yields the
**186 tok/s** Q4_0 GEMV throughput on the 7B decode harness — see
[benchmarks.md](benchmarks.md).

### `DistContext`

```python
with ctx.dist(backend=tc.TC_DIST_SINGLE, world_size=1, rank=0) as d:
    print(d.world_size, d.rank)              # 1, 0
    d.allreduce(grad_buffer, num_elements=n, dtype="f32",
                op=tc.TC_REDUCE_SUM)         # no-op at world_size=1
    d.barrier()
```

Default Apple and portable CPU builds expose `backend="gloo"` /
`TC_DIST_GLOO` with
`gloo+tcp://host:port` rendezvous URLs for TCP all-reduce, broadcast,
allgather, barrier, and dense or sparse TOPK DiLoCo outer steps.

### `DiLoCoContext`

```python
with ctx.dist("single", 1, 0, "single://diloco") as dist:
    with dist.diloco(inner_steps=100, outer_lr=1.0,
                     outer_optimizer="nesterov", compress="none") as d:
        d.add_parameter("tok_embeddings.weight", weight_buffer,
                        num_elements=weight_count, dtype="f32")
        if d.step():
            d.apply_outer()
        print(d.outer_steps_completed, d.last_outer_bytes_sent)
```

The runtime covers local/single-rank DiLoCo outer steps plus dense and
sparse TOPK multi-rank outer steps over `TC_DIST_GLOO`. Dropout-tolerant
WAN recovery and advanced compression modes raise `TensorcoreError` with
explicit unsupported status codes until those paths land.

Methods: `world_size`, `rank`, `allreduce(buf, n, dtype, op)`,
`broadcast(buf, n, dtype, root)`, `allgather(src, dst, n_per_rank, dtype)`,
`barrier()`, `close()`. The multi-Mac TB5 transport is the v0.5 work — see
[distributed.md](distributed.md).

### `GgufFile`

```python
with tc.GgufFile("llama-7b.gguf") as gguf:
    print(gguf.tensor_count(), "tensors")
    cfg = gguf.llama_config()
    print(cfg.embedding_length, cfg.attention_head_count)
    arch = gguf.meta_get_str("general.architecture")
    print(arch)                              # "llama"
```

Methods: `tensor_count()`, `metadata_count()`, `get_tensor(name)`,
`tensor_at(i)`, `meta_get_str(key)`, `meta_get_i64(key, default=)`,
`meta_get_f64(key, default=)`, `meta_array_count(key)`,
`meta_array_get_str(key, i)`, `llama_config()`, `tensor_to_buffer(ctx, name)`,
`load_supported_tensors(ctx)`, `close()`.

### `LoadedModel`

```python
with tc.Context() as ctx, tc.GgufFile("llama-7b.gguf") as gguf:
    model = ctx.load_supported_tensors(gguf)
    print(f"loaded {model.tensor_count()}, skipped {model.skipped_tensor_count()}")

    q = model.quantized_matrix("blk.0.attn_q.weight")
    print(q.N, q.K, q.quant_type)            # e.g. 4096 4096 TC_QUANT_Q4_0
```

Methods: `tensor_count()`, `skipped_tensor_count()`, `tensor_at(i)`,
`get_tensor(name)`, `quantized_matrix(name)`, `close()`.

### `LoadedTensor` and `QuantizedMatrix`

`LoadedTensor` is a `dict` subclass with a strong reference to the owning
`LoadedModel` and a guarded `buffer` accessor (raises if the model is
closed).

`QuantizedMatrix` carries `N`, `K`, `quant_type`, `gguf_type`, `n_bytes`,
`buffer`, plus convenience methods:

```python
q = model.quantized_matrix("blk.0.attn_q.weight")
x = ctx.buffer_from_array(activation)       # [1, K] fp16
y = q.output(M=1)                            # alloc [1, N] fp16

q.gemv(x, y, M=1)                            # sync
# or
with ctx.stream() as s:
    q.gemv_async(x, y, s, M=1)
    s.sync()
```

## Testing

`python/tests/test_basic.py` exercises every wrapped op:

- fp16 GEMM 256³ sync + async vs a NumPy reference (rms_scaled ≤ 5e-3)
- batched GEMM
- attention forward, forward_async, backward
- every training kernel forward path
- Q4_0 sync + async
- Q8_0 GPU quantize + GEMV
- GGUF round-trip and bulk load
- distributed single-host primitives plus GLOO smoke coverage
- LoadedModel / LoadedTensor / QuantizedMatrix wrappers

Registered as the CTest target `python_basic` when Python + NumPy are
available; runs in <1s on M2 Ultra.

## Errors

Every wrapped call checks the C return code and raises `TensorcoreError`
on non-`TC_OK`. The error message includes the integer code and the
`tc_status_string` rendering.

```python
try:
    tc.gemm(ctx, A, B, C, M=0, N=256, K=256, dtype="f16", accum="f32")
except tc.TensorcoreError as e:
    print(e)   # "tensorcore error -6: invalid_shape"
    print(e.status)   # -6
```

## What the binding doesn't do

- No automatic gradient tracking. This is `tensorcore`, not a framework.
  PyTorch-style autograd lives at a higher layer.
- No serialization of GGUF metadata into rich Python types beyond the
  `llama_config()` helper — generic metadata access is via the typed
  getters.
- No threadsafety guarantees. The C API is reentrant per-context but the
  Python wrappers don't release the GIL inside dispatches. Use one
  Python thread per context.
- No mutable view of a `Buffer` as a NumPy array — `to_numpy` copies. If
  you need zero-copy NumPy access to unified memory, build it on top of
  `buffer_map()` (`ctypes.cast` + `numpy.frombuffer`) at your own risk.
