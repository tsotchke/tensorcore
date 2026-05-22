# tensorcore — internal architecture

This document describes how the library is put together — what each layer
does, how a call flows from C ABI through Metal back to the user. It's the
prerequisite for adding kernels, debugging dispatch failures, or reasoning
about a new backend.

## The shape of a call

```
                 ┌──────────────────────────────────────────────┐
                 │  Caller (C, C++, Python, Eshkol, Swift, ...) │
                 └────────────────────┬─────────────────────────┘
                                      │  tc_gemm(...)
                                      ▼
                 ┌──────────────────────────────────────────────┐
                 │  include/tensorcore/*.h  — opaque C ABI       │
                 └────────────────────┬─────────────────────────┘
                                      │
                                      ▼
                 ┌──────────────────────────────────────────────┐
                 │  lib/ops/*.mm  — op dispatch (Obj-C++)        │
                 │  - pick kernel based on family + dtype + shape│
                 │  - look up MTLComputePipelineState in cache   │
                 │  - encode buffers, dispatch threadgroups      │
                 │  - record tc_last_backend                     │
                 └────────────────────┬─────────────────────────┘
                                      │
                       ┌──────────────┼─────────────────┬─────────────────┐
                       ▼              ▼                 ▼                 ▼
              ┌──────────────┐ ┌────────────┐  ┌──────────────┐ ┌───────────────┐
              │ kernels/     │ │ lib/       │  │ lib/         │ │ lib/fallback/ │
              │ metal/       │ │ tensorops/ │  │ distributed/ │ │ (MPS / cblas) │
              │ (.metal,     │ │ (M5 path)  │  │              │ │               │
              │  metallib)   │ │            │  │              │ │               │
              └──────────────┘ └────────────┘  └──────────────┘ └───────────────┘
```

Every public entry point lives in `lib/ops/<group>.mm`. Every entry point
in `lib/ops/*.mm` ends in either:
1. an `[encoder dispatchThreadgroups:...]` call against a precompiled
   `MTLComputePipelineState`, or
2. a `tc_set_last_backend(...)` + call into a fallback in `lib/fallback/`.

That's it. There is no graph, no IR, no scheduler. The library is a
collection of opinionated kernels with a tiny dispatch layer in front.

## Layout, file by file

```
include/tensorcore/
  status.h          ← tc_status_t error codes; tc_status_string
  dtype.h           ← tc_dtype_t enum (10 dtypes); tc_dtype_size, tc_dtype_name
  device.h          ← tc_context, tc_buffer, tc_stream opaque types
                      tc_family_t (Apple7..Apple11), tc_device_info
                      tc_init/shutdown, tc_buffer_alloc/free/map,
                      tc_stream_create/destroy/sync
  gemm.h            ← tc_gemm_desc, tc_gemm[/_async/_batched]
                      tc_backend_t enum, tc_last_backend, tc_backend_name
  attention.h       ← tc_attention_desc (with GQA, window, ALiBi fields)
                      tc_attention_forward[/_async], tc_attention_backward
  training.h        ← RMSnorm, LayerNorm, RoPE, SwiGLU, softmax, AdamW,
                      fused RMSnorm+GEMV
  conv.h            ← tc_conv2d_forward, tc_conv2d_backward_input/_weight
  quantized.h       ← tc_quant_t, tc_quantize_weights, tc_gemv_quantized[/_async]
                      tc_quantized_size
  gguf.h            ← tc_gguf_file, tc_gguf_loaded_model, *open/close,
                      metadata getters, tensor iteration, bulk load, matrix info
  distributed.h     ← tc_dist_ctx, tc_dist_backend_t, tc_reduce_op_t,
                      tc_dist_init/finalize, tc_allreduce, tc_broadcast,
                      tc_allgather, tc_barrier
  tensorcore.h      ← umbrella; includes all of the above
                      TENSORCORE_VERSION_{MAJOR,MINOR,PATCH}, tc_version()

lib/core/
  status.c          ← tc_status_string lookup table
  dtype.c           ← tc_dtype_name lookup
  device.mm         ← tc_init, tc_shutdown, tc_device_info_get
                      tc_buffer_alloc/free/map/size
                      tc_stream_*  — MTLDevice + MTLCommandQueue
                      tc_last_backend (thread-local), tc_backend_name
  pipeline_cache.mm ← name → MTLComputePipelineState cache
                      tc_pipeline_get, function-constant variants
  buffer_pool.mm    ← power-of-2 LIFO MTLBuffer pool keyed on size class
  autotune.cpp      ← (M, N, K, dtype) → best-tile heuristic
  internal.h        ← tc_kernel_kind_t, tc_set_last_backend, pool helpers

lib/ops/
  gemm.mm           ← simdgroup_matrix dispatch, fallback ladder, batched/async
  attention.mm      ← FlashAttention dispatch (D=64 / D=128, causal, GQA,
                      window, ALiBi via function constants); backward
  training.mm       ← RMSnorm, LayerNorm, RoPE, SwiGLU, softmax, AdamW,
                      fused RMSnorm+GEMV
  conv.mm           ← im2col → tc_gemm → col2im strategy
  quantized.mm      ← Q4_0 / Q8_0 quantize + GEMV; async-stream batching

lib/io/
  gguf.c            ← memory-mapped GGUF v3 reader (metadata + tensors),
                      bulk loader, matrix descriptor builder

lib/distributed/
  distributed.mm    ← dispatch over backend enum
  ring_local.mm     ← single-host ring all-reduce (threads + fork transport)

lib/fallback/
  mps_gemm.mm       ← MetalPerformanceShaders GEMM (any shape, any chip)
  accelerate_gemm.c ← cblas_sgemm CPU fallback (bit-exact reference)

lib/tensorops/
  tensorops_select.c← runtime feature-flag plumbing
  tensorops_m5.mm   ← mpp::tensor_ops MMA paths (SDK 26.0+ gated)

lib/c_api/
  c_api.mm          ← tc_version() and any ABI-only shims

kernels/metal/
  gemm_simdgroup.metal        ← 64×64 tile, BK=32, vec4 loads, fp32 accum
  gemm_simdgroup_128.metal    ← 128×128 tile (TC_USE_128_TILE=1)
  gemm_async.metal            ← async_copy variant (Xcode 16 only)
  gemm_async_128.metal        ← async_copy + 128 tile
  gemm_quantized.metal        ← Q4_0 / Q8_0 GEMV (M ≤ 4)
  gemm_quantized_v2.metal     ← faster Q4_0 GEMV; default since v0.1.6
  flash_attention.metal       ← D=64 fwd; causal + window + ALiBi constants
  flash_attention_d128.metal  ← D=128 fwd
  flash_attention_backward.metal       ← D=64 bwd (LSE-saved scheme)
  flash_attention_backward_d128.metal  ← D=128 bwd
  fused_norm_gemv.metal       ← fused RMSnorm + GEMV
  training_kernels.metal      ← RMSnorm, LayerNorm, RoPE, SwiGLU, softmax,
                                AdamW
  conv2d.metal                ← im2col helper kernels
  conv2d_backward.metal       ← col2im atomic accumulate
  tensorops_gemm.metal        ← M5 mpp::tensor_ops GEMM (SDK 26+)
  tensorops_flash_attention.metal ← M5 attention (SDK 26+)
  metal_simdgroup_event.h     ← simdgroup_event utility header

eshkol/
  bridge/
    tensorcore_codegen.cpp    ← FFI shim registered into eshkol-platform
    INTEGRATION.md            ← drop-in instructions
  tensorcore.esk              ← .esk type signatures
  hello_tensorcore.esk        ← sample .esk program

python/
  tensorcore/__init__.py      ← ctypes wrapper for the whole C ABI
  tests/test_basic.py         ← Python smoke test (gated CTest python_basic)

bench/
  bench_gemm.c                ← square 256..4096 sweep, all dtypes
  bench_attention.c           ← D=64 attention, common transformer shapes
  bench_inference_7b.c        ← Q4_0 7B decode latency harness

examples/
  hello_gemm.c                ← minimal C usage
  gguf_inspect.c              ← CLI for GGUF metadata + tensor copy
```

## Lifecycle

### `tc_init` (lib/core/device.mm)

1. Acquire the default `MTLDevice`.
2. Classify the device into `tc_family_t` by calling `supportsFamily:` for
   each `MTLGPUFamilyApple{7..11}`.
3. Detect `supports_bf16_simdgroup` (Apple9+), `supports_i8_simdgroup`
   (Apple10+), `supports_tensorops_m5` (SDK 26.0+ build plus M5/Metal4 runtime).
4. Build the default `MTLCommandQueue`.
5. Locate and load `tensorcore.metallib` (search order documented in
   [integrating_tensorcore.md](integrating_tensorcore.md)).
6. Initialize the pipeline cache (lazy; pipelines are compiled on first
   use, never at init).
7. Initialize the buffer pool (zero buffers preallocated; pool fills as
   you allocate).

`tc_init` is idempotent. Calling it twice returns `TC_ERR_ALREADY_INITIALIZED`.
Calling it from multiple threads is safe; the global context is guarded.

### `tc_buffer_alloc` (lib/core/buffer_pool.mm)

The pool is keyed by power-of-2 size class. On allocation:

1. Round the requested size up to the next power-of-2 class.
2. If the bucket has a free buffer, pop the most recent (LIFO; warm).
3. Otherwise, `[device newBufferWithLength: storageMode:Shared]`.
4. Wrap in `tc_buffer*` and return.

`tc_buffer_free` returns the buffer to its bucket; nothing is released back
to Metal until `tc_shutdown`.

Memory is `MTLStorageModeShared` — unified memory, CPU-mappable. Reads and
writes from the CPU are coherent without explicit synchronization on Apple
Silicon (per Apple's documented memory model for `Shared` storage on UMA
parts).

### `tc_gemm` (lib/ops/gemm.mm)

1. Validate shapes (`M, N, K > 0`, leading dims sensible).
2. Decide a kernel based on (dtype, family, shape):
   - fp16/fp32 + Apple7+ → `gemm_simdgroup` 64×64 tile
   - bf16 + Apple9+ → `gemm_simdgroup_bf16` variant
   - bf16 + Apple7..8 → fallback path: bit-cast bf16↔fp32, call fp32 kernel
   - i8 + Apple10+ → `gemm_simdgroup_i8` variant
   - i8 + Apple7..9 → fallback: widen to fp32, call fp32 kernel
   - 128×128 tile if `TC_USE_128_TILE=1` and shape supports it
   - M5 + SDK 26+ + tensorops on → `mpp::tensor_ops::matmul2d` path
   - shape outside kernel coverage → MPS fallback
   - if all GPU paths fail → Accelerate (`cblas_sgemm`)
3. Get the `MTLComputePipelineState` from the cache.
4. Encode buffers, dispatch threadgroups.
5. Commit and wait (sync) or hand the command buffer to the stream's
   pending buffer (async).
6. `tc_set_last_backend(...)` so the caller can introspect.

The same pattern applies to attention, training, conv, and quantized
ops — the kernel lookup and fallback ladder differ, but the structure is
identical.

### `tc_shutdown`

1. Drain pending command buffers on the default stream.
2. Release the pipeline cache (kernels are reclaimed by Metal).
3. Release the buffer pool — Metal frees the underlying memory.
4. Drop the command queue and device.

After `tc_shutdown`, the context handle is invalid. A subsequent `tc_init`
returns a fresh context.

## Autotune

`lib/core/autotune.cpp` is the host-side heuristic that picks GEMM and
attention tile shapes per `(family, dtype, M, N, K)`. The internal
surface:

- `tc_autotune_gemm_tile_for_family(family, dtype, M, N, K)` — returns a
  `tc_gemm_tile` (BM/BN/BK, simdgroup layout) recommended for the input.
- `tc_autotune_attention_tile_for_family(family, head_dim)` — returns a
  `tc_attention_tile` (Br, Bc, K-block size) sized to the threadgroup
  memory budget at that head_dim.
- `tc_autotune_run_sweep(...)`, `tc_autotune_load_cache(...)`,
  `tc_autotune_save_cache(...)` — the per-host sweep that fills the
  cache; persisted in a small JSON file the runtime reads at init.

The autotune surface is **not public** (these symbols live in
`lib/core/autotune.cpp`, not `include/`). External callers should not
depend on it; it exists so `lib/ops/gemm.mm` and `lib/ops/attention.mm`
can make uniform, hardware-aware tile decisions without inlining the
heuristic at every dispatch site.

## The pipeline cache

`lib/core/pipeline_cache.mm` is the key infrastructure for dispatch
performance. Compiling an `MTLComputePipelineState` from a `MTLFunction` is
~5-50 ms — far too slow for per-call. We compile each kernel at most once,
keyed by:

- function name (`@"gemm_f16_64x64"`)
- function constants set (causal flag, transpose flags, dtype constants)

The cache is a dictionary; entries are added on first use and live for the
process lifetime. The `function_constant` mechanism is how we compile one
`.metal` source into many specialized pipelines without `#define` recompiles.

## Function constants — Metal's compile-time switches

A Metal function constant looks like:

```metal
constant bool g_causal      [[function_constant(0)]];
constant bool g_use_lse     [[function_constant(1)]];
constant uint g_head_dim    [[function_constant(2)]];
```

At pipeline-creation time the host passes an `MTLFunctionConstantValues` to
the function, and Metal specializes the SPIR/AIR. The same `.metal` source
becomes a *causal D=64 with LSE* pipeline or *non-causal D=128 without LSE*
pipeline, with no runtime branches and no preprocessor games.

We use function constants for:

- `gemm_simdgroup.metal`: dtype, transpose_a, transpose_b
- `flash_attention.metal`: causal, use_lse, use_window, use_alibi
- `gemm_quantized.metal`: quant format

This is the Metal-native answer to CUDA's `__device__ template` instantiation.

## The fallback ladder

Every op has a backend ordering. For GEMM:

```
simdgroup_matrix  ←  best path, M-series Apple7+
   ↓ kernel not available (Apple7 + i8, etc.)
tensorops_m5       ←  Apple11 + SDK 26+ specifically
   ↓ shape not handled by kernels
mps                ←  MPSMatrix on the GPU
   ↓ MPS rejected the call
accelerate_cpu     ←  cblas_sgemm; always works; slowest
```

`tc_last_backend()` reports which path served the most recent GEMM or
attention call on this thread (see scope note below). This is how you
diagnose "why is this slow?": if you see `TC_BACKEND_MPS` or
`TC_BACKEND_ACCELERATE_CPU` on a path you expected
`TC_BACKEND_SIMDGROUP_MATRIX`, you found a kernel-coverage gap.

For attention on Metal builds, only the SIMDGROUP path exists today;
unsupported D values return `TC_ERR_UNSUPPORTED_DTYPE` rather than
falling back. CPU-only builds use the portable attention implementation.

**Scope of `tc_last_backend`:** at the v0.1.x checkpoint, the diagnostic
is updated from `lib/ops/gemm.mm` (5 sites), `lib/ops/attention.mm` (3
sites), and `lib/tensorops/tensorops_m5.mm` (2 sites). Training, conv, and
quantized kernels do not currently touch it. Widening to every dispatch
is a v0.2 polish item.

## Streams

A `tc_stream` corresponds to an MTLCommandBuffer that stays open across
dispatches. The async API (`tc_gemm_async`, `tc_attention_forward_async`,
`tc_gemv_quantized_async`) encodes into the stream's pending buffer without
committing. `tc_stream_sync` commits and waits.

This matters for inference: the Q4_0 7B decode harness reaches
**186 tok/s @ 632 GB/s** on M2 Ultra when GEMVs are batched into a single
stream — ~85% of theoretical LPDDR5 bandwidth, ~3× llama.cpp's published
numbers on the same chip. Without the stream, the command-buffer round
trip dominates the cheap GEMVs and tokens/sec collapses.

The default stream (NULL passed to a non-async API) commits and waits on
every call; it exists for ergonomics, not performance.

## Distributed

`lib/distributed/distributed.mm` is the dispatch layer; `ring_local.mm` is
the single-host implementation. Backend enum:

- `TC_DIST_SINGLE`: no-op all-reduce. `world_size=1` always succeeds; useful
  for testing the API and exercising the Eshkol bindings without a cluster.
- `TC_DIST_RING`: TB5 ring (phase v0.5, depends on macOS 26.2+ JACCL).
- `TC_DIST_GLOO`: CPU fallback over Ethernet (phase v0.5).

The single-host ring is exercised two ways in tests:

- `tests/test_distributed_ring.c`: threads + shared memory (correctness of
  the algorithm).
- `tests/test_distributed_ring_fork.c`: `fork()` + socketpairs (the real
  topology that the TB5 backend will use; transport swaps from
  socketpair to TB5/RDMA).

Bit-exact validated for 4 ranks × 1024 fp32.

## Eshkol bridge

`eshkol/bridge/tensorcore_codegen.cpp` is the FFI shim. It declares the C
ABI as external linkage and is compiled into the Eshkol LLVM module.
Activation is opt-in via `ESHKOL_ENABLE_TENSORCORE=1`. With the env unset,
Eshkol builds clean as before; with it set, `__tc-*` builtins resolve to
the corresponding `tc_*` C ABI calls. See
[eshkol_integration.md](eshkol_integration.md) and `eshkol/bridge/INTEGRATION.md`
for the drop-in steps.

## Build-time gates

Two SDK gates and three CMake options shape the build:

| Knob | Effect |
|---|---|
| `xcrun --show-sdk-version >= 26.0` | compiles `tensorops_*.metal`, defines `TC_HAVE_METAL4_SDK=1`, picks `-std=metal4.0` |
| `xcrun --show-sdk-version <  26.0` | compiles the legacy `gemm_async*.metal` (uses private `__asm` intrinsics that Xcode 17+ rejects) |
| `-DTC_BUILD_TESTS=ON/OFF` | enable/disable correctness suite |
| `-DTC_BUILD_BENCH=ON/OFF` | enable/disable TFLOPS / tok/s harness |
| `-DTC_BUILD_EXAMPLES=ON/OFF` | enable/disable `hello_gemm` and `gguf_inspect` |
| `-DTC_ENABLE_TENSOROPS=ON/OFF` | wire the M5 `mpp::tensor_ops` path into dispatch (defaults ON; takes effect only on M5 + SDK 26+) |
| `-DTC_ENABLE_METAL=ON/OFF` | gate the Apple Metal backend (defaults ON on Apple, OFF elsewhere). OFF builds only the portable CPU backend. |

The build always compiles a single `tensorcore.metallib`; which `.metal`
files are included depends only on the SDK gate, not on what chip the
build host has.

## Runtime gates

The dispatch layer reads `tc_device_info` at op time and skips paths the
chip doesn't support. There is no install-time configuration — one binary,
every M-series.

Environment overrides:

- `TC_METALLIB=<path>` — override `tensorcore.metallib` location.
- `TC_USE_128_TILE=1` — opt into the 128×128 GEMM tile (regresses v0.1).
- `TC_Q4_USE_V1=1` — use the original Q4_0 GEMV kernel for comparison.

These are all read once at dispatch time, by `getenv`; no global init step
is required.

## What this design optimizes for

- **Read-the-code-in-an-afternoon.** Everything is in C / ObjC++ / Metal,
  no codegen, no templates beyond function constants. `tensorcore.h` plus
  `lib/ops/gemm.mm` is enough to understand 80% of the library.
- **One library, every M-series.** Family detection at runtime; SDK
  detection at build time. Don't build different binaries.
- **Visible dispatch.** `tc_last_backend()` tells you which path served
  every call. No black boxes.
- **Fallback gracefully.** Old chips, new chips, weird shapes: you get a
  slower but correct result, not a failure.

## What this design *doesn't* try to do

- **Graph capture / scheduling.** That's the `tc_compile` story on the v0.7
  roadmap, not v0.1.
- **Op fusion at the framework level.** We ship a few hand-fused kernels
  (`tc_fused_rmsnorm_gemv`) and call it a day for v0.1-v0.3.
- **Multi-tenant scheduling.** One process, one context. If you want N
  processes, run N processes.
- **An IR.** Eshkol has one; we don't.

## Where to go next

- Add a kernel: [CONTRIBUTING.md](../CONTRIBUTING.md) and `kernels/metal/`.
- Trace a dispatch failure: `tc_last_backend()` + [troubleshooting.md](troubleshooting.md).
- Understand a specific op: [gemm.md](gemm.md), [attention.md](attention.md),
  [training_kernels.md](training_kernels.md), [conv2d.md](conv2d.md),
  [quantized.md](quantized.md).
