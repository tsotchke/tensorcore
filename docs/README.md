# tensorcore — documentation index

This directory is the full reference for `tensorcore`: the *what*, the *how*,
the *why*, and the *where it's going*. Top-level files in the repo root
(`README.md`, `ROADMAP.md`, `CHANGELOG.md`, `ONBOARDING.md`, `CONTRIBUTING.md`)
are the entry points; everything below goes deeper.

## Reading order for a new contributor

1. **[README.md](../README.md)** — the thesis ("CUDA for Apple"), what v0.1
   ships, and the relationship to the surrounding projects.
2. **[ONBOARDING.md](../ONBOARDING.md)** — 30-second tour, the working set,
   constraints to know.
3. **[architecture.md](architecture.md)** — how the library is put together
   internally: device init, pipeline cache, buffer pool, op dispatch,
   fallback ladder.
4. **[api_reference.md](api_reference.md)** — every public C symbol, grouped
   by header, with shapes and dtype rules.
   Or **[api_overview.md](api_overview.md)** for the flat one-line-per-
   symbol map.
5. **[cuda_comparison.md](cuda_comparison.md)** — the explicit
   tensorcore-vs-CUDA map. If you came from NVIDIA-land, start here.
6. **[ROADMAP.md](../ROADMAP.md)** — what's next and how confident we are.

## Reading order for a downstream consumer

1. **[integrating_tensorcore.md](integrating_tensorcore.md)** — build,
   install, link via CMake / pkg-config / Python.
2. **[api_reference.md](api_reference.md)** — the C ABI you'll be calling.
3. **[gguf.md](gguf.md)** + **[quantized.md](quantized.md)** — if you're
   loading a real model and running inference.
4. **[python.md](python.md)** — if you'd rather work in Python.
5. **[troubleshooting.md](troubleshooting.md)** — when something doesn't
   work.

## Reading order for an Eshkol / sibling-project integrator

1. **[eshkol_integration.md](eshkol_integration.md)** — bridge layout,
   builtins, opt-in env flag.
2. **[architecture.md](architecture.md)** — so you understand what the
   FFI layer is wrapping.
3. **[ROADMAP.md](../ROADMAP.md) §v0.4** — the consolidation plan that
   makes the three Metal backends in `eshkol-platform`,
   `quantum_geometric_tensor`, and `semiclassical_qllm` collapse to one.

## Topic index

### Concepts

- **[architecture.md](architecture.md)** — internals: device, pipeline
  cache, buffer pool, op dispatch, fallback ladder, autotune.
- **[dtypes.md](dtypes.md)** — the 10-dtype spectrum, what's native vs
  emulated, accumulation rules.
- **[family_gating.md](family_gating.md)** — Apple7..Apple11 detection,
  per-dtype hardware gates, SDK gates, and how the dispatch picks a path.
- **[cuda_comparison.md](cuda_comparison.md)** — direct
  cuBLAS / cuDNN / CUTLASS / NCCL / Triton ↔ tensorcore equivalents.

### Kernels

- **[gemm.md](gemm.md)** — `tc_gemm` and friends: tile sizes, kernel
  variants, env flags, autotune.
- **[attention.md](attention.md)** — `tc_attention_forward` and backward:
  FlashAttention-2 design, D=64 / D=128 paths, causal / GQA / sliding
  window / ALiBi.
- **[training_kernels.md](training_kernels.md)** — RMSnorm, LayerNorm,
  RoPE, SwiGLU, softmax, AdamW, and fused RMSnorm+GEMV.
- **[conv2d.md](conv2d.md)** — im2col + GEMM strategy, forward + backward.
- **[quantized.md](quantized.md)** — Q4_0 / Q8_0 packed format, GPU
  quantization, GEMV path, async batching.

### Subsystems

- **[gguf.md](gguf.md)** — the GGUF v3 reader, metadata helpers, bulk
  tensor loading, matrix descriptors.
- **[distributed.md](distributed.md)** — distributed primitives, single /
  ring / GLOO backends, the world_size=1 path, portable CPU TCP baseline,
  and fork tests.
- **[diloco.md](diloco.md)** — low-communication outer-loop training for
  cross-site meshes, plus current implementation status.
- **[python.md](python.md)** — the `tensorcore` Python binding, ctypes
  layout, numpy interop.

### Operations

- **[integrating_tensorcore.md](integrating_tensorcore.md)** — link tensorcore
  into another C / C++ / Python / CMake project.
- **[eshkol_integration.md](eshkol_integration.md)** — the Eshkol FFI
  bridge (drop-in shim, opt-in env flag).
- **[benchmarks.md](benchmarks.md)** — measured TFLOPS and tok/s per shape,
  dtype, and chip; how to reproduce.
- **[troubleshooting.md](troubleshooting.md)** — metallib not found,
  family gating misfires, dtype-unsupported errors, etc.
- **[codebase_audit.md](codebase_audit.md)** — what ICC's deterministic
  codebase tool sees: file count, call-graph stats, dead-code candidates,
  ground-truth confirmations of the architecture doc.
- **[kernels.md](kernels.md)** — per-file walkthrough of every `.metal`
  kernel: tile layouts, function constants, threadgroup memory budget.
- **[ci_and_scripts.md](ci_and_scripts.md)** — what each CI workflow
  runs and what every helper script in `scripts/` does.

### Recipes — assembling real workloads

- **[inference.md](inference.md)** — end-to-end Llama decode step from
  GGUF load through `tc_attention_forward` to next-token logits.
- **[training_loop.md](training_loop.md)** — one full transformer-block
  forward + backward + AdamW; every tensorcore call in order.

### Foundations

- **[memory_model.md](memory_model.md)** — unified memory, buffer pool,
  streams, command-buffer batching, threading.
- **[numerics.md](numerics.md)** — `rms_scaled` error metric, fp32
  accumulators, bit-exact guarantees, what the test suite enforces.
- **[faq.md](faq.md)** — common confusions answered in one place.

### Advanced topics

- **[precision_emulation.md](precision_emulation.md)** — SF64 / DF64 /
  FP24 / FP53 precision modes inherited from the eshkol-platform lineage.
- **[release_process.md](release_process.md)** — how a release goes from
  version bump → tag → CI → wheel → GitHub release artifact.
- **[development_setup.md](development_setup.md)** — zero-to-running
  guide for a fresh Mac (Apple Silicon path) and for non-Apple
  platforms (portable CPU only).
- **[observability.md](observability.md)** — runtime introspection:
  `tc_last_backend`, autotune cache, hardware evidence JSON, env
  knobs.
- **[glossary.md](glossary.md)** — every term used in the docs and
  source defined in one place.
- **[extending.md](extending.md)** — kernel-add tutorial with a worked
  example (a hypothetical `tc_gelu_forward`), worked through all five
  layers from `.metal` source to test.

### Per-directory READMEs

- **[../examples/README.md](../examples/README.md)** — what each
  compilable example demonstrates and how to read them.
- **[../tests/README.md](../tests/README.md)** — what each default and
  portable-CPU correctness test covers, including tolerances and skip
  semantics.
- **[../bench/README.md](../bench/README.md)** — what each TFLOPS /
  tok/s harness measures and how to interpret its output.

## Where the source lives

| What you want to read | Where to look |
|---|---|
| Public C ABI | `include/tensorcore/*.h` |
| Op dispatch (host) | `lib/ops/{gemm,attention,training,conv,quantized}.mm` |
| Device init / pipeline cache / buffer pool | `lib/core/{device,pipeline_cache,buffer_pool}.mm` |
| Autotune | `lib/core/autotune.cpp` |
| Metal kernels | `kernels/metal/*.metal` |
| GGUF reader | `lib/io/gguf.c` |
| Distributed | `lib/distributed/{distributed,ring_local}.mm`, `lib/distributed/{distributed_cpu,gloo_tcp}.cpp` |
| MPS + Accelerate fallbacks | `lib/fallback/{mps_gemm.mm,accelerate_gemm.c}` |
| M5 / Metal 4 TensorOps | `lib/tensorops/tensorops_m5.mm` (SDK-gated) |
| Eshkol bridge | `eshkol/bridge/tensorcore_codegen.cpp` |
| Python binding | `python/tensorcore/__init__.py` |
| Correctness tests | `tests/*.c` |
| Benchmarks | `bench/*.c` |
| Examples | `examples/*.c` |
