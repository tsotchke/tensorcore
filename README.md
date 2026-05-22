# tensorcore

**CUDA for Apple Silicon.**

`tensorcore` is the missing software layer that turns the matrix units on
M-series GPUs into a training-grade foundation. It does for Metal what
cuBLAS + cuDNN + CUTLASS + NCCL + ggml-quants combined do for CUDA: one
hardware-aware library, one C ABI, one binary that runs unchanged from
**M1 (Apple7) through M5 (Apple11)**.

```
                          ┌──────────────────────┐
                          │   tensorcore         │
                          │   ─ tc_gemm          │  ← cuBLAS
                          │   ─ tc_attention_*   │  ← cuDNN attention
                          │   ─ tc_conv2d_*      │  ← cuDNN conv
                          │   ─ tc_rmsnorm / RoPE│  ← cuDNN norms
                          │   ─ tc_swiglu / softmax / AdamW
                          │   ─ tc_gemv_quantized│  ← ggml Q4_0 / Q8_0
                          │   ─ tc_gguf_*        │  ← GGUF v3 reader
                          │   ─ tc_allreduce / broadcast / allgather
                          │                       ─ NCCL primitives
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  Apple GPU            │
                          │  ─ simdgroup_matrix   │  (M1+)
                          │  ─ mpp::tensor_ops    │  (M5+)
                          └───────────────────────┘
```

## The thesis

NVIDIA's moat in AI is **software stack maturity × silicon × interconnect**.
Two of those three are pure software. Apple has the silicon: per-watt
inference that NVIDIA can't match without changing chips, and unified
memory that removes the entire host/device transfer problem class. What
Apple is missing is the *cuBLAS-grade kernel library on top of Metal*.

`tensorcore` is the bet that closing that software gap completely changes
the economics for any team training models that fit in ≤32 Macs of unified
memory.

For the direct mapping of every CUDA primitive to its tensorcore
equivalent, see **[docs/cuda_comparison.md](docs/cuda_comparison.md)**.

## What v0.1 ships (measured, M2 Ultra)

| Component | Status | Numbers |
|---|---|---|
| `tc_gemm` fp32 | bit-exact vs Accelerate | 2.46 TFLOPS @ 4096³ |
| `tc_gemm` fp16 (Apple7+) | scaled-RMS err ≤ 5e-3 vs ref | **17.88 TFLOPS @ 4096³ (~66% of peak)** |
| `tc_gemm` bf16 (Apple9+ native, Apple7..8 fallback) | scaled-RMS ≤ 3e-3 | correctness verified |
| `tc_gemm` int8 (Apple10+ native, Apple7..9 fallback) | bit-exact i32 accum | correctness verified |
| `tc_gemm_*_128` 128×128 tile | env-flag opt-in | regresses v0.1; v0.2 retunes |
| `tc_attention_forward` fp16 D=64, causal/GQA/window/ALiBi | scaled-RMS ≤ 1e-3 | 7.07 TFLOPS @ B=1, H=32, S=4096 |
| `tc_attention_forward` fp16 D=128 | correctness verified | bench harness v0.2 |
| `tc_attention_backward` fp16 D=64 | scaled-RMS ≤ 3e-3 | LSE-saved scheme |
| Q4_0 / Q8_0 quantized GEMV plus GPU quantize | bit-exact vs dequant ref | 7B decode harness |
| Q4_0 async-stream batched GEMV | ~79% of LPDDR5 peak bw | **186 tok/s, 632 GB/s @ synthetic 7B decode** |
| RMSnorm / LayerNorm / RoPE / SwiGLU / softmax / AdamW | fused Metal kernels | C tests + Python smoke |
| Fused RMSnorm+GEMV | inference projection primitive | correctness vs separate path |
| Conv2D fwd + backward (im2col + GEMM) | scaled-RMS ≤ 1e-3 | multi-batch validated |
| GGUF reader | v3 metadata, tensors, bulk copy, Q4/Q8 descriptors | synthetic + Q4 GEMV end-to-end |
| Python ctypes binding | full ABI surface, NumPy interop | covered by CTest `python_basic` |
| Distributed (single-host ring) | bit-exact 4 ranks fp32 | thread + fork transports |
| MPS + Accelerate fallback | wired, exercised by dispatch | — |
| **Portable CPU backend** | builds on Linux / Intel-Mac with `TC_ENABLE_METAL=OFF`; covers buffers, streams, GEMM, attention/training/conv, GGUF, distributed-single, DiLoCo, and sparse compression. | for non-Apple mesh workers |
| CTest suite | 24/24 pass on M2 Ultra (22 library/package tests + 2 example smokes) | `ctest --test-dir build` |
| CMake / pkg-config / Python install | `tensorcore::tensorcore[_shared]`, `tensorcore.pc` | tested out-of-tree |

## Public C ABI — `include/tensorcore/*.h`

A 1.3K-line C ABI you can read end-to-end in an afternoon. Fifteen public
headers including the umbrella. Grouped:

- **Lifecycle:** `tc_init`, `tc_shutdown`, `tc_device_info_get`,
  `tc_buffer_alloc`/`_free`/`_map`/`_size`, `tc_stream_create`/`_destroy`/`_sync`.
- **GEMM:** `tc_gemm`, `tc_gemm_async`, `tc_gemm_batched` (fp16, bf16, fp32,
  int8). Diagnostics: `tc_last_backend`, `tc_backend_name`.
- **Attention:** `tc_attention_forward`/`_async`, `tc_attention_backward`.
  Causal, GQA, sliding window, ALiBi, LSE save — all via the same descriptor.
- **Training kernels:** `tc_rmsnorm_*`, `tc_layernorm_*`, `tc_rope_forward`,
  `tc_swiglu_*`, `tc_softmax_*`, `tc_adamw_step`, `tc_fused_rmsnorm_gemv`.
- **Conv2D:** `tc_conv2d_forward`, `tc_conv2d_backward_input`,
  `tc_conv2d_backward_weight`.
- **Quantized:** `tc_quantize_weights`, `tc_gemv_quantized`/`_async`,
  `tc_quantized_size`.
- **GGUF:** `tc_gguf_open`/`_close`, metadata getters, tensor iteration,
  `tc_gguf_load_supported_tensors`, matrix descriptor helpers,
  `tc_gguf_get_llama_config`.
- **Distributed:** `tc_dist_init`/`_finalize`, `tc_allreduce`,
  `tc_broadcast`, `tc_allgather`, `tc_barrier`.

Complete reference: **[docs/api_reference.md](docs/api_reference.md)**.

## Apple GPU family gating

| Family | Chips | Native MMA dtypes | TensorOps M5 |
|---|---|---|---|
| Apple7 | M1 | fp16, fp32 | — |
| Apple8 | M2 | fp16, fp32 | — |
| Apple9 | M3, A17 Pro | + bf16 | — |
| Apple10 | M4 | + int8 | — |
| Apple11 | M5 | (all of the above) | ✓ (SDK 26.0+ + M5 runtime) |

bf16 and int8 are software-fallback on older silicon, with the dispatch
choosing the fastest available path. One library binary; no per-chip
builds. See **[docs/family_gating.md](docs/family_gating.md)**.

## Where it slots in

```
                          ┌──────────────────┐
                          │   eshkol         │  (compiler/runtime)
                          └────────┬─────────┘
                                   │ FFI bridge (opt-in)
                ┌──────────────────┼──────────────────┐
                │                  │                  │
   ┌────────────▼────────┐ ┌───────▼────────┐ ┌───────▼─────────┐
   │ eshkol-platform     │ │ qgt            │ │ semiclassical   │
   │ (Metal stub now)    │ │ (45 kernels)   │ │ _qllm           │
   └────────────┬────────┘ └───────┬────────┘ └───────┬─────────┘
                │                  │                  │
                └──────────────────┼──────────────────┘
                                   │
                          ┌────────▼─────────┐
                          │   tensorcore     │  ← THIS
                          └────────┬─────────┘
                                   │  Metal API
                          ┌────────▼─────────┐
                          │  Apple GPU       │
                          └──────────────────┘
```

After [ROADMAP.md](ROADMAP.md) §v0.4, the three sibling projects retire
their bespoke Metal backends and consume one shared kernel library. The
SF64 / Ozaki-II / FP24 / FP53 precision modes that today live inside
`eshkol-platform/lib/backend/gpu/gpu_memory.mm` move into tensorcore as
named dtypes.

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

ctest --test-dir build --output-on-failure          # 24/24
./build/bench/bench_gemm                             # TFLOPS sweep
./build/bench/bench_attention                        # FlashAttention TFLOPS
./build/bench/bench_inference_7b                     # Q4_0 7B decode harness
./build/examples/hello_gemm                          # minimal C usage
./build/examples/gguf_inspect model.gguf             # inspect a GGUF file
./build/examples/gguf_inspect model.gguf --load-supported
```

On M3 Max, fp16 simdgroup_matrix GEMM should land within ~10% of MLX's
hand-tuned kernels (the v0.2 target). On M2 Ultra you should see
~17 TFLOPS at 4096³.

`bench_gemm` prints the median TFLOPS and the backend that served each
call. If you don't see `simdgroup_matrix`, see
**[docs/troubleshooting.md](docs/troubleshooting.md)**.

## Install and link

```sh
cmake --install build --prefix /opt/tensorcore
```

The install carries the umbrella headers, both libraries (static + shared),
the metallib, a CMake package config, and a pkg-config file:

```cmake
find_package(tensorcore CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE tensorcore::tensorcore_shared)
```

```sh
export PKG_CONFIG_PATH=/opt/tensorcore/lib/pkgconfig
cc main.c $(pkg-config --cflags --libs tensorcore) -o my_app
```

Python:

```sh
python3 -m pip install -e . --no-build-isolation
export TENSORCORE_LIB=/opt/tensorcore/lib/libtensorcore.dylib
python3 -c 'import tensorcore as tc; print(tc.version())'
```

Complete integration guide: **[docs/integrating_tensorcore.md](docs/integrating_tensorcore.md)**.
For a copyable out-of-tree project, see
**[examples/native_sdk_consumer](examples/native_sdk_consumer)**.

## Layout

```
tensorcore/
├── include/tensorcore/   ← Public C ABI headers (stable across versions)
├── lib/
│   ├── core/             ← Device init, pipeline cache, buffer pool, autotune
│   ├── ops/              ← gemm.mm, attention.mm, training.mm, conv.mm, quantized.mm
│   ├── fallback/         ← MPS + Accelerate paths
│   ├── tensorops/        ← Metal 4 / M5 TensorOps (SDK-gated)
│   ├── distributed/      ← Single-host ring + multi-Mac stubs
│   ├── io/               ← GGUF v3 reader
│   └── c_api/            ← ABI shims
├── kernels/metal/        ← .metal sources → default.metallib
├── cmake/                ← compile_metallib.cmake, tensorcoreConfig.cmake.in, .pc.in
├── tests/                ← CTest correctness, ABI, Python, and CPU-portability tests
├── bench/                ← TFLOPS / tok/s harness
├── examples/             ← hello_gemm, gguf_inspect
├── eshkol/               ← .esk bindings + FFI bridge for the Eshkol toolchain
│                            (see [eshkol/bridge/INTEGRATION.md](eshkol/bridge/INTEGRATION.md)
│                            for the drop-in steps)
├── python/               ← ctypes Python binding (full ABI surface)
└── docs/                 ← Architecture, API reference, ROADMAP, integration guides
```

## What's next (v0.2)

- **20+ TFLOPS fp16 4096³ on M2 Ultra** via double-buffered K-loads + 128×128
  tile retune.
- **FlashAttention parity with MFA** (Br=64 for D=128 on Apple9+, K-block
  early-exit pruning, split-K).
- **Full mixed-precision training loop test** (small transformer block,
  matched against PyTorch-MPS gradients).
- **RoPE backward, fused-AdamW for fp16 grads.**
- **M ≥ 4 quantized GEMV** so prefill works at scale.

The honest "compete-with-NVIDIA" picture, the per-watt advantage, and the
silicon-bound vs software-bound axes are all in
**[ROADMAP.md](ROADMAP.md)**.

## Documentation

- **[docs/](docs/)** — full documentation tree (architecture, API per
  header, CUDA comparison, dtypes, every kernel area, GGUF, Python,
  benchmarks, troubleshooting, ICC-grounded codebase audit).
- **[ONBOARDING.md](ONBOARDING.md)** — 30-second tour for a new contributor.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — how to add a kernel, add a
  backend target, run the suites.
- **[ROADMAP.md](ROADMAP.md)** — what's next and how confident we are.
- **[CHANGELOG.md](CHANGELOG.md)** — what's already shipped, per
  checkpoint.
- **[SECURITY.md](SECURITY.md)** — threat model, supported versions,
  how to report a vulnerability.
- **[examples/README.md](examples/README.md)** — what each compilable
  example (`hello_gemm`, `gguf_inspect`, `decode_step`, `training_step`)
  demonstrates.
- **[tests/README.md](tests/README.md)** — what each default and
  portable-CPU correctness test covers and the tolerance it enforces.
- **[bench/README.md](bench/README.md)** — what each bench measures
  (GEMM TFLOPS sweep, FlashAttention TFLOPS, 7B Q4_0 decode latency).

## License

MIT. See [LICENSE](LICENSE).
