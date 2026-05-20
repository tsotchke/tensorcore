# tensorcore

**A CUDA-equivalent tensor-core acceleration layer for Apple Silicon — built
to make Mac competitive with NVIDIA on training and inference, at every scale
from on-device to small clusters of M-series Ultras.**

`tensorcore` is the canonical kernel library that the Eshkol toolchain (and
its sibling projects — `quantum_geometric_tensor`, `semiclassical_qllm`,
`attention`/GeoRefine) calls into for AI training and inference on Apple
Silicon GPUs. It does for Metal what cuBLAS + cuDNN + CUTLASS combined do for
CUDA: a single, hardware-aware library that turns the Apple GPU's matrix
units into fast, training-grade primitives.

## Why this exists

A pre-build audit of the existing Metal code on this machine found:

- `eshkol-platform/lib/backend/gpu/gpu_memory.mm` (4016 lines) — full Metal
  init, buffer pool, hardware-profile detection, 7-tier precision system
  (SF64, Ozaki-II, DF64, F32, FP24, FP53). **Already has** `simdgroup_matrix`
  for f32; no fp16/bf16/i8 path.
- `quantum_geometric_tensor/src/metal/` — 22 `.metal` files, 45+ kernels
  (attention, softmax, RK4, Adam, error-correction). **Does not** use
  `simdgroup_matrix`. Defines `AMXConfig` structs that never get invoked.
- `semiclassical_qllm/src/backend/backend_metal.m` + `sheaf_metal.c` —
  Riemannian-Adam + sheaf operations on Metal. Scalar SIMD matmul only.
- `attention/native_runtime/backends/metal.py` — Python shim.

Three hand-rolled Metal backends, none of them using Apple's actual
tensor-core primitive (`simdgroup_multiply_accumulate`, 8×8 MMA, fp16/bf16/i8
inputs with fp32 accum). `tensorcore` closes that gap, so the three downstream
projects can retire their bespoke kernels and consume one shared library.

## What v0.1 ships (measured on M2 Ultra)

| Component | Status | Numbers |
|---|---|---|
| `tc_gemm` fp32 | bit-exact vs Accelerate | 2.36 TFLOPS @ 4096³ |
| `tc_gemm` fp16 (M1+) | scaled-RMS err 5e-3 vs ref | **16.93 TFLOPS @ 4096³ (~63% of peak)** |
| `tc_gemm` bf16 (M3+, kernel built) | dispatch+gate verified | (run on M3+) |
| `tc_gemm` int8 (M4+, kernel built) | dispatch+gate verified | (run on M4+) |
| `tc_gemm_*_128` 128×128 tile | env-flag opt-in | regresses v0.1; v0.2 tunes |
| `tc_attention_forward` fp16 D=64 | scaled-RMS err 1e-3 vs fp64 ref | 6.70 TFLOPS @ S=4096 |
| `tc_attention_forward` fp16 D=128 | correctness verified | (bench harness v0.2) |
| MPS + Accelerate fallback | wired, exercised by dispatch | — |
| 6/6 correctness tests | pass on M2 Ultra | `ctest --test-dir build` |

### Public C ABI (`include/tensorcore/`)

- `tc_init` / `tc_shutdown` / `tc_device_info_get`
- `tc_buffer_alloc` / `tc_buffer_free` / `tc_buffer_map`
- `tc_gemm` / `tc_gemm_async` / `tc_gemm_batched` — fp16/bf16/fp32/int8 with
  alpha/beta scaling and transpose function constants
- `tc_attention_forward` / `tc_attention_forward_async` — D=64 and D=128 today
- `tc_last_backend` / `tc_backend_name` — diagnostic which path served the call

### Metal kernels (`kernels/metal/`)

- `gemm_simdgroup.metal` — 64×64 tile, BK=32, vec4 cooperative loads, fp32
  accum, dtype-templated (half / bfloat / float / int8)
- `gemm_simdgroup_128.metal` — 128×128 tile, available via `TC_USE_128_TILE=1`
- `flash_attention.metal` — fused QK·softmax·PV, online softmax, fp32 accum,
  function-constant causal+lse switches, D=64
- `flash_attention_d128.metal` — same for head_dim=128 (the llama / mistral
  standard)

### Apple GPU family gating

bf16 kernels gated to Apple9+ (M3, A17 Pro+).
int8 kernels gated to Apple10+ (M4+).
Metal-4 `MTL4MachineLearningCommandEncoder` gated to Apple11+/M5 via
`TC_ENABLE_TENSOROPS=ON` (stub; phase v0.3 will fill in).
All gating is runtime-detected via `MTLGPUFamily*`.

## What's on the v0.2 horizon (see ROADMAP.md)

- 20+ TFLOPS fp16 at 4096³ via double-buffered K-loads + 128×128 tile retune
- FlashAttention backward (the LSE-saved scheme)
- Fused training kernels: RMSnorm/LayerNorm fwd+bwd, RoPE, SwiGLU, AdamW
- D=128 FlashAttention bench + Br=64 path on Apple9+ (M3+) using larger TG mem
- Tile autotune per family (port the occupancy-aware scoring from
  `eshkol-platform/lib/backend/gpu/gpu_memory.mm:400-600`)
- Eshkol FFI bridge in `eshkol-platform/lib/ffi/tensorcore_ffi.cpp`
- Metal-4 TensorOps M5 fast path
- Multi-Mac distributed (TB5 ring + JACCL) — phase v0.5

## Build

```sh
cd ~/Desktop/tensorcore
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run smoke test + correctness suite
ctest --test-dir build --output-on-failure

# Bench (sweeps 256..4096 square GEMM, fp16/fp32/bf16, plus attention)
./build/bench/bench_gemm
./build/bench/bench_attention
```

`bench_gemm` prints the median TFLOPS and the backend that served each call.
On an M3 Max, fp16 simdgroup_matrix GEMM should land within ~10% of MLX's
hand-tuned kernels (which are our v0.2 target).

## Layout

```
tensorcore/
├── include/tensorcore/   ← Public C ABI headers (stable across versions)
├── lib/
│   ├── core/             ← Device init, pipeline cache, buffer pool
│   ├── ops/              ← gemm.mm, attention.mm — dispatch + encoding
│   ├── fallback/         ← MPS + Accelerate paths
│   ├── tensorops/        ← (Phase 4) Metal 4 / M5 TensorOps
│   └── c_api/            ← ABI shims
├── kernels/metal/        ← .metal sources, precompiled to default.metallib
├── cmake/                ← compile_metallib.cmake helper
├── tests/                ← Correctness suite vs Accelerate / fp64 reference
├── bench/                ← TFLOPS / tokens-per-sec harness
├── eshkol/               ← .esk bindings (functional once the FFI bridge ships)
└── docs/                 ← ROADMAP, eshkol_integration, etc.
```

## Relationship to the surrounding projects

```
                          ┌──────────────────┐
                          │   eshkol         │  (compiler/runtime)
                          └────────┬─────────┘
                                   │ FFI
                ┌──────────────────┼──────────────────┐
                │                  │                  │
   ┌────────────▼────────┐ ┌───────▼────────┐ ┌───────▼─────────┐
   │ eshkol-platform     │ │ qgt            │ │ semiclassical   │
   │ (Metal stub, was)   │ │ (45 kernels)   │ │ _qllm           │
   └────────────┬────────┘ └───────┬────────┘ └───────┬─────────┘
                │                  │                  │
                └──────────────────┼──────────────────┘
                                   │
                          ┌────────▼─────────┐
                          │   tensorcore     │  ← THIS
                          │  (this project)  │
                          └────────┬─────────┘
                                   │  Metal API
                          ┌────────▼─────────┐
                          │  Apple GPU       │
                          │  simdgroup_matrix│
                          │  + TensorOps M5  │
                          └──────────────────┘
```

After phase 5 of the roadmap, the three sibling projects share one Metal
backend; the duplicated `lib/backend/gpu/gpu_memory.mm` and qgt's `src/metal/`
become thin adapters over `tc_gemm` / `tc_attention_forward`.

## License

MIT.  See LICENSE.
