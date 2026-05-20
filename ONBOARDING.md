# Onboarding to tensorcore

You're picking up `tensorcore` cold. Here's the working set.

## What this is

A CUDA-equivalent kernel library for Apple Silicon GPUs. Real `simdgroup_matrix`
GEMM (~17.5 TFLOPS fp16 on M2 Ultra, ~65% of peak), fused FlashAttention
forward + backward, full transformer training kernel set (RMSnorm/LayerNorm/
RoPE/SwiGLU/AdamW/softmax), Conv2D forward, distributed primitives stub.
Written in Metal + Objective-C++ with a C ABI. Eshkol bridge included.

`include/tensorcore/tensorcore.h` is the umbrella header — that's the surface
area for external consumers.

## 30-second tour

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure   # 9/9 tests
./build/bench/bench_gemm                       # TFLOPS numbers
./build/examples/hello_gemm                    # Minimal use
```

## Where the value lives

| Concern | File(s) |
|---|---|
| The kernels | `kernels/metal/*.metal` |
| Public ABI | `include/tensorcore/*.h` |
| Device init / pipeline cache / buffer pool | `lib/core/{device,pipeline_cache,buffer_pool}.mm` |
| Op dispatch | `lib/ops/{gemm,attention,training,conv}.mm` |
| M5 path | `lib/tensorops/tensorops_m5.mm` (SDK 26+ gated) |
| Distributed | `lib/distributed/distributed.mm` |
| Eshkol bridge | `eshkol/bridge/tensorcore_codegen.cpp` |
| Build wiring | `CMakeLists.txt`, `cmake/compile_metallib.cmake` |
| The plan | `ROADMAP.md` |
| Current state | `CHANGELOG.md` |

## Important constraints to know

1. **Apple GPU AGX ISA is undocumented.** We don't write assembly. The hottest
   abstraction we use is MSL's `simdgroup_matrix` + `mpp::tensor_ops` (Metal 4).
   See `ROADMAP.md` "honest 'beat CUDA' picture" section.

2. **Metal 4 / `mpp::tensor_ops` requires SDK 26.0+** (Xcode shipped with
   macOS 26). CMake detects this and only compiles those sources when present.
   On older SDKs the Metal 4 sources are silently excluded.

3. **The M5 Neural Accelerator is the only path to >25 TFLOPS** on Apple
   Silicon. It's reachable through `mpp::tensor_ops::matmul2d` (NOT
   `MTL4MachineLearningCommandEncoder` — that's for pre-compiled CoreML).
   On M5+ runtime, our dispatch automatically prefers the tensor_ops path.

4. **fp32 GEMM is bit-exact against Accelerate.** If you change the kernel
   and `test_gemm_f32` shows nonzero error, you broke something subtle.

5. **fp16 with fp32 accumulators is the standard precision for AI training.**
   Our `simdgroup_matrix` GEMM uses this internally; the same applies to
   FlashAttention.

6. **Threadgroup memory budget on M-series is 32 KB.** Don't exceed it.
   This is why FlashAttention D=128 uses Br=Bc=16 — the larger tile blows the
   budget. v0.2 will use aliased regions for Br=64.

## Adding work

- New kernel? See `CONTRIBUTING.md` "Adding a kernel".
- New backend? Same doc, "Adding a backend target".
- Just trying things? `examples/hello_gemm.c` is your starting point.

## When you have an M5

Just rebuild. The Metal 4 dispatch lights up automatically. Then run the bench
suite and watch the TFLOPS jump from ~17 to ~80-110.

## Open work (v0.2+)

`ROADMAP.md` is authoritative. The short list:

- 20+ TFLOPS fp16 on M2 Ultra via `simdgroup_async_copy` (MFA patterns)
- FlashAttention backward at D=128
- Real Thunderbolt-5 ring + JACCL ZeRO-2/3 sharding for multi-Mac
- M5 + macOS 26 SDK validation (`mpp::tensor_ops` actual perf measurement)
- Eshkol bridge integration into `eshkol-platform` (instructions in
  `eshkol/bridge/INTEGRATION.md`)

Anything else you find, check `ROADMAP.md` — if it's not listed, file an issue.
