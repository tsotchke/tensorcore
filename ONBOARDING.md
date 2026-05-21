# Onboarding to tensorcore

You're picking up `tensorcore` cold. This page is the working set.

## What this is, in one paragraph

`tensorcore` is **CUDA for Apple Silicon**. A C-ABI kernel library that
wraps Metal's `simdgroup_matrix` (M1+) and `mpp::tensor_ops` (M5+) into a
single training-grade foundation: ~17 TFLOPS fp16 GEMM on M2 Ultra,
fused FlashAttention forward + backward, full transformer training kernel
set, Q4_0 / Q8_0 GEMV with a GGUF v3 reader, single-host distributed
primitives, MPS + Accelerate fallbacks, and a Python binding. One binary,
every M-series chip, runtime family detection.

`include/tensorcore/tensorcore.h` is the umbrella header — that's the
surface area for external consumers.

## 30-second tour

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure   # 22/22 pass
./build/bench/bench_gemm                       # TFLOPS sweep
./build/examples/hello_gemm                    # Minimal use
./build/examples/gguf_inspect llama.gguf       # Inspect a real GGUF
```

If `ctest` shows 22/22 pass, your environment is healthy. If `bench_gemm`
prints `backend=simdgroup_matrix`, you're on the fast lane.

## Reading order

1. **[README.md](README.md)** — the "CUDA for Apple" thesis, what v0.1
   ships, where tensorcore sits in the ecosystem.
2. **[docs/cuda_comparison.md](docs/cuda_comparison.md)** — if you came
   from CUDA-land, start here.
3. **[docs/architecture.md](docs/architecture.md)** — how the library is
   put together internally: device init, pipeline cache, buffer pool,
   op dispatch, fallback ladder.
4. **[docs/api_reference.md](docs/api_reference.md)** — every public C
   symbol, grouped by header.
5. **[ROADMAP.md](ROADMAP.md)** — what's next, what's silicon-bound, what's
   software-bound.

## Where the value lives

| Concern | File(s) |
|---|---|
| The kernels | `kernels/metal/*.metal` |
| Public ABI | `include/tensorcore/*.h` |
| Device init / pipeline cache / buffer pool | `lib/core/{device,pipeline_cache,buffer_pool}.mm` |
| Op dispatch (host) | `lib/ops/{gemm,attention,training,conv,quantized}.mm` |
| GGUF reader | `lib/io/gguf.c` |
| M5 / Metal 4 TensorOps | `lib/tensorops/tensorops_m5.mm` (SDK-gated) |
| MPS + Accelerate fallback | `lib/fallback/{mps_gemm.mm,accelerate_gemm.c}` |
| Distributed primitives | `lib/distributed/{distributed,ring_local}.mm` |
| Eshkol FFI bridge | `eshkol/bridge/tensorcore_codegen.cpp` |
| Python ctypes binding | `python/tensorcore/__init__.py` |
| Build wiring | `CMakeLists.txt`, `cmake/compile_metallib.cmake` |
| The plan | `ROADMAP.md` |
| Current state | `CHANGELOG.md` |

## Constraints to know

1. **Apple GPU AGX ISA is undocumented.** We don't write assembly. The
   hottest abstraction we use is MSL's `simdgroup_matrix` + `mpp::tensor_ops`
   (Metal 4). See [docs/cuda_comparison.md](docs/cuda_comparison.md) and
   [ROADMAP.md](ROADMAP.md) on the honest competitive picture.

2. **Metal 4 / `mpp::tensor_ops` requires SDK 26.0+** (Xcode shipped with
   macOS 26). CMake detects this and only compiles those sources when
   present. On older SDKs the Metal 4 sources are silently excluded and
   the runtime falls back to `simdgroup_matrix`. See
   [docs/family_gating.md](docs/family_gating.md).

3. **The M5 Neural Accelerator is the only path to >25 TFLOPS** on Apple
   Silicon. It's reachable through `mpp::tensor_ops::matmul2d` (NOT
   `MTL4MachineLearningCommandEncoder` — that's for pre-compiled CoreML).
   On M5+ runtime, our dispatch prefers the tensor_ops path.

4. **fp32 GEMM is bit-exact against Accelerate.** If you change the kernel
   and `test_gemm_f32` shows nonzero error, you broke something subtle.

5. **fp16 with fp32 accumulators is the standard precision for AI
   training.** Our `simdgroup_matrix` GEMM uses this internally; the same
   applies to FlashAttention. See [docs/dtypes.md](docs/dtypes.md) for the
   accumulation policy table.

6. **Threadgroup memory budget on M-series is 32 KB.** Don't exceed it.
   This is why FlashAttention D=128 uses Br=Bc=16 — the larger tile blows
   the budget. v0.2 will use aliased regions for Br=64.

7. **`tc_last_backend()` is a per-thread diagnostic.** It's updated by
   GEMM, attention, and TensorOps dispatch sites; training/conv/quantized
   ops don't currently touch it. Read it immediately after a GEMM or
   attention call to know which path served it.

## Adding work

- **New kernel?** See [CONTRIBUTING.md](CONTRIBUTING.md) "Adding a kernel".
- **New backend target?** See [CONTRIBUTING.md](CONTRIBUTING.md) "Adding a
  backend target".
- **Doc update?** Edit in `docs/`, mirror the audited surface — see
  [docs/codebase_audit.md](docs/codebase_audit.md) for how ICC verifies
  the docs against the code.
- **Just trying things?** `examples/hello_gemm.c` is your starting point.
- **Integrating into an external project?** Start at
  [docs/integrating_tensorcore.md](docs/integrating_tensorcore.md).

## When you have an M5

Just rebuild on macOS 26.0+. The Metal 4 dispatch lights up
automatically once `tc_device_info.supports_tensorops_m5 == true`. Then
run the bench suite and watch the TFLOPS jump from ~17 to ~80-110 (this
is the v0.3 expectation; we'll know the real number when M5 hardware
lands in the lab).

## Open work (v0.2+)

[ROADMAP.md](ROADMAP.md) is authoritative. Short list:

- 20+ TFLOPS fp16 on M2 Ultra via `simdgroup_async_copy` (MFA patterns) +
  128×128 tile retune.
- FlashAttention backward at D=128.
- Real Thunderbolt-5 ring + JACCL ZeRO-2/3 sharding for multi-Mac
  (v0.5, depends on macOS 26.2 substrate).
- M5 + macOS 26 SDK validation (`mpp::tensor_ops` actual perf measurement).
- Eshkol-platform consolidation (v0.4) — collapse three Metal backends
  into one tensorcore-based path.

Anything else you find, check [ROADMAP.md](ROADMAP.md) — if it's not
listed, file an issue.

## Validate your environment

If you only have time for three checks:

```sh
ctest --test-dir build --output-on-failure                     # correctness
./build/bench/bench_gemm | grep "4096" | head -1               # perf
./build/examples/hello_gemm 2>&1 | grep backend                # dispatch path
```

If those three pass, the rest of the library is plug-and-play.

## Running the CI smoke locally

The CI script `scripts/ci_python_smoke.sh` defaults `PREFIX` to
`${RUNNER_TEMP:-/tmp}/tensorcore-install`. Locally that's `/tmp/tensorcore-install`,
which won't exist until you install:

```sh
cmake --install build --prefix /tmp/tensorcore-install
bash scripts/ci_python_smoke.sh
```

The script writes a venv at `/tmp/tensorcore-venv`, installs the
package editable, and asserts that `tc.status_string(tc.TC_OK) == "ok"`,
`tc.dtype_name("fp53") == "fp53"`, `tc.backend_name(TC_BACKEND_TENSOROPS_M5)
== "tensorops_m5"`, and the tensorops kernel selector returns the
expected names.

If your changes broke any of those, you'll see a `SystemExit("…
mismatch")` line near the end of the output.
