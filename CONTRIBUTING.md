# Contributing to tensorcore

`tensorcore` is a small, opinionated, single-purpose library: **CUDA for
Apple Silicon, with measurable correctness and graceful family-aware
fallbacks**. This page tells you how to add to it without breaking that.

## Setup

```sh
git clone <repo> && cd tensorcore
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

You need:

- Apple Silicon (arm64). The build will succeed on x86_64 macOS but
  `simdgroup_matrix` kernels won't run.
- Xcode 16.0+ command-line tools (`xcode-select --install`).
- SDK 26.0+ unlocks the Metal 4 / `mpp::tensor_ops` path. Older SDKs
  build clean without it.

Optional:

- Python 3.10+ and NumPy if you want the `python_basic` CTest target.

## Code layout

```
include/tensorcore/    # Public C ABI (stable across versions)
lib/
  core/                # Device init, pipeline cache, buffer pool, autotune
  ops/                 # gemm.mm, attention.mm, training.mm, conv.mm, quantized.mm
  fallback/            # MPS + Accelerate paths
  tensorops/           # Metal 4 / M5 (SDK-gated)
  distributed/         # Multi-Mac collectives
  io/                  # GGUF v3 reader
  c_api/               # ABI shims
kernels/metal/         # .metal sources -> default.metallib
cmake/                 # compile_metallib.cmake, package config templates
tests/                 # 20 correctness tests
bench/                 # TFLOPS / tok/s bench
examples/              # hello_gemm, gguf_inspect
eshkol/                # .esk bindings + FFI bridge for the Eshkol toolchain
python/                # ctypes Python binding (full ABI surface)
docs/                  # Architecture, API reference, ROADMAP, integration guides
```

For the file-by-file breakdown, see
[docs/architecture.md](docs/architecture.md).

## Adding a kernel

1. **Write the `.metal` kernel in `kernels/metal/`.** Use
   `simdgroup_matrix` for matmul-shaped work; threadgroup-memory
   reductions for row-wise ops. Use function constants (not preprocessor
   macros) for compile-time switches like causal / transpose / dtype.

2. **Add the file to `TC_METAL_SOURCES`** in `CMakeLists.txt`. If the
   kernel uses Metal 4 features (`mpp::tensor_ops`), gate it behind
   `if(TC_HAVE_METAL4)`. If it uses Apple-private `__asm` AIR intrinsics
   (the `gemm_async*` kernels do), gate it behind
   `if(TC_SDK_VERSION VERSION_LESS "26.0")` — Xcode 17+ rejects those.

3. **Declare the public ABI** in `include/tensorcore/<group>.h`. Use
   opaque struct types, status-code returns, no globals. Match the
   existing descriptor patterns:
   - sizes / dtypes in the descriptor struct, buffers as separate args
   - default values for unset fields encoded as `0` (e.g.
     `kv_heads=0` means "no GQA")
   - new fields go at the **end** of existing descriptor structs to
     preserve the ABI

4. **Implement host dispatch** in `lib/ops/<group>.mm`. The pattern:

   ```objc++
   tc_status_t tc_my_op(tc_context* ctx, ...) {
       /* 1. Validate inputs */
       /* 2. Pick a kernel (consider family / dtype / shape) */
       NSError* err = nil;
       id<MTLComputePipelineState> pso =
           tc_pipeline_get(ctx, @"my_kernel_name", &err);
       /* 3. Get an MTLCommandBuffer (from stream, or default) */
       /* 4. Encode buffers, dispatch threadgroups */
       /* 5. Commit + wait (sync), or hand to stream (async) */
       tc_set_last_backend(TC_BACKEND_SIMDGROUP_MATRIX);
       return TC_OK;
   }
   ```

   Function constants get set via
   `MTLFunctionConstantValues` before `tc_pipeline_get`. Look at
   `lib/ops/attention.mm` for the canonical example with multiple
   constants (causal / use_lse / use_window / use_alibi).

5. **Add a correctness test** in `tests/test_<group>.c`. Compare against
   an fp64 CPU reference and use the `rms_scaled` metric:

   ```
   rms_scaled = ||y - yref|| / (||yref|| + eps)
   ```

   Per-cell relative error blows up near zero; use it only for cases
   where you expect bit-exact agreement (fp32 GEMM vs Accelerate, int8
   accumulation).

6. **Register the test** in `tests/CMakeLists.txt`.

7. **(Optional) Add a bench**. The harness pattern in `bench/bench_gemm.c`
   is the template: warmup, 10 timed iterations, report median TFLOPS
   plus `tc_last_backend()`.

8. **Update [CHANGELOG.md](CHANGELOG.md)** and any relevant doc in `docs/`.

## Adding a backend target

`tc_last_backend()` reports which dispatch path served each call. To add
a new path:

1. Add a `TC_BACKEND_*` value in `include/tensorcore/gemm.h` (after the
   existing entries; do not renumber).
2. Update `tc_backend_name` in `lib/core/device.mm` to render the new
   name.
3. Call `tc_set_last_backend(TC_BACKEND_MY_NEW_PATH)` from your dispatch
   site immediately before commit / handoff to a stream.

Today the diagnostic is updated from GEMM, attention, and TensorOps
sites only — see [docs/codebase_audit.md](docs/codebase_audit.md). v0.2
widens it to every op.

## Style

- **C11 for C, C++17 for C++/ObjC++.** No exceptions in the public API.
- **Arc-managed ObjC** (`-fobjc-arc`) — no manual `release`. Compile
  flags in `CMakeLists.txt` are non-negotiable for ObjC++ sources.
- **Comments stay rare.** No comments stating what code does — write
  comments only when the *why* isn't obvious from a careful read.
- **Public headers are opaque.** Opaque struct types, status-code
  returns, no globals. Descriptors are passed by `const` pointer.
- **Tests must be deterministic.** Seed `rand()` per case; never depend
  on wall-clock state. Set tolerances based on theoretical accumulation
  noise, not empirical fudge.
- **No copyright headers in source files.** The LICENSE at the root
  covers everything; per-file headers are noise.

## Numerical guarantees we commit to

| dtype | Path | Guarantee |
|---|---|---|
| fp32 | `tc_gemm` | bit-exact against `cblas_sgemm` |
| fp16 | `tc_gemm` | rms_scaled ≤ 5e-3 vs fp64 reference at 4096³ |
| bf16 | `tc_gemm` (Apple9+ native or Apple7..8 fallback) | rms_scaled ≤ 3e-3 |
| int8 | `tc_gemm` (Apple10+ native or Apple7..9 fallback) | bit-exact i32 accumulation up to K = 2^16 |
| fp16 | `tc_attention_forward` | rms_scaled ≤ 1e-3 vs fp64 reference |
| fp16 | `tc_attention_backward` | rms_scaled ≤ 3e-3 (D=64) |
| fp16 | Q4_0 / Q8_0 GEMV | rms_scaled ≤ 2e-4 vs dequantized reference |
| fp16 | normalizations / RoPE / SwiGLU / softmax | rms_scaled ≤ 5e-3 vs fp64 reference |

Don't merge a kernel that regresses any of these. The tests in `tests/`
catch most of them; the bench harness catches the rest.

## Submitting

PRs welcome. Include:

- A correctness test for any new kernel.
- Bench numbers if you touched perf-critical code, with the chip and the
  shape called out. `bench/bench_gemm.c` style is fine.
- A `ROADMAP.md` update if the PR closes a v0.x item.
- A `CHANGELOG.md` entry under the active checkpoint section.

Coding conventions are enforced by `-Wall -Wextra -Wpedantic` plus
`-fobjc-arc`. The build is warning-clean today; keep it that way.

## CI

There are two CI workflows + one self-hosted hardware workflow:

- `.github/workflows/ci.yml` — build, test, install smoke, Python smoke
  on macos-14 and macos-15 runners. Triggers on push / PR to
  main/master. **This is the gate.**
- `.github/workflows/release.yml` — build wheel + GitHub release on
  `v*` tag push. The wheel is the `tensorcore_apple-*.whl` published on
  the release.
- `.github/workflows/hardware-evidence.yml` — manual workflow that runs
  on a self-hosted M-series runner with `REQUIRE_GPU=1` and (optionally)
  `REQUIRE_METAL4_TENSOROPS=1`. Exercises the real hardware path.

To run the CI Python smoke locally:

```sh
cmake --install build --prefix /tmp/tensorcore-install
bash scripts/ci_python_smoke.sh
```

Without the install, the script's default `PREFIX=/tmp/tensorcore-install`
won't have the dylib.

## Ground-truthing changes against the codebase

If a PR moves a lot of code, re-run the
[ICC](https://github.com/tsotchke/infinite_context_coder) deterministic
audit and update [docs/codebase_audit.md](docs/codebase_audit.md):

```sh
ICC_HOME=~/Desktop/infinite_context_coder
$ICC_HOME/bin/icc register --name tensorcore --path ~/Desktop/tensorcore \
    --skip-dir build --skip-dir CMakeFiles --skip-dir .cache --skip-dir .claude
$ICC_HOME/bin/icc index --repo tensorcore
$ICC_HOME/bin/icc build-memory --repo tensorcore
$ICC_HOME/bin/icc architecture-summary --repo tensorcore --bundle --include-cheatsheet
$ICC_HOME/bin/icc trace-callees --repo tensorcore --symbol tc_gemm --depth 2
$ICC_HOME/bin/icc find-dead-code --repo tensorcore --limit 20
```

This is how the doc tree was ground-truthed; do it again whenever
something significant moves.

## Code of conduct

Be precise, be kind, be brief.
