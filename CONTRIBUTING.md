# Contributing to tensorcore

## Setup

```sh
git clone <repo> && cd tensorcore
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

You need Apple Silicon + Xcode 16.0+. SDK 26.0+ unlocks the Metal 4 path.

## Code layout

```
include/tensorcore/    # Public C ABI (stable across versions)
lib/
  core/                # Device init, pipeline cache, buffer pool, autotune
  ops/                 # gemm.mm, attention.mm, training.mm, conv.mm
  fallback/            # MPS + Accelerate paths
  tensorops/           # Metal 4 / M5 (SDK-gated)
  distributed/         # Multi-Mac collectives
  c_api/               # ABI shims
kernels/metal/         # .metal sources -> default.metallib
cmake/                 # compile_metallib.cmake
tests/                 # Correctness suite
bench/                 # TFLOPS bench
examples/              # Minimal usage examples
eshkol/                # .esk bindings + bridge file
docs/                  # Architecture, ROADMAP, integration guides
```

## Adding a kernel

1. Write the `.metal` kernel in `kernels/metal/`. Use `simdgroup_matrix` for
   matmul-shaped work, threadgroup-memory reductions for row-wise ops.
2. Add the file to `TC_METAL_SOURCES` in `CMakeLists.txt`.
3. Declare the public ABI in `include/tensorcore/<group>.h`.
4. Implement host dispatch in `lib/ops/<group>.mm` (encode buffers, dispatch
   threadgroups, sync). Use `tc_pipeline_get(ctx, @"kernel_name", &err)`.
5. Add a correctness test in `tests/test_<group>.c` comparing against an
   fp64 CPU reference. Use `rms_scaled` metric (not per-cell rel-err which
   blows up near zero).
6. Register the test in `tests/CMakeLists.txt`.

## Adding a backend target

The `tc_last_backend()` enum tracks which path served each call. To add a new
backend:

1. Add a `TC_BACKEND_*` value in `include/tensorcore/gemm.h`.
2. Update `tc_backend_name` in `lib/core/device.mm`.
3. Set `tc_set_last_backend(...)` from your dispatch.

## Style

- C11 for C, C++17 for C++/ObjC++. No exceptions in public API.
- Arc-managed ObjC (`-fobjc-arc`) — no manual `release`.
- No comments stating what code does. Only why, when it's not obvious.
- Public headers: opaque struct types, status-code returns, no globals.
- Tests must be deterministic — seed `rand()` per case.

## Submitting

PRs welcome. Include:
- Correctness test for any new kernel
- Bench numbers if it touches perf-critical code
- ROADMAP.md update if it closes a v0.x item

## Code of conduct

Be precise, be kind, be brief.
