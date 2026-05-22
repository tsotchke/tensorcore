# Apple GPU family gating

`tensorcore` ships **one binary** that runs on every M-series chip from
M1 through M5 (and the A17 Pro / A18 series, by virtue of being Apple9).
The way it does this is by detecting the GPU family at runtime, picking
the best kernel path for that family, and falling back gracefully when a
path isn't available.

This page explains how the detection works, what each family unlocks, and
how to diagnose a misfire.

## The family enum

```c
typedef enum {
    TC_FAMILY_UNKNOWN = 0,
    TC_FAMILY_APPLE7  = 7,    /* M1                                   */
    TC_FAMILY_APPLE8  = 8,    /* M2                                   */
    TC_FAMILY_APPLE9  = 9,    /* M3, A17 Pro     — adds bf16 MMA      */
    TC_FAMILY_APPLE10 = 10,   /* M4              — adds int8 MMA      */
    TC_FAMILY_APPLE11 = 11,   /* M5              — adds TensorOps M5  */
} tc_family_t;
```

The numbers match Apple's `MTLGPUFamilyApple{N}` constants exactly. `0`
means "not classified" — should never happen in practice; if you see it,
the device is non-Apple silicon and many ops will fail.

## What each family unlocks

| Family | Chips | Native MMA dtypes | TensorOps M5 | Notes |
|---|---|---|---|---|
| Apple7 | M1 | fp16, fp32 | — | bf16 / int8 via fallback |
| Apple8 | M2 | fp16, fp32 | — | bf16 / int8 via fallback |
| Apple9 | M3, A17 Pro | fp16, bf16, fp32 | — | int8 via fallback |
| Apple10 | M4 | fp16, bf16, fp32, int8 | — | full simdgroup_matrix coverage |
| Apple11 | M5 | fp16, bf16, fp32, int8 | ✓ (SDK 26.0+ + M5 runtime) | `mpp::tensor_ops::matmul2d` adds the fast small-shape path |

"Native MMA" means `simdgroup_matrix` instructions of that dtype are
available on the silicon. When a dtype is not native, the dispatch routes
through a software fallback (bf16 → fp32 cast, int8 → fp32 widen) that
produces the right answer at lower throughput.

## How detection works

`lib/core/device.mm` runs the following at `tc_init` time:

1. Get the default `MTLDevice` (`MTLCreateSystemDefaultDevice()`).
2. Find the highest family the device supports:
   ```objc
   for (NSInteger f = MTLGPUFamilyApple11; f >= MTLGPUFamilyApple7; --f) {
       if ([device supportsFamily:(MTLGPUFamily)f]) {
           info.family = (tc_family_t)f;
           break;
       }
   }
   ```
3. Set capability flags:
   - `supports_bf16_simdgroup = family >= Apple9`
   - `supports_i8_simdgroup   = family >= Apple10`
   - `supports_tensorops_m5   = family >= Apple11 && SDK 26.0+ && TC_ENABLE_TENSOROPS=ON`
4. Cache `device.name`, `max_buffer_bytes`, `recommended_working_set_bytes`,
   `max_threadgroup_memory`, `max_threads_per_threadgroup`, and
   `unified_memory` (always `true` on M-series).

These are exposed via `tc_device_info_get` and read by `lib/ops/*.mm` at
dispatch time.

## Dispatch decisions

### GEMM (`lib/ops/gemm.mm`)

```
input dtype × family → kernel path
─────────────────────────────────────────────────────
F16 / F32         × any                      → simdgroup_matrix native
BF16              × Apple9+                  → simdgroup_matrix native
BF16              × Apple7..8                → MPS bf16 (or fp32 cast fallback)
I8                × Apple10+                 → simdgroup_matrix int
I8                × Apple7..9                → fp32 widen fallback
F32 (very small)  × any                      → MPS (latency wins)
unsupported shape × any                      → Accelerate cblas_sgemm
F32 (TC_ENABLE_TENSOROPS=1 + Apple11 + SDK 26+) → tensorops_m5
```

`tc_last_backend()` reports which row matched.

### Attention (`lib/ops/attention.mm`)

Coverage is uniform — `simdgroup_matrix` fp16/bf16 with fp32 accum on every
family Apple7+. The kernel doesn't fall back to MPS or CPU; an
unsupported shape (e.g. `head_dim > 128`) returns `TC_ERR_INVALID_SHAPE`.

The D=128 forward + backward kernels use `Br = Bc = 16` on Apple7..8 (to
stay under the 32 KB threadgroup memory limit) and could be `Br = 64` on
Apple9+ — that's a v0.2 task documented in [attention.md](attention.md).

### Training kernels (`lib/ops/training.mm`)

All elementwise / row-reduction kernels; no family gating beyond fp16
availability (Apple7+).

### Conv2D (`lib/ops/conv.mm`)

Same as GEMM — family gating is inherited via the call into `tc_gemm`.

### Quantized (`lib/ops/quantized.mm`)

The Q4_0 / Q8_0 kernels are pure MSL with `half` activations + `fp16`
scales + nibble unpacking. They work on Apple7+ unconditionally.

The newer `gemm_quantized_v2.metal` is the default Q4_0 path; the
original kernel is reachable via `TC_Q4_USE_V1=1` for comparison.

## How to override / inspect

### Inspect at runtime

```c
tc_device_info info;
tc_device_info_get(ctx, &info);
printf("family=Apple%d  bf16=%d  i8=%d  tensorops=%d\n",
       (int)info.family,
       info.supports_bf16_simdgroup,
       info.supports_i8_simdgroup,
       info.supports_tensorops_m5);
```

### Force a specific backend

You can't force, but you can disable. To compare paths:

- Build without TensorOps: `cmake -DTC_ENABLE_TENSOROPS=OFF` (default is `ON`, but it only has an effect with SDK 26.0+ and Apple11/M5+ runtime support).
  M5 then falls back to simdgroup_matrix.
- Force MPS: deliberately call a shape outside the kernel's tile coverage
  (e.g. `M % 64 != 0`). This is brittle; we'll add an `TC_FORCE_BACKEND`
  env in v0.2.

### See which path served the last call

```c
tc_status_t s = tc_gemm(ctx, &d, A, B, C);
printf("backend = %s\n", tc_backend_name(tc_last_backend()));
```

Possible answers:

```
simdgroup_matrix      ← best path, you're on the fast lane (TC_ENABLE_METAL=ON)
tensorops_m5          ← M5 + SDK 26+ + tensorops enabled
mps                   ← MetalPerformanceShaders (slower; covers exotic shapes)
accelerate_cpu        ← cblas_sgemm; correct but CPU-bound
sf64_emulated         ← SoftFloat-64; exact but slow
ozaki_ii              ← CRT-based exact GEMM (research path)
portable_cpu          ← pure-C CPU backend (TC_ENABLE_METAL=OFF, non-Apple builds)
none                  ← no call made yet on this thread, or pre-tc_init
```

If you expected `simdgroup_matrix` and got something else, see
[troubleshooting.md](troubleshooting.md).

## Non-Apple platforms — portable CPU backend

When CMake is invoked with `TC_ENABLE_METAL=OFF` (the default on
non-Apple platforms), tensorcore builds only the **portable CPU
backend**. The Metal kernels, MPS fallback, and TensorOps path are all
excluded; the dispatch ladder collapses to one entry:

```
portable_cpu          ← pure-C reference, builds on Linux / Intel-Mac /
                        anywhere with a C++17 toolchain
```

The portable CPU backend is **not** a perf path — it's a *correctness*
path that keeps the C ABI usable on non-Apple mesh workers. Its
intended uses:

- CPU nodes in a hybrid Apple-GPU + Linux-CPU mesh
- Linux CI runners (for things that don't require Metal)
- Cross-compilation sanity checks
- Tests that should run anywhere

GEMM, attention, conv, training kernels, quantized GEMV, and the
distributed collectives all have CPU implementations in `lib/core/device_cpu.cpp`
and `lib/ops/*_cpu.cpp`. The numerical accuracy matches the GPU path's
fp64-reference tolerance; raw throughput is much lower (no
`simdgroup_matrix`, no SIMD).

On macOS, you can also force this path explicitly:

```sh
cmake -B build_cpu -DTC_ENABLE_METAL=OFF
cmake --build build_cpu -j
ctest --test-dir build_cpu
```

The build produces the same `libtensorcore.dylib` / `.a` / headers + a
trivial `metallib` placeholder so consumer code doesn't need to special-
case the absence of Metal.

`tc_last_backend()` will report `portable_cpu` for every dispatch on
this build.

## SDK gating vs family gating

These are two different gates and the difference matters:

| Gate | Set at | What it controls |
|---|---|---|
| **SDK gate** (build time) | `xcrun --show-sdk-version >= 26.0` | Whether `tensorops_*.metal` and the `lib/tensorops/tensorops_m5.mm` host code are compiled in. With an older SDK, the M5 path is *absent* from the binary entirely. |
| **Family gate** (runtime) | `tc_device_info.family` + `supports_tensorops_m5` | Whether the M5 path is *taken* on the current hardware. With an Apple9 chip running a binary built with SDK 26+, the M5 path exists but is never dispatched. |

Practical implication: a single binary built on a macOS 26+ host
*everywhere unlocks the M5 path on M5 silicon and runs unchanged on
M1-M4*. Conversely, a binary built on an older host can't reach the M5 path
even on M5 hardware. Rebuild on macOS 26+ before deploying to M5.

A second SDK gate concerns the `gemm_async*.metal` kernels: they use private
AIR `__asm` intrinsics that Xcode 16 accepts but Xcode 17+ rejects. The
build flips the include list based on SDK version — older SDKs include the
async kernels, newer SDKs skip them (and the dispatch silently falls back
to the sync path).

## Tested matrices

The test suite is family-aware. Each kernel test skips itself politely if
the chip doesn't support the path being tested:

| Test | Skip condition |
|---|---|
| `test_gemm_bf16` | Apple < 9 → skipped pre-v0.1.3; now runs (validates the fp32 fallback) |
| `test_gemm_i8` | Apple < 10 → skipped pre-v0.1.3; now runs (validates the fp32 widen fallback) |
| `test_attention_correctness` | always runs (fp16) |
| `test_attention_backward` | always runs (fp16); validates D=64 and D=128 |
| `test_quantized` | always runs |
| `test_fused_norm_gemv` | always runs |
| `test_distributed_ring_fork` | always runs (uses fork + socketpair) |
| `test_diloco` | always runs (local/single-rank DiLoCo path) |
| `test_sparse_compress` | always runs (host-side sparse pack/unpack) |
| `test_tensorops_runtime` | skips politely until Metal 4 TensorOps is available |

The default Apple suite is **24 tests** at this checkpoint: 22
correctness/Python tests plus the two native example smokes. It takes
~5-15 seconds on a local Apple workstation.

## What family you're running on, the lazy way

```sh
cd build && ./examples/hello_gemm 2>&1 | head -5
```

The first line of `hello_gemm` prints the backend that served the GEMM —
which transitively tells you whether the runtime classified your device
correctly. If you see `simdgroup_matrix`, you're on the fast path. If you
see `accelerate_cpu`, the runtime failed to bind the Metal device — see
[troubleshooting.md](troubleshooting.md).
