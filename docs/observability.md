# Observability

What tensorcore exposes at runtime so you can answer "what's it doing
right now?" without `printf`-debugging into a Metal kernel.

## What you can ask

| Question | How |
|---|---|
| What chip am I on? | `tc_device_info_get(ctx, &info)` → `info.name`, `info.family`, capability flags |
| Which dispatch path served my last call? | `tc_backend_name(tc_last_backend())` (only for GEMM / attention / tensorops; see scope note below) |
| Which tile shape is autotune using? | `~/.cache/tensorcore/autotune.json` (or the equivalent on the host) |
| Did the M5 TensorOps path actually engage? | `tc_device_info.supports_tensorops_m5` + `tc_last_backend() == TC_BACKEND_TENSOROPS_M5` |
| Did a release pass on this hardware? | `build/release_smoke_runtime_evidence.json` after `scripts/release_smoke.sh` |
| Is my build using the M5 SDK path? | CMake log `tensorcore: SDK X.Y -- ENABLING Metal 4 / mpp::tensor_ops path` |

## Device-info introspection

The cheapest, most useful diagnostic. Get a one-shot report of what
tensorcore sees:

```c
tc_device_info info;
tc_device_info_get(ctx, &info);
printf("name=%s family=Apple%d unified=%d bf16=%d i8=%d tensorops=%d\n",
       info.name, (int)info.family,
       info.unified_memory,
       info.supports_bf16_simdgroup,
       info.supports_i8_simdgroup,
       info.supports_tensorops_m5);
printf("max_buffer=%llu working_set=%llu tg_mem=%u sg_width=%u\n",
       (unsigned long long)info.max_buffer_bytes,
       (unsigned long long)info.recommended_working_set_bytes,
       info.max_threadgroup_memory,
       info.thread_execution_width);
```

This is what `hello_gemm` prints on startup. It's the diagnostic you
quote in bug reports.

## Backend tracing

`tc_last_backend()` reports which dispatch path served the most recent
**GEMM or attention** call on this thread. The return value is the
`tc_backend_t` enum:

```c
typedef enum {
    TC_BACKEND_NONE             = 0,
    TC_BACKEND_SIMDGROUP_MATRIX = 1,
    TC_BACKEND_TENSOROPS_M5     = 2,
    TC_BACKEND_MPS              = 3,
    TC_BACKEND_ACCELERATE_CPU   = 4,
    TC_BACKEND_SF64_EMULATED    = 5,
    TC_BACKEND_OZAKI_II         = 6,
} tc_backend_t;

const char* tc_backend_name(tc_backend_t b);  /* "simdgroup_matrix", ... */
```

**Scope:** `tc_set_last_backend` is currently called only from GEMM
(`lib/ops/gemm.mm`), attention (`lib/ops/attention.mm`), and TensorOps
(`lib/tensorops/tensorops_m5.mm`). Training / conv / quantized kernels
don't currently update it; the diagnostic shows the last GEMM-or-
attention path. Widening to every dispatch is a v0.2 polish item.

Typical use: instrument bench harnesses to print the backend per call:

```c
tc_status_t s = tc_gemm(ctx, &d, A, B, C);
if (s == TC_OK) {
    printf("backend=%-20s %.2f TFLOPS\n",
           tc_backend_name(tc_last_backend()),
           tflops);
}
```

## The autotune cache

`lib/core/autotune.cpp` runs a one-time sweep at first init that picks
GEMM and attention tile shapes per family. The result is cached on disk
so subsequent inits skip the sweep:

```
[tensorcore] autotune: running sweep (one-time)
[tensorcore] autotune: cached → {"version":1,
                                  "gemm":{"BM":64,"BN":64,"BK":32,
                                          "WM":2,"WN":2,"TM":4,"TN":4,
                                          "threads":128},
                                  "attention_d64":{"Br":32,"Bc":32,
                                                    "WM":2,"WN":2,
                                                    "threads":128},
                                  "attention_d128":{"Br":16,"Bc":16,
                                                     "WM":2,"WN":2,
                                                     "threads":128}}
```

The JSON above is exactly what the autotune writes to its cache file on
M2 Ultra. You'll see the same line on subsequent inits replaced by
`[tensorcore] autotune: loaded cached config`.

Cache location: per-user, OS-cache-dir style. To force a fresh sweep
(e.g. after a kernel update), delete the cache. To inspect what the
autotune chose without restarting, dump the JSON:

```sh
cat "$HOME/Library/Caches/tensorcore/autotune.json"   # macOS default
```

## Build-time gates

CMake prints which paths the binary will support:

```
-- tensorcore: SDK 26.0 -- ENABLING Metal 4 / mpp::tensor_ops path
-- tensorcore: SDK 26.0 -- DISABLING async_copy GEMM kernels
   (Xcode 17+ rejects __asm intrinsics)
-- tensorcore: configured (Metal4=ON, tests=ON, bench=ON)
--   metallib -> /Users/.../build/tensorcore.metallib
```

These are the static facts about your binary. The runtime gates
(`info.supports_tensorops_m5`, etc.) are the dynamic facts about your
chip.

## Release smoke evidence

`scripts/release_smoke.sh` emits a structured JSON record of what
happened during the smoke run. The file:

```
build/release_smoke_runtime_evidence.json
```

Contents (example shape):

```json
{
  "tensorcore_version": "0.1.22",
  "host": {
    "device_name": "Apple M2 Ultra",
    "family": 8,
    "unified_memory": true,
    "supports_bf16_simdgroup": false,
    "supports_i8_simdgroup": false,
    "supports_tensorops_m5": false
  },
  "build": {
    "sdk_version": "26.0",
    "metal4": true,
    "async_copy_kernels": false
  },
  "checks": {
    "fp16_gemm": {"backend": "simdgroup_matrix", "rms_scaled": 1.2e-3},
    "attention_d64": {"backend": "simdgroup_matrix", "rms_scaled": 2.2e-4},
    ...
  }
}
```

In CI, this artifact is uploaded by the `Hardware Evidence` workflow as
proof that a particular commit ran correctly on a particular chip. For
downstream integrators it answers "did this version actually exercise the
M5 TensorOps path on M5 hardware before shipping?"

## Python introspection

The same data is reachable from Python via the binding:

```python
import tensorcore as tc
with tc.Context() as ctx:
    info = ctx.device_info()
    print(info.name, "Apple", info.family)
    print("bf16=", info.supports_bf16_simdgroup,
          "i8=", info.supports_i8_simdgroup,
          "tensorops_m5=", info.supports_tensorops_m5)

    # Run a GEMM
    A = ctx.buffer(256*256*2)
    B = ctx.buffer(256*256*2)
    C = ctx.buffer(256*256*2)
    ctx.gemm(A, B, C, M=256, N=256, K=256)

    print("backend:", ctx.last_backend_name())   # "simdgroup_matrix"
```

Plus standalone diagnostics:

```python
tc.status_string(tc.TC_OK)                       # "ok"
tc.dtype_name("fp53")                            # "fp53"
tc.backend_name(tc.TC_BACKEND_TENSOROPS_M5)      # "tensorops_m5"
tc.tensorops_gemm_kernel_name("f16")             # "tc4_gemm_f16"
tc.tensorops_gemm_kernel_name("i8", "i32")       # None (unsupported combo)
```

These are the same calls `scripts/ci_python_smoke.sh` asserts on every
push.

## Environment variables

Runtime overrides — all read once at dispatch time via `getenv`, no
global init step required:

| Variable | Effect |
|---|---|
| `TENSORCORE_LIB` | Override the `libtensorcore.dylib` location (Python binding only). |
| `TC_METALLIB` | Override the `tensorcore.metallib` search; useful when running from a non-standard install layout. |
| `TC_USE_128_TILE` | `=1` opts into the 128×128 GEMM tile (regresses on M2; v0.2 retunes). |
| `TC_Q4_USE_V1` | `=1` reverts to the original Q4_0 GEMV kernel for A/B comparison vs the v0.1.6+ default. |

Build / CMake knobs:

| Knob | Effect |
|---|---|
| `-DTC_BUILD_TESTS=ON/OFF` | Enable/disable the 20-test correctness suite. |
| `-DTC_BUILD_BENCH=ON/OFF` | Enable/disable the bench harness. |
| `-DTC_BUILD_EXAMPLES=ON/OFF` | Enable/disable `hello_gemm` and `gguf_inspect`. |
| `-DTC_ENABLE_METAL=ON/OFF` | ON by default on macOS; OFF on Linux/Windows. When OFF, only the **portable CPU backend** builds and the C ABI works without Metal. |
| `-DTC_ENABLE_TENSOROPS=ON/OFF` | Wire the M5 `mpp::tensor_ops` dispatch path. **Defaults to ON**; takes effect only when SDK ≥ 26.0 + Apple11/M5 + runtime supports the encoder. Set OFF to force the non-TensorOps Metal path for A/B comparison. |
| `-DCMAKE_BUILD_TYPE=Release/Debug` | Standard CMake. |

## Logging at startup

Every fresh `tc_init` prints two lines to stderr:

```
[tensorcore] loaded metallib: <path>
[tensorcore] device="Apple M2 Ultra" family=Apple8 unified=yes vram=147456MB bf16_sg=no i8_sg=no tensorops_m5=no
```

Plus, on first init only:

```
[tensorcore] autotune: running sweep (one-time)
[tensorcore] autotune: cached → {...}
```

These can't be suppressed at v0.1.x. The intent is to be obvious about
what happened — if the wrong chip is classified or the wrong metallib
is loaded, you see it. v0.2 adds a `TC_LOG_LEVEL` env knob.

## Completion oracles (`.icc/completion-oracles.yaml`)

The repo ships an ICC completion-oracle config that defines what "this
build is shippable" means in machine-checkable terms. Eight criteria:

| Criterion | What it asserts | Action on failure |
|---|---|---|
| `release_smoke_runtime_evidence` | `build/release_smoke_runtime_evidence.json` exists and parses | run `scripts/release_smoke.sh` |
| `release_smoke_failure_free` | the smoke ran without errors | fix the runtime failure |
| `public_core_paths_covered` | every public dispatch path's runtime status is recorded (`passed` or `skipped_no_gpu`) | extend `release_smoke.sh` coverage |
| `packaging_and_consumers_passed` | wheel + editable install + CMake `find_package` consumer + pkg-config consumer all build and run | fix `scripts/release_smoke.sh` packaging section |
| `no_synthetic_public_prod_paths` | no symbol under `include/`, `lib/`, or `python/tensorcore/` is marked `synthetic_model_only` (i.e. a stub used in tests only) | wire a real path or explicitly mark as a fallback |
| `metal4_tensorops_compile_status_recorded` | the M5 path's compile status is recorded (`compiled` or `skipped_sdk_too_old`) | emit the status from the smoke |
| `metal4_tensorops_compiled_with_sdk26` | the M5 path was actually compiled (requires SDK 26+ host) | run release smoke on a host with SDK 26+ |
| `metal4_tensorops_runtime_covered` | the M5 runtime path was exercised on M5 hardware | run the hardware-evidence workflow on a self-hosted M5 runner |

ICC checks these against the runtime evidence JSON; in CI they form the
"can we ship this commit?" contract that goes beyond green `ctest`.

The criteria are aliased to `cuda-for-apple` and
`tensorcore-public-integration` so external dashboards consuming ICC's
output can reference the same target by either name.

## What's missing (v0.2 +)

- **Per-call profile counters.** `MTLCounters` API access for kernel
  duration / occupancy / memory traffic. Useful for "this kernel is
  slow; why?" investigation.
- **Trace dumps.** A `TC_TRACE=1` env that prints every dispatch to
  stderr with timing.
- **Per-context backend statistics.** `tc_context_stats(ctx, &out)`
  returning {dispatches_by_backend, total_kernel_time_us, ...}.

These land in v0.2 alongside the kernel-coverage `tc_last_backend`
widening.

## See also

- [api_reference.md § Diagnostics](api_reference.md#gemm-gemmh) for the
  C ABI calls listed above.
- [troubleshooting.md](troubleshooting.md) for what to do when the
  numbers above look wrong.
- [ci_and_scripts.md § release_smoke.sh](ci_and_scripts.md) for the
  evidence-JSON schema produced by the deep smoke.
- [family_gating.md](family_gating.md) for what each capability flag
  unlocks.
