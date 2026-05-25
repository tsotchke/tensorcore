# lib/hip/ — HIP/chipStar backend

Vendor-neutral GPU compute backend. Targets every GPU with a SPIR-V
compiler — Intel (Level Zero), NVIDIA (OpenCL), AMD (OpenCL), ARM Mali
(OpenCL), Aurora-class HPC clusters. The path that takes tensorcore
beyond Apple-only without requiring a CUDA dependency.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  tensorcore C ABI (tc_gemm, tc_attention_forward, ...)           │
└────────────────────────────────┬─────────────────────────────────┘
                                 │  dispatch on device.kind
                                 ▼
┌──────────────────┬──────────────────┬──────────────────┐
│  Metal backend   │  HIP backend     │  CPU SIMD        │
│  (Apple GPU)     │  (this dir)      │  (any host)      │
└──────────────────┴────────┬─────────┴──────────────────┘
                            │
                  ┌─────────┴─────────┐
                  │  chipStar (HIP-on-│
                  │  SPIR-V dispatcher)│
                  └─────────┬─────────┘
                            │
        ┌───────────────────┼───────────────────┬─────────────┐
        │                   │                   │             │
        ▼                   ▼                   ▼             ▼
   Intel Level Zero    NVIDIA OpenCL       AMD OpenCL    ARM Mali OpenCL
   (Arc, Ponte         (RTX 30/40, H100,   (Vega, RDNA,  (Tegra, mobile)
    Vecchio, iGPU)      A100, ...)          MI300, ...)
```

## Files in this directory

- `device.cpp` — `tc_hip_init`, `tc_hip_device_info_get`, device enumeration.
  Without `TC_ENABLE_HIP`, it compiles to deterministic unsupported behavior.
- `buffer.cpp` — `tc_hip_buffer_alloc`, `tc_hip_buffer_map`. Pinned-memory
  allocation through chipStar (`hipHostMalloc` with `hipHostMallocMapped`).
  Equivalent to Apple's `MTLStorageModeShared` for the iGPU / Tegra case;
  separate device/host buffers on dGPUs with explicit transfer.
- `pipeline_cache.cpp` — name → compiled-SPIR-V kernel cache, parallel
  to `lib/core/pipeline_cache.mm` but for HIP.
- `gemm.cpp` — `tc_gemm` dispatch into the HIP path. Uses chipStar's
  hipBLAS port for the bulk of the work; only the dispatch wrapper lives
  here. Without `TC_ENABLE_HIP`, the internal HIP GEMM hook returns
  `TC_ERR_UNSUPPORTED_FAMILY`.
- `attention.cpp` — FlashAttention-2 forward / backward, ported from
  `kernels/metal/flash_attention.metal` to HIP. The algorithm is
  identical; only the thread-block / warp / shared-memory primitives
  differ.
- `kernels/*.hip` — HIP source files compiled to SPIR-V by chipStar's
  clang. Mirrors `kernels/metal/*.metal`.

## chipStar dependency

Required at build time:
- chipStar 1.1+ (https://github.com/CHIP-SPV/chipStar)
- LLVM/Clang 19 (chipStar's branch preferred for fewer surprises)
- SPIRV-LLVM-Translator 19
- An OpenCL or Level Zero runtime for at least one device on the build
  host (or you can build kernels offline and ship the SPIR-V).

CMake detects via `find_package(HIP)` once chipStar is installed.
`TC_ENABLE_HIP` cache variable forces or disables.

At runtime: the chipStar runtime (`libCHIP.so`) plus the appropriate ICD
loader. `clinfo` should show at least one SPIR-V-capable device for the
backend to come up.

Run `scripts/probe_hip_toolchain.py --json /tmp/hip-toolchain.json` before
the build on a new machine. The evidence checker can require build-toolchain
readiness (`hipcc` + HIP CMake config), SPIR-V runtime readiness
(`llvm-spirv` + OpenCL/Level Zero), or full hipBLAS GEMM readiness.

## Kernel porting strategy

The Metal kernels in `kernels/metal/` are written in MSL with
`simdgroup_matrix` MMA intrinsics. The HIP equivalents go in
`kernels/hip/` and use:

| Metal concept | HIP equivalent |
|---|---|
| `simdgroup_matrix` MMA | `__hip_matrix_*` cooperative matrix ops |
| `simdgroup` (32 threads) | warp / subgroup (32 on NVIDIA, 64 on AMD, 16 on Intel) |
| `threadgroup memory` (32 KB) | `__shared__` LDS (48-100 KB depending on vendor) |
| `[[buffer(N)]]` | kernel argument |
| `function_constant` | template specialization at JIT time |

Because chipStar compiles HIP source to SPIR-V, the same `.hip` file
runs on every target vendor. Vendor-specific tuning (warp size, LDS
budget, register pressure) becomes runtime device-info lookups, not
multiple source files.

## Library re-use via chipStar

chipStar ships ports of the standard HIP/ROCm math stack:
- `hipBLAS` (rocBLAS via chipStar) — fp16/fp32/fp64 GEMM, all transposes
- `hipFFT` — FFTs
- `hipSOLVER` — LAPACK-class linear algebra
- `hipSPARSE` — sparse matrix ops
- `rocPRIM` / `hipCUB` — parallel primitives
- `rocThrust` — Thrust-style algorithms

These provide vendor-tuned implementations across all our supported
SPIR-V devices. tensorcore's `tc_gemm` HIP path delegates to hipBLAS;
custom kernels (attention, quantized GEMV, training kernels) live in
`kernels/hip/`.

## Status

**Phase 0**: API + scaffolding. Header declares `tc_hip_*`; `device.cpp`
and `gemm.cpp` build as unsupported stubs unless `TC_ENABLE_HIP` is
compiled in.

**Phase 1.1-1.3**: chipStar install on cosbox, xavier, old-donkey.

**Phase 1.4** (current): device init plus optional fp32 hipBLAS dispatch.
`TC_ENABLE_HIP=ON` now builds runtime diagnostics when the HIP runtime target
is present, even if hipBLAS is absent. When hipBLAS is found too, `tc_gemm`
routes to `TC_BACKEND_HIP` after `tc_hip_init` succeeds, using a host-staged
`hipblasSgemm` path until HIP-owned buffer allocation lands.
`test_hip_device`, `test_hip_gemm`, and `scripts/ci_hip_smoke.sh` validate
runtime availability, `hipblas_sgemm_staged`, and explicit CPU fallback.

**Phase 1.5-1.6**: fp16/bf16/int8 GEMM via hipBLAS plus HIP buffer
policy.

**Phase 1.7**: FlashAttention port.

**Phase 1.8**: cross-vendor bench (NVIDIA RTX 3090 + Tegra Volta +
optional Intel GPU host).

Estimated total: 6-8 weeks once chipStar is installed.

## References

- chipStar: https://github.com/CHIP-SPV/chipStar
- HIP programming guide: https://rocm.docs.amd.com/projects/HIP/
- SPIR-V spec: https://www.khronos.org/spir/
- Level Zero spec: https://spec.oneapi.io/level-zero/
