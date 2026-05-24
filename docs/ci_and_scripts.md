# CI and scripts

`tensorcore` ships a tight CI surface and a small set of helper scripts in
`scripts/`. This page is the field guide: what each workflow runs, what
each script does, when you'd reach for one.

## CI workflows

Three GitHub Actions workflows live under `.github/workflows/`:

### `ci.yml` — gate on every push / PR

Runs on **macos-14** and **macos-15** runners in parallel. Steps per
runner:

1. **`scripts/check_version_consistency.sh`** — fail-fast if `pyproject.toml`,
   `CMakeLists.txt::project(VERSION ...)`, and the
   `TENSORCORE_VERSION_{MAJOR,MINOR,PATCH}` triple in
   `include/tensorcore/tensorcore.h` disagree.
2. **Configure + build** — `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`
3. **`scripts/ci_macos_test.sh`** — runs `ctest` but skips GPU-required
   tests when the runner only exposes the paravirtual GPU.
4. **Install smoke** — `cmake --install` into a temp prefix, then
   `cmake --find-package` and `pkg-config --modversion tensorcore` to
   verify the out-of-tree consumer contract.
5. **`scripts/ci_python_smoke.sh`** — sets up a venv, installs the binding
   editable, asserts `tc.version()`, the diagnostic helpers, and the
   tensorops kernel selector.

This is the gate. PRs need it green to merge.

### `release.yml` — wheel + GitHub release on `v*` tag

Runs on **macos-15**. Steps:

1. Version consistency check.
2. Configure + build + `ci_macos_test.sh`.
3. **`scripts/release_smoke.sh`** — the deeper smoke (see below).
4. Install native artifacts to a release prefix.
5. Build the wheel via `pip wheel . --no-build-isolation`, with
   `TENSORCORE_NATIVE_DIR` pointing at the install lib dir so the dylib
   + metallib are vendored into the package.
6. **Verify wheel** — uninstall + reinstall + `import tensorcore as tc;
   tc.version()` against the wheel.
7. Upload `tensorcore_apple-*.whl` as an artifact.
8. **Publish** — `gh release create` and `gh release upload` for the
   tag.

### `hardware-evidence.yml` — manual, self-hosted runner

Triggered via `workflow_dispatch`. Runs on a `[self-hosted, macOS, ARM64]`
runner with real GPU exposure. Executes `scripts/release_smoke.sh` with
`REQUIRE_GPU=1` and (optionally) `REQUIRE_METAL4_TENSOROPS=1`. Uploads
`build/release_smoke_runtime_evidence.json` as the artifact — a
machine-readable record of (chip, family, TensorOps availability, backend
chosen per call) that downstream integrators can consume.

## Scripts

### `check_version_consistency.sh`

Reads version from `pyproject.toml`, `CMakeLists.txt`, and
`include/tensorcore/tensorcore.h`. Asserts all three agree. Prints
`tensorcore version OK: X.Y.Z` on success; lists the disagreements
otherwise. First step in CI; cheap; catches the most common release
mistake.

### `ci_macos_test.sh`

Runs `ctest --test-dir build --output-on-failure`, with explicit
`-E` exclusion of tests that need a real GPU when the runner is
paravirtualized (detected via `system_profiler SPDisplaysDataType`).
On a real-GPU host this runs the full 20-test suite.

### `ci_python_smoke.sh`

Sets up a venv at `${RUNNER_TEMP:-/tmp}/tensorcore-venv`, installs the
Python package editable, and runs an inline smoke script that asserts:

- `tc.version()` matches `pyproject.toml::version`
- `tc.status_string(tc.TC_OK) == "ok"`
- `tc.dtype_name("fp53") == "fp53"`
- `tc.backend_name(tc.TC_BACKEND_TENSOROPS_M5) == "tensorops_m5"`
- `tc.backend_name(tc.TC_BACKEND_METAL_COMPUTE) == "metal_compute"`
- `tc.backend_name(tc.TC_BACKEND_CUDA) == "cuda"`
- `tc.backend_name(tc.TC_BACKEND_HIP) == "hip"`
- `tc.last_backend_name() == "none"` (before any kernel runs)
- `tc.tensorops_gemm_kernel_name("f16") == "tc4_gemm_f16"`
- `tc.tensorops_gemm_kernel_name("i8", "i32") is None`

Run locally:

```sh
cmake --install build --prefix /tmp/tensorcore-install
bash scripts/ci_python_smoke.sh
```

The install step is required because the script defaults `PREFIX` to
`/tmp/tensorcore-install`.

### `ci_pytorch_smoke.sh`

Optional bridge smoke for `bindings/pytorch`. If PyTorch is importable, the
script force-builds `tensorcore_torch` against
`${TENSORCORE_LIB_DIR:-build-portable-cpu-current}` and validates:

- fp32 matmul against PyTorch
- bf16 matmul against fp32-accum then bf16-rounded reference
- non-contiguous inputs
- `K == 0` and empty-result matmuls
- dtype/shape error paths
- opt-in `torch.matmul` dispatcher routing and autograd fallback
- importing the extension after the ctypes wrapper already initialized the
  native library
- `tensorcore_torch` package import registers PyTorch's PrivateUse1 backend
  name as `tensorcore`, installs `torch.tensorcore`, and generates the
  usual tensor helper methods
- `tensorcore_torch.pytorch_backend_state()` and
  `torch.tensorcore.backend_state()` report the same structured capability
  snapshot: registered runtime shim, generated tensor helpers, matmul
  extension loaded, and host-memory allocator/storage/factory kernels marked
  `available`
- `tensorcore_torch.matmul_eligibility()` exposes the dispatcher gate used
  by the opt-in `torch.matmul` hook, including explicit fallback reasons for
  dtype, rank, and shape mismatches
- direct tensor allocation with `torch.empty(..., device="tensorcore")`,
  explicit `to_tensorcore()` / `to_cpu()` round-trips, and PrivateUse1
  matmul dispatch

If PyTorch is not importable, the script skips by default. Set
`REQUIRE_PYTORCH=1` to make that a hard failure.
Set `TENSORCORE_PYTORCH_SMOKE_EVIDENCE_PATH=/tmp/pytorch.json` to emit a
machine-readable node-health artifact with the skip/pass status, torch
version, backend state, matmul checks, and direct device-allocation status.
Validate it with:

```sh
python3 scripts/check_pytorch_smoke_evidence.py /tmp/pytorch.json
python3 scripts/check_pytorch_smoke_evidence.py /tmp/pytorch.json \
  --require-pytorch --require-backend-allocation
```

Run locally:

```sh
cmake --build build-portable-cpu-current --parallel
REQUIRE_PYTORCH=1 REQUIRE_PYTORCH_BACKEND=1 scripts/ci_pytorch_smoke.sh
```

### `ci_portable_cpu.sh`

Builds with `TC_ENABLE_METAL=OFF`, runs the portable CTest suite,
installs the native SDK, verifies CMake and pkg-config consumers, then
runs an inline Python smoke against the installed shared library. The
CTest phase covers `TC_DIST_GLOO` with four forked localhost ranks,
including the TCP ring fp32 SUM path. The Python phase covers
DiLoCo-over-GLOO with two forked localhost ranks, and runs isolated
subprocess GEMM smokes with
`TC_USE_AVX2_GEMM=1`, `TC_USE_AVX2_GEMM=1 TC_AVX2_THREADS=1`,
`TC_USE_NEON_GEMM=1`, and `TC_USE_AMX_GEMM=1`.
The portable CTest suite also builds direct AMX regression binaries for the
tile kernel and edge-tile alpha/beta wrapper; they skip unless
`TC_RUN_AMX_GEMM_TEST=1` is set. The separate `test_amx_probe` CTest
validates AMX availability, ISA-version, cluster-count, and fp16/bf16
gating metadata without executing raw AMX instructions.

`bench_gemm_shared` is the shared-runtime GEMM benchmark. Use it for AVX2
OpenMP throughput work, for example:

```sh
TC_USE_AVX2_GEMM=1 TC_AVX2_THREADS=64 \
  TC_BENCH_DTYPES=f32 TC_BENCH_SIZES=2048,4096 \
  build/bench/bench_gemm_shared
```

AMX uses reverse-engineered Apple instructions, so the AMX subprocess
treats SIGILL as a skip by default instead of taking down the whole smoke
on hosts that block the instruction. Set `REQUIRE_AMX_GEMM=1` to require
the Python AMX opt-in path on local Apple-Silicon verification machines;
set `TC_RUN_AMX_GEMM_TEST=1` when running the direct C AMX regressions on
known-good local hardware.

### `run_live_mesh_smoke.sh`

Runs the operational four-rank mesh smoke across Atlas, Enki, old-donkey,
and cosbox. With `TC_MESH_PREPARE=1`, it uploads the local portable CPU
rank binary to Enki, archives the current committed checkout to the Linux
nodes, builds their portable `test_dist_remote` target, launches all four
ranks, and verifies per-rank logs. The default `TC_MESH_TEST=all` covers
direct GLOO ring fp32 SUM and the DiLoCo sparse TOPK outer-step path.
`TC_MESH_DILOCO_CYCLES`, `TC_MESH_DILOCO_INNER_STEPS`, and
`TC_MESH_DILOCO_ELEMENTS` scale the training-sync soak.

```sh
TC_MESH_PREPARE=1 scripts/run_live_mesh_smoke.sh
TC_MESH_TEST=allreduce scripts/run_live_mesh_smoke.sh   # transport-only
TC_MESH_TEST=diloco scripts/run_live_mesh_smoke.sh      # training-sync only
```

### `run_live_mesh_training_demo.sh`

Runs `examples/mesh_training_demo` across the same four-rank mesh: Atlas
rank 0, Enki rank 1, old-donkey rank 2, and cosbox rank 3. This is the
full demo loop rather than the compact transport probe: RMSNorm -> GEMM ->
softmax+CE -> backward -> AdamW, with DiLoCo outer sync and activation
checkpointing enabled by default. With `TC_MESH_PREPARE=1`, the script
archives the current committed checkout to the Linux hosts, builds their
`mesh_training_demo` target, copies the local Apple binary to Enki, and
builds cosbox with `TC_ENABLE_CUDA=ON` unless `TC_MESH_RANK3_CUDA=0`.
Set `TC_MESH_RANK1_PREPARE=linux` when Enki/rank 1 should also be built
from the archived checkout instead of receiving the local binary.
Use `TC_MESH_RANK1_PATH`, `TC_MESH_RANK2_PATH`, and `TC_MESH_RANK3_PATH`
when a remote host needs a custom toolchain prefix for `cmake`, `nvcc`, or
runtime helper binaries; the path prefix is applied during both prepare and
rank launch.

```sh
TC_MESH_PREPARE=1 scripts/run_live_mesh_training_demo.sh
TC_MESH_RANK1_PREPARE=linux TC_MESH_PREPARE=1 scripts/run_live_mesh_training_demo.sh
TC_MESH_RANK3_PATH=/usr/local/cuda/bin TC_MESH_PREPARE=1 \
  scripts/run_live_mesh_training_demo.sh
TC_MESH_TRAINING_INNER=8 TC_MESH_TRAINING_OUTER=5 scripts/run_live_mesh_training_demo.sh
TC_MESH_TRAINING_CHECKPOINT=0 scripts/run_live_mesh_training_demo.sh
TC_MESH_LOCAL_ONLY=1 TC_MESH_TRAINING_OUTER=1 \
  TC_MESH_TRAINING_EVIDENCE_PATH=/tmp/live-mesh-training-local.json \
  scripts/run_live_mesh_training_demo.sh
TC_MESH_TRAINING_EVIDENCE_PATH=/tmp/live-mesh-training.json scripts/run_live_mesh_training_demo.sh
TC_GLOO_ADVERTISE_HOSTS=100.96.130.16,100.111.56.36,100.121.14.68,100.86.83.35 \
  TC_MESH_TRAINING_EVIDENCE_PATH=/tmp/live-mesh-training.json \
  scripts/run_live_mesh_training_demo.sh
python3 scripts/check_live_mesh_training_evidence.py /tmp/live-mesh-training.json \
  --require-direct-ring --require-checkpoint --require-cuda-rank3
python3 scripts/check_live_mesh_training_evidence.py /tmp/live-mesh-training-local.json \
  --require-direct-ring --require-checkpoint --require-local-only
```

The optional evidence JSON uses schema
`tensorcore.live_mesh_training.evidence.v1` and records per-rank rendezvous,
outer-step losses, selected backend, direct-ring route counts, and checkpoint
discard/realize counters.

### `check_live_mesh_training_evidence.py`

Validates the JSON artifact emitted by `run_live_mesh_training_demo.sh` via
`TC_MESH_TRAINING_EVIDENCE_PATH`. Use `--min-outer-steps`,
`--require-direct-ring`, `--require-checkpoint`, and
`--require-cuda-rank3` to enforce the operational contract for a live run.
Use `--require-local-only` for the localhost multi-rank regression mode.

### `check_operational_evidence.py`

Validates a complete operational evidence bundle by delegating to the release,
SDK26, CUDA, HIP, PyTorch, and live-mesh evidence checkers, then applying
bundle-level policy. For production promotion, use the clean-head flags so
stale or dirty-tree release, SDK26, PyTorch, and live-mesh evidence cannot
satisfy the current head's deployment gate.

```sh
python3 scripts/check_operational_evidence.py \
  --release /tmp/release/release_smoke_runtime_evidence.json \
  --sdk26 /tmp/sdk26/release_smoke_runtime_evidence.json \
  --cuda /tmp/cuda-smoke.json \
  --pytorch /tmp/pytorch.json \
  --live-mesh /tmp/live-mesh-training.json \
  --require-release --require-sdk26 --require-cuda --require-pytorch \
  --require-pytorch-backend-allocation --require-live-mesh \
  --require-release-clean-head --require-sdk26-clean-head \
  --require-cuda-clean-head --require-pytorch-clean-head \
  --require-live-clean-head \
  --min-live-outer-steps 2 \
  --require-direct-ring --require-checkpoint --require-cuda-rank3
```

Add `--hip /tmp/hip.json --require-hip --require-hip-clean-head` on mesh
subsets that include a required HIP/chipStar accelerator host.

### `ci_cuda_smoke.sh`

Configures a Linux CUDA build with `TC_ENABLE_CUDA=ON`, runs its CTest
suite, then runs fp32/fp16 Python GEMM smokes and Python training dispatch
smokes through the default CUDA policy. When `TC_ENABLE_CUDA=ON`, CTest includes
`test_cuda_gemm`, which asserts managed-memory cuBLAS dispatch and applies
a 4096^3 fp32 perf gate on high-end Ampere+ devices. On CUDA devices that
report support, the CTest path also covers bf16/fp32-accum and int8/i32-accum
cuBLAS GEMM plus managed-memory RMSNorm/LayerNorm/RoPE/SwiGLU/softmax/AdamW
training dispatch, including RMSNorm/LayerNorm/RoPE/SwiGLU/softmax backward
and both fp32/fp16-gradient AdamW paths. The Python smoke asserts numerical
GEMM output, `backend=cuda`, expected managed-memory cuBLAS kernel names,
explicit `TC_DISABLE_CUDA_GEMM=1` CPU fallback, and CUDA dispatch for RMSNorm,
LayerNorm, RoPE, SwiGLU, softmax, and AdamW fp32/fp16-gradient updates.

If `TENSORCORE_CUDA_SMOKE_EVIDENCE_PATH` is set, the script writes
`tensorcore.cuda_smoke.evidence.v1`-style JSON with `runtime_status` set to
`passed`, `skipped_not_built`, or `skipped_runtime_unavailable`. Set
`REQUIRE_CUDA=1` to make skipped evidence fail the script on a host that is
expected to have a working NVIDIA runtime. Evidence records `git_head` and
`git_dirty`; archive-based deployments can supply that provenance via
`TENSORCORE_SOURCE_GIT_HEAD` / `TENSORCORE_SOURCE_GIT_DIRTY` or the
`.tensorcore_source_head` / `.tensorcore_source_dirty` files written by the
live-mesh prepare step.

```sh
TENSORCORE_CUDA_SMOKE_EVIDENCE_PATH=/tmp/cuda.json \
  TC_CUDA_BUILD_DIR=build-cuda scripts/ci_cuda_smoke.sh
python3 scripts/check_cuda_smoke_evidence.py /tmp/cuda.json --require-cuda
python3 scripts/check_cuda_smoke_evidence.py /tmp/cuda.json \
  --require-cuda --require-clean-head
```

### `ci_hip_smoke.sh`

Configures a Linux chipStar/HIP build with `TC_ENABLE_HIP=ON`, runs CTest,
then writes optional JSON evidence for the HIP runtime state. If HIP runtime
targets are missing, the script records `runtime_status=skipped_not_built`;
if HIP builds but no runtime device is available, it records
`skipped_runtime_unavailable`; if chipStar initializes but hipBLAS is not
installed, it records `runtime_only_no_hipblas`. On a working chipStar +
hipBLAS host, it asserts fp32 GEMM dispatch through `backend=hip`,
`kernel=hipblas_sgemm_staged`, and verifies `TC_DISABLE_HIP_GEMM=1` falls
back to a non-HIP backend. Set `TC_HIP_PREFIX=/path/to/chipstar-install`
when chipStar is outside the default CMake prefix paths. HIP evidence records
`git_head` and `git_dirty`; archive-based deployments can supply that
provenance via `TENSORCORE_SOURCE_GIT_HEAD` / `TENSORCORE_SOURCE_GIT_DIRTY`
or the `.tensorcore_source_head` / `.tensorcore_source_dirty` files written
by the live-mesh prepare step.

```sh
TENSORCORE_HIP_SMOKE_EVIDENCE_PATH=/tmp/hip.json scripts/ci_hip_smoke.sh
python3 scripts/check_hip_smoke_evidence.py /tmp/hip.json
python3 scripts/check_hip_smoke_evidence.py /tmp/hip.json --require-clean-head

REQUIRE_HIP=1 scripts/ci_hip_smoke.sh  # fails unless HIP dispatch passes
```

### `check_public_headers.sh`

For every header in `include/tensorcore/`, compiles a minimal C *and* a
minimal C++ TU that does nothing but `#include "tensorcore/foo.h"`.
Catches:
- missing `extern "C"` guards
- headers that don't compile standalone
- accidental dependency on a build-tree macro

Pure-source check; no GPU needed.

### `check_public_exports.sh`

Compares the dylib's actual exported symbol table (from `nm -gU`)
against the union of the `EXTERN_C` declarations in
`include/tensorcore/*.h`. Catches:
- a private symbol that escaped via the wrong visibility annotation
- a public symbol the dylib forgot to export

Backed by `cmake/tensorcore.exports`, which the linker uses to filter.

### `check_python_ffi_surface.py`

Parses the C ABI surface (function names + signatures from the headers)
and asserts the Python binding's ctypes argtypes/restype declarations
agree.

### `check_python_abi_layout.py`

Asserts the Python `ctypes.Structure` layouts (`TCGemmDesc`,
`TCAttentionDesc`, `TCDeviceInfo`, etc.) match what the C struct produces
under `sizeof` and field offsets. Catches ABI drift between header and
binding.

### `check_python_constants.py`

Asserts the Python `TC_*` module constants match the enum values declared
in the public headers. Catches drift like "Python says
`TC_BACKEND_PORTABLE_CPU = 6` but the header declares 7."

### `create_native_sdk_archive.sh`

From a populated `--prefix` install dir, packages the headers, libraries,
metallib, CMake config, and pkg-config files into a versioned
`tensorcore-native-sdk-X.Y.Z-darwin-arm64.tar.gz`. The archive carries
no build-tree paths; the install paths it references are relative to
the archive root.

```sh
cmake --install build --prefix /private/tmp/tensorcore-install
scripts/create_native_sdk_archive.sh /private/tmp/tensorcore-install
```

### `check_native_sdk_archive.sh`

Validates a native SDK archive:

- File structure matches the contract
- `pkgconfig/tensorcore.pc` resolves
- Compiles a minimal C consumer against the archived headers / dylib
- Asserts `tc.version()` matches the embedded version

```sh
scripts/check_native_sdk_archive.sh tensorcore-native-sdk-0.1.22-darwin-arm64.tar.gz
```

### `create_release_checksums.sh`

Emits a SHA manifest (`tensorcore-release-checksums-X.Y.Z.txt`) for the
wheel + native SDK archive of a given release. Used by
`release.yml` to publish reproducibility evidence alongside the
artifacts.

### `release_smoke.sh`

The deep smoke. ~1284 lines. Runs *everything*:

1. Build with `-DCMAKE_BUILD_TYPE=Release`.
2. Full `ctest` (gated by `REQUIRE_GPU` for hardware-only tests).
3. Install to a temp prefix.
4. Build native SDK archive + verify it via the consumer test.
5. Build the wheel into a temp dir + reinstall + smoke-test against the
   wheel.
6. Verify `tc_last_backend()` reports `SIMDGROUP_MATRIX` for a fp16 GEMM
   call (the "is the dispatch actually using the GPU?" check).
7. With `REQUIRE_METAL4_TENSOROPS=1`, additionally assert
   `tc_device_info.supports_tensorops_m5 == true` and that a GEMM dispatch
   reports `TC_BACKEND_TENSOROPS_M5`.
8. Emit a `build/release_smoke_runtime_evidence.json` artifact with
   chip/runtime status, package/consumer coverage, Metal 4 TensorOps compile
   and runtime status, and clean git-head provenance when the source checkout
   exposes `.git`.

Env knobs:

| Variable | Default | Effect |
|---|---|---|
| `REQUIRE_GPU` | `0` | Fail if no real GPU is exposed. CI uses `1` on self-hosted runners, `0` on macos-14 / macos-15. |
| `REQUIRE_METAL4_TENSOROPS` | `0` | Additionally fail if the M5 TensorOps path isn't taken. Hardware-evidence workflow opt-in. |
| `BUILD_DIR` | `$ROOT/build` | Override the build directory. |
| `PREFIX` | `/private/tmp/tensorcore-install` | Where to install the native artifacts. |
| `PY_PREFIX`, `WHEEL_DIR`, `WHEEL_PREFIX` | per-invocation temp dirs | Wheel build/install paths. |
| `TENSORCORE_RELEASE_SMOKE_EVIDENCE_PATH` | `$BUILD_DIR/release_smoke_runtime_evidence.json` | Set empty to disable evidence emission. |

Run locally:

```sh
scripts/release_smoke.sh                              # software-only smoke
REQUIRE_GPU=1 scripts/release_smoke.sh                # require real GPU
REQUIRE_GPU=1 REQUIRE_METAL4_TENSOROPS=1 scripts/release_smoke.sh  # M5+ only
```

This is the deepest "is this build releaseable?" check. If
`release_smoke.sh` passes locally on M-series hardware, the release
workflow will pass too.

## What to reach for when

| You want to… | Run |
|---|---|
| Sanity-check a local build before committing | `ctest --test-dir build --output-on-failure` |
| Validate the public headers compile alone | `scripts/check_public_headers.sh` |
| Validate the dylib export list matches the headers | `scripts/check_public_exports.sh` |
| Validate the Python binding hasn't drifted | `scripts/check_python_{ffi_surface,abi_layout,constants}.py` |
| Build and verify a native SDK tarball | `scripts/create_native_sdk_archive.sh && scripts/check_native_sdk_archive.sh` |
| Run the CI Python smoke locally | `cmake --install build --prefix /tmp/tensorcore-install && scripts/ci_python_smoke.sh` |
| Pre-release wide smoke | `scripts/release_smoke.sh` (add `REQUIRE_GPU=1` if you have one) |
| Cross-check version triple | `scripts/check_version_consistency.sh` |
