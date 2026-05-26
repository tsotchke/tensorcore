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
version, backend state, matmul checks, direct device-allocation status, and
ICC-readable function coverage for the Python shim plus native extension
dispatch paths.
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

### `run_fallback_runtime_smoke.py`

Focused runtime evidence for the intentional GEMM fallback surface. The script
reuses the existing `test_gemm_f32`, `test_gemm_bf16`, and `test_gemm_i8`
binaries from `${BUILD_DIR:-build}` and records an ICC-readable trace for:

- Accelerate fp32 GEMM fallback (`tc_accelerate_gemm_f32`)
- MPS fp32 fallback dispatch (`tc_mps_gemm`)
- bf16 software fallback through fp32 conversion
- i8 software fallback through fp32 conversion
- the fallback layout and dtype helpers used by those paths

The default artifact is `build/fallback_runtime_evidence.json`. Validate it
with:

```sh
python3 scripts/check_fallback_runtime_evidence.py \
  build/fallback_runtime_evidence.json --require-pass
```

Run locally after building the Metal test binaries:

```sh
python3 scripts/run_fallback_runtime_smoke.py --require-pass
```

The artifact's `files` section is intentionally shaped like the release-smoke
coverage payload so ICC can consume it as `--trace-file` evidence when ranking
fallback-path readiness.

### `run_metallib_build_rule_evidence.py`

Focused evidence for `cmake/compile_metallib.cmake`. The runner generates a
small out-of-tree CMake project that includes the real `compile_metallib`
module, calls `tc_compile_metallib`, builds a one-kernel `probe.metallib`, and
records the artifact SHA-256 plus ICC-readable function coverage.

The default artifact is
`build/metallib-build-rule-evidence/metallib_build_rule_evidence.json`.
Validate it with:

```sh
python3 scripts/check_metallib_build_rule_evidence.py \
  build/metallib-build-rule-evidence/metallib_build_rule_evidence.json \
  --require-pass
```

Run locally:

```sh
python3 scripts/run_metallib_build_rule_evidence.py --require-pass
```

On non-Apple hosts, or Apple hosts without `xcrun metal`/`metallib`, the
artifact reports `status=blocked` with a specific reason instead of treating
the missing external Metal backend as a source failure.

### `run_python_packaging_evidence.py`

Focused evidence for `setup.py` packaging paths. The runner uses local native
artifacts, defaults to `build/`, runs the native artifact copy path through
`build_py`, runs the macOS validation-tool path through `_run_tool`, builds a
wheel with an explicit platform tag, and records copied artifact hashes plus
the wheel hash.

The default artifact is
`build/python-packaging-evidence/python_packaging_evidence.json`. Validate it
with:

```sh
python3 scripts/check_python_packaging_evidence.py \
  build/python-packaging-evidence/python_packaging_evidence.json \
  --require-pass
```

Run locally after building native artifacts:

```sh
python3 scripts/run_python_packaging_evidence.py --require-pass
```

On hosts without the required native artifacts, or non-macOS hosts where the
`lipo` validation path is not applicable, the artifact reports
`status=blocked` with a specific reason.

### `run_distributed_runtime_evidence.py`

Focused evidence for local distributed runtime paths. The runner executes the
forked GLOO ring, dense DiLoCo, and sparse DiLoCo smokes from an existing build
directory, captures command traces, and emits ICC-readable coverage for:

- `lib/distributed/gloo_tcp.cpp`
- `lib/distributed/diloco.cpp`, including the TOPK sparse-delta path when
  `test_diloco_sparse_fork` passes

The default artifact is `build/distributed_runtime_evidence.json`. Validate it
with:

```sh
python3 scripts/check_distributed_runtime_evidence.py \
  build/distributed_runtime_evidence.json --require-pass
```

Run locally after building tests:

```sh
python3 scripts/run_distributed_runtime_evidence.py --require-pass
```

If the host cannot bind loopback sockets, the artifact reports
`status=blocked` with `loopback_unavailable` instead of treating the test skip
as a code failure.

### `run_amx_bench_evidence.py`

Focused evidence for local AMX and GEMM benchmark paths. The runner executes
the AMX metadata probe, the opt-in direct AMX GEMM regression, and a tiny
`bench_gemm` run with `TC_BENCH_SIZES=16`, `TC_BENCH_DTYPES=f32`,
`TC_BENCH_WARMUP=0`, and `TC_BENCH_ITERS=1`. It records ICC-readable coverage
for:

- `lib/ops/gemm_cpu_amx.cpp`
- `bench/bench_gemm.c`

The default artifact is `build/amx_bench_evidence.json`. Validate it with:

```sh
python3 scripts/check_amx_bench_evidence.py \
  build/amx_bench_evidence.json --require-pass
```

Run locally after building the portable CPU AMX tests and GEMM benchmark:

```sh
python3 scripts/run_amx_bench_evidence.py --require-pass
```

TensorOps M5 layout helpers are probed as an optional layout check. On current
non-SDK26 or non-M5 hosts the artifact stays `status=passed` for AMX/bench
coverage while recording `summary.optional_blocked_reasons` such as
`tensorops_layout:skipped_no_metal4_sdk` or `tensorops_layout:skipped_no_m5`.

### `run_cpu_ops_runtime_evidence.py`

Focused evidence for portable CPU GEMM and Conv2D helper paths. The runner uses
the portable CPU build, executes `test_portable_cpu` and `test_conv2d`, and
emits ICC-readable coverage for:

- `lib/ops/gemm_cpu.cpp:gemm_compute`
- `lib/ops/conv2d_cpu.cpp:direct_sgemm_f32`

The default artifact is `build/cpu_ops_runtime_evidence.json`. Validate it
with:

```sh
python3 scripts/check_cpu_ops_runtime_evidence.py \
  build/cpu_ops_runtime_evidence.json --require-pass
```

Run locally after building the portable CPU suite:

```sh
python3 scripts/run_cpu_ops_runtime_evidence.py --require-pass
```

### `run_metal_ops_runtime_evidence.py`

Focused evidence for local Metal attention and Conv2D host dispatch. The
runner executes `test_attention_correctness` and the Metal-build `test_conv2d`
with `TC_TRACE=1`, then emits ICC-readable coverage for:

- `lib/ops/attention.mm:encode_forward`
- `lib/ops/conv.mm:conv_bytes`

The default artifact is `build/metal_ops_runtime_evidence.json`. Validate it
with:

```sh
python3 scripts/check_metal_ops_runtime_evidence.py \
  build/metal_ops_runtime_evidence.json --require-pass
```

Run locally after building the Metal test suite:

```sh
python3 scripts/run_metal_ops_runtime_evidence.py --require-pass
```

The runner also records async-copy shader availability as an optional blocked
check. Existing host traces show public ops/backends, not Metal shader
function-line execution, so `async_copy` remains unclaimed until a Metal
capture, shader instrumentation, or selected-kernel trace exists.

### `run_eshkol_tensorcore_bridge_smoke.py`

Focused evidence for the Eshkol bridge surface. The script uses a local
`eshkol-run` binary, defaults to `~/Desktop/eshkol/build/eshkol-run` when
available, and tries to compile and run:

- `eshkol/hello_tensorcore.esk`
- `eshkol/tensorcore_bridge_smoke.esk`

It emits `build/eshkol_tensorcore_bridge_evidence.json` with structured
compile/runtime attempts, source module-load checks, and ICC-readable function
coverage once the bridge actually runs. The Scheme module resolves `__tc-*`
through the `tc_eshkol_*` C shims exported by `libtensorcore`; if the selected
native backend cannot initialize, the smoke omits the success markers and the
artifact records a failed runtime attempt. Validate the artifact with:

```sh
python3 scripts/check_eshkol_tensorcore_bridge_evidence.py \
  build/eshkol_tensorcore_bridge_evidence.json
```

Run locally:

```sh
python3 scripts/run_eshkol_tensorcore_bridge_smoke.py \
  --build-dir build-portable-cpu-current --require-pass
python3 scripts/check_eshkol_tensorcore_bridge_evidence.py \
  build/eshkol_tensorcore_bridge_evidence.json --require-pass
```

Use the portable CPU build when you need backend-independent bridge evidence.
Use the default Metal build when specifically validating the local Apple GPU
path.

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

### `ci_windows_cpu.ps1`

Builds the Windows x86 portable CPU target with MSVC or another CMake
Windows generator. It configures `TC_ENABLE_METAL=OFF`, disables CUDA/HIP
unless explicitly tested elsewhere, runs CTest with `-C Release` including
the Winsock-backed local `TC_DIST_GLOO` split-rank smoke, installs the
native SDK, and imports Python against the produced `tensorcore.dll`.
If `cmake` / `ctest` are not on PATH, the script also checks the Visual
Studio Build Tools bundled CMake location.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\ci_windows_cpu.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\ci_windows_cpu.ps1 `
  -Generator "Visual Studio 17 2022" -Platform x64
```

Use `-SkipPython` only for first-pass compiler bring-up on a machine that
does not have Python installed yet; deployment validation should keep the
Python smoke enabled.

### `bootstrap_windows_cpu.ps1`

Checks a Windows x86 host for Visual Studio Build Tools, CMake/CTest, and
Python, then runs `ci_windows_cpu.ps1` with explicit tool paths. With
`-Install`, it downloads Visual Studio Build Tools 2022 and Python for a
first-time machine. The Build Tools install requires an Administrator
PowerShell; Python installs for the current user.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\bootstrap_windows_cpu.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\bootstrap_windows_cpu.ps1 -Install
```

### `run_windows_host_smoke.sh`

Runs the Windows bootstrap on a Tailscale/SSH-reachable host from a Unix
controller. The SSH target is required through `TC_WINDOWS_SSH` or a private
local config file at `~/.config/tensorcore/windows-host.env`. The script
clones `https://github.com/tsotchke/tensorcore.git` to `src/tensorcore` if
needed, fast-forwards `master`, then launches `bootstrap_windows_cpu.ps1`.

```sh
mkdir -p ~/.config/tensorcore
printf '%s\n' 'TC_WINDOWS_SSH=tsotchke@desktop-jack-blupc' \
  > ~/.config/tensorcore/windows-host.env
TC_WINDOWS_SSH_KEY="$HOME/.ssh/id_ed25519_jack" scripts/run_windows_host_smoke.sh
TC_WINDOWS_EVIDENCE_PATH=/tmp/windows-host.json \
  TC_WINDOWS_SSH_KEY="$HOME/.ssh/id_ed25519_jack" \
  scripts/run_windows_host_smoke.sh
python3 scripts/check_windows_host_smoke_evidence.py /tmp/windows-host.json \
  --require-windows --require-clean-head --require-python
```

Set `TC_WINDOWS_SSH`, `TC_WINDOWS_REPO`, `TC_WINDOWS_REF`, or
`TC_WINDOWS_REMOTE_URL` for other Windows hosts. The default update is
non-destructive; set `TC_WINDOWS_RESET=1` only when the remote checkout is a
dedicated smoke workspace that can be hard-reset to `origin/<ref>`.
`TC_WINDOWS_EVIDENCE_PATH` writes
`tensorcore.windows_host_smoke.evidence.v1` after the remote bootstrap passes.

### `run_windows_cuda_probe.sh`

Runs a non-destructive CUDA readiness probe on a Tailscale/SSH-reachable
Windows host. It fast-forwards the remote checkout like
`run_windows_host_smoke.sh`, then records NVIDIA driver/device visibility,
`nvidia-smi` compute-app admission state, CUDA toolkit / `nvcc` discovery,
and optional Windows CUDA configure/build/CTest proof.
This is the scheduling gate for Jack's RTX lane before any GPU work is allowed.

```sh
TC_WINDOWS_CUDA_EVIDENCE_PATH=/tmp/windows-cuda.json \
  scripts/run_windows_cuda_probe.sh
python3 scripts/check_windows_cuda_probe_evidence.py /tmp/windows-cuda.json \
  --require-driver --require-admission-clear --require-clean-head
```

Set `TC_WINDOWS_CUDA_BUILD_SMOKE=1` to have the probe configure a CUDA-enabled
Ninja/MSVC build, run full CTest, and require `test_cuda_gemm`:

```sh
TC_WINDOWS_CUDA_BUILD_SMOKE=1 \
TC_WINDOWS_CUDA_EVIDENCE_PATH=/tmp/windows-cuda.json \
  scripts/run_windows_cuda_probe.sh
python3 scripts/check_windows_cuda_probe_evidence.py /tmp/windows-cuda.json \
  --require-ready --require-build-smoke
```

Driver-only evidence is still useful before a Windows lane is promoted into
the scheduler: it proves the GPU/driver and admission state while keeping
scheduling conservative until the build toolchain is complete.
On Windows display GPUs, `nvidia-smi --query-compute-apps` may report ordinary
desktop/WDDM processes as `[Insufficient Permissions]` even when the full
`nvidia-smi` process table has no visible CUDA entries. The probe records those
rows as `ignored_opaque_wddm` and allows admission only when the visible CUDA
process table is empty.

Use `scripts/check_windows_cuda_resource_admission.py` as the scheduler
`admission_cmd` for Jack-style Windows CUDA lanes. Use
`scripts/mesh_windows_worker_identity.py` as a workload-specific
`worker_identity_cmd` after the job's `start_cmd` and `post_start_probe_cmd`
can identify the actual Windows CUDA worker process.

### `run_windows_gloo_smoke.ps1`

CTest helper for Windows portable CPU builds. It reserves a loopback TCP
port, starts two `test_dist_remote.exe` ranks against that rendezvous URL,
prints per-rank stdout/stderr, and fails if either process exits nonzero or
times out. It is normally launched by CTest rather than run directly.

### `run_live_mesh_smoke.sh`

Runs the operational four-rank mesh smoke across the configured private mesh.
Set `TC_MESH_CONFIG`, or export `TC_MESH_RANK0_HOST` and
`TC_MESH_RANK{1,2,3}_SSH`; private hostnames and tailnet addresses are
intentionally not source defaults. With `TC_MESH_PREPARE=1`, it uploads the
local portable CPU rank binary to rank 1, archives the current committed
checkout to the Linux nodes, builds their portable `test_dist_remote` target,
launches all four ranks, and verifies per-rank logs. The default
`TC_MESH_TEST=all` covers
direct GLOO ring fp32 SUM and the DiLoCo sparse TOPK outer-step path.
`TC_MESH_DILOCO_CYCLES`, `TC_MESH_DILOCO_INNER_STEPS`, and
`TC_MESH_DILOCO_ELEMENTS` scale the training-sync soak.

```sh
TC_MESH_PREPARE=1 scripts/run_live_mesh_smoke.sh
TC_MESH_TEST=allreduce scripts/run_live_mesh_smoke.sh   # transport-only
TC_MESH_TEST=diloco scripts/run_live_mesh_smoke.sh      # training-sync only
```

### `run_live_mesh_training_demo.sh`

Runs `examples/mesh_training_demo` across the same configured four-rank mesh.
Set `TC_MESH_CONFIG`, or export `TC_MESH_RANK0_HOST` and
`TC_MESH_RANK{1,2,3}_SSH`, before running against physical hosts. This is the
full demo loop rather than the compact transport probe: RMSNorm -> GEMM ->
softmax+CE -> backward -> AdamW, with DiLoCo outer sync and activation
checkpointing enabled by default. With `TC_MESH_PREPARE=1`, the script
archives the current committed checkout to the Linux hosts, builds their
`mesh_training_demo` target, copies the local Apple binary to Enki, and
builds cosbox with `TC_ENABLE_CUDA=ON` unless `TC_MESH_RANK3_CUDA=0`.
When CUDA is requested, rank 3 must report `backend=cuda`; set
`TC_MESH_ALLOW_CUDA_FALLBACK=1` only when a CPU fallback is intentional and
should be recorded as such in the evidence.
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
TC_GLOO_ADVERTISE_HOSTS=<rank0>,<rank1>,<rank2>,<rank3> \
  TC_MESH_TRAINING_EVIDENCE_PATH=/tmp/live-mesh-training.json \
  scripts/run_live_mesh_training_demo.sh
python3 scripts/check_live_mesh_training_evidence.py /tmp/live-mesh-training.json \
  --require-direct-ring --require-checkpoint --require-cuda-rank3 \
  --require-explicit-backends --require-no-backend-fallback
python3 scripts/check_live_mesh_training_evidence.py /tmp/live-mesh-training.json \
  --require-direct-ring --require-checkpoint --require-cuda-rank3 \
  --require-rank1-source-prepare \
  --require-explicit-backends --require-no-backend-fallback
python3 scripts/check_live_mesh_training_evidence.py /tmp/live-mesh-training-local.json \
  --require-direct-ring --require-checkpoint --require-local-only \
  --require-explicit-backends --require-no-backend-fallback
```

The optional evidence JSON uses schema
`tensorcore.live_mesh_training.evidence.v1` and records per-rank rendezvous,
outer-step losses, selected backend, direct-ring route counts, checkpoint
discard/realize counters, requested-vs-observed backend summaries, CUDA
fallback ranks, and per-rank launch/prepare metadata.

### `mesh_resource_scheduler.py`

Coordinates shared mesh resources such as `cosbox:cuda3090` through the
Tsotchke arbiter. It reads a small jobs JSON, probes known live work,
releases only verified-stale leases, adopts live known holders, and launches
new jobs only after claiming the selected resource. Jobs can target an exact
`resource`, a `resources` list, or a `resource_pool` selector over the checked
inventory; pool jobs expand into per-resource placements and can use command
templates such as `{resource}` and `{node}`. Tenant counts are tracked across
resources so idle machines are assigned fair-share first, then by priority.
CUDA-exclusive jobs add admission, post-start, and worker-identity gates before
a launch is considered healthy. Unknown leases and unknown liveness block
scheduling instead of killing another agent's work. When `--inventory-json` is
supplied, inventory rows with `backend: "cuda"` infer `cuda_exclusive` for
omitted job classes and reject explicit `generic` CUDA jobs before any lease can
be claimed. The checked-in default job set lives at
`configs/mesh_resource_jobs.json`; deploy that file to the live state path when
the control-plane contract changes. Checked-in running jobs must launch through
repo-owned starters; host-local systemd starts belong in paused adoption-only
rows until the unit or equivalent launcher is installed from a git checkout.

Run one dry pass:

```sh
scripts/mesh_resource_scheduler.py \
  --arbiter-cmd tsotchke-arbiter \
  --inventory-json configs/mesh_resources.json \
  --jobs-json configs/mesh_resource_jobs.json \
  --dry-run --pretty-json
```

Run the daemon loop and persist last-state evidence:

```sh
scripts/mesh_resource_scheduler.py \
  --arbiter-cmd tsotchke-arbiter \
  --inventory-json configs/mesh_resources.json \
  --jobs-json ~/.tsotchke/state/mesh-resource-jobs.json \
  --state-json ~/.tsotchke/state/mesh-resource-scheduler-state.json \
  --loop --interval-sec 30 \
  --admission-timeout-sec 10 \
  --post-start-timeout-sec 30 \
  --worker-identity-timeout-sec 10
```

Fixture coverage:

```sh
python3 scripts/mesh_resource_scheduler_selftest.py
python3 scripts/mesh_arbiter_with_inventory_selftest.py
python3 scripts/mesh_deploy_git_checkout_selftest.py
python3 scripts/check_mesh_git_access_selftest.py
python3 scripts/check_mesh_resource_preflights_selftest.py
python3 scripts/check_mesh_resource_preflight_evidence_selftest.py
python3 scripts/check_mesh_resource_config_selftest.py
python3 scripts/check_georefine_qwen_live_selftest.py
python3 scripts/check_georefine_qwen_artifact_selftest.py
python3 scripts/start_georefine_qwen_cr025_selftest.py
python3 scripts/start_qllm_olddonkey_precompute_chain_selftest.py
python3 scripts/check_mesh_resource_inventory.py
python3 scripts/check_mesh_resource_jobs.py
python3 scripts/mesh_system_audit_selftest.py
```

Use `scripts/mesh_arbiter_with_inventory.py` when arbiter capacity/status
output should be seeded from `configs/mesh_resources.json` before the scheduler
claims leases.

### `mesh_deploy_git_checkout.py`

Clones or fast-forwards a git checkout on a mesh node over SSH. Use it when
adding a new machine or rebuilding an existing one, so scheduler commands can
call repo-local scripts from `~/src/tensorcore` instead of private wrapper
paths.
The helper uploads its remote shell script without depending on stdin streaming;
if `scp` is blocked, it falls back to short chunked SSH commands.

```sh
python3 scripts/mesh_deploy_git_checkout.py \
  --target old-donkey \
  --repo-url https://github.com/tsotchke/tensorcore.git \
  --repo-dir '~/src/tensorcore' \
  --ref master \
  --require-clean \
  --json
```

Use the same command for the `computer_mesh` repo, then put its `tsotchke/bin`
directory on `PATH` for the scheduler service so `tsotchke-arbiter` resolves
from a git checkout.

Fixture coverage:

```sh
python3 scripts/mesh_deploy_git_checkout_selftest.py
```

### `check_mesh_git_access.py`

Checks that a mesh node can run `git ls-remote` against a repo without
interactive credentials. Use it before unpausing private workload rows:

```sh
python3 scripts/check_mesh_git_access.py \
  --target cosbox \
  --repo-url git@github.com:Tsotchke-Corporation/GeoRefineInternal.git \
  --resource cosbox:cuda3090 \
  --ref HEAD \
  --json
```

Pass `--resource` when using this helper as a scheduler `preflight_cmd`; the
preflight runner requires helper JSON to match the scheduler row's resource.

Fixture coverage:

```sh
python3 scripts/check_mesh_git_access_selftest.py
```

### `check_mesh_resource_preflights.py`

Runs `preflight_cmd` entries from `configs/mesh_resource_jobs.json`. By default
it checks paused rows only, excluding rows with
`metadata.preflight_default=false`. Use `--job-id` to run one of those opt-out
preflights deliberately. The JSON output records those opt-out rows in
`skipped_default_job_ids` so an empty default sweep is auditable. A passing
preflight command must emit a JSON object with `schema`, `ok: true`, and the
same `resource` as the scheduler row; a zero exit without that JSON is treated
as invalid evidence.

```sh
python3 scripts/check_mesh_resource_preflights.py \
  --job-id georefine-m2-cosbox \
  --json
```

Use `scripts/check_mesh_resource_preflight_evidence.py
--require-skipped-default-job <id>` or the operational-bundle flag
`--mesh-preflight-skipped-default-job <id>` when the desired proof is "this
default sweep intentionally did not run the opt-out row." For known external
blockers that should stay visible in the default sweep, use
`--allow-failure <job-id>:<reason>` or operational-bundle
`--mesh-preflight-allowed-failure <job-id>:<reason>` to require the row and its
exact current blocker without treating the full preflight bundle as passing.

Fixture coverage:

```sh
python3 scripts/check_mesh_resource_preflights_selftest.py
python3 scripts/check_mesh_resource_preflight_evidence_selftest.py
```

`check_mesh_resource_preflight_evidence.py` validates saved output from the
preflight runner for operational evidence bundles. Use `--require-pass` before
treating paused external rows as ready to unpause. Passing result rows must
have `rc: 0`, include the helper's nested JSON payload with `ok: true`, and
that payload's resource must match the scheduler row. `--require-pass` also
requires at least one result row.

### `start_georefine_qwen_cr025.py`

Starts the GeoRefine Qwen CR025 supervised run from a remote git checkout. It
clones or fast-forwards GeoRefine on the target host, refuses dirty checkouts by
default, preserves the CR025 `m2_supervised_run` command, and emits scheduler
JSON. Keep the scheduler row paused until the target host has the GeoRefine
Python environment installed. Use `--preflight-only --json` to verify checkout,
Python imports, calibration text, and evaluation text without launching work.
The default repo URL is the SSH GitHub remote for the private GeoRefine repo.

Fixture coverage:

```sh
python3 scripts/start_georefine_qwen_cr025_selftest.py
```

### `start_qllm_olddonkey_precompute_chain.py`

Starts the old-donkey qLLM teacher-logit chain from a remote git checkout. It
clones or fast-forwards `semiclassical_qllm` into the dedicated scheduler
checkout `/data/qllm/checkouts/semiclassical_qllm-tensorcore`, generates a
per-launch tmux chain script, and invokes `scripts/precompute_teacher_logits.py`
directly for the known shard queue. Keep the scheduler row paused until the
target host has the qLLM Python environment and phase-1 shard data installed.
Use `--preflight-only --json` to verify checkout, Python imports, `tmux`,
`nvidia-smi`, and shard files without launching the tmux chain.
The default repo URL is the SSH GitHub remote for the private qLLM repo.

Fixture coverage:

```sh
python3 scripts/start_qllm_olddonkey_precompute_chain_selftest.py
```

### `mesh_system_audit.py`

Audits the whole mesh control plane: inventory, expanded scheduler jobs,
scheduler freshness, arbiter resources, reserved/blocked policy, scheduler
lease worker identity, and optional live CUDA process ownership.

```sh
scripts/mesh_system_audit.py \
  --inventory-json configs/mesh_resources.json \
  --jobs-json ~/.tsotchke/state/mesh-resource-jobs.json \
  --scheduler-state-json ~/.tsotchke/state/mesh-resource-scheduler-state.json \
  --arbiter-cmd "scripts/mesh_arbiter_with_inventory.py --inventory-json configs/mesh_resources.json --arbiter-cmd tsotchke-arbiter -- status --json" \
  --probe-cuda
```

### `check_cuda_resource_admission.py`

Repo-owned admission gate for exclusive NVIDIA jobs. It refuses launch when
`nvidia-smi` reports unmanaged compute applications, with an optional regex
allowlist for tiny helper processes.

```sh
python3 scripts/check_cuda_resource_admission.py --resource cosbox:cuda3090
```

Fixture coverage:

```sh
python3 scripts/check_cuda_resource_admission_selftest.py
python3 scripts/check_windows_cuda_resource_admission_selftest.py
python3 scripts/check_windows_persistent_launch_selftest.py
python3 scripts/check_windows_cuda_smoke_artifact_selftest.py
python3 scripts/start_windows_cuda_smoke_selftest.py
python3 scripts/check_windows_cuda_scheduled_smoke_evidence_selftest.py
```

`check_windows_persistent_launch.py` verifies that a Windows SSH account can
create, run, mark, and delete a no-op scheduled task for durable scheduler
launches. It does not start CUDA work.

`start_windows_cuda_smoke.py` is repo-owned, but a Windows scheduler row should
stay paused on hosts where SSH child processes are torn down at session exit and
no credentialed scheduled-task or service launcher is installed.
`check_windows_cuda_scheduled_smoke_evidence.py` validates the completed
scheduler evidence; when smoke or identity flags are true, the evidence must
carry the matching `smoke_artifact` and `worker_identity` JSON objects rather
than just setting summary booleans.
The smoke executable is internally bounded to `duration_sec + 15s`, and
foreground timeout recovery is opt-in via `--recover-foreground-timeout` so a
failed foreground launch does not automatically make an extra Windows SSH probe.
Keep `--foreground` out of scheduler rows; it is a manual diagnostic fallback,
not a durable launch mode.
`check_mesh_resource_jobs.py` enforces that `windows_cuda_scheduled_smoke`
rows use the persistent-launch preflight, omit foreground recovery flags, and
keep the CUDA smoke duration at or below 5 seconds. Their post-start probe must
accept either a fresh running artifact or a completed passing artifact, because
the 3 second smoke may finish before the scheduler observes the post-start
state. Their regular `probe_cmd` must still require a live artifact, while
`completion_cmd` requires a completed passing artifact; that prevents a stale
completed smoke from being adopted as live work. The starter itself accepts a
running or completed passing artifact for scheduled-task launch success.

### `mesh_worker_identity.py`

Emits Linux worker identity JSON for `cuda_exclusive` scheduler jobs. Use it
as `worker_identity_cmd` on remote hosts to record hostname, worker PIDs,
optional user-systemd unit metadata, matching `nvidia-smi` compute
applications, and `cuda_pids`.

```sh
python3 scripts/mesh_worker_identity.py \
  --resource cosbox:cuda3090 \
  --unit qllm-phase1.service \
  --match-regex qllm.train_geometric_lm_torch \
  --require-active-unit \
  --require-matching-process \
  --require-matched-cuda
```

Fixture coverage:

```sh
python3 scripts/mesh_worker_identity_selftest.py
python3 scripts/mesh_cuda_worker_identity_selftest.py
python3 scripts/mesh_windows_worker_identity_selftest.py
```

### `check_live_mesh_training_evidence.py`

Validates the JSON artifact emitted by `run_live_mesh_training_demo.sh` via
`TC_MESH_TRAINING_EVIDENCE_PATH`. Use `--min-outer-steps`,
`--require-direct-ring`, `--require-checkpoint`, and
`--require-cuda-rank3` to enforce the operational contract for a live run.
Add `--require-explicit-backends --require-no-backend-fallback` when CUDA
fallback must be impossible for the evidence to pass.
Use `--require-local-only` for the localhost multi-rank regression mode.
Use `--require-rank1-source-prepare` when the evidence must prove rank 1
was prepared from the archived checkout during that run.

### `check_operational_evidence.py`

Validates a complete operational evidence bundle by delegating to the release,
SDK26, CUDA, HIP, HIP toolchain, PyTorch, Windows, Windows CUDA probe, mesh
preflight, and live-mesh evidence checkers, then applying bundle-level policy.
For production promotion, use the clean-head flags so stale or dirty-tree
evidence cannot satisfy the current head's deployment gate.

```sh
python3 scripts/check_operational_evidence.py \
  --release /tmp/release/release_smoke_runtime_evidence.json \
  --sdk26 /tmp/sdk26/release_smoke_runtime_evidence.json \
  --cuda /tmp/cuda-smoke.json \
  --pytorch /tmp/pytorch.json \
  --windows /tmp/windows-host.json \
  --windows-cuda /tmp/windows-cuda.json \
  --windows-cuda-smoke /tmp/windows-cuda-smoke.json \
  --mesh-preflights /tmp/mesh-preflights.json \
  --live-mesh /tmp/live-mesh-training.json \
  --require-release --require-sdk26 --require-cuda --require-pytorch \
  --require-pytorch-backend-allocation --require-windows \
  --require-windows-python --require-windows-cuda-driver \
  --require-windows-cuda-toolchain --require-windows-cuda-admission-clear \
  --require-windows-cuda-ready --require-windows-cuda-build-smoke \
  --require-windows-cuda-scheduled-smoke \
  --require-mesh-preflights --require-mesh-preflights-pass \
  --require-live-mesh \
  --require-release-clean-head --require-sdk26-clean-head \
  --require-cuda-clean-head --require-pytorch-clean-head \
  --require-windows-clean-head --require-windows-cuda-clean-head \
  --require-live-clean-head \
  --min-live-outer-steps 2 \
  --require-direct-ring --require-checkpoint --require-cuda-rank3 \
  --require-explicit-backends --require-no-backend-fallback
```

Add `--hip /tmp/hip.json --require-hip --require-hip-clean-head` on mesh
subsets that include a required HIP/chipStar accelerator host. Use
`--require-hip-build --require-hip-clean-head` when the deployment only
requires proving that chipStar/HIP compiled and initialized far enough to
emit runtime-unavailable diagnostics.
Add `--hip-toolchain /tmp/hip-toolchain.json --require-hip-toolchain` when
the deployment must prove `hipcc` plus HIP CMake config, add
`--require-hip-spirv-runtime` when the host must expose a SPIR-V-capable GPU
runtime, and add `--require-ready-hip-toolchain` when hipBLAS-ready
SPIR-V-capable GPU runtime evidence is required before scheduling work to
that host.

Add `--windows /tmp/windows-host.json --require-windows
--require-windows-clean-head` when Jack or another Windows node is part of
the deployment proof. Add `--require-windows-python` when the Windows Python
binding smoke must be present, not skipped.
Add `--windows-cuda /tmp/windows-cuda.json --require-windows-cuda-driver
--require-windows-cuda-admission-clear --require-windows-cuda-clean-head`
to prove a Windows NVIDIA lane is visible and not currently occupied; add
`--require-windows-cuda-toolchain`, `--require-windows-cuda-ready`, and
`--require-windows-cuda-build-smoke` before treating that lane as CUDA-build
ready.
Add `--windows-cuda-smoke /tmp/windows-cuda-smoke.json
--require-windows-cuda-scheduled-smoke` when a scheduler-owned Windows CUDA
smoke has produced completed evidence with matching smoke artifact and worker
identity payloads.
Add `--mesh-preflights /tmp/mesh-preflights.json --require-mesh-preflights
--require-mesh-preflights-pass` before unpausing external scheduler rows; use
`--mesh-preflight-job <job-id>` to require specific paused launchable rows in
that evidence. During paused rollout, use
`--mesh-preflight-allowed-failure <job-id>:<reason>` for an explicit known
external blocker such as Jack's `scheduled_task_did_not_run`.

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
by the live-mesh prepare step. The JSON also embeds the HIP toolchain probe
described below, so skipped HIP smoke artifacts still preserve path and
OpenCL/SPIR-V diagnostics.

```sh
TENSORCORE_HIP_SMOKE_EVIDENCE_PATH=/tmp/hip.json scripts/ci_hip_smoke.sh
python3 scripts/check_hip_smoke_evidence.py /tmp/hip.json
python3 scripts/check_hip_smoke_evidence.py /tmp/hip.json --require-hip-build
python3 scripts/check_hip_smoke_evidence.py /tmp/hip.json --require-clean-head
python3 scripts/check_hip_smoke_evidence.py /tmp/hip.json --require-toolchain

REQUIRE_HIP=1 scripts/ci_hip_smoke.sh  # fails unless HIP dispatch passes
```

### `probe_hip_toolchain.py`

Captures the chipStar/OpenCL/SPIR-V host setup without building tensorcore.
The evidence records `hipcc`, `clang`, `llvm-spirv`, `clinfo`, HIP and
hipBLAS CMake package files, OpenCL ICDs, Level Zero loader discovery,
`clinfo` GPU/SPIR-V capability summaries, and path hints for
`TC_HIP_PREFIX`, `PATH`, `CMAKE_PREFIX_PATH`, and `LD_LIBRARY_PATH`.

```sh
python3 scripts/probe_hip_toolchain.py --json /tmp/hip-toolchain.json
python3 scripts/check_hip_toolchain_evidence.py /tmp/hip-toolchain.json
python3 scripts/check_hip_toolchain_evidence.py /tmp/hip-toolchain.json \
  --require-build-toolchain --require-spirv-runtime
python3 scripts/check_hip_toolchain_evidence.py /tmp/hip-toolchain.json \
  --require-ready --require-clean-head
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
| Emit ICC-readable fallback runtime evidence | `python3 scripts/run_fallback_runtime_smoke.py --require-pass` |
| Prove the Metal library build rule | `python3 scripts/run_metallib_build_rule_evidence.py --require-pass` |
| Prove Python native packaging paths | `python3 scripts/run_python_packaging_evidence.py --require-pass` |
| Prove local distributed runtime paths | `python3 scripts/run_distributed_runtime_evidence.py --require-pass` |
| Prove local AMX and GEMM benchmark paths | `python3 scripts/run_amx_bench_evidence.py --require-pass` |
| Prove portable CPU GEMM and Conv2D helpers | `python3 scripts/run_cpu_ops_runtime_evidence.py --require-pass` |
| Prove Metal attention and Conv2D dispatch helpers | `python3 scripts/run_metal_ops_runtime_evidence.py --require-pass` |
| Probe Eshkol bridge runtime evidence | `python3 scripts/run_eshkol_tensorcore_bridge_smoke.py --build-dir build-portable-cpu-current --require-pass` |
| Pre-release wide smoke | `scripts/release_smoke.sh` (add `REQUIRE_GPU=1` if you have one) |
| Cross-check version triple | `scripts/check_version_consistency.sh` |
