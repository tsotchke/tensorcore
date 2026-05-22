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

### `ci_portable_cpu.sh`

Builds with `TC_ENABLE_METAL=OFF`, runs the portable CTest suite,
installs the native SDK, verifies CMake and pkg-config consumers, then
runs an inline Python smoke against the installed shared library. The
Python phase covers `TC_DIST_GLOO` and DiLoCo-over-GLOO with two forked
localhost ranks, and runs isolated subprocess GEMM smokes with
`TC_USE_AVX2_GEMM=1`, `TC_USE_NEON_GEMM=1`, and `TC_USE_AMX_GEMM=1`.

AMX uses reverse-engineered Apple instructions, so the AMX subprocess
treats SIGILL as a skip by default instead of taking down the whole smoke
on hosts that block the instruction. Set `REQUIRE_AMX_GEMM=1` to require
the AMX opt-in path on local Apple-Silicon verification machines.

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
   (chip name, family, capability flags, backend per representative call,
   version triple).

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
