# Development setup

Zero to "all tests pass" on a fresh Mac. Or on a Linux box if you only
want the portable CPU backend.

## Apple Silicon (the primary path)

### 1. Xcode command-line tools

```sh
xcode-select --install
xcrun --show-sdk-version   # should be 14.0+ at minimum; 26.0+ unlocks Metal 4
```

If you're on macOS 26 (Tahoe) and the SDK version reports `14.x`, run
`sudo xcode-select -s /Applications/Xcode.app` so the full Xcode is the
selected toolchain — the CLT-only path doesn't ship the macOS 26 SDK.

### 2. CMake (3.20+)

```sh
brew install cmake
cmake --version
```

### 3. Python (3.10+) and NumPy

For the `python_basic` CTest target and the Python binding:

```sh
brew install python@3.13   # or whatever 3.10+ your Brew has
python3 -m pip install --upgrade pip
python3 -m pip install numpy
```

If you prefer pyenv / asdf / system Python, any 3.10+ works. NumPy is
the only runtime dependency.

### 4. Clone + build + test

```sh
git clone https://github.com/tsotchke/tensorcore.git
cd tensorcore
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build --output-on-failure
```

Expected: **24/24 tests pass** in roughly 5-15 seconds on a local Apple
build.

If `ctest` shows fewer than 24 tests, your Python or NumPy may not be
visible to CMake -- `python_basic` skips, dropping the count by one.

### 5. Quick smoke

```sh
./build/examples/hello_gemm
```

Expected first line: `[tensorcore] loaded metallib: …/build/tensorcore.metallib`.
Then `[tensorcore] device="Apple M2 Ultra"` (or whatever chip), then
`tc_gemm: ok  backend=simdgroup_matrix`. If you see
`backend=simdgroup_matrix`, your fast path is wired correctly.

### 6. The bench (optional)

```sh
./build/bench/bench_gemm
./build/bench/bench_attention
./build/bench/bench_inference_7b
```

See [benchmarks.md](benchmarks.md) for expected numbers per chip.

## Non-Apple platforms (portable CPU only)

For Linux CPU nodes or Intel Macs participating in a hybrid mesh:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTC_ENABLE_METAL=OFF
cmake --build build -j8
ctest --test-dir build --output-on-failure
```

This builds only the portable CPU backend: pure C/C++17, no Metal, no
Apple frameworks. The C ABI is identical to the Metal-enabled build;
ops that still depend on an external backend return
`TC_ERR_UNSUPPORTED_FAMILY` cleanly.

What works on the portable CPU build:

- `tc_init` / `tc_shutdown` / `tc_device_info_get`
- `tc_buffer_alloc` / `tc_buffer_free` / `tc_buffer_map` / `tc_buffer_size`
- `tc_stream_create` / `tc_stream_sync` / `tc_stream_destroy`
- `tc_gemm` (all dtypes, transpose flags, batched, async)
- `tc_attention_forward` / `tc_attention_backward`
- `tc_rmsnorm_*`, `tc_layernorm_*`, `tc_rope_*`, `tc_swiglu_*`,
  `tc_softmax_*`, `tc_adamw_step`, `tc_fused_rmsnorm_gemv`,
  `tc_fused_layernorm_gemv`
- `tc_conv2d_forward`, `tc_conv2d_backward_input`,
  `tc_conv2d_backward_weight`
- `tc_quantize_weights` / `tc_gemv_quantized` (Q4_0, Q8_0)
- `tc_gguf_*` (full reader surface)
- `tc_dist_*` with `TC_DIST_SINGLE` backend (`world_size=1` no-ops)
- `tc_dist_*` with `TC_DIST_GLOO` on default Apple and portable CPU builds:
  TCP rendezvous,
  fp32 SUM/AVG/MIN/MAX all-reduce, fp16 SUM/AVG all-reduce, byte-level
  broadcast, allgather, and barrier
- DiLoCo single-rank, dense multi-rank, and sparse TOPK multi-rank outer
  steps over `TC_DIST_GLOO`
- opt-in CPU GEMM variants via `TC_USE_AVX2_GEMM=1`,
  `TC_USE_NEON_GEMM=1`, and `TC_USE_AMX_GEMM=1`; the portable CI script
  smokes these in isolated Python subprocesses
- sparse top-k compression helpers
- memory-tier stub baseline and portable CPU activation-checkpointing
  discard/realize
- HIP and CUDA backend diagnostics with deterministic unsupported stubs
- `tc_status_string` / `tc_dtype_name` / `tc_backend_name`

What doesn't (returns `TC_ERR_UNSUPPORTED_FAMILY`):

- `tc_dist_*` with `TC_DIST_RING`
- `tc_dist_*` with `TC_DIST_GLOO` bf16/int8 reductions and public generic
  sparse packed wire-format APIs
- HIP/chipStar execution
- CUDA execution
- DiLoCo dropout tolerance and non-shipped compression modes

`tc_last_backend()` reports `portable_cpu` for every served call on
this build.

## Optional CUDA / HIP Scaffolding

The direct NVIDIA CUDA and chipStar HIP backends are opt-in build paths:

```bash
cmake -S . -B build-cuda -DTC_ENABLE_METAL=OFF -DTC_ENABLE_CUDA=ON
cmake -S . -B build-hip  -DTC_ENABLE_METAL=OFF -DTC_ENABLE_HIP=ON
```

`TC_ENABLE_CUDA=ON` requires CMake's `CUDAToolkit` package with
`CUDA::cudart` and `CUDA::cublas`. `TC_ENABLE_HIP=ON` requires HIP runtime
and hipBLAS imported targets. If those dependencies are missing, configure
prints a warning and falls back to the deterministic unsupported stubs used
by default builds. Installed CMake packages preserve the effective backend
flags and rediscover CUDA/HIP dependencies before loading exported targets.

The portable build does not build or install `tensorcore.metallib`.
Consumers should treat the metallib as a Metal-backend artifact and skip
that lookup when `TC_ENABLE_METAL=OFF`.

## IDE integration

`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` is on by default (see
`CMakeLists.txt:41`). After `cmake -B build`, a `compile_commands.json`
sits at `build/compile_commands.json`. Symlink or copy to the repo
root if your editor needs it there:

```sh
ln -sf build/compile_commands.json .
```

clangd, VSCode's C/C++ extension, and Xcode's bridge support all read
this file. Metal files use clangd's `metal` mode (the file extension is
enough).

For Python development, point your IDE at `python/tensorcore/__init__.py`
(the binding) and `python/tests/test_basic.py` (the test suite).

## Running the deep smoke (release_smoke.sh)

The deepest pre-release check. ~30 seconds on M2 Ultra:

```sh
cmake --install build --prefix /tmp/tensorcore-install
REQUIRE_GPU=1 scripts/release_smoke.sh
```

This builds the wheel, the native SDK archive, verifies the install via
CMake `find_package` and pkg-config, runs the full ctest, and writes a
JSON evidence file to `build/release_smoke_runtime_evidence.json`. See
[ci_and_scripts.md § release_smoke.sh](ci_and_scripts.md) for the
schema.

## Common environment knobs

Build-time:

```sh
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTC_ENABLE_METAL=ON       # OFF for portable CPU
  -DTC_ENABLE_TENSOROPS=ON   # M5 mpp::tensor_ops dispatch (SDK 26+)
  -DTC_BUILD_TESTS=ON
  -DTC_BUILD_BENCH=ON
  -DTC_BUILD_EXAMPLES=ON
```

Runtime:

```sh
export TENSORCORE_LIB=/opt/tensorcore/lib/libtensorcore.dylib
export TC_METALLIB=/opt/tensorcore/lib/tensorcore.metallib
export TC_USE_128_TILE=1     # opt into 128×128 GEMM tile (regresses on M2)
export TC_Q4_USE_V1=1        # revert to original Q4_0 GEMV kernel
```

See [observability.md](observability.md) for the complete list.

## When `make` is enough

The repo ships a developer Makefile. From the repo root:

```sh
make build         # cmake -B build && cmake --build build -j
make test          # the 22 ctest cases
make bench         # GEMM + attention + 7B Q4_0 benches
make hello         # ./build/examples/hello_gemm
make decode        # ./build/examples/decode_step
make train         # ./build/examples/training_step
make smoke         # release_smoke.sh with REQUIRE_GPU=1
make install       # cmake --install to /tmp/tensorcore-install
make wheel         # build + install + reimport the wheel
make check-version # version triple consistency
make docs-check    # docs/ link integrity
make icc-audit     # ICC index + doc-coverage + shell-hardening
make all           # build + test
```

See `make help` for the full menu.

## What you don't need

- **MLX.** Optional; tensorcore is independent. If you want to compare
  fp16 GEMM TFLOPS against MLX, install separately:
  `pip install mlx`.
- **PyTorch / TensorFlow.** Tensorcore's core build only needs the C ABI.
  An experimental PyTorch bridge lives in `bindings/pytorch` for fp32/bf16
  CPU matmul and opt-in `torch.matmul` dispatch experiments. It uses
  zero-copy tensor wrappers when the runtime accepts the tensor pointer and
  falls back to staged buffers otherwise.
- **CUDA.** Obviously.
- **Real GGUF model.** The bench harness uses synthetic Q4_0 weights;
  full inference against a real GGUF is a v0.2 deliverable.
- **GitHub auth.** Cloning is anonymous; only the `gh` CLI in
  [release_process.md](release_process.md) needs auth.

## Troubleshooting first build

If `cmake -B build` fails with `find_library(METAL_FRAMEWORK)` errors,
your Xcode CLT install is incomplete:

```sh
sudo xcode-select -s /Applications/Xcode.app
xcrun --show-sdk-path   # should print a path under /Applications/Xcode.app
```

If `cmake --build build` fails on `.metal` compilation with
`error: unknown intrinsic 'air.…'`, your Xcode is 17+ and you have stale
CMake cache from an earlier 16.x configure. Wipe and reconfigure:

```sh
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

If `ctest` shows `python_basic` failing with a NumPy error, run with
`PYTHONPATH=$PWD/python TENSORCORE_LIB=$PWD/build/libtensorcore.dylib
python3 python/tests/test_basic.py` directly to see the error.

See [troubleshooting.md](troubleshooting.md) for runtime issues after
the build succeeds.

## See also

- [integrating_tensorcore.md](integrating_tensorcore.md) — once tensorcore
  works locally, this is how you wire it into another project.
- [ci_and_scripts.md](ci_and_scripts.md) — what the CI workflows + helper
  scripts do.
- [release_process.md](release_process.md) — from version bump to
  published wheel.
- [observability.md](observability.md) — runtime introspection knobs.
