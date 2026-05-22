# Troubleshooting

When something doesn't work, start here. Most failure modes have a
specific signature — read the matching section.

## Library load / init failures

### `dyld: Library not loaded: @rpath/libtensorcore.dylib`

The runtime linker can't find the dylib. Check, in this order:

1. `cmake --install build --prefix /opt/tensorcore` actually ran and
   `ls /opt/tensorcore/lib/libtensorcore.dylib` shows the file.
2. If you're linking via CMake, your consumer set
   `tensorcore::tensorcore_shared` as the target. The shared target carries
   an `INTERFACE_LINK_LIBRARIES` that points at the install lib dir.
3. If you're linking via pkg-config, `pkg-config --libs tensorcore` ran
   from a shell that has `PKG_CONFIG_PATH` pointing at
   `/opt/tensorcore/lib/pkgconfig`.
4. If all else fails: `export DYLD_LIBRARY_PATH=/opt/tensorcore/lib`.

### `tc_init returned TC_ERR_NO_DEVICE (-3)`

`MTLCreateSystemDefaultDevice()` returned NULL. Means:

- You're on an Intel Mac. tensorcore is arm64-only.
- You're inside a container or chroot that doesn't expose the GPU.
- You're on a CI runner without a GPU (GitHub macOS runners *do* expose
  the GPU at the simulator level; check your runner image).

Diagnostic: `system_profiler SPDisplaysDataType | grep -A 5 "Chipset Model"`.

### `metallib not found: TC_ERR_KERNEL_NOT_FOUND (-9) on first dispatch`

The library loaded but the runtime can't find `tensorcore.metallib`. The
search order is:

1. `TC_METALLIB` env var, if set.
2. `tensorcore.metallib` next to the loaded `libtensorcore.dylib`.
3. `../lib/tensorcore.metallib` relative to the dylib.
4. The build-tree path baked in at compile time (only in local builds).
5. `default.metallib` in CWD.

Fix: `export TC_METALLIB=/opt/tensorcore/lib/tensorcore.metallib`. Verify
the file exists.

### Build claims "Metal 4 enabled" but M5 path isn't taken

Three independent gates. Each must pass:

| Gate | Check |
|---|---|
| Build-time SDK gate | `xcrun --show-sdk-version` ≥ 26.0 when you ran CMake |
| Runtime family gate | `tc_device_info_get().family >= TC_FAMILY_APPLE11` |
| Runtime flag gate | `tc_device_info_get().supports_tensorops_m5 == true` (depends on macOS >= 26.0 + `-DTC_ENABLE_TENSOROPS=ON`) |

If you have an M5 and built with SDK 26.0+, but `supports_tensorops_m5`
is false at runtime, check the runtime Metal 4 report, the device name
probe, and the CMake flag. See [family_gating.md](family_gating.md).

## Wrong backend served the call

Run:

```c
tc_status_t s = tc_gemm(ctx, &d, A, B, C);
printf("backend = %s\n", tc_backend_name(tc_last_backend()));
```

If you expected `SIMDGROUP_MATRIX` and got something else:

### `MPS`

Your shape fell outside the kernel's tile coverage. The 64×64 tile expects
`M, N % 64 == 0`. The dispatch routes shapes that don't fit cleanly to
MPS. Padding to the next multiple of 64 (and slicing the result) avoids
this; v0.2 adds tail-handling inside the main kernel.

### `ACCELERATE_CPU`

The GPU path failed entirely. Check:

- Is `TC_DTYPE_F32` the only dtype you're passing? fp32 GEMM uses the GPU
  simdgroup_matrix path, not Accelerate, on Apple7+.
- Did `tc_init` succeed? If not, every subsequent dispatch falls back.
- Did the pipeline-cache compilation fail? Look for warnings on stderr at
  first dispatch.

### `NONE`

No dispatch happened yet on this thread, or `tc_last_backend` was called
before any op. Expected on a fresh thread.

## Numerical errors

### "My fp16 GEMM doesn't match my fp32 reference"

It's not supposed to. fp16 has 10 bits of mantissa; a 4096-wide
inner-product accumulates ~12 bits of rounding noise. The right comparison
is **rms_scaled error** vs an fp64 reference:

```
rms_scaled = ||y - yref|| / (||yref|| + ε)
```

fp16 GEMM at 4096³ should land at rms_scaled ≤ 5e-3. If you're seeing
that and asking whether it's right — yes, that's the bound.

### "My bf16 GEMM has higher error than fp16"

bf16 has 7 bits of mantissa vs fp16's 10. Expected. The published v0.1.3
fallback path lands ~2.7e-3 rms_scaled at 256³; the larger shapes scale
similarly (error grows ~ sqrt(K)).

### "fp32 GEMM differs from Accelerate by an LSB or two"

fp32 should be bit-exact against `cblas_sgemm`. If you see any difference
at all, you've found a bug — please file an issue with the shape and
descriptor. Cross-check by inspecting `tc_last_backend()`; if it's not
`SIMDGROUP_MATRIX`, you took a different path.

### "My int8 GEMM result is wrong"

Two likely causes:

1. You set `accum_dtype = TC_DTYPE_I8`. It must be `TC_DTYPE_I32`.
2. K exceeds 2^16 and the i8×i8 product overflows the fp32-widen fallback
   on Apple7..9. Mostly a theoretical case but worth noting.

## Build issues

### `Xcode 17 rejects __asm in gemm_async.metal`

This is the gate documented in `CMakeLists.txt` lines ~109-124. The build
should automatically skip the async kernels on SDK ≥ 26.0. If you see the
error anyway, your SDK detection misfired:

```sh
xcrun --show-sdk-version
```

If that prints `26.0` or later, but CMake still tried to compile the
async kernels, your `CMakeCache.txt` is stale. Wipe `build/` and
re-configure.

### `find_library(METAL_FRAMEWORK ...) failed`

The Xcode command-line tools aren't installed, or the SDK path is set to
something that doesn't have Metal:

```sh
xcode-select --print-path
xcode-select --install        # if needed
```

### `cmake --install` doesn't include the metallib

Make sure the metallib build target finished before you ran install. The
file appears at `build/tensorcore.metallib`; if `ls build/` doesn't show
it, `cmake --build build` failed on the metal compile.

Look for `metal: error:` lines in the build output. If a single kernel
fails to compile, the metallib is not produced and runtime dispatch will
return `TC_ERR_KERNEL_NOT_FOUND`.

## Python binding issues

### `import tensorcore` raises `TensorcoreError: cannot find libtensorcore.dylib`

Set `TENSORCORE_LIB` to the absolute path of the dylib:

```sh
export TENSORCORE_LIB=/opt/tensorcore/lib/libtensorcore.dylib
```

Or `cmake --install build --prefix /opt/tensorcore` and the binding will
find it under the standard prefix.

### `ctypes.ArgumentError: argument 4: <class 'TypeError'>`

You passed a NumPy array where the binding expects a `tc_buffer*`. Allocate
a buffer (`tc.buffer_alloc`) and then `tc.buffer_write(buf, your_array)`.

### Python tests pass but I get garbage results

Did `cmake --build build` finish *after* you last edited a kernel? The
metallib needs to be rebuilt; ctypes loads the dylib but it loads the
metallib lazily, and if `build/tensorcore.metallib` is stale you'll see
wrong-result symptoms instead of compile errors.

## GGUF issues

### `tc_gguf_open returned TC_ERR_INVALID_ARG`

The file isn't a valid GGUF v3 file. Diagnostics:

```sh
hexdump -C model.gguf | head -1
# should start with 47 47 55 46 = "GGUF"
```

If the magic is wrong, you have a ggml-v2 file or a corrupt download. If
the magic is right, your version may be older than v3.

### `tc_gguf_load_supported_tensors` reports many skipped

The model uses k-quants (Q4_K_M, Q5_K_M, etc.) or another format we don't
handle in v0.1. List the tensor types:

```c
for (uint64_t i = 0; i < tc_gguf_tensor_count(gguf); ++i) {
    tc_gguf_tensor_info info;
    tc_gguf_tensor_at(gguf, i, &info);
    printf("%s  type=%d\n", info.name, info.type);
}
```

`type = TC_GGUF_TYPE_UNSUPPORTED = -1` is what gets skipped. v0.2 adds the
k-quant family.

### "I loaded a model but inference outputs nonsense"

Sanity-check the matrix descriptor. GGUF stores matrices as `[K, N]`;
the kernel takes `N, K`. Don't transpose manually — use
`tc_gguf_loaded_tensor_quantized_matrix_info` to derive `N, K` correctly.

## Distributed issues

### `tc_dist_init(TC_DIST_RING, world_size=4, ...)` returns `TC_ERR_UNSUPPORTED_FAMILY`

The multi-Mac ring backend isn't implemented in v0.1. `TC_DIST_SINGLE`
with `world_size=1` is functional in every build, and default Apple plus
portable CPU builds support `TC_DIST_GLOO` over TCP. The standalone
single-host ring algorithm is validated in `tests/test_distributed_ring_fork.c`;
`TC_DIST_GLOO` also uses TCP ring sockets for fp32 SUM at `world_size >= 3`.
The TB5/JACCL multi-Mac `TC_DIST_RING` backend lands in v0.5.

### `tc_dist_init(TC_DIST_GLOO, ...)` fails or hangs

Use a rendezvous URL like `gloo+tcp://127.0.0.1:29500` for local tests or
`gloo+tcp://host0:port` across hosts, start every rank with the same URL,
and make sure the port is reachable. `TC_DIST_GLOO` is wired in default
Apple and portable CPU builds; other builds return explicit unsupported or
internal errors when the POSIX TCP transport is not available.

### Forked-rank test hangs

The fork test uses socketpair-blocking reads. If one rank dies, the
others wait forever. Run with `ctest --output-on-failure` and look at
which `rank X failed:` line came out.

## When all else fails

1. Re-run `ctest --test-dir build --output-on-failure` to confirm the
   library itself is healthy.
2. Run `bench/bench_gemm` and check that `backend=simdgroup_matrix` shows
   up — that's the "library is reaching the GPU correctly" check.
3. Capture the shape, dtype, descriptor, and `tc_last_backend()` value for
   the failing call.
4. Open an issue at the project repo with the above plus your chip
   (`tc_device_info.name`), macOS version, and `xcrun --show-sdk-version`.
