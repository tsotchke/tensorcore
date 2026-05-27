# Integrating tensorcore into Eshkol

`tensorcore` exposes a C ABI (`include/tensorcore/tensorcore.h`). Eshkol-side
access happens through `eshkol/tensorcore.esk`, which declares `__tc-*` names
with Eshkol's `extern` form and maps them to the `tc_eshkol_*` helpers in
`include/tensorcore/eshkol_bridge.h`.

**Status:** the bridge is runtime-proven. The smoke compiles and runs
`eshkol/hello_tensorcore.esk` plus `eshkol/tensorcore_bridge_smoke.esk`
against `libtensorcore`, with passing evidence on the portable CPU backend.

## What the bridge does

The bridge declares Eshkol-visible builtins that thunk into flat C helpers:

| Eshkol builtin             | C entry point                  |
|----------------------------|--------------------------------|
| `__tc-init`                | `tc_eshkol_init`               |
| `__tc-shutdown`            | `tc_eshkol_shutdown`           |
| `__tc-device-*`            | `tc_eshkol_device_*`           |
| `__tc-buffer-alloc`        | `tc_eshkol_buffer_alloc`       |
| `__tc-buffer-free`         | `tc_eshkol_buffer_free`        |
| `__tc-buffer-map`          | `tc_eshkol_buffer_map`         |
| `__tc-gemm`                | `tc_eshkol_gemm`               |
| `__tc-attention-forward`   | `tc_eshkol_attention_forward`  |
| `__tc-last-backend`        | `tc_eshkol_last_backend_code`  |
| `__tc-last-backend-name`   | `tc_eshkol_last_backend`       |
| `__tc-version`             | `tc_eshkol_version`            |
| `__tc-status-string`       | `tc_eshkol_status_string`      |

Opaque handles (`tc_context*`, `tc_buffer*`, `tc_stream*`) cross the FFI as
boxed pointers. `tc_buffer_map` exposes the unified-memory pointer so Eshkol
vectors can be constructed without copy.

## How the bridge is integrated

Build tensorcore and put `eshkol/tensorcore.esk` on `ESHKOL_PATH`:

```sh
cmake --build build-portable-cpu-current --target tensorcore tensorcore_shared
export ESHKOL_PATH=~/Desktop/tensorcore/eshkol
```

Then compile/link Eshkol sources with `-ltensorcore`:

```sh
eshkol-run -I ~/Desktop/tensorcore/eshkol \
  --lib-path ~/Desktop/tensorcore/build-portable-cpu-current \
  --lib tensorcore \
  ~/Desktop/tensorcore/eshkol/hello_tensorcore.esk
```

`eshkol/bridge/tensorcore_codegen.cpp` remains as an optional raw C ABI
declaration file for compiler-side integrations, but the runtime-proven path is
the checked-in Scheme `extern` bridge.

## Calling convention

The Eshkol-side calling convention keeps the public tensorcore ABI intact while
flattening the pieces that are awkward for Eshkol's scalar/pointer FFI:

- Opaque handles cross as boxed pointers (Eshkol type `(Pointer Void)`).
- Status codes return as fixnums; the smoke only prints success markers when
  required calls return `TC_OK`.
- Diagnostics expose the canonical C ABI device name, backend name, version,
  and status renderer through pointer-returning wrappers because this bridge
  intentionally stays within Eshkol's flat pointer/scalar FFI.
- Buffer maps return host-addressable `(Pointer UInt8)` so Eshkol
  vectors are constructed in-place without copy.
- GEMM and attention descriptors are built inside the `tc_eshkol_*` C helpers,
  so Eshkol callers pass scalar shape/dtype fields directly.

The `.esk` files in this directory (`tensorcore.esk`, `hello_tensorcore.esk`)
describe the intended Eshkol-side interface and a sample program.

`scripts/run_eshkol_tensorcore_bridge_smoke.py` records the current runtime
state in `build/eshkol_tensorcore_bridge_evidence.json`. It selects
`build-portable-cpu-current` by default when that build exists, which gives
backend-independent runtime evidence; pass `--build-dir build` when you need
to validate the local Metal path. On a Metal build where the host exposes no usable Metal
device, the artifact stays `status=blocked` with `skipped_no_gpu` runtime
checks instead of reporting a bridge failure. The runner normalizes expected
no-GPU command tails in that blocked artifact, while retaining hashes, so
generic readiness scanners do not mistake the skip for a bridge failure.

## Compile evidence

`eshkol/bridge/COMPILE-EVIDENCE.txt` and
`eshkol/bridge/tensorcore_codegen.compile-verified.txt` are the snapshots
of "this builds clean at this checkpoint." Update on every bridge
surface change.

The full symbol list at the current checkpoint lives in
`eshkol/bridge/tensorcore_codegen.symbols.txt`.

## Backend selection

The bridge is linked against whichever tensorcore build directory is passed to
`eshkol-run`. Use `build-portable-cpu-current` for deterministic CI and local
runtime evidence. Use the default Metal build when validating Apple GPU
dispatch; on hosts without an initialized Metal device, the smoke records a
runtime failure instead of printing success markers.

## v0.4 — consolidation

[ROADMAP.md](../ROADMAP.md) §v0.4 moves the three Metal backends in
`eshkol-platform`, `quantum_geometric_tensor`, and `semiclassical_qllm`
onto tensorcore. The consequences:

- `eshkol-platform/lib/backend/tensor_*_codegen.cpp` emits `tc_gemm` /
  `tc_attention_forward` instead of bespoke `gpu_memory.mm` dispatch.
- `quantum_geometric_tensor/src/metal/` (45+ kernels) becomes a thin
  adapter layer over the tensorcore kernel surface.
- `semiclassical_qllm/src/backend/backend_metal.m` becomes a similar
  adapter.
- The SF64 / Ozaki-II / FP24 / FP53 precision modes move from
  `eshkol-platform/lib/backend/gpu/gpu_memory.mm` into tensorcore as
  `TC_DTYPE_SF64` / `TC_DTYPE_DF64` / `TC_DTYPE_FP24` / `TC_DTYPE_FP53`
  paths.

After v0.4: **one Metal kernel layer across the entire ecosystem.**

## v0.2 surface additions

v0.1 covers GEMM + attention + lifecycle. v0.2 widens the bridge to the
training kernels, conv, quantized, and GGUF surfaces. The naming pattern keeps
the same split: `__tc-*` names in Eshkol, `tc_eshkol_*` helpers in C, and those
helpers call the canonical `tc_*` ABI:

```
__tc-rmsnorm-forward  →  tc_eshkol_rmsnorm_forward  →  tc_rmsnorm_forward
__tc-rope-forward     →  tc_eshkol_rope_forward     →  tc_rope_forward
__tc-adamw-step       →  tc_eshkol_adamw_step       →  tc_adamw_step
__tc-conv2d-forward   →  tc_eshkol_conv2d_forward   →  tc_conv2d_forward
__tc-gemv-quantized   →  tc_eshkol_gemv_quantized   →  tc_gemv_quantized
__tc-gguf-open        →  tc_eshkol_gguf_open        →  tc_gguf_open
__tc-gguf-load-supported-tensors
                      →  tc_eshkol_gguf_load_supported_tensors
                      →  tc_gguf_load_supported_tensors
```

The v0.2 work is adding the extra C shims plus Eshkol `extern` declarations and
smoke coverage.

## See also

- [api_reference.md](api_reference.md) — full C ABI surface the bridge
  wraps.
- [architecture.md](architecture.md) — internal layering of the C side
  the FFI is fronting.
- [ROADMAP.md](../ROADMAP.md) §v0.4 — the consolidation plan.
- [../eshkol/bridge/INTEGRATION.md](../eshkol/bridge/INTEGRATION.md) —
  drop-in step-by-step bridge integration with file paths and verification
  commands, kept inside the bridge checkout so it travels with the file.
