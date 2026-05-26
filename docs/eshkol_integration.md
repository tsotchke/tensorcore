# Integrating tensorcore into Eshkol

`tensorcore` exposes a C ABI (`include/tensorcore/tensorcore.h`). Eshkol-side
access happens through a small bridge file (`eshkol/bridge/tensorcore_codegen.cpp`)
that registers `__tc-*` builtins inside the Eshkol codegen.

**Status:** the bridge is functional. v0.1.4 dropped it into
`eshkol-platform/lib/backend/` (auto-globbed by their CMakeLists) and
v0.1.5 mirrored the integration into the canonical `~/Desktop/eshkol/`
checkout. Both build clean; the REPL is identical with and without the
opt-in env flag.

## What the bridge does

The bridge registers Eshkol builtins that thunk into the tensorcore C ABI:

| Eshkol builtin             | C entry point                  |
|----------------------------|--------------------------------|
| `__tc-init`                | `tc_init`                      |
| `__tc-shutdown`            | `tc_shutdown`                  |
| `__tc-device-info`         | `tc_device_info_get`           |
| `__tc-buffer-alloc`        | `tc_buffer_alloc`              |
| `__tc-buffer-free`         | `tc_buffer_free`               |
| `__tc-buffer-map`          | `tc_buffer_map`                |
| `__tc-gemm`                | `tc_gemm`                      |
| `__tc-attention-forward`   | `tc_attention_forward`         |
| `__tc-last-backend`        | `tc_last_backend`              |
| `__tc-version`             | `tc_version`                   |
| `__tc-status-string`       | `tc_status_string`             |

Opaque handles (`tc_context*`, `tc_buffer*`, `tc_stream*`) cross the FFI as
boxed pointers. `tc_buffer_map` exposes the unified-memory pointer so Eshkol
vectors can be constructed without copy.

## How the bridge is integrated

Drop the bridge file into the eshkol-side backend dir:

```sh
cp eshkol/bridge/tensorcore_codegen.cpp \
   ~/Desktop/eshkol-platform/lib/backend/
```

The platform's `CMakeLists.txt` globs `lib/backend/*.cpp`, so it picks the
file up automatically — no list edit needed. Activation is opt-in via:

```sh
export ESHKOL_ENABLE_TENSORCORE=1
```

When set, the one-line addition in the eshkol-platform codegen-context
initialization path declares the `tc_*` symbols as `ExternalLinkage` and
the `__tc-*` builtins resolve to them. When unset, the bridge file
compiles but registers nothing — `eshkol-static` builds identically to a
pre-bridge build.

Verification (v0.1.4 → v0.1.5):

- `eshkol-static` builds 100% clean in both modes.
- REPL `(+ 1 2) → 3` identically in both modes.
- Mirrored into both `~/Desktop/eshkol/` (main) and
  `~/Desktop/eshkol-platform/` (active development); separate commits.

## Calling convention

The Eshkol-side calling convention mirrors the C ABI exactly:

- Opaque handles cross as boxed pointers (Eshkol type `(Pointer Void)`).
- Status codes return as fixnums; host wrappers raise on non-`TC_OK`.
- Buffer maps return host-addressable `(Pointer UInt8)` so Eshkol
  vectors are constructed in-place without copy.
- Descriptor structs are passed by reference — Eshkol wrappers
  build them on the stack before the call.

The `.esk` files in this directory (`tensorcore.esk`, `hello_tensorcore.esk`)
describe the intended Eshkol-side interface and a sample program.

`scripts/run_eshkol_tensorcore_bridge_smoke.py` records the current runtime
state in `build/eshkol_tensorcore_bridge_evidence.json`. Until the Eshkol-side
`__tc-*` wrappers resolve to the native `tc_*` declarations, that evidence is
expected to be `status=blocked`; use `--require-pass` only when promoting the
bridge to a real runtime-proven path.

## Compile evidence

`eshkol/bridge/COMPILE-EVIDENCE.txt` and
`eshkol/bridge/tensorcore_codegen.compile-verified.txt` are the snapshots
of "this builds clean at this checkpoint." Update on every bridge
surface change.

The full symbol list at the current checkpoint lives in
`eshkol/bridge/tensorcore_codegen.symbols.txt`.

## Why opt-in

The bridge is opt-in for two reasons:

1. **Reversibility.** Eshkol-platform is shared with other backends
   (CUDA, ROCm, CPU). The env-flag activation means tensorcore is a
   build-time-optional addition; users on non-Apple hardware never see
   it. Set `ESHKOL_ENABLE_TENSORCORE=0` or remove the bridge file and the
   build is back to the previous state.
2. **Staged rollout.** Once we're confident the bridge stays clean
   across eshkol-platform refactors, the default flips. Today the
   conservative stance is explicit opt-in for testing.

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
training kernels, conv, quantized, and GGUF surfaces. The naming pattern
follows the C ABI:

```
__tc-rmsnorm-forward  →  tc_rmsnorm_forward
__tc-rope-forward     →  tc_rope_forward
__tc-adamw-step       →  tc_adamw_step
__tc-conv2d-forward   →  tc_conv2d_forward
__tc-gemv-quantized   →  tc_gemv_quantized
__tc-gguf-open        →  tc_gguf_open
__tc-gguf-load-supported-tensors
                      →  tc_gguf_load_supported_tensors
```

The bridge file already declares these symbols at `ExternalLinkage`; the
v0.2 work is just the Eshkol-side builtin registration plus the type
declarations in `tensorcore.esk`.

## See also

- [api_reference.md](api_reference.md) — full C ABI surface the bridge
  wraps.
- [architecture.md](architecture.md) — internal layering of the C side
  the FFI is fronting.
- [ROADMAP.md](../ROADMAP.md) §v0.4 — the consolidation plan.
- [../eshkol/bridge/INTEGRATION.md](../eshkol/bridge/INTEGRATION.md) —
  drop-in step-by-step bridge integration with file paths and verification
  commands, kept inside the bridge checkout so it travels with the file.
