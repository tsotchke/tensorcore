# Integrating tensorcore into Eshkol

`tensorcore` exposes a C ABI (`include/tensorcore/tensorcore.h`). Eshkol-side
access happens through a small bridge file in `eshkol-platform/lib/ffi/` that
registers `__tc-*` builtins.

Status (v0.1): bridge file pending. The `.esk` files in this directory describe
the intended surface and become functional once the bridge ships.

## What the bridge does

The bridge file (`eshkol-platform/lib/ffi/tensorcore_ffi.cpp`) registers Eshkol
builtins that thunk into the tensorcore C ABI:

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

Opaque handles (`tc_context*`, `tc_buffer*`, `tc_stream*`) cross the FFI as
boxed pointers. `tc_buffer_map` exposes the unified-memory pointer so Eshkol
vectors can be constructed without copy.

## Endgame

Once the bridge is in place, the three Metal backends in `eshkol-platform`,
`quantum_geometric_tensor`, and `semiclassical_qllm` can all be replaced by
calls into tensorcore. The `tensor_*_codegen.cpp` files in eshkol-platform
should emit calls to the `tc_gemm`/`tc_attention_forward` ABI instead of
their bespoke `gpu_memory.mm`. See `ROADMAP.md` phase 5.
