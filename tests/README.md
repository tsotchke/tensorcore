# tests/

29 default CTest entries cover every public ABI surface: 26 native
correctness tests, the Python binding smoke, and two executable example
smokes in the Metal build. The portable
CPU-only build registers `test_portable_cpu.c`, `test_conv2d.c`,
`test_diloco.c`, `test_sparse_compress.c`, `test_gloo_fork.c`, and
`test_gloo_ring_fork.c`, plus `test_checkpoint.c` and the
DiLoCo-over-GLOO fork tests. Each native test is a single `.c` file (or
`.mm` for the buffer pool, which needs ObjC++).
Numerical tests compare against an fp64 CPU reference or a bit-exact CPU
oracle and pass the tolerances documented in
[../docs/numerics.md](../docs/numerics.md).

```sh
ctest --test-dir build --output-on-failure   # 29 default Apple tests
```

Runs in ~5-15s on M2 Ultra.

With `TC_ENABLE_METAL=OFF`, `test_portable_cpu.c` covers the portable
buffer/device path plus padded f32/f16 GEMM, batched GEMM, i8 GEMM,
quantized GEMV, `TC_DIST_SINGLE` collectives, memory-tier and
checkpoint baseline APIs, HIP/CUDA inactive diagnostics, local DiLoCo, and
the localhost GLOO TCP collective and DiLoCo-over-GLOO smokes. The
portable build also runs Conv2D, DiLoCo, sparse-compression, broker GLOO TCP,
opt-in ring GLOO TCP, activation checkpointing, DiLoCo-over-GLOO, and sparse
TOPK DiLoCo-over-GLOO tests.
`scripts/ci_portable_cpu.sh` adds installed SDK consumer checks plus
subprocess smokes for the opt-in AVX2, NEON, and AMX GEMM environment
variants.

## Test inventory

| # | Test | Coverage |
|---:|---|---|
| 1 | `test_device.c` | `tc_init`, `tc_shutdown`, `tc_device_info_get`; basic buffer alloc/free/map |
| 2 | `test_gemm_f16.c` | fp16 GEMM at multiple shapes vs fp64 reference (rms_scaled ≤ 5e-3) |
| 3 | `test_gemm_f32.c` | fp32 GEMM **bit-exact** vs `cblas_sgemm` |
| 4 | `test_gemm_bf16.c` | bf16 GEMM; native on Apple9+, fp32-cast fallback on Apple7..8 |
| 5 | `test_gemm_i8.c` | int8 GEMM; native on Apple10+, fp32-widen fallback on Apple7..9; bit-exact for K ≤ 2^16 |
| 6 | `test_attention_correctness.c` | FlashAttention forward: causal, GQA (3 cases), sliding window, ALiBi |
| 7 | `test_attention_backward.c` | FlashAttention backward at D=64 and D=128 vs numerical-differences reference |
| 8 | `test_training_kernels.c` | RMSnorm fwd+bwd, LayerNorm fwd+bwd, RoPE fwd+bwd, SwiGLU fwd+bwd, softmax fwd+bwd, AdamW |
| 9 | `test_transformer_block.c` | Full forward + backward of one Llama-style block at small shapes |
| 10 | `test_e2e_training.c` | A few iterations of forward + backward + AdamW; checks parameter convergence |
| 11 | `test_conv2d.c` | Conv2D forward + dInput + dWeight (multi-batch validated) |
| 12 | `test_distributed_ring.c` | Single-host ring all-reduce via threads — bit-exact across 4 ranks × 1024 fp32 |
| 13 | `test_quantized.c` | Q4_0 sync + async, Q4_0 tail N, Q8_0 GPU quantize + GEMV, invalid-quant sizing |
| 14 | `test_fused_norm_gemv.c` | fused RMSNorm/LayerNorm GEMV against separate norm-forward + `tc_gemm` paths |
| 15 | `test_distributed_ring_fork.c` | Ring all-reduce via `fork()` + socketpairs — **same transport pattern** v0.5 TB5 will use |
| 16 | `test_gguf.c` | Synthetic GGUF round-trip, metadata, bulk load, skip-unsupported count, Q4 GEMV from GGUF |
| 17 | `test_tensorops_select.c` | M5 TensorOps dtype × accum selector (works without M5 hardware) |
| 18 | `test_tensorops_runtime.c` | TensorOps runtime path coverage (skips politely on non-M5) |
| 19 | `test_diloco.c` | Local/single-rank DiLoCo outer steps, counters, and unsupported multi-rank guards |
| 20 | `test_sparse_compress.c` | DiLoCo top-k sparse compression pack/unpack accuracy and merge behavior |
| 21 | `test_gloo_fork.c` | Four forked ranks over broker GLOO TCP; fp32/fp16 allreduce, broadcast, allgather, barrier |
| 22 | `test_diloco_gloo_fork.c` | Multi-rank DiLoCo over GLOO with forked localhost ranks |
| 23 | `test_diloco_sparse_fork.c` | TOPK sparse DiLoCo over GLOO; validates sparse wire-byte reduction |
| 24 | `test_gloo_ring_fork.c` | Four forked ranks with `TC_GLOO_RING=1`; direct TCP ring fp32 SUM |
| 25 | `test_checkpoint.c` | CPU discard/realize checkpoint lifecycle; skips on Metal until handle-preserving MTLBuffer discard lands |
| 26 | `test_buffer_pool.mm` | LIFO recycling, bucket size classes, concurrent allocate/free |
| 27 | `python_basic` | The Python binding's `tests/test_basic.py` — full ABI surface exercised from ctypes |
| 28 | `example_decode_step` | Native decode-step smoke using the installed C ABI |
| 29 | `example_training_step` | Native training-step smoke using the installed C ABI |

## Tolerances

All tests use the `rms_scaled` error metric:

```
rms_scaled = ||Y - Y_ref|| / (||Y_ref|| + ε)
```

per [../docs/numerics.md](../docs/numerics.md). Bit-exact tests are
called out specifically (fp32 GEMM, int8 GEMM with i32 accum, ring
all-reduce).

## Skip semantics

Each test can skip itself politely if the runtime can't exercise the
path:

- **No real GPU** (paravirtual runner): all GPU-only tests skip with a
  printed "no Metal device" message; only the no-device subset runs.
- **Apple7..8 + bf16**: was pre-v0.1.3 skipped; now runs via the
  fp32-cast fallback path. Passes everywhere.
- **Apple7..9 + int8**: was pre-v0.1.3 skipped; now runs via fp32-widen
  fallback. Bit-exact everywhere.
- **Apple7..10 + Metal 4 TensorOps**: `test_tensorops_runtime` skips
  with a "TensorOps not available" message; `test_tensorops_select`
  always runs (selector logic is host-side).
- **No Python or NumPy installed**: `python_basic` is excluded from
  CTest at configure time.

`scripts/ci_macos_test.sh` runs the no-device / paravirtual subset on
CI runners that don't expose a real Metal device.

## Numerical references

Tests build their references in fp64 on the CPU, then compare against
the fp16 / bf16 / int8 GPU output. This is **not** comparing GPU vs CPU
of the same dtype — it's comparing GPU low-precision vs CPU high-
precision, which is what catches accumulation bugs and dtype mistakes.

The Q4_0 reference is special: the kernel and the CPU reference both
dequantize the same blocks the same way, so the comparison validates the
GEMV path against a naive dequant-then-multiply. Quantization itself
isn't tested here — see `tests/test_quantized.c::test_quantize_q4_0`
for that.

## Adding a test

See [../CONTRIBUTING.md § Adding a kernel](../CONTRIBUTING.md#adding-a-kernel)
step 5-6. Pattern:

```c
/* tests/test_<group>.c */
#include "tensorcore/tensorcore.h"
#include <stdio.h>

int main(void) {
    tc_context* ctx;
    tc_init(&ctx);

    /* 1. Allocate inputs */
    /* 2. Initialize with a fixed seed (`srand(seed)`) */
    /* 3. Compute reference on host in fp64 */
    /* 4. Compute on GPU via tc_<op> */
    /* 5. Compute rms_scaled error */
    /* 6. Assert below tolerance */

    tc_shutdown(ctx);
    return 0;
}
```

Register in `tests/CMakeLists.txt`:

```cmake
add_executable(test_mything test_mything.c)
target_link_libraries(test_mything PRIVATE tensorcore)
target_include_directories(test_mything PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_mything COMMAND test_mything)
```

## See also

- [../docs/numerics.md](../docs/numerics.md) — the tolerance contract
  every test enforces.
- [../docs/codebase_audit.md](../docs/codebase_audit.md) — ICC's view
  of test coverage.
- [../scripts/ci_macos_test.sh](../scripts/ci_macos_test.sh) — what the
  CI does on a real-vs-paravirtual GPU.
