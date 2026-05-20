# Changelog

## v0.1.0 — Foundation

### Kernels (Metal)
- `gemm_simdgroup.metal`: 64×64 GEMM, BK=32, vec4 cooperative loads, fp16/bf16/fp32/i8 with fp32 accumulators
- `gemm_simdgroup_128.metal`: 128×128 large-tile variant (opt-in via `TC_USE_128_TILE=1`)
- `flash_attention.metal`: fused FA-2 forward, D=64
- `flash_attention_d128.metal`: fused FA-2 forward, D=128
- `flash_attention_backward.metal`: split-kernel dQ + dK/dV backward (D=64)
- `training_kernels.metal`: RMSnorm fwd+bwd, LayerNorm fwd+bwd, RoPE fwd, SwiGLU fwd+bwd, softmax fwd+bwd, fused AdamW step
- `conv2d.metal`: im2col + bias-add (forward, via tc_gemm)
- `tensorops_gemm.metal`: Metal 4 `mpp::tensor_ops::matmul2d` path (SDK 26+, M5 Neural Accelerator)
- `tensorops_flash_attention.metal`: Metal 4 FlashAttention skeleton (SDK 26+, validation pending M5)

### Public C ABI
- Lifecycle: `tc_init`, `tc_shutdown`, `tc_device_info_get`, `tc_version`
- Buffers: `tc_buffer_alloc`, `tc_buffer_free`, `tc_buffer_map`, `tc_buffer_size`
- Streams: `tc_stream_create`, `tc_stream_destroy`, `tc_stream_sync`
- GEMM: `tc_gemm`, `tc_gemm_async`, `tc_gemm_batched`
- Attention: `tc_attention_forward`, `tc_attention_forward_async`, `tc_attention_backward`
- Training: `tc_rmsnorm_forward`/`_backward`, `tc_layernorm_forward`/`_backward`, `tc_rope_forward`, `tc_swiglu_forward`/`_backward`, `tc_softmax_forward`/`_backward`, `tc_adamw_step`
- Conv: `tc_conv2d_forward`
- Distributed: `tc_dist_init`, `tc_dist_finalize`, `tc_allreduce`, `tc_broadcast`, `tc_allgather`, `tc_barrier` (single-host backend live; ring TB5 + Gloo gated for v0.5)
- Diagnostics: `tc_last_backend`, `tc_backend_name`, `tc_status_string`, `tc_dtype_name`

### Runtime
- `lib/core/device.mm`: Apple GPU family detect (Apple7..Apple11) + unified-memory probe
- `lib/core/pipeline_cache.mm`: thread-safe `MTLComputePipelineState` cache, function-constant specialization
- `lib/core/buffer_pool.mm`: power-of-2 bucketed MTLBuffer pool (LIFO recycle, 8/bucket cap)
- `lib/core/autotune.cpp`: family-keyed tile selection + cache load/save
- `lib/tensorops/tensorops_m5.mm`: Metal 4 host dispatch (SDK-gated)
- `lib/distributed/distributed.mm`: single-host backend; TB5 ring + Gloo stubs

### Fallbacks
- `lib/fallback/mps_gemm.mm`: MPSMatrixMultiplication path
- `lib/fallback/accelerate_gemm.c`: CPU `cblas_sgemm` (AMX on M1-M3, SME on M4+)

### Tests (9 total, 100% passing on M2 Ultra)
- `test_device`: smoke + family detect
- `test_gemm_f32`: bit-exact vs Accelerate (max_abs=0 across all shapes)
- `test_gemm_f16`: RMS-scaled error vs Accelerate, <1.5e-2 across 64..512
- `test_gemm_bf16`: kernel skip-clean on Apple<9 (no runtime exercise on M2)
- `test_gemm_i8`: kernel skip-clean on Apple<10
- `test_attention_correctness`: D=64 and D=128 vs fp64 reference, <2e-2 RMS-scaled
- `test_attention_backward`: dQ/dK/dV all <1% RMS-scaled vs fp64 analytic gradient
- `test_training_kernels`: 6/6 kernels (RMSnorm/LayerNorm/SwiGLU/softmax/RoPE/AdamW)
- `test_transformer_block`: full forward through every kernel + AdamW step

### Measured perf (Apple M2 Ultra, family Apple8, ~27 TFLOPS fp16 peak)
| Workload | TFLOPS | % of peak |
|---|---|---|
| GEMM fp16 4096³ | 17.59 | ~65% |
| GEMM fp32 4096³ | 2.38 | ~60% (bit-exact) |
| FA fwd fp16 D=64 S=4096 | 6.72 | — |

### Eshkol integration
- `eshkol/tensorcore.esk`: Scheme-level bindings (`tc-init`, `tc-gemm-fp16`, etc.)
- `eshkol/hello_tensorcore.esk`: minimal example
- `eshkol/bridge/tensorcore_codegen.cpp`: drop-in for `eshkol-platform/lib/backend/` — declares 14 `tc_*` ExternalLinkage LLVM symbols, mirrors `builtin_declarations.cpp` pattern
- `eshkol/bridge/INTEGRATION.md`: 4-step recipe

### Build
- CMake 3.20+, macOS 12.0+ (Apple7+ runtime check). C11/C++17.
- SDK detection auto-includes Metal 4 sources when SDK >= 26.0; skipped cleanly on older SDKs (today: macOS 15.1 + Xcode 16.2 + SDK 15.2).
- `compile_metallib.cmake` helper: `.metal` → `.air` → `default.metallib` precompile (qgt-style, no runtime compile overhead).

### Known limitations (documented in ROADMAP.md)
- v0.1 bf16/i8 paths unexercised at runtime (M2 lacks the silicon).
- v0.1 conv2d covers forward only and processes batches serially.
- v0.1 distributed: only single-host backend live; multi-Mac TB5 ring lands v0.5.
- v0.1 attention backward: D=64 only.
- v0.1 autotune: family-keyed static table; bench-driven sweep + cache persistence are wired but not yet self-tuning at init.
- Metal 4 `mpp::tensor_ops` attention kernel has placeholder softmax step pending M5 hardware validation.
