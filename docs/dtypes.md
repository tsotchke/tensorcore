# dtypes

`tensorcore` has ten dtypes today, organized so a kernel can index a dispatch
table by `(uint8_t)dtype`. Five are first-class ML dtypes; five are
precision-emulation modes inherited from the `eshkol-platform` lineage. New
dtypes append; nothing renumbers.

```c
typedef enum {
    TC_DTYPE_F16  = 0,
    TC_DTYPE_BF16 = 1,
    TC_DTYPE_F32  = 2,
    TC_DTYPE_I8   = 3,
    TC_DTYPE_I32  = 4,
    TC_DTYPE_F64  = 5,
    TC_DTYPE_SF64 = 6,
    TC_DTYPE_DF64 = 7,
    TC_DTYPE_FP24 = 8,
    TC_DTYPE_FP53 = 9,
} tc_dtype_t;
```

## First-class ML dtypes

### `TC_DTYPE_F16` — IEEE 754 binary16

**Storage:** 2 bytes / element. **Apple7+ native** in `simdgroup_matrix`.

The default training and inference dtype. fp16 inputs with fp32 accumulators
is the design center for `tc_gemm`, `tc_attention_*`, and every training
kernel. Range is the standard ~6.1e-5 to 6.5e4 — well within the activation
range of a healthy transformer.

### `TC_DTYPE_BF16` — bfloat16

**Storage:** 2 bytes / element. **Apple9+ native** in `simdgroup_matrix`
(M3 / A17 Pro and newer). **Software fallback on Apple7..8.**

Same exponent range as fp32 (8 bits), only 7 bits of mantissa. The right
choice when you have outlier activations or you're training from scratch
and want to skip the loss-scaling logic that fp16 requires.

The software fallback bit-casts bf16 ↔ fp32 (`bf16 = high 16 bits of
fp32`), routes through fp32 `tc_gemm`, and round-trips back. Validated
at ~2.7e-3 RMS-scaled error vs an fp64 reference on Apple8.

### `TC_DTYPE_F32` — IEEE 754 binary32

**Storage:** 4 bytes / element. **Apple7+ native.**

The reference dtype: `tc_gemm` fp32 is **bit-exact** against
`cblas_sgemm`. Use it when correctness is the constraint and TFLOPS isn't.
2.36 TFLOPS @ 4096³ on M2 Ultra; ~60% of fp32 peak.

### `TC_DTYPE_I8` — signed 8-bit integer

**Storage:** 1 byte / element. **Apple10+ native** in `simdgroup_matrix`
(M4 and newer). **Software fallback on Apple7..9.**

Used for post-training quantization and Q-LoRA-style fine-tunes. The fallback
widens to fp32; since fp32 has 24 bits of mantissa, the i8×i8 product
accumulation is exact up to K = 2^16 (more than any realistic matmul). Tests
report bit-exact agreement (0 errors / 65K cells at 256³) against an
i32-reference on Apple8.

### `TC_DTYPE_I32` — signed 32-bit integer

**Storage:** 4 bytes / element.

Used as:
- the accumulator dtype for i8 GEMM (`accum_dtype = TC_DTYPE_I32`),
- an index dtype where ops take one.

Never used as an input dtype for matmul today.

## Precision-emulation dtypes (from eshkol-platform)

These dtypes encode arbitrary-precision and custom-precision floats. The
software implementations live in `eshkol-platform/lib/backend/gpu/gpu_memory.mm`
today; the v0.4 consolidation phase moves the relevant kernels into
tensorcore proper.

### `TC_DTYPE_F64` — IEEE 754 binary64

**Storage:** 8 bytes / element. **Emulated.** Apple GPUs do not have native
fp64. Compute happens via either SF64 (SoftFloat) or DF64 (double-float)
depending on the op.

### `TC_DTYPE_SF64` — SoftFloat-64 storage

**Storage:** 8 bytes / element, stored as `uint2`. Pure-software IEEE 754
binary64 implemented in MSL. Slow but exact; used for the precision-critical
paths in scientific compute (quantum_geometric_tensor, semiclassical_qllm).
Backend reports `TC_BACKEND_SF64_EMULATED`.

### `TC_DTYPE_DF64` — double-float (f32 + f32 unevaluated sum)

**Storage:** 8 bytes / element. Two fp32 values whose unevaluated sum
represents an approximately fp52-precision number. Faster than SF64;
cheaper than full IEEE 754 fp64; works for accumulation-dominated kernels.

### `TC_DTYPE_FP24` — 24-bit ML format

**Storage:** 4 bytes / element (3 bytes used). Custom format from
eshkol-platform; trades dynamic range for compactness on activations.

### `TC_DTYPE_FP53` — 53-bit format

**Storage:** 8 bytes / element. Custom format from eshkol-platform; designed
for the precision-critical eigensolver paths in quantum compute.

## What's supported where

| Operation | F16 | BF16 | F32 | I8 | I32 | F64 / SF64 / DF64 / FP24 / FP53 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `tc_gemm` | ✓ | ✓ (fallback < Apple9) | ✓ | ✓ (fallback < Apple10) | — | reserved (v0.4) |
| `tc_gemm_async` | ✓ | ✓ | ✓ | ✓ | — | — |
| `tc_attention_*` | ✓ | ✓ | — | — | — | — |
| `tc_rmsnorm_*` | ✓ | — | — | — | — | — |
| `tc_layernorm_*` | ✓ | — | — | — | — | — |
| `tc_rope_*` | ✓ | — | — | — | — | — |
| `tc_swiglu_*` | ✓ | — | — | — | — | — |
| `tc_softmax_*` | ✓ | — | — | — | — | — |
| `tc_adamw_step` (grads) | ✓ | — | ✓ | — | — | — |
| `tc_fused_*norm_gemv` | ✓ | — | — | — | — | — |
| `tc_conv2d_*` | ✓ | — | — | — | — | — |
| `tc_gemv_quantized` (X) | ✓ | — | — | — | — | — |
| `tc_quantize_weights` (in) | ✓ | — | — | — | — | — |
| `tc_allreduce` | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| `tc_broadcast` | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| `tc_allgather` | ✓ | ✓ | ✓ | ✓ | ✓ | — |

A blank cell means "not implemented in v0.1." `tc_attention_*` doesn't
support fp32 deliberately — the entire point of FlashAttention is to fit
the on-chip tile in 32 KB of threadgroup memory, which only works at
fp16/bf16.

## Accumulation policy

The accumulator dtype is independent from the input dtype. For every
GEMM-shaped kernel, the rules are:

| Input dtype | Default `accum_dtype` | Why |
|---|---|---|
| F16 | F32 | Standard ML practice. fp16 mantissa is 10 bits; long inner-products lose precision rapidly without fp32 accumulation. |
| BF16 | F32 | Same reason; bf16 has 7-bit mantissa. |
| F32 | F32 | Bit-exact against Accelerate. |
| I8 | I32 | Required by the `simdgroup_matrix` int path. |

You can request other accumulators (e.g. fp16 accum for inference) by
setting `tc_gemm_desc.accum_dtype` — but if you go below fp32 with fp16
inputs you'll see precision loss above K ≈ 2048.

## Storage size helper

```c
static inline size_t tc_dtype_size(tc_dtype_t d);
```

| Dtype | Bytes |
|---|---:|
| `F16`, `BF16` | 2 |
| `F32`, `I32`, `FP24` | 4 |
| `I8` | 1 |
| `F64`, `SF64`, `DF64`, `FP53` | 8 |

`tc_dtype_size(UNKNOWN_VAL)` returns 0 — caller should treat 0 as an
invalid-arg signal.

## Conventions used throughout the library

- All "fp16" in this doc means IEEE 754 binary16 (Apple's `half` /
  `MTLPixelFormatR16Float` / `__fp16`). Not Apple's old `__fp16` storage
  type with different semantics.
- All "bf16" means the Google bfloat16 layout (sign + 8 exp + 7 mantissa).
- "rms_scaled" error metric (used throughout the tests): `||y - yref|| /
  (||yref|| + epsilon)`. This is robust to per-cell rounding noise; per-cell
  relative error blows up near zero.
- `accum_dtype` is honored but not advertised in failure modes — if a chip
  can't do the requested accumulation natively, the fallback ladder takes
  over and `tc_last_backend` reports the actual path.

## Dtype roadmap

- v0.2: enable bf16 / int8 fast paths everywhere they're supported (today
  only GEMM uses them).
- v0.3: introduce `TC_DTYPE_F8_E4M3` / `F8_E5M2` emulated via per-tile-scale
  fp16 → bf16 (matches NVIDIA's Transformer Engine semantics).
- v0.4: SF64 / DF64 / FP24 / FP53 GEMM and elementwise kernels move from
  `eshkol-platform/lib/backend/gpu/gpu_memory.mm` into tensorcore.
- v0.5+: native fp8 if Apple silicon ships it; emulated fp4 for inference.
