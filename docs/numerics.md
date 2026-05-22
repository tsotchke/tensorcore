# Numerics

How `tensorcore` thinks about precision. What it guarantees, what it
doesn't, and how to compare against a reference. This page is the contract
that the test suite enforces; it's the thing you read when you ask "is
this number wrong?"

## The metrics

### `rms_scaled` — the workhorse

For comparing a computed tensor `Y` against a reference `Y_ref`:

```
rms_scaled = ||Y - Y_ref|| / (||Y_ref|| + ε)
```

Where `||·||` is the Frobenius norm and `ε` is a small constant (`1e-30`)
that prevents division-by-zero when the reference itself is zero.

**Why this and not per-cell relative error.** A matmul output that
includes a tile near `0` will have huge per-cell relative errors purely
from rounding; the relative metric blows up at the cells that *don't
matter*. `rms_scaled` is the energy-weighted error — it tracks the
cells that carry signal.

Used everywhere in `tests/test_*.c`. The function is `rms_scaled_error`
in each test file (small inlined helper, ~5 lines).

### `bit_exact`

Two tensors agree to the last bit. Used when we expect the kernel and
the reference to produce identical results:

- fp32 GEMM matches `cblas_sgemm` exactly (`tests/test_gemm_f32.c`)
- int8 GEMM with `accum_dtype=TC_DTYPE_I32` matches an i32-reference
  exactly up to `K=2^16` (`tests/test_gemm_i8.c`)
- Q4 dequant-and-multiply matches the CPU reference exactly when both
  use the same block layout

### Maximum absolute error

For attention / softmax outputs in [0,1] where rms_scaled is less
informative than per-cell error. Used as a sanity bound alongside
rms_scaled in `test_attention_correctness.c`.

## What the library guarantees

| Path | Comparison target | Tolerance |
|---|---|---|
| `tc_gemm` fp32 | `cblas_sgemm` | **bit-exact** |
| `tc_gemm` fp16 (Apple7+) | fp64 reference | rms_scaled ≤ 5e-3 |
| `tc_gemm` bf16 (Apple9+ native or fp32 fallback) | fp64 reference | rms_scaled ≤ 3e-3 |
| `tc_gemm` int8 → i32 (Apple10+ native or fp32 widen fallback) | i32 reference up to K=2^16 | **bit-exact** |
| `tc_attention_forward` fp16 | fp64 reference | rms_scaled ≤ 1e-3 at S=4096 |
| `tc_attention_backward` fp16 D=64/D=128 | numerical-differences reference | rms_scaled ≤ 3e-3 |
| Q4_0 / Q8_0 GEMV | dequantized CPU reference | rms_scaled ≤ 2e-4 |
| `tc_rmsnorm_*`, `tc_layernorm_*`, `tc_rope_*` | fp64 reference | rms_scaled ≤ 5e-3 |
| `tc_swiglu_*`, `tc_softmax_*` | fp64 reference | rms_scaled ≤ 5e-3 |
| `tc_adamw_step` | scalar fp64 update repeated per element | rms_scaled ≤ 1e-5 |
| `tc_fused_*norm_gemv` | separate norm-forward + `tc_gemm` paths | rms_scaled ≤ 5e-3 |
| `tc_conv2d_*` | fp64 reference | rms_scaled ≤ 1e-3 |
| `tc_allreduce` / `tc_broadcast` | per-rank algorithm | **bit-exact** for the SINGLE backend and threads/fork ring backends |

These tolerances are baked into the test suite. Don't merge a kernel
that regresses them.

## Why fp16 has 5e-3 error and not 0

fp16 has 10 bits of mantissa. A length-K inner product accumulates ~12
bits of rounding noise at K=4096 — `sqrt(4096) ≈ 64`, mantissa step
≈ 2^-10, so error ~`64 × 2^-10 ≈ 6e-2` per cell at fp16-throughout.
With **fp32 accumulators inside the kernel** (which we always use), the
accumulation noise drops to fp32 levels (~`64 × 2^-23 ≈ 8e-6`) until the
final downcast, and the cast itself reintroduces fp16-step noise (~`2^-10
≈ 1e-3`).

`rms_scaled ≤ 5e-3` at 4096³ is what falls out of this analysis. Real
measurements typically come in at 1-3e-3.

If you see rms_scaled > 1e-2 on a fp16 kernel, something is wrong.
Either the accumulator is fp16 (won't happen in our kernels — we audit
this), or your reference is wrong.

## Why bf16 has higher error than fp16

bf16 has 7 bits of mantissa (vs fp16's 10) but the same 8 bits of
exponent as fp32. Downcast noise from fp32 to bf16 is `~2^-7 ≈ 8e-3`
per element. With energy averaging, rms_scaled lands around 2-3e-3 — the
3e-3 contract has a small safety margin.

If you migrate code from fp16 to bf16 (e.g. to skip loss scaling),
expect the rms_scaled to roughly triple. That's the tradeoff for the
larger dynamic range.

## Why fp32 is bit-exact and not "essentially exact"

fp32 inputs + fp32 accumulators + fp32 outputs + IEEE round-to-nearest-
even means the same multiply-add sequence produces the same result.
`tc_gemm` orders its inner sums identically to `cblas_sgemm` (row-major
contiguous; 4-element vec accumulators), so the floating-point operations
happen in the same order. The result is the same up to the last bit.

This is the test we'd want to break first if we change anything in the
fp32 path. It's that load-bearing.

## Why int8 is bit-exact up to K=2^16

`i8 × i8 → i16` per multiply; summing K of them needs `i16 + log2(K)`
bits to never overflow. For K=2^16, that's 16 + 16 = 32 bits — exactly
what `i32` accumulators give. Below K=2^16, the fp32-widen fallback (used
on Apple7..9) is also bit-exact because fp32's 24-bit mantissa is more
than 24 bits ≥ 16 + log2(K) for K ≤ 2^8.

Above K=2^16, accumulation can overflow on the native int8 path; the
test caps K at 16384 to stay in-bound. Real Q-LoRA workloads stay well
under this.

## Why Q4 has tighter error than fp16

Because the comparison is **against the dequantized version of the same
weights**, not against a different precision. The kernel reads the same
4-bit weights the reference reads; the only error is the kernel's
internal accumulation, which is fp32. Hence rms_scaled ≤ 2e-4 — much
tighter than fp16-vs-fp64 because we're not testing quantization itself,
we're testing that the dequant+multiply implementation agrees with the
naive one.

## How tests compare

Inside `tests/test_*.c`, the pattern is:

```c
/* 1. Compute on GPU */
tc_status_t s = tc_gemm(ctx, &d, A, B, C);

/* 2. Compute reference on CPU in fp64 */
for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
        double sum = 0.0;
        for (int k = 0; k < K; ++k)
            sum += (double)A_host[i*K + k] * (double)B_host[k*N + j];
        ref[i*N + j] = (float)sum;
    }

/* 3. Compare */
double err = rms_scaled_error(C_host, ref, M * N);
assert(err <= TOLERANCE);
```

This is what the default CTest suite does, with shape and dtype
variations across the native, Python, and example smokes in `tests/`.

## When you see a number that looks wrong

Order of operations:

1. **Verify the test harness, not the kernel.** Most "wrong" results
   come from the reference: wrong order of operations, wrong shape,
   wrong dtype, missing alpha/beta scale. The reference being wrong is
   2-3× more common than the kernel being wrong.
2. **Check `tc_last_backend()`.** If you expected `SIMDGROUP_MATRIX` and
   got `ACCELERATE_CPU` or `MPS`, you're testing a different code path.
3. **Try smaller shapes.** If 256³ is fine and 4096³ is not, that's a
   rounding-noise issue. If both are not fine, it's a kernel bug.
4. **Verify the dtype mapping.** It's easy to pass fp16 storage where
   the kernel expects fp32 — `sizeof` mismatches don't always crash
   immediately.
5. **Run with `TC_USE_128_TILE=1`** or another env override to see if
   the issue is path-specific.

## Determinism

Every kernel is deterministic given the same input. There's no random
state, no atomic non-determinism (the int8 GEMM uses fp32-widen on older
chips, but it's still bit-exact). `tc_conv2d_backward_input` uses fp32
atomic adds during col2im, which on Apple Silicon are deterministic
(the atomic-add unit is a separate atomic compute domain; same order of
arrival = same result).

`tests/test_distributed_ring*.c` validates bit-exact agreement for
4-rank ring all-reduce: the same code reaches the same final answer
across all ranks.

## Reproducibility across chips

The same library binary produces the same fp32 GEMM output on M1, M2,
M3, M4, and M5 — all bit-exact against Accelerate. The fp16/bf16 paths
produce numerically *equivalent* output (rms_scaled within tolerance),
not bit-exact, because the simdgroup_matrix unit's internal order of
operations can differ across Apple GPU families. fp16 results on M2
Ultra and fp16 results on M3 Max will land within ~1 ULP of each other
on any given cell, but not bit-equal.

The fallback path (e.g. bf16 on Apple7..8 routed through fp32) is
deterministic and reproducible across all M-series — the cast pattern is
fixed.

## See also

- [dtypes.md](dtypes.md) — every dtype's storage, range, and accumulation
  policy.
- [gemm.md](gemm.md) — GEMM fallback ladder and the family gating that
  determines which numerical path you took.
- [family_gating.md](family_gating.md) — how to know which path served
  your call.
