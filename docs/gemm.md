# GEMM

`tc_gemm` is the workhorse: matrix multiply with optional alpha/beta scaling,
optional transpose flags, sync / async / batched variants, and a dispatch
that picks among `simdgroup_matrix`, the M5 TensorOps path, MPS, and
Accelerate based on dtype × family × shape.

## Surface

See [api_reference.md § GEMM](api_reference.md#gemm-gemmh) for the exact
signatures. The descriptor is:

```c
typedef struct {
    int32_t   M, N, K;
    tc_dtype_t a_dtype, b_dtype, c_dtype, accum_dtype;
    bool       transpose_a, transpose_b;
    float      alpha, beta;
    int32_t    lda, ldb, ldc;
} tc_gemm_desc;
```

Layout matches BLAS row-major. `C[M, N] = alpha * A[M, K] @ B[K, N] +
beta * C[M, N]`. Leading dims default to row-major contiguous when 0.

## Kernel variants

| Kernel source | Tile | dtype | When used |
|---|---|---|---|
| `gemm_simdgroup.metal` | 64×64, BK=32 | fp16, bf16, fp32 | default `simdgroup_matrix` path |
| `gemm_simdgroup.metal` (i8 variant) | 64×64, BK=32 | int8 → i32 accum | Apple10+ |
| `gemm_simdgroup_128.metal` | 128×128, BK=32 | fp16, fp32 | opt-in via `TC_USE_128_TILE=1` |
| `gemm_async.metal` | 64×64 with async_copy | fp16 | SDK < 26 only; uses private `__asm` |
| `gemm_async_128.metal` | 128×128 with async_copy | fp16 | SDK < 26 only |
| `tensorops_gemm.metal` | tile from `mpp::tensor_ops` | fp16, bf16, fp32 | Apple11 + SDK 26.0+ + `TC_ENABLE_TENSOROPS=ON` |
| `gemm_quantized_v2.metal` | M ≤ 4 GEMV | Q4_0 → fp16 | `tc_gemv_quantized` default |
| `gemm_quantized.metal` | M ≤ 4 GEMV | Q4_0/Q8_0 → fp16 | `tc_gemv_quantized` fallback (`TC_Q4_USE_V1=1`) |

## The 64×64 tile (default)

The default kernel does the following per CTA (threadgroup):

- Tile shape: 64 rows of A × 64 cols of B × 32-deep K-block.
- **4 simdgroups per CTA** (32 threads/simdgroup × 4 simdgroups = 128
  threads), arranged WM=2 × WN=2.
- Each simdgroup owns a 32×32 subtile of the 64×64 output (TM=4, TN=4
  of 8×8 `simdgroup_matrix` fragments — 16 fragments / simdgroup).
- K-loop: load 64×32 A-tile + 32×64 B-tile cooperatively (vec4 loads) into
  threadgroup memory, then 4× `simdgroup_multiply_accumulate(C, A, B)` to
  drain BK=32 of K.

The fp32 accumulator path is unconditional — we never run with fp16 accum
in production. fp32 accumulation is *the* difference between "valid
training kernel" and "broken kernel" at K > 2048.

### Why 64×64?

The 32 KB threadgroup memory budget on M-series caps the largest A+B+C
working set we can stage. 64×32 fp16 A + 32×64 fp16 B + 64×64 fp32 C =
4 KB + 4 KB + 16 KB = 24 KB, plus space for double-buffered K-load
(v0.2). 128×128 is feasible on paper but the v0.1 kernel uses too many
registers per simdgroup, which throttles occupancy. v0.2 retunes.

### Measured perf

On M2 Ultra (current checkpoint):

| Shape | dtype | TFLOPS | % peak |
|---|---|---:|---:|
| 4096³ | F16 | **17.88** | ~66% |
| 2048³ | F16 | 11.19 | — |
| 1024³ | F16 | 4.80 | — |
| 4096³ | F32 | 2.46  | ~60% |
| 2048³ | F32 | 2.26  | — |

`bench/bench_gemm.c` does the full sweep 256..4096 across fp16 / bf16 /
fp32 and prints median TFLOPS plus `tc_last_backend()` for every cell.

## The 128×128 tile (opt-in)

`gemm_simdgroup_128.metal` is the v0.2 target. Today it builds, dispatches,
and produces correct results, but it regresses M2 performance (~10 TFLOPS
@ 4096³ vs 17 TFLOPS for 64×64) due to register pressure.

Activate via:

```sh
TC_USE_128_TILE=1 ./build/bench/bench_gemm
```

Tuning is the v0.2 work:
- Reduce register-pressure-aware sg layout (target WM=4×WN=2, TM=2×TN=4).
- Double-buffer K-block loads (one tile staged while computing on prev).
- Async dispatch with multi-CB pipelining.

## The TensorOps M5 path

On Apple11 silicon (M5+) with SDK 26.0+ and `TC_ENABLE_TENSOROPS=ON`, the
dispatch can route to `mpp::tensor_ops::matmul2d` via the
`MTL4MachineLearningCommandEncoder`. Apple's reported speedup is up to 4×
at small-shape inference (the M5 "neural accelerators").

`lib/tensorops/tensorops_m5.mm` is the host glue;
`kernels/metal/tensorops_gemm.metal` is the kernel. The dispatch is
conservative in v0.1: it lights up on M5 + SDK 26 + flag-on, runs the
correctness suite, and lets the autotune decide between simdgroup_matrix
and tensorops based on a first-call bench.

This is also what `tensorops_flash_attention.metal` is — the same encoder
path applied to attention.

## Async, batched, and streams

```c
tc_stream* s; tc_stream_create(ctx, &s);
tc_gemm_async(ctx, &d, A, B, C, s);
tc_gemm_async(ctx, &d, A, B, D, s);     /* batched / pipelined */
tc_stream_sync(s);
```

The stream holds a pending `MTLCommandBuffer` across dispatches; only
`tc_stream_sync` commits and waits. For inference, this saves the
command-buffer round trip between layers — the bench's async-batched
Q4_0 GEMV path on the synthetic 7B decode shape reaches **186 tok/s @
632 GB/s effective bandwidth (~79% of LPDDR5 peak)**, ~3× ahead of
llama.cpp's published M2 Ultra numbers on the same shape. See
[benchmarks.md](benchmarks.md).

For batched same-shape GEMM with elementwise strides, use
`tc_gemm_batched`:

```c
tc_gemm_batched_desc bd = { .base = d, .batch = 32,
                            .stride_a = M*K, .stride_b = K*N, .stride_c = M*N };
tc_gemm_batched(ctx, &bd, A, B, C);
```

Internally this issues 32 dispatches against the same pipeline state with
buffer offsets; the per-batch CB amortization makes this faster than 32
separate `tc_gemm` calls.

## Fallback ladder

If `simdgroup_matrix` can't serve a call:

```
Apple10+ int8           → simdgroup_matrix (TC_BACKEND_SIMDGROUP_MATRIX)
Apple7..9 int8          → fp32-widen fallback via tc_gemm (TC_BACKEND_SIMDGROUP_MATRIX after cast)
Apple9+  bf16           → simdgroup_matrix (TC_BACKEND_SIMDGROUP_MATRIX)
Apple7..8 bf16          → fp32-cast fallback via tc_gemm (TC_BACKEND_SIMDGROUP_MATRIX after cast)
small/odd shapes        → MPSMatrix (TC_BACKEND_MPS)
all GPU paths failed    → cblas_sgemm (TC_BACKEND_ACCELERATE_CPU)
```

Use `tc_last_backend()` to see which row matched.

## Transposes

The transpose flags are honored by reading the operand with swapped strides
inside the kernel (no separate transpose pass). This is implemented via a
function constant; the pipeline cache stores four variants (NN / NT / TN /
TT) per dtype.

## Numerical guarantees

| dtype | Guarantee | Validated by |
|---|---|---|
| fp32 | bit-exact against `cblas_sgemm` | `tests/test_gemm_f32.c` |
| fp16 | rms_scaled error ≤ 5e-3 vs fp64 reference | `tests/test_gemm_f16.c` |
| bf16 | rms_scaled error ≤ 3e-3 vs fp64 reference (fallback) | `tests/test_gemm_bf16.c` |
| int8 | bit-exact i32 accumulation up to K = 2^16 | `tests/test_gemm_i8.c` |

`rms_scaled` is `||y - yref|| / (||yref|| + ε)`. Per-cell relative error
isn't meaningful for matmul outputs that can be near-zero.

## Knobs

| Env / flag | Effect |
|---|---|
| `TC_USE_128_TILE=1` | use the 128×128 GEMM tile (regresses on M2; v0.2 retunes) |
| `TC_ENABLE_TENSOROPS=ON` (CMake) | compile the M5 TensorOps path |
| `TC_METALLIB=<path>` | override the metallib search; useful when running from a non-standard install layout |

## Adding a new GEMM kernel

Walkthrough in [CONTRIBUTING.md](../CONTRIBUTING.md). Short version:

1. Write the `.metal` kernel in `kernels/metal/`. Use `simdgroup_matrix`
   for matmul-shaped work; function constants for static switches.
2. Add to `TC_METAL_SOURCES` in `CMakeLists.txt` (mind the SDK gates).
3. Encode the dispatch in `lib/ops/gemm.mm`. Pick a unique pipeline-cache
   key and `tc_set_last_backend(TC_BACKEND_...)`.
4. Write a correctness test (`tests/test_gemm_*.c`) against an fp64
   reference. Use `rms_scaled` for fp16/bf16, bit-exact for fp32 and int8.
5. Bench it (`bench/bench_gemm.c` — add the shape if needed).
6. Update [benchmarks.md](benchmarks.md) and the changelog.
