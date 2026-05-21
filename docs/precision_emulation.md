# Precision emulation — SF64, DF64, FP24, FP53

`tc_dtype_t` includes four dtypes inherited from the `eshkol-platform`
precision-arithmetic lineage: `TC_DTYPE_SF64`, `TC_DTYPE_DF64`,
`TC_DTYPE_FP24`, `TC_DTYPE_FP53`. They are not first-class ML dtypes;
they encode arbitrary- and custom-precision floats for scientific
compute (quantum simulation, eigensolvers, error-correction codes).

v0.1 reserves the enum values and the dispatch infrastructure. The
kernels themselves still live in `eshkol-platform/lib/backend/gpu/gpu_memory.mm`;
v0.4 consolidation moves them into `tensorcore`.

This page explains what each format is, why it exists, and what to
expect when v0.4 lands.

## Why the GPU needs custom precision

Apple Silicon GPUs have no native fp64. CUDA cards do (~half the
fp32 rate); Apple silicon does not. For the eshkol-platform's downstream
clients — quantum_geometric_tensor, semiclassical_qllm — fp64-grade
precision *is the workload*. Tensor decompositions of stabilizer
codes, real-space Hamiltonian eigenvalue solvers, geometric Riemannian
integrators — these need precision that fp32 / fp16 can't deliver.

Four solutions, ranked by accuracy / speed:

| Format | Precision (decimal digits) | Storage | Speed |
|---|---:|---|---|
| fp32 | ~7 | 4 B | native; reference baseline |
| FP24 | ~5 | 4 B (3 used) | native fp32 with mantissa truncation |
| DF64 | ~16 | 8 B (2× fp32) | ~4× fp32 |
| FP53 | ~16 | 8 B | ~3-4× fp32 |
| SF64 | ~17 (full IEEE binary64) | 8 B | ~10-20× fp32 |

DF64 and FP53 are the practical sweet spot — accuracy comparable to
native fp64 at affordable cost. SF64 is the conservative path: bit-exact
IEEE binary64 simulation when "compute exactly what an x86 fp64 unit
would compute" is the requirement.

## `TC_DTYPE_F64` — IEEE 754 binary64

Storage: 8 bytes / element. **Emulated.** Apple GPUs do not have native
fp64. When you pass `TC_DTYPE_F64` to a tensorcore call, the dispatch
chooses between SF64 and DF64 based on what the kernel expects; if the
op doesn't support emulated fp64 yet (v0.1 status), you get
`TC_ERR_UNSUPPORTED_DTYPE`.

## `TC_DTYPE_SF64` — SoftFloat-64

Storage: 8 bytes / element, internally a `uint2`. Pure-software IEEE 754
binary64 implemented in MSL.

- Bit-exact agreement with x86-64 fp64 under the same rounding mode.
- Slow: ~10-20× the cost of fp32 per operation.
- Backend reports `TC_BACKEND_SF64_EMULATED` (enum value 5) when chosen.

Used when correctness across CPU/GPU boundaries is the constraint — e.g.
validating a GPU result against a CPU oracle.

## `TC_DTYPE_DF64` — double-float (f32 + f32 unevaluated sum)

Storage: 8 bytes / element: two fp32 values `hi` and `lo` whose
unevaluated sum represents an approximately-fp52-precision number.

- Equivalent to ~16 decimal digits of accuracy.
- Faster than SF64 (~4× fp32 per op) because the operations decompose
  into fp32 primitives that the hardware does run.
- The "lo" carries the rounding remainder; care with the order of
  operations (especially in accumulating GEMMs) is mandatory.

Used for accumulation-dominated workloads where the loss-of-precision
in fp32 sums is the bottleneck — e.g. long-trajectory ODE integrators.

DF64 is the dtype most projects choose first when they realize fp32
isn't enough: same memory footprint as fp64, half the speed loss of
SF64, and "close enough" to IEEE fp64 for almost everything that
isn't pathological.

## `TC_DTYPE_FP24` — 24-bit ML format (eshkol-platform)

Storage: 4 bytes / element, only 3 bytes used (sign + 8 exp + 15
mantissa). Custom format from `eshkol-platform`.

- More precision than fp16 (15 mantissa bits vs 10).
- Same dynamic range as fp16 (5-bit exponent? No — 8-bit, matches fp32).
  Wait — actually the original format is: sign + 8 exp + 15 mantissa,
  packed in 24 bits with the high byte of a 32-bit word zeroed.
- Storage savings vs fp32: 25%.
- Used as an activation dtype where fp16's dynamic range isn't enough
  but fp32's full mantissa is overkill.

Not widely used outside the eshkol-platform ecosystem; primarily for
research workloads.

## `TC_DTYPE_FP53` — 53-bit format (eshkol-platform)

Storage: 8 bytes / element. Custom format from `eshkol-platform`,
designed for the precision-critical eigensolver paths in
quantum_geometric_tensor.

- 53-bit mantissa = full IEEE fp64 precision in the mantissa.
- Extended exponent range (15 bits, vs fp64's 11) — handles wide
  dynamic range without overflow.
- Storage same as fp64 / SF64 / DF64.

Used where you need fp64's precision *plus* extra exponent headroom — the
specific case is large-condition-number linear algebra where intermediate
quantities span many orders of magnitude.

## Current status (v0.1.x)

- **Enum values reserved** (`TC_DTYPE_F64 = 5`, `TC_DTYPE_SF64 = 6`,
  `TC_DTYPE_DF64 = 7`, `TC_DTYPE_FP24 = 8`, `TC_DTYPE_FP53 = 9`).
- **`tc_dtype_size()` returns the correct byte count** for each
  (`F64/SF64/DF64/FP53 = 8`, `FP24 = 4`).
- **`tc_dtype_name()` returns a stable string** for each (`"f64"`,
  `"sf64"`, etc.).
- **`tc_gemm` accepts them only on the dispatch level**; the kernels
  themselves live in `eshkol-platform/lib/backend/gpu/gpu_memory.mm`
  and are not yet linked from tensorcore. Calls return
  `TC_ERR_UNSUPPORTED_DTYPE` today unless your build links the
  external implementation.

## v0.4 consolidation

The plan is:

1. Lift `eshkol-platform/lib/backend/gpu/gpu_memory.mm`'s SF64 / DF64 /
   FP24 / FP53 implementations into `kernels/metal/precision_*.metal`
   files.
2. Register them in the dispatch tables so `tc_gemm` with
   `accum_dtype = TC_DTYPE_DF64` (etc.) dispatches automatically.
3. Add correctness tests against fp128 / arbitrary-precision references.

After v0.4, the three downstream Metal backends in `eshkol-platform`,
`quantum_geometric_tensor`, and `semiclassical_qllm` collapse onto
tensorcore. The custom-precision dtypes are the technically-difficult
piece of that consolidation: the matmul-on-`uint2` pattern needs careful
threadgroup memory layout to avoid throughput collapse.

## Why this matters

The bet: Apple Silicon's biggest applied-math limitation is "no native
fp64 GPU compute." This has historically locked Mac out of any
scientific workload that needs more than fp32 precision. With DF64 /
SF64 on tensorcore, that constraint goes away for everything that's
GEMM-shaped.

The numbers from `eshkol-platform`'s existing implementation: a 1024³
DF64 GEMM runs at ~0.6 TFLOPS on M2 Ultra (vs ~10 TFLOPS fp32 at the
same shape). 16× slower than fp32, but ~16× faster than the
"download to CPU and use Accelerate fp64 BLAS" alternative.

For Mac users in the quantum-simulation, scientific-computing, and
research-math worlds, that's the difference between "this workload runs
on my Mac" and "this workload doesn't."

## References

- `eshkol-platform/lib/backend/gpu/gpu_memory.mm` — current implementation
  (~4016 lines, includes the SF64/DF64/FP24/FP53 paths)
- `eshkol-platform/lib/backend/gpu/metal_softfloat.h` — the SoftFloat-64
  emulation primitives in MSL
- David Bailey's QD library — the C reference for double-float / quad-
  double arithmetic that DF64 follows
- IEEE 754-2008 binary64 — the SF64 target specification

## See also

- [dtypes.md](dtypes.md) — the full dtype table with first-class ML
  dtypes (F16, BF16, F32, I8, I32) alongside these.
- [ROADMAP.md](../ROADMAP.md) §v0.4 — the consolidation plan.
- [eshkol_integration.md](eshkol_integration.md) — how tensorcore bridges
  to the eshkol-platform lineage.
