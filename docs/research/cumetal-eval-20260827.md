# CuMetal evaluation — CUDA as single-source kernel language for tensorcore

**Date:** 2026-08-27
**Host:** Atlas, Apple M2 Ultra (192 GB unified), macOS 15.1, Xcode 16.2 (Build 16C5032a)
**Subject:** [CuMetal](https://github.com/Lulzx/cuda-metal) 0.1.3 (Homebrew `lulzx/tap/cumetal`)
**Question:** Can tensorcore's CUDA kernels become the single source of truth, retiring the
hand-maintained Metal backend in `kernels/metal/`?

**Verdict: no. Not at 0.1.3 on this toolchain.** Zero of tensorcore's bandwidth-bound kernels
executed. Two of the three blocking defects fail *silently* — clean compile, `cudaSuccess`,
wrong or absent results.

---

## 1. Install state

`brew install lulzx/tap/cumetal` succeeded (pulls llvm 22.1.8, z3, python@3.14; 35 MB installed).

`cumetal doctor` reports **all green**:

```
[✓] Apple Silicon          arm64
[✓] macOS                  15.1 (14+ required)
[✓] CuMetal 0.1.3          cumetalc / include / lib all present
[✓] CUDA-capable Clang     /opt/homebrew/opt/llvm/bin/clang++
[✓] Metal toolchain        metal, metallib (Xcode 16.2)
[•] Binary compatibility shim   Not installed (optional)
No issues found!
```

**Doctor is not a sufficient readiness check.** It confirms `metal`/`metallib` *exist* but never
checks that they accept the AIR version CuMetal emits. On this machine they do not — see §4. A
green doctor coexists with a compiler that cannot produce a loadable metallib for any non-trivial
kernel.

## 2. Smoke test — PASS

`vectorAdd`, 2^20 f32 elements, `<<<4096, 256>>>`:

```
launch status: cudaSuccess
n=1048576 max_abs_err=0  c[0]=0 c[n-1]=786431 (expect 786431)
SMOKE PASS
```

Bit-exact on the M2 Ultra GPU. Compile time 3.6 s (first run; ~0.8 s warm).

## 3. Real kernel — tensorcore RMSNorm and AdamW: **did not execute**

Selected `rmsnorm_forward_kernel` from `lib/cuda/training.cu` — a bandwidth-bound fp16 serving
kernel (row-wise `rsqrt(mean(x²)+eps) * gamma`) that has an exact hand-written counterpart,
`tc_rmsnorm_forward` in `kernels/metal/training_kernels.metal`. Carried `adamw_step_fp32_kernel` /
`tc_adamw_step_f32` alongside as a pure-streaming control. Both kernels plus the
`block_reduce_sum_f32` helper were copied **verbatim** into a standalone harness.

`cumetalc` accepted the file with **exit 0 and no diagnostics whatsoever**. At runtime:

```
CUMETAL: registered kernel missing metallib/name/args: func=... kernel='_Z22rmsnorm_forward_kernel...' metallib=''
CUMETAL: registered kernel missing metallib/name/args: func=... kernel='_Z22adamw_step_fp32_kernel...' metallib=''
RMSNORM N=8192 D=4096  max_rel_err(Y)=1.000e+00  max_rel_err(rstd)=1.000e+00  MISMATCH
ADAMW   n=33554432     max_rel_err=1.892e-01                                  MISMATCH
```

`cudaDeviceSynchronize()` and `cudaGetLastError()` both returned **`cudaSuccess`**. The output
buffers were never written. A production caller checking CUDA error codes would see a clean run and
propagate garbage.

## 4. Root cause — AIR version mismatch, swallowed in link mode

`cumetalc --cuda-device` exposes the two device routes and the real error:

- **Route A (works).** PTX → **MSL source** → `xcrun metal -c` → `xcrun metallib`. AIR version comes
  from Apple's own compiler, so it is always correct.
- **Route B (broken here).** PTX → **LLVM IR** → `xcrun metal` on the IR. CuMetal stamps the module
  `air version 2.8.0`; Xcode 16.2 ships AIR 2.7:

```
air-lld: error: air version set to 2.8.0 (!0 = !{i32 2, i32 8, i32 0}), but expecting 2.7 in _Z2kkPfPKfi
metal: error: air-lld command failed with exit code 1
error: air version set to 2.8.0 ... but expecting 2.7 in _Z2kkPfPKfi
cumetalc failed: failed to package metallib with xcrun metallib
```

Route A's MSL translator **bails on any libdevice math call, any `__shared__`, and on
`__shfl_xor_sync`** (`cumetalc failed: legacy backend did not produce MSL`), pushing exactly those
kernels onto Route B, where they die. Since every real tensorcore kernel uses at least one of the
three, every real tensorcore kernel is dead.

**The severity is in the swallowing.** In `--cuda-device` mode the error above is printed. In the
default compile-and-link mode, `cumetalc kernel.cu -o kernel` **exits 0, prints nothing, and emits a
runnable binary** whose kernels are simply absent. Verified directly:

```
$ cumetalc z_sqrtf.cu -o z_sqrtf_x
EXITCODE=0        # no stderr, 51 KB executable produced, kernel is a no-op
```

Neither documented workaround helps:
- `--backend cumetal-ir` → `PTX opcode 'st.param.b32' has no CuMetal IR normalization`
- `--fallback-experimental` / `--mode experimental` → writes a `cumetal-experimental` container that
  passes CuMetal's own validator but that the Metal runtime will not load (same runtime warning).

**Untested variable, and the highest-value follow-up:** AIR 2.8 corresponds to the Xcode 26 / Metal 4
toolchain. Installing a newer Metal Toolchain component may unblock Route B entirely and change this
verdict for §5's "did not run" rows. This was deliberately **not** attempted here — it is a large
system mutation on a live serving host. It does **not** affect §6, which is a codegen defect
independent of toolchain version.

## 5. Coverage limits hit

Method: one minimal kernel per feature, output buffer pre-seeded to a sentinel, checked for
modification. Every row marked *silent* compiled with exit 0 and returned `cudaSuccess` from both
`cudaDeviceSynchronize()` and `cudaGetLastError()`.

| CUDA feature | Result | How it failed |
| --- | --- | --- |
| `sqrtf`, `rsqrtf`, `sqrt(double)`, `__fsqrt_rn` | did not run | **silent** |
| `expf`, `__expf`, `logf`, `tanhf`, `powf`, `sinf` | did not run | **silent** |
| `fmaxf`, `fabsf`, `floorf`, `fmaf` | did not run | **silent** |
| `__shared__` + `__syncthreads()` | did not run | **silent** |
| `__shfl_xor_sync` | did not run | **silent** |
| `out[i] = a[i]` (pure copy) | did not run | **silent** |
| `__shfl_down_sync` | ran | — |
| `atomicAdd` (f32, global) | ran | — |
| `__half2float` / `__float2half` (device) | ran | — |
| f32 add, sub, mul, div | ran, bit-exact | — |
| `a*b + c` (FMA contraction) | **ran, wrong answer** | **silent** — see §6 |
| `__float2half` / `__half2float` on **host** | did not compile | *explicit* compile error |

Two notes on that table.

**The pure-copy row is not a typo.** `out[i] = a[i]` fails while `out[i] = a[i] + b[i]` succeeds. The
Route A / Route B boundary is not predictable from source complexity, so "avoid the unsupported
feature list" is not a workable mitigation strategy — there is no reliable way to know which side of
the line a kernel lands on short of running it.

**Host-side fp16 is a source-compatibility break.** CuMetal's `cuda_fp16.h` declares
`__float2half`/`__half2float` as `__device__` only; NVIDIA declares them `__host__ __device__`. Any
tensorcore host-side reference or test code that converts halves will fail to compile. This one is at
least loud. Workaround: on CuMetal's device path `__half` is `_Float16`, so a plain `static_cast`
works on the host.

## 6. Silent miscompile: `a*b + c` truncates to integer

The most damaging finding, and independent of the toolchain mismatch in §4. A kernel of the form
`out[i] = a[i]*b[i] + c[i]` compiles cleanly, dispatches, returns `cudaSuccess` — and returns
`trunc(a*b + c)`:

| a | b | c | CuMetal | exact `a*b+c` |
| --- | --- | --- | --- | --- |
| 1.5 | 2 | 0.25 | **3** | 3.25 |
| 2.25 | 4 | 0.5 | **9** | 9.5 |
| 0.3 | 3 | 0.05 | **0** | 0.95 |
| 10 | 0.125 | 0.375 | **1** | 1.625 |
| 7 | 0.5 | 0.25 | **3** | 3.75 |
| 0.5 | 0.5 | 0.125 | **0** | 0.375 |
| 100.5 | 2 | 0.75 | **201** | 201.75 |
| 3 | 1 | 0.125 | **3** | 3.125 |

8 of 8 truncated. It reproduces through a temporary as well (`float t = a[i]*2.0f; out[i] = t + b[i]`
→ 777 instead of 777.25), so it is the multiply-add contraction, not a single expression shape. A
plain `a + b` or `a * b` alone is bit-exact.

This defect emits no warning at any stage. It corrupts results in a kernel that appears, by every
signal available to the caller, to have run correctly. FMA contraction is present in essentially
every numeric kernel tensorcore has.

## 7. Bandwidth

M2 Ultra, ~800 GB/s peak. 50 timed iterations after warmup, wall-clock around
`commit`/`waitUntilCompleted` (Metal) or `cudaDeviceSynchronize` (CuMetal). Hand-written Metal
reference is `kernels/metal/training_kernels.metal` compiled with `xcrun metal` and driven from an
Objective-C++ harness.

| Kernel | Implementation | ms | GB/s | % of 800 | Correct? |
| --- | --- | ---: | ---: | ---: | --- |
| ADD, 32M f32 (3×4 B/elem) | **CuMetal** (CUDA source) | 0.920 | **437.5** | 55 % | bit-exact |
| TRIAD, 32M f32 (3×4 B/elem) | Metal, hand-written | 0.819 | **491.8** | 61 % | yes |
| RMSNorm fp16, 8192×4096 | Metal `tc_rmsnorm_forward` | 0.491 | **273.5** compulsory / 410.3 issued | 34 % / 51 % | yes |
| RMSNorm fp16, 8192×4096 | **CuMetal** (`rmsnorm_forward_kernel`, verbatim) | — | **did not run** | — | no output |
| AdamW f32, 32M (7×4 B/elem) | Metal `tc_adamw_step_f32` | 1.501 | **625.9** | 78 % | yes |
| AdamW f32, 32M | **CuMetal** (`adamw_step_fp32_kernel`, verbatim) | — | **did not run** | — | no output |

*Compulsory* counts the algorithmic minimum (read X + write Y). *Issued* counts the second read of X
that both implementations perform in phase 2; the ratio between implementations is unaffected.

**The only measurable CuMetal-vs-Metal pair is the elementwise add: 437.5 / 491.8 = 89 % of
hand-written Metal.** That is a genuinely encouraging number and it clears the 80 % bar — but it
describes a kernel with no math function, no shared memory, no reduction, and no FMA. It is a
statement about CuMetal's dispatch and memory path, which is fine. It says nothing about the kernels
tensorcore actually serves, none of which run at all.

## 8. Verdict

**CUDA-as-single-source is not viable for tensorcore's bandwidth-bound serving kernels at CuMetal
0.1.3 on this toolchain.** The hand-maintained Metal backend in `kernels/metal/` cannot be retired.

The ≥80 % viability bar is technically met (89 %) by the single trivial kernel that translates, but
the bar is irrelevant when the coverage rate on real kernels is 0 of 2 and the failure mode is
silence. Three blockers, in descending order of severity:

1. **`a*b + c` silently truncates to integer** (§6). A codegen defect, not a toolchain issue. Nothing
   can be trusted to run correctly until this is fixed, including kernels that appear to work.
2. **Failures are silent end to end** (§3, §4, §5). Clean compile, exit 0, `cudaSuccess` from both
   sync and `getLastError`, unwritten buffers. This is worse than not supporting a feature: it
   defeats every error check tensorcore already has.
3. **Route A coverage excludes math functions, shared memory, and `__shfl_xor_sync`** (§5) — the
   union of which covers every real tensorcore kernel — and the Route A/B boundary is not predictable
   from source (a pure copy fails while an add succeeds).

Re-evaluate when CuMetal (a) fixes FMA codegen, (b) fails loudly — non-zero exit when a kernel gets
no metallib, and a runtime error code rather than a stderr line plus `cudaSuccess`, (c) pins the
emitted AIR version to the installed toolchain or has `doctor` verify it, and (d) covers math
intrinsics and threadgroup reductions in the MSL route.

Worth retrying §4's "did not run" rows against a newer Metal Toolchain (AIR 2.8 / Xcode 26) first —
that is cheap and could move blocker 3 substantially. Blockers 1 and 2 are upstream work regardless.

## 9. Reproduction

Harnesses, bisection cases, and the Metal reference live in `~/Desktop/tensorcore/.scratch/cumetal-eval/`
(untracked scratch, not part of this commit):

- `tc_rmsnorm_cumetal.cu` — tensorcore RMSNorm + AdamW, verbatim, under CuMetal
- `mref.mm` + `bench_extra.metal` — Objective-C++ Metal reference driving `training_kernels.metal`
- `fma.cu` — the §6 truncation reproducer (smallest and most important)
- `gen2.sh` — the §5 feature sweep
- `cm_add.cu` — the §7 CuMetal bandwidth measurement
