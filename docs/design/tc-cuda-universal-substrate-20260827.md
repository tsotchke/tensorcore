# tc-cuda — CUDA as a compatibility on-ramp to the tensorcore substrate

**Date:** 2026-08-27
**Status:** design, not implemented. No library code exists for any part of this.
**Predecessor:** `docs/research/cumetal-eval-20260827.md` (CuMetal 0.1.3, verdict: not viable)
**Related:** `~/Desktop/infinite_context_coder/docs/mothra/cosbox-27b-coder-feasibility-20260827.md`

---

## 1. What tc-cuda is

tc-cuda accepts **CUDA kernel source and CUDA host API calls** and executes them on
tensorcore's backends — Metal today, HIP and portable CPU later. It is the same basic
principle CuMetal attempts. It is not the same engineering contract.

The one-sentence difference: **CuMetal's default behaviour on a construct it cannot handle
is to produce a binary whose kernel silently does nothing. tc-cuda's default behaviour is
to refuse to build.**

tc-cuda is a *compatibility on-ramp*, not the native path. Its job is to let existing CUDA
code — ours in `lib/cuda/`, the Kimi engine's in `~/Desktop/computer_mesh/ops/kimi/`,
third-party kernels from the ecosystem — run on Apple silicon without a hand-port, at a
performance level that makes the port unnecessary rather than merely possible.

### Non-goals for v1, stated so they cannot be assumed

- **No PTX, cubin, or fatbin ingestion.** Source only. The CuMetal evidence is that the
  binary/IR route is precisely where correctness dies (§5 below).
- **No `nvcc` compatibility as a drop-in.** tc-cuda is invoked deliberately, on a declared
  file set, with a declared subset.
- **No "best effort".** There is no mode in which tc-cuda produces output it cannot certify.
- **Not a replacement for `kernels/metal/`.** Hand-written Metal remains the performance
  ceiling and the reference. tc-cuda is measured *against* it.

---

## 2. Requirements, derived by inverting the CuMetal failure modes

Every requirement below is traceable to an observed defect. This table is the spine of the
design; the rest of the document implements it.

| # | Observed in CuMetal 0.1.3 | tc-cuda requirement |
| --- | --- | --- |
| R1 | `cumetalc z_sqrtf.cu -o x` exits 0, prints nothing, emits a binary whose kernel is absent (eval §4) | **Closed kernel manifest.** The compiler enumerates every `__global__` in the translation unit and records its lowering status. Link **fails** if any status is not `lowered`. |
| R2 | Route A "bails" on `__shared__`, libdevice math, `__shfl_xor_sync` with a generic message; Route B says `PTX opcode 'st.param.b32' has no CuMetal IR normalization` (eval §4) | **Named-construct diagnostics.** Every rejection names the CUDA construct in CUDA's own vocabulary, with file:line, and links to the subset table entry. An unnamed rejection is itself a bug. |
| R3 | Missing kernel is a stderr line; `cudaGetLastError()` and `cudaDeviceSynchronize()` both return `cudaSuccess` (eval §3) | **Hard runtime error.** Launching a kernel with no lowered body returns a distinct non-zero status `tcCudaErrorKernelNotLowered` from the launch call itself, and every subsequent sync/error query keeps returning it. There is no path from "no kernel" to "success". |
| R4 | `out[i] = a[i]*b[i] + c[i]` returns `trunc(a*b+c)`, 8 of 8 cases, no diagnostic (eval §6) | **Contraction is specified and tested.** The FMA/contraction policy is declared per translation unit, emitted explicitly, and `.scratch/cumetal-eval/fma.cu` becomes a **permanent conformance vector** promoted into the repo at `tests/conformance/tc-cuda/fma_contraction.cu`. No backend ships without 8/8 bit-exact. |
| R5 | `out[i] = a[i]` fails while `out[i] = a[i] + b[i]` succeeds — the route boundary is not predictable from source (eval §5) | **One route, no fallback.** A single lowering path. There is no second route to silently fall onto, so there is no boundary to be unpredictable about. |
| R6 | `air version set to 2.8.0 ... but expecting 2.7` — the emitted AIR version is hard-coded and does not match the installed toolchain (eval §4) | **Never emit a versioned IR we do not own.** Hand MSL *source text* to `xcrun metal`; the AIR version is then always Apple's. Additionally, a build-time probe compile through the full chain is a hard configure-step gate. |
| R7 | `cumetal doctor` reports all green on a machine where no real kernel can compile (eval §1) | **`tc-cuda doctor` proves, it does not check.** It compiles and *executes* a real kernel end to end and compares against a golden. Presence of a binary on `$PATH` is never evidence. |
| R8 | `__float2half` declared `__device__`-only; NVIDIA declares it `__host__ __device__` (eval §5) | **Host/device attribute parity is part of the subset contract.** Any header symbol whose host/device availability differs from CUDA's is a subset violation, tested by a compile-only host conformance case. |
| R9 | CuMetal's own experimental container passes CuMetal's validator but the Metal runtime will not load it (eval §4) | **The only acceptable validator is the target runtime.** Certification loads the artefact through the real backend and runs it. Self-validation is not evidence. |

R1 needs one escape hatch or it is unusable during bring-up. That hatch is a checked-in
`tc-cuda.waivers` file naming each kernel that is knowingly not lowered, with a reason.
A waiver makes the exception *reviewable in the diff*; it never makes it silent. Waived
kernels still fail at runtime per R3 — a waiver waives the build error, never the execution
guarantee.

---

## 3. The v1 CUDA subset — 55 constructs, surveyed not guessed

The subset is the union of what `~/Desktop/tensorcore/lib/cuda/*.cu` (3,064 lines) and
`~/Desktop/computer_mesh/ops/kimi/*.cu` actually use. It was produced by enumeration over
those sources, not by reading the CUDA programming guide.

### A. Function and declaration forms — 4

| | Construct | Note |
| --- | --- | --- |
| A1 | `__global__ void f(...)` | |
| A2 | `__device__` helper, incl. `inline` / `static` | |
| A3 | `template<int D>` non-type integral template | explicit instantiation only; `training.cu:939,1126` |
| A4 | `__restrict__` on pointer parameters | must lower to a real noalias, not be dropped |

### B. Launch geometry and indexing — 3

| | Construct | Note |
| --- | --- | --- |
| B1 | `threadIdx` / `blockIdx` / `blockDim` / `gridDim`, `.x .y .z` | `.y` is used (`training.cu`, `kimi_moe_cuda.cu`) |
| B2 | `<<<grid, block>>>` | 22 launch sites in `lib/cuda` alone |
| B3 | `<<<grid, block, dynamic_smem_bytes>>>` | paired with C2 |

### C. Memory — 4

| | Construct | Note |
| --- | --- | --- |
| C1 | `__shared__` static-extent array | 9 uses in `lib/cuda`, 7 in Kimi |
| C2 | `extern __shared__` dynamic shared array | 3 uses in `lib/cuda` |
| C3 | module-scope `__device__` variable + `cudaMemcpyToSymbol` / `FromSymbol` | 2 uses |
| C4 | global pointer load/store incl. `const T* __restrict__` | 87 `__restrict__` uses |

### D. Synchronisation and warp collectives — 4

| | Construct | Note |
| --- | --- | --- |
| D1 | `__syncthreads()` | 19 + 14 uses |
| D2 | `__shfl_sync` | |
| D3 | `__shfl_xor_sync` | 12 uses in `lib/cuda`; **highest-risk construct, see §7** |
| D4 | `__shfl_down_sync` | |

### E. Atomics — 1

| | Construct | Note |
| --- | --- | --- |
| E1 | `atomicAdd(float* /* global */, float)` | 11 + 3 uses. Shared-memory and integer atomics are **out**. |

### F. Half precision — 8

| | Construct | Note |
| --- | --- | --- |
| F1 | `__half` scalar, **host and device** storage | 144 uses; R8 applies |
| F2 | `__half2` packed | 52 uses |
| F3 | `__half2float` / `__float2half`, `__host__ __device__` both | 57 / 9 uses |
| F4 | `__float2half_rn` | 13 uses |
| F5 | `__float2half2_rn` | 5 uses |
| F6 | `__hfma2` | 5 uses |
| F7 | `__hmul2` | |
| F8 | `__ushort_as_half` | 7 uses; bit-cast, must not round-trip through float |

### G. fp32 libdevice math — 13

`sqrtf`, `rsqrtf`, `expf`, `logf`, `tanhf`, `powf`, `sinf`, `cosf`, `fmaxf`, `fminf`,
`fabsf`, `floorf`, `fmaf`.

Each is a separate construct because each needs its own numerical golden. CuMetal's Route A
rejected this entire class; MSL's `metal::` namespace has a direct counterpart for all 13,
so the gap there is effort, not feasibility.

### H. Rounding-explicit and fast intrinsics — 4

| | Construct | Note |
| --- | --- | --- |
| H1 | `__expf` | 6+2 uses; maps to `metal::fast::exp`, **not** `metal::exp` |
| H2 | `__logf` | 2 uses |
| H3 | `__fmul_rn` | 3 uses; **contraction barrier** — must not be fused |
| H4 | `__fadd_rn` | 3 uses; **contraction barrier** |

H3/H4 are load-bearing for R4. Where a kernel author wrote `__fmul_rn`, they explicitly
asked *not* to be contracted; a lowering that fuses them anyway is the CuMetal bug in
mirror image.

### I. Integer and bit arithmetic — 2

| | Construct | Note |
| --- | --- | --- |
| I1 | `uint8_t` / `unsigned short` / `int` / `long` / `size_t` arithmetic, shifts, masks, casts | the whole of `dequant_q4_t_kernel` is this |
| I2 | integer division and modulo by a runtime value | `idx / nblocks`, `idx % nblocks` |

### J. Host runtime API — 12 families

`cudaMalloc` · `cudaMallocHost` · `cudaMallocManaged` · `cudaFree` · `cudaMemcpy` (+`Async`,
4 directions) · `cudaMemset` · `cudaMemcpyToSymbol`/`FromSymbol` · `cudaStream*`
(Create/WithFlags/Destroy/Synchronize) · `cudaEvent*` (Create/Record/Synchronize/
ElapsedTime/Destroy) · `cudaDeviceSynchronize` · error surface (`cudaGetLastError`,
`cudaGetErrorName`, `cudaSuccess`, `cudaError_t`) · device query
(`cudaGetDevice`/`SetDevice`/`GetDeviceCount`/`GetDeviceProperties`/`cudaPointerGetAttributes`/
`cudaFuncSetAttribute`).

`cudaMallocManaged` is nearly free on unified memory and is one of the few places tc-cuda is
structurally *better* off than CUDA.

### Total: 4+3+4+4+1+8+13+4+2+12 = **55 constructs**

### Explicitly out of scope in v1 — every one is a build error naming the construct

`wmma` / `mma_sync` / any tensor-core intrinsic · inline PTX (`asm volatile`) ·
`cp.async` · `ldmatrix` · cooperative groups · CUDA graphs · dynamic parallelism ·
texture and surface objects · `__constant__` proper · warp-vote (`__ballot_sync`,
`__any_sync`, `__all_sync`) · `__syncwarp` · `__ldg` · shared-memory atomics · integer
atomics · `atomicCAS`/`Exch`/`Max` · double-precision math · device-side `printf` ·
device-side `malloc`/`free`/`assert` · recursion · virtual functions and RTTI in device
code · type-parametric templates (only non-type integral in A3) · `__launch_bounds__` ·
multi-GPU peer access · stream callbacks · `float4`/`double2` vector types.

Two of these deserve a note. **`__ldg`** and **`__launch_bounds__`** are performance hints,
not semantics; the tempting move is to accept and ignore them. Do not — an ignored hint is a
silent performance cliff, which is R1's failure mode wearing a different hat. Reject them in
v1 and add them deliberately when there is a backend mapping to point at.

---

## 4. cuBLAS is a separate surface, and it is the bigger half

The survey turned up something the CuMetal evaluation did not reach: **the Kimi engine's
GEMMs are not kernels at all.** `~/Desktop/computer_mesh/ops/kimi/*.cu` contains 33 cuBLAS
call sites — `cublasSgemm` (23), `cublasGemmStridedBatchedEx` (18), `cublasGemmBatchedEx`
(10), `cublasGemmEx` (9), `cublasSetMathMode`, `cublasLt`. The hand-written `__global__`
kernels there are the *glue*: dequant, MLA fusion, routing. All the FLOPs go through cuBLAS.

So tc-cuda has two surfaces, and they need different treatment:

**Surface 1 — kernel lowering.** CUDA-C source in, backend kernel out. Certified by
per-backend numerical goldens. This is §3 and §5.

**Surface 2 — library API interception.** `cublasGemmEx(...)` in, a `tc_gemm` call out. No
lowering happens; this is an ABI shim that translates descriptors (layout, transpose flags,
`cudaDataType` compute type, alpha/beta, batch strides) onto tensorcore's existing GEMM.
Certified differently: against cuBLAS's *own* output captured on a real NVIDIA host, since
there is no source to derive a golden from.

Conflating the two is a design error waiting to happen. A cuBLAS call that tc-cuda cannot
map — a math mode we do not implement, an epilogue in `cublasLt` — must be a **link-time**
error naming the entry point, on the same R1/R2 terms as an unlowerable kernel. It must not
degrade to a slow path, and it must never be a no-op.

Surface 2 is out of scope for v1 lowering work but its *error behaviour* is in scope from
day one: v1 links against a cuBLAS stub whose every entry point is a named hard error. That
costs almost nothing and it means the first person to point tc-cuda at the Kimi engine gets
`tc-cuda: cublasGemmStridedBatchedEx at kimi_moe_cuda.cu:412 is not implemented` instead of
a mystery.

---

## 5. Lowering route — two candidates, one recommendation

### Route (a) — CUDA-C → MSL source, via a clang AST consumer

Parse the `.cu` with clang's CUDA frontend (libTooling), walk the AST, emit MSL text, hand
the text to `xcrun metal` / `xcrun metallib`.

### Route (b) — NVVM-IR → AIR, via LLVM

CuMetal's route. Compile to NVVM/PTX, normalise the IR, stamp it as AIR, hand it to
`xcrun metal`.

### Comparison against the R-requirements

| | Route (a) source → MSL | Route (b) NVVM-IR → AIR |
| --- | --- | --- |
| **R2 named diagnostics** | The construct is still *present*. `__shfl_xor_sync` is a `CallExpr` to a named function at a known `SourceLocation`. The diagnostic writes itself. | The construct is gone by the time we see it. The best available message is CuMetal's actual message: `PTX opcode 'st.param.b32' has no CuMetal IR normalization` — unactionable to the kernel author. **Fails R2 structurally.** |
| **R4 contraction** | We choose, per expression, whether to emit `a*b+c`, `fma(a,b,c)`, or a barrier-preserving form, and we can honour H3/H4. | Contraction happens inside a pipeline we do not own. The 8/8 truncation bug lives exactly here. |
| **R6 AIR version** | Never emits AIR. Apple's compiler sets the version. The eval §4 failure is *structurally impossible*. | Requires stamping a version we do not own, tracking Apple's private AIR versioning forever. A build-time probe reduces the blast radius but does not remove the dependency. |
| **R5 one route** | Natural: one frontend, one emitter. | Also possible, but (b) tends to grow a source fallback for what the IR path cannot normalise — which is exactly how CuMetal acquired its unpredictable boundary. |
| **Multi-backend reach** | N textual emitters over **one** frontend AST. HIP is nearly identity from CUDA-C (this is what `hipify` does). CPU is a grid-loop emission. | N LLVM backends. The Metal one (AIR) is not upstream, not documented, and not ours. |
| **Ecosystem binaries** | Cannot ingest PTX/cubin. | Could, in principle. |
| **Cost** | A libTooling consumer plus per-backend emitters. Real work, but *our* work, and debuggable by reading the emitted MSL. | Smaller if AIR emission worked. It does not, on this toolchain. |

### Recommendation: **Route (a)**, with a hard rule that v1 ships no IR route at all.

Three reasons, in order of weight.

**First, Route (a) is the only one that can satisfy R2, and R2 is the whole point.** The
entire thesis of tc-cuda is that unsupported means *named error*, not silence. A construct
name is a source-level concept. Once you are in NVVM-IR you have destroyed the information
the diagnostic needs. Everything else in this document is downstream of that.

**Second, the CuMetal evidence points here, hard.** Their Route A — the MSL source route —
is the one that produces bit-exact results (eval §7: 437.5 GB/s, 89% of hand-written Metal,
bit-exact). Their Route B is the one that emits AIR 2.8 against a 2.7 toolchain and kills
every real kernel. Route A's coverage gaps are `__shared__`, libdevice math, and
`__shfl_xor_sync` — and MSL has `threadgroup`, `metal::exp`/`sqrt`/`rsqrt`/`tanh`/`pow`, and
`simd_shuffle_xor` respectively. Every gap has a direct counterpart. **CuMetal's working
route is unfinished, not infeasible; its finished route is broken by construction.**

**Third, "universal substrate" argues for (a), not against it.** The intuition that one IR
reaches many targets is right in general and wrong for *our* target set. Our targets are
Metal (no upstream LLVM backend), HIP (already CUDA-C source-compatible), and portable CPU
(a grid loop). All three are cheaper from an AST than from IR.

The cost is real and should be stated plainly: we give up ecosystem *binaries*, and a
libTooling frontend is a larger v1 than shimming an existing compiler. The first is a
declared non-goal (§1). The second buys R1 through R9, which CuMetal does not have at any
price.

### Sketch of the lowering, for the constructs that carry risk

| CUDA | MSL |
| --- | --- |
| `__global__ void f(T* p, int n)` | `kernel void f(device T* p [[buffer(0)]], constant int& n [[buffer(1)]], uint3 tid [[thread_position_in_threadgroup]], uint3 gid [[threadgroup_position_in_grid]], uint3 tgs [[threads_per_threadgroup]])` |
| `threadIdx.x` / `blockIdx.x` / `blockDim.x` | `tid.x` / `gid.x` / `tgs.x` |
| `__shared__ float s[N]` | `threadgroup float s[N]` |
| `extern __shared__ char s[]` | `threadgroup char* s [[threadgroup(0)]]` + explicit `setThreadgroupMemoryLength` at dispatch |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup \| mem_flags::mem_device)` — **both** flags; CUDA's barrier orders global accesses too |
| `__shfl_xor_sync(0xffffffff, v, m)` | `simd_shuffle_xor(v, m)` — **only** for full-mask, non-divergent; see §7 |
| `atomicAdd(gp, x)` | `atomic_fetch_add_explicit((device atomic_float*)gp, x, memory_order_relaxed)` |
| `__half` | `half` (device); `_Float16` (host) with a `__host__ __device__` conversion shim per R8 |
| `__ushort_as_half(s)` | `as_type<half>(s)` — bit-cast, never arithmetic |
| `__expf(x)` | `metal::fast::exp(x)` |
| `expf(x)` | `metal::exp(x)` |
| `__fmul_rn(a,b)` | `a * b` inside `#pragma METAL fp math_mode(safe)` region, or a `precise::` form; **never** contractible |
| `a*b + c` (default) | `fma(a, b, c)`, matching nvcc's default `-fmad=true` — declared, emitted, and tested by R4's vector |

---

## 6. Certification — the device dimension is P0 and it is shared work

`mothra.kernel_certificate.v1` today has **no device dimension**. The certifier at
`~/Desktop/infinite_context_coder/scripts/mothra_kernel_certify_service.py` builds rows keyed
`(kernel, kind, dtype)` — line 703, `key = (entry.kernel, kind, entry.dtype)` — and the
single receipt on disk is 40 rows of CPU-torch Riemannian primitives. "Kernel" there means
*mathematical primitive*, not GPU compute kernel. The service source contains no occurrence
of cuda, gpu, metal, device, or sm_86. Zero GPU kernels certified is a **schema gap**, not an
empty table.

tc-cuda cannot be certified without fixing that, and neither can the cosbox Ampere
qualification the mothra lane needs. **It is one change and it serves both.** Doing it once,
first, is the reason P0 exists as its own phase.

### The schema change

Extend each row's key from `(kernel, kind, dtype)` to
`(kernel, kind, dtype, backend, arch)`, and add a required per-row `device` block:

```
"device": {
  "backend":   "metal" | "cuda" | "hip" | "cpu",
  "arch":      "apple-m2-ultra" | "sm_86" | "sm_120" | "arm64",
  "toolchain": "xcrun metal 32023.619 (Xcode 16.2, AIR 2.7)",
  "driver":    "macOS 15.1",
  "source":    "tc-cuda" | "handwritten" | "eshkol" | "reference"
}
```

Three rules that make it a gate rather than a label:

1. **A declared device that did not run is a FAIL, not an omission.** If the certificate's
   scope names `backend: metal` and no Metal row appears, the verdict is FAIL. This is the
   direct inversion of CuMetal's "did not run" rows reading as blank.
2. **`source` is mandatory and compared.** A tc-cuda row and a handwritten row for the same
   `(kernel, dtype)` are cross-checked against each other, not only against the golden.
3. **`toolchain` is captured from the tool, not configured.** R6 and R7 depend on the
   receipt recording what actually compiled the artefact.

The existing 40-row CPU receipt must re-emit PASS with `backend: "cpu"`, `source:
"reference"` populated — a zero-behaviour-change migration is the proof the schema extension
is sound.

### Conformance vectors — permanent, per backend

Promoted out of scratch into `tests/conformance/tc-cuda/`:

- **`fma_contraction.cu`** — the eval §6 reproducer, verbatim. 8 rows, bit-exact. CuMetal
  scores 0/8. This is the single highest-value test in the suite: it is small, it is fast,
  and it catches the class of defect that is otherwise invisible.
- **`shuffle_semantics.cu`** — full-mask reduction, and a *divergent* shuffle that must be
  **rejected at compile time**, not lowered (see §7).
- **`barrier_scope.cu`** — a `__syncthreads()` ordering a global write, proving the
  `mem_device` flag is present.
- **`half_host_device.cu`** — compile-only; `__float2half` used on the host (R8).
- **`silent_noop.cu`** — a kernel using an out-of-scope construct. Asserts non-zero exit,
  asserts the construct name is in the message, asserts no binary was produced. This is the
  test that CuMetal would fail and the reason tc-cuda exists.

Negative tests are first-class here. A suite that only tests what works cannot distinguish
tc-cuda from CuMetal.

---

## 7. The highest-risk assumption

**That CUDA warp semantics survive the translation to Metal SIMD-group semantics.**

`__shfl_xor_sync` appears 12 times in `lib/cuda/training.cu`, all inside
`block_reduce_sum_f32` and its callers. The lowering to `simd_shuffle_xor` looks like a
one-liner. It rests on three assumptions, and each can be wrong quietly:

1. **Width.** CUDA's warp is architecturally 32. Metal's SIMD-group is 32 on every shipping
   Apple GPU but is an *implementation detail*, queryable at runtime as
   `threads_per_simdgroup`, not a language guarantee. A reduction whose loop bound is a
   literal 32 is correct today and silently wrong on any future part that differs.
2. **Mask.** `__shfl_xor_sync` takes a participation mask. `simd_shuffle_xor` does not — it
   has no expression for partial participation. A non-`0xffffffff` mask has **no correct
   lowering**, only a plausible-looking one.
3. **Divergence.** CUDA post-Volta has independent thread scheduling and `_sync` intrinsics
   carry a reconvergence guarantee. Metal's SIMD-group shuffles under divergence are
   undefined. A shuffle inside an `if` is the failure case, and it will not announce itself.

This is the assumption most likely to produce a CuMetal-shaped outcome — a kernel that
compiles, dispatches, returns success, and is numerically wrong under a load pattern that
does not show up in a golden.

**Mitigation, and it must be in v1 or the guarantee is hollow:**

- Reject any `__shfl_*_sync` whose mask argument is not the literal `0xffffffff`, by name,
  at compile time.
- Reject any `__shfl_*_sync` the frontend cannot prove is outside divergent control flow.
  Conservative: if the enclosing region is not straight-line from the kernel entry or a
  uniform-condition branch, refuse. Over-rejection is the correct error direction.
- Emit `threads_per_simdgroup` as a runtime check against the assumed width, and fail the
  dispatch if it differs. Never fold 32 in as a literal.
- `shuffle_semantics.cu` is a permanent conformance vector for all three.

The honest position: this mitigation makes tc-cuda **reject some CUDA that is legal and
correct on NVIDIA**. That is the trade. R1 through R9 are worth a subset that is smaller than
advertised; they are not worth a subset that is silently wrong.

---

## 8. Relationship to the Eshkol bridge — two roads, one gate

These must not compete, and the way to prevent it is to say plainly what each is for.

| | **tc-cuda** | **Eshkol codegen** |
| --- | --- | --- |
| Input | existing CUDA-C source | Eshkol source |
| Purpose | **compatibility on-ramp** — run code that already exists | **native path** — the long-term single source |
| Users | our `lib/cuda/`, the Kimi engine, third-party kernels | new kernels, everything greenfield |
| Ceiling | whatever the CUDA source expresses | whatever the substrate can express |
| Lifetime | as long as CUDA source exists in the ecosystem — indefinite | permanent |
| Certification | `mothra.kernel_certificate` with `source: "tc-cuda"` | same schema, `source: "eshkol"` |

**They share the gate, and the gate is the integration point.** The device dimension from §6
is built once and both use it. A kernel that exists on both paths produces two rows with the
same `(kernel, kind, dtype, backend, arch)` and different `source` — and the certifier
cross-checks them. That is a feature: it turns the two efforts into each other's oracle
instead of into competitors for the same budget.

Current honest state of the Eshkol side, for calibration:
`eshkol/bridge/tensorcore_codegen.cpp` is 210 lines declaring 14 `tc_*` symbols as
`ExternalLinkage`. It emits no device code — no PTX, no MSL, no nvcc invocation, no backend
selection. `tensorcore_bridge_smoke.esk`'s documented expected result is
`backend=portable-cpu`. The eshkol repo has no NVPTX target and no MSL emission; its 26 MSL
kernels are hand-written C string literals in `lib/backend/gpu/metal_softfloat.h`.

So the sequencing is not a conflict at all: **tc-cuda is nearer to producing a certified GPU
kernel than the Eshkol bridge is**, and the device-dimension work it forces is the exact
prerequisite the Eshkol bridge will need when it gets there. tc-cuda goes first *because* it
unblocks the native path, not in spite of it.

The rule that keeps them from drifting: **neither path may ship a kernel that the other
path's certificate schema cannot describe.**

---

## 9. Phase plan

Each exit is a measurement or an artefact. None is "implemented".

### P0 — device dimension in the certificate schema

Scope: `~/Desktop/infinite_context_coder/scripts/mothra_kernel_certify_service.py` only.
No tensorcore code. Serves the cosbox Ampere lane identically.

**Exits:**
1. Row key is `(kernel, kind, dtype, backend, arch)`; the `device` block of §6 is required.
2. A scope declaring a backend that produced no rows yields **FAIL** with a named blocker.
3. `bin/icc mothra-kernel-certify` accepts a device selector and records the *observed*
   toolchain string, not a configured one.
4. The existing 40-row receipt re-emits **PASS** with `backend: "cpu"`, `source:
   "reference"` — zero behaviour change on the migration.
5. A hand-written Metal kernel (`tc_rmsnorm_forward`) is certified as the first non-CPU row.
   This is what makes P0 real rather than a schema edit.

### P1 — one real kernel: rmsnorm fp16

`rmsnorm_forward_kernel` from `lib/cuda/training.cu`, **verbatim**, lowered by tc-cuda to
Metal. Chosen because it has an exact hand-written counterpart to measure against, and
because it is the kernel CuMetal could not run.

**Exits:**
1. Compiles verbatim, no source edits, with a kernel manifest showing `lowered`.
2. Certificate row `{kernel: rmsnorm_forward, dtype: fp16, backend: metal,
   arch: apple-m2-ultra, source: tc-cuda}` = **PASS**, at the same tolerance the
   handwritten row uses.
3. Cross-check against the `source: "handwritten"` row passes.
4. **≥ 218.8 GB/s** compulsory on 8192×4096 — 80% of the handwritten `tc_rmsnorm_forward`
   measurement of 273.5 GB/s (eval §7).
5. `fma_contraction.cu` passes **8/8 bit-exact**.
6. `silent_noop.cu` passes: a kernel using `__ballot_sync` exits non-zero, the message
   contains the string `__ballot_sync`, and no binary is produced.
7. `shuffle_semantics.cu` passes, including the rejection cases of §7.

Exit 6 is not optional decoration. It is the exit that distinguishes this project from the
one we evaluated and rejected.

### P2 — the dequant_q4_t family

`dequant_q4_t_kernel` and `dequant_q8_t_kernel` from
`~/Desktop/computer_mesh/ops/kimi/kimi_dense_cuda.cu` and `kimi_moe_cuda.cu`. These are pure
integer/bit-manipulation plus `__ushort_as_half` — no reduction, no shuffle, no float
associativity. They should be **bit-exact**, not tolerance-matched, and if they are not, the
lowering has a defect worth finding before anything harder is attempted.

**Exits:**
1. Both kernels certified **bit-identical** to CUDA reference output over the same Q4_0
   blob. Not a tolerance. Any deviation is a bug.
2. The cuBLAS shim decision is landed either way: `cublasGemmEx` and
   `cublasGemmStridedBatchedEx` either intercept to `tc_gemm` with a certified row, or they
   are named hard errors at link time. **Not** a silent absence.
3. One Kimi dense layer runs end to end on Metal via tc-cuda, logits matching the CUDA run
   within the engine's own tolerance.
4. Subset coverage measured: the fraction of `__global__` kernels across `lib/cuda/` and
   `ops/kimi/` that lower without waiver, reported as a number. This is the honest analogue
   of eval §5's coverage table, and the first phase where it can be non-trivial.

### P3 — subset freeze and docs

**Exits:**
1. The 55-construct table is machine-readable at `docs/tc-cuda/subset.v1.json`, and the
   compiler's accept-list is **generated from it**. Doc and code cannot drift because they
   are one artefact.
2. Every out-of-scope construct in §3 has a negative test asserting non-zero exit and the
   construct name in the message. Coverage of the out-of-scope list is 100% or the freeze
   does not happen.
3. `tc-cuda doctor` compiles *and executes* a real kernel against a golden through the full
   chain and refuses to report green on anything less (R7).
4. `docs/tc-cuda/` published: subset reference, diagnostic catalogue, waiver-file format,
   porting guide for the out-of-scope constructs.
5. Subset v1 frozen. Additions go to v2 with the same survey-then-enumerate discipline —
   never by a compiler silently starting to accept something.

---

## 10. Open questions

1. **Does clang's CUDA frontend parse our `.cu` without a CUDA installation present?** It
   needs `cuda_fp16.h` and friends. Options: vendor a minimal compatible header set (which
   R8 requires anyway, for host/device attribute parity), or require a CUDA toolkit for
   headers only. The vendored-header route is more work and more control; it is probably
   right, but it is untested.
2. **Threadgroup memory limits.** Metal's 32 KB per threadgroup is tighter than CUDA's
   configurable 48–228 KB. `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` has no honest
   Metal equivalent above 32 KB. A request over the limit must be a **launch-time hard
   error** naming the requested size, never a silent clamp.
3. **`cudaMallocManaged` on unified memory** is close to free, but the coherence model is
   not identical. Worth measuring before claiming it as a win.
4. **HIP backend ordering.** HIP is nearly source-identical to CUDA, so the tc-cuda HIP
   "emitter" is close to a pass-through. That makes it cheap, and it makes it a good early
   proof that the AST layer is genuinely backend-agnostic rather than a Metal emitter with
   an AST in front of it. Consider it before P3.
5. **AIR 2.8 / Xcode 26.** The eval flagged installing a newer Metal Toolchain as the
   highest-value untested variable for CuMetal. Under Route (a) it is *irrelevant* — we
   never emit AIR. That is itself a point in Route (a)'s favour worth recording: the
   recommendation does not depend on a system mutation to a live serving host.
