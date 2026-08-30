# CUDA on Metal ecosystem survey — first-party direction for tensorcore

**Date:** 2026-08-28  
**Decision target:** whether tensorcore should build first-party CUDA source/API compatibility on Apple Silicon  
**Method:** current primary-source internet survey reconciled against ICC's tensorcore architecture, feature inventory, long memory, and the M2 Ultra CuMetal evaluation

## Executive conclusion

Tensorcore should build a first-party CUDA compatibility layer for Apple Silicon.
The project is technically feasible, strategically aligned with tensorcore, and
materially de-risked by existing systems that demonstrate each required piece.
The correct v1 is not a CUDA binary emulator. It is:

1. a CUDA-C++ source frontend built on Clang;
2. a typed, inspectable CUDA-semantics representation;
3. one fail-closed lowering path to MSL source;
4. a CUDA Runtime/Driver compatibility ABI backed by tensorcore buffers,
   streams, devices, and Metal dispatch; and
5. CUDA-X interception that maps expensive library operations to tensorcore's
   existing kernels.

```text
CUDA application
    |
    +-- __global__ source --> tc-cuda frontend --> typed GPU semantics
    |                                              |
    |                                              +--> MSL --> metallib --> Metal
    |
    +-- CUDA Runtime ------> tc-cuda runtime ------+--> tc_buffer / tc_stream / device
    |
    +-- cuBLAS/cuDNN/NCCL -> compatibility shims --+--> tc_gemm / attention / conv / dist
```

PTX, fatbin, cubin, SASS, and drop-in binary compatibility should be a separate
post-v1 decision. They are not required to make tensorcore's own CUDA kernels or
source-available ecosystem software run on Macs.

## Evidence from the ecosystem

### CuMetal: direct proof on Apple Silicon

[CuMetal](https://github.com/Lulzx/cuda-metal) is the closest comparator. It
implements CUDA source compilation, PTX lowering, a CUDA-compatible runtime,
partial CUDA-X libraries, and Metal execution. Its current compiler architecture
converges CUDA/NVVM and PTX frontends into a typed SSA GPU IR, legalizes that IR
to typed MSL, and leaves final code generation to Apple's supported Metal tools.
Unsupported constructs in the new backend are compile-time errors rather than
automatic fallback.

Relevant validated design choices:

- CUDA-facing headers can be clean-room implementations.
- Clang can provide the CUDA language frontend.
- A typed intermediate representation can make address spaces, memory ordering,
  synchronization, pointer provenance, and Metal legalization explicit.
- MSL is the stable boundary; Apple owns AIR generation and versioning.
- Source-native registration can avoid fatbinary parsing entirely.
- CUDA's 32-wide warp maps naturally onto Metal SIMD groups of width 32.
- Runtime provenance and semantic quality must be separate fields.

CuMetal also demonstrates the cost of insufficient semantic gates. Tensorcore's
M2 Ultra evaluation of v0.1.3 found silent missing kernels and a silent FMA wrong
answer. CuMetal subsequently added stricter typed lowering and correctness work,
but v0.2.1 itself fixed another broad silent floating-point typing bug. The lesson
is not that the project is infeasible; it is that compiler acceptance can never
serve as correctness evidence.

Sources:

- [CuMetal README](https://github.com/Lulzx/cuda-metal)
- [CuMetal compiler architecture](https://github.com/Lulzx/cuda-metal/blob/main/docs/compiler-architecture.md)
- [CuMetal status](https://github.com/Lulzx/cuda-metal/blob/main/docs/status.md)
- [CuMetal v0.2.1 release](https://github.com/Lulzx/cuda-metal/releases/tag/v0.2.1)
- tensorcore's [v0.1.3 hardware evaluation](cumetal-eval-20260827.md)

### SCALE: proof that source-compatible CUDA need not be NVIDIA-owned

[SCALE](https://docs.scale-lang.com/stable/) supplies a replacement CUDA
compiler plus Runtime/Driver/math APIs and CUDA-X wrappers for AMD GPUs. It
compiles nvcc-dialect CUDA without requiring applications to migrate to another
source language. SCALE validates the product shape tensorcore should pursue:
preserve CUDA source, replace the compiler/runtime, and delegate high-level
libraries to an optimized native substrate.

SCALE is not a Metal implementation and is not an implementation dependency.
Its relevance is architectural: source compatibility is a viable product, not
merely a one-off transpiler.

### ZLUDA: proof of binary/API compatibility, and a warning about scope

[ZLUDA](https://github.com/vosen/ZLUDA) implements a drop-in CUDA replacement on
ROCm/HIP. It contains PTX parsing/lowering, driver interception, caching, and
partial cuBLAS, cuDNN, cuSPARSE, cuFFT, NCCL, and OptiX compatibility. It has run
substantial unmodified applications and demonstrates that a CUDA ABI shim can be
useful independently of source migration.

Its limitations are equally instructive:

- binary compatibility makes PTX availability a deployment constraint;
- CUDA-X breadth dominates real application compatibility;
- startup JIT and cache identity become product concerns;
- differences in rounding, denormals, warp geometry, and vendor intrinsics can
  surface as numerical or liveness failures; and
- compatibility breadth can consume the project before its core semantics are
  certifiable.

ZLUDA supports a later binary-compatibility track. It does not justify putting
that track on tensorcore v1's critical path.

### HIPIFY and Clang: proof that the frontend is available

LLVM Clang already parses CUDA as a language and exposes the AST, source
locations, semantic analysis, and diagnostics required by a source-to-MSL
compiler. AMD's HIP/HIPIFY ecosystem demonstrates that most ordinary CUDA
device source can be transformed or compiled through a compatible C++ frontend,
while CUDA library calls need explicit API mapping.

This supports a Clang LibTooling/AST consumer rather than building a CUDA parser
from scratch. It also preserves source-level constructs long enough to issue
diagnostics such as `unsupported __ballot_sync at file.cu:42`, which are no
longer naturally available after lowering to PTX.

Sources:

- [Clang compiler documentation](https://clang.llvm.org/docs/UsersManual.html)
- [Clang CUDA language selection](https://clang.llvm.org/docs/CommandGuide/clang.html)
- [AMD HIPIFY documentation](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/)

### Public specifications are sufficient for the initial surface

NVIDIA publishes the CUDA programming guide, Runtime API, Driver API, math and
library references, and PTX ISA. Apple publishes the Metal Shading Language
specification and Metal framework documentation. These sources describe enough
observable behavior to implement a source/API-compatible subset without
decompiling NVIDIA software.

Sources:

- [CUDA documentation](https://docs.nvidia.com/cuda/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Apple Metal resources](https://developer.apple.com/metal/resources/)
- [Apple Metal libraries](https://developer.apple.com/documentation/metal/metal-libraries)

## Legal boundary, stated narrowly

This is an engineering survey, not legal advice. The current CUDA SDK agreement
prohibits reverse engineering, decompiling, or disassembling the SDK. It also
prohibits reverse engineering SDK-generated output for translation to a
non-NVIDIA platform. That language creates avoidable contractual risk around a
pipeline that invokes `nvcc` and translates its PTX/fatbin/cubin output.

It does not make an independently implemented CUDA-compatible source language,
runtime API, or library ABI technically impossible. The conservative v1 boundary
is therefore:

- do not decompile or disassemble NVIDIA binaries;
- do not copy NVIDIA SDK headers or source without a component-specific license;
- do not require `nvcc` or consume its generated artifacts;
- derive interfaces and behavior from public specifications and independently
  authored conformance programs;
- retain provenance for compatible headers and behavioral tests; and
- obtain jurisdiction-specific counsel before distributing PTX/binary
  translation or using CUDA trademarks in product naming.

Source: [CUDA 13.3 EULA, section 1.2](https://docs.nvidia.com/cuda/eula/).

## ICC-grounded fit with tensorcore

ICC reports that tensorcore already owns the expensive execution substrate:

- 274 indexed public `tc_*` symbols;
- first-class buffer, device, stream, backend, and capability contracts;
- optimized GEMM, attention, convolution, normalization, activation,
  quantization, optimizer, and distributed operations;
- physical Metal, CUDA, and portable-CPU evidence;
- a direct CUDA backend containing managed allocation and native CUDA kernels;
- dispatch provenance through `tc_record_dispatch`; and
- fail-closed release/evidence policy that rejects inferred backend success.

The existing design surveyed tensorcore's CUDA kernels and Kimi kernels into a
bounded v1 set of 55 CUDA construct/API families. This is the critical leverage:
we do not need to implement all of CUDA to deliver immediate value.

The compiler layer should be generic CUDA compatibility owned alongside
tensorcore, but it must not absorb Eshkol-specific language lowering. ICC's
existing ownership boundary remains correct: Eshkol owns Eshkol AST/IR/codegen;
tc-cuda owns CUDA compatibility against tensorcore's public substrate.

## Build, borrow, and reject matrix

| Capability | Decision | Reason |
| --- | --- | --- |
| CUDA C++ parsing/sema | Borrow Clang | Mature CUDA dialect, AST, source locations, diagnostics |
| CUDA-compatible headers | Build clean-room minimal set | Control licensing and exact v1 surface |
| Typed GPU semantics IR | Build | Central authority for CUDA semantics and backend legalization |
| CUDA source to MSL | Build | Core differentiator; direct access to tensorcore and Metal contracts |
| MSL to AIR/metallib | Borrow Apple tools | Public supported boundary; avoids private AIR coupling |
| Runtime memory/streams/events | Build over tensorcore/Metal | Existing substrate and unified-memory advantage |
| cuBLAS compatibility | Build thin shim to `tc_gemm` | Reuses optimized, certified implementation |
| cuDNN compatibility | Build thin shims to tensorcore ops | Attention/conv/norm primitives already exist |
| NCCL compatibility | Build thin shim to `tc_dist` | Preserve tensorcore transport and identity semantics |
| PTX parser/lowering | Defer and reassess | Useful for binaries; higher semantic and contractual risk |
| fatbin/cubin registration | Defer | Not needed for source-first value |
| SASS execution/decompilation | Reject | Hardware-specific, undocumented, unnecessary, high risk |
| CuMetal source | Study; selectively reuse only after review | Apache-2.0 permits reuse with obligations, but wholesale adoption imports architecture and correctness debt |
| ZLUDA source | Study for ABI/test patterns | AMD/PTX architecture is useful but not a Metal substrate |

## Recommended architecture

### 1. Frontend

Use Clang LibTooling to enumerate every `__global__` kernel and its reachable
device-call closure. Reject any construct not present in a machine-readable
subset manifest. The manifest generates both compiler acceptance tables and
documentation so they cannot drift.

### 2. Typed semantics layer

Represent CUDA concepts explicitly before Metal legalization:

- grid/block/thread indices;
- global, shared/threadgroup, constant, and private address spaces;
- barriers and memory scopes;
- warp collectives with active-mask constraints;
- atomics and ordering;
- half/half2 conversions and bit casts;
- floating-point contraction and rounding policy; and
- logical kernel arguments independently of Metal binding indices.

This layer should not model PTX registers. It should model the CUDA source
semantics tensorcore promises.

### 3. MSL legalization and emission

Lower typed CUDA operations to typed MSL and pass emitted source to `xcrun
metal` and `xcrun metallib`. Never stamp AIR versions. Retain emitted MSL as an
inspectable build artifact and include compiler/schema identity in cache keys.

### 4. Runtime and registration

Define a versioned source-native module descriptor containing:

- embedded metallib bytes or an installed metallib reference;
- kernel names and host-stub identities;
- logical argument descriptors;
- concrete Metal binding descriptors;
- static and dynamic threadgroup-memory requirements;
- required SIMD width and device features; and
- compiler/subset/schema versions.

Validate the complete descriptor before making any kernel launchable. A missing
or unlowered kernel must produce a sticky non-success CUDA-compatible error.

### 5. CUDA-X interception

Treat library compatibility separately from kernel compilation. Translate
descriptors and layouts, then call tensorcore. Unsupported math modes,
epilogues, layouts, or datatypes must be named link/runtime errors—never CPU
fallbacks presented as successful GPU compatibility.

## Execution plan and hard gates

### P0 — governance and conformance foundation

- Land the clean-room/provenance policy.
- Land `subset.v1.json` and generate documentation plus compiler tables.
- Promote the existing FMA, missing-kernel, and sentinel-output reproducers.
- Add device/backend/compiler/subset dimensions to runtime certificates.
- Gate: every unsupported construct fails with file, line, and CUDA name.

### P1 — source frontend and vector execution

- Compile vector add directly from `.cu` through Clang AST to MSL.
- Implement allocation, copy, launch, synchronize, and error APIs.
- Use source-native module registration; no PTX or fatbinary path.
- Gate: bit-exact output, Metal provenance, and malformed-descriptor negatives.

### P2 — tensorcore's real kernel subset

- Implement shared memory, barriers, shuffles, fp16/half2, atomics, and math.
- Compile tensorcore RMSNorm and AdamW verbatim.
- Gate: numerical parity and at least 80% of hand-written Metal bandwidth for
  the declared shapes; zero silent no-op or silent fallback.

### P3 — library interception

- Implement the required `cublasGemmEx`, batched, and strided-batched subset
  over `tc_gemm`.
- Add Kimi dequant kernels and execute one dense layer end to end.
- Gate against NVIDIA-produced reference tensors captured by an independent
  test harness, with tensorcore/Metal dispatch provenance.

### P4 — application proof

- Compile and execute a source-available CUDA application with no source edits.
- Report kernel coverage, library coverage, fallbacks, performance, and quality
  separately.
- Gate: no unsupported call can be hidden by an unused or fallback path.

### P5 — binary compatibility decision

Only after P0-P4, evaluate PTX and fatbinary ingestion as a separate product and
legal workstream. Require a written legal decision, a typed semantics import,
and the same fail-closed numerical certification. Cubin/SASS remains excluded.

## Final decision

Proceed with first-party `tc-cuda` implementation. Preserve the existing
hand-written Metal kernels as the performance/reference path. Use CuMetal,
SCALE, ZLUDA, HIPIFY, Clang, CUDA documentation, and Metal documentation as
comparators and inputs—not as the runtime authority.

The next code change should be P0/P1 scaffolding, not another broad design pass:
a machine-readable subset, clean-room minimal headers, a Clang-based kernel
manifest tool, and a single vector-add path that emits MSL and executes through
a versioned tensorcore module descriptor.
