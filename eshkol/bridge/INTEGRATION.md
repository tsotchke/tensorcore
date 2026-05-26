# Eshkol ↔ tensorcore integration

This directory contains the production-grade bridge file that wires
`tensorcore` into the Eshkol LLVM codegen as external builtins.

## Files

- **`tensorcore_codegen.cpp`** — drop-in for `eshkol-platform/lib/backend/`.
  Declares the `tc_*` C ABI as ExternalLinkage functions in the Eshkol
  LLVM module, matching the pattern used by `builtin_declarations.cpp`
  and `tensor_codegen.cpp`.
- **`../tensorcore.esk`** — Scheme-level bindings (high-level surface).
- **`../hello_tensorcore.esk`** — minimal usage example.

## Integration in eshkol-platform (4 steps)

### 1. Copy the bridge file

```sh
cp ~/Desktop/tensorcore/eshkol/bridge/tensorcore_codegen.cpp \
   ~/Desktop/eshkol-platform/lib/backend/
```

### 2. Link tensorcore in eshkol-platform's CMake

Add to `eshkol-platform/CMakeLists.txt`:

```cmake
find_package(tensorcore REQUIRED PATHS ${CMAKE_SOURCE_DIR}/../tensorcore/build)
target_link_libraries(eshkol PUBLIC tensorcore::tensorcore)
target_sources(eshkol PRIVATE lib/backend/tensorcore_codegen.cpp)
```

Or, if not using `find_package`, link directly:

```cmake
target_link_libraries(eshkol PUBLIC
    ${CMAKE_SOURCE_DIR}/../tensorcore/build/libtensorcore.a
    "-framework Metal" "-framework MetalPerformanceShaders"
    "-framework MetalPerformanceShadersGraph" "-framework Accelerate")
target_include_directories(eshkol PUBLIC
    ${CMAKE_SOURCE_DIR}/../tensorcore/include)
```

### 3. Hook into codegen init

Edit `eshkol-platform/lib/backend/codegen_context.cpp` (or equivalent
context init site) to call the registration function:

```cpp
extern "C" void eshkol_register_tensorcore_builtins(CodegenContext*);

CodegenContext::CodegenContext(...) {
    // ... existing setup ...
    BuiltinDeclarations(*this);
    eshkol_register_tensorcore_builtins(this);   // NEW
}
```

### 4. Install the Scheme bindings

```sh
cp ~/Desktop/tensorcore/eshkol/tensorcore.esk \
   ~/Desktop/eshkol-platform/lib/tensorcore.esk
```

Then in Eshkol code:

```scheme
(require tensorcore)

(define ctx (tc-init))
(define A (tc-buffer-alloc ctx (* M K 2)))
(define B (tc-buffer-alloc ctx (* K N 2)))
(define C (tc-buffer-alloc ctx (* M N 2)))
(tc-gemm-fp16 ctx A B C M N K)
(display (tc-last-backend))
```

## What the bridge does

The bridge declares 14 external functions (the full tensorcore C ABI) in
the Eshkol LLVM module via `llvm::Function::Create` with `ExternalLinkage`.
When the Eshkol JIT or AOT compiler emits IR that calls one of these
names, the linker resolves the symbol against `libtensorcore.a` —
identical to how Eshkol's existing `eshkol_deep_equal`, `tensor_matmul`,
etc., are wired.

Once integrated, `tc_*` calls from Eshkol code dispatch into the
simdgroup_matrix kernels (or the Metal 4 / TensorOps path on M5+) with
zero overhead — direct LLVM IR call into the linked C ABI.

## Validation plan

After integration:

1. **Smoke test**: Run `ESHKOL_ENABLE_TENSORCORE=1 ESHKOL_PATH=~/Desktop/tensorcore/eshkol eshkol-run -I ~/Desktop/tensorcore/eshkol ~/Desktop/tensorcore/eshkol/hello_tensorcore.esk`. Should print device info + `gemm OK backend=simdgroup-matrix`.
2. **Cross-check vs eshkol-platform's existing GEMM**: Run both paths on identical inputs, compare element-wise. Should match within fp16 tolerance.
3. **Bench**: Compare tensorcore-via-Eshkol vs eshkol-platform's `gpu_memory.mm` matmul. tensorcore should be ≥1.2× on shapes ≥1024³ (because it uses `simdgroup_matrix` with fp32 accum while gpu_memory.mm uses f32-only simdgroup).

## Migration plan for the three downstream projects

After phase v0.5 of `tensorcore`'s ROADMAP, the migration shape is:

| Project | Before | After |
|---|---|---|
| `eshkol-platform/lib/backend/gpu/gpu_memory.mm` | 4016-line bespoke Metal | thin shim calling `tc_gemm`, `tc_attention_forward` |
| `quantum_geometric_tensor/src/metal/` | 22 `.metal` files | replaced by tensorcore kernels; QGT-specific ops layered on top |
| `semiclassical_qllm/src/backend/backend_metal.m` | bespoke Riemannian-Adam Metal | calls `tc_gemm` for the matmul-heavy part, keeps Riemannian retraction in its own kernels |

The SF64 / Ozaki-II / FP24 / FP53 precision-tier kernels in
`eshkol-platform/lib/backend/gpu/metal_softfloat.h` migrate into tensorcore
as `TC_DTYPE_SF64` / `TC_DTYPE_OZAKI` / etc. extensions — no behavior change,
just one canonical location.
