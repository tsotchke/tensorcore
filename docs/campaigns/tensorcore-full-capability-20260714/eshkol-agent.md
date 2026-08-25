# Agent note — Eshkol TensorCore compiler adapter

Repository: `~/Desktop/eshkol`

ICC task: `eshkol-tensorcore-adapter-ownership`

Campaign task: `eshkol-tensorcore-compiler-adapter`.

## Mission

Make Eshkol's canonical repository the authority for compiler/codegen lowering
and Eshkol-specific FFI behavior against TensorCore's public ABI. Preserve
existing working integration while removing duplicate authority through a
tested, compatibility-aware migration.

## Ground truth to preserve

- Eshkol already contains `lib/backend/tensorcore_codegen.cpp`.
- TensorCore also contains Eshkol `.esk` modules, flat `tc_eshkol_*` helpers,
  a compiler-oriented `eshkol/bridge/tensorcore_codegen.cpp`, and bridge smoke
  evidence.
- TensorCore's first-party generic bindings may remain with TensorCore, but
  compiler lowering and Eshkol language policy belong here.
- Existing bridge behavior must remain usable during migration; deleting one
  copy before conformance and release tests would break downstream users.

## Ownership boundary

Own:

- Eshkol AST/IR/codegen lowering, builtin registration, calling convention,
  compiler diagnostics, and Eshkol-facing conformance tests.
- Packaging/discovery of the TensorCore ABI from Eshkol builds.

Do not own:

- TensorCore kernel, buffer, transport, collective, or training algorithms.
- qLLM/QGTL model semantics or computer_mesh orchestration.
- A forked copy of TensorCore public headers or enums.

## Work order

1. Inventory every Eshkol/TensorCore integration path in both repositories and
   classify it as public ABI helper, language binding, compiler lowering,
   example, test, or obsolete duplicate.
2. Establish one canonical compiler adapter in Eshkol. Define a compatibility
   window for TensorCore's compiler-side bridge instead of deleting it
   immediately.
3. Consume installed/public TensorCore headers and capability discovery. Do not
   duplicate status, dtype, device-family, or feature enums. The handed-off
   contract is `tc_runtime_capabilities_get` ABI v1 in
   `include/tensorcore/capabilities.h`; unknown versions and unknown feature
   bits must fail closed.
4. Cover lifecycle, buffer ownership, GEMM/attention descriptors, error/status
   conversion, unavailable backends, and mixed ABI versions.
5. Run LLVM `verifyModule` on generated code and execute a linked portable-CPU
   smoke. Add representative Metal/CUDA runtime evidence when available.
6. Compare canonical and compatibility paths on the same programs until their
   outputs and failure semantics match.
7. Update packaging and documentation so downstream Eshkol users install the
   adapter from Eshkol and TensorCore as a library dependency.
8. Remove or demote duplicate authority only after both repositories' release
   and conformance gates pass.

## Acceptance gates

- One documented canonical compiler/codegen implementation exists in Eshkol.
- Generated modules pass LLVM verification and execute the intended TensorCore
  backend or report an explicit unavailable/fallback state.
- ABI version mismatch and unsupported capabilities fail deterministically.
- The compatibility migration has tests and a stated removal/version policy.
- TensorCore no longer needs knowledge of Eshkol compiler internals.

## ICC start and handoff

```sh
~/Desktop/infinite_context_coder/bin/icc task-show \
  --repo eshkol \
  --task-id eshkol-tensorcore-adapter-ownership \
  --format markdown
```

The current ICC compiler oracle is blocked, including missing `verifyModule`
evidence. Handoff to TensorCore must enumerate which TensorCore-side files stay
as generic ABI/binding support, which are compatibility-only, and which can be
removed after the migration window.
