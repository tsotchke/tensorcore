# Agent note — qLLM TensorCore adapters and decentralized trainer

Repository: `~/Desktop/semiclassical_qllm`

ICC task: `qllm-tensorcore-decentralized-training-adapter`

Campaign tasks: `qllm-tensorcore-dispatch-adapter` and
`qllm-tensorcore-decentralized-trainer`.

## Mission

Own qLLM's complete integration with TensorCore: model tensor/layout mapping,
dispatch and fallback policy, persistent TensorCore context/buffer lifecycle,
real decentralized training, and qLLM checkpoint/resume semantics. The adapter
must live and build in qLLM, using only TensorCore's public ABI.

## Ground truth to preserve

- The current dirty qLLM working tree contains `src/core/matmul_dispatch.c`,
  `src/core/tensorcore_adapter.c`, public qLLM adapter headers, fallback tests,
  and a real linked TensorCore smoke.
- `QLLM_ENABLE_TENSORCORE` now performs optional public-header/library
  discovery, compiles the adapter into `qllm_core`, and retains the dependency
  on the final shared library.
- The qLLM-owned adapter has a persistent context, reusable grow-only buffers,
  dtype/layout conversion, backend diagnostics, and fallback-only stubs. These
  are uncommitted user/agent changes and must be preserved.
- Availability currently means only that `tc_init` succeeded; the adapter has
  not yet consumed TensorCore's versioned capability query.
- Fallback numerical tests are useful and must continue to pass without a
  TensorCore installation.
- `docs/SCALE_OUT_ARCHITECTURE.md` names
  `python/qllm/distributed_trainer.py`, but that module is absent.
- qLLM already has TensorCore scheduler/deployment guards and a soft
  `tensorcore_torch` integration; reuse those contracts instead of creating a
  third integration path.

## Ownership boundary

Own:

- qLLM's optional build/link discovery for TensorCore.
- Tensor layout, dtype, transpose, stride, alpha/beta, and lifecycle mapping.
- AUTO/forced/fallback selection and qLLM-visible counters/diagnostics.
- Model parameters, optimizer, data sampler, RNG, scheduler, and checkpoint
  lineage for decentralized training.

Do not own:

- New qLLM-specific exports inside TensorCore.
- TensorCore transport, collective, compression, or DiLoCo algorithms.
- Machine placement, Tailscale reconciliation, or lease recovery.
- QGTL policy or Selene candidate promotion.

## Work order

1. Preserve and review the active qLLM CMake/adapter diff; it already implements
   the narrow optional TensorCore target and contains unrelated user work.
2. After `tc_init`, call `tc_runtime_capabilities_get` ABI v1 and require
   `TC_CAPABILITY_GEMM_F32` through the fail-closed helper. Map
   `TC_ERR_ABI_MISMATCH` deterministically and retain fallback-only builds.
3. Extend the existing persistent-context/reusable-buffer adapter rather than
   replacing it or reintroducing per-call initialization/allocation.
4. Preserve three explicit modes: AUTO, forced TensorCore, and forced fallback.
   Forced TensorCore must return a clear backend error when unavailable; AUTO
   may fall back, but must report which backend actually ran.
5. Add numerical tests for alpha/beta, shapes, transposes/layouts, dtypes,
   zero-size behavior, failure cleanup, and repeated calls. Add one real linked
   TensorCore smoke that proves its backend counter increased.
6. Implement `python/qllm/distributed_trainer.py` as the qLLM-owned bridge from
   real qLLM parameters and training steps to the safe public DiLoCo surface.
7. Checkpoint model, optimizer, sampler/RNG, scheduler, DiLoCo state, outer
   round, topology/membership epoch, source data identity, and lineage hashes.
8. Prove a two-host heterogeneous run for at least three outer rounds with
   held-out loss reduction, sparse byte evidence, checkpoint interruption, and
   exact resume behavior.

## Dependencies

- TensorCore capability ABI v1 is now handed off in
  `include/tensorcore/capabilities.h`: use
  `tc_runtime_capabilities_get`, `TC_CAPABILITY_GEMM_F32`, and
  `tc_runtime_capability_available`. Unknown versions/features fail closed.
- Do not enable async training until TensorCore provides snapshot-safe
  completion/error semantics.
- Consume placement, fabric, rank, lease, and evidence paths from
  `computer_mesh`; do not hard-code the six hosts named in the scale-out doc.

## Acceptance gates

- Clean qLLM builds work both with and without TensorCore.
- No qLLM adapter source is added to TensorCore.
- AUTO selects TensorCore only when runtime capability and shape policy agree.
- Real execution, not a weak test double, increments the TensorCore dispatch
  counter and matches the fallback numerical oracle.
- The distributed trainer exists at its documented import path and trains real
  qLLM parameters.
- Resume preserves exact lineage and reaches the same declared tolerance as an
  uninterrupted control.
- Held-out quality, wire bytes, backend, machine identities, and failures are
  emitted as structured evidence.

## ICC start and handoff

```sh
~/Desktop/infinite_context_coder/bin/icc task-show \
  --repo qllm \
  --task-id qllm-tensorcore-decentralized-training-adapter \
  --format markdown
```

Before handoff, provide the TensorCore and mesh agents with the exact qLLM build
option, adapter header/source, runtime evidence schema, checkpoint schema
version, and commands for fallback-only and real-backend tests.
