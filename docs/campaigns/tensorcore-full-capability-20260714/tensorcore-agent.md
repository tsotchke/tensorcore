# Agent note — TensorCore runtime and distributed substrate

Repository: `~/Desktop/tensorcore`

ICC task: `tensorcore-decentralized-training-frontier`

Campaign tasks: `tc-public-runtime-capabilities`, `tc-peer-identity`,
`tc-release-contract-closure`, `tc-sanitizer-input-hardening`,
`tc-verification-integrity`, `tc-diloco-correctness`,
`tc-distributed-topology-scale`, `tc-backend-hardware-proof`,
`tc-production-release-candidate`, and `frontier-system-gates`.

## Mission

Make TensorCore the stable, backend-neutral compute and distributed-training
substrate for the complete mesh. Preserve the working implementation, close
unsafe or misleading contracts, and expose only public primitives that qLLM,
QGTL, Eshkol, and other consumers can use without TensorCore importing their
model or policy semantics.

## Ground truth to preserve

- Remote tensor transport hardening is the campaign's verified baseline.
- `tc_diloco_*` is real: local outer steps, SGD/Nesterov/Adam, dense Gloo,
  sparse top-k Gloo payloads, error feedback, counters, Python bindings,
  multi-rank fork tests, and live mesh evidence already exist.
- Sparse top-k over Gloo is the only current mode that reduces wire volume.
- Current proof is a small deterministic multi-rank workload, not a real qLLM
  training curve.

## Known limits that must remain explicit

- `TC_DILOCO_COMPRESS_FP16` currently sends dense FP32.
- FP8, low-rank, and signSGD are reserved and rejected at initialization.
- `tolerate_dropouts` does not implement elastic membership or recovery and is
  rejected at initialization rather than silently ignored.
- Async outer execution now uses immutable snapshots, private worker state,
  explicit READY/FAILED observation, and caller-thread rebase/commit.
- Versioned transport identity authentication is available for remote tensor,
  mesh, and Gloo. Topology and membership epochs are still not authoritative;
  authentication proves configured identity possession, not scheduler intent.

## Execution state

- `tc-public-runtime-capabilities` is implemented but limited as of the
  2026-08-24 closure reset. The native query is sound, but Python, Rust and the
  API reference still disagree with the native current DiLoCo state ABI.
  `tc_runtime_capabilities_get` ABI v1 now reports known/available production
  features and compiled/active backend masks through a size-versioned public
  structure.
- Positive, unavailable, minimum-prefix, larger-caller, future-version, and
  invalid-size native tests pass on clean portable and Metal builds. The
  release contract is not closed: public export, Python FFI and Python constant
  gates fail, and the Python/Rust default still emits ABI v1 rather than v2.
- Snapshot-safe async and versioned DiLoCo checkpoint/resume are available.
  Elastic membership and real FP16 wire packing remain known but unavailable.
- Mutual HMAC-SHA-256 peer authentication, expected-identity/rank binding,
  replay rejection, key-overlap rotation, fail-closed legacy mixing, and
  authenticated direct-ring sockets are implemented for remote tensor, mesh,
  and Gloo. The contract authenticates but does not encrypt payloads.

## Ownership boundary

Own:

- Public C ABI, ABI/version discovery, device/backend capabilities.
- Kernel, buffer, transport, collective, and DiLoCo state-machine behavior.
- Generic TensorCore language bindings and substrate conformance evidence.

Do not own:

- qLLM tensor layouts, dispatch thresholds, optimizer/sampler checkpoints, or
  trainer policy.
- QGTL training-island policy or caller-authored evidence interpretation.
- Eshkol compiler lowering.
- Mesh placement, leases, job admission, or Selene promotion policy.
- qLLM-specific helper exports such as the nonexistent
  `tc_matmul_f32_passthrough`.

## Work order

1. **Closure reset:** reconcile native headers, export lists, Python, Rust and
   documentation; prove explicit v1 compatibility and default v2 round trips.
2. **Closure reset:** remove sanitizer-visible invalid-enum behavior and make
   off-manifold diagnostics correct across multiple optimizer instances.
3. **Closure reset:** repair stale/fail-open verification fixtures and make ICC
   production-readiness evidence trustworthy.
4. **Completed:** audit `tc_device_info_get`, backend introspection, ABI-version,
   and feature-query surfaces; publish one size/versioned capability contract.
5. **Completed:** add positive, unavailable, and mixed-version ABI tests so
   downstream adapters can fail closed without weak-symbol inventions.
6. **Completed:** redesign async DiLoCo around immutable per-round snapshots and a pending
   anchor/state swap at a declared boundary. Expose completion and error state;
   make finalize and checkpoint operations race-free.
7. **Completed:** serialize complete DiLoCo state: anchor, optimizer moments, error-feedback
   residuals, inner/outer counters, round identity, membership epoch, and
   compression metadata. Prove exact resume.
8. Either implement actual FP16 wire packing with numerical and byte accounting
   or reject the mode. Apply that rule to every advertised compression mode.
9. Define bounded round timeouts, authenticated membership epochs, replay
   protection, rank loss/rejoin, and checkpoint resynchronization before making
   dropout tolerance functional.
10. Replace centralized or unbounded collective behavior with topology-aware,
   bounded-state algorithms and prove reconnect plus concurrent ordering.
11. Collect physical correctness and performance evidence on representative
   Metal M4/M5, CUDA Blackwell/Ampere, Jetson, HIP, and CPU nodes.

## Acceptance gates

- No concurrent access to live training buffers from the async outer worker.
- Background failures are observable and cannot be counted as completed rounds.
- Checkpoint/resume is numerically equivalent to an uninterrupted reference.
- Wire-byte counters match packet/payload evidence for every compression mode.
- Rank loss, rejoin, duplicate/replayed messages, timeout, and mixed-version
  peers have negative tests.
- Public ABI changes are versioned and handed to qLLM, QGTL, and Eshkol agents.
- Hardware evidence identifies the actual backend selected; fallback is never
  reported as native acceleration.

## ICC start and handoff

```sh
~/Desktop/infinite_context_coder/bin/icc task-show \
  --repo tensorcore \
  --task-id tensorcore-decentralized-training-frontier \
  --format markdown

~/Desktop/infinite_context_coder/bin/icc work \
  --repo tensorcore \
  --task-id tensorcore-decentralized-training-frontier \
  --goal 'harden and extend the implemented DiLoCo runtime without regressing sparse Gloo' \
  --mode plan --allow-stale --format markdown
```

Record each tranche with `icc task-attempt --outcome in_progress` until all
owned gates have direct runtime evidence. Do not mark the task verified merely
because local CTest passes.
