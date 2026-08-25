# Agent note — QGTL decentralized policy and TensorCore adapter

Repository: `~/Desktop/quantum_geometric_tensor`

ICC task: `qgtl-tensorcore-decentralized-policy-adapter`

Campaign task: `qgtl-tensorcore-decentralized-policy-adapter`.

## Mission

Preserve QGTL's implemented heterogeneous training policy and connect it to
TensorCore through QGTL-owned adapter code. Replace caller-asserted or dry-run
evidence with measurements from real training, communication, checkpoint, and
failure-recovery execution.

## Ground truth to preserve

- `qgtl_training_island_t` captures backend, rank/world, devices, model and
  optimizer size, local/sync steps, merge rule, checkpoints, and evidence path.
- Merge policy includes allreduce, delta average, and DiLoCo outer SGD.
- Low-communication round validation checks loss deltas, dense/delta bytes,
  communication fraction, and checkpoint paths.
- LoRA, QLoRA, and S-LoRA adapter accounting validates frozen-base and trainable
  state sizes.
- Checkpoint-recovery evidence structures and validation exist.
- The current DiLoCo policy object does not invoke `tc_diloco_*`.
- Recovery construction is explicitly dry-run; `start_training` records
  lifecycle state but does not execute a model training loop.
- Existing MPI collectives and tests are a separate baseline and must not be
  mislabeled as TensorCore DiLoCo execution.

## Ownership boundary

Own:

- Training-island policy, merge-rule selection, adapter accounting, and QGTL
  evidence semantics.
- The adapter mapping those policy objects to TensorCore public handles and
  runtime results.
- QGTL model/gradient/parameter layout and checkpoint integration.

Do not own:

- TensorCore DiLoCo algorithms, compression codecs, membership protocol, or
  buffer implementation.
- qLLM training semantics or mesh placement.
- Synthetic evidence fields supplied without a runtime provenance record.

## Work order

1. Inspect the very dirty QGTL worktree and active completion campaign before
   changing CMake or distributed sources. Reindex ICC if source drift remains.
2. Define a narrow optional TensorCore adapter target in QGTL. Community builds
   without TensorCore must fail closed or retain the declared MPI path without
   pretending the TensorCore policy executed.
3. Map validated island policy to TensorCore context, distributed context,
   DiLoCo config, registered parameters, checkpoints, and status. Use only
   public headers and symbols.
4. Make policy validation reflect actual runtime support. Reserved TensorCore
   compression/dropout modes cannot be accepted as executable.
5. Populate low-communication evidence from runtime counters and structured
   traces. Caller-provided byte/loss values may be expectations, never proof.
6. Implement real failure injection and checkpoint restart evidence. Preserve a
   separate dry-run type/status so operators cannot confuse plans with results.
7. Add multi-rank CPU tests, then heterogeneous mesh evidence with explicit
   backend and device provenance.
8. Compare QGTL policy outcomes against dense MPI/allreduce and host-local
   baselines on the same workload and quality metric.

## Dependencies

- TensorCore capability ABI v1 is available through
  `tc_runtime_capabilities_get`. Gate policy with the known/available bits for
  `TC_CAPABILITY_DILOCO`, `TC_CAPABILITY_DILOCO_SPARSE_GLOO`, snapshot-safe
  async, and checkpoint/resume. Elastic-membership and FP16-wire bits remain
  unavailable; continue waiting for those lifecycle gates before enabling the
  corresponding policies.
- Consume machine/rank/fabric placement and job identity from computer_mesh.
- Coordinate any shared checkpoint or evidence schema fields with qLLM and
  Selene, but keep QGTL-specific types in QGTL.

## Acceptance gates

- The DiLoCo merge rule causes a real public TensorCore call in a QGTL-owned
  source file.
- Unsupported runtime policy combinations fail validation before job launch.
- Evidence bytes and loss values trace to a runtime event and raw artifact.
- An injected failed rank is restarted from a real checkpoint and rejoins with
  correct lineage.
- Tests distinguish MPI, TensorCore, dry-run, fallback, and unavailable states.
- No QGTL adapter implementation is added to TensorCore.

## ICC start and handoff

```sh
~/Desktop/infinite_context_coder/bin/icc task-show \
  --repo quantum_geometric_tensor \
  --task-id qgtl-tensorcore-decentralized-policy-adapter \
  --format markdown
```

Handoff must include the QGTL build option, adapter path, policy-to-runtime
mapping, checkpoint/evidence schema versions, unsupported combinations, and
commands for dry-run, MPI baseline, and real TensorCore tests.
