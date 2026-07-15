# Agent note — Selene federated checkpoint promotion

Repository: `~/Desktop/Selene`

ICC task: `selene-federated-checkpoint-promotion`

Campaign task: `selene-federated-checkpoint-promotion`.

## Mission

Own the policy that turns decentralized qLLM training outputs into an accepted
Selene checkpoint: deterministic merge, lineage, held-out evaluation, value
alignment, anti-forgetting, human gates, atomic promotion, rollback, and audit.
Consume runtime and mesh evidence without absorbing their implementations.

## Ground truth to preserve

- `continual_organism_board.json` already defines D-1 remote training shards,
  D-2 federated checkpoint merge, and D-3 TensorCore training integration.
- D-1 is blocked by the existing HQ-2 human authorization gate for remote
  training-node access. Do not bypass it.
- `ORGANISM_ARCHITECTURE.md` already requires adversarial quorum, Noesis
  verdict, EWC/SI anti-forgetting, constitution checks, lineage, and rollback.
- `selene_exec_mesh.py` performs a real SSH remote substrate probe with queue
  audit states, but the payload is deliberately read-only and is not a training
  step.
- Deterministic federated checkpoint merge and promotion are planned, not yet
  proven by a real multi-shard candidate run.

## Ownership boundary

Own:

- Candidate/checkpoint lineage and deterministic merge policy.
- Evaluation corpus identity, held-out quality, value/constitution,
  anti-forgetting, quorum, promotion, rollback, and autobiographical audit.
- The decision record explaining why a candidate was accepted or rejected.

Do not own:

- TensorCore runtime/DiLoCo code.
- qLLM model/trainer adapter or optimizer implementation.
- computer_mesh placement, leases, SSH/Tailscale reconciliation, or evidence
  transport.
- QGTL's training-island policy.

## Work order

1. Preserve HQ-2 and every existing value gate. Separate read-only substrate
   probes from authorized remote training submissions.
2. Define a versioned candidate manifest consuming qLLM checkpoint lineage,
   model/config hash, data identity, sampler/RNG state, optimizer/DiLoCo state,
   training metrics, mesh job/rank identity, and raw evidence references.
3. Specify a deterministic merge rule and ordering. The same candidate set and
   policy version must produce the same merged hash or deterministic rejection.
4. Prevent unsafe parameter averaging across incompatible model/config,
   tokenizer, optimizer, adapter, data-policy, or lineage versions.
5. Evaluate each shard and merged candidate on held-out quality,
   anti-forgetting, value/constitution, safety, and declared resource criteria.
6. Require adversarial/quorum and Noesis verdict evidence before durable
   promotion. Keep candidate, accepted, rejected, and rolled-back states
   append-only and auditable.
7. Promote atomically and retain the previous accepted checkpoint. Prove
   rollback on injected quality/value failure and interrupted promotion.
8. Demonstrate a real authorized two-shard run where the merged candidate is at
   least as good as the best shard on the declared evaluation.

## Dependencies

- qLLM must provide real resumable candidates and a versioned checkpoint
  manifest.
- computer_mesh must provide authenticated job/rank/topology evidence and
  unambiguous completion/failure state.
- TensorCore/QGTL evidence may inform training provenance but cannot substitute
  for Selene's promotion evaluation.

## Acceptance gates

- Merge is deterministic and rejects incompatible candidates before touching
  the accepted checkpoint.
- The merged candidate meets or exceeds the best shard on the declared held-out
  evaluation and passes anti-forgetting thresholds.
- Value, constitution, quorum, and human-authorization gates are enforced.
- Promotion and rollback are atomic, replayable, and lineage-preserving.
- Every accepted/rejected decision cites candidate hashes, policy version,
  evaluation data identity, metrics, verdicts, and raw evidence.

## ICC start and handoff

```sh
~/Desktop/infinite_context_coder/bin/icc task-show \
  --repo Selene \
  --task-id selene-federated-checkpoint-promotion \
  --format markdown
```

The ICC dossier is currently blocked because checkpoint, training-progress,
completion, journal, resume, and data-batch evidence are not proven. Do not mark
the task verified from the existing remote probe alone.
