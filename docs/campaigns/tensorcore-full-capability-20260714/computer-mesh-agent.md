# Agent note — computer_mesh topology and decentralized-training control plane

Repository: `~/Desktop/computer_mesh`

ICC task: `computer-mesh-decentralized-training-control-plane`

Campaign tasks: `mesh-topology-authority`,
`mesh-decentralized-training-control-plane`, and `fleet-pareto-acceptance`.

## Mission

Represent the entire real computational topology and operate decentralized
training on it: observation, identity reconciliation, placement, admission,
leases, health, rank recovery, checkpoint-writer safety, structured evidence,
and final fleet acceptance. Model and compiler adapters remain outside this
repository.

## Ground truth to preserve

- The discovery snapshot found 19 declared machines, 20 active-profile
  Tailscale identities including a phone, 15 online identities, 14 online
  compute machines, nine live accelerator hosts, and thirteen GPU devices.
- These are time-stamped observations, not desired-state constants.
- A second inactive Tailscale profile exists and requires non-destructive
  reconciliation.
- TensorCore's scheduler snapshot represented nine resources on five machines,
  not the complete fleet.
- `tsotchke-tensorcore` records real cross-host dense-allreduce and sparse
  DiLoCo measurements. The measured cosbox/old-donkey path is 1 GbE-class, so
  sparse deltas are the present training default.
- The helper's evidence is currently a fixed snapshot, not a continuously
  refreshed source of truth.

## Ownership boundary

Own:

- Declarative machine authority and stable machine/device identity mapping.
- Tailscale-profile, cloud, hardware, health, and scheduler observation.
- Drift classification, eligibility, quarantine, taints, leases, admission,
  placement, recovery, and evidence retention.
- Job and fleet-acceptance schemas.

Do not own:

- TensorCore algorithms or public ABI.
- qLLM tensor/trainer adapters, QGTL merge policy, Eshkol lowering, or Selene
  promotion decisions.
- Silent enrollment or deletion based only on a Tailscale observation.

## Work order

1. Reconcile `nodes.json`, `computer_mesh/mesh_inventory.py`, every configured
   Tailscale profile, cloud inventories, and TensorCore resources by stable
   identity. Preserve declared-only and observed-only rows as drift.
2. Model `declared`, `observed`, `online`, `eligible`, `quarantined`, `retired`,
   and `non_compute` separately. Never infer machine count from logical graph
   vertices.
3. Add accelerator identity, memory, backend, health freshness, fabric class,
   measured peer bandwidth/latency, taints, and evidence age to placement.
4. Define the decentralized-training job schema: owner repository, model and
   data identity, checkpoint lineage, rank/site topology, synchronization
   cadence, compression, security identity, resource requirements, retry and
   recovery policy, and completion oracle.
5. Reconcile desired jobs continuously. Make rank/lease claims atomic and
   ensure only one accepted writer owns a checkpoint generation.
6. Route dense, sparse, or host-local training from measured fabric evidence;
   reject disaggregation or dense synchronization when admission thresholds do
   not pass.
7. Emit per-rank and aggregate evidence for backend, host/device identity,
   inner/outer progress, loss, bytes, checkpoint, membership, failure, retry,
   and completion.
8. Build the final fleet matrix from live observations; phones, offline nodes,
   quarantined devices, and unsupported backends must remain visible with an
   explicit classification.

## Safety rules

- Tailscale encryption and ACLs remain mandatory until TensorCore peer
  authentication has passed negative and rotation tests.
- Do not expose unauthenticated TensorCore listeners to untrusted networks.
- Reconciliation must be non-destructive by default and must not rewrite user
  SSH/Tailscale state merely to eliminate drift.
- Do not launch expensive or mutating remote training while performing an
  inventory refresh.

## Acceptance gates

- Every declared and observed identity appears exactly once in the reconciled
  topology with provenance and freshness.
- Placement decisions are reproducible from the recorded topology snapshot and
  job policy.
- Expired leases, node loss, rank rejoin, stale health, and checkpoint-writer
  conflict have deterministic recovery tests.
- A real qLLM/QGTL decentralized job can be observed from admission through
  completion or explicit failure.
- Fleet acceptance covers every eligible topology/accelerator class and reports
  exclusions instead of silently shrinking the fleet.

## ICC start and handoff

```sh
~/Desktop/infinite_context_coder/bin/icc task-show \
  --repo computer_mesh \
  --task-id computer-mesh-decentralized-training-control-plane \
  --format markdown
```

The generic ICC oracle currently scores this task more optimistically than the
campaign-specific gates. Treat `configs/full_capability_campaign.json` as the
completion authority. Hand topology and job-schema versions to TensorCore,
qLLM, QGTL, and Selene agents before enabling live reconciliation.
