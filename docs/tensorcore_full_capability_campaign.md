# TensorCore Full-Capability Campaign

This campaign brings TensorCore from a locally correct kernel library to the
evidence-backed compute substrate for the complete `computer_mesh` topology.
ICC is the campaign control plane: every tranche starts with an ICC work
dossier and ends with structured runtime evidence, a scoped production audit,
and a verified task attempt. The durable dependency graph lives in
`.icc/assistant-goals.yaml`; the cross-repository ownership and execution graph
lives in `configs/full_capability_campaign.json` and is checked by
`scripts/check_full_capability_campaign.py`.

## Baseline snapshot

The 2026-07-13 discovery snapshot found several different counts that must not
be conflated:

- `computer_mesh` declares 19 machines. Its graph has 50 logical vertices;
  those vertices include services and topology concepts, not 50 computers.
- The active shared Tailscale profile exposes 20 identities: 19 machines plus
  one Pixel device. Fifteen identities were online, of which fourteen were
  compute machines. A second inactive Tailscale profile exists and still needs
  non-destructive inventory reconciliation.
- Nine live hosts exposed accelerators, for thirteen live GPU devices in the
  snapshot: seven Blackwell devices, four other NVIDIA/Jetson devices, and two
  Apple GPUs.
- TensorCore's scheduler described nine resources across five machines. It did
  not represent the complete live accelerator fleet, and all desired jobs were
  paused with no continuously running reconciler.

These are time-stamped observations, not hard-coded desired counts. The future
authority must classify declared, observed, online, eligible, quarantined, and
retired resources without silently dropping nodes or treating phones and
services as compute workers.

## Campaign invariants

1. `computer_mesh` is the declarative machine authority; Tailscale and cloud
   providers are observation sources. Differences become explicit drift, not
   implicit deletion or enrollment.
2. TensorCore schedules only resources with a stable machine identity,
   accelerator identity, health state, admission policy, and fresh runtime
   evidence.
3. Distributed listeners bind the requested overlay/private address. Tailscale
   ACLs remain mandatory until application-level peer authentication is proven;
   no campaign stage exposes an unauthenticated listener to an untrusted
   network.
4. A compiled backend is not a capable backend. Metal, CUDA, HIP, TensorOps,
   sparse, quantized, and distributed paths require physical runtime evidence
   on representative hardware.
5. Consumer integration is part of capability. qLLM AUTO dispatch must link to
   TensorCore's public runtime ABI through a qLLM-owned adapter and demonstrate
   checkpoint/resume and cross-node execution. Consumer-specific adapters,
   layouts, heuristics, optimizer state, and checkpoint policy live in their
   respective public repositories; TensorCore does not import those consumers.
6. ICC evidence is necessary but its attestations are not tamper-resistant
   until an attestation secret is configured. Unsigned evidence is labeled as
   such.

## Existing decentralized-training baseline

The campaign starts from implemented work and will not replace it with a new
parallel design:

- TensorCore exposes `tc_diloco_*`, implements local SGD outer synchronization,
  outer SGD/Nesterov/Adam, dense Gloo synchronization, sparse top-k Gloo payloads
  with error feedback, counters, Python bindings, multi-rank fork tests, and a
  checkpointed live mesh-training evidence runner.
- QGTL already defines and validates heterogeneous training islands, allreduce,
  delta-average and DiLoCo-outer-SGD merge policies, LoRA/QLoRA/S-LoRA state
  accounting, low-communication round evidence, and checkpoint-recovery
  evidence.
- `computer_mesh` records cross-host dense-allreduce and sparse-DiLoCo evidence
  and already recommends sparse deltas on the measured 1 GbE-class path.
- Selene's organism plan already names remote training shards, deterministic
  federated checkpoint merge, value gates, anti-forgetting, and rollback as the
  learning-promotion policy.

That baseline has precise limits. TensorCore's current FP16 DiLoCo mode still
sends dense FP32; FP8, low-rank and signSGD are unsupported; dropout tolerance
is staged; and the async worker lacks immutable parameter snapshots, completion
status, and background error propagation. QGTL's policy structs do not yet
invoke TensorCore, and its recovery constructor is dry-run evidence. qLLM's
documented `python/qllm/distributed_trainer.py` is absent; its matmul dispatcher
is not in a CMake target and calls weak names TensorCore does not export. Live
training proof is deterministic and multi-rank, but still a small workload
rather than a real qLLM loss curve.

These are campaign inputs, not reasons to discard the implemented foundation.

## Repository ownership

| Repository | Owns |
|---|---|
| `tensorcore` | Public C ABI, runtime capability discovery, kernels, transports, collectives, DiLoCo state machine, generic first-party bindings and substrate conformance evidence |
| `semiclassical_qllm` | qLLM tensor/layout conversion, dispatch and fallback heuristics, TensorCore context lifecycle, real distributed trainer, model checkpoint/sampler/optimizer semantics |
| `quantum_geometric_tensor` | Training-island policy and the adapter that maps QGTL merge/recovery policy to the TensorCore runtime |
| `computer_mesh` | Machine identity, complete Tailscale/cloud observation, topology reconciliation, placement, admission, leases, health, recovery and fleet observability |
| `eshkol` | Compiler/codegen lowering and Eshkol-specific FFI adapter against the public TensorCore ABI |
| `Selene` | Federated candidate merge and promotion policy, evaluation, value/anti-forgetting gates, lineage and rollback |

First-party Python, Rust, Swift, PyTorch and WebAssembly packages may remain in
TensorCore when they are bindings to TensorCore itself. Application- or
compiler-specific behavior belongs to the consuming repository. The dependency
arrow is always consumer to TensorCore.

## ICC campaign dossiers

The initial cross-repository dossiers are:

- `tensorcore-decentralized-training-frontier`
- `qllm-tensorcore-decentralized-training-adapter`
- `qgtl-tensorcore-decentralized-policy-adapter`
- `computer-mesh-decentralized-training-control-plane`
- `eshkol-tensorcore-adapter-ownership`
- `selene-federated-checkpoint-promotion`

They are generated with `~/Desktop/infinite_context_coder/bin/icc work` and
include the related repositories so changes are planned from the actual ABI,
policy, topology, tests, and evidence rather than from a repository in
isolation.

## Agent handoffs

The repository-specific working briefs are collected in the
**[2026-07-14 agent handoff index](campaigns/tensorcore-full-capability-20260714/README.md)**.
There is one self-contained note for each owning agent: TensorCore, qLLM,
computer_mesh, QGTL, Eshkol, and Selene. Each note records the agent's ICC
task, current ground truth, owned work, dependencies, acceptance evidence,
and boundaries with the other repositories.

## Execution waves

1. Preserve and tell the truth: keep the verified transport baseline, enumerate
   implemented decentralized-training behavior, and make unsupported or unsafe
   behavior explicit.
2. Public contracts and topology authority: add versioned runtime capability
   discovery, bind peer identity, and reconcile every declared and observed
   node without conflating logical graph vertices with compute machines.
3. Repo-owned adapters and training correctness: make async DiLoCo snapshot-safe
   and resumable, then land qLLM, QGTL, and Eshkol adapters in their owners with
   real execution tests.
4. Elastic control plane and recovery: continuously place, lease, observe,
   checkpoint, fail and recover decentralized jobs on measured topology.
5. Hardware frontier and fleet acceptance: prove representative Metal, CUDA,
   Jetson, HIP and CPU paths, pass every primary-source frontier gate, and
   publish the fleet Pareto matrix.

The machine-readable graph gives every task an owner, ICC task id, dependency
set, wave, status, and falsifiable acceptance criteria. It also assigns all
thirteen gates in `configs/beyond_sota_research.json` to their responsible
repositories.

## Stages and exit gates

The ordered stages are evaluated by:

```sh
~/Desktop/infinite_context_coder/bin/icc capability-roadmap --repo tensorcore --format markdown
```

- Remote transport hardening: exact bind semantics, safe request bounds,
  pointer-lifetime barriers, concurrent framing, and bounded shutdown are
  verified by the full native suite and an AddressSanitizer transport suite.
- Topology authority: every declared and observed identity is classified, and
  scheduler-resource drift is machine-readable and test-gated.
- Transport identity: authenticated, versioned peer negotiation and negative
  tests are proven before fleet-wide listener activation.
- Scheduler reconciliation: desired state, leases, health, admission, and
  recovery operate continuously across every eligible accelerator.
- Native consumers and hardware: qLLM dispatch, Apple M4/M5 selection, and all
  representative backends carry physical runtime evidence.
- Decentralized training: TensorCore DiLoCo is snapshot-safe, resumable and
  honest about wire compression; qLLM and QGTL own their adapters; real model
  loss, checkpoint continuity, sparse bytes and injected failure recovery are
  proven across heterogeneous nodes.
- Distributed scale: reconnect, bounded round state, ordering, rank churn, and
  topology-aware collectives survive multi-node failure tests.
- Fleet acceptance: one reproducible matrix ties machine identity, device
  identity, build, test, dispatch, recovery, and performance evidence together.

Campaign completion is a fleet property. A green local CTest run is a required
component, but it is never by itself the completion oracle.

Validate the program graph with:

```sh
python3 scripts/check_full_capability_campaign.py
python3 scripts/check_full_capability_campaign_selftest.py
~/Desktop/infinite_context_coder/bin/icc capability-roadmap --repo tensorcore --format markdown
```
