# TensorCore full-capability campaign — agent notes

These are the individual handoff notes for the ICC-driven TensorCore
full-capability campaign. Each note is self-contained enough for its repository
agent to start safely, while `configs/full_capability_campaign.json` remains the
machine-readable authority for owners, dependencies, waves, and acceptance
gates.

## Shared rules

1. Run the named ICC task before changing source. Refresh the repository index
   if ICC reports source drift.
2. Preserve existing work. Every participating repository currently has a dirty
   or active worktree; inspect overlapping diffs before editing.
3. Consumer-specific adapters stay in their consuming repositories. TensorCore
   owns the stable public ABI and substrate behavior, not qLLM layouts, QGTL
   policy, Eshkol compiler lowering, mesh orchestration, or Selene promotion
   policy.
4. A compiled path is not a completed path. Acceptance requires runtime
   evidence identifying the machine, accelerator, backend, build, workload,
   checkpoint, quality result, failure behavior, and raw logs where applicable.
5. Do not claim Beyond-SOTA status from projections or isolated
   microbenchmarks. Use the thirteen gates in
   `configs/beyond_sota_research.json` and publish unresolved limits.
6. Tailscale is an observation and encrypted transport source, not the
   declarative topology authority. `computer_mesh` declarations remain
   authoritative; differences are explicit drift.

## Notes

| Agent | Repository | Note | First dependency |
|---|---|---|---|
| TensorCore runtime agent | `tensorcore` | [tensorcore-agent.md](tensorcore-agent.md) | Verified transport baseline |
| qLLM integration agent | `semiclassical_qllm` | [qllm-agent.md](qllm-agent.md) | TensorCore public capability contract |
| Mesh control-plane agent | `computer_mesh` | [computer-mesh-agent.md](computer-mesh-agent.md) | Complete topology reconciliation |
| QGTL policy adapter agent | `quantum_geometric_tensor` | [qgtl-agent.md](qgtl-agent.md) | Safe TensorCore DiLoCo contract |
| Eshkol compiler adapter agent | `eshkol` | [eshkol-agent.md](eshkol-agent.md) | TensorCore public ABI conformance |
| Selene promotion agent | `Selene` | [selene-agent.md](selene-agent.md) | Real qLLM candidates and mesh evidence |

## Cross-agent contract flow

```text
computer_mesh topology ──► TensorCore identity/capabilities
          │                         │
          │                         ├──► qLLM-owned runtime/trainer adapter
          │                         ├──► QGTL-owned policy/runtime adapter
          │                         └──► Eshkol-owned compiler adapter
          │
          └──► placement, leases, health, recovery, evidence
                                      │
qLLM checkpoints + evaluation ────────┴──► Selene merge/promotion/rollback
```

Any public contract change must be reported to downstream agents with the
header/schema path, version change, compatibility behavior, negative tests,
and the ICC evidence or task attempt that proves it.

## Coordinator checks

From the TensorCore checkout:

```sh
python3 scripts/check_full_capability_campaign.py
python3 scripts/check_full_capability_campaign_selftest.py
python3 scripts/check_beyond_sota_research.py
python3 scripts/check_docs_links.py
~/Desktop/infinite_context_coder/bin/icc capability-roadmap \
  --repo tensorcore --format markdown
```

Campaign completion remains a fleet property. An agent completes a workstream
only after its owned acceptance gates pass and every dependent agent has a
usable, versioned handoff.
