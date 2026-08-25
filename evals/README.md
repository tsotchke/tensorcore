# tensorcore adversarial evals

Operational-edge scenarios that probe how tensorcore + ICC + Selene
degrade under realistic pressure. Each scenario file is a
self-contained Python harness that:

1. Sets up the adversarial state (a dirty worktree, a stale ICC
   artifact, a missing Qwen endpoint, a wedged disk, etc.).
2. Drives the production code path through it.
3. Checks the *behaviour* against a spec — not "did it succeed",
   but "did it degrade the way we promised?"

These exist because the substrate + oracle work proves the happy
path; this set proves we don't silently corrupt user state or
hand back wrong answers when the world misbehaves.

Run a single scenario:

```bash
python3 evals/dirty_worktree_recovery.py
```

Run the whole suite + emit ICC-grade JSON evidence:

```bash
python3 evals/run_all.py --out build/adversarial_evals_evidence.json
```

ICC oracle: `adversarial-evals-runtime-evidence` (per-scenario PASS
criteria in `.icc/completion-oracles.yaml`).

## The 6 must-have scenarios (weakness-map #1)

| file | scenario | what it proves |
|---|---|---|
| `dirty_worktree_recovery.py`     | dirty-worktree-recovery     | tensorcore + ICC don't clobber user-side uncommitted changes when re-indexing or rebuilding |
| `stale_artifact_repair.py`       | stale-artifact-repair       | when ICC artifacts (index, memory, git-history) lag HEAD, repair is a single command that doesn't lose state |
| `qwen_unavailable_degraded.py`   | qwen-unavailable-degraded-mode | local control-plane guidance stays useful when the Qwen model server is down or sandbox-blocked |
| `disk_pressure.py`               | disk-pressure               | the build / smoke pipeline either succeeds or fails LOUDLY when the build dir runs out of space — no silently-truncated artifacts |
| `fail_gate_surface.py`           | fail-gate-surface           | an intentionally-failing smoke surfaces in `production-audit` within one cycle, doesn't get swallowed by another oracle's PASS |
| `new_development_suggestions.py` | new-development-suggestions | when a user asks "what should I build next?", the recommendation references real evidence (weakness-map, dead-code, contract-gaps) and not hallucinated work |

Each scenario is independent. Pass/fail of one doesn't gate the
others — that's the point of breadth.
