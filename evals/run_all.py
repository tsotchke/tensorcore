#!/usr/bin/env python3
"""Run every adversarial eval scenario, aggregate the per-scenario
JSON into a single ICC-keyed payload, write to --out.

Each scenario is a standalone Python entrypoint that returns the
canonical {checks.adversarial_evals.scenario: {...}} shape. We
invoke them as subprocesses (one each) so a single broken scenario
can't take the whole suite down.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCENARIOS = [
    "dirty_worktree_recovery.py",
    "stale_artifact_repair.py",
    "qwen_unavailable_degraded.py",
    "disk_pressure.py",
    "fail_gate_surface.py",
    "new_development_suggestions.py",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    evals_dir = Path(__file__).resolve().parent
    scenarios = {}
    overall_passed = True
    t0 = time.time()
    for sc in SCENARIOS:
        proc = subprocess.run(["python3", str(evals_dir / sc)],
                                capture_output=True, text=True, timeout=120)
        try:
            payload = json.loads(proc.stdout) if proc.stdout.strip() else None
        except json.JSONDecodeError:
            payload = None
        sid = sc.replace(".py", "").replace("_", "-")
        if payload and "checks" in payload and "adversarial_evals" in payload["checks"]:
            sub = payload["checks"]["adversarial_evals"]
            # Intentionally don't embed full scenario details — some
            # scenarios (e.g. failed-gates) contain synthetic
            # "runtime_status: failed" markers as part of what they're
            # probing, and bubbling those up to the suite-level JSON
            # makes ICC's free-text failure extractor think the suite
            # itself failed. The per-scenario JSON is on disk if you
            # need the diagnostic (run the scenario directly).
            scenarios[sid] = {
                "runtime_status": sub.get("runtime_status", "failed"),
                "passed": sub.get("scenario", {}).get("passed", False),
                "wall_seconds": sub.get("wall_seconds"),
            }
        else:
            scenarios[sid] = {
                "runtime_status": "failed",
                "passed": False,
                "error": f"subprocess rc={proc.returncode}",
                "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
            }
        if not scenarios[sid]["passed"]:
            overall_passed = False
        print(f"[{sid}] {'✓' if scenarios[sid]['passed'] else '✗'}",
              file=sys.stderr)

    payload = {
        "checks": {
            "adversarial_evals_suite": {
                "runtime_status": "passed" if overall_passed else "failed",
                "n_scenarios": len(SCENARIOS),
                "n_passed": sum(1 for s in scenarios.values() if s["passed"]),
                "scenarios": scenarios,
                "wall_seconds": round(time.time() - t0, 3),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        }
    }
    txt = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(txt)
        print(f"wrote {out}", file=sys.stderr)
    else:
        sys.stdout.write(txt)
    return 0 if overall_passed else 2


if __name__ == "__main__":
    sys.exit(main())
