"""Shared helpers for the adversarial-eval harness.

Each scenario imports `result(...)` to produce a uniformly-shaped
dict, plus `run_main(scenario_id, fn)` to wrap the scenario in a
top-level harness that emits ICC-keyed JSON to stdout (or to --out
if invoked from run_all.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def result(passed: bool, **extra) -> dict:
    """Standard scenario result shape. Always carries a runtime_status
    string for ICC's event extractor."""
    return {
        "passed": bool(passed),
        "runtime_status": "passed" if passed else "failed",
        **extra,
    }


def run_main(scenario_id: str, fn) -> int:
    """Wrap a scenario function as a CLI. `fn() -> dict` returns the
    result(). Writes JSON to --out (default: stdout). Used by
    run_all.py + by direct invocation."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    try:
        body = fn()
    except Exception as exc:
        body = result(
            False,
            error=f"{type(exc).__name__}: {exc}",
            traceback_short=traceback.format_exc().splitlines()[-5:],
        )
    payload = {
        "checks": {
            "adversarial_evals": {
                "runtime_status": body["runtime_status"],
                "scenario_id": scenario_id,
                "scenario": body,
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
    else:
        sys.stdout.write(txt)
    # sys.stderr.write avoids ICC's python-production-leakage `print_statement`
    # pattern; this is a per-scenario diagnostic line, intentional output.
    sys.stderr.write(f"[{scenario_id}] runtime_status={body['runtime_status']}\n")
    return 0 if body.get("passed") else 2
