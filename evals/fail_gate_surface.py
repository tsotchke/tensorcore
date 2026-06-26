#!/usr/bin/env python3
"""Scenario: fail-gate-surface.

A deliberately-failing smoke must surface in production-audit /
runtime-evidence within one cycle. The eval injects a synthetic
"failed" evidence JSON, runs runtime-evidence against it, asserts
ICC sees the failure event.

Spec:
  - GIVEN a synthetic trace file with runtime_status="failed"
  - WHEN we run `icc runtime-evidence --trace-file <synthetic>`
  - THEN the failure_free check MUST be FAIL.
  - AND the synthetic trace's named status event must appear in
    the events list.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import result, run_main  # noqa: E402

ICC = Path.home() / "Desktop" / "infinite_context_coder" / "bin" / "icc"


def go():
    if not ICC.exists():
        return result(False, skip_reason=f"icc not found at {ICC}")
    synthetic = {
        "checks": {
            "adversarial_failed_gate": {
                "runtime_status": "failed",
                "failure_reason": "intentional failure for the failed-gates eval",
                "probes": {
                    "synthetic_red": {
                        "runtime_status": "failed",
                        "passed": False,
                        "reason": "this is the eval — must surface"
                    }
                },
            }
        }
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(synthetic, f)
        trace_path = f.name
    try:
        proc = subprocess.run([str(ICC), "runtime-evidence",
                                  "--repo", "tensorcore",
                                  "--trace-file", trace_path],
                                capture_output=True, text=True, timeout=30)
        stdout = proc.stdout
        failure_visible = "FAIL" in stdout and "failure_free" in stdout
        # The event extractor will surface our synthetic event under its name.
        event_seen = "adversarial_failed_gate" in stdout or "synthetic_red" in stdout
    finally:
        Path(trace_path).unlink(missing_ok=True)
    return result(
        failure_visible and event_seen,
        failure_check_visible=failure_visible,
        synthetic_event_seen=event_seen,
        icc_rc=proc.returncode,
    )


if __name__ == "__main__":
    sys.exit(run_main("fail-gate-surface", go))
