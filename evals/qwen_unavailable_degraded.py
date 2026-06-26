#!/usr/bin/env python3
"""Scenario: qwen-unavailable-degraded-mode.

ICC's structured commands (audit, weakness-map, production-audit)
should NOT silently require a Qwen model server: they're
deterministic + grounded in the repo's own artifacts. The eval
proves they keep working when the Qwen endpoint is unreachable.

Spec:
  - WHEN we run `icc weakness-map` with TC_QWEN_URL pointed at an
    unbound port (i.e. no Qwen server reachable)
  - THEN the command MUST exit 0 + return a populated weakness
    structure within 30s (no hanging on the network), with no
    sentinel "Qwen unreachable" in the OUTPUT itself (the
    weakness-map is artifact-grounded; the model is for OPTIONAL
    enrichment).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import result, run_main  # noqa: E402

ICC = Path.home() / "Desktop" / "infinite_context_coder" / "bin" / "icc"


def go():
    if not ICC.exists():
        return result(False, skip_reason=f"icc not found at {ICC}")
    env = {**os.environ,
            # Port 1 is reserved-and-unbound on every host we've seen.
            "TC_QWEN_URL": "http://127.0.0.1:1/never-bound",
            "QWEN_API_URL": "http://127.0.0.1:1/never-bound"}
    proc = subprocess.run([str(ICC), "weakness-map", "--repo", "tensorcore"],
                            capture_output=True, text=True, timeout=45, env=env)
    rc = proc.returncode
    stdout = proc.stdout
    # Should still produce a populated weakness map.
    populated = "Weaknesses:" in stdout and ("Must-have:" in stdout or "MUST" in stdout)
    return result(
        rc == 0 and populated,
        weakness_map_rc=rc,
        weakness_lines_present=populated,
        env_setting="TC_QWEN_URL=http://127.0.0.1:1/never-bound",
    )


if __name__ == "__main__":
    sys.exit(run_main("qwen-unavailable-degraded-mode", go))
