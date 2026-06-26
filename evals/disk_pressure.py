#!/usr/bin/env python3
"""Scenario: disk-pressure.

When the build dir runs out of space, smoke scripts MUST either
fail loudly or report a structured `skipped_no_space` status —
they must NOT write a silently-truncated artifact that ICC then
reads as PASS.

Spec:
  - GIVEN we point the smoke at an unwritable output path
    (a directory we don't have write access to under a
    constructed root)
  - WHEN we run `scripts/check_python_substrate_runtime.py
    --out /nonexistent-root/.../evidence.json`
  - THEN the script MUST exit non-zero AND must NOT have written
    any artifact at the requested path.

Approximates true disk-full because actually filling the disk is
destructive — the unwritable-path case exercises the same error
branch in the smoke's `out_path.write_text` call.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import result, run_main  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def go():
    # Path the kernel will reject — under /dev/null/, which can't be
    # turned into a directory.
    bad_out = Path("/dev/null/cant_write_here/evidence.json")
    smoke = REPO_ROOT / "scripts" / "check_python_substrate_runtime.py"
    env = {**os.environ,
            "TENSORCORE_LIB": str(REPO_ROOT / "build" / "libtensorcore.dylib")}
    proc = subprocess.run(["python3", str(smoke), "--out", str(bad_out)],
                            capture_output=True, text=True, timeout=60,
                            env=env)
    rc = proc.returncode
    artifact_absent = not bad_out.exists()
    return result(
        rc != 0 and artifact_absent,
        smoke_rc=rc,
        artifact_absent=artifact_absent,
        bad_out_path=str(bad_out),
    )


if __name__ == "__main__":
    sys.exit(run_main("disk-pressure", go))
