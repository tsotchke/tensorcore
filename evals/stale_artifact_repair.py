#!/usr/bin/env python3
"""Scenario: stale-artifact-repair.

ICC's index / memory / git-history artifacts can lag behind HEAD
after commits. The eval proves that ICC reports staleness honestly
and that the three-command refresh (index + build-memory +
build-git-history) restores a clean state in one cycle.

Spec:
  - WHEN we drop a trivial change into a tracked file (we touch the
    audit doc, which is benign) so the source_fingerprint diverges
  - THEN `icc source-drift` MUST report Stale=True
  - WHEN we run the canonical refresh
  - THEN `icc source-drift` MUST report Stale=False afterwards

Doesn't actually leave the touched file behind — restores it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import result, run_main  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ICC = Path.home() / "Desktop" / "infinite_context_coder" / "bin" / "icc"


def drift_stale() -> bool:
    out = subprocess.run([str(ICC), "source-drift", "--repo", "tensorcore"],
                            capture_output=True, text=True, timeout=30)
    return "Stale: `True`" in out.stdout


def go():
    if not ICC.exists():
        return result(False, skip_reason=f"icc not found at {ICC}")
    target = REPO_ROOT / "docs" / "icc_audit_2026-06-26.md"
    if not target.exists():
        return result(False, skip_reason=f"missing target file {target}")
    original = target.read_bytes()
    try:
        # Mutate it (add a trailing space-newline) to force a fingerprint diff.
        target.write_bytes(original + b" \n")
        stale_seen = drift_stale()

        # Restore + re-index — this is the canonical "repair" path.
        target.write_bytes(original)
        subprocess.run([str(ICC), "index", "--repo", "tensorcore"],
                          capture_output=True, text=True, timeout=60)
        subprocess.run([str(ICC), "build-memory", "--repo", "tensorcore"],
                          capture_output=True, text=True, timeout=60)
        subprocess.run([str(ICC), "build-git-history", "--repo", "tensorcore"],
                          capture_output=True, text=True, timeout=60)
        clean_after = not drift_stale()
    finally:
        # Always restore even on exception.
        target.write_bytes(original)
    return result(
        stale_seen and clean_after,
        stale_after_mutation=stale_seen,
        clean_after_repair=clean_after,
    )


if __name__ == "__main__":
    sys.exit(run_main("stale-artifact-repair", go))
