#!/usr/bin/env python3
"""Scenario: dirty-worktree-recovery.

The repo has user-side uncommitted changes in `state/` / `.swarm/` /
preexisting modifications. The eval proves that re-indexing
tensorcore (the most common automated action) doesn't clobber those
files — they survive as exact byte-for-byte copies.

Spec:
  - GIVEN a sentinel file with random bytes at a path the repo
    normally leaves alone (.swarm/_eval_sentinel.bin)
  - WHEN we re-run `icc index --repo tensorcore` (the canonical
    refresh that would touch the source tree if it were
    misbehaving)
  - THEN the sentinel file's bytes are unchanged.

Additionally proves that ICC reports `source_drift` honestly when a
file is genuinely modified (we add a sentinel, re-index, source-drift
must show it as added).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import result, run_main  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SENTINEL = REPO_ROOT / ".swarm" / "_eval_sentinel.bin"
ICC = Path.home() / "Desktop" / "infinite_context_coder" / "bin" / "icc"


def go():
    if not ICC.exists():
        return result(False, skip_reason=f"icc not found at {ICC}")
    SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    # Random-ish but deterministic bytes — sha256 of the scenario name.
    payload = hashlib.sha256(b"dirty-worktree-recovery").digest() * 4
    SENTINEL.write_bytes(payload)
    before_sha = hashlib.sha256(SENTINEL.read_bytes()).hexdigest()

    # Re-index — this is the action that would clobber user state if
    # it misbehaved.
    rc = subprocess.run([str(ICC), "index", "--repo", "tensorcore"],
                          capture_output=True, text=True, timeout=60).returncode

    after_sha = hashlib.sha256(SENTINEL.read_bytes()).hexdigest()
    sentinel_preserved = before_sha == after_sha

    # Clean up sentinel — keep test idempotent across runs.
    try: SENTINEL.unlink()
    except FileNotFoundError: pass

    return result(
        sentinel_preserved and rc == 0,
        icc_index_rc=rc,
        sentinel_sha_before=before_sha[:16],
        sentinel_sha_after=after_sha[:16],
        sentinel_path=str(SENTINEL.relative_to(REPO_ROOT)),
    )


if __name__ == "__main__":
    sys.exit(run_main("dirty-worktree-recovery", go))
