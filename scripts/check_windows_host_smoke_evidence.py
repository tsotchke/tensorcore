#!/usr/bin/env python3
"""Validate machine-readable evidence from scripts/run_windows_host_smoke.sh."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = "tensorcore.windows_host_smoke.evidence.v1"
VALID_STATUSES = {"passed", "skipped_no_smoke"}


def fail(message: str) -> int:
    print(f"Windows host smoke evidence invalid: {message}", file=sys.stderr)
    return 1


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def require_dict(value: Any, name: str) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument("--git-head", default=git_head())
    parser.add_argument("--require-windows", action="store_true")
    parser.add_argument("--require-clean-head", action="store_true")
    parser.add_argument("--require-python", action="store_true")
    args = parser.parse_args()

    try:
        evidence = json.loads(args.path.read_text(encoding="utf-8"))
    except Exception as exc:
        return fail(f"could not read JSON: {exc}")

    if evidence.get("schema") != SCHEMA:
        return fail(f"schema must be {SCHEMA!r}")
    if evidence.get("schema_version") != 1:
        return fail("schema_version must be 1")

    status = evidence.get("runtime_status")
    if status not in VALID_STATUSES:
        return fail(f"unexpected runtime_status={status!r}")
    if args.require_windows and status != "passed":
        return fail(f"--require-windows needs passed evidence, got {status}")

    if args.require_clean_head:
        if not args.git_head:
            return fail("expected git head is unavailable for Windows evidence check")
        if evidence.get("git_dirty") is not False:
            return fail("Windows evidence must be from a clean git tree")
        if evidence.get("git_head") != args.git_head:
            return fail(
                "Windows evidence git_head mismatch: "
                f"{evidence.get('git_head')!r} != {args.git_head!r}"
            )

    host = require_dict(evidence.get("host"), "host")
    if host is None:
        return fail("host must be an object")
    if not host.get("computer_name"):
        return fail("host.computer_name is required")
    if "Windows" not in str(host.get("os") or ""):
        return fail("host.os must identify Windows")

    bootstrap = require_dict(evidence.get("bootstrap"), "bootstrap")
    if bootstrap is None:
        return fail("bootstrap must be an object")

    if status == "passed":
        if bootstrap.get("ran") is not True:
            return fail("passed evidence must have bootstrap.ran=true")
        if not evidence.get("git_head"):
            return fail("passed evidence must include git_head")
        if not evidence.get("repo"):
            return fail("passed evidence must include repo")
        if args.require_python and bootstrap.get("skip_python") is not False:
            return fail("--require-python cannot accept skip_python=true")

    print(
        "Windows host smoke evidence OK: "
        f"status={status} host={host.get('computer_name')} "
        f"head={str(evidence.get('git_head') or '')[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
