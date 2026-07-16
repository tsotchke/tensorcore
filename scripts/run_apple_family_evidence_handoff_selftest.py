#!/usr/bin/env python3
"""Portable contract tests for the borrowed-hardware handoff command."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "scripts" / "run_apple_family_evidence_handoff.py"
HEAD = "a" * 40


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HANDOFF), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def main() -> int:
    independent = run(
        "--chip",
        "M4",
        "--expected-head",
        HEAD,
        "--hardware-resource",
        "collaborator:example:m4",
        "--authority-owner",
        "collaborator:example",
        "--dry-run",
        "--json",
    )
    if independent.returncode != 0:
        raise AssertionError(independent.stderr or independent.stdout)
    plan = json.loads(independent.stdout)
    assert plan["status"] == "planned"
    assert plan["chip"] == "M4"
    assert len(plan["commands"]) == 3
    assert "-DTC_BUILD_BENCH=OFF" in plan["commands"][0]
    assert "--expected-chip M4" in plan["commands"][1]
    assert "--require-clean-head" in plan["commands"][2]

    reserved = run(
        "--chip",
        "M4",
        "--expected-head",
        HEAD,
        "--hardware-resource",
        "enki:metal_m4_tsotchke_chan",
        "--authority-owner",
        "tensorcore:unauthorized",
        "--dry-run",
    )
    assert reserved.returncode != 0
    assert "tsotchke-chan" in reserved.stderr

    m5 = run(
        "--chip",
        "M5",
        "--expected-head",
        HEAD,
        "--hardware-resource",
        "collaborator:example:m5",
        "--authority-owner",
        "collaborator:example",
        "--dry-run",
        "--json",
    )
    if m5.returncode != 0:
        raise AssertionError(m5.stderr or m5.stdout)
    assert json.loads(m5.stdout)["chip"] == "M5"

    short_head = run(
        "--chip",
        "M5",
        "--expected-head",
        "abc123",
        "--hardware-resource",
        "collaborator:example:m5",
        "--authority-owner",
        "collaborator:example",
        "--dry-run",
    )
    assert short_head.returncode != 0
    assert "full lowercase 40-character" in short_head.stderr

    print("Apple family evidence handoff selftest OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
