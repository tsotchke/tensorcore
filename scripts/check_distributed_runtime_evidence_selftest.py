#!/usr/bin/env python3
"""Fixture tests for the distributed runtime evidence checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_distributed_runtime_evidence.py"


def coverage() -> dict[str, Any]:
    return {
        "lib/distributed/gloo_tcp.cpp": {
            "executed_lines": [404, 530, 583, 592, 667],
            "functions": {
                "tcp_connect_timeout": {"start_line": 404, "executed_lines": [404]},
                "advertised_peer_info": {"start_line": 530, "executed_lines": [530]},
                "ring_connect_timeout_ms": {"start_line": 583, "executed_lines": [583]},
                "accept_with_timeout": {"start_line": 592, "executed_lines": [592]},
                "close_gloo_state": {"start_line": 667, "executed_lines": [667]},
            },
        },
        "lib/distributed/diloco.cpp": {
            "executed_lines": [241, 280, 387],
            "functions": {
                "apply_outer_optimizer": {"start_line": 280, "executed_lines": [280]},
                "compute_delta_topk": {"start_line": 241, "executed_lines": [241]},
                "do_outer_step": {"start_line": 387, "executed_lines": [387]},
            },
        },
    }


def passed_evidence() -> dict[str, Any]:
    return {
        "schema": "tensorcore.distributed_runtime_evidence.v1",
        "meta": {
            "format": 1,
            "source": "tensorcore_distributed_runtime_probe",
            "git_head": "abc123",
            "git_dirty": False,
        },
        "status": "passed",
        "paths": {
            "build_dir": "/repo/build",
            "evidence": "/repo/build/distributed_runtime_evidence.json",
        },
        "checks": {
            "gloo_ring_fork": {"status": "passed", "binary": "/repo/build/tests/test_gloo_ring_fork", "trace": "gloo_ring_fork"},
            "diloco_gloo_fork": {"status": "passed", "binary": "/repo/build/tests/test_diloco_gloo_fork", "trace": "diloco_gloo_fork"},
            "diloco_sparse_fork": {"status": "passed", "binary": "/repo/build/tests/test_diloco_sparse_fork", "trace": "diloco_sparse_fork"},
        },
        "trace": [
            {"name": "gloo_ring_fork", "cmd": ["test"], "cwd": "/repo", "rc": 0},
            {"name": "diloco_gloo_fork", "cmd": ["test"], "cwd": "/repo", "rc": 0},
            {"name": "diloco_sparse_fork", "cmd": ["test"], "cwd": "/repo", "rc": 0},
        ],
        "files": coverage(),
        "summary": {
            "checks_passed": True,
            "blocked_reasons": [],
            "failure_reasons": [],
            "required_functions": [
                "lib/distributed/diloco.cpp:apply_outer_optimizer",
                "lib/distributed/diloco.cpp:compute_delta_topk",
                "lib/distributed/diloco.cpp:do_outer_step",
                "lib/distributed/gloo_tcp.cpp:accept_with_timeout",
                "lib/distributed/gloo_tcp.cpp:advertised_peer_info",
                "lib/distributed/gloo_tcp.cpp:close_gloo_state",
                "lib/distributed/gloo_tcp.cpp:ring_connect_timeout_ms",
                "lib/distributed/gloo_tcp.cpp:tcp_connect_timeout",
            ],
            "covered_functions": [
                "lib/distributed/diloco.cpp:apply_outer_optimizer",
                "lib/distributed/diloco.cpp:compute_delta_topk",
                "lib/distributed/diloco.cpp:do_outer_step",
                "lib/distributed/gloo_tcp.cpp:accept_with_timeout",
                "lib/distributed/gloo_tcp.cpp:advertised_peer_info",
                "lib/distributed/gloo_tcp.cpp:close_gloo_state",
                "lib/distributed/gloo_tcp.cpp:ring_connect_timeout_ms",
                "lib/distributed/gloo_tcp.cpp:tcp_connect_timeout",
            ],
            "missing_functions": [],
        },
    }


def blocked_evidence() -> dict[str, Any]:
    evidence = passed_evidence()
    evidence["status"] = "blocked"
    evidence["checks"]["gloo_ring_fork"] = {
        "status": "blocked",
        "blocked_reason": "loopback_unavailable",
        "binary": "/repo/build/tests/test_gloo_ring_fork",
        "trace": "gloo_ring_fork",
    }
    evidence["trace"][0]["rc"] = 77
    evidence["files"] = {
        "lib/distributed/diloco.cpp": coverage()["lib/distributed/diloco.cpp"],
    }
    evidence["summary"]["checks_passed"] = False
    evidence["summary"]["blocked_reasons"] = ["gloo_ring_fork:loopback_unavailable"]
    evidence["summary"]["covered_functions"] = [
        "lib/distributed/diloco.cpp:apply_outer_optimizer",
        "lib/distributed/diloco.cpp:compute_delta_topk",
        "lib/distributed/diloco.cpp:do_outer_step",
    ]
    evidence["summary"]["missing_functions"] = [
        "lib/distributed/gloo_tcp.cpp:accept_with_timeout",
        "lib/distributed/gloo_tcp.cpp:advertised_peer_info",
        "lib/distributed/gloo_tcp.cpp:close_gloo_state",
        "lib/distributed/gloo_tcp.cpp:ring_connect_timeout_ms",
        "lib/distributed/gloo_tcp.cpp:tcp_connect_timeout",
    ]
    return evidence


def run_checker(evidence: dict[str, Any], *args: str) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(evidence, handle)
        path = pathlib.Path(handle.name)
    try:
        return subprocess.run(
            [sys.executable, str(CHECKER), str(path), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        path.unlink(missing_ok=True)


def assert_passes(evidence: dict[str, Any], *args: str) -> None:
    result = run_checker(evidence, *args)
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)


def assert_fails(evidence: dict[str, Any], needle: str, *args: str) -> None:
    result = run_checker(evidence, *args)
    if result.returncode == 0:
        raise AssertionError("checker unexpectedly passed")
    output = result.stderr + result.stdout
    if needle not in output:
        raise AssertionError(f"expected {needle!r} in checker output:\n{output}")


def main() -> int:
    passed = passed_evidence()
    assert_passes(passed)
    assert_passes(passed, "--require-pass", "--git-head", "abc123", "--require-clean-head")

    blocked = blocked_evidence()
    assert_passes(blocked)
    assert_fails(blocked, "--require-pass needs passed evidence", "--require-pass")

    missing_function = copy.deepcopy(passed)
    del missing_function["files"]["lib/distributed/gloo_tcp.cpp"]["functions"]["accept_with_timeout"]
    assert_fails(missing_function, "missing function coverage", "--require-pass")

    dirty = copy.deepcopy(passed)
    dirty["meta"]["git_dirty"] = True
    assert_fails(dirty, "clean git tree", "--git-head", "abc123", "--require-clean-head")

    stale = copy.deepcopy(passed)
    stale["meta"]["git_head"] = "stale"
    assert_fails(stale, "git_head mismatch", "--git-head", "abc123", "--require-clean-head")

    print("distributed runtime evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
