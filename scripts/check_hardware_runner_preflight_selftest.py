#!/usr/bin/env python3
"""Fixture tests for the hardware runner preflight checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_hardware_runner_preflight.py"


def runner(name: str, status: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "busy": False,
        "labels": ["ARM64", "m5", "macOS", "metal4-tensorops", "sdk26", "self-hosted"],
    }


def evidence(status: str = "matching_runner_online") -> dict[str, Any]:
    matching = [runner("m5", "online")]
    api_rc = 0
    api_error = ""
    registered = 1
    if status == "runner_api_unavailable":
        matching = []
        api_rc = 1
        api_error = "permission denied"
        registered = 0
    elif status == "matching_runner_offline":
        matching = [runner("m5", "offline")]
    elif status == "blocked_no_matching_runner":
        matching = []
        registered = 1

    online = [item for item in matching if item["status"] == "online"]
    return {
        "schema": "tensorcore.hardware_runner_preflight.v1",
        "meta": {
            "format": 1,
            "source": "tensorcore_hardware_runner_preflight",
            "head_sha": "abc123",
            "run_id": "456",
            "run_attempt": "1",
            "workflow": "Hardware Evidence",
        },
        "status": status,
        "repository": "owner/repo",
        "required_labels": ["self-hosted", "macOS", "ARM64", "m5", "sdk26", "metal4-tensorops"],
        "require_metal4_tensorops": "true",
        "runner_api_rc": api_rc,
        "runner_api_error": api_error,
        "registered_runner_count": registered,
        "matching_runner_count": len(matching),
        "online_matching_runner_count": len(online),
        "matching_runners": matching,
        "diagnostics": [
            {
                "id": f"hardware_runner_preflight.{status}",
                "diagnostic_class": {
                    "runner_api_unavailable": "token_unavailable",
                    "matching_runner_online": "runner_online",
                    "matching_runner_offline": "runner_offline",
                    "blocked_no_matching_runner": "runner_absent",
                }[status],
                "status": "passed" if status == "matching_runner_online" else "failed",
                "message": status,
                "recommended_action": "act",
            }
        ],
    }


def run_checker(payload: dict[str, Any], *args: str) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle)
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


def assert_passes(payload: dict[str, Any], *args: str) -> None:
    result = run_checker(payload, *args)
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)


def assert_fails(payload: dict[str, Any], needle: str, *args: str) -> None:
    result = run_checker(payload, *args)
    if result.returncode == 0:
        raise AssertionError("checker unexpectedly passed")
    output = result.stderr + result.stdout
    if needle not in output:
        raise AssertionError(f"expected {needle!r} in checker output:\n{output}")


def main() -> int:
    online = evidence()
    assert_passes(online, "--expected-head", "abc123", "--require-online-runner")
    assert_passes(online, "--require-runner-api", "--require-metal4-tensorops")

    unavailable = evidence("runner_api_unavailable")
    assert_passes(unavailable, "--require-metal4-tensorops")
    assert_fails(unavailable, "runner API must be available", "--require-runner-api")
    assert_fails(unavailable, "online matching runner required", "--require-online-runner")

    mismatch = copy.deepcopy(online)
    mismatch["meta"]["head_sha"] = "other"
    assert_fails(mismatch, "head mismatch", "--expected-head", "abc123")

    bad_count = copy.deepcopy(online)
    bad_count["matching_runner_count"] = 0
    assert_fails(bad_count, "matching_runner_count")

    missing_diag = copy.deepcopy(online)
    missing_diag["diagnostics"] = []
    assert_fails(missing_diag, "diagnostics must not be empty")

    missing_metal4_label = copy.deepcopy(online)
    missing_metal4_label["required_labels"] = ["self-hosted", "macOS", "ARM64"]
    assert_fails(missing_metal4_label, "missing required runner labels")

    mislabeled_runner = copy.deepcopy(online)
    mislabeled_runner["matching_runners"][0]["labels"] = ["self-hosted", "macOS", "ARM64"]
    assert_fails(mislabeled_runner, "missing required labels")

    print("hardware runner preflight checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
