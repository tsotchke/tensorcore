#!/usr/bin/env python3
"""Fixture tests for fetch_m5_tensorops_runtime_evidence.py."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_m5_tensorops_runtime_evidence.py"

spec = importlib.util.spec_from_file_location("fetch_m5_tensorops_runtime_evidence", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load fetch_m5_tensorops_runtime_evidence.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["fake"], 0, stdout=stdout, stderr="")


def run_list_payload(items: list[dict[str, Any]]) -> str:
    return json.dumps(items)


def test_latest_run_id_selects_successful_matching_head() -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return completed(
            run_list_payload(
                [
                    {
                        "databaseId": 101,
                        "headSha": "other",
                        "status": "completed",
                        "conclusion": "success",
                    },
                    {
                        "databaseId": 102,
                        "headSha": "abc123",
                        "status": "completed",
                        "conclusion": "failure",
                    },
                    {
                        "databaseId": 103,
                        "headSha": "abc123",
                        "status": "completed",
                        "conclusion": "success",
                    },
                ]
            )
        )

    original = fetch.run
    fetch.run = fake_run
    try:
        assert fetch.latest_run_id("owner/repo", "abc123", 10) == "103"
        assert calls[0][:5] == ["gh", "run", "list", "--repo", "owner/repo"]
    finally:
        fetch.run = original


def test_latest_run_id_fails_without_match() -> None:
    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        del cmd
        return completed(
            run_list_payload(
                [
                    {
                        "databaseId": 201,
                        "headSha": "abc123",
                        "status": "in_progress",
                        "conclusion": None,
                    }
                ]
            )
        )

    original = fetch.run
    fetch.run = fake_run
    try:
        try:
            fetch.latest_run_id("owner/repo", "abc123", 10)
        except SystemExit as exc:
            assert "no successful hardware-evidence.yml run found" in str(exc)
        else:
            raise AssertionError("latest_run_id unexpectedly found a run")
    finally:
        fetch.run = original


def main() -> int:
    test_latest_run_id_selects_successful_matching_head()
    test_latest_run_id_fails_without_match()
    print("M5 TensorOps runtime evidence fetch selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
