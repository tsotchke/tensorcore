#!/usr/bin/env python3
"""Selftests for scripts/start_georefine_qwen_rank_probe.py."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
STARTER = ROOT / "scripts" / "start_georefine_qwen_rank_probe.py"


def run_starter(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(STARTER), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_print_script_renders_scheduler_contract() -> None:
    result = run_starter(
        "--target",
        "cosbox",
        "--resource",
        "cosbox:cuda3090",
        "--worker-resource",
        "gpu:cosbox:0",
        "--authority-lease-id",
        "lease-test",
        "--authority-owner",
        "georefine:test",
        "--repo-url",
        "git@example.invalid/georefine.git",
        "--ref",
        "main",
        "--repo-dir",
        "/repos/georefine",
        "--qllm-repo-dir",
        "/repos/qllm-tools",
        "--run-dir",
        "/runs/qwen-rank-probe",
        "--evidence-root",
        "/runs/evidence",
        "--cal-text",
        "/data/cal.txt",
        "--eval-text",
        "/data/eval.txt",
        "--model",
        "test/model",
        "--device",
        "cuda",
        "--dtype",
        "auto",
        "--run-target",
        "test-run",
        "--preflight-only",
        "--print-script",
        "--json",
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr + result.stdout)
    payload = json.loads(result.stdout)
    script = payload["script"]
    assert payload["ok"] is True
    assert payload["schema"] == "tensorcore.georefine_qwen_rank_probe.start.v1"
    assert "authority_lease_id=lease-test" in script
    assert "run_dir=/runs/qwen-rank-probe" in script
    assert "--authority-source tensorcore-scheduler" in script
    assert "--worker-lease-mode mirror" in script
    assert '--run-target "$run_target"' in script
    assert "run_target=test-run" in script
    assert script.index("qllm_resource_lease_missing") < script.index("preflight_ok")
    assert script.index("authority_lease_id_missing") < script.index("preflight_ok")
    assert "worker_resource_missing" in script
    assert "gpu:cosbox:0" in script


def test_invalid_numeric_arguments_fail() -> None:
    result = run_starter(
        "--target",
        "cosbox",
        "--resource",
        "cosbox:cuda3090",
        "--worker-resource",
        "gpu:cosbox:0",
        "--authority-owner",
        "georefine:test",
        "--repo-url",
        "git@example.invalid/georefine.git",
        "--ref",
        "main",
        "--repo-dir",
        "/repos/georefine",
        "--qllm-repo-dir",
        "/repos/qllm-tools",
        "--run-dir",
        "/runs/qwen-rank-probe",
        "--evidence-root",
        "/runs/evidence",
        "--cal-text",
        "/data/cal.txt",
        "--eval-text",
        "/data/eval.txt",
        "--model",
        "test/model",
        "--device",
        "cuda",
        "--dtype",
        "auto",
        "--run-target",
        "test-run",
        "--compression-ratio",
        "1.5",
        "--print-script",
        "--json",
    )
    if result.returncode == 0:
        raise AssertionError("invalid compression ratio unexpectedly passed")
    if "--compression-ratio must be in (0, 1]" not in result.stderr:
        raise AssertionError(result.stderr + result.stdout)


def test_worker_resource_is_required() -> None:
    result = run_starter(
        "--target",
        "cosbox",
        "--resource",
        "cosbox:cuda3090",
        "--authority-owner",
        "georefine:test",
        "--repo-url",
        "git@example.invalid/georefine.git",
        "--ref",
        "main",
        "--repo-dir",
        "/repos/georefine",
        "--qllm-repo-dir",
        "/repos/qllm-tools",
        "--run-dir",
        "/runs/qwen-rank-probe",
        "--evidence-root",
        "/runs/evidence",
        "--cal-text",
        "/data/cal.txt",
        "--eval-text",
        "/data/eval.txt",
        "--model",
        "test/model",
        "--device",
        "cuda",
        "--dtype",
        "auto",
        "--run-target",
        "test-run",
        "--print-script",
        "--json",
    )
    if result.returncode == 0:
        raise AssertionError("missing worker resource unexpectedly passed")
    if "--worker-resource is required" not in result.stderr:
        raise AssertionError(result.stderr + result.stdout)


def main() -> int:
    test_print_script_renders_scheduler_contract()
    test_invalid_numeric_arguments_fail()
    test_worker_resource_is_required()
    print("GeoRefine Qwen rank probe starter selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
