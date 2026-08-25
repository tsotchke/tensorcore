#!/usr/bin/env python3
"""Selftests for scripts/start_georefine_qllm_native_probe.py."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
STARTER = ROOT / "scripts" / "start_georefine_qllm_native_probe.py"


def run_starter(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(STARTER), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def base_args() -> list[str]:
    return [
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
        "--repo-dir",
        "/repos/qllm",
        "--qllm-repo-dir",
        "/repos/qllm",
        "--python-bin",
        "/venv/bin/python",
        "--run-dir",
        "/protected/georefine/runs/native",
        "--evidence-root",
        "/protected/georefine/evidence",
    ]


def test_print_script_renders_native_probe_contract() -> None:
    result = run_starter(*base_args(), "--preflight-only", "--print-script", "--json")
    if result.returncode != 0:
        raise AssertionError(result.stderr + result.stdout)
    payload = json.loads(result.stdout)
    script = payload["script"]

    assert payload["ok"] is True
    assert payload["schema"] == "tensorcore.georefine_qllm_native_probe.start.v1"
    assert "georefine_native_coherence_probe.py" in script
    assert "check_georefine_native_coherence_artifact.py" in script
    assert "--authority-lease-id \"$authority_lease_id\"" in script
    assert "--worker-resource \"$worker_resource\"" in script
    assert "--preflight-only" in script
    assert "preflight_ok" in script
    assert "experiments.georefine.m2_compress" not in script
    assert "m2_compress" not in script
    assert "import torch" not in script


def test_worker_resource_is_required_before_ssh() -> None:
    args = base_args()
    idx = args.index("--worker-resource")
    del args[idx:idx + 2]
    result = run_starter(*args, "--print-script", "--json")
    if result.returncode == 0:
        raise AssertionError("missing worker resource unexpectedly passed")
    assert "--worker-resource is required" in result.stderr


def test_authority_lease_id_is_required_before_ssh() -> None:
    args = base_args()
    idx = args.index("--authority-lease-id")
    del args[idx:idx + 2]
    result = run_starter(*args, "--print-script", "--json")
    if result.returncode == 0:
        raise AssertionError("missing authority lease id unexpectedly passed")
    assert "--authority-lease-id is required" in result.stderr


def test_untrusted_run_dir_is_rejected_before_ssh() -> None:
    args = base_args()
    args[args.index("--run-dir") + 1] = "/tmp/native"
    result = run_starter(*args, "--print-script", "--json")
    if result.returncode == 0:
        raise AssertionError("untrusted run dir unexpectedly passed")
    assert "--run-dir must be an absolute protected path" in result.stderr


def main() -> int:
    tests = [
        test_print_script_renders_native_probe_contract,
        test_worker_resource_is_required_before_ssh,
        test_authority_lease_id_is_required_before_ssh,
        test_untrusted_run_dir_is_rejected_before_ssh,
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {test.__name__}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
