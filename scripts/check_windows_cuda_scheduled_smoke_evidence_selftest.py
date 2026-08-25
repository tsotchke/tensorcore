#!/usr/bin/env python3
"""Fixture tests for the Windows scheduled CUDA smoke evidence checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_windows_cuda_scheduled_smoke_evidence.py"


def evidence() -> dict[str, Any]:
    return {
        "schema": "tensorcore.windows_cuda_scheduled_smoke.evidence.v1",
        "schema_version": 1,
        "checked_at_unix": time.time(),
        "phase": "completed",
        "resource": "jack-blupc:cuda3060",
        "job": "jack-cuda3060-smoke",
        "driver_visible": True,
        "toolchain_found": True,
        "wddm_admission_ok": True,
        "build_smoke_passed": True,
        "runtime_smoke_passed": True,
        "scheduler_lease_held": True,
        "worker_identity_recorded": True,
        "lease_id": "lease-jack",
        "smoke_artifact": {
            "schema": "tensorcore.windows_cuda_smoke.v1",
            "ok": True,
            "resource": "jack-blupc:cuda3060",
            "state": "completed",
            "build_ok": True,
            "runtime_ok": True,
        },
        "worker_identity": {
            "schema": "tensorcore.mesh_worker_identity.v1",
            "ok": True,
            "resource": "jack-blupc:cuda3060",
            "worker_host": "DESKTOP-JACK-BLUPC",
            "worker_pid": 1234,
        },
    }


def write_json(directory: pathlib.Path, name: str, data: dict[str, Any]) -> pathlib.Path:
    path = directory / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def run_checker(path: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(path), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def assert_passes(path: pathlib.Path, *args: str) -> None:
    result = run_checker(path, *args)
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)


def assert_fails(path: pathlib.Path, needle: str, *args: str) -> None:
    result = run_checker(path, *args)
    if result.returncode == 0:
        raise AssertionError("checker unexpectedly passed")
    output = result.stderr + result.stdout
    if needle not in output:
        raise AssertionError(f"expected {needle!r} in checker output:\n{output}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        directory = pathlib.Path(tmp)
        good = write_json(directory, "good.json", evidence())
        assert_passes(good, "--require-complete", "--max-age-sec", "60")

        missing_worker = copy.deepcopy(evidence())
        missing_worker["worker_identity_recorded"] = False
        missing_worker_path = write_json(directory, "missing-worker.json", missing_worker)
        assert_fails(
            missing_worker_path,
            "worker identity must be recorded",
            "--require-worker-identity",
        )

        missing_artifact = copy.deepcopy(evidence())
        del missing_artifact["smoke_artifact"]
        missing_artifact_path = write_json(directory, "missing-artifact.json", missing_artifact)
        assert_fails(missing_artifact_path, "smoke_artifact is required", "--require-complete")

        string_bool = copy.deepcopy(evidence())
        string_bool["runtime_smoke_passed"] = "true"
        string_bool_path = write_json(directory, "string-bool.json", string_bool)
        assert_fails(string_bool_path, "runtime_smoke_passed must be boolean")

        artifact_mismatch = copy.deepcopy(evidence())
        artifact_mismatch["smoke_artifact"]["resource"] = "other:cuda"
        artifact_mismatch_path = write_json(directory, "artifact-mismatch.json", artifact_mismatch)
        assert_fails(artifact_mismatch_path, "smoke_artifact resource must match")

        worker_mismatch = copy.deepcopy(evidence())
        worker_mismatch["worker_identity"]["resource"] = "other:cuda"
        worker_mismatch_path = write_json(directory, "worker-mismatch.json", worker_mismatch)
        assert_fails(worker_mismatch_path, "worker_identity resource must match")

        stale = copy.deepcopy(evidence())
        stale["checked_at_unix"] = 1
        stale_path = write_json(directory, "stale.json", stale)
        assert_fails(stale_path, "evidence is stale", "--max-age-sec", "1")

        incomplete = copy.deepcopy(evidence())
        incomplete["runtime_smoke_passed"] = False
        incomplete_path = write_json(directory, "incomplete.json", incomplete)
        assert_fails(incomplete_path, "CUDA runtime smoke must pass", "--require-complete")

        launched = copy.deepcopy(evidence())
        launched["phase"] = "launched"
        launched_path = write_json(directory, "launched.json", launched)
        assert_fails(launched_path, "phase must be completed", "--require-complete")

    print("Windows scheduled CUDA smoke evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
