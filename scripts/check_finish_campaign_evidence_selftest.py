#!/usr/bin/env python3
"""Fixture tests for the TensorCore finish-campaign evidence join."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_finish_campaign_evidence.py"
HEAD = "a" * 40


def clean_meta() -> dict[str, Any]:
    return {"git_head": HEAD, "git_dirty": False}


def release() -> dict[str, Any]:
    return {
        "schema": "tensorcore.release_smoke.runtime_evidence.v1",
        "meta": clean_meta(),
        "status": "passed",
        "run": {"phase": "complete"},
        "checks": {
            "public_exports": {"passed": True},
            "public_headers": {"passed": True},
            "python_ffi_surface": {"passed": True},
            "python_constants": {"passed": True},
            "python_abi_layout": {"passed": True},
            "tests": {"passed": True, "gpu_device_available": True},
            "artifact_privacy": {"passed": True},
            "installed_wheel_smoke": {"passed": True},
            "consumers": {"cmake": {"passed": True}, "pkg_config": {"passed": True}},
            "packaging_and_consumers": {"runtime_covered": True},
        },
    }


def sanitizer() -> dict[str, Any]:
    return {
        "schema": "tensorcore.release_sanitizers.v1",
        "meta": clean_meta(),
        "status": "passed",
        "checks": {
            "address": {"enabled": True},
            "undefined": {"enabled": True},
            "ctest": {"passed": True},
        },
    }


def verification() -> dict[str, Any]:
    names = (
        "documentation", "version_consistency", "source_privacy",
        "history_privacy", "campaign", "script_selftests",
        "adversarial_evals", "icc_precommit", "architecture_model",
    )
    return {
        "schema": "tensorcore.release_verification.v1",
        "meta": clean_meta(),
        "status": "passed",
        "checks": {name: {"passed": True} for name in names},
    }


def windows() -> dict[str, Any]:
    return {
        "schema": "tensorcore.windows_host_smoke.evidence.v2",
        "schema_version": 2,
        "git_head": HEAD,
        "git_dirty": False,
        "runtime_status": "passed",
        "checkout": "configured",
        "source": "configured",
        "host": {"platform": "windows", "architecture": "AMD64", "os": "Microsoft Windows"},
    }


def cuda(capability: str, bf16: bool) -> dict[str, Any]:
    kernels = {
        name: {"status": "passed", "backend": "cuda"}
        for name in ("cuda_gemm_sgemm", "cuda_gemm_hgemm", "cuda_gemm_i8")
    }
    kernels["cuda_gemm_bf16"] = {
        "status": "passed" if bf16 else "skipped_unsupported",
        "backend": "cuda" if bf16 else None,
    }
    return {
        "git_head": HEAD,
        "git_dirty": False,
        "runtime_status": "passed",
        "backend": "cuda",
        "cuda_build_enabled": True,
        "device": {"compute_capability": capability},
        "gemm_kernels": kernels,
        "training_kernels": {"adamw_step_fp32": {"backend": "cuda"}},
    }


def write(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def run(directory: pathlib.Path, blackwell: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    fixtures = {
        "release.json": release(),
        "sanitizer.json": sanitizer(),
        "verification.json": verification(),
        "windows.json": windows(),
        "xavier.json": cuda("7.2", False),
        "blackwell.json": blackwell,
    }
    for name, value in fixtures.items():
        write(directory / name, value)
    return subprocess.run(
        [
            sys.executable, str(CHECKER),
            "--release-evidence", str(directory / "release.json"),
            "--sanitizer-evidence", str(directory / "sanitizer.json"),
            "--verification-evidence", str(directory / "verification.json"),
            "--windows-evidence", str(directory / "windows.json"),
            "--xavier-evidence", str(directory / "xavier.json"),
            "--blackwell-evidence", str(directory / "blackwell.json"),
            "--output", str(directory / "finish.jsonl"),
            "--git-head", HEAD,
        ],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as raw:
        directory = pathlib.Path(raw)
        good = run(directory, cuda("12.0", True))
        if good.returncode != 0:
            raise AssertionError(good.stderr or good.stdout)
        rows = [json.loads(line) for line in (directory / "finish.jsonl").read_text().splitlines()]
        gates = [row for row in rows if row.get("kind") == "tensorcore_finish_gate"]
        if len(gates) != 6 or any(row.get("value") != "PASS" for row in gates):
            raise AssertionError(f"unexpected finish-gate rows: {gates!r}")

        stale = cuda("12.0", True)
        stale["git_head"] = "b" * 40
        failed = run(directory, stale)
        if failed.returncode == 0 or "Blackwell CUDA evidence is not bound" not in failed.stderr:
            raise AssertionError(f"stale Blackwell fixture did not fail:\n{failed.stderr}{failed.stdout}")

    print("finish campaign evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
