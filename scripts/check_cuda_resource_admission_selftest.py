#!/usr/bin/env python3
"""Selftest for scripts/check_cuda_resource_admission.py."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_cuda_resource_admission.py"


def fake_nvidia_smi(directory: pathlib.Path, rows: list[str]) -> pathlib.Path:
    path = directory / "nvidia-smi"
    lines = ["#!/bin/sh\n"]
    for row in rows:
        lines.append(f"printf '%s\\n' {row!r}\n")
    path.write_text("".join(lines), encoding="utf-8")
    path.chmod(0o755)
    return path


def run_gate(nvidia_smi: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--nvidia-smi",
            str(nvidia_smi),
            "--json",
            *args,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        idle = run_gate(fake_nvidia_smi(root, []))
        if idle.returncode != 0:
            raise AssertionError(idle.stdout + idle.stderr)
        if json.loads(idle.stdout)["reason"] != "ok":
            raise AssertionError("idle admission should pass")

        busy = run_gate(fake_nvidia_smi(root, ["1234, python qllm trainer, 8192"]))
        if busy.returncode == 0:
            raise AssertionError("unmanaged CUDA process should block admission")
        busy_payload = json.loads(busy.stdout)
        if busy_payload["reason"] != "blocked_cuda_compute_apps":
            raise AssertionError("unexpected busy reason")
        if busy_payload["blocked"][0]["used_memory_mib"] != 8192:
            raise AssertionError("GPU memory parse failed")

        allowed = run_gate(
            fake_nvidia_smi(root, ["4321, tensorcore-admission-probe, 16"]),
            "--allow-process-regex",
            "tensorcore-admission-probe",
            "--allowed-process-max-memory-mib",
            "64",
        )
        if allowed.returncode != 0:
            raise AssertionError(allowed.stdout + allowed.stderr)
        if json.loads(allowed.stdout)["allowed"][0]["pid"] != 4321:
            raise AssertionError("allowlist parse failed")

    print("CUDA resource admission selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
