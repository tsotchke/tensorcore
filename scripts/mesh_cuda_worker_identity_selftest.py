#!/usr/bin/env python3
"""Selftest for scripts/mesh_cuda_worker_identity.py."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mesh_cuda_worker_identity.py"


def fake_nvidia_smi(directory: pathlib.Path) -> pathlib.Path:
    path = directory / "nvidia-smi"
    path.write_text(
        "#!/bin/sh\n"
        "printf '1234, python qllm trainer, 8192\\n'\n"
        "printf '9999, unrelated worker, 64\\n'\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def run_identity(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        nvidia_smi = fake_nvidia_smi(pathlib.Path(tmp))
        ok = run_identity(
            "--nvidia-smi",
            str(nvidia_smi),
            "--process-substring",
            "qllm",
            "--require-cuda-process",
        )
        if ok.returncode != 0:
            raise AssertionError(ok.stderr)
        payload = json.loads(ok.stdout)
        if payload["schema"] != "tensorcore.mesh_cuda_worker_identity.v1":
            raise AssertionError("unexpected schema")
        if payload["cuda_pids"] != [1234]:
            raise AssertionError(f"unexpected cuda_pids: {payload['cuda_pids']!r}")
        if payload["cuda_processes"][0]["used_gpu_memory_mb"] != 8192:
            raise AssertionError("used GPU memory parse failed")

        missing = run_identity(
            "--nvidia-smi",
            str(nvidia_smi),
            "--process-substring",
            "does-not-exist",
            "--require-cuda-process",
        )
        if missing.returncode == 0:
            raise AssertionError("missing process should fail when required")

    print("mesh CUDA worker identity selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
