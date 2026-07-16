#!/usr/bin/env python3
"""Portable fixtures for the Apple-family runtime evidence contract."""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_apple_family_runtime_evidence.py"
EXPECTATIONS = {
    "M2": ("Apple M2 Ultra", 8, False, False, False, "15.2"),
    "M4": ("Apple M4 Max", 9, True, False, False, "26.0"),
    "M5": ("Apple M5 Max", 10, True, False, True, "26.0"),
}


def attempt(name: str, stdout: str, stderr: str) -> dict[str, Any]:
    digest = hashlib.sha256((stdout + "\0" + stderr).encode()).hexdigest()
    return {
        "name": name,
        "cmd": [f"/repo/build/tests/{name}"],
        "rc": 0,
        "stdout_tail": stdout,
        "stderr_tail": stderr,
        "output_sha256": digest,
    }


def evidence(chip: str) -> dict[str, Any]:
    name, family, bf16, i8, tensorops, sdk = EXPECTATIONS[chip]
    device_line = (
        f'[tensorcore] device="{name}" family=Apple{family} unified=yes vram=1MB '
        f"bf16_sg={'yes' if bf16 else 'no'} i8_sg={'yes' if i8 else 'no'} "
        f"tensorops_m5={'yes' if tensorops else 'no'}\n"
    )
    bf16_note = (
        f"[note] device family=Apple{family} supports bf16 simdgroup_matrix\n"
        if bf16
        else f"[note] device family=Apple{family} lacks bf16 simdgroup_matrix; testing MPS fallback path instead\n"
    )
    tensor_status = (
        f'tensorops_runtime_status=passed backend=tensorops_m5 family=Apple{family} device="{name}"\n'
        if chip == "M5"
        else f'tensorops_runtime_status=skipped_no_m5 family=Apple{family} device="{name}"\n'
    )
    trace = [
        attempt(
            "test_device",
            f"family policy    : Apple {chip}=Apple{family} verified\n",
            device_line,
        ),
        attempt(
            "test_gemm_bf16",
            bf16_note
            + "mps_bf16_sw_fallback max_abs=0.000e+00\n"
            + "padded transpose beta bf16 backend=mps padding=OK  OK\n",
            device_line,
        ),
        attempt(
            "test_gemm_i8",
            f"[note] device family=Apple{family} uses the public MPS i8 fallback\n"
            + "mps_i8_sw_fallback errors=0/12 max_abs=0\n"
            + "M=64 backend=mps errors=0/4096 OK\n"
            + "padded transpose beta i8 backend=mps errors=0/1155 padding=OK  OK\n",
            device_line,
        ),
        attempt("test_tensorops_runtime", tensor_status, device_line),
    ]
    return {
        "schema": "tensorcore.apple_family_runtime_evidence.v1",
        "meta": {
            "format": 1,
            "source": "tensorcore_apple_family_probe",
            "git_head": "abc123",
            "git_dirty": False,
            "policy_sha256": "0" * 64,
        },
        "host": {
            "system": "Darwin",
            "machine": "arm64",
            "chip": chip,
            "sdk_version": sdk,
            "identity_probe": {"cmd": ["system_profiler"], "rc": 0, "output_sha256": "1" * 64},
        },
        "reservation": {
            "resource": "enki:metal_m4_tsotchke_chan" if chip == "M4" else None,
            "authority_owner": "tsotchke-chan:public-evidence" if chip == "M4" else "tensorcore:public-evidence",
            "authorized": True,
        },
        "device": {
            "name": name,
            "chip": chip,
            "family": family,
            "bf16_simdgroup": bf16,
            "i8_simdgroup": i8,
            "tensorops_m5": tensorops,
        },
        "checks": {test["name"]: {"status": "passed"} for test in trace},
        "trace": trace,
        "status": "passed",
        "summary": {"blocked_reasons": [], "failure_reasons": []},
    }


def run(data: dict[str, Any], *args: str) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(data, handle)
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


def passes(data: dict[str, Any], *args: str) -> None:
    result = run(data, *args)
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)


def fails(data: dict[str, Any], needle: str, *args: str) -> None:
    result = run(data, *args)
    text = result.stderr + result.stdout
    if result.returncode == 0:
        raise AssertionError("checker unexpectedly passed")
    if needle not in text:
        raise AssertionError(f"expected {needle!r} in checker output:\n{text}")


def main() -> int:
    for chip in EXPECTATIONS:
        fixture = evidence(chip)
        passes(fixture, "--require-pass", "--require-chip", chip)
    passes(evidence("M2"), "--require-clean-head", "--git-head", "abc123")

    wrong_family = copy.deepcopy(evidence("M4"))
    wrong_family["device"]["family"] = 10
    fails(wrong_family, "device.family must be 9")

    integer_claim = copy.deepcopy(evidence("M5"))
    integer_claim["device"]["i8_simdgroup"] = True
    fails(integer_claim, "device.i8_simdgroup must be False")

    unauthorized = copy.deepcopy(evidence("M4"))
    unauthorized["reservation"]["authority_owner"] = "tensorcore:public-evidence"
    unauthorized["reservation"]["authorized"] = False
    fails(unauthorized, "tsotchke-chan prefix")

    old_sdk = copy.deepcopy(evidence("M5"))
    old_sdk["host"]["sdk_version"] = "15.2"
    fails(old_sdk, "SDK 26.0 or newer")

    dirty = copy.deepcopy(evidence("M2"))
    dirty["meta"]["git_dirty"] = True
    fails(dirty, "clean tracked git tree", "--require-clean-head", "--git-head", "abc123")

    skipped_m5 = copy.deepcopy(evidence("M5"))
    skipped_m5["trace"][-1]["stdout_tail"] = "tensorops_runtime_status=skipped_no_m5\n"
    fails(skipped_m5, "TensorOps trace must report passed")

    print("Apple family runtime evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
