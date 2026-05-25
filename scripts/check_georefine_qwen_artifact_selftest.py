#!/usr/bin/env python3
"""Selftests for scripts/check_georefine_qwen_artifact.py."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_georefine_qwen_artifact.py"


def write_cert(directory: pathlib.Path, payload: dict) -> None:
    (directory / "m2_certificate.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def run_checker(directory: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(directory), "--max-size-ratio", "0.10"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def assert_passes(payload: dict) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = pathlib.Path(tmp)
        write_cert(directory, payload)
        result = run_checker(directory)
    if result.returncode != 0:
        raise AssertionError(result.stderr + result.stdout)
    out = json.loads(result.stdout)
    assert out["heldout_ppl"] == payload["ppl_compressed_eval"]
    assert out["stored_size_bytes"] == payload["size_compressed_bytes"]


def assert_fails(payload: dict, needle: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = pathlib.Path(tmp)
        write_cert(directory, payload)
        result = run_checker(directory)
    if result.returncode == 0:
        raise AssertionError("checker unexpectedly passed")
    if needle not in result.stdout + result.stderr:
        raise AssertionError(result.stdout + result.stderr)


def complete_cert() -> dict:
    return {
        "completed": True,
        "ppl_compressed_eval": 17.25,
        "size_compressed_bytes": 123456,
        "size_original_bytes": 2000000,
        "size_ratio": 0.061728,
    }


def main() -> int:
    assert_passes(complete_cert())

    incomplete = complete_cert()
    incomplete["completed"] = False
    assert_fails(incomplete, "completed is not true")

    no_ppl = complete_cert()
    del no_ppl["ppl_compressed_eval"]
    assert_fails(no_ppl, "held-out PPL")

    no_size = complete_cert()
    del no_size["size_compressed_bytes"]
    assert_fails(no_size, "stored size")

    too_large = complete_cert()
    too_large["size_ratio"] = 0.101
    assert_fails(too_large, "exceeds max")

    computed_ratio = complete_cert()
    del computed_ratio["size_ratio"]
    assert_passes(computed_ratio)

    print("georefine qwen artifact checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
