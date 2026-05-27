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


def run_checker(directory: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(directory), "--max-size-ratio", "0.10", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def lossless_chat_verification() -> dict:
    return {
        "latest": {
            "status": "ok",
            "passed": True,
            "blockers": [],
            "probes": {
                "chat_roundtrip": {
                    "passed": True,
                },
                "base_vs_artifact": {
                    "passed": True,
                },
            },
        }
    }


def assert_passes(payload: dict, *, sidecar: dict | None = None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = pathlib.Path(tmp)
        write_cert(directory, payload)
        if sidecar is not None:
            (directory / "m2_chat_verification.json").write_text(
                json.dumps(sidecar),
                encoding="utf-8",
            )
        result = run_checker(directory)
    if result.returncode != 0:
        raise AssertionError(result.stderr + result.stdout)
    out = json.loads(result.stdout)
    assert out["heldout_ppl"] == payload["ppl_compressed_eval"]
    assert out["stored_size_bytes"] == payload["size_compressed_bytes"]
    assert out["chat_lossless"] is True


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
        "ppl_baseline_eval": 17.0,
        "ppl_delta_fraction_eval": 0.0147058824,
        "size_compressed_bytes": 123456,
        "size_original_bytes": 2000000,
        "size_ratio": 0.061728,
        "quality_gate": {"passed": True},
        "m2_target_kl_achievement": {"post_storage_kl_mean": 0.08},
        "chat_verification": lossless_chat_verification(),
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

    sidecar_chat = complete_cert()
    del sidecar_chat["chat_verification"]
    assert_passes(sidecar_chat, sidecar=lossless_chat_verification())

    no_chat = complete_cert()
    del no_chat["chat_verification"]
    assert_fails(no_chat, "chat verification")

    failed_chat = complete_cert()
    failed_chat["chat_verification"]["latest"]["probes"]["base_vs_artifact"]["passed"] = False
    assert_fails(failed_chat, "chat verification")

    no_quality_gate = complete_cert()
    del no_quality_gate["quality_gate"]
    assert_fails(no_quality_gate, "quality_gate")

    high_delta = complete_cert()
    high_delta["ppl_delta_fraction_eval"] = 0.06
    with tempfile.TemporaryDirectory() as tmp:
        directory = pathlib.Path(tmp)
        write_cert(directory, high_delta)
        result = run_checker(directory, "--max-ppl-delta", "0.05")
    assert result.returncode != 0
    assert "PPL delta" in result.stdout + result.stderr

    high_kl = complete_cert()
    high_kl["m2_target_kl_achievement"]["post_storage_kl_mean"] = 0.11
    with tempfile.TemporaryDirectory() as tmp:
        directory = pathlib.Path(tmp)
        write_cert(directory, high_kl)
        result = run_checker(directory, "--max-target-kl", "0.10")
    assert result.returncode != 0
    assert "target KL" in result.stdout + result.stderr

    print("georefine qwen artifact checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
