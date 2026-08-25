#!/usr/bin/env python3
"""Portable transport-integrity tests for Apple-family evidence intake."""

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
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_apple_family_runtime_evidence_selftest as fixtures  # noqa: E402


INTAKE = SCRIPTS / "intake_apple_family_runtime_evidence.py"
HEAD = "a" * 40


def evidence(chip: str) -> dict[str, Any]:
    data = fixtures.evidence(chip)
    data["meta"]["git_head"] = HEAD
    if chip == "M4":
        data["reservation"] = {
            "resource": "collaborator:alice:macbook",
            "scope": "independent",
            "authority_owner": "collaborator:alice",
            "authorized": True,
        }
    return data


def encoded(data: dict[str, Any]) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def record(raw: bytes, *, chip: str = "M4", head: str = HEAD, digest: str | None = None) -> str:
    return (
        "APPLE_FAMILY_EVIDENCE_HANDOFF "
        f"chip={chip} head={head} sha256={digest or hashlib.sha256(raw).hexdigest()} "
        "path=/contributor/build output/apple_family_runtime_evidence.json"
    )


def run(
    raw: bytes,
    handoff: str,
    *,
    chip: str = "M4",
    head: str = HEAD,
    record_file: bool = False,
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        evidence_path = root / "evidence.json"
        evidence_path.write_bytes(raw)
        argv = [
            sys.executable,
            str(INTAKE),
            str(evidence_path),
            "--expected-head",
            head,
            "--require-chip",
            chip,
            "--json",
        ]
        if record_file:
            record_path = root / "handoff.log"
            record_path.write_text(f"collector output\n{handoff}\n", encoding="utf-8")
            argv.extend(("--handoff-record-file", str(record_path)))
        else:
            argv.extend(("--handoff-record", handoff))
        return subprocess.run(
            argv,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )


def passes(result: subprocess.CompletedProcess[str], chip: str) -> None:
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    report = json.loads(result.stdout)
    assert report["schema"] == "tensorcore.apple_family_evidence_intake.v1"
    assert report["status"] == "accepted"
    assert report["chip"] == chip
    assert report["git_head"] == HEAD


def fails(result: subprocess.CompletedProcess[str], needle: str) -> None:
    output = result.stdout + result.stderr
    if result.returncode == 0:
        raise AssertionError("intake unexpectedly accepted invalid evidence")
    if needle not in output:
        raise AssertionError(f"expected {needle!r} in intake output:\n{output}")


def main() -> int:
    m4_raw = encoded(evidence("M4"))
    passes(run(m4_raw, record(m4_raw)), "M4")

    m5_raw = encoded(evidence("M5"))
    passes(run(m5_raw, record(m5_raw, chip="M5"), chip="M5", record_file=True), "M5")

    fails(run(m4_raw, record(m4_raw, digest="0" * 64)), "digest mismatch")
    fails(run(m4_raw, record(m4_raw), chip="M5"), "does not match required M5")
    fails(run(m4_raw, record(m4_raw), head="b" * 40), "does not match expected")
    fails(run(m4_raw, record(m4_raw) + "\n" + record(m4_raw)), "exactly one")
    fails(run(m4_raw + b"\n", record(m4_raw)), "digest mismatch")

    wrong_family = copy.deepcopy(evidence("M4"))
    wrong_family["device"]["family"] = 10
    wrong_family_raw = encoded(wrong_family)
    fails(run(wrong_family_raw, record(wrong_family_raw)), "device.family must be 9")

    duplicate_json = b'{"schema":"first","schema":"second"}'
    fails(run(duplicate_json, record(duplicate_json)), "duplicate object key")
    fails(run(m4_raw, "not a handoff record"), "exactly one")

    print("Apple family evidence intake selftest OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
