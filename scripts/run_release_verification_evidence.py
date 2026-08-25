#!/usr/bin/env python3
"""Run fail-closed release verification and emit a public-safe exact-head receipt."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


def git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def git_dirty() -> bool | None:
    try:
        for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
            if subprocess.run(
                ["git", *args], cwd=ROOT, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode != 0:
                return True
        return False
    except Exception:
        return None


def redact(text: str) -> str:
    text = re.sub(r"/(?:Users|home)/[^/\s\"']+", "<home>", text)
    return re.sub(r"(?i)\b[A-Z]:[\\/]+Users[\\/]+[^\\/\s\"']+", "<home>", text)


def run(command: list[str]) -> tuple[bool, str]:
    result = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.returncode == 0, redact(result.stdout)


def find_icc(explicit: str | None) -> str | None:
    candidates = [
        explicit,
        os.environ.get("ICC"),
        shutil.which("icc"),
        str(ROOT.parent / "infinite_context_coder" / "bin" / "icc"),
    ]
    for candidate in candidates:
        if candidate and pathlib.Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def check(name: str, command: list[str], results: dict[str, Any]) -> None:
    passed, output = run(command)
    results[name] = {"passed": passed, "status": "passed" if passed else "failed"}
    if not passed:
        tail = "\n".join(output.splitlines()[-20:])
        print(f"[tensorcore/verification] {name} FAILED\n{tail}", file=sys.stderr)
    else:
        print(f"[tensorcore/verification] {name} passed")


def write_evidence(path: pathlib.Path, checks: dict[str, Any]) -> dict[str, Any]:
    passed = all(isinstance(value, dict) and value.get("passed") is True for value in checks.values())
    artifact = {
        "schema": "tensorcore.release_verification.v1",
        "meta": {
            "git_head": git_output("rev-parse", "HEAD"),
            "git_dirty": git_dirty(),
            "source": "tensorcore_release_verification",
        },
        "status": "passed" if passed else "failed",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "checks": checks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=pathlib.Path,
        default=ROOT / ".scratch" / "release-verification-evidence.json",
    )
    parser.add_argument("--icc-bin")
    parser.add_argument(
        "--release-evidence", type=pathlib.Path,
        default=ROOT / ".scratch" / "release-evidence.json",
        help="Exact-head release smoke evidence used by runtime architecture invariants.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    icc = find_icc(args.icc_bin)
    clean = git_dirty() is False
    checks: dict[str, Any] = {
        "clean_head": {
            "passed": clean,
            "status": "passed" if clean else "failed",
        }
    }
    if not clean:
        print("[tensorcore/verification] clean_head FAILED", file=sys.stderr)

    check("documentation", [sys.executable, "scripts/check_docs_links.py"], checks)
    check("version_consistency", ["scripts/check_version_consistency.sh"], checks)
    check("source_privacy", [sys.executable, "scripts/check_release_privacy.py"], checks)
    check("history_privacy", [sys.executable, "scripts/check_release_privacy.py", "--history"], checks)
    check(
        "campaign",
        [sys.executable, "scripts/check_full_capability_campaign.py"],
        checks,
    )

    selftests = sorted((ROOT / "scripts").glob("*_selftest.py"))
    selftest_passed = True
    for path in selftests:
        passed, output = run([sys.executable, str(path.relative_to(ROOT))])
        if not passed:
            selftest_passed = False
            tail = "\n".join(output.splitlines()[-20:])
            print(f"[tensorcore/verification] selftest {path.name} FAILED\n{tail}", file=sys.stderr)
    checks["script_selftests"] = {
        "passed": selftest_passed,
        "status": "passed" if selftest_passed else "failed",
        "count": len(selftests),
    }
    print(
        f"[tensorcore/verification] script_selftests "
        f"{'passed' if selftest_passed else 'FAILED'} ({len(selftests)})"
    )

    check("adversarial_evals", [sys.executable, "evals/run_all.py"], checks)
    if icc is None:
        checks["icc_precommit"] = {"passed": False, "status": "failed", "reason": "icc_unavailable"}
        checks["architecture_model"] = {"passed": False, "status": "failed", "reason": "icc_unavailable"}
        print("[tensorcore/verification] ICC executable unavailable", file=sys.stderr)
    elif not args.release_evidence.is_file():
        checks["icc_precommit"] = {"passed": False, "status": "failed", "reason": "release_evidence_unavailable"}
        checks["architecture_model"] = {"passed": False, "status": "failed", "reason": "release_evidence_unavailable"}
        print("[tensorcore/verification] exact-head release evidence unavailable", file=sys.stderr)
    else:
        check(
            "icc_precommit",
            [
                icc, "pre-commit-check", "--repo", "tensorcore",
                "--architecture-model", ".icc/architecture-model.yaml",
                "--trace-file", str(args.release_evidence),
                "--format", "json",
            ],
            checks,
        )
        check(
            "architecture_model",
            [
                icc, "architecture-verify", "--repo", "tensorcore",
                "--model", ".icc/architecture-model.yaml",
                "--trace-file", str(args.release_evidence),
                "--format", "json",
            ],
            checks,
        )

    artifact = write_evidence(args.output, checks)
    privacy_ok, privacy_output = run(
        [sys.executable, "scripts/check_release_artifact_privacy.py", str(args.output)]
    )
    if not privacy_ok:
        print(privacy_output, file=sys.stderr)
        return 1
    if artifact["status"] != "passed":
        return 1
    print(f"[tensorcore/verification] release verification evidence OK ({len(checks)} gates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
