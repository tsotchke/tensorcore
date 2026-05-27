#!/usr/bin/env python3
"""Fetch and validate self-hosted M5 TensorOps hardware evidence.

This is the consumer-side companion to .github/workflows/hardware-evidence.yml.
It does not run the hardware smoke locally; it downloads the workflow artifact
from a self-hosted M5 run and validates that the release-smoke evidence belongs
to the expected clean git head and passed the Metal 4 TensorOps runtime gate.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = "hardware-evidence.yml"
ARTIFACT = "tensorcore-hardware-evidence"


def run(
    cmd: list[str],
    *,
    cwd: pathlib.Path = ROOT,
    capture: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if check and proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        raise SystemExit(proc.returncode)
    return proc


def git_output(*args: str) -> str:
    return run(["git", *args]).stdout.strip()


def default_repo() -> str:
    try:
        value = git_output("config", "--get", "remote.origin.url")
    except SystemExit:
        return ""
    if value.startswith("git@github.com:"):
        value = value.removeprefix("git@github.com:").removesuffix(".git")
    elif value.startswith("https://github.com/"):
        value = value.removeprefix("https://github.com/").removesuffix(".git")
    return value


def load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"could not read JSON at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"JSON root at {path} must be an object")
    return data


def latest_run_id(repo: str, expected_head: str, limit: int) -> str:
    proc = run(
        [
            "gh",
            "run",
            "list",
            "--repo",
            repo,
            "--workflow",
            WORKFLOW,
            "--json",
            "databaseId,headSha,status,conclusion,event,createdAt",
            "--limit",
            str(limit),
        ]
    )
    try:
        runs = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"could not parse gh run list JSON: {exc}") from exc
    if not isinstance(runs, list):
        raise SystemExit("gh run list did not return a JSON list")
    for item in runs:
        if not isinstance(item, dict):
            continue
        if item.get("headSha") != expected_head:
            continue
        if item.get("status") != "completed" or item.get("conclusion") != "success":
            continue
        run_id = item.get("databaseId")
        if run_id:
            return str(run_id)
    raise SystemExit(
        f"no successful {WORKFLOW} run found for head {expected_head}; "
        "pass --run-id after dispatching the workflow if the run is still fresh"
    )


def dispatch(repo: str, ref: str) -> None:
    run(
        [
            "gh",
            "workflow",
            "run",
            WORKFLOW,
            "--repo",
            repo,
            "--ref",
            ref,
            "--field",
            "require_metal4_tensorops=true",
        ],
        capture=False,
    )
    print(
        "dispatched Hardware Evidence with require_metal4_tensorops=true; "
        "rerun this script with --latest-for-head or --run-id after it completes"
    )


def download_artifact(repo: str, run_id: str, output_dir: pathlib.Path, keep: bool) -> pathlib.Path:
    if output_dir.exists() and not keep:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "gh",
            "run",
            "download",
            run_id,
            "--repo",
            repo,
            "--name",
            ARTIFACT,
            "--dir",
            str(output_dir),
        ],
        capture=False,
    )
    evidence = output_dir / "release_smoke_runtime_evidence.json"
    if not evidence.exists():
        matches = sorted(output_dir.rglob("release_smoke_runtime_evidence.json"))
        if matches:
            evidence = matches[0]
        else:
            raise SystemExit(
                f"artifact {ARTIFACT!r} from run {run_id} did not contain "
                "release_smoke_runtime_evidence.json"
            )
    return evidence


def validate(evidence: pathlib.Path, expected_head: str) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "check_release_evidence.py"),
        str(evidence),
        "--require-gpu",
        "--require-metal4-tensorops",
        "--git-head",
        expected_head,
        "--require-clean-head",
    ]
    run(cmd, capture=False)
    data = load_json(evidence)
    metal4 = data.get("checks", {}).get("metal4_tensorops", {})
    print(
        "M5 TensorOps hardware evidence accepted: "
        f"head={data.get('meta', {}).get('git_head')} "
        f"compile={metal4.get('compile_status')} "
        f"runtime={metal4.get('runtime_status')} "
        f"artifact={evidence}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=default_repo(), help="GitHub repo as owner/name.")
    parser.add_argument("--expected-head", default=git_output("rev-parse", "HEAD"))
    parser.add_argument("--ref", default=None, help="Ref to dispatch; defaults to expected head.")
    parser.add_argument("--run-id", help="Existing hardware-evidence workflow run id to fetch.")
    parser.add_argument(
        "--latest-for-head",
        action="store_true",
        help="Find the latest successful hardware-evidence run for --expected-head.",
    )
    parser.add_argument(
        "--dispatch",
        action="store_true",
        help="Dispatch hardware-evidence.yml with require_metal4_tensorops=true and exit.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=ROOT / "build" / "m5-tensorops-hardware-evidence",
    )
    parser.add_argument("--keep-output-dir", action="store_true")
    parser.add_argument("--run-list-limit", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.repo:
        raise SystemExit("--repo is required when remote.origin.url is not a GitHub repo")
    if args.dispatch:
        dispatch(args.repo, args.ref or args.expected_head)
        return 0
    run_id = args.run_id
    if not run_id:
        if not args.latest_for_head:
            raise SystemExit("pass --run-id, --latest-for-head, or --dispatch")
        run_id = latest_run_id(args.repo, args.expected_head, args.run_list_limit)
    evidence = download_artifact(args.repo, run_id, args.output_dir, args.keep_output_dir)
    validate(evidence, args.expected_head)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
