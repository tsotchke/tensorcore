#!/usr/bin/env python3
"""Check noninteractive git access from a mesh node."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from typing import Any


SCHEMA = "tensorcore.mesh_git_access.v1"


def render_remote_command(repo_url: str, ref: str) -> str:
    parts = ["GIT_TERMINAL_PROMPT=0", "git", "ls-remote", "--exit-code", shlex.quote(repo_url)]
    if ref:
        parts.append(shlex.quote(ref))
    return " ".join(parts)


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    command = render_remote_command(args.repo_url, args.ref)
    try:
        proc = subprocess.run(
            ["ssh", args.target, command],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        payload = {
            "schema": SCHEMA,
            "ok": False,
            "reason": "git_access_timeout",
            "target": args.target,
            "repo_url": args.repo_url,
            "ref": args.ref,
        }
        if args.resource:
            payload["resource"] = args.resource
        return payload
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": proc.returncode == 0,
        "reason": "ok" if proc.returncode == 0 else "git_access_failed",
        "target": args.target,
        "repo_url": args.repo_url,
        "ref": args.ref,
        "rc": proc.returncode,
        "checked_at_unix": int(time.time()),
        "stdout_tail": proc.stdout.strip()[-1000:],
        "stderr_tail": proc.stderr.strip()[-1000:],
    }
    if args.resource:
        payload["resource"] = args.resource
    if proc.returncode == 0:
        line = next((item for item in proc.stdout.splitlines() if item.strip()), "")
        fields = line.split()
        if len(fields) >= 2:
            payload["head"] = fields[0]
            payload["matched_ref"] = fields[1]
        else:
            payload["ok"] = False
            payload["reason"] = "invalid_ls_remote_output"
    elif "Permission denied (publickey)" in proc.stderr:
        payload["reason"] = "git_publickey_denied"
    elif "could not read Username" in proc.stderr or "terminal prompts disabled" in proc.stderr:
        payload["reason"] = "git_credentials_required"
    return payload


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--resource", default="")
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = run_check(args)
    if args.json:
        json.dump(payload, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"{args.target}: ok={payload.get('ok')} reason={payload.get('reason')}")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
