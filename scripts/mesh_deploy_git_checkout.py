#!/usr/bin/env python3
"""Clone or fast-forward a git checkout on a mesh node over SSH."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
import sys
import tempfile
import uuid
from typing import Any


SCHEMA = "tensorcore.mesh_git_deploy.v1"


def shq(value: str) -> str:
    return shlex.quote(value)


def render_remote_script(args: argparse.Namespace) -> str:
    require_clean = "1" if args.require_clean else "0"
    return f"""#!/bin/sh
set -eu
repo_url={shq(args.repo_url)}
repo_dir={shq(args.repo_dir)}
ref={shq(args.ref)}
require_clean={require_clean}

case "$repo_dir" in
  "~") repo_dir="$HOME" ;;
  "~/"*) repo_dir="$HOME/${{repo_dir#\\~/}}" ;;
esac

json_escape() {{
  printf '%s' "$1" | sed 's/\\\\/\\\\\\\\/g; s/"/\\\\"/g'
}}

emit() {{
  ok="$1"
  reason="$2"
  head="${{3:-}}"
  branch="${{4:-}}"
  if [ -z "$branch" ]; then
    branch=detached
  fi
  printf '{{"branch":"%s","head":"%s","ok":%s,"reason":"%s","ref":"%s","repo_dir":"%s","repo_url":"%s","schema":"{SCHEMA}"}}\\n' \
    "$(json_escape "$branch")" \
    "$(json_escape "$head")" \
    "$ok" \
    "$(json_escape "$reason")" \
    "$(json_escape "$ref")" \
    "$(json_escape "$repo_dir")" \
    "$(json_escape "$repo_url")"
}}

if [ ! -d "$repo_dir/.git" ]; then
  parent=$(dirname "$repo_dir")
  mkdir -p "$parent"
  if ! git clone --filter=blob:none --branch "$ref" "$repo_url" "$repo_dir"; then
    emit false clone_failed
    exit 1
  fi
else
  current_url=$(git -C "$repo_dir" remote get-url origin 2>/dev/null || true)
  if [ "$current_url" != "$repo_url" ]; then
    git -C "$repo_dir" remote set-url origin "$repo_url"
  fi
  git -C "$repo_dir" fetch origin "$ref" || git -C "$repo_dir" fetch origin
  git -C "$repo_dir" checkout "$ref"
  if git -C "$repo_dir" symbolic-ref -q HEAD >/dev/null 2>&1; then
    git -C "$repo_dir" pull --ff-only origin "$ref"
  fi
fi

head=$(git -C "$repo_dir" rev-parse HEAD)
branch=$(git -C "$repo_dir" branch --show-current 2>/dev/null || true)
dirty=$(git -C "$repo_dir" status --porcelain)
if [ "$require_clean" = "1" ] && [ -n "$dirty" ]; then
  printf '%s\\n' "$dirty" >&2
  emit false dirty_checkout "$head" "$branch"
  exit 3
fi

emit true ok "$head" "$branch"
"""


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="SSH target host")
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--ref", default="master")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--print-script", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def run_deploy(args: argparse.Namespace) -> dict[str, Any]:
    script = render_remote_script(args)
    if args.print_script:
        return {
            "schema": SCHEMA,
            "ok": True,
            "target": args.target,
            "script": script,
        }
    try:
        proc = run_remote_script(args.target, script, timeout=args.timeout_sec)
    except subprocess.TimeoutExpired:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "deploy_timeout",
            "target": args.target,
        }
    if proc.returncode != 0:
        payload = parse_remote_payload(proc.stdout)
        if payload is not None:
            invalid = validate_remote_payload(payload, args)
            if invalid is not None:
                invalid["rc"] = proc.returncode
                invalid["stderr_tail"] = proc.stderr.strip()[-1000:]
                invalid["stdout_tail"] = proc.stdout.strip()[-1000:]
                return invalid
            payload.setdefault("ok", False)
            payload.setdefault("target", args.target)
            payload["reason"] = classify_remote_failure(str(payload.get("reason") or ""), proc.stderr)
            payload["rc"] = proc.returncode
            payload["stderr_tail"] = proc.stderr.strip()[-1000:]
            payload["stdout_tail"] = proc.stdout.strip()[-1000:]
            return payload
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": classify_remote_failure("", proc.stderr),
            "target": args.target,
            "rc": proc.returncode,
            "stdout_tail": proc.stdout.strip()[-1000:],
            "stderr_tail": proc.stderr.strip()[-1000:],
        }
    payload = parse_remote_payload(proc.stdout)
    if payload is None:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "invalid_deploy_json",
            "target": args.target,
            "stdout_tail": proc.stdout.strip()[-1000:],
            "stderr_tail": proc.stderr.strip()[-1000:],
        }
    invalid = validate_remote_payload(payload, args)
    if invalid is not None:
        invalid["stdout_tail"] = proc.stdout.strip()[-1000:]
        invalid["stderr_tail"] = proc.stderr.strip()[-1000:]
        return invalid
    payload.setdefault("target", args.target)
    return payload


def validate_remote_payload(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    if payload.get("schema") != SCHEMA:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "invalid_deploy_schema",
            "target": args.target,
            "deploy_schema": payload.get("schema"),
        }
    if payload.get("repo_url") != args.repo_url:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "deploy_repo_url_mismatch",
            "target": args.target,
            "repo_url": args.repo_url,
            "deploy_repo_url": payload.get("repo_url"),
        }
    return None


def parse_remote_payload(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def classify_remote_failure(reason: str, stderr: str) -> str:
    if "Permission denied (publickey)" in stderr:
        return "git_publickey_denied"
    if "could not read Username" in stderr or "terminal prompts disabled" in stderr:
        return "git_credentials_required"
    if reason == "clone_failed":
        return "clone_failed"
    return reason or "deploy_failed"


def run_remote_script(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
    local_name = ""
    remote_path = f"/tmp/tensorcore-mesh-deploy-{os.getpid()}-{uuid.uuid4().hex}.sh"
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(script)
            local_name = handle.name
        upload = subprocess.run(
            ["scp", "-q", local_name, f"{target}:{remote_path}"],
            stdin=subprocess.DEVNULL,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if upload.returncode != 0:
            return run_remote_inline_script(target, script, timeout=timeout)
        try:
            return subprocess.run(
                ["ssh", target, "sh", remote_path],
                stdin=subprocess.DEVNULL,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        finally:
            try:
                subprocess.run(
                    ["ssh", target, "rm", "-f", remote_path],
                    stdin=subprocess.DEVNULL,
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=max(1.0, min(timeout, 10.0)),
                    check=False,
                )
            except subprocess.TimeoutExpired:
                pass
    finally:
        if local_name:
            try:
                os.unlink(local_name)
            except OSError:
                pass


def run_remote_inline_script(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
    remote_path = f"/tmp/tensorcore-mesh-deploy-{os.getpid()}-{uuid.uuid4().hex}.sh"
    b64_path = f"{remote_path}.b64"
    quoted_path = shlex.quote(remote_path)
    quoted_b64_path = shlex.quote(b64_path)
    script_b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    init = subprocess.run(
        ["ssh", target, f"rm -f {quoted_path} {quoted_b64_path}; umask 077; : > {quoted_b64_path}"],
        stdin=subprocess.DEVNULL,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if init.returncode != 0:
        return init
    for offset in range(0, len(script_b64), 3000):
        chunk = script_b64[offset:offset + 3000]
        append = subprocess.run(
            ["ssh", target, f"printf '%s' {shlex.quote(chunk)} >> {quoted_b64_path}"],
            stdin=subprocess.DEVNULL,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if append.returncode != 0:
            return append
    remote_command = (
        f"(base64 -d {quoted_b64_path} > {quoted_path} 2>/dev/null || "
        f"base64 --decode {quoted_b64_path} > {quoted_path}) && sh {quoted_path}; "
        f"rc=$?; rm -f {quoted_path} {quoted_b64_path}; exit $rc"
    )
    return subprocess.run(
        ["ssh", target, remote_command],
        stdin=subprocess.DEVNULL,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = run_deploy(args)
    if args.print_script and not args.json:
        print(payload["script"], end="")
    elif args.json:
        json.dump(payload, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        if payload.get("ok"):
            print(f"{args.target}: {payload.get('repo_dir')} at {payload.get('head')}")
        else:
            print(f"{args.target}: deploy failed: {payload.get('reason')}", file=sys.stderr)
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
