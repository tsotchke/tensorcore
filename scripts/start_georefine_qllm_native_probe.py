#!/usr/bin/env python3
"""Start the native qLLM GeoRefine coherence probe over SSH.

This is an additive launcher family. It does not replace the legacy
Qwen rank-probe starter and it never calls experiments.georefine.m2_compress.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import PurePosixPath


SCHEMA = "tensorcore.georefine_qllm_native_probe.start.v1"
UNTRUSTED_PATH_PREFIXES = ("/tmp", "/var/tmp", "/private/tmp")
UNTRUSTED_PATH_COMPONENTS = {"bytehole"}


def shq(value: str) -> str:
    return shlex.quote(value)


def is_protected_path(value: str) -> bool:
    if not value.startswith("/") or value == "/":
        return False
    path = str(PurePosixPath(value))
    if path in UNTRUSTED_PATH_PREFIXES or path.startswith(tuple(f"{root}/" for root in UNTRUSTED_PATH_PREFIXES)):
        return False
    return UNTRUSTED_PATH_COMPONENTS.isdisjoint(PurePosixPath(path).parts)


def render_remote_script(args: argparse.Namespace) -> str:
    preflight_only = "1" if args.preflight_only else "0"
    return f"""#!/bin/sh
set -eu

schema={shq(SCHEMA)}
resource={shq(args.resource)}
worker_resource={shq(args.worker_resource)}
authority_lease_id={shq(args.authority_lease_id)}
authority_owner={shq(args.authority_owner)}
repo_dir={shq(args.repo_dir)}
qllm_repo_dir={shq(args.qllm_repo_dir)}
python_bin={shq(args.python_bin)}
run_dir={shq(args.run_dir)}
evidence_root={shq(args.evidence_root)}
preflight_only={preflight_only}

case "$repo_dir" in
  "~") repo_dir="$HOME" ;;
  "~/"*) repo_dir="$HOME/${{repo_dir#\\~/}}" ;;
esac
case "$qllm_repo_dir" in
  "~") qllm_repo_dir="$HOME" ;;
  "~/"*) qllm_repo_dir="$HOME/${{qllm_repo_dir#\\~/}}" ;;
esac

json_escape() {{
  printf '%s' "$1" | sed 's/\\\\/\\\\\\\\/g; s/"/\\\\"/g'
}}

emit() {{
  ok="$1"
  reason="$2"
  printf '{{"ok":%s,"reason":"%s","repo_dir":"%s","qllm_repo_dir":"%s","resource":"%s","run_dir":"%s","schema":"%s"}}\\n' \\
    "$ok" "$(json_escape "$reason")" "$(json_escape "$repo_dir")" \\
    "$(json_escape "$qllm_repo_dir")" "$(json_escape "$resource")" \\
    "$(json_escape "$run_dir")" "$(json_escape "$schema")"
}}

if [ -z "$worker_resource" ]; then
  emit false worker_resource_missing
  exit 1
fi
if [ -z "$authority_lease_id" ]; then
  emit false authority_lease_id_missing
  exit 1
fi
if [ -z "$authority_owner" ]; then
  authority_owner=georefine:native-qllm-probe
fi
if [ ! -d "$qllm_repo_dir" ]; then
  emit false qllm_repo_dir_missing
  exit 1
fi
case "$python_bin" in
  "") ;;
  /*) ;;
  *) python_bin="$qllm_repo_dir/$python_bin" ;;
esac
if [ -z "$python_bin" ] || [ ! -x "$python_bin" ]; then
  if [ -x "$qllm_repo_dir/bin/python" ]; then
    python_bin="$qllm_repo_dir/bin/python"
  elif [ -x "$qllm_repo_dir/.venv/bin/python" ]; then
    python_bin="$qllm_repo_dir/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    python_bin=$(command -v python3)
  else
    emit false python_not_found
    exit 1
  fi
fi
probe_script="$qllm_repo_dir/scripts/georefine_native_coherence_probe.py"
checker_script="$qllm_repo_dir/scripts/check_georefine_native_coherence_artifact.py"
if [ ! -r "$probe_script" ]; then
  emit false qllm_native_probe_missing
  exit 1
fi
if [ ! -r "$checker_script" ]; then
  emit false qllm_native_checker_missing
  exit 1
fi

set -- "$python_bin" "$probe_script" \\
  --run-dir "$run_dir" \\
  --evidence-root "$evidence_root" \\
  --qllm-root "$qllm_repo_dir" \\
  --test-bin-dir "$qllm_repo_dir/build/bin" \\
  --resource "$resource" \\
  --worker-resource "$worker_resource" \\
  --authority-lease-id "$authority_lease_id" \\
  --authority-owner "$authority_owner" \\
  --json

if [ "$preflight_only" = "1" ]; then
  "$@" --preflight-only >/dev/null
  emit true preflight_ok
  exit 0
fi

"$@"
"$python_bin" "$checker_script" "$run_dir" --json >/dev/null
"""


def validate_args(args: argparse.Namespace) -> None:
    if not args.target:
        raise SystemExit("--target is required")
    if not args.worker_resource:
        raise SystemExit("--worker-resource is required")
    if not args.authority_lease_id:
        raise SystemExit("--authority-lease-id is required")
    for label, value in (
        ("repo-dir", args.repo_dir),
        ("qllm-repo-dir", args.qllm_repo_dir),
        ("run-dir", args.run_dir),
        ("evidence-root", args.evidence_root),
    ):
        if not is_protected_path(value):
            raise SystemExit(f"--{label} must be an absolute protected path")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--resource", required=True)
    parser.add_argument("--worker-resource", default="")
    parser.add_argument("--authority-lease-id", default="")
    parser.add_argument("--authority-owner", default="")
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--qllm-repo-dir", required=True)
    parser.add_argument("--python-bin", default="")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--compression-ratio", default="0.70")
    parser.add_argument("--max-size-ratio", default="0.30")
    parser.add_argument("--target-kl", default="0.80")
    parser.add_argument("--target-kl-kd-steps", default="0")
    parser.add_argument("--quality-floor", default="0.05")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--print-script", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    script = render_remote_script(args)
    if args.print_script:
        payload = {"schema": SCHEMA, "ok": True, "script": script}
        print(json.dumps(payload, sort_keys=True) if args.json else script)
        return 0
    proc = subprocess.run(
        ["ssh", args.target, script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
