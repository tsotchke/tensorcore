#!/usr/bin/env python3
"""Check a Windows CUDA smoke artifact over SSH."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from typing import Any


SCHEMA = "tensorcore.windows_cuda_smoke.check.v1"


def ps_encode(command: str) -> str:
    return base64.b64encode(command.encode("utf-16le")).decode("ascii")


def ps_literal(value: str) -> str:
    return value.replace("'", "''")


def default_artifact_path(resource: str) -> str:
    safe = resource.replace(":", "-").replace("/", "-").replace("\\", "-")
    return f"%LOCALAPPDATA%\\tensorcore\\{safe}-smoke.json"


def artifact_path_assignment(resource: str, artifact_path: str) -> str:
    if artifact_path:
        return f"$ArtifactPath = [Environment]::ExpandEnvironmentVariables('{ps_literal(artifact_path)}')"
    safe = resource.replace(":", "-").replace("/", "-").replace("\\", "-")
    return (
        "$ArtifactDir = Join-Path $env:LOCALAPPDATA 'tensorcore'\n"
        f"$ArtifactPath = Join-Path $ArtifactDir '{safe}-smoke.json'"
    )


def render_probe(resource: str, artifact_path: str) -> str:
    return f"""
$ErrorActionPreference = 'Continue'
{artifact_path_assignment(resource, artifact_path)}
$exists = Test-Path -LiteralPath $ArtifactPath
$artifact = $null
$processAlive = $false
$processProbeReason = 'artifact_heartbeat_only'
$reason = 'missing_artifact'
if ($exists) {{
  try {{
    $artifact = Get-Content -Raw -LiteralPath $ArtifactPath | ConvertFrom-Json
    $reason = 'ok'
  }} catch {{
    $reason = 'invalid_artifact_json'
  }}
}}
[ordered]@{{
  schema = '{SCHEMA}'
  ok = $exists -and $null -ne $artifact
  reason = $reason
  resource = '{ps_literal(resource)}'
  artifact_path = $ArtifactPath
  process_alive = $processAlive
  process_probe_reason = $processProbeReason
  checked_at_unix = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
  artifact = $artifact
}} | ConvertTo-Json -Depth 10 -Compress
"""


def run_remote_powershell(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", target, "powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", ps_encode(script)],
        stdin=subprocess.DEVNULL,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def evaluate(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if payload.get("schema") != SCHEMA:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "invalid_probe_schema",
            "resource": args.resource,
            "probe_schema": payload.get("schema"),
        }
    if payload.get("resource") != args.resource:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "probe_resource_mismatch",
            "resource": args.resource,
            "probe_resource": payload.get("resource"),
        }
    artifact = payload.get("artifact") if isinstance(payload.get("artifact"), dict) else {}
    if artifact and artifact.get("schema") != "tensorcore.windows_cuda_smoke.v1":
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "invalid_artifact_schema",
            "resource": args.resource,
            "artifact_schema": artifact.get("schema"),
        }
    if artifact and artifact.get("resource") != args.resource:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "artifact_resource_mismatch",
            "resource": args.resource,
            "artifact_resource": artifact.get("resource"),
        }
    state = artifact.get("state")
    ok = payload.get("ok") is True
    reason = str(payload.get("reason") or "unknown")
    checked_now = payload.get("checked_at_unix") or 0
    heartbeat = artifact.get("heartbeat_unix") or artifact.get("checked_at_unix")
    live_age = None
    if heartbeat is not None:
        try:
            live_age = float(checked_now) - float(heartbeat)
        except (TypeError, ValueError):
            live_age = None
    if live_age is not None:
        payload["live_artifact_age_sec"] = live_age
    if args.max_age_sec > 0 and artifact:
        checked = artifact.get("checked_at_unix") or artifact.get("completed_at_unix")
        try:
            age = float(checked_now) - float(checked)
        except (TypeError, ValueError):
            age = args.max_age_sec + 1
        payload["artifact_age_sec"] = age
        if age > args.max_age_sec:
            payload["ok"] = False
            payload["reason"] = "artifact_stale"
            return payload
    if args.require_live:
        if state != "running":
            ok = False
            reason = f"not_live:{state}"
        elif payload.get("process_alive") is True:
            ok = True
        elif live_age is not None and 0 <= live_age <= args.live_max_age_sec:
            ok = True
            payload["live_artifact_fresh"] = True
        elif live_age is not None:
            ok = False
            reason = "live_artifact_stale"
        else:
            ok = False
            reason = "live_process_not_found"
    if args.require_live_or_complete:
        if state == "completed":
            if (
                artifact.get("ok") is not True
                or artifact.get("build_ok") is not True
                or artifact.get("runtime_ok") is not True
            ):
                ok = False
                reason = "complete_artifact_not_ok"
            else:
                ok = True
        elif state == "running":
            if payload.get("process_alive") is True:
                ok = True
            elif live_age is not None and 0 <= live_age <= args.live_max_age_sec:
                ok = True
                payload["live_artifact_fresh"] = True
            elif live_age is not None:
                ok = False
                reason = "live_artifact_stale"
            else:
                ok = False
                reason = "live_process_not_found"
        else:
            ok = False
            reason = f"not_live_or_complete:{state}"
    if args.require_complete:
        if state != "completed":
            ok = False
            reason = f"not_complete:{state}"
        elif artifact.get("ok") is not True or artifact.get("build_ok") is not True or artifact.get("runtime_ok") is not True:
            ok = False
            reason = "complete_artifact_not_ok"
    payload["ok"] = ok
    payload["reason"] = "ok" if ok else reason
    return payload


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    try:
        proc = run_remote_powershell(
            args.target,
            render_probe(args.resource, args.artifact_path),
            timeout=args.timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return {"schema": SCHEMA, "ok": False, "reason": "probe_timeout", "resource": args.resource}
    if proc.returncode != 0:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "ssh_or_powershell_failed",
            "resource": args.resource,
            "rc": proc.returncode,
            "stdout_tail": proc.stdout.strip()[-1000:],
            "stderr_tail": proc.stderr.strip()[-1000:],
        }
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "invalid_probe_json",
            "resource": args.resource,
            "stdout_tail": proc.stdout.strip()[-1000:],
            "stderr_tail": proc.stderr.strip()[-1000:],
        }
    if not isinstance(payload, dict):
        return {"schema": SCHEMA, "ok": False, "reason": "non_object_probe_json", "resource": args.resource}
    return evaluate(payload, args)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--resource", required=True)
    parser.add_argument("--artifact-path", default="")
    parser.add_argument("--require-live", action="store_true")
    parser.add_argument("--require-live-or-complete", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--live-max-age-sec", type=float, default=15.0)
    parser.add_argument("--max-age-sec", type=float, default=0.0)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = run_check(args)
    if args.json:
        json.dump(payload, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"{args.resource}: ok={payload.get('ok')} reason={payload.get('reason')}")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
