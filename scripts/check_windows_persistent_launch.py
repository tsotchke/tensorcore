#!/usr/bin/env python3
"""Check whether a Windows host can create a persistent scheduler launch task."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from typing import Any


SCHEMA = "tensorcore.windows_persistent_launch.v1"


def ps_encode(command: str) -> str:
    return base64.b64encode(command.encode("utf-16le")).decode("ascii")


def ps_literal(value: str) -> str:
    return value.replace("'", "''")


def render_probe(resource: str, token: str) -> str:
    task_name = f"TensorcorePersistentLaunchPreflight_{token}"
    return f"""
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
$Resource = '{ps_literal(resource)}'
$TaskName = '{ps_literal(task_name)}'
$ArtifactDir = Join-Path $env:LOCALAPPDATA 'tensorcore'
New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null
$MarkerPath = Join-Path $ArtifactDir ($TaskName + '.marker')
Remove-Item -Force -LiteralPath $MarkerPath -ErrorAction SilentlyContinue
$TaskCommand = 'cmd.exe /c echo ok > "' + $MarkerPath + '"'
$startTime = (Get-Date).AddMinutes(1).ToString('HH:mm')
& schtasks.exe /Delete /F /TN $TaskName *> $null
$createOutput = @(& schtasks.exe /Create /F /TN $TaskName /SC ONCE /ST $startTime /TR $TaskCommand 2>&1)
$createRc = $LASTEXITCODE
$runOutput = @()
$runRc = -1
$markerExists = $false
if ($createRc -eq 0) {{
  $runOutput = @(& schtasks.exe /Run /TN $TaskName 2>&1)
  $runRc = $LASTEXITCODE
  $deadline = (Get-Date).AddSeconds(8)
  while ((Get-Date) -lt $deadline) {{
    if (Test-Path -LiteralPath $MarkerPath) {{
      $markerExists = $true
      break
    }}
    Start-Sleep -Milliseconds 250
  }}
  & schtasks.exe /Delete /F /TN $TaskName *> $null
}}
Remove-Item -Force -LiteralPath $MarkerPath -ErrorAction SilentlyContinue
$tail = ($createOutput -join [Environment]::NewLine)
if ($tail.Length -gt 1000) {{ $tail = $tail.Substring($tail.Length - 1000) }}
$runTail = ($runOutput -join [Environment]::NewLine)
if ($runTail.Length -gt 1000) {{ $runTail = $runTail.Substring($runTail.Length - 1000) }}
$reason = if ($createRc -ne 0) {{
  'scheduled_task_create_failed'
}} elseif ($runRc -ne 0) {{
  'scheduled_task_run_failed'
}} elseif (-not $markerExists) {{
  'scheduled_task_did_not_run'
}} else {{
  'ok'
}}
[ordered]@{{
  schema = '{SCHEMA}'
  ok = ($reason -eq 'ok')
  reason = $reason
  resource = $Resource
  computer_name = $env:COMPUTERNAME
  user = $env:USERNAME
  task_name = $TaskName
  marker_path = $MarkerPath
  create_rc = $createRc
  create_tail = $tail
  run_rc = $runRc
  run_tail = $runTail
  marker_written = $markerExists
  checked_at_unix = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
}} | ConvertTo-Json -Depth 6 -Compress
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


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    token = f"{int(time.time())}_{os.getpid()}"
    try:
        proc = run_remote_powershell(args.target, render_probe(args.resource, token), timeout=args.timeout_sec)
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
    if payload.get("schema") != SCHEMA:
        return {"schema": SCHEMA, "ok": False, "reason": "invalid_probe_schema", "resource": args.resource}
    if payload.get("resource") != args.resource:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "probe_resource_mismatch",
            "resource": args.resource,
            "probe_resource": payload.get("resource"),
        }
    return payload


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--resource", required=True)
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
