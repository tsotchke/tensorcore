#!/usr/bin/env python3
"""Emit Windows worker identity for a mesh-scheduled CUDA job."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from typing import Any


POWERSHELL = r"""
$ErrorActionPreference = 'Continue'
$pattern = '__MATCH_REGEX__'
function ToIntOrNull($value) {
  $text = [string]$value
  $out = 0
  if ([int]::TryParse(($text -replace '[^0-9-]', ''), [ref]$out)) { return $out }
  return $null
}
function ParseApps($rows) {
  $apps = @()
  foreach ($line in $rows) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $parts = $line.Split(',', 3)
    if ($parts.Count -lt 3) { continue }
    $apps += [ordered]@{
      pid = ToIntOrNull $parts[0].Trim()
      process_name = $parts[1].Trim()
      used_memory_mib = ToIntOrNull $parts[2].Trim()
      raw = $line.Trim()
    }
  }
  return $apps
}
$matched = @()
$procs = @(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine,ExecutablePath)
foreach ($proc in $procs) {
  $cmd = [string]$proc.CommandLine
  if (-not [string]::IsNullOrWhiteSpace($pattern) -and $cmd -match $pattern) {
    $matched += [ordered]@{
      pid = [int]$proc.ProcessId
      ppid = [int]$proc.ParentProcessId
      name = [string]$proc.Name
      executable = [string]$proc.ExecutablePath
      args = $cmd
    }
  }
}
$apps = @()
$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvsmi) {
  $rows = @(& $nvsmi.Source '--query-compute-apps=pid,process_name,used_gpu_memory' '--format=csv,noheader,nounits' 2>&1)
  if ($LASTEXITCODE -eq 0) { $apps = @(ParseApps $rows) }
}
[ordered]@{
  computer_name = $env:COMPUTERNAME
  user = $env:USERNAME
  matched_processes = $matched
  cuda = [ordered]@{
    ok = ($null -ne $nvsmi)
    apps = $apps
  }
} | ConvertTo-Json -Depth 8 -Compress
"""


def render_script(pattern: str) -> str:
    return POWERSHELL.replace("__MATCH_REGEX__", pattern.replace("'", "''"))


def ps_encode(command: str) -> str:
    return base64.b64encode(command.encode("utf-16le")).decode("ascii")


def run_remote_powershell(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
    script_name = f"tensorcore-windows-worker-identity-{os.getpid()}.ps1"
    upload_command = (
        "$ProgressPreference = 'SilentlyContinue'; "
        f"$Path = Join-Path $env:TEMP '{script_name}'; "
        "Set-Content -LiteralPath $Path -Encoding UTF8 -Value ([Console]::In.ReadToEnd())"
    )
    run_command = (
        "$ProgressPreference = 'SilentlyContinue'; "
        f"$Path = Join-Path $env:TEMP '{script_name}'; & $Path"
    )
    cleanup_command = (
        "$ProgressPreference = 'SilentlyContinue'; "
        f"$Path = Join-Path $env:TEMP '{script_name}'; "
        "Remove-Item -Force -LiteralPath $Path -ErrorAction SilentlyContinue"
    )
    common = ["ssh", target, "powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass"]
    upload = subprocess.run(
        common + ["-EncodedCommand", ps_encode(upload_command)],
        input=script,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if upload.returncode != 0:
        return upload
    try:
        return subprocess.run(
            common + ["-EncodedCommand", ps_encode(run_command)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    finally:
        subprocess.run(
            common + ["-EncodedCommand", ps_encode(cleanup_command)],
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=max(1.0, min(timeout, 5.0)),
            check=False,
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--resource", required=True)
    parser.add_argument("--match-regex", required=True)
    parser.add_argument("--require-matching-process", action="store_true")
    parser.add_argument("--require-matched-cuda", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    proc = run_remote_powershell(
        args.target,
        render_script(args.match_regex),
        timeout=args.timeout_sec,
    )
    if proc.returncode != 0:
        return {
            "schema": "tensorcore.mesh_worker_identity.v1",
            "ok": False,
            "reason": "ssh_or_powershell_failed",
            "resource": args.resource,
            "worker_host": args.target,
            "stderr_tail": proc.stderr.strip()[-1000:],
            "stdout_tail": proc.stdout.strip()[-1000:],
        }
    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "schema": "tensorcore.mesh_worker_identity.v1",
            "ok": False,
            "reason": "invalid_probe_json",
            "resource": args.resource,
            "worker_host": args.target,
            "stderr_tail": proc.stderr.strip()[-1000:],
            "stdout_tail": proc.stdout.strip()[-1000:],
        }
    matched = raw.get("matched_processes") if isinstance(raw, dict) else []
    if not isinstance(matched, list):
        matched = []
    apps = ((raw.get("cuda") or {}).get("apps") if isinstance(raw, dict) else []) or []
    matched_pids = []
    for row in matched:
        try:
            matched_pids.append(int(row["pid"]))
        except Exception:
            pass
    cuda_pids = []
    matched_cuda_pids = []
    ignored_opaque_wddm = []
    for app in apps:
        if (
            app.get("process_name") == "[Insufficient Permissions]"
            and app.get("used_memory_mib") is None
        ):
            ignored_opaque_wddm.append(app)
            continue
        try:
            pid = int(app["pid"])
        except Exception:
            continue
        cuda_pids.append(pid)
        if pid in matched_pids:
            matched_cuda_pids.append(pid)
    ok = True
    reasons = []
    if args.require_matching_process and not matched_pids:
        ok = False
        reasons.append("no_matching_process")
    if args.require_matched_cuda and not matched_cuda_pids:
        ok = False
        reasons.append("no_matched_cuda_process")
    return {
        "schema": "tensorcore.mesh_worker_identity.v1",
        "ok": ok,
        "reason": "ok" if ok else ",".join(reasons),
        "resource": args.resource,
        "worker_host": raw.get("computer_name") or args.target if isinstance(raw, dict) else args.target,
        "worker_pid": matched_pids[0] if matched_pids else None,
        "worker_pids": matched_pids,
        "matched_processes": matched,
        "cuda_pids": cuda_pids,
        "matched_cuda_pids": matched_cuda_pids,
        "ignored_opaque_wddm": ignored_opaque_wddm,
        "cuda": raw.get("cuda") if isinstance(raw, dict) else {},
        "worker_systemd_unit": None,
        "worker_cgroup": None,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = run_probe(args)
    if args.json:
        json.dump(payload, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"{args.resource}: ok={payload.get('ok')} reason={payload.get('reason')}")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
