#!/usr/bin/env python3
"""Emit Windows worker identity for a mesh-scheduled CUDA job."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
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


ARTIFACT_POWERSHELL = r"""
$ErrorActionPreference = 'Continue'
$ArtifactPath = [Environment]::ExpandEnvironmentVariables('__ARTIFACT_PATH__')
if ([string]::IsNullOrWhiteSpace($ArtifactPath)) {
  $ArtifactDir = Join-Path $env:LOCALAPPDATA 'tensorcore'
  $ArtifactPath = Join-Path $ArtifactDir '__DEFAULT_ARTIFACT_NAME__'
}
$artifact = $null
$reason = 'missing_artifact'
if (Test-Path -LiteralPath $ArtifactPath) {
  try {
    $artifact = Get-Content -Raw -LiteralPath $ArtifactPath | ConvertFrom-Json
    $reason = 'ok'
  } catch {
    $reason = 'invalid_artifact_json'
  }
}
[ordered]@{
  computer_name = $env:COMPUTERNAME
  user = $env:USERNAME
  artifact_path = $ArtifactPath
  reason = $reason
  artifact = $artifact
} | ConvertTo-Json -Depth 10 -Compress
"""


def render_script(pattern: str) -> str:
    return POWERSHELL.replace("__MATCH_REGEX__", pattern.replace("'", "''"))


def default_artifact_name(resource: str) -> str:
    safe = resource.replace(":", "-").replace("/", "-").replace("\\", "-")
    return f"{safe}-smoke.json"


def render_artifact_script(resource: str, artifact_path: str) -> str:
    return (
        ARTIFACT_POWERSHELL
        .replace("__ARTIFACT_PATH__", artifact_path.replace("'", "''"))
        .replace("__DEFAULT_ARTIFACT_NAME__", default_artifact_name(resource).replace("'", "''"))
    )


def ps_encode(command: str) -> str:
    return base64.b64encode(command.encode("utf-16le")).decode("ascii")


def run_remote_powershell(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
    script_name = f"tensorcore-windows-worker-identity-{os.getpid()}.ps1"
    cleanup_command = (
        "$ProgressPreference = 'SilentlyContinue'; "
        f"Remove-Item -Force -LiteralPath '{script_name}' -ErrorAction SilentlyContinue"
    )
    common = ["ssh", target, "powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass"]
    local_path = ""
    tmp_dir = "/private/tmp" if os.path.isdir("/private/tmp") and os.access("/private/tmp", os.W_OK) else None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", prefix=script_name + ".", suffix=".tmp", dir=tmp_dir, delete=False
        ) as handle:
            handle.write(script)
            local_path = handle.name
        upload = subprocess.run(
            ["scp", "-q", local_path, f"{target}:{script_name}"],
            stdin=subprocess.DEVNULL,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if upload.returncode != 0:
            return upload
        return subprocess.run(
            common + ["-File", script_name],
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
                common + ["-EncodedCommand", ps_encode(cleanup_command)],
                stdin=subprocess.DEVNULL,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=max(1.0, min(timeout, 5.0)),
                check=False,
            )
        except subprocess.TimeoutExpired:
            pass
        if local_path:
            try:
                os.unlink(local_path)
            except FileNotFoundError:
                pass


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--resource", required=True)
    parser.add_argument("--match-regex", default="")
    parser.add_argument("--artifact-path", default="")
    parser.add_argument("--allow-completed-artifact", action="store_true")
    parser.add_argument("--require-matching-process", action="store_true")
    parser.add_argument("--require-matched-cuda", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def run_artifact_probe(args: argparse.Namespace) -> dict[str, Any]:
    try:
        proc = run_remote_powershell(
            args.target,
            render_artifact_script(args.resource, args.artifact_path),
            timeout=args.timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return {
            "schema": "tensorcore.mesh_worker_identity.v1",
            "ok": False,
            "reason": "probe_timeout",
            "resource": args.resource,
            "worker_host": args.target,
        }
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
    if not isinstance(raw, dict):
        return {
            "schema": "tensorcore.mesh_worker_identity.v1",
            "ok": False,
            "reason": "invalid_probe_payload",
            "resource": args.resource,
            "worker_host": args.target,
            "stderr_tail": proc.stderr.strip()[-1000:],
            "stdout_tail": proc.stdout.strip()[-1000:],
        }
    artifact = raw.get("artifact") if isinstance(raw, dict) and isinstance(raw.get("artifact"), dict) else {}
    state = artifact.get("state")
    allowed_states = {"running"}
    if args.allow_completed_artifact:
        allowed_states.add("completed")
    reasons = []
    if raw.get("reason") != "ok":
        reasons.append(str(raw.get("reason") or "artifact_probe_failed"))
    if artifact.get("schema") != "tensorcore.windows_cuda_smoke.v1":
        reasons.append("invalid_artifact_schema")
    if artifact.get("resource") != args.resource:
        reasons.append("artifact_resource_mismatch")
    if state not in allowed_states:
        reasons.append(f"artifact_not_live:{state}")
    if state == "completed" and artifact.get("runtime_ok") is not True:
        reasons.append("artifact_runtime_not_ok")
    worker_pid = artifact.get("smoke_pid") or artifact.get("cuda_pid")
    cuda_pid = artifact.get("cuda_pid")
    matched_processes = []
    if worker_pid:
        matched_processes.append({
            "pid": worker_pid,
            "ppid": None,
            "name": "tensorcore_windows_cuda_smoke",
            "executable": artifact.get("executable"),
            "args": f"artifact_state={state} token={artifact.get('token')}",
        })
    cuda_pids = [cuda_pid] if cuda_pid else []
    ok = not reasons and bool(worker_pid)
    if not worker_pid:
        reasons.append("artifact_missing_worker_pid")
    return {
        "schema": "tensorcore.mesh_worker_identity.v1",
        "ok": ok,
        "reason": "ok" if ok else ",".join(reasons),
        "resource": args.resource,
        "worker_host": raw.get("computer_name") or args.target if isinstance(raw, dict) else args.target,
        "worker_pid": worker_pid,
        "worker_pids": [worker_pid] if worker_pid else [],
        "matched_processes": matched_processes,
        "cuda_pids": cuda_pids,
        "matched_cuda_pids": cuda_pids,
        "ignored_opaque_wddm": [],
        "cuda": {"ok": bool(cuda_pids), "apps": []},
        "smoke_artifact": artifact,
        "worker_systemd_unit": None,
        "worker_cgroup": None,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.artifact_path or args.allow_completed_artifact:
        return run_artifact_probe(args)
    try:
        proc = run_remote_powershell(
            args.target,
            render_script(args.match_regex),
            timeout=args.timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return {
            "schema": "tensorcore.mesh_worker_identity.v1",
            "ok": False,
            "reason": "probe_timeout",
            "resource": args.resource,
            "worker_host": args.target,
        }
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
