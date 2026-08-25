#!/usr/bin/env python3
"""Windows SSH admission check for an exclusive NVIDIA CUDA resource."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from typing import Any


SCHEMA = "tensorcore.cuda_resource_admission.v1"


POWERSHELL = r"""
$ErrorActionPreference = 'Continue'
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
function VisibleProcessRows($rows) {
  $seen = $false
  $out = @()
  foreach ($line in $rows) {
    if ($line -match '^\|\s*Processes:') { $seen = $true; continue }
    if (-not $seen) { continue }
    if ($line -notmatch '^\|') { continue }
    if ($line -match '^\|\s*(GPU|=|\+|-)' -or $line -match 'Process name') { continue }
    if ($line -match '^\|\s*\d+\s+\S+\s+\S+\s+\d+\s+\S+\s+(.+?)\s+((?:N/A)|(?:\d+)MiB)\s*\|') {
      $out += $line.Trim()
    }
  }
  return $out
}
function FindNvcc {
  $cmd = Get-Command nvcc -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  $candidates = @()
  if (-not [string]::IsNullOrWhiteSpace($env:CUDA_PATH)) { $candidates += $env:CUDA_PATH }
  $candidates += (Join-Path $env:USERPROFILE 'src\cuda-redist-12.6\toolkit')
  $candidates += 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'
  foreach ($candidate in $candidates) {
    $nvcc = Join-Path $candidate 'bin\nvcc.exe'
    if (Test-Path $nvcc) {
      $env:CUDA_PATH = $candidate
      $env:Path = ((Join-Path $candidate 'bin') + ';' + $env:Path)
      return $nvcc
    }
  }
  return $null
}

$resource = '__RESOURCE__'
$allowedMemory = __ALLOWED_MEMORY__
$requireToolchain = __REQUIRE_TOOLCHAIN__
$allowOpaque = __ALLOW_OPAQUE__
$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$devices = @()
$blocked = @()
$ignored = @()
$visible = @()
$reason = 'nvidia_smi_not_found'
$driverOk = $false
$admissionOk = $false
if ($nvsmi) {
  $gpuRows = @(& $nvsmi.Source '--query-gpu=name,driver_version,memory.total,compute_cap' '--format=csv,noheader,nounits' 2>&1)
  if ($LASTEXITCODE -eq 0) {
    foreach ($line in $gpuRows) {
      $parts = $line.Split(',', 4)
      if ($parts.Count -ge 4) {
        $devices += [ordered]@{
          name = $parts[0].Trim()
          driver_version = $parts[1].Trim()
          memory_total_mib = ToIntOrNull $parts[2].Trim()
          compute_capability = $parts[3].Trim()
        }
      }
    }
  }
  $driverOk = ($devices.Count -gt 0)
  $tableRows = @(& $nvsmi.Source 2>&1)
  if ($LASTEXITCODE -eq 0) { $visible = @(VisibleProcessRows $tableRows) }
  $appRows = @(& $nvsmi.Source '--query-compute-apps=pid,process_name,used_gpu_memory' '--format=csv,noheader,nounits' 2>&1)
  if ($LASTEXITCODE -eq 0) {
    $apps = @(ParseApps $appRows)
    foreach ($app in $apps) {
      if ($null -eq $app.used_memory_mib -or $app.used_memory_mib -gt $allowedMemory) {
        $blocked += $app
      }
    }
    $opaque = @($blocked | Where-Object { $null -eq $_.used_memory_mib -and $_.process_name -eq '[Insufficient Permissions]' })
    if ($blocked.Count -eq 0) {
      $admissionOk = $true
      $reason = 'ok'
    } elseif ($allowOpaque -and $opaque.Count -eq $blocked.Count -and $visible.Count -eq 0) {
      $admissionOk = $true
      $reason = 'ok_opaque_wddm_rows_no_visible_cuda_processes'
      $ignored = $opaque
      $blocked = @()
    } else {
      $reason = 'blocked_cuda_compute_apps'
    }
  } else {
    $reason = 'nvidia_smi_compute_apps_failed'
  }
}
$nvcc = FindNvcc
$toolchainOk = -not [string]::IsNullOrWhiteSpace($nvcc)
if ($driverOk -and $admissionOk -and $requireToolchain -and -not $toolchainOk) {
  $reason = 'cuda_toolkit_not_found'
}
$ok = $driverOk -and $admissionOk -and ((-not $requireToolchain) -or $toolchainOk)
[ordered]@{
  schema = 'tensorcore.cuda_resource_admission.v1'
  ok = $ok
  reason = $reason
  resource = $resource
  checked_at_unix = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
  windows = $true
  driver_ok = $driverOk
  toolchain_ok = $toolchainOk
  admission_ok = $admissionOk
  device_count = $devices.Count
  devices = $devices
  visible_processes = $visible
  allowed_process_max_memory_mib = $allowedMemory
  blocked = $blocked
  ignored_opaque_wddm = $ignored
  cuda_toolkit = [ordered]@{
    nvcc_found = $toolchainOk
    nvcc_path = $nvcc
    cuda_path = $env:CUDA_PATH
  }
} | ConvertTo-Json -Depth 8 -Compress
"""


def ps_bool(value: bool) -> str:
    return "$true" if value else "$false"


def render_script(args: argparse.Namespace) -> str:
    return (
        POWERSHELL
        .replace("__RESOURCE__", args.resource.replace("'", "''"))
        .replace("__ALLOWED_MEMORY__", str(args.allowed_process_max_memory_mib))
        .replace("__REQUIRE_TOOLCHAIN__", ps_bool(args.require_toolchain))
        .replace("__ALLOW_OPAQUE__", ps_bool(not args.disallow_opaque_wddm))
    )


def ps_encode(command: str) -> str:
    return base64.b64encode(command.encode("utf-16le")).decode("ascii")


def run_remote_powershell(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
    script_name = f"tensorcore-windows-cuda-admission-{os.getpid()}.ps1"
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


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    try:
        proc = run_remote_powershell(args.target, render_script(args), timeout=args.timeout_sec)
    except subprocess.TimeoutExpired:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "probe_timeout",
            "resource": args.resource,
            "blocked": [],
        }
    if proc.returncode != 0:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "ssh_or_powershell_failed",
            "resource": args.resource,
            "rc": proc.returncode,
            "blocked": [],
            "stdout_tail": proc.stdout.strip()[-1000:],
            "stderr_tail": proc.stderr.strip()[-1000:],
        }
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return payload
    return {
        "schema": SCHEMA,
        "ok": False,
        "reason": "invalid_probe_json",
        "resource": args.resource,
        "blocked": [],
        "stdout_tail": proc.stdout.strip()[-1000:],
        "stderr_tail": proc.stderr.strip()[-1000:],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--resource", required=True)
    parser.add_argument("--allowed-process-max-memory-mib", type=int, default=64)
    parser.add_argument("--require-toolchain", action="store_true")
    parser.add_argument("--disallow-opaque-wddm", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


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
