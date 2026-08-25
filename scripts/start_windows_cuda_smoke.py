#!/usr/bin/env python3
"""Start a short Windows CUDA smoke under the mesh scheduler."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any


SCHEMA = "tensorcore.windows_cuda_smoke.start.v1"

CHILD_SCRIPT = r"""
param(
  [string]$ArtifactPath,
  [string]$Resource,
  [int]$DurationSec,
  [string]$Token,
  [string]$TaskName = ''
)
$ErrorActionPreference = 'Continue'

function UnixNow {
  return [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
}

function Write-SmokeStatus($Fields) {
  $payload = [ordered]@{
    schema = 'tensorcore.windows_cuda_smoke.v1'
    resource = $Resource
    token = $Token
    state = $Fields.state
    ok = $Fields.ok
    smoke_pid = $PID
    checked_at_unix = UnixNow
  }
  foreach ($key in $Fields.Keys) {
    if (-not $payload.Contains($key)) {
      $payload[$key] = $Fields[$key]
    }
  }
  $parent = Split-Path -Parent $ArtifactPath
  if (-not [string]::IsNullOrWhiteSpace($parent)) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
  }
  $tmp = $ArtifactPath + '.tmp'
  $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $tmp -Encoding UTF8
  Move-Item -Force -LiteralPath $tmp -Destination $ArtifactPath
}

function Finish-Smoke([int]$Code) {
  if (-not [string]::IsNullOrWhiteSpace($TaskName)) {
    & schtasks.exe /Delete /F /TN $TaskName *> $null
  }
  exit $Code
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

function FindVsDevCmd {
  $candidates = @()
  if (-not [string]::IsNullOrWhiteSpace(${env:ProgramFiles(x86)})) {
    $candidates += (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat')
    $candidates += (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat')
  }
  if (-not [string]::IsNullOrWhiteSpace($env:ProgramFiles)) {
    $candidates += (Join-Path $env:ProgramFiles 'Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat')
    $candidates += (Join-Path $env:ProgramFiles 'Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat')
  }
  foreach ($candidate in $candidates) {
    if (Test-Path $candidate) { return $candidate }
  }
  return $null
}

function CmdQuote([string]$Value) {
  return '"' + $Value + '"'
}

$workDir = Join-Path $env:TEMP "tensorcore_windows_cuda_smoke_$Token"
New-Item -ItemType Directory -Force -Path $workDir | Out-Null
$source = Join-Path $workDir "tensorcore_windows_cuda_smoke_$Token.cu"
$exe = Join-Path $workDir "tensorcore_windows_cuda_smoke_$Token.exe"
$buildLog = Join-Path $workDir "build.log"
$runOut = Join-Path $workDir "run.stdout.log"
$runErr = Join-Path $workDir "run.stderr.log"

Write-SmokeStatus @{
  state = 'building'
  ok = $false
  work_dir = $workDir
  artifact_path = $ArtifactPath
}

$nvcc = FindNvcc
if (-not $nvcc) {
  Write-SmokeStatus @{
    state = 'failed'
    ok = $false
    reason = 'cuda_toolkit_not_found'
    work_dir = $workDir
    artifact_path = $ArtifactPath
  }
  Finish-Smoke 1
}
$vsDevCmd = FindVsDevCmd
if (-not $vsDevCmd) {
  Write-SmokeStatus @{
    state = 'failed'
    ok = $false
    reason = 'vsdevcmd_not_found'
    nvcc_path = $nvcc
    work_dir = $workDir
    artifact_path = $ArtifactPath
  }
  Finish-Smoke 1
}

@'
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <chrono>

__global__ void gemm_kernel(const float* a, const float* b, float* c, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n || col >= n) return;
  float sum = 0.0f;
  for (int k = 0; k < n; ++k) {
    sum += a[row * n + k] * b[k * n + col];
  }
  c[row * n + col] = sum;
}

int main(int argc, char** argv) {
  int duration = 30;
  if (argc > 1) duration = std::atoi(argv[1]);
  cudaDeviceProp prop{};
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return 10;
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 11;
  constexpr int n = 16;
  constexpr int count = n * n;
  float ha[count], hb[count], hc[count];
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      ha[row * n + col] = static_cast<float>((row + 1) * 0.25f + col);
      hb[row * n + col] = static_cast<float>((col + 1) * 0.125f - row);
      hc[row * n + col] = 0.0f;
    }
  }
  float *da = nullptr, *db = nullptr, *dc = nullptr;
  if (cudaMalloc(&da, count * sizeof(float)) != cudaSuccess) return 20;
  if (cudaMalloc(&db, count * sizeof(float)) != cudaSuccess) return 21;
  if (cudaMalloc(&dc, count * sizeof(float)) != cudaSuccess) return 22;
  if (cudaMemcpy(da, ha, count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 23;
  if (cudaMemcpy(db, hb, count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 24;
  dim3 block(16, 16);
  dim3 grid(1, 1);
  gemm_kernel<<<grid, block>>>(da, db, dc, n);
  if (cudaDeviceSynchronize() != cudaSuccess) return 30;
  if (cudaMemcpy(hc, dc, count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 31;
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      float expected = 0.0f;
      for (int k = 0; k < n; ++k) {
        expected += ha[row * n + k] * hb[k * n + col];
      }
      if (std::fabs(hc[row * n + col] - expected) > 1.0e-3f) return 40;
    }
  }
  std::printf("tensorcore_windows_cuda_smoke gemm ok device=%s duration=%d\n", prop.name, duration);
  std::fflush(stdout);
  std::this_thread::sleep_for(std::chrono::seconds(duration));
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  cudaDeviceReset();
  return 0;
}
'@ | Set-Content -LiteralPath $source -Encoding ASCII

$cudaPath = $env:CUDA_PATH
if ([string]::IsNullOrWhiteSpace($cudaPath)) {
  $cudaPath = Split-Path -Parent (Split-Path -Parent $nvcc)
}
$cudaBin = Split-Path -Parent $nvcc
$batchPath = Join-Path $workDir "build.cmd"
$buildSteps = @(
  '@echo off',
  ('call ' + (CmdQuote $vsDevCmd) + ' -arch=x64 -host_arch=x64'),
  'if errorlevel 1 exit /b %errorlevel%',
  ('set "CUDA_PATH=' + $cudaPath + '"'),
  ('set "PATH=' + $cudaBin + ';%PATH%"'),
  ((CmdQuote $nvcc) + ' -std=c++17 -O2 ' + (CmdQuote $source) + ' -o ' + (CmdQuote $exe))
)
Set-Content -LiteralPath $batchPath -Encoding ASCII -Value $buildSteps
$buildOutput = @(& cmd.exe /c (CmdQuote $batchPath) 2>&1)
$buildRc = $LASTEXITCODE
$buildOutput | Set-Content -LiteralPath $buildLog -Encoding UTF8
if ($buildRc -ne 0) {
  Write-SmokeStatus @{
    state = 'failed'
    ok = $false
    reason = 'build_failed'
    nvcc_path = $nvcc
    vsdevcmd_path = $vsDevCmd
    build_rc = $buildRc
    build_log_tail = (($buildOutput -join [Environment]::NewLine))
    work_dir = $workDir
    artifact_path = $ArtifactPath
  }
  Finish-Smoke $buildRc
}

$startedAt = UnixNow
Write-SmokeStatus @{
  state = 'running'
  ok = $true
  reason = 'running'
  nvcc_path = $nvcc
  vsdevcmd_path = $vsDevCmd
  executable = $exe
  build_ok = $true
  build_rc = $buildRc
  duration_sec = $DurationSec
  heartbeat_unix = $startedAt
  expected_complete_after_unix = ($startedAt + $DurationSec + 30)
  work_dir = $workDir
  artifact_path = $ArtifactPath
}

$timedOut = $false
$exitCode = $null
$cudaPid = 0
$runtimeTimeoutSec = [Math]::Max(1, $DurationSec + 15)
$runtimeTimeoutMs = [int][Math]::Min([int]::MaxValue, [Math]::Max(1000, ($runtimeTimeoutSec * 1000)))
try {
  $psi = [System.Diagnostics.ProcessStartInfo]::new()
  $psi.FileName = $exe
  $psi.Arguments = [string]$DurationSec
  $psi.WorkingDirectory = $workDir
  $psi.UseShellExecute = $false
  $psi.CreateNoWindow = $true
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $runProc = [System.Diagnostics.Process]::new()
  $runProc.StartInfo = $psi
  $null = $runProc.Start()
  $cudaPid = $runProc.Id
  $stdoutTask = $runProc.StandardOutput.ReadToEndAsync()
  $stderrTask = $runProc.StandardError.ReadToEndAsync()
  Write-SmokeStatus @{
    state = 'running'
    ok = $true
    reason = 'running'
    nvcc_path = $nvcc
    vsdevcmd_path = $vsDevCmd
    executable = $exe
    build_ok = $true
    build_rc = $buildRc
    cuda_pid = $cudaPid
    duration_sec = $DurationSec
    runtime_timeout_sec = $runtimeTimeoutSec
    heartbeat_unix = UnixNow
    expected_complete_after_unix = ($startedAt + $runtimeTimeoutSec)
    work_dir = $workDir
    artifact_path = $ArtifactPath
  }
  if (-not $runProc.WaitForExit($runtimeTimeoutMs)) {
    $timedOut = $true
    try { $runProc.Kill() } catch {}
    try { $runProc.WaitForExit(5000) | Out-Null } catch {}
    $exitCode = 124
  } else {
    $exitCode = $runProc.ExitCode
  }
  $stdoutTask.Result | Set-Content -LiteralPath $runOut -Encoding UTF8
  $stderrTask.Result | Set-Content -LiteralPath $runErr -Encoding UTF8
} catch {
  $exitCode = 1
  $_.Exception.Message | Set-Content -LiteralPath $runErr -Encoding UTF8
}
$stdout = ''
$stderr = ''
if (Test-Path -LiteralPath $runOut) { $stdout = Get-Content -Raw -LiteralPath $runOut }
if (Test-Path -LiteralPath $runErr) { $stderr = Get-Content -Raw -LiteralPath $runErr }
$stdoutLooksOk = ($stdout -match 'tensorcore_windows_cuda_smoke gemm ok')
if ((-not $timedOut) -and $null -eq $exitCode -and $stdoutLooksOk -and [string]::IsNullOrWhiteSpace($stderr)) {
  $exitCode = 0
}
$ok = ((-not $timedOut) -and $exitCode -eq 0)
$finalReason = 'runtime_failed'
if ($timedOut) {
  $finalReason = 'runtime_timeout'
} elseif ($ok) {
  $finalReason = 'ok'
}
Write-SmokeStatus @{
  state = 'completed'
  ok = $ok
  reason = $finalReason
  nvcc_path = $nvcc
  vsdevcmd_path = $vsDevCmd
  executable = $exe
  build_ok = $true
  runtime_ok = $ok
  runtime_timeout = $timedOut
  runtime_timeout_sec = $runtimeTimeoutSec
  exit_code = $exitCode
  cuda_pid = $cudaPid
  stdout_tail = $stdout
  stderr_tail = $stderr
  duration_sec = $DurationSec
  heartbeat_unix = UnixNow
  expected_complete_after_unix = ($startedAt + $DurationSec + 30)
  work_dir = $workDir
  artifact_path = $ArtifactPath
  completed_at_unix = UnixNow
}
if ($ok) { Finish-Smoke 0 }
if ($timedOut) { Finish-Smoke 124 }
if ($null -ne $exitCode) { Finish-Smoke $exitCode }
Finish-Smoke 1
"""


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


def render_parent(args: argparse.Namespace, token: str) -> str:
    child_b64 = base64.b64encode(CHILD_SCRIPT.encode("utf-8")).decode("ascii")
    foreground_block = ""
    if args.foreground:
        foreground_block = f"""
& $ChildPath -ArtifactPath $ArtifactPath -Resource $Resource -DurationSec $DurationSec -Token $Token
$childRc = $LASTEXITCODE
$payload = ReadSmokePayload
$state = if ($null -ne $payload) {{ $payload.state }} else {{ 'unknown' }}
$reason = 'foreground_missing_artifact'
if ($null -ne $payload) {{
  $reason = [string]$payload.reason
  if ([string]::IsNullOrWhiteSpace($reason)) {{ $reason = [string]$payload.state }}
}}
$launcherPid = 0
if ($null -ne $payload -and $payload.smoke_pid) {{ $launcherPid = [int]$payload.smoke_pid }}
[ordered]@{{
  schema = '{SCHEMA}'
  ok = ($null -ne $payload -and $payload.state -eq 'completed' -and $payload.ok -eq $true)
  reason = $reason
  resource = $Resource
  token = $Token
  launcher_pid = $launcherPid
  launch_mode = 'foreground'
  child_rc = $childRc
  artifact_path = $ArtifactPath
  state = $state
  payload = $payload
}} | ConvertTo-Json -Depth 10 -Compress
exit 0
"""
    return f"""
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
function CmdArg([string]$Value) {{
  return '"' + ($Value -replace '"','\\"') + '"'
}}
function PsSingle([string]$Value) {{
  return "'" + ($Value -replace "'", "''") + "'"
}}
function TailText($Lines) {{
  $text = ($Lines -join [Environment]::NewLine)
  if ($text.Length -gt 1000) {{ return $text.Substring($text.Length - 1000) }}
  return $text
}}
function ReadSmokePayload {{
  if (-not (Test-Path -LiteralPath $ArtifactPath)) {{ return $null }}
  try {{
    $candidate = Get-Content -Raw -LiteralPath $ArtifactPath | ConvertFrom-Json
    if ($candidate.token -eq $Token) {{ return $candidate }}
  }} catch {{}}
  return $null
}}
function WaitSmokePayload([double]$Seconds) {{
  $deadline = (Get-Date).AddSeconds([Math]::Max(0.0, $Seconds))
  $last = $null
  while ((Get-Date) -lt $deadline) {{
    Start-Sleep -Milliseconds 500
    $candidate = ReadSmokePayload
    if ($null -ne $candidate) {{
      $last = $candidate
      if ($candidate.state -in @('running','completed','failed')) {{ return $candidate }}
    }}
  }}
  return $last
}}
function StartDirectChild {{
  $directArgs = '-NoProfile -ExecutionPolicy Bypass -File ' + (CmdArg $ChildPath) +
    ' -ArtifactPath ' + (CmdArg $ArtifactPath) +
    ' -Resource ' + (CmdArg $Resource) +
    ' -DurationSec ' + [string]$DurationSec +
    ' -Token ' + (CmdArg $Token)
  $direct = Start-Process -FilePath 'powershell.exe' -ArgumentList $directArgs -WindowStyle Hidden -PassThru
  return $direct.Id
}}
{artifact_path_assignment(args.resource, args.artifact_path)}
$Resource = '{ps_literal(args.resource)}'
$DurationSec = {int(args.duration_sec)}
$Token = '{ps_literal(token)}'
$TaskName = "TensorcoreWindowsCudaSmoke_" + ($Token -replace '[^A-Za-z0-9_-]', '_')
$ChildPath = Join-Path $env:TEMP ("tensorcore_windows_cuda_smoke_" + $Token + ".ps1")
$TaskScriptPath = Join-Path $env:TEMP ("tcwcs_" + $Token + ".ps1")
$ChildText = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{child_b64}'))
Set-Content -LiteralPath $ChildPath -Encoding UTF8 -Value $ChildText
Remove-Item -Force -LiteralPath $ArtifactPath -ErrorAction SilentlyContinue
{foreground_block}
$InvokeLine = '  & ' + (PsSingle $ChildPath) + ' -ArtifactPath ' + (PsSingle $ArtifactPath) + ' -Resource ' + (PsSingle $Resource) + ' -DurationSec ' + [string]$DurationSec + ' -Token ' + (PsSingle $Token) + ' -TaskName ' + (PsSingle $TaskName)
$CleanupLine = '  Remove-Item -Force -LiteralPath ' + (PsSingle $TaskScriptPath) + ' -ErrorAction SilentlyContinue'
$TaskScript = @('$ErrorActionPreference = ''Continue''', 'try {{', $InvokeLine, '}} finally {{', $CleanupLine, '}}')
Set-Content -LiteralPath $TaskScriptPath -Encoding UTF8 -Value $TaskScript
$TaskCommand = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File ' + (CmdArg $TaskScriptPath)
$startTime = (Get-Date).AddMinutes(1).ToString('HH:mm')
& schtasks.exe /Delete /F /TN $TaskName *> $null
$createOutput = @(& schtasks.exe /Create /F /TN $TaskName /SC ONCE /ST $startTime /TR $TaskCommand 2>&1)
$createRc = $LASTEXITCODE
if ($createRc -eq 0) {{
  $runOutput = @(& schtasks.exe /Run /TN $TaskName 2>&1)
  $runRc = $LASTEXITCODE
}} else {{
  $runOutput = @()
  $runRc = -1
}}
$directPid = 0
$launchMode = 'scheduled_task'
if ($createRc -ne 0 -or $runRc -ne 0) {{
  & schtasks.exe /Delete /F /TN $TaskName *> $null
  $scheduledTaskReason = if ($createRc -ne 0) {{ 'scheduled_task_create_failed' }} else {{ 'scheduled_task_run_failed' }}
  $createTail = TailText $createOutput
  $runTail = TailText $runOutput
  [ordered]@{{
    schema = '{SCHEMA}'
    ok = $false
    reason = $scheduledTaskReason
    resource = $Resource
    token = $Token
    launcher_pid = 0
    direct_launcher_pid = 0
    launch_mode = $launchMode
    scheduled_task_name = $TaskName
    scheduled_task_create_rc = $createRc
    scheduled_task_run_rc = $runRc
    scheduled_task_create_tail = $createTail
    scheduled_task_run_tail = $runTail
    artifact_path = $ArtifactPath
    state = 'unknown'
    payload = $null
  }} | ConvertTo-Json -Depth 10 -Compress
  exit 0
}}
$payload = WaitSmokePayload ([double]{int(args.start_wait_sec)})
$state = if ($null -ne $payload) {{ $payload.state }} else {{ 'unknown' }}
$ready = ($null -ne $payload -and $payload.state -in @('running','completed','failed'))
$reason = 'start_wait_timeout'
if ($ready) {{
  $reason = [string]$payload.reason
  if ([string]::IsNullOrWhiteSpace($reason)) {{ $reason = [string]$payload.state }}
  if ($launchMode -ne 'scheduled_task') {{ $reason = 'non_durable_' + $launchMode }}
}}
$launcherPid = 0
if ($null -ne $payload -and $payload.smoke_pid) {{ $launcherPid = [int]$payload.smoke_pid }}
$startOk = (
  $null -ne $payload -and
  $launchMode -eq 'scheduled_task' -and
  (
    $payload.state -eq 'running' -or
    ($payload.state -eq 'completed' -and $payload.ok -eq $true)
  )
)
[ordered]@{{
  schema = '{SCHEMA}'
  ok = $startOk
  reason = $reason
  resource = $Resource
  token = $Token
  launcher_pid = $launcherPid
  direct_launcher_pid = $directPid
  launch_mode = $launchMode
  scheduled_task_name = $TaskName
  scheduled_task_create_rc = $createRc
  scheduled_task_run_rc = $runRc
  scheduled_task_create_tail = TailText $createOutput
  scheduled_task_run_tail = TailText $runOutput
  artifact_path = $ArtifactPath
  state = $state
  payload = $payload
}} | ConvertTo-Json -Depth 10 -Compress
"""


def render_foreground_recovery(args: argparse.Namespace, token: str) -> str:
    return f"""
$ErrorActionPreference = 'Continue'
{artifact_path_assignment(args.resource, args.artifact_path)}
$Resource = '{ps_literal(args.resource)}'
$Token = '{ps_literal(token)}'
function UnixNow {{
  return [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
}}
function Write-Result($Payload, [bool]$Ok, [string]$Reason) {{
  $launcherPid = 0
  if ($null -ne $Payload -and $Payload.smoke_pid) {{ $launcherPid = [int]$Payload.smoke_pid }}
  $state = if ($null -ne $Payload) {{ $Payload.state }} else {{ 'unknown' }}
  [ordered]@{{
    schema = '{SCHEMA}'
    ok = $Ok
    reason = $Reason
    resource = $Resource
    token = $Token
    launcher_pid = $launcherPid
    launch_mode = 'foreground_recovered_after_timeout'
    artifact_path = $ArtifactPath
    state = $state
    payload = $Payload
  }} | ConvertTo-Json -Depth 10 -Compress
}}
if (-not (Test-Path -LiteralPath $ArtifactPath)) {{
  Write-Result $null $false 'foreground_recovery_missing_artifact'
  exit 0
}}
try {{
  $artifact = Get-Content -Raw -LiteralPath $ArtifactPath | ConvertFrom-Json
}} catch {{
  Write-Result $null $false 'foreground_recovery_invalid_artifact'
  exit 0
}}
if ($artifact.token -ne $Token) {{
  Write-Result $artifact $false 'foreground_recovery_token_mismatch'
  exit 0
}}
$workDir = [string]$artifact.work_dir
$runOut = Join-Path $workDir 'run.stdout.log'
$runErr = Join-Path $workDir 'run.stderr.log'
$stdout = ''
$stderr = ''
if (Test-Path -LiteralPath $runOut) {{ $stdout = Get-Content -Raw -LiteralPath $runOut }}
if (Test-Path -LiteralPath $runErr) {{ $stderr = Get-Content -Raw -LiteralPath $runErr }}
$runtimeOk = ($stdout -match 'tensorcore_windows_cuda_smoke gemm ok' -and [string]::IsNullOrWhiteSpace($stderr))
if (-not $runtimeOk) {{
  Write-Result $artifact $false 'foreground_recovery_runtime_not_ok'
  exit 0
}}
$now = UnixNow
$payload = [ordered]@{{
  schema = 'tensorcore.windows_cuda_smoke.v1'
  resource = $Resource
  token = $Token
  state = 'completed'
  ok = $true
  smoke_pid = $artifact.smoke_pid
  checked_at_unix = $now
  reason = 'ok'
  nvcc_path = $artifact.nvcc_path
  vsdevcmd_path = $artifact.vsdevcmd_path
  executable = $artifact.executable
  build_ok = $true
  build_rc = $artifact.build_rc
  runtime_ok = $true
  runtime_timeout = $false
  exit_code = 0
  stdout_tail = $stdout
  stderr_tail = $stderr
  duration_sec = $artifact.duration_sec
  heartbeat_unix = $now
  expected_complete_after_unix = $artifact.expected_complete_after_unix
  work_dir = $workDir
  artifact_path = $ArtifactPath
  completed_at_unix = $now
}}
$tmp = $ArtifactPath + '.tmp'
$payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $tmp -Encoding UTF8
Move-Item -Force -LiteralPath $tmp -Destination $ArtifactPath
Write-Result $payload $true 'foreground_recovered'
exit 0
"""


def run_remote_powershell(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
    script_name = f"tensorcore-windows-cuda-smoke-start-{os.getpid()}.ps1"
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


def run_start(args: argparse.Namespace) -> dict[str, Any]:
    token = f"tensorcore_windows_cuda_smoke_{int(time.time())}_{os.getpid()}"
    try:
        proc = run_remote_powershell(
            args.target,
            render_parent(args, token),
            timeout=args.timeout_sec,
        )
    except subprocess.TimeoutExpired:
        if args.foreground and args.recover_foreground_timeout:
            recovered = recover_foreground_completion(args, token)
            if recovered.get("ok"):
                return recovered
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "start_timeout",
            "resource": args.resource,
            "token": token,
        }
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
            "reason": "invalid_start_json",
            "resource": args.resource,
            "stdout_tail": proc.stdout.strip()[-1000:],
            "stderr_tail": proc.stderr.strip()[-1000:],
        }
    if isinstance(payload, dict):
        if payload.get("schema") != SCHEMA:
            return {
                "schema": SCHEMA,
                "ok": False,
                "reason": "invalid_start_schema",
                "resource": args.resource,
                "start_schema": payload.get("schema"),
            }
        if payload.get("resource") != args.resource:
            return {
                "schema": SCHEMA,
                "ok": False,
                "reason": "start_resource_mismatch",
                "resource": args.resource,
                "start_resource": payload.get("resource"),
            }
        return payload
    return {"schema": SCHEMA, "ok": False, "reason": "non_object_start_json", "resource": args.resource}


def recover_foreground_completion(args: argparse.Namespace, token: str) -> dict[str, Any]:
    try:
        proc = run_remote_powershell(
            args.target,
            render_foreground_recovery(args, token),
            timeout=min(max(args.timeout_sec, 10.0), 20.0),
        )
    except subprocess.TimeoutExpired:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "foreground_recovery_timeout",
            "resource": args.resource,
            "token": token,
        }
    if proc.returncode != 0:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "foreground_recovery_ssh_or_powershell_failed",
            "resource": args.resource,
            "token": token,
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
            "reason": "foreground_recovery_invalid_json",
            "resource": args.resource,
            "token": token,
            "stdout_tail": proc.stdout.strip()[-1000:],
            "stderr_tail": proc.stderr.strip()[-1000:],
        }
    if isinstance(payload, dict):
        return payload
    return {
        "schema": SCHEMA,
        "ok": False,
        "reason": "foreground_recovery_non_object_json",
        "resource": args.resource,
        "token": token,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--resource", required=True)
    parser.add_argument("--artifact-path", default="")
    parser.add_argument("--duration-sec", type=int, default=45)
    parser.add_argument("--start-wait-sec", type=int, default=20)
    parser.add_argument("--timeout-sec", type=float, default=40.0)
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--recover-foreground-timeout", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = run_start(args)
    if args.json:
        json.dump(payload, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"{args.resource}: ok={payload.get('ok')} state={payload.get('state')}")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
