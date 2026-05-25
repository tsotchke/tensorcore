#!/usr/bin/env bash
set -euo pipefail

windows_config="${TC_WINDOWS_CONFIG:-$HOME/.config/tensorcore/windows-host.env}"
if [[ -f "$windows_config" ]]; then
    # Private machine coordinates belong in local config, never in source.
    # shellcheck source=/dev/null
    source "$windows_config"
fi

windows_ssh="${TC_WINDOWS_SSH:-}"
windows_repo="${TC_WINDOWS_REPO:-src/tensorcore}"
windows_remote_url="${TC_WINDOWS_REMOTE_URL:-https://github.com/tsotchke/tensorcore.git}"
windows_ref="${TC_WINDOWS_REF:-master}"
windows_reset="${TC_WINDOWS_RESET:-0}"
windows_timeout="${TC_WINDOWS_SSH_CONNECT_TIMEOUT:-10}"
windows_evidence_path="${TC_WINDOWS_CUDA_EVIDENCE_PATH:-}"
allowed_process_max_memory_mib="${TC_WINDOWS_CUDA_ALLOWED_PROCESS_MAX_MEMORY_MIB:-64}"
windows_build_smoke="${TC_WINDOWS_CUDA_BUILD_SMOKE:-0}"
evidence_marker="__TENSORCORE_WINDOWS_CUDA_PROBE_EVIDENCE__"

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_windows_cuda_probe.sh

Environment:
  TC_WINDOWS_CONFIG       Optional local env file, default ~/.config/tensorcore/windows-host.env
  TC_WINDOWS_SSH          SSH target. Required via env or TC_WINDOWS_CONFIG.
  TC_WINDOWS_SSH_KEY      Optional private key path
  TC_WINDOWS_REPO         Remote repo path, default src/tensorcore
  TC_WINDOWS_REMOTE_URL   Remote clone URL
  TC_WINDOWS_REF          Branch/ref to test, default master
  TC_WINDOWS_RESET=1      Hard-reset remote repo to origin/<ref>
  TC_WINDOWS_CUDA_EVIDENCE_PATH  Optional local JSON evidence output path
  TC_WINDOWS_CUDA_BUILD_SMOKE=1  Also configure, build, and CTest CUDA on Windows
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ -z "$windows_ssh" ]]; then
    echo "tensorcore/windows-cuda: TC_WINDOWS_SSH is required via env or $windows_config" >&2
    exit 2
fi

ps_quote() {
    local value=$1
    value=${value//\'/\'\'}
    printf "'%s'" "$value"
}

ssh_opts=(-o BatchMode=yes -o ConnectTimeout="$windows_timeout")
if [[ -n "${TC_WINDOWS_SSH_KEY:-}" ]]; then
    ssh_opts+=(-i "$TC_WINDOWS_SSH_KEY" -o IdentitiesOnly=yes)
fi

repo_q=$(ps_quote "$windows_repo")
url_q=$(ps_quote "$windows_remote_url")
ref_q=$(ps_quote "$windows_ref")
reset_q=$(ps_quote "$windows_reset")
marker_q=$(ps_quote "$evidence_marker")
allowed_memory_q=$(ps_quote "$allowed_process_max_memory_mib")
build_smoke_q=$(ps_quote "$windows_build_smoke")

remote_command=$(cat <<PS
\$ErrorActionPreference = 'Stop'
\$ProgressPreference = 'SilentlyContinue'
Set-StrictMode -Version 3.0

\$Repo = $repo_q
\$RemoteUrl = $url_q
\$Ref = $ref_q
\$Reset = $reset_q
\$EvidenceMarker = $marker_q
\$AllowedProcessMaxMemoryMiB = [int]$allowed_memory_q
\$BuildSmoke = $build_smoke_q

function Find-CommandPath([string]\$Name) {
    \$cmd = Get-Command \$Name -ErrorAction SilentlyContinue
    if (\$null -eq \$cmd) { return \$null }
    return \$cmd.Source
}

function Tail-Text([string]\$Value, [int]\$Limit = 1000) {
    if ([string]::IsNullOrEmpty(\$Value)) { return '' }
    if (\$Value.Length -le \$Limit) { return \$Value }
    return \$Value.Substring(\$Value.Length - \$Limit)
}

function To-NullableInt([string]\$Value) {
    \$digits = (\$Value -replace '[^0-9-]', '')
    if ([string]::IsNullOrWhiteSpace(\$digits)) { return \$null }
    try { return [int]\$digits } catch { return \$null }
}

function Update-Repo {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw 'git not found on PATH. Install Git for Windows first.'
    }
    \$Parent = Split-Path -Parent \$Repo
    if (-not [string]::IsNullOrWhiteSpace(\$Parent) -and -not (Test-Path \$Parent)) {
        New-Item -ItemType Directory -Force -Path \$Parent | Out-Null
    }
    \$GitDir = Join-Path \$Repo '.git'
    if (-not (Test-Path \$GitDir)) {
        Write-Output "[tensorcore/windows-cuda] clone \$RemoteUrl -> \$Repo"
        git clone --branch \$Ref \$RemoteUrl \$Repo
        if (\$LASTEXITCODE -ne 0) { throw "git clone failed with exit code \$LASTEXITCODE" }
    } else {
        Write-Output "[tensorcore/windows-cuda] update existing checkout"
        git -C \$Repo fetch --prune origin \$Ref
        if (\$LASTEXITCODE -ne 0) { throw "git fetch failed with exit code \$LASTEXITCODE" }
        if (\$Reset -eq '1') {
            git -C \$Repo reset --hard "origin/\$Ref"
            if (\$LASTEXITCODE -ne 0) { throw "git reset failed with exit code \$LASTEXITCODE" }
        } else {
            git -C \$Repo checkout \$Ref
            if (\$LASTEXITCODE -ne 0) { throw "git checkout failed with exit code \$LASTEXITCODE" }
            git -C \$Repo pull --ff-only origin \$Ref
            if (\$LASTEXITCODE -ne 0) { throw "git pull --ff-only failed with exit code \$LASTEXITCODE" }
        }
    }
}

function Parse-GpuRows([string[]]\$Rows) {
    \$devices = @()
    foreach (\$line in \$Rows) {
        if ([string]::IsNullOrWhiteSpace(\$line)) { continue }
        \$parts = @(\$line -split ',')
        if (\$parts.Count -lt 4) { continue }
        \$devices += [ordered]@{
            name = \$parts[0].Trim()
            driver_version = \$parts[1].Trim()
            memory_total_mib = To-NullableInt \$parts[2].Trim()
            compute_capability = \$parts[3].Trim()
        }
    }
    return \$devices
}

function Parse-ComputeApps([string[]]\$Rows) {
    \$apps = @()
    foreach (\$line in \$Rows) {
        if ([string]::IsNullOrWhiteSpace(\$line)) { continue }
        \$parts = @(\$line -split ',')
        if (\$parts.Count -lt 3) {
            \$apps += [ordered]@{ raw = \$line; parse_error = 'expected pid, process_name, used_gpu_memory' }
            continue
        }
        \$apps += [ordered]@{
            pid = To-NullableInt \$parts[0].Trim()
            process_name = \$parts[1].Trim()
            used_memory_mib = To-NullableInt \$parts[2].Trim()
            raw = \$line
        }
    }
    return \$apps
}

function Parse-NvidiaSmiProcessTable([string[]]\$Rows) {
    \$processes = @()
    \$inProcesses = \$false
    foreach (\$line in \$Rows) {
        if (\$line -match '^\|\s*Processes:') {
            \$inProcesses = \$true
            continue
        }
        if (-not \$inProcesses) { continue }
        if (\$line -notmatch '^\|') { continue }
        if (\$line -match '^\|\s*(GPU|=|\+|-)' -or \$line -match 'Process name') { continue }
        if (\$line -match '^\|\s*(\d+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(.+?)\s+((?:N/A)|(?:\d+)MiB)\s*\|') {
            \$processes += [ordered]@{
                gpu = To-NullableInt \$Matches[1]
                gi = \$Matches[2]
                ci = \$Matches[3]
                pid = To-NullableInt \$Matches[4]
                type = \$Matches[5]
                process_name = \$Matches[6].Trim()
                used_memory = \$Matches[7]
            }
        }
    }
    return \$processes
}

function Is-OpaqueWddmApp(\$App) {
    return (
        \$null -eq \$App.used_memory_mib -and
        [string]\$App.process_name -eq '[Insufficient Permissions]'
    )
}

function Cmd-Quote([string]\$Value) {
    return '"' + \$Value + '"'
}

function Parse-CtestSummary([string]\$Text) {
    \$summary = [ordered]@{
        tests_total = \$null
        tests_failed = \$null
        tests_passed = \$null
        tests_skipped = \$null
        cuda_gemm_passed = \$false
    }
    if (\$Text -match '100% tests passed, 0 tests failed out of (\d+)') {
        \$summary.tests_total = [int]\$Matches[1]
        \$summary.tests_failed = 0
    } elseif (\$Text -match '(\d+)% tests passed, (\d+) tests failed out of (\d+)') {
        \$summary.tests_total = [int]\$Matches[3]
        \$summary.tests_failed = [int]\$Matches[2]
    }
    if (\$Text -match 'Test\s+#?\d+:\s+test_cuda_gemm\s+\.*\s+Passed') {
        \$summary.cuda_gemm_passed = \$true
    }
    if (\$Text -match 'The following tests did not run:') {
        \$skipped = 0
        foreach (\$line in \$Text -split [Environment]::NewLine) {
            if (\$line -match '^\s*\d+\s+-\s+\S+\s+\(Skipped\)') {
                \$skipped += 1
            }
        }
        \$summary.tests_skipped = \$skipped
    }
    if (\$null -ne \$summary.tests_total -and \$null -ne \$summary.tests_failed) {
        \$SkippedCount = 0
        if (\$null -ne \$summary.tests_skipped) {
            \$SkippedCount = [int]\$summary.tests_skipped
        }
        \$summary.tests_passed = \$summary.tests_total - \$summary.tests_failed - \$SkippedCount
    }
    return \$summary
}

function Run-BuildSmoke([string]\$RuntimeStatus, \$CudaToolkit) {
    \$BuildDir = Join-Path \$Repo 'build-windows-cuda-smoke'
    \$result = [ordered]@{
        ran = \$true
        ok = \$false
        build_dir = \$BuildDir
        reason = \$null
        rc = \$null
        command_lines = @()
        output_tail = ''
        tests_total = \$null
        tests_passed = \$null
        tests_failed = \$null
        tests_skipped = \$null
        cuda_gemm_passed = \$false
    }
    if (\$RuntimeStatus -ne 'ready') {
        \$result.reason = 'runtime_not_ready'
        return \$result
    }
    \$Vs = Join-Path \${env:ProgramFiles(x86)} 'Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
    if (-not (Test-Path \$Vs)) {
        \$result.reason = 'vsdevcmd_not_found'
        return \$result
    }
    if (-not \$CudaToolkit.nvcc_found -or [string]::IsNullOrWhiteSpace(\$CudaToolkit.cuda_path)) {
        \$result.reason = 'cuda_toolkit_not_found'
        return \$result
    }

    Remove-Item -Recurse -Force \$BuildDir -ErrorAction SilentlyContinue
    \$CudaPath = [string]\$CudaToolkit.cuda_path
    \$CudaBin = Join-Path \$CudaPath 'bin'
    \$Nvcc = Join-Path \$CudaBin 'nvcc.exe'
    \$BatchPath = Join-Path \$env:TEMP 'tensorcore-windows-cuda-smoke.cmd'
    \$CallVs = 'call ' + (Cmd-Quote \$Vs) + ' -arch=x64 -host_arch=x64'
    \$SetCudaPath = 'set "CUDA_PATH=' + \$CudaPath + '"'
    \$SetPath = 'set "PATH=' + \$CudaBin + ';%PATH%"'
    \$Configure = 'cmake -S ' + (Cmd-Quote \$Repo) + ' -B ' + (Cmd-Quote \$BuildDir) +
        ' -G Ninja -DTC_ENABLE_METAL=OFF -DTC_ENABLE_CUDA=ON -DTC_BUILD_TESTS=ON' +
        ' -DTC_BUILD_BENCH=OFF -DTC_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release' +
        ' -DCUDAToolkit_ROOT=' + (Cmd-Quote \$CudaPath) +
        ' -DCMAKE_CUDA_COMPILER=' + (Cmd-Quote \$Nvcc) +
        ' -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl'
    \$BuildCommand = 'cmake --build ' + (Cmd-Quote \$BuildDir) + ' --parallel 4'
    \$CtestCommand = 'ctest --test-dir ' + (Cmd-Quote \$BuildDir) + ' --output-on-failure --timeout 240'
    \$steps = @(
        '@echo off',
        \$CallVs,
        'if errorlevel 1 exit /b %errorlevel%',
        \$SetCudaPath,
        \$SetPath,
        \$Configure,
        'if errorlevel 1 exit /b %errorlevel%',
        \$BuildCommand,
        'if errorlevel 1 exit /b %errorlevel%',
        \$CtestCommand
    )
    \$result.command_lines = \$steps
    Set-Content -LiteralPath \$BatchPath -Encoding ASCII -Value \$steps
    \$PreviousErrorActionPreference = \$ErrorActionPreference
    \$ErrorActionPreference = 'Continue'
    try {
        \$Output = @(& cmd.exe /c (Cmd-Quote \$BatchPath) 2>&1)
        \$Rc = \$LASTEXITCODE
    } finally {
        \$ErrorActionPreference = \$PreviousErrorActionPreference
        Remove-Item -Force -LiteralPath \$BatchPath -ErrorAction SilentlyContinue
    }
    \$Text = \$Output -join [Environment]::NewLine
    \$summary = Parse-CtestSummary \$Text
    \$result.rc = \$Rc
    \$result.output_tail = Tail-Text \$Text 6000
    \$result.tests_total = \$summary.tests_total
    \$result.tests_passed = \$summary.tests_passed
    \$result.tests_failed = \$summary.tests_failed
    \$result.tests_skipped = \$summary.tests_skipped
    \$result.cuda_gemm_passed = \$summary.cuda_gemm_passed
    \$result.ok = (
        \$Rc -eq 0 -and
        \$summary.tests_failed -eq 0 -and
        \$summary.cuda_gemm_passed
    )
    if (-not \$result.ok -and [string]::IsNullOrWhiteSpace(\$result.reason)) {
        \$result.reason = 'build_or_ctest_failed'
    }
    return \$result
}

Write-Output "[tensorcore/windows-cuda] host=\$env:COMPUTERNAME repo=\$Repo ref=\$Ref"
Update-Repo
\$FullHead = (git -C \$Repo rev-parse HEAD)
if (\$LASTEXITCODE -ne 0) { throw "git rev-parse HEAD failed with exit code \$LASTEXITCODE" }
\$FinalStatus = @(git -C \$Repo status --porcelain)
if (\$LASTEXITCODE -ne 0) { throw "git status failed with exit code \$LASTEXITCODE" }

\$NvidiaSmiPath = Find-CommandPath 'nvidia-smi'
\$Nvidia = [ordered]@{
    found = (\$null -ne \$NvidiaSmiPath)
    path = \$NvidiaSmiPath
    query_rc = \$null
    stderr_tail = ''
}
\$Devices = @()
\$ComputeApps = @()
\$Admission = [ordered]@{
    ok = \$false
    reason = 'nvidia_smi_not_found'
    allowed_process_max_memory_mib = \$AllowedProcessMaxMemoryMiB
    compute_app_count = 0
    blocked = @()
}

if (\$NvidiaSmiPath) {
    \$GpuOutput = @(& \$NvidiaSmiPath '--query-gpu=name,driver_version,memory.total,compute_cap' '--format=csv,noheader,nounits' 2>&1)
    \$Nvidia.query_rc = \$LASTEXITCODE
    if (\$LASTEXITCODE -eq 0) {
        \$Devices = @(Parse-GpuRows \$GpuOutput)
    } else {
        \$Nvidia.stderr_tail = Tail-Text (\$GpuOutput -join [Environment]::NewLine)
    }

    \$ProcessTableOutput = @(& \$NvidiaSmiPath 2>&1)
    \$ProcessTableRc = \$LASTEXITCODE
    \$VisibleProcessRows = @()
    if (\$ProcessTableRc -eq 0) {
        \$VisibleProcessRows = @(Parse-NvidiaSmiProcessTable \$ProcessTableOutput)
    }
    \$Nvidia.process_table_rc = \$ProcessTableRc
    \$Nvidia.visible_processes = \$VisibleProcessRows
    if (\$ProcessTableRc -ne 0) {
        \$Nvidia.process_table_stderr_tail = Tail-Text (\$ProcessTableOutput -join [Environment]::NewLine)
    }

    \$AppsOutput = @(& \$NvidiaSmiPath '--query-compute-apps=pid,process_name,used_gpu_memory' '--format=csv,noheader,nounits' 2>&1)
    \$AppsRc = \$LASTEXITCODE
    if (\$AppsRc -eq 0) {
        \$ComputeApps = @(Parse-ComputeApps \$AppsOutput)
        \$Blocked = @()
        foreach (\$app in \$ComputeApps) {
            \$mem = \$app.used_memory_mib
            if (\$null -eq \$mem -or \$mem -gt \$AllowedProcessMaxMemoryMiB) {
                \$Blocked += \$app
            }
        }
        \$OpaqueWddm = @(\$Blocked | Where-Object { Is-OpaqueWddmApp \$_ })
        if (\$Blocked.Count -gt 0 -and \$OpaqueWddm.Count -eq \$Blocked.Count -and \$VisibleProcessRows.Count -eq 0) {
            \$Admission.ok = \$true
            \$Admission.reason = 'ok_opaque_wddm_rows_no_visible_cuda_processes'
            \$Admission.ignored_opaque_wddm_app_count = \$OpaqueWddm.Count
            \$Admission.ignored_opaque_wddm = \$OpaqueWddm
            \$Admission.blocked = @()
        } else {
            \$Admission.ok = (\$Blocked.Count -eq 0)
            \$Admission.reason = if (\$Admission.ok) { 'ok' } else { 'blocked_cuda_compute_apps' }
            \$Admission.blocked = \$Blocked
        }
        \$Admission.compute_app_count = \$ComputeApps.Count
    } else {
        \$Admission.ok = \$false
        \$Admission.reason = 'nvidia_smi_compute_apps_failed'
        \$Admission.stderr_tail = Tail-Text (\$AppsOutput -join [Environment]::NewLine)
    }
}

\$NvccPath = Find-CommandPath 'nvcc'
\$ToolkitCandidates = @()
if (-not [string]::IsNullOrWhiteSpace(\$env:CUDA_PATH)) {
    \$ToolkitCandidates += \$env:CUDA_PATH
}
\$RepoParent = Split-Path -Parent \$Repo
if (-not [string]::IsNullOrWhiteSpace(\$RepoParent)) {
    \$ToolkitCandidates += (Join-Path \$RepoParent 'cuda-redist-12.6\toolkit')
}
\$ToolkitCandidates += (Join-Path \$env:USERPROFILE 'src\cuda-redist-12.6\toolkit')
foreach (\$Candidate in \$ToolkitCandidates) {
    if (\$NvccPath) { break }
    \$CandidateNvcc = Join-Path \$Candidate 'bin\nvcc.exe'
    if (Test-Path \$CandidateNvcc) {
        \$NvccPath = \$CandidateNvcc
        \$env:CUDA_PATH = \$Candidate
        \$env:Path = ((Join-Path \$Candidate 'bin') + ';' + \$env:Path)
    }
}
\$NvccVersion = @()
if (\$NvccPath) {
    \$NvccVersion = @(& \$NvccPath '--version' 2>&1)
}
\$ToolkitRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
\$ToolkitDirs = @()
if (Test-Path \$ToolkitRoot) {
    \$ToolkitDirs = @(Get-ChildItem \$ToolkitRoot -Directory | ForEach-Object { \$_.FullName })
}
\$ToolkitDirs = @(\$ToolkitDirs + (\$ToolkitCandidates | Where-Object { Test-Path \$_ }) | Select-Object -Unique)
\$CudaToolkit = [ordered]@{
    nvcc_found = (\$null -ne \$NvccPath)
    nvcc_path = \$NvccPath
    nvcc_version = (\$NvccVersion -join [Environment]::NewLine)
    cuda_path = \$env:CUDA_PATH
    toolkit_dirs = \$ToolkitDirs
}

\$DeviceCount = \$Devices.Count
\$RuntimeStatus = 'unavailable'
if (\$DeviceCount -gt 0 -and \$CudaToolkit.nvcc_found -and \$Admission.ok) {
    \$RuntimeStatus = 'ready'
} elseif (\$DeviceCount -gt 0 -and -not \$Admission.ok) {
    \$RuntimeStatus = 'admission_blocked'
} elseif (\$DeviceCount -gt 0) {
    \$RuntimeStatus = 'driver_only'
}

\$BuildSmokeResult = [ordered]@{ ran = \$false }
if (\$BuildSmoke -eq '1') {
    Write-Output '[tensorcore/windows-cuda] build smoke requested'
    \$BuildSmokeResult = Run-BuildSmoke \$RuntimeStatus \$CudaToolkit
}

\$Evidence = [ordered]@{
    schema = 'tensorcore.windows_cuda_probe.evidence.v1'
    schema_version = 1
    runtime_status = \$RuntimeStatus
    checked_at_unix = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    git_head = \$FullHead
    git_dirty = (\$FinalStatus.Count -ne 0)
    ref = \$Ref
    repo = \$Repo
    remote_url = \$RemoteUrl
    host = [ordered]@{
        computer_name = \$env:COMPUTERNAME
        user = \$env:USERNAME
        os = [System.Environment]::OSVersion.VersionString
    }
    nvidia_smi = \$Nvidia
    device_count = \$DeviceCount
    devices = \$Devices
    cuda_toolkit = \$CudaToolkit
    admission = \$Admission
    build_smoke = \$BuildSmokeResult
}

Write-Output (\$EvidenceMarker + (\$Evidence | ConvertTo-Json -Depth 8 -Compress))
PS
)

echo "[tensorcore/windows-cuda] ssh $windows_ssh"
ps_encode() {
    printf "%s" "$1" | iconv -f UTF-8 -t UTF-16LE | base64 | tr -d '\n'
}

remote_script_name="tensorcore-windows-cuda-probe.ps1"
upload_command="\$ProgressPreference = 'SilentlyContinue'; \$Path = Join-Path \$env:TEMP '$remote_script_name'; Set-Content -LiteralPath \$Path -Encoding UTF8 -Value ([Console]::In.ReadToEnd())"
run_command="\$ProgressPreference = 'SilentlyContinue'; \$Path = Join-Path \$env:TEMP '$remote_script_name'; & \$Path"
cleanup_command="\$ProgressPreference = 'SilentlyContinue'; \$Path = Join-Path \$env:TEMP '$remote_script_name'; Remove-Item -Force -LiteralPath \$Path -ErrorAction SilentlyContinue"

run_remote() {
    printf "%s\n" "$remote_command" |
        ssh "${ssh_opts[@]}" "$windows_ssh" powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass \
            -EncodedCommand "$(ps_encode "$upload_command")"
    local status=0
    ssh "${ssh_opts[@]}" "$windows_ssh" powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass \
        -EncodedCommand "$(ps_encode "$run_command")" || status=$?
    ssh "${ssh_opts[@]}" "$windows_ssh" powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass \
        -EncodedCommand "$(ps_encode "$cleanup_command")" >/dev/null 2>&1 || true
    return "$status"
}

if [[ -n "$windows_evidence_path" ]]; then
    evidence_tmp=$(mktemp "${TMPDIR:-/tmp}/tensorcore-windows-cuda.XXXXXX")
    set +e
    run_remote \
        | tee "$evidence_tmp" \
        | sed "/^${evidence_marker}/d"
    ssh_status=${PIPESTATUS[0]}
    set -e
    if [[ $ssh_status -ne 0 ]]; then
        exit "$ssh_status"
    fi
    evidence_line=$(grep "^${evidence_marker}" "$evidence_tmp" | tail -n 1 || true)
    rm -f "$evidence_tmp"
    if [[ -z "$evidence_line" ]]; then
        echo "[tensorcore/windows-cuda] missing evidence marker" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$windows_evidence_path")"
    printf "%s\n" "${evidence_line#"$evidence_marker"}" >"$windows_evidence_path"
    echo "[tensorcore/windows-cuda] evidence: $windows_evidence_path"
    echo "[tensorcore/windows-cuda] OK"
else
    run_remote | sed "/^${evidence_marker}/d"
    echo "[tensorcore/windows-cuda] OK"
fi
