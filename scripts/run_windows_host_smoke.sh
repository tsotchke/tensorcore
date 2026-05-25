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
windows_install="${TC_WINDOWS_INSTALL:-0}"
windows_skip_python="${TC_WINDOWS_SKIP_PYTHON:-0}"
windows_no_smoke="${TC_WINDOWS_NO_SMOKE:-0}"
windows_timeout="${TC_WINDOWS_SSH_CONNECT_TIMEOUT:-10}"
windows_evidence_path="${TC_WINDOWS_EVIDENCE_PATH:-}"
evidence_marker="__TENSORCORE_WINDOWS_HOST_EVIDENCE__"

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_windows_host_smoke.sh

Environment:
  TC_WINDOWS_CONFIG               Optional local env file, default ~/.config/tensorcore/windows-host.env
  TC_WINDOWS_SSH                  SSH target. Required via env or TC_WINDOWS_CONFIG.
  TC_WINDOWS_SSH_KEY              Optional private key path
  TC_WINDOWS_REPO                 Remote repo path, default src/tensorcore
  TC_WINDOWS_REMOTE_URL           Remote clone URL
  TC_WINDOWS_REF                  Branch/ref to test, default master
  TC_WINDOWS_RESET=1              Hard-reset remote repo to origin/<ref>
  TC_WINDOWS_INSTALL=1            Let bootstrap install missing prerequisites
  TC_WINDOWS_SKIP_PYTHON=1        Skip Python smoke on first compiler bring-up
  TC_WINDOWS_NO_SMOKE=1           Only check/update host; do not run bootstrap
  TC_WINDOWS_EVIDENCE_PATH        Optional local JSON evidence output path
  TC_WINDOWS_SSH_CONNECT_TIMEOUT  SSH connect timeout seconds, default 10
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ -z "$windows_ssh" ]]; then
    echo "tensorcore/windows-host: TC_WINDOWS_SSH is required via env or $windows_config" >&2
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
install_q=$(ps_quote "$windows_install")
skip_python_q=$(ps_quote "$windows_skip_python")
no_smoke_q=$(ps_quote "$windows_no_smoke")
emit_evidence_q=$(ps_quote "$([[ -n "$windows_evidence_path" ]] && echo 1 || echo 0)")
marker_q=$(ps_quote "$evidence_marker")

remote_command=$(cat <<PS
\$ErrorActionPreference = 'Stop'
\$ProgressPreference = 'SilentlyContinue'
Set-StrictMode -Version 3.0

\$Repo = $repo_q
\$RemoteUrl = $url_q
\$Ref = $ref_q
\$Reset = $reset_q
\$Install = $install_q
\$SkipPython = $skip_python_q
\$NoSmoke = $no_smoke_q
\$EmitEvidence = $emit_evidence_q
\$EvidenceMarker = $marker_q

Write-Output "[tensorcore/windows-host] host=\$env:COMPUTERNAME repo=\$Repo ref=\$Ref"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'git not found on PATH. Install Git for Windows first.'
}

\$Parent = Split-Path -Parent \$Repo
if (-not [string]::IsNullOrWhiteSpace(\$Parent) -and -not (Test-Path \$Parent)) {
    New-Item -ItemType Directory -Force -Path \$Parent | Out-Null
}

\$GitDir = Join-Path \$Repo '.git'
if (-not (Test-Path \$GitDir)) {
    Write-Output "[tensorcore/windows-host] clone \$RemoteUrl -> \$Repo"
    git clone --branch \$Ref \$RemoteUrl \$Repo
    if (\$LASTEXITCODE -ne 0) { throw "git clone failed with exit code \$LASTEXITCODE" }
} else {
    Write-Output "[tensorcore/windows-host] update existing checkout"
    git -C \$Repo fetch --prune origin \$Ref
    if (\$LASTEXITCODE -ne 0) { throw "git fetch failed with exit code \$LASTEXITCODE" }
    if (\$Reset -eq '1') {
        Write-Output "[tensorcore/windows-host] reset --hard origin/\$Ref"
        git -C \$Repo reset --hard "origin/\$Ref"
        if (\$LASTEXITCODE -ne 0) { throw "git reset failed with exit code \$LASTEXITCODE" }
    } else {
        git -C \$Repo checkout \$Ref
        if (\$LASTEXITCODE -ne 0) { throw "git checkout failed with exit code \$LASTEXITCODE" }
        git -C \$Repo pull --ff-only origin \$Ref
        if (\$LASTEXITCODE -ne 0) { throw "git pull --ff-only failed with exit code \$LASTEXITCODE" }
    }
}

\$Head = (git -C \$Repo rev-parse --short HEAD)
if (\$LASTEXITCODE -ne 0) { throw "git rev-parse failed with exit code \$LASTEXITCODE" }
\$FullHead = (git -C \$Repo rev-parse HEAD)
if (\$LASTEXITCODE -ne 0) { throw "git rev-parse HEAD failed with exit code \$LASTEXITCODE" }
\$Status = (git -C \$Repo status --short --branch)
Write-Output "[tensorcore/windows-host] head=\$Head"
Write-Output \$Status

if (\$NoSmoke -ne '1') {
    \$Bootstrap = Join-Path \$Repo 'scripts\bootstrap_windows_cpu.ps1'
    if (-not (Test-Path \$Bootstrap)) { throw "bootstrap script not found: \$Bootstrap" }
    \$Args = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', \$Bootstrap, '-RepoDir', \$Repo)
    if (\$Install -eq '1') { \$Args += '-Install' }
    if (\$SkipPython -eq '1') { \$Args += '-SkipPython' }
    Write-Output "[tensorcore/windows-host] run bootstrap_windows_cpu.ps1"
    \$EmptyInput = New-TemporaryFile
    try {
        \$BootstrapProcess = Start-Process -FilePath powershell -ArgumentList \$Args -NoNewWindow -Wait -PassThru -RedirectStandardInput \$EmptyInput
        if (\$BootstrapProcess.ExitCode -ne 0) {
            throw "bootstrap failed with exit code \$(\$BootstrapProcess.ExitCode)"
        }
    } finally {
        Remove-Item -Force \$EmptyInput -ErrorAction SilentlyContinue
    }
}

if (\$EmitEvidence -eq '1') {
    \$FinalStatus = @(git -C \$Repo status --porcelain)
    if (\$LASTEXITCODE -ne 0) { throw "git status failed with exit code \$LASTEXITCODE" }
    \$Evidence = [ordered]@{
        schema = 'tensorcore.windows_host_smoke.evidence.v1'
        schema_version = 1
        runtime_status = if (\$NoSmoke -eq '1') { 'skipped_no_smoke' } else { 'passed' }
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
        update = [ordered]@{
            reset = (\$Reset -eq '1')
        }
        bootstrap = [ordered]@{
            ran = (\$NoSmoke -ne '1')
            install_requested = (\$Install -eq '1')
            skip_python = (\$SkipPython -eq '1')
        }
    }
    Write-Output (\$EvidenceMarker + (\$Evidence | ConvertTo-Json -Depth 6 -Compress))
}
PS
)

echo "[tensorcore/windows-host] ssh $windows_ssh"
ps_encode() {
    printf "%s" "$1" | iconv -f UTF-8 -t UTF-16LE | base64 | tr -d '\n'
}

remote_script_name="tensorcore-windows-host-smoke.ps1"
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
    evidence_tmp=$(mktemp "${TMPDIR:-/tmp}/tensorcore-windows-host.XXXXXX")
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
        echo "[tensorcore/windows-host] missing evidence marker" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$windows_evidence_path")"
    printf "%s\n" "${evidence_line#"$evidence_marker"}" >"$windows_evidence_path"
    echo "[tensorcore/windows-host] evidence: $windows_evidence_path"
    echo "[tensorcore/windows-host] OK"
else
    run_remote
    echo "[tensorcore/windows-host] OK"
fi
