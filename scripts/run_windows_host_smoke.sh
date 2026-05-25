#!/usr/bin/env bash
set -euo pipefail

windows_ssh="${TC_WINDOWS_SSH:-tsotchke@100.68.70.96}"
windows_repo="${TC_WINDOWS_REPO:-src/tensorcore}"
windows_remote_url="${TC_WINDOWS_REMOTE_URL:-https://github.com/tsotchke/tensorcore.git}"
windows_ref="${TC_WINDOWS_REF:-master}"
windows_reset="${TC_WINDOWS_RESET:-0}"
windows_install="${TC_WINDOWS_INSTALL:-0}"
windows_skip_python="${TC_WINDOWS_SKIP_PYTHON:-0}"
windows_no_smoke="${TC_WINDOWS_NO_SMOKE:-0}"
windows_timeout="${TC_WINDOWS_SSH_CONNECT_TIMEOUT:-10}"

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_windows_host_smoke.sh

Environment:
  TC_WINDOWS_SSH                  SSH target, default tsotchke@100.68.70.96
  TC_WINDOWS_SSH_KEY              Optional private key path
  TC_WINDOWS_REPO                 Remote repo path, default src/tensorcore
  TC_WINDOWS_REMOTE_URL           Remote clone URL
  TC_WINDOWS_REF                  Branch/ref to test, default master
  TC_WINDOWS_RESET=1              Hard-reset remote repo to origin/<ref>
  TC_WINDOWS_INSTALL=1            Let bootstrap install missing prerequisites
  TC_WINDOWS_SKIP_PYTHON=1        Skip Python smoke on first compiler bring-up
  TC_WINDOWS_NO_SMOKE=1           Only check/update host; do not run bootstrap
  TC_WINDOWS_SSH_CONNECT_TIMEOUT  SSH connect timeout seconds, default 10
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
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
    powershell @Args
    if (\$LASTEXITCODE -ne 0) { throw "bootstrap failed with exit code \$LASTEXITCODE" }
}

Write-Output "[tensorcore/windows-host] OK"
PS
)

echo "[tensorcore/windows-host] ssh $windows_ssh"
encoded_command=$(printf "%s" "$remote_command" | iconv -f UTF-8 -t UTF-16LE | base64 | tr -d '\n')
ssh "${ssh_opts[@]}" "$windows_ssh" powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand "$encoded_command"
