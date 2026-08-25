#!/usr/bin/env python3
"""Start the old-donkey qLLM teacher-logit precompute chain over SSH."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
import sys
import tempfile
import uuid
from typing import Any


SCHEMA = "tensorcore.qllm_olddonkey_precompute_chain.start.v1"
RESOURCE = "old-donkey:cuda3050"
DEFAULT_REPO_URL = "git@github.com:Tsotchke-Corporation/semiclassical_qllm.git"
DEFAULT_REPO_DIR = "/data/qllm/checkouts/semiclassical_qllm-tensorcore"
DEFAULT_REF = "claude/session-2026-05-08"
DEFAULT_PYTHON_BIN = "/data/venv/qllm/bin/python"
DEFAULT_SHARD_DIR = "/data/qllm/corpus/phase1.shards"
DEFAULT_RUN_DIR = "/data/qllm/runs"
DEFAULT_SESSION = "qllm-precompute-chain"
DEFAULT_SHARDS = (
    "ds07_shard02",
    "ds07_shard03",
    "ds03_shard01",
    "ds02_shard01",
    "ds06_shard01",
    "ds06_shard02",
)


def shq(value: str) -> str:
    return shlex.quote(value)


def render_remote_script(args: argparse.Namespace) -> str:
    require_clean = "1" if args.require_clean else "0"
    preflight_only = "1" if args.preflight_only else "0"
    shards = " ".join(shq(item) for item in args.shards)
    return f"""#!/bin/sh
set -eu

schema={shq(SCHEMA)}
repo_url={shq(args.repo_url)}
repo_dir={shq(args.repo_dir)}
ref={shq(args.ref)}
python_bin={shq(args.python_bin)}
shard_dir={shq(args.shard_dir)}
run_dir={shq(args.run_dir)}
session={shq(args.session)}
teacher={shq(args.teacher)}
batch={shq(str(args.batch))}
top_k={shq(str(args.top_k))}
seq={shq(str(args.seq))}
dtype={shq(args.dtype)}
device={shq(args.device)}
require_clean={require_clean}
preflight_only={preflight_only}
shards={shq(shards)}

case "$repo_dir" in
  "~") repo_dir="$HOME" ;;
  "~/"*) repo_dir="$HOME/${{repo_dir#\\~/}}" ;;
esac

json_escape() {{
  printf '%s' "$1" | sed 's/\\\\/\\\\\\\\/g; s/"/\\\\"/g'
}}

emit() {{
  ok="$1"
  reason="$2"
  pane_pid="${{3:-0}}"
  head="${{4:-}}"
  printf '{{"head":"%s","ok":%s,"pane_pid":%s,"reason":"%s","repo_dir":"%s","resource":"old-donkey:cuda3050","schema":"%s","session":"%s"}}\\n' \\
    "$(json_escape "$head")" "$ok" "$pane_pid" "$(json_escape "$reason")" \\
    "$(json_escape "$repo_dir")" "$(json_escape "$schema")" "$(json_escape "$session")"
}}

if [ ! -d "$repo_dir/.git" ]; then
  mkdir -p "$(dirname "$repo_dir")"
  if ! git clone --filter=blob:none --branch "$ref" "$repo_url" "$repo_dir"; then
    emit false clone_failed
    exit 1
  fi
else
  current_url=$(git -C "$repo_dir" remote get-url origin 2>/dev/null || true)
  if [ "$current_url" != "$repo_url" ]; then
    git -C "$repo_dir" remote set-url origin "$repo_url"
  fi
  git -C "$repo_dir" fetch origin "$ref" || git -C "$repo_dir" fetch origin
  git -C "$repo_dir" checkout "$ref"
  if git -C "$repo_dir" symbolic-ref -q HEAD >/dev/null 2>&1; then
    git -C "$repo_dir" pull --ff-only origin "$ref"
  fi
fi

head=$(git -C "$repo_dir" rev-parse HEAD)
dirty=$(git -C "$repo_dir" status --porcelain)
if [ "$require_clean" = "1" ] && [ -n "$dirty" ]; then
  printf '%s\\n' "$dirty" >&2
  emit false dirty_checkout 0 "$head"
  exit 1
fi
if [ ! -x "$python_bin" ]; then
  emit false python_not_found 0 "$head"
  exit 1
fi
if [ ! -f "$repo_dir/scripts/precompute_teacher_logits.py" ]; then
  emit false precompute_script_missing 0 "$head"
  exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
  emit false tmux_not_found 0 "$head"
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  emit false nvidia_smi_not_found 0 "$head"
  exit 1
fi
if [ ! -d "$shard_dir" ]; then
  emit false shard_dir_missing 0 "$head"
  exit 1
fi
for name in $shards; do
  if [ ! -r "$shard_dir/$name.bin" ]; then
    emit false "shard_missing:$name" 0 "$head"
    exit 1
  fi
done
if ! (cd "$repo_dir" && "$python_bin" - <<'PY'
import torch
import transformers
PY
); then
  emit false python_env_not_ready 0 "$head"
  exit 1
fi

if pgrep -af "precompute_teacher_logits.py.*--shard $shard_dir/" >/dev/null; then
  emit true already_live 0 "$head"
  exit 0
fi

if [ "$preflight_only" = "1" ]; then
  emit true preflight_ok 0 "$head"
  exit 0
fi

mkdir -p "$run_dir"
chain_script="$run_dir/tensorcore_olddonkey_precompute_chain.sh"
cat >"$chain_script" <<'CHAIN'
#!/usr/bin/env bash
set -uo pipefail

repo_dir="${{1:?repo_dir required}}"
python_bin="${{2:?python_bin required}}"
shard_dir="${{3:?shard_dir required}}"
log="${{4:?chain log required}}"
teacher="${{5:?teacher required}}"
batch="${{6:?batch required}}"
top_k="${{7:?top_k required}}"
seq="${{8:?seq required}}"
dtype="${{9:?dtype required}}"
device="${{10:?device required}}"
read -r -a shard_names <<<"${{11:?shards required}}"

export HF_HUB_OFFLINE="${{HF_HUB_OFFLINE:-1}}"
export TRANSFORMERS_OFFLINE="${{TRANSFORMERS_OFFLINE:-1}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"

echo "=== old-donkey tensorcore precompute chain start $(date -u +%FT%TZ) ===" | tee -a "$log"
echo "repo=$repo_dir" | tee -a "$log"
echo "queue: ${{shard_names[*]}}" | tee -a "$log"

while pgrep -f precompute_teacher_logits.py >/dev/null; do
  echo "[wait] in-flight precompute still running ($(date -u +%H:%M))" | tee -a "$log"
  sleep 300
done

for name in "${{shard_names[@]}}"; do
  shard="$shard_dir/$name.bin"
  shard_log="$(dirname "$log")/precompute-$name.log"
  echo "=== [resume] $name $(date -u +%FT%TZ) ===" | tee -a "$log"
  df -h "$(dirname "$shard_dir")" | tail -1 | tee -a "$shard_log"
  nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | tee -a "$shard_log"
  "$python_bin" "$repo_dir/scripts/precompute_teacher_logits.py" \\
    --shard "$shard" \\
    --teacher "$teacher" \\
    --top-k "$top_k" \\
    --seq "$seq" \\
    --batch "$batch" \\
    --device "$device" \\
    --dtype "$dtype" \\
    --resume 2>&1 | tee -a "$shard_log"
  rc=${{PIPESTATUS[0]}}
  echo "=== [resume] $name done rc=$rc $(date -u +%FT%TZ) ===" | tee -a "$log"
  if [[ $rc -ne 0 ]]; then
    echo "[resume] ABORT - $name failed" | tee -a "$log"
    exit "$rc"
  fi
done
echo "=== old-donkey tensorcore precompute chain done $(date -u +%FT%TZ) ===" | tee -a "$log"
CHAIN
chmod +x "$chain_script"

if tmux has-session -t "$session" 2>/dev/null; then
  tmux kill-session -t "$session"
fi

tmux new-session -d -s "$session" "bash '$chain_script' '$repo_dir' '$python_bin' '$shard_dir' '$run_dir/olddonkey_chain_resume.log' '$teacher' '$batch' '$top_k' '$seq' '$dtype' '$device' '$shards'"
sleep 2
pane_pid=$(tmux list-panes -t "$session" -F '#{{pane_pid}}' 2>/dev/null | head -n 1 || printf '0')
if pgrep -af "precompute_teacher_logits.py.*--shard $shard_dir/" >/dev/null; then
  emit true started "$pane_pid" "$head"
  exit 0
fi
if tmux has-session -t "$session" 2>/dev/null; then
  emit true started_pending "$pane_pid" "$head"
  exit 0
fi
emit false tmux_start_failed "$pane_pid" "$head"
exit 1
"""


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--repo-dir", default=DEFAULT_REPO_DIR)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--shard-dir", default=DEFAULT_SHARD_DIR)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--teacher", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shards", nargs="+", default=list(DEFAULT_SHARDS))
    parser.add_argument("--require-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--print-script", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def run_start(args: argparse.Namespace) -> dict[str, Any]:
    script = render_remote_script(args)
    if args.print_script:
        return {"schema": SCHEMA, "ok": True, "target": args.target, "script": script}
    try:
        proc = run_remote_script(args.target, script, timeout=args.timeout_sec)
    except subprocess.TimeoutExpired:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "start_timeout",
            "resource": RESOURCE,
            "target": args.target,
        }
    if proc.returncode != 0:
        payload = parse_remote_payload(proc.stdout)
        if payload is not None:
            invalid = validate_remote_payload(payload, target=args.target)
            if invalid is not None:
                invalid["rc"] = proc.returncode
                invalid["stderr_tail"] = proc.stderr.strip()[-1000:]
                invalid["stdout_tail"] = proc.stdout.strip()[-1000:]
                return invalid
            payload.setdefault("ok", False)
            payload.setdefault("target", args.target)
            payload["reason"] = classify_remote_failure(str(payload.get("reason") or ""), proc.stderr)
            payload["rc"] = proc.returncode
            payload["stderr_tail"] = proc.stderr.strip()[-1000:]
            payload["stdout_tail"] = proc.stdout.strip()[-1000:]
            return payload
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "remote_start_failed",
            "resource": RESOURCE,
            "target": args.target,
            "rc": proc.returncode,
            "stdout_tail": proc.stdout.strip()[-1000:],
            "stderr_tail": proc.stderr.strip()[-1000:],
        }
    payload = parse_remote_payload(proc.stdout)
    if payload is None:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "invalid_start_json",
            "resource": RESOURCE,
            "target": args.target,
            "stdout_tail": proc.stdout.strip()[-1000:],
            "stderr_tail": proc.stderr.strip()[-1000:],
        }
    invalid = validate_remote_payload(payload, target=args.target)
    if invalid is not None:
        invalid["stdout_tail"] = proc.stdout.strip()[-1000:]
        invalid["stderr_tail"] = proc.stderr.strip()[-1000:]
        return invalid
    payload.setdefault("target", args.target)
    return payload


def validate_remote_payload(payload: dict[str, Any], *, target: str) -> dict[str, Any] | None:
    if payload.get("schema") != SCHEMA:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "invalid_start_schema",
            "resource": RESOURCE,
            "target": target,
            "start_schema": payload.get("schema"),
        }
    if payload.get("resource") != RESOURCE:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "start_resource_mismatch",
            "target": target,
            "resource": RESOURCE,
            "start_resource": payload.get("resource"),
        }
    return None


def parse_remote_payload(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def classify_remote_failure(reason: str, stderr: str) -> str:
    if reason == "clone_failed":
        if "Permission denied (publickey)" in stderr:
            return "git_publickey_denied"
        if "could not read Username" in stderr or "terminal prompts disabled" in stderr:
            return "git_credentials_required"
    return reason or "remote_start_failed"


def run_remote_script(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
    local_name = ""
    remote_path = f"/tmp/tensorcore-qllm-chain-{os.getpid()}-{uuid.uuid4().hex}.sh"
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(script)
            local_name = handle.name
        upload = subprocess.run(
            ["scp", "-q", local_name, f"{target}:{remote_path}"],
            stdin=subprocess.DEVNULL,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if upload.returncode != 0:
            return run_remote_inline_script(target, script, timeout=timeout)
        try:
            return subprocess.run(
                ["ssh", target, "sh", remote_path],
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
                    ["ssh", target, "rm", "-f", remote_path],
                    stdin=subprocess.DEVNULL,
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=max(1.0, min(timeout, 10.0)),
                    check=False,
                )
            except subprocess.TimeoutExpired:
                pass
    finally:
        if local_name:
            try:
                os.unlink(local_name)
            except OSError:
                pass


def run_remote_inline_script(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
    remote_path = f"/tmp/tensorcore-qllm-chain-{os.getpid()}-{uuid.uuid4().hex}.sh"
    b64_path = f"{remote_path}.b64"
    quoted_path = shlex.quote(remote_path)
    quoted_b64_path = shlex.quote(b64_path)
    script_b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    init = subprocess.run(
        ["ssh", target, f"rm -f {quoted_path} {quoted_b64_path}; umask 077; : > {quoted_b64_path}"],
        stdin=subprocess.DEVNULL,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if init.returncode != 0:
        return init
    for offset in range(0, len(script_b64), 3000):
        chunk = script_b64[offset:offset + 3000]
        append = subprocess.run(
            ["ssh", target, f"printf '%s' {shlex.quote(chunk)} >> {quoted_b64_path}"],
            stdin=subprocess.DEVNULL,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if append.returncode != 0:
            return append
    remote_command = (
        f"(base64 -d {quoted_b64_path} > {quoted_path} 2>/dev/null || "
        f"base64 --decode {quoted_b64_path} > {quoted_path}) && sh {quoted_path}; "
        f"rc=$?; rm -f {quoted_path} {quoted_b64_path}; exit $rc"
    )
    return subprocess.run(
        ["ssh", target, remote_command],
        stdin=subprocess.DEVNULL,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = run_start(args)
    if args.print_script and not args.json:
        print(payload["script"], end="")
    elif args.json:
        json.dump(payload, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"{args.target}: ok={payload.get('ok')} reason={payload.get('reason')}")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
