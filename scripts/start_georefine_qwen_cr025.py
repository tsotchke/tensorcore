#!/usr/bin/env python3
"""Start the GeoRefine Qwen CR025 supervised run from a git checkout over SSH."""

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


SCHEMA = "tensorcore.georefine_qwen_cr025.start.v1"
RESOURCE = "cosbox:cuda3090"
DEFAULT_REPO_URL = "git@github.com:Tsotchke-Corporation/GeoRefineInternal.git"
DEFAULT_REPO_DIR = "~/src/georefine"
DEFAULT_REF = "master"
DEFAULT_RUN_DIR = (
    "/home/tyr/bytehole/georefine/runs/"
    "qwen3_5_0_8b_cr025_emb1024_fint4_cal4096_kd2048_t010_q005_20260525_043850"
)


def shq(value: str) -> str:
    return shlex.quote(value)


def render_remote_script(args: argparse.Namespace) -> str:
    require_clean = "1" if args.require_clean else "0"
    trust_remote_code = "1" if args.trust_remote_code else "0"
    live_agent = "1" if args.live_agent else "0"
    preflight_only = "1" if args.preflight_only else "0"
    return f"""#!/bin/sh
set -eu

schema={shq(SCHEMA)}
repo_url={shq(args.repo_url)}
repo_dir={shq(args.repo_dir)}
ref={shq(args.ref)}
run_dir={shq(args.run_dir)}
python_bin={shq(args.python_bin)}
start_log={shq(args.start_log)}
require_clean={require_clean}
trust_remote_code={trust_remote_code}
live_agent={live_agent}
preflight_only={preflight_only}
cal_text=/home/tyr/bytehole/georefine/qwen35_calibration/wikitext_merged_calibration_20260525.txt
eval_text=/home/tyr/projects/GeoRefineInternal/results/m2/qwen35_0_8b_investor_eval_wikitext_validation_20260522.txt

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
  pid="${{3:-0}}"
  head="${{4:-}}"
  printf '{{"head":"%s","ok":%s,"pid":%s,"reason":"%s","repo_dir":"%s","resource":"cosbox:cuda3090","run_dir":"%s","schema":"%s"}}\\n' \\
    "$(json_escape "$head")" "$ok" "$pid" "$(json_escape "$reason")" \\
    "$(json_escape "$repo_dir")" "$(json_escape "$run_dir")" "$(json_escape "$schema")"
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

case "$python_bin" in
  "") ;;
  /*) ;;
  *) python_bin="$repo_dir/$python_bin" ;;
esac
if [ -z "$python_bin" ] || [ ! -x "$python_bin" ]; then
  if [ -x "$repo_dir/bin/python" ]; then
    python_bin="$repo_dir/bin/python"
  elif [ -x "$repo_dir/.venv/bin/python" ]; then
    python_bin="$repo_dir/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    python_bin=$(command -v python3)
  else
    emit false python_not_found 0 "$head"
    exit 1
  fi
fi

if ! (cd "$repo_dir" && "$python_bin" - <<'PY'
import experiments.georefine.m2_compress
import experiments.georefine.m2_supervised_run
PY
); then
  emit false python_env_not_ready 0 "$head"
  exit 1
fi
if [ ! -r "$cal_text" ]; then
  emit false calibration_text_missing 0 "$head"
  exit 1
fi
if [ ! -r "$eval_text" ]; then
  emit false eval_text_missing 0 "$head"
  exit 1
fi

if "$python_bin" - "$run_dir" <<'PY'
import json
import math
import pathlib
import sys

def number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None

def first_number(data, *keys):
    for key in keys:
        value = number(data.get(key))
        if value is not None:
            return value
    return None

def target_kl_value(cert):
    achievement = cert.get("m2_target_kl_achievement")
    if isinstance(achievement, dict):
        for key in ("post_storage_kl_mean", "best_achieved"):
            value = number(achievement.get(key))
            if value is not None:
                return value
    return first_number(cert, "m2_kl_mean", "final_kl_mean", "kl_mean")

cert = pathlib.Path(sys.argv[1]) / "m2_certificate.json"
try:
    data = json.loads(cert.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if not isinstance(data, dict) or data.get("completed") is not True:
    raise SystemExit(1)
quality_gate = data.get("quality_gate")
if not isinstance(quality_gate, dict) or quality_gate.get("passed") is not True:
    raise SystemExit(1)
heldout_ppl = first_number(
    data,
    "ppl_compressed_eval",
    "final_heldout_ppl",
    "final_held_out_ppl",
    "heldout_ppl",
    "held_out_ppl",
    "eval_ppl",
    "ppl",
)
baseline_ppl = first_number(
    data,
    "ppl_baseline_eval",
    "baseline_heldout_ppl",
    "baseline_held_out_ppl",
    "baseline_eval_ppl",
)
ppl_delta = first_number(
    data,
    "ppl_delta_fraction_eval",
    "final_heldout_ppl_delta_fraction",
    "heldout_ppl_delta_fraction",
    "held_out_ppl_delta_fraction",
    "eval_ppl_delta_fraction",
)
if ppl_delta is None and heldout_ppl and baseline_ppl and baseline_ppl > 0:
    ppl_delta = (heldout_ppl - baseline_ppl) / baseline_ppl
size_ratio = first_number(data, "size_ratio", "final_size_ratio", "stored_size_ratio")
stored_size = first_number(data, "size_compressed_bytes", "final_stored_size_bytes", "stored_size_bytes")
original_size = first_number(data, "size_original_bytes")
if size_ratio is None and stored_size and original_size and original_size > 0:
    size_ratio = stored_size / original_size
target_kl = target_kl_value(data)
if (
    heldout_ppl is not None and heldout_ppl > 0
    and ppl_delta is not None and ppl_delta <= 0.05
    and size_ratio is not None and size_ratio <= 0.10
    and target_kl is not None and target_kl <= 0.10
):
    raise SystemExit(0)
raise SystemExit(1)
PY
then
  emit true already_complete 0 "$head"
  exit 0
fi

if "$python_bin" - "$run_dir" <<'PY'
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
status = run_dir / "m2_supervisor_status.json"
try:
    data = json.loads(status.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if data.get("state") != "running":
    raise SystemExit(1)
run_text = str(run_dir)
for key in ("compressor_pid", "supervisor_pid"):
    pid = data.get(key)
    if not isinstance(pid, int) or pid <= 1:
        continue
    cmdline = pathlib.Path("/proc") / str(pid) / "cmdline"
    try:
        text = cmdline.read_bytes().replace(b"\\0", b" ").decode("utf-8", "replace")
    except Exception:
        continue
    if run_text in text:
        raise SystemExit(0)
raise SystemExit(1)
PY
then
  emit true already_live 0 "$head"
  exit 0
fi

if [ "$preflight_only" = "1" ]; then
  emit true preflight_ok 0 "$head"
  exit 0
fi

mkdir -p "$run_dir"
if [ -z "$start_log" ]; then
  start_log="$run_dir/scheduler_start.log"
fi

set -- "$python_bin" -m experiments.georefine.m2_supervised_run \\
  --artifact-dir "$run_dir" \\
  --heartbeat-seconds 15 \\
  --target-size-ratio 0.10 \\
  --quality-floor 0.05 \\
  --max-state-age-seconds 240
if [ "$live_agent" != "1" ]; then
  set -- "$@" --no-live-agent
fi
set -- "$@" -- "$python_bin" -m experiments.georefine.m2_compress \\
  --model Qwen/Qwen3.5-0.8B \\
  --output-dir "$run_dir" \\
  --device cuda \\
  --dtype auto \\
  --cal-text-file "$cal_text" \\
  --cal-tokens 4096 \\
  --eval-text-file "$eval_text" \\
  --eval-tokens 1024 \\
  --compression-ratio 0.25 \\
  --min-rank 4 \\
  --streaming-asvd \\
  --resume \\
  --skip-if-exists \\
  --no-canonicalize \\
  --no-null-head-filter \\
  --factor-embeddings \\
  --embedding-rank 1024 \\
  --factor-inactive-linears \\
  --inactive-linear-prefix model.visual \\
  --inactive-linear-compression-ratio 0.20 \\
  --inactive-linear-min-params 262144 \\
  --target-kl 0.10 \\
  --target-kl-max-iterations 4 \\
  --target-kl-layers-per-iter 64 \\
  --target-kl-rank-growth 1.15 \\
  --target-kl-kd-steps 2048 \\
  --target-kl-kd-lr 3e-5 \\
  --target-kl-kd-temperature 2.0 \\
  --target-kl-kd-hidden-state-weight 0.03 \\
  --target-kl-kd-chunk-size 64 \\
  --quantize-factors int4 \\
  --quantize-residual-tensors none \\
  --storage-quantization-recovery auto \\
  --residual-quantization-preflight off \\
  --max-size-ratio 0.10 \\
  --fail-on-size-gate \\
  --quality-floor 0.05 \\
  --quality-floor-retries 0 \\
  --fail-on-quality-gate \\
  --require-held-out-quality \\
  --fail-on-verdict LOSSY \\
  --verbose
if [ "$trust_remote_code" = "1" ]; then
  set -- "$@" --trust-remote-code
fi

cd "$repo_dir"
nohup "$@" >>"$start_log" 2>&1 &
pid=$!
sleep 2
if kill -0 "$pid" >/dev/null 2>&1; then
  emit true started "$pid" "$head"
  exit 0
fi
emit false launch_failed "$pid" "$head"
exit 1
"""


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--repo-dir", default=DEFAULT_REPO_DIR)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--python-bin", default="")
    parser.add_argument("--start-log", default="")
    parser.add_argument("--require-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--live-agent", action=argparse.BooleanOptionalAction, default=False)
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
    remote_path = f"/tmp/tensorcore-georefine-cr025-{os.getpid()}-{uuid.uuid4().hex}.sh"
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
    remote_path = f"/tmp/tensorcore-georefine-cr025-{os.getpid()}-{uuid.uuid4().hex}.sh"
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
