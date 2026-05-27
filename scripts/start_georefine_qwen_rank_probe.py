#!/usr/bin/env python3
"""Start a scheduler-owned GeoRefine Qwen rank-search probe over SSH."""

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


SCHEMA = "tensorcore.georefine_qwen_rank_probe.start.v1"
DEFAULT_REF = os.environ.get("TC_GEOREFINE_REF", "")


def shq(value: str) -> str:
    return shlex.quote(value)


def render_remote_script(args: argparse.Namespace) -> str:
    require_clean = "1" if args.require_clean else "0"
    trust_remote_code = "1" if args.trust_remote_code else "0"
    live_agent = "1" if args.live_agent else "0"
    preflight_only = "1" if args.preflight_only else "0"
    sync_repo = "1" if args.sync_repo else "0"
    chat_verify_fail_on_drift = "1" if args.chat_verify_fail_on_drift else "0"
    return f"""#!/bin/sh
set -eu

schema={shq(SCHEMA)}
resource={shq(args.resource)}
worker_resource={shq(args.worker_resource)}
authority_lease_id={shq(args.authority_lease_id)}
authority_owner={shq(args.authority_owner)}
repo_url={shq(args.repo_url)}
repo_dir={shq(args.repo_dir)}
qllm_repo_dir={shq(args.qllm_repo_dir)}
lease_helper={shq(args.lease_helper)}
ref={shq(args.ref)}
run_dir={shq(args.run_dir)}
output_root={shq(args.output_root)}
evidence_root={shq(args.evidence_root)}
python_bin={shq(args.python_bin)}
start_log={shq(args.start_log)}
require_clean={require_clean}
trust_remote_code={trust_remote_code}
live_agent={live_agent}
preflight_only={preflight_only}
sync_repo={sync_repo}
chat_verify_fail_on_drift={chat_verify_fail_on_drift}
cal_text={shq(args.cal_text)}
eval_text={shq(args.eval_text)}
cal_images_dir={shq(args.cal_images_dir)}
cal_preset={shq(args.cal_preset)}
exclude_layers={shq(args.exclude_layers)}
embedding_rank={int(args.embedding_rank)}
compression_ratio={float(args.compression_ratio)}
target_kl={float(args.target_kl)}
target_kl_kd_steps={int(args.target_kl_kd_steps)}
max_size_ratio={float(args.max_size_ratio)}
quality_floor={float(args.quality_floor)}
target_kl_max_iterations={int(args.target_kl_max_iterations)}
target_kl_layers_per_iter={int(args.target_kl_layers_per_iter)}
target_kl_rank_growth={float(args.target_kl_rank_growth)}
target_kl_kd_lr={float(args.target_kl_kd_lr)}
target_kl_kd_temperature={float(args.target_kl_kd_temperature)}
target_kl_kd_hidden_state_weight={float(args.target_kl_kd_hidden_state_weight)}
target_kl_kd_chunk_size={int(args.target_kl_kd_chunk_size)}
model={shq(args.model)}
device={shq(args.device)}
dtype={shq(args.dtype)}
cal_tokens={int(args.cal_tokens)}
eval_tokens={int(args.eval_tokens)}
min_rank={int(args.min_rank)}
heartbeat_seconds={int(args.heartbeat_seconds)}
max_state_age_seconds={int(args.max_state_age_seconds)}
quality_floor_retries={int(args.quality_floor_retries)}
lease_ttl_sec={int(args.lease_ttl_sec)}
run_target={shq(args.run_target)}
quantize_factors={shq(args.quantize_factors)}
chat_verify_phases={shq(args.chat_verify_phases)}
chat_verify_max_kl={float(args.chat_verify_max_kl)}
chat_verify_max_l1={float(args.chat_verify_max_l1)}
chat_verify_max_base_rank={int(args.chat_verify_max_base_rank)}

case "$repo_dir" in
  "~") repo_dir="$HOME" ;;
  "~/"*) repo_dir="$HOME/${{repo_dir#\\~/}}" ;;
esac
case "$qllm_repo_dir" in
  "~") qllm_repo_dir="$HOME" ;;
  "~/"*) qllm_repo_dir="$HOME/${{qllm_repo_dir#\\~/}}" ;;
esac

json_escape() {{
  printf '%s' "$1" | sed 's/\\\\/\\\\\\\\/g; s/"/\\\\"/g'
}}

emit() {{
  ok="$1"
  reason="$2"
  pid="${{3:-0}}"
  head="${{4:-}}"
  printf '{{"head":"%s","ok":%s,"pid":%s,"reason":"%s","repo_dir":"%s","resource":"%s","run_dir":"%s","schema":"%s"}}\\n' \\
    "$(json_escape "$head")" "$ok" "$pid" "$(json_escape "$reason")" \\
    "$(json_escape "$repo_dir")" "$(json_escape "$resource")" "$(json_escape "$run_dir")" "$(json_escape "$schema")"
}}

if [ "$sync_repo" = "1" ] && [ ! -d "$repo_dir/.git" ]; then
  mkdir -p "$(dirname "$repo_dir")"
  if ! git clone --filter=blob:none --branch "$ref" "$repo_url" "$repo_dir"; then
    emit false clone_failed
    exit 1
  fi
elif [ "$sync_repo" = "1" ]; then
  current_url=$(git -C "$repo_dir" remote get-url origin 2>/dev/null || true)
  if [ "$current_url" != "$repo_url" ]; then
    git -C "$repo_dir" remote set-url origin "$repo_url"
  fi
  git -C "$repo_dir" fetch origin "$ref" || git -C "$repo_dir" fetch origin
  git -C "$repo_dir" checkout "$ref"
  if git -C "$repo_dir" symbolic-ref -q HEAD >/dev/null 2>&1; then
    git -C "$repo_dir" pull --ff-only origin "$ref"
  fi
elif [ ! -d "$repo_dir/.git" ]; then
  emit false repo_dir_missing_with_no_sync
  exit 1
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
if [ -z "$run_dir" ]; then
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  rank_label=$(printf '%04d' "$embedding_rank")
  ratio_label=$(printf '%s' "$compression_ratio" | tr -d '.')
  tkl_label=$(printf '%s' "$target_kl" | tr -d '.')
  size_label=$(printf '%s' "$max_size_ratio" | tr -d '.')
  run_dir="${{output_root}}/qwen35_0_8b_cr${{ratio_label}}_emb${{rank_label}}_tkl${{tkl_label}}_size${{size_label}}_${{stamp}}"
fi
if [ -z "$worker_resource" ]; then
  emit false worker_resource_missing 0 "$head"
  exit 1
fi
if [ -z "$authority_owner" ]; then
  authority_owner=georefine:qwen-rank-probe
fi
if [ ! -r "$cal_text" ]; then
  emit false calibration_text_missing 0 "$head"
  exit 1
fi
if [ ! -r "$eval_text" ]; then
  emit false eval_text_missing 0 "$head"
  exit 1
fi
if [ -n "$cal_images_dir" ] && [ ! -d "$cal_images_dir" ]; then
  emit false calibration_images_missing 0 "$head"
  exit 1
fi

if "$python_bin" - "$run_dir" "$max_size_ratio" "$quality_floor" "$target_kl" <<'PY'
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
max_size_ratio = float(sys.argv[2])
quality_floor = float(sys.argv[3])
target_kl_threshold = float(sys.argv[4])
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
    and ppl_delta is not None and ppl_delta <= quality_floor
    and size_ratio is not None and size_ratio <= max_size_ratio
    and target_kl is not None and target_kl <= target_kl_threshold
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

lease_wrapper="$qllm_repo_dir/scripts/qllm_resource_lease.py"
reconciler_script="$qllm_repo_dir/scripts/georefine_live_control_reconciler.py"
finalizer_script="$qllm_repo_dir/scripts/finalize_georefine_artifact.py"
run_intent_json="$run_dir/run_intent.json"
final_manifest_output="$evidence_root/$(basename "$run_dir").trusted_final_manifest.json"
if [ ! -r "$lease_wrapper" ]; then
  emit false qllm_resource_lease_missing 0 "$head"
  exit 1
fi
if [ ! -r "$reconciler_script" ]; then
  emit false qllm_reconciler_missing 0 "$head"
  exit 1
fi
if [ ! -r "$finalizer_script" ]; then
  emit false qllm_finalizer_missing 0 "$head"
  exit 1
fi
if [ -z "$authority_lease_id" ]; then
  emit false authority_lease_id_missing 0 "$head"
  exit 1
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
  --heartbeat-seconds "$heartbeat_seconds" \\
  --target-size-ratio "$max_size_ratio" \\
  --quality-floor "$quality_floor" \\
  --max-state-age-seconds "$max_state_age_seconds"
if [ "$live_agent" != "1" ]; then
  set -- "$@" --no-live-agent
fi
set -- "$@" -- "$python_bin" -m experiments.georefine.m2_compress \\
  --model "$model" \\
  --output-dir "$run_dir" \\
  --device "$device" \\
  --dtype "$dtype" \\
  --cal-text-file "$cal_text" \\
  --cal-tokens "$cal_tokens" \\
  --eval-text-file "$eval_text" \\
  --eval-tokens "$eval_tokens" \\
  --cal-preset "$cal_preset" \\
  --compression-ratio "$compression_ratio" \\
  --min-rank "$min_rank" \\
  --streaming-asvd \\
  --resume \\
  --skip-if-exists \\
  --no-canonicalize \\
  --no-null-head-filter \\
  --factor-embeddings \\
  --embedding-rank "$embedding_rank" \\
  --target-kl "$target_kl" \\
  --target-kl-max-iterations "$target_kl_max_iterations" \\
  --target-kl-layers-per-iter "$target_kl_layers_per_iter" \\
  --target-kl-rank-growth "$target_kl_rank_growth" \\
  --target-kl-kd-steps "$target_kl_kd_steps" \\
  --target-kl-kd-lr "$target_kl_kd_lr" \\
  --target-kl-kd-temperature "$target_kl_kd_temperature" \\
  --target-kl-kd-hidden-state-weight "$target_kl_kd_hidden_state_weight" \\
  --target-kl-kd-chunk-size "$target_kl_kd_chunk_size" \\
  --quantize-factors "$quantize_factors" \\
  --quantize-residual-tensors none \\
  --storage-quantization-recovery auto \\
  --residual-quantization-preflight off \\
  --max-size-ratio "$max_size_ratio" \\
  --fail-on-size-gate \\
  --quality-floor "$quality_floor" \\
  --quality-floor-retries "$quality_floor_retries" \\
  --fail-on-quality-gate \\
  --require-held-out-quality \\
  --fail-on-verdict LOSSY \\
  --verbose
if [ -n "$cal_images_dir" ]; then
  set -- "$@" --cal-images-dir "$cal_images_dir"
fi
if [ -n "$exclude_layers" ]; then
  set -- "$@" --exclude-layers "$exclude_layers"
fi
if [ "$chat_verify_fail_on_drift" = "1" ]; then
  set -- "$@" \\
    --chat-verify-fail-on-drift \\
    --chat-verify-phases "$chat_verify_phases" \\
    --chat-verify-max-kl "$chat_verify_max_kl" \\
    --chat-verify-max-l1 "$chat_verify_max_l1" \\
    --chat-verify-max-base-rank "$chat_verify_max_base_rank"
fi
if [ "$trust_remote_code" = "1" ]; then
  set -- "$@" --trust-remote-code
fi
metadata_json=$(printf '{{"surface":"tensorcore_scheduler","service":"georefine-qwen-rank-probe","run_dir":"%s"}}' "$(json_escape "$run_dir")")
set -- "$python_bin" "$lease_wrapper"
if [ -n "$lease_helper" ]; then
  if [ ! -r "$lease_helper" ]; then
    emit false qllm_lease_helper_missing 0 "$head"
    exit 1
  fi
  set -- "$@" --helper "$lease_helper"
fi
set -- "$@" \\
  --resource "$worker_resource" \\
  --owner "$authority_owner" \\
  --ttl-sec "$lease_ttl_sec" \\
  --heartbeat-failure-policy terminate \\
  --worker-lease-mode mirror \\
  --verify-gpu-identity \\
  --exclusive-cuda 1 \\
  --metadata-json "$metadata_json" \\
  --run-intent-json "$run_intent_json" \\
  --artifact-dir "$run_dir" \\
  --controller-mode observe \\
  --allowed-mutator tensorcore-georefine-reconciler \\
  --run-target "$run_target" \\
  --authority-resource "$resource" \\
  --authority-lease-id "$authority_lease_id" \\
  --authority-owner "$authority_owner" \\
  --authority-source tensorcore-scheduler \\
  --require-substrate-contract \\
  --finalizer-script "$finalizer_script" \\
  --finalizer-python "$python_bin" \\
  --final-manifest-evidence-root "$evidence_root" \\
  --final-manifest-output "$final_manifest_output" \\
  --finalizer-max-size-ratio "$max_size_ratio" \\
  --finalizer-max-ppl-delta "$quality_floor" \\
  --finalizer-max-target-kl "$target_kl" \\
  --finalizer-failure-policy fail \\
  --reconciler-script "$reconciler_script" \\
  --reconciler-python "$python_bin" \\
  --reconciler-actor tensorcore-georefine-reconciler \\
  --reconciler-require-active-lease \\
  --reconciler-failure-policy terminate \\
  -- \\
  "$@"

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
    parser.add_argument("--resource", default=os.environ.get("TC_GEOREFINE_RESOURCE", ""))
    parser.add_argument("--worker-resource", default=os.environ.get("TC_GEOREFINE_WORKER_RESOURCE", ""))
    parser.add_argument("--authority-lease-id", default="")
    parser.add_argument("--authority-owner", default=os.environ.get("TC_GEOREFINE_AUTHORITY_OWNER", ""))
    parser.add_argument("--repo-url", default=os.environ.get("TC_GEOREFINE_REPO_URL", ""))
    parser.add_argument("--repo-dir", default=os.environ.get("TC_GEOREFINE_REPO_DIR", ""))
    parser.add_argument("--qllm-repo-dir", default=os.environ.get("TC_GEOREFINE_QLLM_REPO_DIR", ""))
    parser.add_argument("--lease-helper", default=os.environ.get("TC_GEOREFINE_LEASE_HELPER", ""))
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--output-root", default=os.environ.get("TC_GEOREFINE_OUTPUT_ROOT", ""))
    parser.add_argument("--evidence-root", default=os.environ.get("TC_GEOREFINE_EVIDENCE_ROOT", ""))
    parser.add_argument("--python-bin", default="")
    parser.add_argument("--start-log", default="")
    parser.add_argument("--cal-text", default=os.environ.get("TC_GEOREFINE_CAL_TEXT", ""))
    parser.add_argument("--eval-text", default=os.environ.get("TC_GEOREFINE_EVAL_TEXT", ""))
    parser.add_argument("--embedding-rank", type=int, default=1024)
    parser.add_argument("--compression-ratio", type=float, default=0.70)
    parser.add_argument("--target-kl", type=float, default=0.80)
    parser.add_argument("--target-kl-kd-steps", type=int, default=2048)
    parser.add_argument("--target-kl-kd-lr", type=float, default=3e-5)
    parser.add_argument("--target-kl-kd-temperature", type=float, default=2.0)
    parser.add_argument("--target-kl-kd-hidden-state-weight", type=float, default=0.03)
    parser.add_argument("--target-kl-kd-chunk-size", type=int, default=64)
    parser.add_argument("--target-kl-max-iterations", type=int, default=4)
    parser.add_argument("--target-kl-layers-per-iter", type=int, default=64)
    parser.add_argument("--target-kl-rank-growth", type=float, default=1.15)
    parser.add_argument("--max-size-ratio", type=float, default=0.30)
    parser.add_argument("--quality-floor", type=float, default=0.05)
    parser.add_argument("--model", default=os.environ.get("TC_GEOREFINE_MODEL", ""))
    parser.add_argument("--device", default=os.environ.get("TC_GEOREFINE_DEVICE", ""))
    parser.add_argument("--dtype", default=os.environ.get("TC_GEOREFINE_DTYPE", ""))
    parser.add_argument("--cal-tokens", type=int, default=int(os.environ.get("TC_GEOREFINE_CAL_TOKENS", "4096")))
    parser.add_argument("--eval-tokens", type=int, default=int(os.environ.get("TC_GEOREFINE_EVAL_TOKENS", "1024")))
    parser.add_argument("--min-rank", type=int, default=int(os.environ.get("TC_GEOREFINE_MIN_RANK", "4")))
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=int(os.environ.get("TC_GEOREFINE_HEARTBEAT_SECONDS", "15")),
    )
    parser.add_argument(
        "--max-state-age-seconds",
        type=int,
        default=int(os.environ.get("TC_GEOREFINE_MAX_STATE_AGE_SECONDS", "240")),
    )
    parser.add_argument(
        "--quality-floor-retries",
        type=int,
        default=int(os.environ.get("TC_GEOREFINE_QUALITY_FLOOR_RETRIES", "0")),
    )
    parser.add_argument("--lease-ttl-sec", type=int, default=int(os.environ.get("TC_GEOREFINE_LEASE_TTL_SEC", "21600")))
    parser.add_argument("--run-target", default=os.environ.get("TC_GEOREFINE_RUN_TARGET", ""))
    parser.add_argument("--cal-images-dir", default="")
    parser.add_argument("--cal-preset", default="default")
    parser.add_argument("--exclude-layers", default="")
    parser.add_argument(
        "--quantize-factors",
        choices=("none", "int8", "int4", "int2"),
        default="int4",
    )
    parser.add_argument(
        "--chat-verify-fail-on-drift",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--chat-verify-phases", default="all")
    parser.add_argument("--chat-verify-max-kl", type=float, default=1.0)
    parser.add_argument("--chat-verify-max-l1", type=float, default=1.0)
    parser.add_argument("--chat-verify-max-base-rank", type=int, default=5)
    parser.add_argument("--require-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sync-repo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--live-agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--print-script", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    required_strings = {
        "--resource": args.resource,
        "--worker-resource": args.worker_resource,
        "--authority-owner": args.authority_owner,
        "--repo-dir": args.repo_dir,
        "--qllm-repo-dir": args.qllm_repo_dir,
        "--ref": args.ref,
        "--evidence-root": args.evidence_root,
        "--cal-text": args.cal_text,
        "--eval-text": args.eval_text,
        "--model": args.model,
        "--device": args.device,
        "--dtype": args.dtype,
        "--run-target": args.run_target,
    }
    if args.sync_repo:
        required_strings["--repo-url"] = args.repo_url
    if not args.run_dir:
        required_strings["--output-root"] = args.output_root
    for flag, value in required_strings.items():
        if not str(value or "").strip():
            parser.error(f"{flag} is required; pass it explicitly or set the matching TC_GEOREFINE_* environment variable")
    if args.embedding_rank < 1:
        parser.error("--embedding-rank must be positive")
    if args.compression_ratio <= 0 or args.compression_ratio > 1:
        parser.error("--compression-ratio must be in (0, 1]")
    if args.max_size_ratio <= 0 or args.max_size_ratio > 1:
        parser.error("--max-size-ratio must be in (0, 1]")
    if args.target_kl <= 0:
        parser.error("--target-kl must be positive")
    if args.target_kl_kd_steps < 0:
        parser.error("--target-kl-kd-steps must be >= 0")
    if args.target_kl_kd_lr <= 0:
        parser.error("--target-kl-kd-lr must be positive")
    if args.target_kl_kd_temperature <= 0:
        parser.error("--target-kl-kd-temperature must be positive")
    if args.target_kl_kd_hidden_state_weight < 0:
        parser.error("--target-kl-kd-hidden-state-weight must be >= 0")
    if args.target_kl_kd_chunk_size < 1:
        parser.error("--target-kl-kd-chunk-size must be positive")
    if args.target_kl_max_iterations < 0:
        parser.error("--target-kl-max-iterations must be >= 0")
    if args.target_kl_layers_per_iter < 1:
        parser.error("--target-kl-layers-per-iter must be positive")
    if args.target_kl_rank_growth <= 1.0:
        parser.error("--target-kl-rank-growth must be > 1")
    if args.chat_verify_max_kl <= 0:
        parser.error("--chat-verify-max-kl must be positive")
    if args.chat_verify_max_l1 <= 0:
        parser.error("--chat-verify-max-l1 must be positive")
    if args.chat_verify_max_base_rank < 1:
        parser.error("--chat-verify-max-base-rank must be >= 1")
    if args.cal_tokens < 1:
        parser.error("--cal-tokens must be positive")
    if args.eval_tokens < 1:
        parser.error("--eval-tokens must be positive")
    if args.min_rank < 1:
        parser.error("--min-rank must be positive")
    if args.heartbeat_seconds < 1:
        parser.error("--heartbeat-seconds must be positive")
    if args.max_state_age_seconds < 1:
        parser.error("--max-state-age-seconds must be positive")
    if args.quality_floor_retries < 0:
        parser.error("--quality-floor-retries must be >= 0")
    if args.lease_ttl_sec < 1:
        parser.error("--lease-ttl-sec must be positive")
    return args


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
            "resource": args.resource,
            "target": args.target,
        }
    if proc.returncode != 0:
        payload = parse_remote_payload(proc.stdout)
        if payload is not None:
            invalid = validate_remote_payload(payload, target=args.target, resource=args.resource)
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
            "resource": args.resource,
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
            "resource": args.resource,
            "target": args.target,
            "stdout_tail": proc.stdout.strip()[-1000:],
            "stderr_tail": proc.stderr.strip()[-1000:],
        }
    invalid = validate_remote_payload(payload, target=args.target, resource=args.resource)
    if invalid is not None:
        invalid["stdout_tail"] = proc.stdout.strip()[-1000:]
        invalid["stderr_tail"] = proc.stderr.strip()[-1000:]
        return invalid
    payload.setdefault("target", args.target)
    return payload


def validate_remote_payload(payload: dict[str, Any], *, target: str, resource: str) -> dict[str, Any] | None:
    if payload.get("schema") != SCHEMA:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "invalid_start_schema",
            "resource": resource,
            "target": target,
            "start_schema": payload.get("schema"),
        }
    if payload.get("resource") != resource:
        return {
            "schema": SCHEMA,
            "ok": False,
            "reason": "start_resource_mismatch",
            "target": target,
            "resource": resource,
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
    remote_path = f"/tmp/tensorcore-georefine-rank-probe-{os.getpid()}-{uuid.uuid4().hex}.sh"
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
    remote_path = f"/tmp/tensorcore-georefine-rank-probe-{os.getpid()}-{uuid.uuid4().hex}.sh"
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
