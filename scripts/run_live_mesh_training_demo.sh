#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

local_only="${TC_MESH_LOCAL_ONLY:-0}"
world="${TC_MESH_WORLD:-4}"
default_rank0_host="100.96.130.16"
if [[ "$local_only" == "1" ]]; then
    default_rank0_host="127.0.0.1"
fi
rank0_host="${TC_MESH_RANK0_HOST:-$default_rank0_host}"
port="${TC_MESH_PORT:-}"
inner_steps="${TC_MESH_TRAINING_INNER:-3}"
outer_steps="${TC_MESH_TRAINING_OUTER:-2}"
checkpoint="${TC_MESH_TRAINING_CHECKPOINT:-1}"
prepare="${TC_MESH_PREPARE:-0}"
remote_jobs="${TC_MESH_REMOTE_JOBS:-8}"
rank3_cuda="${TC_MESH_RANK3_CUDA:-1}"
trace="${TC_MESH_TRACE:-1}"
ring="${TC_MESH_RING:-1}"
timeout_ms="${TC_MESH_RING_CONNECT_TIMEOUT_MS:-10000}"
advertise_hosts="${TC_GLOO_ADVERTISE_HOSTS:-}"
log_dir="${TC_MESH_LOG_DIR:-${TMPDIR:-/tmp}/tensorcore-live-mesh-training-$(date +%Y%m%d-%H%M%S)}"
evidence_path="${TC_MESH_TRAINING_EVIDENCE_PATH:-}"

local_build_dir="${TC_MESH_TRAINING_LOCAL_BUILD_DIR:-$ROOT/build-live-mesh-training-cpu}"
rank0_bin="${TC_MESH_RANK0_BIN:-$local_build_dir/examples/mesh_training_demo}"
rank1_ssh="${TC_MESH_RANK1_SSH:-enki}"
rank1_dir="${TC_MESH_RANK1_DIR:-/tmp/tensorcore-live-mesh-training}"
rank1_prepare="${TC_MESH_RANK1_PREPARE:-copy-local}"
rank1_bin="${TC_MESH_RANK1_BIN:-}"
if [[ -z "$rank1_bin" ]]; then
    if [[ "$rank1_prepare" == "linux" ]]; then
        rank1_bin="$rank1_dir/build/examples/mesh_training_demo"
    else
        rank1_bin="$rank1_dir/mesh_training_demo_cpu"
    fi
fi
rank1_path="${TC_MESH_RANK1_PATH:-}"
rank2_ssh="${TC_MESH_RANK2_SSH:-old-donkey}"
rank2_dir="${TC_MESH_RANK2_DIR:-/tmp/tensorcore-live-mesh-training}"
rank2_bin="${TC_MESH_RANK2_BIN:-$rank2_dir/build/examples/mesh_training_demo}"
rank2_path="${TC_MESH_RANK2_PATH:-}"
rank3_ssh="${TC_MESH_RANK3_SSH:-cosbox}"
rank3_dir="${TC_MESH_RANK3_DIR:-/tmp/tensorcore-live-mesh-training}"
rank3_bin="${TC_MESH_RANK3_BIN:-$rank3_dir/build/examples/mesh_training_demo}"
rank3_path="${TC_MESH_RANK3_PATH:-}"

usage() {
    cat <<EOF
Usage: TC_MESH_PREPARE=1 $0

Runs the full mesh_training_demo across Atlas, Enki, old-donkey, and cosbox.
Rank 3 is built with CUDA by default; set TC_MESH_RANK3_CUDA=0 for CPU-only.
Set TC_MESH_LOCAL_ONLY=1 to run all ranks on this host for regression evidence.

Environment:
  TC_MESH_LOCAL_ONLY=$local_only
  TC_MESH_WORLD=$world
  TC_MESH_TRAINING_INNER=$inner_steps
  TC_MESH_TRAINING_OUTER=$outer_steps
  TC_MESH_TRAINING_CHECKPOINT=$checkpoint
  TC_MESH_RANK3_CUDA=$rank3_cuda
  TC_MESH_PREPARE=$prepare
  TC_MESH_PORT=<port>
  TC_MESH_RANK1_PREPARE=copy-local|linux
  TC_MESH_RANK{1,2,3}_PATH=<extra remote PATH prefix>
  TC_GLOO_ADVERTISE_HOSTS=<rank0,rank1,...>
  TC_MESH_LOG_DIR=$log_dir
  TC_MESH_TRAINING_EVIDENCE_PATH=<json>
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "$world" != "4" && "$local_only" != "1" ]]; then
    echo "run_live_mesh_training_demo expects TC_MESH_WORLD=4 unless TC_MESH_LOCAL_ONLY=1" >&2
    exit 2
fi

mkdir -p "$log_dir"

if [[ ! -x "$rank0_bin" || "${TC_MESH_BUILD_LOCAL:-0}" == "1" ]]; then
    cmake -S "$ROOT" -B "$local_build_dir" \
        -DTC_ENABLE_METAL=OFF \
        -DTC_ENABLE_CUDA=OFF \
        -DTC_BUILD_TESTS=OFF \
        -DTC_BUILD_BENCH=OFF \
        -DTC_BUILD_EXAMPLES=ON \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "$local_build_dir" --target mesh_training_demo --parallel
fi
if [[ ! -x "$rank0_bin" ]]; then
    echo "rank 0 binary not found: $rank0_bin" >&2
    exit 1
fi

choose_port() {
    python3 - "$rank0_host" <<'PY'
import socket
import sys

host = sys.argv[1]
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((host, 0))
    print(sock.getsockname()[1])
finally:
    sock.close()
PY
}

if [[ -z "$port" ]]; then
    if ! port="$(choose_port)"; then
        echo "could not auto-select a port on $rank0_host; set TC_MESH_PORT" >&2
        exit 1
    fi
fi
url="tcp://$rank0_host:$port"

prepare_linux_rank() {
    local host="$1"
    local remote_dir="$2"
    local enable_cuda="$3"
    local remote_path="${4:-}"
    local source_head
    local source_dirty=0
    source_head="$(git -C "$ROOT" rev-parse HEAD)"
    if ! git -C "$ROOT" diff --quiet || ! git -C "$ROOT" diff --cached --quiet; then
        source_dirty=1
    fi
    echo "[mesh-training] preparing $host:$remote_dir from git HEAD (CUDA=$enable_cuda)"
    git -C "$ROOT" archive --format=tar HEAD | ssh "$host" \
        "rm -rf '$remote_dir' && mkdir -p '$remote_dir' && tar -xf - -C '$remote_dir'"
    ssh "$host" \
        "printf '%s\n' '$source_head' > '$remote_dir/.tensorcore_source_head' && printf '%s\n' '$source_dirty' > '$remote_dir/.tensorcore_source_dirty'"
    local path_prefix=""
    if [[ -n "$remote_path" ]]; then
        path_prefix="PATH='$remote_path':\$PATH; export PATH;"
    fi
    ssh "$host" \
        "$path_prefix cmake -S '$remote_dir' -B '$remote_dir/build' -DTC_ENABLE_METAL=OFF -DTC_ENABLE_CUDA='$enable_cuda' -DTC_BUILD_TESTS=OFF -DTC_BUILD_BENCH=OFF -DTC_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release && cmake --build '$remote_dir/build' --target mesh_training_demo --parallel '$remote_jobs'"
}

if [[ "$prepare" == "1" && "$local_only" != "1" ]]; then
    if [[ "$rank1_prepare" == "copy-local" ]]; then
        echo "[mesh-training] preparing rank 1 portable binary"
        ssh "$rank1_ssh" "mkdir -p '$rank1_dir'"
        scp "$rank0_bin" "$rank1_ssh:$rank1_bin"
        ssh "$rank1_ssh" "chmod +x '$rank1_bin'"
    elif [[ "$rank1_prepare" == "linux" ]]; then
        prepare_linux_rank "$rank1_ssh" "$rank1_dir" OFF "$rank1_path"
    else
        echo "unsupported TC_MESH_RANK1_PREPARE=$rank1_prepare" >&2
        exit 2
    fi
    prepare_linux_rank "$rank2_ssh" "$rank2_dir" OFF "$rank2_path"
    if [[ "$rank3_cuda" == "1" ]]; then
        prepare_linux_rank "$rank3_ssh" "$rank3_dir" ON "$rank3_path"
    else
        prepare_linux_rank "$rank3_ssh" "$rank3_dir" OFF "$rank3_path"
    fi
fi

remote_check() {
    local host="$1"
    local bin="$2"
    ssh "$host" "test -x '$bin'"
}

if [[ "$local_only" != "1" ]]; then
    remote_check "$rank1_ssh" "$rank1_bin" || {
        echo "rank 1 binary missing on $rank1_ssh: $rank1_bin" >&2
        echo "rerun with TC_MESH_PREPARE=1" >&2
        exit 1
    }
    remote_check "$rank2_ssh" "$rank2_bin" || {
        echo "rank 2 binary missing on $rank2_ssh: $rank2_bin" >&2
        echo "rerun with TC_MESH_PREPARE=1" >&2
        exit 1
    }
    remote_check "$rank3_ssh" "$rank3_bin" || {
        echo "rank 3 binary missing on $rank3_ssh: $rank3_bin" >&2
        echo "rerun with TC_MESH_PREPARE=1" >&2
        exit 1
    }
fi

rank_env=(
    "TC_GLOO_RING=$ring"
    "TC_GLOO_TRACE=$trace"
    "TC_GLOO_RING_CONNECT_TIMEOUT_MS=$timeout_ms"
)
if [[ -n "$advertise_hosts" ]]; then
    rank_env+=("TC_GLOO_ADVERTISE_HOSTS=$advertise_hosts")
fi

rank_args=(
    --world "$world"
    --url "$url"
    --inner "$inner_steps"
    --outer "$outer_steps"
)
if [[ "$checkpoint" == "1" ]]; then
    rank_args+=(--checkpoint)
fi

pids=()
cleanup() {
    local pid
    for pid in "${pids[@]:-}"; do
        kill "$pid" >/dev/null 2>&1 || true
    done
}
trap cleanup INT TERM

run_local_rank() {
    local rank="$1"
    local log="$log_dir/rank${rank}.log"
    env "${rank_env[@]}" "$rank0_bin" --rank "$rank" "${rank_args[@]}" >"$log" 2>&1 &
    pids[$rank]=$!
}

run_remote_rank() {
    local rank="$1"
    local host="$2"
    local bin="$3"
    local dir="$4"
    local log="$log_dir/rank${rank}.log"
    local remote_path="${5:-}"
    local path_prefix=""
    if [[ -n "$remote_path" ]]; then
        path_prefix="PATH='$remote_path':\$PATH; export PATH; "
    fi
    local remote_cmd="${path_prefix}cd '$dir' && TC_GLOO_RING='$ring' TC_GLOO_TRACE='$trace' TC_GLOO_RING_CONNECT_TIMEOUT_MS='$timeout_ms'"
    if [[ -n "$advertise_hosts" ]]; then
        remote_cmd="$remote_cmd TC_GLOO_ADVERTISE_HOSTS='$advertise_hosts'"
    fi
    remote_cmd="$remote_cmd LD_LIBRARY_PATH='$dir/build':\${LD_LIBRARY_PATH:-} '$bin' --rank '$rank' --world '$world' --url '$url' --inner '$inner_steps' --outer '$outer_steps'"
    if [[ "$checkpoint" == "1" ]]; then
        remote_cmd="$remote_cmd --checkpoint"
    fi
    ssh "$host" "$remote_cmd" >"$log" 2>&1 &
    pids[$rank]=$!
}

echo "[mesh-training] logs: $log_dir"
echo "[mesh-training] url: $url"
run_local_rank 0
sleep "${TC_MESH_LAUNCH_DELAY:-0.5}"
if [[ "$local_only" == "1" ]]; then
    for ((rank = 1; rank < world; rank++)); do
        run_local_rank "$rank"
    done
else
    run_remote_rank 1 "$rank1_ssh" "$rank1_bin" "$rank1_dir" "$rank1_path"
    run_remote_rank 2 "$rank2_ssh" "$rank2_bin" "$rank2_dir" "$rank2_path"
    run_remote_rank 3 "$rank3_ssh" "$rank3_bin" "$rank3_dir" "$rank3_path"
fi

rc=0
for ((rank = 0; rank < world; rank++)); do
    if ! wait "${pids[$rank]}"; then
        echo "[mesh-training] rank $rank failed; log follows" >&2
        sed "s/^/[rank $rank] /" "$log_dir/rank${rank}.log" >&2 || true
        rc=1
    fi
done
trap - INT TERM

if [[ "$rc" != "0" ]]; then
    exit "$rc"
fi

write_evidence() {
    local status="$1"
    local path="$2"
    if [[ -z "$path" ]]; then
        return
    fi
    python3 - "$ROOT" "$log_dir" "$path" "$status" "$world" "$url" \
        "$inner_steps" "$outer_steps" "$checkpoint" "$ring" "$trace" \
        "$timeout_ms" "$prepare" "$rank3_cuda" "$local_only" <<'PY'
import json
import pathlib
import re
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
log_dir = pathlib.Path(sys.argv[2])
path = pathlib.Path(sys.argv[3])
status = sys.argv[4]
world = int(sys.argv[5])
url = sys.argv[6]
inner_steps = int(sys.argv[7])
outer_steps = int(sys.argv[8])
checkpoint_enabled = sys.argv[9] == "1"
ring_enabled = sys.argv[10] == "1"
trace_enabled = sys.argv[11] == "1"
timeout_ms = int(sys.argv[12])
prepare = sys.argv[13] == "1"
rank3_cuda = sys.argv[14] == "1"
local_only = sys.argv[15] == "1"

patterns = {
    "direct": re.compile(
        r"\[tensorcore:gloo rank (?P<rank>\d+)\] direct_ring=(?P<enabled>\w+) "
        r"next_rank=(?P<next_rank>\d+) next=(?P<next>\S+) timeout_ms=(?P<timeout>\d+)"
    ),
    "route": re.compile(
        r"\[tensorcore:gloo rank (?P<rank>\d+)\] allreduce_f32_sum "
        r"route=(?P<route>\w+) elements=(?P<elements>\d+)"
    ),
    "rendezvous": re.compile(
        r"\[rank (?P<rank>\d+)/(?P<world>\d+)\] rendezvous "
        r"(?P<seconds>[0-9.eE+-]+)s via (?P<url>\S+)"
    ),
    "outer": re.compile(
        r"\[rank (?P<rank>\d+)\] outer (?P<step>\d+)/(?P<total>\d+) "
        r"loss=(?P<loss>[0-9.eE+-]+) bytes=(?P<bytes>\d+) backend=(?P<backend>\S+)"
    ),
    "ok": re.compile(
        r"\[rank (?P<rank>\d+)\] mesh_training_demo OK first_loss="
        r"(?P<first_loss>[0-9.eE+-]+) last_loss=(?P<last_loss>[0-9.eE+-]+) "
        r"outer_steps=(?P<outer_steps>\d+) elapsed=(?P<elapsed>[0-9.eE+-]+)s"
    ),
    "checkpoint": re.compile(
        r"\[rank (?P<rank>\d+)\] checkpoint x_norm discards=(?P<discards>\d+) "
        r"realizes=(?P<realizes>\d+) peak_discarded=(?P<peak>\d+) "
        r"final_discarded=(?P<final>\d+)"
    ),
}

ranks = {rank: {"rank": rank, "outer": [], "routes": []} for rank in range(world)}
for rank in range(world):
    log_path = log_dir / f"rank{rank}.log"
    ranks[rank]["log"] = str(log_path)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if match := patterns["direct"].search(line):
            data = match.groupdict()
            ranks[rank]["direct_ring"] = {
                "enabled": data["enabled"] == "enabled",
                "next_rank": int(data["next_rank"]),
                "next": data["next"],
                "timeout_ms": int(data["timeout"]),
            }
        elif match := patterns["route"].search(line):
            data = match.groupdict()
            ranks[rank]["routes"].append({
                "route": data["route"],
                "elements": int(data["elements"]),
            })
        elif match := patterns["rendezvous"].search(line):
            data = match.groupdict()
            ranks[rank]["rendezvous"] = {
                "seconds": float(data["seconds"]),
                "world": int(data["world"]),
                "url": data["url"],
            }
        elif match := patterns["outer"].search(line):
            data = match.groupdict()
            ranks[rank]["outer"].append({
                "step": int(data["step"]),
                "total": int(data["total"]),
                "loss": float(data["loss"]),
                "bytes": int(data["bytes"]),
                "backend": data["backend"],
            })
        elif match := patterns["ok"].search(line):
            data = match.groupdict()
            ranks[rank]["ok"] = True
            ranks[rank]["first_loss"] = float(data["first_loss"])
            ranks[rank]["last_loss"] = float(data["last_loss"])
            ranks[rank]["outer_steps_completed"] = int(data["outer_steps"])
            ranks[rank]["elapsed_seconds"] = float(data["elapsed"])
        elif match := patterns["checkpoint"].search(line):
            data = match.groupdict()
            ranks[rank]["checkpoint"] = {
                "discards": int(data["discards"]),
                "realizes": int(data["realizes"]),
                "peak_discarded": int(data["peak"]),
                "final_discarded": int(data["final"]),
            }

rank_list = [ranks[rank] for rank in range(world)]
ring_route_events = sum(
    1 for rank in rank_list for route in rank["routes"] if route["route"] == "ring"
)
direct_ring_ranks = sum(1 for rank in rank_list if rank.get("direct_ring", {}).get("enabled"))
checkpoint_ranks = sum(1 for rank in rank_list if "checkpoint" in rank)
cuda_ranks = [
    rank["rank"]
    for rank in rank_list
    if any(outer["backend"] == "cuda" for outer in rank["outer"])
]
loss_decreased = all(
    rank.get("last_loss", float("inf")) < rank.get("first_loss", float("-inf"))
    for rank in rank_list
)
all_ranks_passed = all(rank.get("ok") for rank in rank_list)
all_requested_outer_steps = all(
    rank.get("outer_steps_completed") == outer_steps for rank in rank_list
)

def git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()

evidence = {
    "schema": "tensorcore.live_mesh_training.evidence.v1",
    "meta": {
        "format": 1,
        "source": "run_live_mesh_training_demo",
        "git_head": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
    },
    "status": status,
    "run": {
        "world": world,
        "url": url,
        "log_dir": str(log_dir),
        "inner_steps": inner_steps,
        "outer_steps": outer_steps,
        "checkpoint_enabled": checkpoint_enabled,
        "ring_enabled": ring_enabled,
        "trace_enabled": trace_enabled,
        "ring_connect_timeout_ms": timeout_ms,
        "prepare": prepare,
        "rank3_cuda_requested": rank3_cuda,
        "local_only": local_only,
    },
    "summary": {
        "passed": status == "passed",
        "all_ranks_passed": all_ranks_passed,
        "all_requested_outer_steps": all_requested_outer_steps,
        "loss_decreased": loss_decreased,
        "direct_ring_ranks": direct_ring_ranks,
        "ring_route_events": ring_route_events,
        "checkpoint_ranks": checkpoint_ranks,
        "cuda_ranks": cuda_ranks,
    },
    "ranks": rank_list,
}

path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

for ((rank = 0; rank < world; rank++)); do
    log="$log_dir/rank${rank}.log"
    grep -q "\[rank $rank\] mesh_training_demo OK" "$log" || {
        echo "[mesh-training] rank $rank did not report mesh_training_demo OK" >&2
        sed "s/^/[rank $rank] /" "$log" >&2
        exit 1
    }
    grep -q "outer_steps=$outer_steps" "$log" || {
        echo "[mesh-training] rank $rank did not finish $outer_steps outer steps" >&2
        sed "s/^/[rank $rank] /" "$log" >&2
        exit 1
    }
    if [[ "$checkpoint" == "1" ]]; then
        grep -q "checkpoint x_norm discards=" "$log" || {
            echo "[mesh-training] rank $rank did not report checkpoint counters" >&2
            sed "s/^/[rank $rank] /" "$log" >&2
            exit 1
        }
    fi
done

echo "[mesh-training] summary"
grep -hE "direct_ring=|route=|rendezvous|outer |mesh_training_demo|checkpoint x_norm" \
    "$log_dir"/rank*.log || true
write_evidence passed "$evidence_path"
if [[ -n "$evidence_path" ]]; then
    echo "[mesh-training] evidence: $evidence_path"
fi
echo "[mesh-training] live mesh training demo OK"
