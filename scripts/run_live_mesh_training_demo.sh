#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

world="${TC_MESH_WORLD:-4}"
rank0_host="${TC_MESH_RANK0_HOST:-100.96.130.16}"
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
log_dir="${TC_MESH_LOG_DIR:-${TMPDIR:-/tmp}/tensorcore-live-mesh-training-$(date +%Y%m%d-%H%M%S)}"

local_build_dir="${TC_MESH_TRAINING_LOCAL_BUILD_DIR:-$ROOT/build-live-mesh-training-cpu}"
rank0_bin="${TC_MESH_RANK0_BIN:-$local_build_dir/examples/mesh_training_demo}"
rank1_ssh="${TC_MESH_RANK1_SSH:-enki}"
rank1_dir="${TC_MESH_RANK1_DIR:-/tmp/tensorcore-live-mesh-training}"
rank1_bin="${TC_MESH_RANK1_BIN:-$rank1_dir/mesh_training_demo_cpu}"
rank2_ssh="${TC_MESH_RANK2_SSH:-old-donkey}"
rank2_dir="${TC_MESH_RANK2_DIR:-/tmp/tensorcore-live-mesh-training}"
rank2_bin="${TC_MESH_RANK2_BIN:-$rank2_dir/build/examples/mesh_training_demo}"
rank3_ssh="${TC_MESH_RANK3_SSH:-cosbox}"
rank3_dir="${TC_MESH_RANK3_DIR:-/tmp/tensorcore-live-mesh-training}"
rank3_bin="${TC_MESH_RANK3_BIN:-$rank3_dir/build/examples/mesh_training_demo}"

usage() {
    cat <<EOF
Usage: TC_MESH_PREPARE=1 $0

Runs the full mesh_training_demo across Atlas, Enki, old-donkey, and cosbox.
Rank 3 is built with CUDA by default; set TC_MESH_RANK3_CUDA=0 for CPU-only.

Environment:
  TC_MESH_TRAINING_INNER=$inner_steps
  TC_MESH_TRAINING_OUTER=$outer_steps
  TC_MESH_TRAINING_CHECKPOINT=$checkpoint
  TC_MESH_RANK3_CUDA=$rank3_cuda
  TC_MESH_PREPARE=$prepare
  TC_MESH_PORT=<port>
  TC_MESH_LOG_DIR=$log_dir
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "$world" != "4" ]]; then
    echo "run_live_mesh_training_demo currently expects TC_MESH_WORLD=4" >&2
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
    echo "[mesh-training] preparing $host:$remote_dir from git HEAD (CUDA=$enable_cuda)"
    git -C "$ROOT" archive --format=tar HEAD | ssh "$host" \
        "rm -rf '$remote_dir' && mkdir -p '$remote_dir' && tar -xf - -C '$remote_dir'"
    ssh "$host" \
        "cmake -S '$remote_dir' -B '$remote_dir/build' -DTC_ENABLE_METAL=OFF -DTC_ENABLE_CUDA='$enable_cuda' -DTC_BUILD_TESTS=OFF -DTC_BUILD_BENCH=OFF -DTC_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release && cmake --build '$remote_dir/build' --target mesh_training_demo --parallel '$remote_jobs'"
}

if [[ "$prepare" == "1" ]]; then
    echo "[mesh-training] preparing Enki portable binary"
    ssh "$rank1_ssh" "mkdir -p '$rank1_dir'"
    scp "$rank0_bin" "$rank1_ssh:$rank1_bin"
    ssh "$rank1_ssh" "chmod +x '$rank1_bin'"
    prepare_linux_rank "$rank2_ssh" "$rank2_dir" OFF
    if [[ "$rank3_cuda" == "1" ]]; then
        prepare_linux_rank "$rank3_ssh" "$rank3_dir" ON
    else
        prepare_linux_rank "$rank3_ssh" "$rank3_dir" OFF
    fi
fi

remote_check() {
    local host="$1"
    local bin="$2"
    ssh "$host" "test -x '$bin'"
}

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

rank_env=(
    "TC_GLOO_RING=$ring"
    "TC_GLOO_TRACE=$trace"
    "TC_GLOO_RING_CONNECT_TIMEOUT_MS=$timeout_ms"
)

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
    local remote_cmd="cd '$dir' && TC_GLOO_RING='$ring' TC_GLOO_TRACE='$trace' TC_GLOO_RING_CONNECT_TIMEOUT_MS='$timeout_ms' LD_LIBRARY_PATH='$dir/build':\${LD_LIBRARY_PATH:-} '$bin' --rank '$rank' --world '$world' --url '$url' --inner '$inner_steps' --outer '$outer_steps'"
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
run_remote_rank 1 "$rank1_ssh" "$rank1_bin" "$rank1_dir"
run_remote_rank 2 "$rank2_ssh" "$rank2_bin" "$rank2_dir"
run_remote_rank 3 "$rank3_ssh" "$rank3_bin" "$rank3_dir"

rc=0
for rank in 0 1 2 3; do
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

for rank in 0 1 2 3; do
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
echo "[mesh-training] live mesh training demo OK"
