#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

world="${TC_MESH_WORLD:-4}"
rank0_host="${TC_MESH_RANK0_HOST:-100.96.130.16}"
port="${TC_MESH_PORT:-}"
test_filter="${TC_MESH_TEST:-all}"
elements="${TC_MESH_ELEMENTS:-65536}"
iters="${TC_MESH_ITERS:-2}"
diloco_elements="${TC_MESH_DILOCO_ELEMENTS:-65536}"
diloco_cycles="${TC_MESH_DILOCO_CYCLES:-3}"
diloco_inner_steps="${TC_MESH_DILOCO_INNER_STEPS:-5}"
ring="${TC_MESH_RING:-1}"
trace="${TC_MESH_TRACE:-1}"
timeout_ms="${TC_MESH_RING_CONNECT_TIMEOUT_MS:-10000}"
prepare="${TC_MESH_PREPARE:-0}"
remote_jobs="${TC_MESH_REMOTE_JOBS:-8}"
log_dir="${TC_MESH_LOG_DIR:-${TMPDIR:-/tmp}/tensorcore-live-mesh-$(date +%Y%m%d-%H%M%S)}"

rank0_bin="${TC_MESH_RANK0_BIN:-$ROOT/build-portable-cpu-current/tests/test_dist_remote}"
rank1_ssh="${TC_MESH_RANK1_SSH:-enki}"
rank1_dir="${TC_MESH_RANK1_DIR:-/tmp/tensorcore-live-mesh}"
rank1_bin="${TC_MESH_RANK1_BIN:-$rank1_dir/test_dist_remote_cpu}"
rank2_ssh="${TC_MESH_RANK2_SSH:-old-donkey}"
rank2_dir="${TC_MESH_RANK2_DIR:-/tmp/tensorcore-live-mesh}"
rank2_bin="${TC_MESH_RANK2_BIN:-$rank2_dir/build/tests/test_dist_remote}"
rank3_ssh="${TC_MESH_RANK3_SSH:-cosbox}"
rank3_dir="${TC_MESH_RANK3_DIR:-/tmp/tensorcore-live-mesh}"
rank3_bin="${TC_MESH_RANK3_BIN:-$rank3_dir/build/tests/test_dist_remote}"

usage() {
    cat <<EOF
Usage: TC_MESH_PREPARE=1 $0

Runs a live 4-rank tensorcore GLOO smoke over the mesh:
  rank 0 local Atlas: $rank0_bin
  rank 1 $rank1_ssh: $rank1_bin
  rank 2 $rank2_ssh: $rank2_bin
  rank 3 $rank3_ssh: $rank3_bin

Environment:
  TC_MESH_TEST=$test_filter                 all | allreduce | diloco
  TC_MESH_ELEMENTS=$elements                fp32 elements for allreduce probe
  TC_MESH_ITERS=$iters                      timed allreduce iterations
  TC_MESH_DILOCO_ELEMENTS=$diloco_elements  fp16 parameters for DiLoCo probe
  TC_MESH_DILOCO_CYCLES=$diloco_cycles      DiLoCo outer steps
  TC_MESH_DILOCO_INNER_STEPS=$diloco_inner_steps
  TC_MESH_PREPARE=$prepare                  1 archives/builds remote binaries first
  TC_MESH_PORT=<port>                       rendezvous port; auto-selected if unset
  TC_MESH_LOG_DIR=$log_dir
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "$world" != "4" ]]; then
    echo "run_live_mesh_smoke currently expects TC_MESH_WORLD=4" >&2
    exit 2
fi

mkdir -p "$log_dir"

if [[ ! -x "$rank0_bin" || "${TC_MESH_BUILD_LOCAL:-0}" == "1" ]]; then
    cmake --build "$ROOT/build-portable-cpu-current" --target test_dist_remote --parallel
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
    echo "[mesh] preparing $host:$remote_dir from git HEAD"
    git -C "$ROOT" archive --format=tar HEAD | ssh "$host" \
        "rm -rf '$remote_dir' && mkdir -p '$remote_dir' && tar -xf - -C '$remote_dir'"
    ssh "$host" \
        "cmake -S '$remote_dir' -B '$remote_dir/build' -DTC_ENABLE_METAL=OFF -DTC_ENABLE_CUDA=OFF -DTC_BUILD_TESTS=ON -DTC_BUILD_BENCH=OFF -DTC_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build '$remote_dir/build' --target test_dist_remote --parallel '$remote_jobs'"
}

if [[ "$prepare" == "1" ]]; then
    echo "[mesh] preparing Enki portable binary"
    ssh "$rank1_ssh" "mkdir -p '$rank1_dir'"
    scp "$rank0_bin" "$rank1_ssh:$rank1_bin"
    ssh "$rank1_ssh" "chmod +x '$rank1_bin'"
    prepare_linux_rank "$rank2_ssh" "$rank2_dir"
    prepare_linux_rank "$rank3_ssh" "$rank3_dir"
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
    --test "$test_filter"
    --elements "$elements"
    --iters "$iters"
    --diloco-elements "$diloco_elements"
    --diloco-cycles "$diloco_cycles"
    --diloco-inner-steps "$diloco_inner_steps"
)

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
    local log="$log_dir/rank${rank}.log"
    local remote_cmd="TC_GLOO_RING='$ring' TC_GLOO_TRACE='$trace' TC_GLOO_RING_CONNECT_TIMEOUT_MS='$timeout_ms' '$bin' --rank '$rank' --world '$world' --url '$url' --test '$test_filter' --elements '$elements' --iters '$iters' --diloco-elements '$diloco_elements' --diloco-cycles '$diloco_cycles' --diloco-inner-steps '$diloco_inner_steps'"
    ssh "$host" "$remote_cmd" >"$log" 2>&1 &
    pids[$rank]=$!
}

echo "[mesh] logs: $log_dir"
echo "[mesh] url: $url"
run_local_rank 0
sleep "${TC_MESH_LAUNCH_DELAY:-0.5}"
run_remote_rank 1 "$rank1_ssh" "$rank1_bin"
run_remote_rank 2 "$rank2_ssh" "$rank2_bin"
run_remote_rank 3 "$rank3_ssh" "$rank3_bin"

rc=0
for rank in 0 1 2 3; do
    if ! wait "${pids[$rank]}"; then
        echo "[mesh] rank $rank failed; log follows" >&2
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
    grep -q "\[rank $rank\] OK" "$log" || {
        echo "[mesh] rank $rank did not report OK" >&2
        sed "s/^/[rank $rank] /" "$log" >&2
        exit 1
    }
    if [[ "$ring" == "1" && "$test_filter" != "diloco" ]]; then
        grep -q "direct_ring=enabled" "$log" || {
            echo "[mesh] rank $rank did not enable direct ring" >&2
            sed "s/^/[rank $rank] /" "$log" >&2
            exit 1
        }
        grep -q "allreduce_f32_sum route=ring" "$log" || {
            echo "[mesh] rank $rank did not route allreduce over ring" >&2
            sed "s/^/[rank $rank] /" "$log" >&2
            exit 1
        }
    fi
done

echo "[mesh] summary"
grep -hE "direct_ring=|route=|rendezvous done|allreduce |DiLoCo|\\[rank [0-9]+\\] OK" "$log_dir"/rank*.log || true
echo "[mesh] live mesh smoke OK"
