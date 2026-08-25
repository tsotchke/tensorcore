#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/.scratch/release-sanitizers}"
EVIDENCE_PATH="${TENSORCORE_SANITIZER_EVIDENCE_PATH:-$BUILD_DIR/release_sanitizers_evidence.json}"
PHASE="init"
STATUS="running"
EXIT_STATUS=""
CONFIGURE_STATUS="not_run"
BUILD_STATUS="not_run"
CTEST_STATUS="not_run"

write_evidence() {
    mkdir -p "$(dirname "$EVIDENCE_PATH")"
    export ROOT PHASE STATUS EXIT_STATUS CONFIGURE_STATUS BUILD_STATUS CTEST_STATUS
    python3 - "$EVIDENCE_PATH" <<'PY'
import datetime
import json
import os
import pathlib
import subprocess
import sys


def git_output(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=os.environ["ROOT"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def git_dirty():
    try:
        for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
            if subprocess.run(
                ["git", *args], cwd=os.environ["ROOT"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode != 0:
                return True
        return False
    except Exception:
        return None


status = os.environ["STATUS"]
artifact = {
    "schema": "tensorcore.release_sanitizers.v1",
    "meta": {
        "git_head": git_output("rev-parse", "HEAD"),
        "git_dirty": git_dirty(),
        "source": "tensorcore_release_sanitizers",
    },
    "status": status,
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "run": {
        "phase": os.environ["PHASE"],
        "exit_status": os.environ["EXIT_STATUS"],
    },
    "checks": {
        "configure": {
            "status": os.environ["CONFIGURE_STATUS"],
            "passed": os.environ["CONFIGURE_STATUS"] == "passed",
        },
        "build": {
            "status": os.environ["BUILD_STATUS"],
            "passed": os.environ["BUILD_STATUS"] == "passed",
        },
        "ctest": {
            "status": os.environ["CTEST_STATUS"],
            "passed": os.environ["CTEST_STATUS"] == "passed",
        },
        "address": {"enabled": True},
        "undefined": {"enabled": True},
    },
}
path = pathlib.Path(sys.argv[1])
temp = path.with_name(f".{path.name}.tmp")
temp.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temp.replace(path)
PY
}

on_exit() {
    local exit_status=$?
    if [[ $exit_status -ne 0 ]]; then
        set +e
        STATUS="failed"
        EXIT_STATUS="$exit_status"
        write_evidence
    fi
}
trap on_exit EXIT

cmake -E remove_directory "$BUILD_DIR"

PHASE="configure"
CONFIGURE_STATUS="running"
cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DTC_ENABLE_METAL=OFF \
    -DTC_ENABLE_CUDA=OFF \
    -DTC_ENABLE_HIP=OFF \
    -DTC_BUILD_TESTS=ON \
    -DTC_BUILD_BENCH=OFF \
    -DTC_BUILD_EXAMPLES=OFF \
    -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
    -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined"
CONFIGURE_STATUS="passed"

PHASE="build"
BUILD_STATUS="running"
cmake --build "$BUILD_DIR" --parallel 4
BUILD_STATUS="passed"

PHASE="ctest"
CTEST_STATUS="running"
ASAN_OPTIONS="detect_leaks=0:abort_on_error=1" \
UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1" \
    ctest --test-dir "$BUILD_DIR" --output-on-failure --parallel 4
CTEST_STATUS="passed"

PHASE="complete"
STATUS="passed"
EXIT_STATUS="0"
write_evidence
python3 "$ROOT/scripts/check_release_artifact_privacy.py" "$EVIDENCE_PATH"
echo "[tensorcore] release sanitizer evidence OK"
