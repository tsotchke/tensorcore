#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-"$ROOT/build-m5-tensorops"}"
EVIDENCE_PATH="${TENSORCORE_RELEASE_SMOKE_EVIDENCE_PATH:-"$BUILD_DIR/release_smoke_runtime_evidence.json"}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export BUILD_DIR
export PYTHON_BIN
export REQUIRE_GPU=1
export REQUIRE_METAL4_TENSOROPS=1
export TENSORCORE_RELEASE_SMOKE_EVIDENCE_PATH="$EVIDENCE_PATH"

"$ROOT/scripts/release_smoke.sh"
"$PYTHON_BIN" "$ROOT/scripts/check_release_evidence.py" \
    "$EVIDENCE_PATH" \
    --require-gpu \
    --require-metal4-tensorops
