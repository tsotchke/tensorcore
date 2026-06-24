#!/usr/bin/env bash
# icc_check_oracles.sh — run all three tensorcore completion oracles with the
# right trace files, so ICC sees the produced runtime evidence without being
# polluted by CMake's intentional probe failures.
#
# Run from anywhere; uses the script's directory to anchor paths.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ICC="${ICC_BIN:-$HOME/Desktop/infinite_context_coder/bin/icc}"

cd "$REPO_ROOT"

if [[ ! -x "$ICC" ]]; then
    echo "ICC CLI not found at $ICC — set ICC_BIN or install infinite_context_coder" >&2
    exit 2
fi

trace_or_skip() {
    local trace="$1"
    if [[ ! -f "$trace" ]]; then
        echo "trace missing: $trace — run the appropriate smoke first" >&2
        return 1
    fi
    return 0
}

print_oracle() {
    local target="$1"; shift
    echo "=== $target ==="
    "$ICC" readiness --repo tensorcore --target "$target" "$@" --format markdown \
        | sed -n '5,20p'
    echo
}

# 1. Eshkol bridge (single file)
ESHKOL_TRACE="$REPO_ROOT/build/eshkol_tensorcore_bridge_evidence.json"
trace_or_skip "$ESHKOL_TRACE" || true
print_oracle eshkol-bridge-runtime-evidence --trace-file "$ESHKOL_TRACE"

# 2. PyTorch bridge (single file)
PYT_TRACE="$REPO_ROOT/build/pytorch_bridge_runtime_evidence.json"
trace_or_skip "$PYT_TRACE" || true
print_oracle pytorch-bridge-runtime-evidence --trace-file "$PYT_TRACE"

# 3. cuda-for-apple (release smoke + sdk26 evidence; do NOT use --trace-dir
#    because it would scoop CMake's intentional OpenMP probe-failure logs).
RS_TRACE="$REPO_ROOT/build/release_smoke_runtime_evidence.json"
SDK26_EVIDENCE_DIR="$REPO_ROOT/build/sdk26-compile-evidence-26479827492"
SDK26_TRACE="$SDK26_EVIDENCE_DIR/release_smoke_runtime_evidence.json"
trace_or_skip "$RS_TRACE" || true

if [[ -f "$SDK26_TRACE" ]]; then
    print_oracle cuda-for-apple-public-integration \
        --trace-file "$RS_TRACE" \
        --trace-file "$SDK26_TRACE"
else
    print_oracle cuda-for-apple-public-integration \
        --trace-file "$RS_TRACE"
fi
