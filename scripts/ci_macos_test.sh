#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
DEVICE_LOG="${RUNNER_TEMP:-/tmp}/tensorcore-device.log"

run_no_device_subset() {
    ctest --test-dir "$BUILD_DIR" --output-on-failure \
        -R '^(test_distributed_ring|test_distributed_ring_fork|test_tensorops_select)$'
}

run_paravirtual_subset() {
    ctest --test-dir "$BUILD_DIR" --output-on-failure \
        -R '^(test_device|test_distributed_ring|test_distributed_ring_fork|test_tensorops_select|test_gguf)$'
}

if ! "$BUILD_DIR/tests/test_device" >"$DEVICE_LOG" 2>&1; then
    cat "$DEVICE_LOG"
    echo "No usable Metal device on this runner; running no-device smoke subset."
    run_no_device_subset
    exit 0
fi

cat "$DEVICE_LOG"

if grep -Eq 'family[[:space:]]*:[[:space:]]*Apple(0|[1-6]\b)' "$DEVICE_LOG" ||
   grep -Fq 'Apple Paravirtual device' "$DEVICE_LOG"; then
    echo "Non-production Apple GPU detected; running paravirtual-safe smoke subset."
    run_paravirtual_subset
    exit 0
fi

ctest --test-dir "$BUILD_DIR" --output-on-failure
