#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
DEVICE_LOG="${RUNNER_TEMP:-/tmp}/tensorcore-device.log"

# Defense in depth: refuse to write to an empty or system-critical path,
# even though the default log path is constructed from safe temp directories.
require_output_file_path() {
    local label="$1"
    local path="$2"
    case "$path" in
        ""|"/"|"/etc"|"/etc/"*|"/bin"|"/bin/"*|"/usr"|"/usr/"*|"/sbin"|"/sbin/"*|"/System"|"/System/"*)
            echo "$label: refusing to write to system path: $path" >&2
            exit 2
            ;;
    esac
}
require_output_file_path "DEVICE_LOG" "$DEVICE_LOG"

run_no_device_subset() {
    ctest --test-dir "$BUILD_DIR" --output-on-failure \
        -R '^(test_distributed_ring|test_distributed_ring_fork|test_tensorops_select|test_tensorops_runtime)$'
}

run_paravirtual_subset() {
    ctest --test-dir "$BUILD_DIR" --output-on-failure \
        -R '^(test_device|test_distributed_ring|test_distributed_ring_fork|test_tensorops_select|test_tensorops_runtime|test_gguf)$'
}

require_output_file_path "DEVICE_LOG (test_device redirect)" "$DEVICE_LOG"
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
