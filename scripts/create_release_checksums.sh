#!/usr/bin/env bash
set -euo pipefail

DIST_DIR="${1:-dist}"
OUT="${DIST_DIR}/SHA256SUMS"

# Defense in depth: refuse to write to an empty or system-critical path.
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
require_output_file_path "OUT" "$OUT"

if [ ! -d "$DIST_DIR" ]; then
    echo "release dist directory not found: $DIST_DIR" >&2
    exit 1
fi

artifacts=(
    "$DIST_DIR"/tensorcore_apple-*.whl
    "$DIST_DIR"/tensorcore-native-sdk-*.tar.gz
)

missing=()
for artifact in "${artifacts[@]}"; do
    if [ ! -f "$artifact" ]; then
        missing+=("$artifact")
    fi
done
if [ "${#missing[@]}" -ne 0 ]; then
    echo "release artifact(s) missing for checksum generation:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
fi

OUT_TMP="$(mktemp "${OUT}.XXXXXX")"
cleanup_out_tmp() {
    [ -n "${OUT_TMP:-}" ] && [ -f "$OUT_TMP" ] && command rm -- "$OUT_TMP" 2>/dev/null || true
}
trap cleanup_out_tmp EXIT
(
    cd "$DIST_DIR"
    LC_ALL=C shasum -a 256 \
        tensorcore_apple-*.whl \
        tensorcore-native-sdk-*.tar.gz |
        LC_ALL=C sort -k2
) > "$OUT_TMP"
require_output_file_path "OUT (SHA256SUMS write)" "$OUT"
mv "$OUT_TMP" "$OUT"

(
    cd "$DIST_DIR"
    LC_ALL=C shasum -a 256 -c "$(basename "$OUT")"
)

echo "$OUT"
