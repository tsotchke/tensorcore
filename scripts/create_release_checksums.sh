#!/usr/bin/env bash
set -euo pipefail

DIST_DIR="${1:-dist}"
OUT="${DIST_DIR}/SHA256SUMS"

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

(
    cd "$DIST_DIR"
    LC_ALL=C shasum -a 256 \
        tensorcore_apple-*.whl \
        tensorcore-native-sdk-*.tar.gz |
        LC_ALL=C sort -k2
) > "$OUT"

echo "$OUT"
