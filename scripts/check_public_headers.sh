#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEADER_DIR="${HEADER_DIR:-"$ROOT/include/tensorcore"}"
INCLUDE_DIR="${INCLUDE_DIR:-"$ROOT/include"}"
CC_BIN="${CC:-cc}"
CXX_BIN="${CXX:-c++}"

if [ ! -d "$HEADER_DIR" ]; then
    echo "public header directory not found: $HEADER_DIR" >&2
    exit 1
fi

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/tensorcore-headers.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

count=0
for header in "$HEADER_DIR"/*.h; do
    name="$(basename "$header")"
    rel="tensorcore/$name"
    c_unit="$tmpdir/${name%.h}.c"
    cxx_unit="$tmpdir/${name%.h}.cpp"

    printf '#include "%s"\nint main(void) { return 0; }\n' "$rel" > "$c_unit"
    printf '#include "%s"\nint main() { return 0; }\n' "$rel" > "$cxx_unit"

    "$CC_BIN" -std=c11 -I "$INCLUDE_DIR" -fsyntax-only "$c_unit"
    "$CXX_BIN" -std=c++17 -I "$INCLUDE_DIR" -fsyntax-only "$cxx_unit"
    count=$((count + 1))
done

echo "public headers OK: $count headers"
