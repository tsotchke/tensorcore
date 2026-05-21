#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
DYLIB="${DYLIB:-"$BUILD_DIR/libtensorcore.dylib"}"
EXPORTS_FILE="${EXPORTS_FILE:-cmake/tensorcore.exports}"

if [ ! -f "$DYLIB" ]; then
    echo "shared library not found: $DYLIB" >&2
    exit 1
fi
if [ ! -f "$EXPORTS_FILE" ]; then
    echo "exports file not found: $EXPORTS_FILE" >&2
    exit 1
fi

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/tensorcore-exports.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

expected="$tmpdir/expected"
actual="$tmpdir/actual"
duplicates="$tmpdir/duplicates"
missing="$tmpdir/missing"
extra="$tmpdir/extra"

LC_ALL=C sed '/^[[:space:]]*$/d;/^[[:space:]]*#/d' "$EXPORTS_FILE" | sort > "$expected"
LC_ALL=C sed '/^[[:space:]]*$/d;/^[[:space:]]*#/d' "$EXPORTS_FILE" | sort | uniq -d > "$duplicates"
nm -gU "$DYLIB" | awk 'NF >= 3 { print $3 }' | grep '^_tc_' | LC_ALL=C sort > "$actual"

if [ -s "$duplicates" ]; then
    echo "duplicate public exports in $EXPORTS_FILE:" >&2
    cat "$duplicates" >&2
    exit 1
fi

comm -23 "$expected" "$actual" > "$missing"
comm -13 "$expected" "$actual" > "$extra"

if [ -s "$missing" ] || [ -s "$extra" ]; then
    if [ -s "$missing" ]; then
        echo "expected public exports missing from $DYLIB:" >&2
        cat "$missing" >&2
    fi
    if [ -s "$extra" ]; then
        echo "unexpected public exports in $DYLIB:" >&2
        cat "$extra" >&2
    fi
    exit 1
fi

echo "public export surface OK: $(wc -l < "$actual" | tr -d ' ') symbols"
