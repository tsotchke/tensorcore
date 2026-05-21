#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
DYLIB="${DYLIB:-"$BUILD_DIR/libtensorcore.dylib"}"
EXPORTS_FILE="${EXPORTS_FILE:-cmake/tensorcore.exports}"
HEADER_DIR="${HEADER_DIR:-include/tensorcore}"

if [ ! -f "$DYLIB" ]; then
    echo "shared library not found: $DYLIB" >&2
    exit 1
fi
if [ ! -f "$EXPORTS_FILE" ]; then
    echo "exports file not found: $EXPORTS_FILE" >&2
    exit 1
fi
if [ ! -d "$HEADER_DIR" ]; then
    echo "public header directory not found: $HEADER_DIR" >&2
    exit 1
fi

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/tensorcore-exports.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

expected="$tmpdir/expected"
actual="$tmpdir/actual"
duplicates="$tmpdir/duplicates"
header_expected="$tmpdir/header_expected"
header_missing="$tmpdir/header_missing"
header_extra="$tmpdir/header_extra"
missing="$tmpdir/missing"
extra="$tmpdir/extra"

LC_ALL=C sed '/^[[:space:]]*$/d;/^[[:space:]]*#/d' "$EXPORTS_FILE" | sort > "$expected"
LC_ALL=C sed '/^[[:space:]]*$/d;/^[[:space:]]*#/d' "$EXPORTS_FILE" | sort | uniq -d > "$duplicates"
python3 - "$HEADER_DIR" > "$header_expected" <<'PY'
import pathlib
import re
import sys

header_dir = pathlib.Path(sys.argv[1])
inline_only = {"tc_dtype_size"}
symbols = set()

for path in sorted(header_dir.glob("*.h")):
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//.*", " ", text)
    for match in re.finditer(r"\btc_[A-Za-z0-9_]+\s*\(", text):
        symbol = match.group(0).split("(", 1)[0].strip()
        if symbol not in inline_only:
            symbols.add(f"_{symbol}")

for symbol in sorted(symbols):
    print(symbol)
PY
nm -gU "$DYLIB" | awk 'NF >= 3 { print $3 }' | grep '^_tc_' | LC_ALL=C sort > "$actual"

if [ -s "$duplicates" ]; then
    echo "duplicate public exports in $EXPORTS_FILE:" >&2
    cat "$duplicates" >&2
    exit 1
fi

comm -23 "$header_expected" "$expected" > "$header_missing"
comm -13 "$header_expected" "$expected" > "$header_extra"

if [ -s "$header_missing" ] || [ -s "$header_extra" ]; then
    if [ -s "$header_missing" ]; then
        echo "public header symbols missing from $EXPORTS_FILE:" >&2
        cat "$header_missing" >&2
    fi
    if [ -s "$header_extra" ]; then
        echo "exports not declared in public headers:" >&2
        cat "$header_extra" >&2
    fi
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
