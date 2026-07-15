#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="${1:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CC_BIN="${CC:-cc}"

if [ -z "$ARCHIVE" ]; then
    echo "usage: $0 <tensorcore-native-sdk-*.tar.gz>" >&2
    exit 2
fi
if [ ! -f "$ARCHIVE" ]; then
    echo "native SDK archive not found: $ARCHIVE" >&2
    exit 1
fi

EXPECTED_VERSION="$("$PYTHON_BIN" - "$ROOT/pyproject.toml" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
if not match:
    raise SystemExit("project.version not found in pyproject.toml")
print(match.group(1))
PY
)"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/tensorcore-sdk-check.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT
sdk="$tmpdir/sdk"
mkdir -p "$sdk"

LC_ALL=C tar -tzf "$ARCHIVE" | while IFS= read -r member; do
    case "$member" in
        ""|.|..|/*|../*|*/..|*/../*)
            echo "unsafe archive member: $member" >&2
            exit 1
            ;;
    esac
done
LC_ALL=C tar -xzf "$ARCHIVE" -C "$sdk"

required=(
    "lib/libtensorcore.a"
    "lib/cmake/tensorcore/tensorcoreConfig.cmake"
    "lib/cmake/tensorcore/tensorcoreConfigVersion.cmake"
    "lib/pkgconfig/tensorcore.pc"
)

case "$(uname -s)" in
    Darwin)
        platform="macos"
        shared_lib="lib/libtensorcore.dylib"
        ;;
    Linux)
        platform="linux"
        shared_lib="lib/libtensorcore.so"
        ;;
    *)
        echo "native SDK tarballs are supported on macOS and Linux; use the PowerShell checker on Windows" >&2
        exit 2
        ;;
esac

case "$(uname -m)" in
    arm64|aarch64) archive_arch="arm64" ;;
    x86_64|amd64) archive_arch="x86_64" ;;
    *) archive_arch="$(uname -m)" ;;
esac

expected_name="tensorcore-native-sdk-${EXPECTED_VERSION}-${platform}-${archive_arch}.tar.gz"
if [ "$(basename "$ARCHIVE")" != "$expected_name" ]; then
    echo "native SDK archive name mismatch: expected $expected_name" >&2
    exit 1
fi

required+=("$shared_lib")
config_file="$sdk/lib/cmake/tensorcore/tensorcoreConfig.cmake"
metal_enabled=0
if grep -Fq 'set(tensorcore_ENABLE_METAL "ON")' "$config_file"; then
    metal_enabled=1
    required+=("lib/tensorcore.metallib")
fi
for header in "$ROOT"/include/tensorcore/*.h; do
    required+=("include/tensorcore/$(basename "$header")")
done

missing=()
for rel in "${required[@]}"; do
    if [ ! -e "$sdk/$rel" ]; then
        missing+=("$rel")
    fi
done
if [ "${#missing[@]}" -ne 0 ]; then
    echo "native SDK archive is missing required files:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
fi

if LC_ALL=C grep -R -n -E '/Applications/.+Xcode|MacOSX\.sdk' \
    "$sdk/lib/cmake/tensorcore" "$sdk/lib/pkgconfig/tensorcore.pc" >&2; then
    echo "native SDK metadata contains build-machine SDK paths" >&2
    exit 1
fi

if [ "$platform" = "macos" ] && command -v lipo >/dev/null 2>&1; then
    archs="$(lipo -archs "$sdk/$shared_lib")"
    host_arch="$(uname -m)"
    case " $archs " in
        *" $host_arch "*) ;;
        *)
            echo "libtensorcore.dylib archs '$archs' do not include host arch '$host_arch'" >&2
            exit 1
            ;;
    esac
fi

if [ "$platform" = "macos" ] && command -v otool >/dev/null 2>&1; then
    "$PYTHON_BIN" - "$sdk/$shared_lib" "$EXPECTED_VERSION" <<'PY'
import re
import subprocess
import sys

dylib = sys.argv[1]
expected = sys.argv[2]
major, minor, _patch = expected.split(".", 2)
expected_compat = f"{major}.{minor}.0"

out = subprocess.check_output(["otool", "-L", dylib], text=True)
match = re.search(
    rf"@rpath/libtensorcore\.{re.escape(major)}\.dylib "
    r"\(compatibility version ([^,]+), current version ([^)]+)\)",
    out,
)
if not match:
    raise SystemExit(
        f"libtensorcore.dylib install name is not @rpath/libtensorcore.{major}.dylib"
    )

compat, current = match.groups()
if current != expected:
    raise SystemExit(f"libtensorcore.dylib current version mismatch: expected {expected}, got {current}")
if compat != expected_compat:
    raise SystemExit(
        f"libtensorcore.dylib compatibility version mismatch: "
        f"expected {expected_compat}, got {compat}"
    )
PY
fi

if [ "$platform" = "linux" ] && command -v readelf >/dev/null 2>&1; then
    expected_soname="libtensorcore.so.${EXPECTED_VERSION%%.*}"
    if ! readelf -d "$sdk/$shared_lib" | grep -F '(SONAME)' | grep -Fq "[$expected_soname]"; then
        echo "libtensorcore.so SONAME mismatch: expected $expected_soname" >&2
        exit 1
    fi
fi

consumer="$tmpdir/consumer"
mkdir -p "$consumer"
cmake -S "$ROOT/examples/native_sdk_consumer" -B "$consumer/build" \
    -DCMAKE_PREFIX_PATH="$sdk"
cmake --build "$consumer/build"
if [ "$metal_enabled" -eq 1 ] && [ ! -f "$consumer/build/tensorcore.metallib" ]; then
    echo "tensorcore_copy_metallib did not copy tensorcore.metallib" >&2
    exit 1
fi
DYLD_LIBRARY_PATH="$sdk/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
LD_LIBRARY_PATH="$sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$consumer/build/consumer_shared"
DYLD_LIBRARY_PATH="$sdk/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
LD_LIBRARY_PATH="$sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$consumer/build/consumer_cxx"
"$consumer/build/consumer_static"

if command -v pkg-config >/dev/null 2>&1; then
    pkg_version="$(PKG_CONFIG_PATH="$sdk/lib/pkgconfig" pkg-config --modversion tensorcore)"
    if [ "$pkg_version" != "$EXPECTED_VERSION" ]; then
        echo "pkg-config version mismatch: expected $EXPECTED_VERSION, got $pkg_version" >&2
        exit 1
    fi
    "$CC_BIN" "$ROOT/examples/native_sdk_consumer/main.c" \
        $(PKG_CONFIG_PATH="$sdk/lib/pkgconfig" pkg-config --cflags --libs tensorcore) \
        -o "$consumer/pkg-consumer"
    DYLD_LIBRARY_PATH="$sdk/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
    LD_LIBRARY_PATH="$sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$consumer/pkg-consumer"
else
    "$PYTHON_BIN" - "$sdk/lib/pkgconfig/tensorcore.pc" "$EXPECTED_VERSION" <<'PY'
import pathlib
import sys

pc = pathlib.Path(sys.argv[1]).read_text()
expected = sys.argv[2]
for line in pc.splitlines():
    if line.startswith("Version:"):
        actual = line.split(":", 1)[1].strip()
        if actual != expected:
            raise SystemExit(f"pkg-config version mismatch: expected {expected}, got {actual}")
        break
else:
    raise SystemExit("Version field missing from tensorcore.pc")
PY
fi

echo "native SDK archive OK: $ARCHIVE"
