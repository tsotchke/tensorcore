#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${1:-${PREFIX:-/private/tmp/tensorcore-install}}"
OUT_DIR="${OUT_DIR:-${RUNNER_TEMP:-/tmp}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -d "$PREFIX" ]; then
    echo "native SDK prefix not found: $PREFIX" >&2
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
        echo "native SDK archives are supported on macOS and Linux; use the PowerShell archive script on Windows" >&2
        exit 2
        ;;
esac

case "$(uname -m)" in
    arm64|aarch64) archive_arch="arm64" ;;
    x86_64|amd64) archive_arch="x86_64" ;;
    *) archive_arch="$(uname -m)" ;;
esac

required+=("$shared_lib")
config_file="$PREFIX/lib/cmake/tensorcore/tensorcoreConfig.cmake"
if grep -Fq 'set(tensorcore_ENABLE_METAL "ON")' "$config_file"; then
    required+=("lib/tensorcore.metallib")
fi
for header in "$ROOT"/include/tensorcore/*.h; do
    required+=("include/tensorcore/$(basename "$header")")
done

missing=()
for rel in "${required[@]}"; do
    if [ ! -e "$PREFIX/$rel" ]; then
        missing+=("$rel")
    fi
done
if [ "${#missing[@]}" -ne 0 ]; then
    echo "native SDK prefix is missing required files:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
fi

if [ "$platform" = "macos" ] && command -v lipo >/dev/null 2>&1; then
    archs="$(lipo -archs "$PREFIX/$shared_lib")"
    host_arch="$(uname -m)"
    case " $archs " in
        *" $host_arch "*) ;;
        *)
            echo "libtensorcore.dylib archs '$archs' do not include host arch '$host_arch'" >&2
            exit 1
            ;;
    esac
fi

if command -v pkg-config >/dev/null 2>&1; then
    pkg_version="$(
        PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" pkg-config --modversion tensorcore 2>/dev/null || true
    )"
else
    pkg_version="$("$PYTHON_BIN" - "$PREFIX/lib/pkgconfig/tensorcore.pc" <<'PY'
import pathlib
import sys

for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if line.startswith("Version:"):
        print(line.split(":", 1)[1].strip())
        break
PY
)"
fi
if [ "$pkg_version" != "$EXPECTED_VERSION" ]; then
    echo "pkg-config version mismatch: expected $EXPECTED_VERSION, got ${pkg_version:-<none>}" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
archive="$OUT_DIR/tensorcore-native-sdk-${EXPECTED_VERSION}-${platform}-${archive_arch}.tar.gz"
COPYFILE_DISABLE=1 LC_ALL=C tar -czf "$archive" -C "$PREFIX" .

echo "$archive"
