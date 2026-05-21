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
    "include/tensorcore/tensorcore.h"
    "include/tensorcore/status.h"
    "include/tensorcore/dtype.h"
    "include/tensorcore/device.h"
    "include/tensorcore/gemm.h"
    "include/tensorcore/attention.h"
    "include/tensorcore/training.h"
    "include/tensorcore/conv.h"
    "include/tensorcore/quantized.h"
    "include/tensorcore/distributed.h"
    "include/tensorcore/gguf.h"
    "lib/libtensorcore.a"
    "lib/libtensorcore.dylib"
    "lib/tensorcore.metallib"
    "lib/cmake/tensorcore/tensorcoreConfig.cmake"
    "lib/cmake/tensorcore/tensorcoreConfigVersion.cmake"
    "lib/pkgconfig/tensorcore.pc"
)

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

if command -v lipo >/dev/null 2>&1; then
    archs="$(lipo -archs "$sdk/lib/libtensorcore.dylib")"
    host_arch="$(uname -m)"
    case " $archs " in
        *" $host_arch "*) ;;
        *)
            echo "libtensorcore.dylib archs '$archs' do not include host arch '$host_arch'" >&2
            exit 1
            ;;
    esac
fi

consumer="$tmpdir/consumer"
mkdir -p "$consumer"
cat > "$consumer/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.20)
project(tensorcore_native_sdk_consumer LANGUAGES C)

find_package(tensorcore CONFIG REQUIRED)

add_executable(consumer main.c)
target_link_libraries(consumer PRIVATE tensorcore::tensorcore_shared)
tensorcore_copy_metallib(consumer)

add_executable(static_consumer main.c)
target_link_libraries(static_consumer PRIVATE tensorcore::tensorcore)
tensorcore_copy_metallib(static_consumer)
CMAKE
cat > "$consumer/main.c" <<'C'
#include <stdio.h>
#include <string.h>
#include "tensorcore/tensorcore.h"

#define TC_STR2(x) #x
#define TC_STR(x) TC_STR2(x)

int main(void) {
    const char* expected =
        "tensorcore " TC_STR(TENSORCORE_VERSION_MAJOR)
        "." TC_STR(TENSORCORE_VERSION_MINOR)
        "." TC_STR(TENSORCORE_VERSION_PATCH);
    if (strcmp(tc_version(), expected) != 0) {
        fprintf(stderr, "unexpected version: %s\n", tc_version());
        return 1;
    }
    if (tc_dtype_size(TC_DTYPE_BF16) != 2 ||
        strcmp(tc_dtype_name(TC_DTYPE_FP53), "fp53") != 0 ||
        strcmp(tc_backend_name(TC_BACKEND_ACCELERATE_CPU), "accelerate_cpu") != 0) {
        return 1;
    }
    tc_gguf_tensor_info t = {0};
    t.n_dims = 2;
    t.dims[0] = 32;
    t.dims[1] = 1;
    t.type = TC_GGUF_TYPE_Q4_0;
    t.n_bytes = tc_quantized_size(TC_QUANT_Q4_0, 1, 32);
    tc_gguf_quantized_matrix_info q = {0};
    tc_status_t s = tc_gguf_tensor_quantized_matrix_info(&t, &q);
    if (s != TC_OK || q.N != 1 || q.K != 32 || q.quant_type != TC_QUANT_Q4_0) {
        return 1;
    }
    printf("%s\n", tc_version());
    return 0;
}
C

cmake -S "$consumer" -B "$consumer/build" -DCMAKE_PREFIX_PATH="$sdk"
cmake --build "$consumer/build"
if [ ! -f "$consumer/build/tensorcore.metallib" ]; then
    echo "tensorcore_copy_metallib did not copy tensorcore.metallib" >&2
    exit 1
fi
DYLD_LIBRARY_PATH="$sdk/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
    "$consumer/build/consumer"
"$consumer/build/static_consumer"

if command -v pkg-config >/dev/null 2>&1; then
    pkg_version="$(PKG_CONFIG_PATH="$sdk/lib/pkgconfig" pkg-config --modversion tensorcore)"
    if [ "$pkg_version" != "$EXPECTED_VERSION" ]; then
        echo "pkg-config version mismatch: expected $EXPECTED_VERSION, got $pkg_version" >&2
        exit 1
    fi
    "$CC_BIN" "$consumer/main.c" \
        $(PKG_CONFIG_PATH="$sdk/lib/pkgconfig" pkg-config --cflags --libs tensorcore) \
        -o "$consumer/pkg-consumer"
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
