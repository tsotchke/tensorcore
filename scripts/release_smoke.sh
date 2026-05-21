#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-"$ROOT/build"}"
PREFIX="${PREFIX:-/private/tmp/tensorcore-install}"
PY_PREFIX="${PY_PREFIX:-/private/tmp/tensorcore-py-install}"
WHEEL_DIR="${WHEEL_DIR:-/private/tmp/tensorcore-wheels}"
REQUIRE_GPU="${REQUIRE_GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[tensorcore] configure"
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

echo "[tensorcore] build"
cmake --build "$BUILD_DIR"

echo "[tensorcore] test"
GPU_OK=0
if "$BUILD_DIR/tests/test_device"; then
    GPU_OK=1
    ctest --test-dir "$BUILD_DIR" --output-on-failure
else
    if [ "$REQUIRE_GPU" = "1" ]; then
        echo "Metal device smoke failed and REQUIRE_GPU=1 was set." >&2
        exit 1
    fi
    echo "No usable Metal device in this environment; skipping GPU tests."
    ctest --test-dir "$BUILD_DIR" --output-on-failure -R 'distributed_ring'
fi

echo "[tensorcore] install"
cmake --install "$BUILD_DIR" --prefix "$PREFIX"

echo "[tensorcore] python syntax"
"$PYTHON_BIN" -m py_compile \
    "$ROOT/python/tensorcore/__init__.py" \
    "$ROOT/python/tests/test_basic.py"

echo "[tensorcore] python wheel"
mkdir -p "$WHEEL_DIR"
"$PYTHON_BIN" -m pip wheel "$ROOT" --no-build-isolation -w "$WHEEL_DIR"

echo "[tensorcore] python editable install"
"$PYTHON_BIN" -m pip install -e "$ROOT" --no-build-isolation --prefix "$PY_PREFIX"
PY_VER="$("$PYTHON_BIN" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
PY_SITE="$PY_PREFIX/lib/$PY_VER/site-packages"
TENSORCORE_LIB="$PREFIX/lib/libtensorcore.dylib" "$PYTHON_BIN" - "$PY_SITE" <<'PY'
import site
import sys

site.addsitedir(sys.argv[1])
import tensorcore as tc

assert tc.version().startswith("tensorcore 0.1.6"), tc.version()
print(tc.version())
PY

echo "[tensorcore] installed python smoke"
if [ "$GPU_OK" = "1" ]; then
    TENSORCORE_LIB="$PREFIX/lib/libtensorcore.dylib" \
    PYTHONPATH="$ROOT/python" \
        "$PYTHON_BIN" "$ROOT/python/tests/test_basic.py"
else
    TENSORCORE_LIB="$PREFIX/lib/libtensorcore.dylib" \
    PYTHONPATH="$ROOT/python" \
        "$PYTHON_BIN" -c 'import tensorcore as tc; print(tc.version())'
fi

echo "[tensorcore] out-of-tree CMake consumer"
CONSUMER_DIR="$(mktemp -d /private/tmp/tensorcore-consumer.XXXXXX)"
trap 'rm -rf "$CONSUMER_DIR"' EXIT
cat > "$CONSUMER_DIR/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.20)
project(tensorcore_consumer LANGUAGES C)

find_package(tensorcore CONFIG REQUIRED)

add_executable(consumer main.c)
target_link_libraries(consumer PRIVATE tensorcore::tensorcore_shared)

add_executable(static_consumer static_main.c)
target_link_libraries(static_consumer PRIVATE tensorcore::tensorcore)
tensorcore_copy_metallib(static_consumer)
CMAKE
cat > "$CONSUMER_DIR/main.c" <<'C'
#include <stdio.h>
#include "tensorcore/tensorcore.h"

int main(void) {
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
cat > "$CONSUMER_DIR/static_main.c" <<'C'
#include <stdio.h>
#include "tensorcore/tensorcore.h"

int main(void) {
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init: %s\n", tc_status_string(s));
        return 1;
    }
    tc_shutdown(ctx);
    printf("%s\n", tc_version());
    return 0;
}
C
cmake -S "$CONSUMER_DIR" -B "$CONSUMER_DIR/build" \
    -DCMAKE_PREFIX_PATH="$PREFIX"
cmake --build "$CONSUMER_DIR/build"
"$CONSUMER_DIR/build/consumer"
if [ "$GPU_OK" = "1" ]; then
    "$CONSUMER_DIR/build/static_consumer"
fi

echo "[tensorcore] pkg-config consumer"
if command -v pkg-config >/dev/null 2>&1; then
    CC_BIN="${CC:-cc}"
    PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" \
        "$CC_BIN" "$CONSUMER_DIR/main.c" \
        $(PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" pkg-config --cflags --libs tensorcore) \
        -o "$CONSUMER_DIR/pkg-consumer"
    "$CONSUMER_DIR/pkg-consumer"
else
    echo "pkg-config not found; skipping pkg-config consumer smoke."
fi

echo "[tensorcore] release smoke OK"
