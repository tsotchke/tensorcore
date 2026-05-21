#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-"$ROOT/build"}"
PREFIX="${PREFIX:-/private/tmp/tensorcore-install}"
if [ -z "${PY_PREFIX:-}" ]; then
    PY_PREFIX="$(mktemp -d /private/tmp/tensorcore-py-install.XXXXXX)"
fi
if [ -z "${WHEEL_DIR:-}" ]; then
    WHEEL_DIR="$(mktemp -d /private/tmp/tensorcore-wheels.XXXXXX)"
fi
if [ -z "${WHEEL_PREFIX:-}" ]; then
    WHEEL_PREFIX="$(mktemp -d /private/tmp/tensorcore-wheel-install.XXXXXX)"
fi
REQUIRE_GPU="${REQUIRE_GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
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
export EXPECTED_VERSION

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
    "$ROOT/setup.py" \
    "$ROOT/python/tensorcore/__init__.py" \
    "$ROOT/python/tests/test_basic.py"

echo "[tensorcore] python native loader policy"
"$PYTHON_BIN" - "$ROOT" "$BUILD_DIR/libtensorcore.dylib" <<'PY'
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

root = pathlib.Path(sys.argv[1])
dylib = pathlib.Path(sys.argv[2])
tmp = pathlib.Path(tempfile.mkdtemp(prefix="tensorcore-loader.", dir="/private/tmp"))
try:
    pkg = tmp / "tensorcore"
    pkg.mkdir()
    shutil.copy2(root / "python" / "tensorcore" / "__init__.py", pkg / "__init__.py")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp)
    env.pop("TENSORCORE_LIB", None)
    missing = subprocess.run(
        [sys.executable, "-c", "import tensorcore"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if missing.returncode == 0 or "package-local libtensorcore.dylib not found" not in missing.stderr:
        raise SystemExit("installed-package import did not reject missing native dylib")

    env["TENSORCORE_LIB"] = str(dylib)
    explicit = subprocess.run(
        [sys.executable, "-c", "import tensorcore as tc; print(tc.version())"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if explicit.returncode != 0:
        raise SystemExit(explicit.stderr)
    print(explicit.stdout.strip())
finally:
    shutil.rmtree(tmp, ignore_errors=True)
PY

echo "[tensorcore] python wheel"
mkdir -p "$WHEEL_DIR"
TENSORCORE_NATIVE_DIR="$PREFIX/lib" \
    "$PYTHON_BIN" -m pip wheel "$ROOT" --no-build-isolation -w "$WHEEL_DIR"

WHEEL_PATH="$("$PYTHON_BIN" - "$WHEEL_DIR" <<'PY'
import pathlib
import sys

wheels = sorted(pathlib.Path(sys.argv[1]).glob("tensorcore_apple-*.whl"))
if not wheels:
    raise SystemExit("no tensorcore_apple wheel was built")
print(wheels[-1])
PY
)"
"$PYTHON_BIN" - "$WHEEL_PATH" <<'PY'
import sys
import zipfile

required = {
    "tensorcore/libtensorcore.dylib",
    "tensorcore/tensorcore.metallib",
}
with zipfile.ZipFile(sys.argv[1]) as zf:
    names = set(zf.namelist())
missing = [
    suffix for suffix in sorted(required)
    if not any(name == suffix or name.endswith(f"/purelib/{suffix}") for name in names)
]
if missing:
    raise SystemExit(f"wheel missing native artifacts: {missing}")
PY

PY_VER="$("$PYTHON_BIN" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
echo "[tensorcore] python wheel install"
"$PYTHON_BIN" -m pip install "$WHEEL_PATH" --no-deps --prefix "$WHEEL_PREFIX"
WHEEL_SITE="$WHEEL_PREFIX/lib/$PY_VER/site-packages"
TENSORCORE_LIB= TC_METALLIB= "$PYTHON_BIN" - "$WHEEL_SITE" <<'PY'
import os
import sys

sys.path.insert(0, sys.argv[1])
import tensorcore as tc

expected = os.environ["EXPECTED_VERSION"]
lib = os.path.realpath(tc._find_lib())
expected_suffix = os.path.join("tensorcore", "libtensorcore.dylib")
if not lib.endswith(expected_suffix):
    raise SystemExit(f"package-local dylib was not selected: {lib}")
metallib = os.path.join(os.path.dirname(lib), "tensorcore.metallib")
if not os.path.exists(metallib):
    raise SystemExit(f"package-local metallib missing: {metallib}")
assert tc.version().startswith(f"tensorcore {expected}"), tc.version()
print(tc.version())
PY

echo "[tensorcore] python editable install"
"$PYTHON_BIN" -m pip install -e "$ROOT" --no-build-isolation --prefix "$PY_PREFIX"
PY_SITE="$PY_PREFIX/lib/$PY_VER/site-packages"
TENSORCORE_LIB="$PREFIX/lib/libtensorcore.dylib" "$PYTHON_BIN" - "$PY_SITE" <<'PY'
import site
import os
import sys

site.addsitedir(sys.argv[1])
import tensorcore as tc

expected = os.environ["EXPECTED_VERSION"]
assert tc.version().startswith(f"tensorcore {expected}"), tc.version()
print(tc.version())
PY

echo "[tensorcore] installed wheel python smoke"
if [ "$GPU_OK" = "1" ]; then
    TENSORCORE_LIB= TC_METALLIB= \
    TENSORCORE_TEST_INSTALLED=1 \
    PYTHONPATH="$WHEEL_SITE" \
        "$PYTHON_BIN" "$ROOT/python/tests/test_basic.py"
else
    TENSORCORE_LIB= TC_METALLIB= \
    PYTHONPATH="$WHEEL_SITE" \
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
