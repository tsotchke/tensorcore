#!/usr/bin/env bash
set -euo pipefail

PREFIX="${PREFIX:-${RUNNER_TEMP:-/tmp}/tensorcore-install}"
VENV="${VENV:-${RUNNER_TEMP:-/tmp}/tensorcore-venv}"
EXPECTED_VERSION="$(
    python3 - <<'PY'
import pathlib
import re

text = pathlib.Path("pyproject.toml").read_text()
match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
if not match:
    raise SystemExit("project.version not found in pyproject.toml")
print(match.group(1))
PY
)"

python3 -m venv "$VENV"
PYTHON="$VENV/bin/python3"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$VENV/bin/python"
fi

"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install -e . --no-build-isolation

TENSORCORE_LIB="$PREFIX/lib/libtensorcore.dylib" \
    "$PYTHON" - "$EXPECTED_VERSION" <<'PY'
import sys
import tensorcore as tc

expected = sys.argv[1]
actual = tc.version()
if not actual.startswith(f"tensorcore {expected}"):
    raise SystemExit(f"version mismatch: expected tensorcore {expected}, got {actual}")
if tc.status_string(tc.TC_OK) != "ok":
    raise SystemExit("status_string(TC_OK) mismatch")
if tc.dtype_name("fp53") != "fp53":
    raise SystemExit("dtype_name(fp53) mismatch")
if tc.backend_name(tc.TC_BACKEND_TENSOROPS_M5) != "tensorops_m5":
    raise SystemExit("backend_name(TENSOROPS_M5) mismatch")
if tc.last_backend_name() != "none":
    raise SystemExit("initial last_backend_name mismatch")
if tc.tensorops_gemm_kernel_name("f16") != "tc4_gemm_f16":
    raise SystemExit("tensorops f16 kernel selection mismatch")
if tc.tensorops_gemm_kernel_name("i8", "i32") is not None:
    raise SystemExit("tensorops unsupported i8 kernel selection mismatch")
if tc.TC_DIST_SINGLE != 0 or tc.TC_DIST_RING != 1 or tc.TC_DIST_GLOO != 2:
    raise SystemExit("distributed backend constants mismatch")
if tc.TC_REDUCE_SUM != 0 or tc.TC_REDUCE_AVG != 1:
    raise SystemExit("distributed reduce constants mismatch")
print(actual)
PY
