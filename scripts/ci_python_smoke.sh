#!/usr/bin/env bash
set -euo pipefail

PREFIX="${PREFIX:-${RUNNER_TEMP:-/tmp}/tensorcore-install}"
VENV="${VENV:-${RUNNER_TEMP:-/tmp}/tensorcore-venv}"

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e . --no-build-isolation

TENSORCORE_LIB="$PREFIX/lib/libtensorcore.dylib" \
    python -c 'import tensorcore as tc; print(tc.version())'
