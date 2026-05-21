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
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e . --no-build-isolation

TENSORCORE_LIB="$PREFIX/lib/libtensorcore.dylib" \
    python - "$EXPECTED_VERSION" <<'PY'
import sys
import tensorcore as tc

expected = sys.argv[1]
actual = tc.version()
if not actual.startswith(f"tensorcore {expected}"):
    raise SystemExit(f"version mismatch: expected tensorcore {expected}, got {actual}")
print(actual)
PY
