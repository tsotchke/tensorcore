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
if tc.TC_TIER_L0_DEVICE != 0 or tc.TC_TIER_L4_REMOTE_NVME != 4:
    raise SystemExit("memory tier constants mismatch")
if tc.TC_TIER_HINT_HOT != 0 or tc.TC_TIER_HINT_ICE != 3:
    raise SystemExit("memory tier hint constants mismatch")
if tc.TC_HIP_VENDOR_UNKNOWN != 0 or tc.TC_HIP_VENDOR_NVIDIA != 2:
    raise SystemExit("HIP vendor constants mismatch")
if tc.TC_DILOCO_COMPRESS_NONE != 0 or tc.TC_DILOCO_COMPRESS_TOPK_01PCT != 4:
    raise SystemExit("DiLoCo compression constants mismatch")
if tc.TC_DILOCO_OUTER_SGD != 0 or tc.TC_DILOCO_OUTER_NESTEROV != 1:
    raise SystemExit("DiLoCo optimizer constants mismatch")
if tc.hip_device_count() != 0 or tc.hip_last_kernel_name() != "none":
    raise SystemExit("HIP inactive diagnostics mismatch")
if tc.cuda_device_count() != 0 or tc.cuda_last_kernel_name() != "none":
    raise SystemExit("CUDA inactive diagnostics mismatch")
if (tc.checkpoint_count_resident() != 0 or
        tc.checkpoint_count_discarded() != 0 or
        tc.checkpoint_total_bytes_discarded() != 0):
    raise SystemExit("checkpoint baseline counters mismatch")
print(actual)
PY
