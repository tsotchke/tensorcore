#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${TC_HIP_BUILD_DIR:-$ROOT/build-hip}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIRE_HIP="${REQUIRE_HIP:-0}"
EVIDENCE_PATH="${TENSORCORE_HIP_SMOKE_EVIDENCE_PATH:-}"

cmake_prefix="${CMAKE_PREFIX_PATH:-}"
if [ -n "${TC_HIP_PREFIX:-}" ]; then
    cmake_prefix="${TC_HIP_PREFIX}${cmake_prefix:+;$cmake_prefix}"
elif [ -d "${HOME:-}/chipstar-install" ]; then
    cmake_prefix="${HOME}/chipstar-install${cmake_prefix:+;$cmake_prefix}"
fi

cmake_args=(
    -S "$ROOT"
    -B "$BUILD"
    -DCMAKE_BUILD_TYPE=Release
    -DTC_ENABLE_METAL=OFF
    -DTC_ENABLE_HIP=ON
)
if [ -n "$cmake_prefix" ]; then
    cmake_args+=("-DCMAKE_PREFIX_PATH=$cmake_prefix")
fi

cmake "${cmake_args[@]}"
cmake --build "$BUILD" --parallel
ctest --test-dir "$BUILD" --output-on-failure

case "$(uname -s)" in
    Darwin) shared_lib="libtensorcore.dylib" ;;
    Linux)  shared_lib="libtensorcore.so" ;;
    *)      shared_lib="libtensorcore.so" ;;
esac

hip_runtime_test_present=0
hip_gemm_test_present=0
ctest_list="$(ctest --test-dir "$BUILD" -N)"
if printf '%s\n' "$ctest_list" | grep -q 'test_hip_device'; then
    hip_runtime_test_present=1
fi
if printf '%s\n' "$ctest_list" | grep -q 'test_hip_gemm'; then
    hip_gemm_test_present=1
fi

TC_HIP_RUNTIME_TEST_PRESENT="$hip_runtime_test_present" \
TC_HIP_GEMM_TEST_PRESENT="$hip_gemm_test_present" \
TC_HIP_REQUIRE="$REQUIRE_HIP" \
TC_HIP_EVIDENCE_PATH="$EVIDENCE_PATH" \
TC_ROOT="$ROOT" \
LD_LIBRARY_PATH="$BUILD:${LD_LIBRARY_PATH:-}" \
PYTHONPATH="$ROOT/python" \
TENSORCORE_LIB="$BUILD/$shared_lib" \
"$PYTHON_BIN" - <<'PY'
import ctypes
import json
import math
import os
import pathlib
import subprocess
import sys

import tensorcore as tc


def git_value(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=os.environ["TC_ROOT"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def falsy(value):
    return str(value).strip().lower() in ("0", "false", "no", "off")


def root_file_value(name):
    try:
        return (pathlib.Path(os.environ["TC_ROOT"]) / name).read_text(
            encoding="utf-8"
        ).strip()
    except Exception:
        return None


def source_git_head():
    return (
        os.environ.get("TENSORCORE_SOURCE_GIT_HEAD")
        or git_value("rev-parse", "HEAD")
        or root_file_value(".tensorcore_source_head")
    )


def source_git_dirty():
    override = os.environ.get("TENSORCORE_SOURCE_GIT_DIRTY")
    if override is not None:
        if truthy(override):
            return True
        if falsy(override):
            return False
        return None

    dirty = git_value("status", "--short")
    if dirty is not None:
        return bool(dirty)

    marker = root_file_value(".tensorcore_source_dirty")
    if marker is not None:
        if truthy(marker):
            return True
        if falsy(marker):
            return False
    return None


def base_evidence():
    return {
        "schema_version": 1,
        "git_head": source_git_head(),
        "git_dirty": source_git_dirty(),
        "hip_build_enabled": os.environ.get("TC_HIP_RUNTIME_TEST_PRESENT") == "1",
        "hip_gemm_enabled": os.environ.get("TC_HIP_GEMM_TEST_PRESENT") == "1",
        "require_hip": os.environ.get("TC_HIP_REQUIRE") == "1",
        "runtime_status": "not_run",
        "device_count": 0,
        "device": None,
        "backend": None,
        "kernel": tc.hip_last_kernel_name(),
        "fallback_backend": None,
    }


def write_evidence(evidence):
    path = os.environ.get("TC_HIP_EVIDENCE_PATH")
    if path:
        pathlib.Path(path).write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")


def fill_buffer(buf, values):
    ctypes.memmove(tc.buffer_map(buf), values, ctypes.sizeof(values))


evidence = base_evidence()
require = evidence["require_hip"]

if not evidence["hip_build_enabled"]:
    evidence["runtime_status"] = "skipped_not_built"
    write_evidence(evidence)
    print("HIP smoke skipped: TC_ENABLE_HIP runtime dependencies not found")
    sys.exit(1 if require else 0)

ctx = tc.init()
try:
    evidence["device_count"] = tc.hip_device_count()
    try:
        tc.hip_init(ctx)
    except tc.TensorcoreError as exc:
        evidence["runtime_status"] = "skipped_runtime_unavailable"
        evidence["init_status"] = exc.status
        evidence["init_status_string"] = tc.status_string(exc.status)
        write_evidence(evidence)
        print(f"HIP smoke skipped: {evidence['init_status_string']}")
        sys.exit(1 if require else 0)

    info = tc.hip_device_at(0)
    evidence["device"] = info
    evidence["device_count"] = tc.hip_device_count()

    if not evidence["hip_gemm_enabled"]:
        evidence["runtime_status"] = "runtime_only_no_hipblas"
        write_evidence(evidence)
        print(
            "HIP runtime OK but hipBLAS GEMM skipped: "
            f"{info['device_name']} devices={evidence['device_count']}"
        )
        sys.exit(1 if require else 0)

    A_vals = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
    B_vals = (ctypes.c_float * 4)(5.0, 6.0, 7.0, 8.0)
    C_vals = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0)
    A = tc.buffer_alloc(ctx, ctypes.sizeof(A_vals))
    B = tc.buffer_alloc(ctx, ctypes.sizeof(B_vals))
    C = tc.buffer_alloc(ctx, ctypes.sizeof(C_vals))
    try:
        fill_buffer(A, A_vals)
        fill_buffer(B, B_vals)
        fill_buffer(C, C_vals)
        tc.gemm(ctx, A, B, C, 2, 2, 2, dtype="f32", accum="f32")
        out = (ctypes.c_float * 4).from_address(tc.buffer_map(C).value)
        expected = (19.0, 22.0, 43.0, 50.0)
        if any(math.fabs(out[i] - expected[i]) > 1e-4 for i in range(4)):
            raise SystemExit(f"bad HIP f32 GEMM output: {[out[i] for i in range(4)]}")
        evidence["backend"] = tc.last_backend_name()
        evidence["kernel"] = tc.hip_last_kernel_name()
        if evidence["backend"] != "hip" or evidence["kernel"] != "hipblas_sgemm_staged":
            raise SystemExit(f"HIP dispatch mismatch: {evidence['backend']} {evidence['kernel']}")

        os.environ["TC_DISABLE_HIP_GEMM"] = "1"
        fill_buffer(C, C_vals)
        tc.gemm(ctx, A, B, C, 2, 2, 2, dtype="f32", accum="f32")
        evidence["fallback_backend"] = tc.last_backend_name()
        if evidence["fallback_backend"] == "hip":
            raise SystemExit("TC_DISABLE_HIP_GEMM did not force fallback")
        os.environ.pop("TC_DISABLE_HIP_GEMM", None)
    finally:
        tc.buffer_free(ctx, A)
        tc.buffer_free(ctx, B)
        tc.buffer_free(ctx, C)

    evidence["runtime_status"] = "passed"
    write_evidence(evidence)
    print(
        "HIP smoke OK: "
        f"{info['device_name']} backend=hip kernel=hipblas_sgemm_staged "
        f"fallback={evidence['fallback_backend']}"
    )
finally:
    tc.shutdown(ctx)
PY
