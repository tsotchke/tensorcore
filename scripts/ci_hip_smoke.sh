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
import re
import subprocess
import sys
import struct

sys.path.insert(0, str(pathlib.Path(os.environ["TC_ROOT"]) / "scripts"))
import probe_hip_toolchain

import tensorcore as tc


REQUIRED_FUNCTIONS = {
    "lib/hip/gemm.cpp": [
        "hip_gemm_hgemm",
        "hip_gemm_sgemm",
    ],
}


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
        "gemm_kernels": {},
        "fallback_backend": None,
        "files": {},
        "summary": {},
        "toolchain": probe_hip_toolchain.collect_evidence(os.environ["TC_ROOT"]),
    }


def function_line(rel_path, name):
    path = pathlib.Path(os.environ["TC_ROOT"]) / rel_path
    regex = re.compile(
        rf"^\s*(?:static\s+)?(?:extern\s+\"C\"\s+)?"
        rf"(?:[A-Za-z_][\w:<>,\s\*&]*\s+)+{re.escape(name)}\s*\("
    )
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return 1
    for index, line in enumerate(lines, start=1):
        if not regex.search(line):
            continue
        signature = line
        for continuation in lines[index:]:
            signature += "\n" + continuation
            if "{" in continuation or ";" in continuation:
                break
        if "{" in signature and (";" not in signature or signature.index("{") < signature.index(";")):
            return index
    return 1


def add_function(evidence, rel_path, name):
    line = function_line(rel_path, name)
    entry = evidence.setdefault("files", {}).setdefault(
        rel_path, {"executed_lines": [], "functions": {}}
    )
    if line not in entry["executed_lines"]:
        entry["executed_lines"].append(line)
    entry["functions"][name] = {"start_line": line, "executed_lines": [line]}


def covered_functions(evidence):
    covered = []
    for rel_path, entry in evidence.get("files", {}).items():
        functions = entry.get("functions") if isinstance(entry, dict) else None
        if isinstance(functions, dict):
            covered.extend(f"{rel_path}:{name}" for name in functions)
    return sorted(covered)


def finalize_evidence(evidence):
    for entry in evidence.get("files", {}).values():
        if isinstance(entry, dict) and isinstance(entry.get("executed_lines"), list):
            entry["executed_lines"] = sorted(set(entry["executed_lines"]))
    required = sorted(
        f"{path}:{name}" for path, names in REQUIRED_FUNCTIONS.items() for name in names
    )
    covered = covered_functions(evidence)
    evidence["summary"] = {
        "required_functions": required,
        "covered_functions": covered,
        "missing_functions": sorted(set(required) - set(covered)),
    }
    return evidence


def write_evidence(evidence):
    path = os.environ.get("TC_HIP_EVIDENCE_PATH")
    if path:
        evidence = finalize_evidence(evidence)
        pathlib.Path(path).write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")


def fill_buffer(buf, values):
    ctypes.memmove(tc.buffer_map(buf), values, ctypes.sizeof(values))


def half_bits(value):
    return int.from_bytes(struct.pack("<e", float(value)), "little")


def half_value(bits):
    return struct.unpack("<e", int(bits).to_bytes(2, "little"))[0]


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
        evidence["gemm_kernels"]["hip_gemm_sgemm"] = {
            "status": "passed",
            "backend": evidence["backend"],
            "kernel": evidence["kernel"],
        }

        A16_vals = (ctypes.c_uint16 * 4)(
            half_bits(1.0), half_bits(2.0), half_bits(3.0), half_bits(4.0)
        )
        B16_vals = (ctypes.c_uint16 * 4)(
            half_bits(5.0), half_bits(6.0), half_bits(7.0), half_bits(8.0)
        )
        C16_vals = (ctypes.c_uint16 * 4)(0, 0, 0, 0)
        A16 = B16 = C16 = None
        try:
            A16 = tc.buffer_alloc(ctx, ctypes.sizeof(A16_vals))
            B16 = tc.buffer_alloc(ctx, ctypes.sizeof(B16_vals))
            C16 = tc.buffer_alloc(ctx, ctypes.sizeof(C16_vals))
            fill_buffer(A16, A16_vals)
            fill_buffer(B16, B16_vals)
            fill_buffer(C16, C16_vals)
            tc.gemm(ctx, A16, B16, C16, 2, 2, 2, dtype="f16", accum="f32")
            out16_bits = (ctypes.c_uint16 * 4).from_address(tc.buffer_map(C16).value)
            out16 = [half_value(out16_bits[i]) for i in range(4)]
            if any(math.fabs(out16[i] - expected[i]) > 1e-3 for i in range(4)):
                raise SystemExit(f"bad HIP f16 GEMM output: {out16}")
            backend16 = tc.last_backend_name()
            kernel16 = tc.hip_last_kernel_name()
            if backend16 != "hip" or kernel16 != "hipblas_hgemm_staged":
                raise SystemExit(f"HIP hGEMM mismatch: {backend16} {kernel16}")
            evidence["gemm_kernels"]["hip_gemm_hgemm"] = {
                "status": "passed",
                "backend": backend16,
                "kernel": kernel16,
            }
        finally:
            for buf in (C16, B16, A16):
                if buf is not None:
                    tc.buffer_free(ctx, buf)

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

    add_function(evidence, "lib/hip/gemm.cpp", "hip_gemm_sgemm")
    add_function(evidence, "lib/hip/gemm.cpp", "hip_gemm_hgemm")
    evidence["runtime_status"] = "passed"
    write_evidence(evidence)
    print(
        "HIP smoke OK: "
        f"{info['device_name']} backend=hip "
        "kernels=hipblas_sgemm_staged,hipblas_hgemm_staged "
        f"fallback={evidence['fallback_backend']}"
    )
finally:
    tc.shutdown(ctx)
PY
