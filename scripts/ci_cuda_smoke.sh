#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${TC_CUDA_BUILD_DIR:-$ROOT/build-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIRE_CUDA="${REQUIRE_CUDA:-0}"
EVIDENCE_PATH="${TENSORCORE_CUDA_SMOKE_EVIDENCE_PATH:-}"

cmake -S "$ROOT" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DTC_ENABLE_METAL=OFF \
    -DTC_ENABLE_CUDA=ON
cmake --build "$BUILD" --parallel
ctest --test-dir "$BUILD" --output-on-failure

case "$(uname -s)" in
    Darwin) shared_lib="libtensorcore.dylib" ;;
    Linux)  shared_lib="libtensorcore.so" ;;
    *)      shared_lib="libtensorcore.so" ;;
esac

cuda_test_present=0
ctest_list="$(ctest --test-dir "$BUILD" -N)"
if printf '%s\n' "$ctest_list" | grep -q 'test_cuda_gemm'; then
    cuda_test_present=1
fi

TC_CUDA_BUILD_ENABLED="$cuda_test_present" \
TC_CUDA_REQUIRE="$REQUIRE_CUDA" \
TC_CUDA_EVIDENCE_PATH="$EVIDENCE_PATH" \
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
import struct
import subprocess
import sys

import tensorcore as tc


REQUIRED_FUNCTIONS = {
    "lib/cuda/gemm.cpp": [
        "cuda_gemm_bf16",
        "cuda_gemm_hgemm",
        "cuda_gemm_i8",
        "cuda_gemm_sgemm",
    ],
    "lib/cuda/training.cu": [
        "adamw_step_fp16_kernel",
        "adamw_step_fp32_kernel",
        "block_reduce_sum_f32",
    ],
}


def _git_value(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=os.environ["TC_ROOT"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _falsy(value):
    return str(value).strip().lower() in ("0", "false", "no", "off")


def _root_file_value(name):
    try:
        return (pathlib.Path(os.environ["TC_ROOT"]) / name).read_text(
            encoding="utf-8"
        ).strip()
    except Exception:
        return None


def _source_git_head():
    return (
        os.environ.get("TENSORCORE_SOURCE_GIT_HEAD")
        or _git_value("rev-parse", "HEAD")
        or _root_file_value(".tensorcore_source_head")
    )


def _source_git_dirty():
    override = os.environ.get("TENSORCORE_SOURCE_GIT_DIRTY")
    if override is not None:
        if _truthy(override):
            return True
        if _falsy(override):
            return False
        return None

    dirty = _git_value("status", "--short")
    if dirty is not None:
        return bool(dirty)

    marker = _root_file_value(".tensorcore_source_dirty")
    if marker is not None:
        if _truthy(marker):
            return True
        if _falsy(marker):
            return False
    return None


def _base_evidence():
    return {
        "schema_version": 1,
        "git_head": _source_git_head(),
        "git_dirty": _source_git_dirty(),
        "cuda_build_enabled": os.environ.get("TC_CUDA_BUILD_ENABLED") == "1",
        "require_cuda": os.environ.get("TC_CUDA_REQUIRE") == "1",
        "runtime_status": "not_run",
        "device_count": 0,
        "device": None,
        "backend": None,
        "f32_kernel": tc.cuda_last_kernel_name(),
        "f16_kernel": None,
        "gemm_kernels": {},
        "fallback_backend": None,
        "training_kernels": {},
        "files": {},
        "summary": {},
    }


def _function_line(rel_path, name):
    path = pathlib.Path(os.environ["TC_ROOT"]) / rel_path
    regex = re.compile(
        rf"^\s*(?:static\s+)?(?:__global__\s+)?(?:extern\s+\"C\"\s+)?"
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


def _add_function(evidence, rel_path, name):
    line = _function_line(rel_path, name)
    entry = evidence.setdefault("files", {}).setdefault(
        rel_path, {"executed_lines": [], "functions": {}}
    )
    if line not in entry["executed_lines"]:
        entry["executed_lines"].append(line)
    entry["functions"][name] = {"start_line": line, "executed_lines": [line]}


def _covered_functions(evidence):
    covered = []
    for rel_path, entry in evidence.get("files", {}).items():
        functions = entry.get("functions") if isinstance(entry, dict) else None
        if isinstance(functions, dict):
            covered.extend(f"{rel_path}:{name}" for name in functions)
    return sorted(covered)


def _finalize_evidence(evidence):
    for entry in evidence.get("files", {}).values():
        if isinstance(entry, dict) and isinstance(entry.get("executed_lines"), list):
            entry["executed_lines"] = sorted(set(entry["executed_lines"]))
    required = sorted(
        f"{path}:{name}" for path, names in REQUIRED_FUNCTIONS.items() for name in names
    )
    covered = _covered_functions(evidence)
    evidence["summary"] = {
        "required_functions": required,
        "covered_functions": covered,
        "missing_functions": sorted(set(required) - set(covered)),
    }
    return evidence


def _write_evidence(evidence):
    path = os.environ.get("TC_CUDA_EVIDENCE_PATH")
    if path:
        evidence = _finalize_evidence(evidence)
        pathlib.Path(path).write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")


def _fill_buffer(buf, values):
    ctypes.memmove(tc.buffer_map(buf), values, ctypes.sizeof(values))


def _half_bits(value):
    return int.from_bytes(struct.pack("<e", float(value)), "little")


def _half_value(bits):
    return struct.unpack("<e", int(bits).to_bytes(2, "little"))[0]


def _bf16_bits(value):
    raw = int.from_bytes(struct.pack("<f", float(value)), "little")
    return ((raw + 0x8000) >> 16) & 0xffff


def _bf16_value(bits):
    raw = int(bits) << 16
    return struct.unpack("<f", raw.to_bytes(4, "little"))[0]


def _expect_cuda_training_op(evidence, name, expected_kernel, fn):
    fn()
    backend = tc.last_backend_name()
    kernel = tc.cuda_last_kernel_name()
    evidence["training_kernels"][name] = {
        "backend": backend,
        "kernel": kernel,
    }
    if backend != "cuda":
        raise SystemExit(f"{name} backend was {backend}, expected cuda")
    if kernel != expected_kernel:
        raise SystemExit(f"{name} kernel was {kernel}, expected {expected_kernel}")


def _run_training_dispatch_smoke(ctx, evidence):
    bufs = []

    def alloc(nbytes):
        b = tc.buffer_alloc(ctx, nbytes)
        bufs.append(b)
        return b

    try:
        N, D = 2, 8
        x_vals = (ctypes.c_uint16 * (N * D))(
            *[_half_bits(((i % 7) - 3) * 0.125) for i in range(N * D)]
        )
        gamma_vals = (ctypes.c_uint16 * D)(*[_half_bits(0.75 + 0.02 * i) for i in range(D)])
        dy_vals = (ctypes.c_uint16 * (N * D))(
            *[_half_bits(((i % 5) - 2) * 0.05) for i in range(N * D)]
        )
        X = alloc(ctypes.sizeof(x_vals))
        gamma = alloc(ctypes.sizeof(gamma_vals))
        Y = alloc(ctypes.sizeof(x_vals))
        rstd = alloc(N * ctypes.sizeof(ctypes.c_float))
        dY = alloc(ctypes.sizeof(dy_vals))
        dX = alloc(ctypes.sizeof(x_vals))
        dgamma = alloc(D * ctypes.sizeof(ctypes.c_float))
        _fill_buffer(X, x_vals)
        _fill_buffer(gamma, gamma_vals)
        _fill_buffer(dY, dy_vals)
        _expect_cuda_training_op(
            evidence, "rmsnorm_forward", "cuda_rmsnorm_forward",
            lambda: tc.rmsnorm_forward(ctx, X, gamma, Y, rstd, N, D),
        )
        _expect_cuda_training_op(
            evidence, "rmsnorm_backward", "cuda_rmsnorm_backward",
            lambda: tc.rmsnorm_backward(ctx, X, gamma, dY, rstd, dX, dgamma, N, D),
        )

        beta_vals = (ctypes.c_uint16 * D)(*[_half_bits(0.01 * ((i % 3) - 1)) for i in range(D)])
        beta = alloc(ctypes.sizeof(beta_vals))
        ln_y = alloc(ctypes.sizeof(x_vals))
        mean = alloc(N * ctypes.sizeof(ctypes.c_float))
        ln_rstd = alloc(N * ctypes.sizeof(ctypes.c_float))
        ln_dx = alloc(ctypes.sizeof(x_vals))
        _fill_buffer(beta, beta_vals)
        _expect_cuda_training_op(
            evidence, "layernorm_forward", "cuda_layernorm_forward",
            lambda: tc.layernorm_forward(ctx, X, gamma, beta, ln_y, mean, ln_rstd, N, D),
        )
        _expect_cuda_training_op(
            evidence, "layernorm_backward", "cuda_layernorm_backward",
            lambda: tc.layernorm_backward(ctx, X, gamma, dY, mean, ln_rstd, ln_dx, N, D),
        )

        P = 16
        gate_vals = (ctypes.c_uint16 * P)(*[_half_bits(((i % 9) - 4) * 0.125) for i in range(P)])
        up_vals = (ctypes.c_uint16 * P)(*[_half_bits(0.2 + 0.01 * i) for i in range(P)])
        dout_vals = (ctypes.c_uint16 * P)(*[_half_bits(0.01 * ((i % 5) - 2)) for i in range(P)])
        gate = alloc(ctypes.sizeof(gate_vals))
        up = alloc(ctypes.sizeof(up_vals))
        out = alloc(ctypes.sizeof(gate_vals))
        dout = alloc(ctypes.sizeof(dout_vals))
        dgate = alloc(ctypes.sizeof(gate_vals))
        dup = alloc(ctypes.sizeof(gate_vals))
        _fill_buffer(gate, gate_vals)
        _fill_buffer(up, up_vals)
        _fill_buffer(dout, dout_vals)
        _expect_cuda_training_op(
            evidence, "swiglu_forward", "cuda_swiglu_forward",
            lambda: tc.swiglu_forward(ctx, gate, up, out, P),
        )
        _expect_cuda_training_op(
            evidence, "swiglu_backward", "cuda_swiglu_backward",
            lambda: tc.swiglu_backward(ctx, gate, up, dout, dgate, dup, P),
        )

        soft_x_vals = (ctypes.c_uint16 * (N * D))(
            *[_half_bits(((i % 11) - 5) * 0.2) for i in range(N * D)]
        )
        soft_dy_vals = (ctypes.c_uint16 * (N * D))(
            *[_half_bits(((i % 7) - 3) * 0.03) for i in range(N * D)]
        )
        SX = alloc(ctypes.sizeof(soft_x_vals))
        SY = alloc(ctypes.sizeof(soft_x_vals))
        SdY = alloc(ctypes.sizeof(soft_dy_vals))
        SdX = alloc(ctypes.sizeof(soft_x_vals))
        _fill_buffer(SX, soft_x_vals)
        _fill_buffer(SdY, soft_dy_vals)
        _expect_cuda_training_op(
            evidence, "softmax_forward", "cuda_softmax_forward",
            lambda: tc.softmax_forward(ctx, SX, SY, N, D),
        )
        _expect_cuda_training_op(
            evidence, "softmax_backward", "cuda_softmax_backward",
            lambda: tc.softmax_backward(ctx, SY, SdY, SdX, N, D),
        )

        B, H, S, RD = 1, 2, 4, 32
        rope_vals = (ctypes.c_uint16 * (B * H * S * RD))(
            *[_half_bits(((i % 13) - 6) * 0.05) for i in range(B * H * S * RD)]
        )
        drope_vals = (ctypes.c_uint16 * (B * H * S * RD))(
            *[_half_bits(((i % 11) - 5) * 0.04) for i in range(B * H * S * RD)]
        )
        cos_vals = (ctypes.c_float * (S * (RD // 2)))()
        sin_vals = (ctypes.c_float * (S * (RD // 2)))()
        for pos in range(S):
            for d in range(RD // 2):
                theta = pos / (10000.0 ** (2.0 * d / RD))
                cos_vals[pos * (RD // 2) + d] = math.cos(theta)
                sin_vals[pos * (RD // 2) + d] = math.sin(theta)
        rope_x = alloc(ctypes.sizeof(rope_vals))
        rope_dx = alloc(ctypes.sizeof(drope_vals))
        rope_cos = alloc(ctypes.sizeof(cos_vals))
        rope_sin = alloc(ctypes.sizeof(sin_vals))
        _fill_buffer(rope_x, rope_vals)
        _fill_buffer(rope_dx, drope_vals)
        _fill_buffer(rope_cos, cos_vals)
        _fill_buffer(rope_sin, sin_vals)
        _expect_cuda_training_op(
            evidence, "rope_forward", "cuda_rope_forward",
            lambda: tc.rope_forward(ctx, rope_x, rope_cos, rope_sin, B, H, S, RD),
        )
        _expect_cuda_training_op(
            evidence, "rope_backward", "cuda_rope_backward",
            lambda: tc.rope_backward(ctx, rope_dx, rope_cos, rope_sin, B, H, S, RD),
        )

        A = 16
        p_vals = (ctypes.c_float * A)(*[0.1 + 0.001 * i for i in range(A)])
        z_vals = (ctypes.c_float * A)(*[0.0 for _ in range(A)])
        g32_vals = (ctypes.c_float * A)(*[0.01 * ((i % 5) - 2) for i in range(A)])
        g16_vals = (ctypes.c_uint16 * A)(*[_half_bits(0.01 * ((i % 5) - 2)) for i in range(A)])
        P32 = alloc(ctypes.sizeof(p_vals))
        M32 = alloc(ctypes.sizeof(z_vals))
        V32 = alloc(ctypes.sizeof(z_vals))
        G32 = alloc(ctypes.sizeof(g32_vals))
        P16 = alloc(ctypes.sizeof(p_vals))
        M16 = alloc(ctypes.sizeof(z_vals))
        V16 = alloc(ctypes.sizeof(z_vals))
        G16 = alloc(ctypes.sizeof(g16_vals))
        for buf, vals in ((P32, p_vals), (M32, z_vals), (V32, z_vals), (G32, g32_vals),
                          (P16, p_vals), (M16, z_vals), (V16, z_vals), (G16, g16_vals)):
            _fill_buffer(buf, vals)
        _expect_cuda_training_op(
            evidence, "adamw_step_fp32", "cuda_adamw_step_fp32",
            lambda: tc.adamw_step(ctx, P32, M32, V32, G32, "f32", A,
                                  1e-3, 0.9, 0.999, 1e-8, 0.01, 0.1, 0.001),
        )
        _expect_cuda_training_op(
            evidence, "adamw_step_fp16", "cuda_adamw_step_fp16",
            lambda: tc.adamw_step(ctx, P16, M16, V16, G16, "f16", A,
                                  1e-3, 0.9, 0.999, 1e-8, 0.01, 0.1, 0.001),
        )
    finally:
        for buf in reversed(bufs):
            tc.buffer_free(ctx, buf)


evidence = _base_evidence()
require = evidence["require_cuda"]

if not evidence["cuda_build_enabled"]:
    evidence["runtime_status"] = "skipped_not_built"
    _write_evidence(evidence)
    print("CUDA smoke skipped: TC_ENABLE_CUDA runtime dependencies not found")
    sys.exit(1 if require else 0)

ctx = tc.init()
try:
    try:
        tc.cuda_init(ctx)
    except tc.TensorcoreError as exc:
        evidence["runtime_status"] = "skipped_runtime_unavailable"
        evidence["init_status"] = exc.status
        evidence["init_status_string"] = tc.status_string(exc.status)
        _write_evidence(evidence)
        print(f"CUDA smoke skipped: {evidence['init_status_string']}")
        sys.exit(1 if require else 0)

    if tc.cuda_device_count() <= 0:
        evidence["runtime_status"] = "skipped_runtime_unavailable"
        evidence["device_count"] = 0
        _write_evidence(evidence)
        print("CUDA smoke skipped: no visible CUDA device")
        sys.exit(1 if require else 0)

    info = tc.cuda_device_at(0)
    evidence["device"] = info
    evidence["device_count"] = tc.cuda_device_count()

    A32_vals = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
    B32_vals = (ctypes.c_float * 4)(5.0, 6.0, 7.0, 8.0)
    C32_vals = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0)
    A32 = tc.buffer_alloc(ctx, ctypes.sizeof(A32_vals))
    B32 = tc.buffer_alloc(ctx, ctypes.sizeof(B32_vals))
    C32 = tc.buffer_alloc(ctx, ctypes.sizeof(C32_vals))
    try:
        _fill_buffer(A32, A32_vals)
        _fill_buffer(B32, B32_vals)
        _fill_buffer(C32, C32_vals)
        tc.gemm(ctx, A32, B32, C32, 2, 2, 2, dtype="f32", accum="f32")
        out32 = (ctypes.c_float * 4).from_address(tc.buffer_map(C32).value)
        expected = (19.0, 22.0, 43.0, 50.0)
        if any(math.fabs(out32[i] - expected[i]) > 1e-4 for i in range(4)):
            raise SystemExit(f"bad CUDA f32 GEMM output: {[out32[i] for i in range(4)]}")
        evidence["backend"] = tc.last_backend_name()
        evidence["f32_kernel"] = tc.cuda_last_kernel_name()
        if evidence["backend"] != "cuda":
            raise SystemExit(f"f32 backend was {evidence['backend']}, expected cuda")
        if evidence["f32_kernel"] != "cublas_sgemm_managed":
            raise SystemExit(
                f"f32 kernel was {evidence['f32_kernel']}, expected cublas_sgemm_managed"
            )
        evidence["gemm_kernels"]["cuda_gemm_sgemm"] = {
            "status": "passed",
            "backend": evidence["backend"],
            "kernel": evidence["f32_kernel"],
        }
    finally:
        tc.buffer_free(ctx, A32)
        tc.buffer_free(ctx, B32)
        tc.buffer_free(ctx, C32)

    A16_vals = (ctypes.c_uint16 * 4)(_half_bits(1.0), _half_bits(2.0),
                                     _half_bits(3.0), _half_bits(4.0))
    B16_vals = (ctypes.c_uint16 * 4)(_half_bits(5.0), _half_bits(6.0),
                                     _half_bits(7.0), _half_bits(8.0))
    C16_vals = (ctypes.c_uint16 * 4)(0, 0, 0, 0)
    A16 = tc.buffer_alloc(ctx, ctypes.sizeof(A16_vals))
    B16 = tc.buffer_alloc(ctx, ctypes.sizeof(B16_vals))
    C16 = tc.buffer_alloc(ctx, ctypes.sizeof(C16_vals))
    try:
        _fill_buffer(A16, A16_vals)
        _fill_buffer(B16, B16_vals)
        _fill_buffer(C16, C16_vals)
        tc.gemm(ctx, A16, B16, C16, 2, 2, 2, dtype="f16", accum="f32")
        out16_bits = (ctypes.c_uint16 * 4).from_address(tc.buffer_map(C16).value)
        out16 = [_half_value(out16_bits[i]) for i in range(4)]
        expected = (19.0, 22.0, 43.0, 50.0)
        if any(math.fabs(out16[i] - expected[i]) > 1e-3 for i in range(4)):
            raise SystemExit(f"bad CUDA f16 GEMM output: {out16}")
        if tc.last_backend_name() != "cuda":
            raise SystemExit(f"f16 backend was {tc.last_backend_name()}, expected cuda")
        evidence["f16_kernel"] = tc.cuda_last_kernel_name()
        if evidence["f16_kernel"] != "cublas_gemmex_fp16_tensorop_managed":
            raise SystemExit(
                "f16 kernel was "
                f"{evidence['f16_kernel']}, expected cublas_gemmex_fp16_tensorop_managed"
            )
        evidence["gemm_kernels"]["cuda_gemm_hgemm"] = {
            "status": "passed",
            "backend": "cuda",
            "kernel": evidence["f16_kernel"],
        }
    finally:
        tc.buffer_free(ctx, A16)
        tc.buffer_free(ctx, B16)
        tc.buffer_free(ctx, C16)

    if info.get("supports_bf16"):
        Abf_vals = (ctypes.c_uint16 * 4)(
            _bf16_bits(1.0), _bf16_bits(2.0), _bf16_bits(3.0), _bf16_bits(4.0)
        )
        Bbf_vals = (ctypes.c_uint16 * 4)(
            _bf16_bits(5.0), _bf16_bits(6.0), _bf16_bits(7.0), _bf16_bits(8.0)
        )
        Cbf_vals = (ctypes.c_uint16 * 4)(0, 0, 0, 0)
        Abf = tc.buffer_alloc(ctx, ctypes.sizeof(Abf_vals))
        Bbf = tc.buffer_alloc(ctx, ctypes.sizeof(Bbf_vals))
        Cbf = tc.buffer_alloc(ctx, ctypes.sizeof(Cbf_vals))
        try:
            _fill_buffer(Abf, Abf_vals)
            _fill_buffer(Bbf, Bbf_vals)
            _fill_buffer(Cbf, Cbf_vals)
            tc.gemm(ctx, Abf, Bbf, Cbf, 2, 2, 2, dtype="bf16", accum="f32")
            outbf_bits = (ctypes.c_uint16 * 4).from_address(tc.buffer_map(Cbf).value)
            outbf = [_bf16_value(outbf_bits[i]) for i in range(4)]
            expected = (19.0, 22.0, 43.0, 50.0)
            if any(math.fabs(outbf[i] - expected[i]) > 1e-3 for i in range(4)):
                raise SystemExit(f"bad CUDA bf16 GEMM output: {outbf}")
            kernel = tc.cuda_last_kernel_name()
            if tc.last_backend_name() != "cuda":
                raise SystemExit(f"bf16 backend was {tc.last_backend_name()}, expected cuda")
            if kernel != "cublas_gemmex_bf16_tensorop_managed":
                raise SystemExit(f"bf16 kernel was {kernel}, expected cublas_gemmex_bf16_tensorop_managed")
            evidence["gemm_kernels"]["cuda_gemm_bf16"] = {
                "status": "passed",
                "backend": "cuda",
                "kernel": kernel,
            }
        finally:
            tc.buffer_free(ctx, Abf)
            tc.buffer_free(ctx, Bbf)
            tc.buffer_free(ctx, Cbf)
    else:
        evidence["gemm_kernels"]["cuda_gemm_bf16"] = {
            "status": "skipped_unsupported",
            "reason": "device_no_bf16",
        }

    if info.get("supports_int8_tensor_core"):
        M8, N8, K8 = 16, 16, 16
        A8_vals = (ctypes.c_int8 * (M8 * K8))(*([1] * (M8 * K8)))
        B8_vals = (ctypes.c_int8 * (K8 * N8))(*([1] * (K8 * N8)))
        C8_vals = (ctypes.c_int32 * (M8 * N8))(*([0] * (M8 * N8)))
        A8 = tc.buffer_alloc(ctx, ctypes.sizeof(A8_vals))
        B8 = tc.buffer_alloc(ctx, ctypes.sizeof(B8_vals))
        C8 = tc.buffer_alloc(ctx, ctypes.sizeof(C8_vals))
        try:
            _fill_buffer(A8, A8_vals)
            _fill_buffer(B8, B8_vals)
            _fill_buffer(C8, C8_vals)
            desc = tc.TCGemmDesc(
                M=M8, N=N8, K=K8,
                a_dtype=tc.TC_DTYPE_I8,
                b_dtype=tc.TC_DTYPE_I8,
                c_dtype=tc.TC_DTYPE_I32,
                accum_dtype=tc.TC_DTYPE_I32,
                transpose_a=False,
                transpose_b=False,
                alpha=1.0,
                beta=0.0,
                lda=0,
                ldb=0,
                ldc=0,
            )
            tc._check(tc._lib.tc_gemm(
                tc._as_handle(ctx), ctypes.byref(desc),
                tc._as_handle(A8), tc._as_handle(B8), tc._as_handle(C8),
            ))
            out8 = (ctypes.c_int32 * (M8 * N8)).from_address(tc.buffer_map(C8).value)
            if any(out8[i] != K8 for i in range(M8 * N8)):
                sample = [out8[i] for i in range(min(8, M8 * N8))]
                raise SystemExit(f"bad CUDA int8 GEMM output: {sample}")
            kernel = tc.cuda_last_kernel_name()
            if tc.last_backend_name() != "cuda":
                raise SystemExit(f"int8 backend was {tc.last_backend_name()}, expected cuda")
            if kernel != "cublas_gemmex_i8_tensorop_managed":
                raise SystemExit(f"int8 kernel was {kernel}, expected cublas_gemmex_i8_tensorop_managed")
            evidence["gemm_kernels"]["cuda_gemm_i8"] = {
                "status": "passed",
                "backend": "cuda",
                "kernel": kernel,
            }
        finally:
            tc.buffer_free(ctx, A8)
            tc.buffer_free(ctx, B8)
            tc.buffer_free(ctx, C8)
    else:
        evidence["gemm_kernels"]["cuda_gemm_i8"] = {
            "status": "skipped_unsupported",
            "reason": "device_no_int8_tensor_core",
        }

    os.environ["TC_DISABLE_CUDA_GEMM"] = "1"
    A32 = tc.buffer_alloc(ctx, ctypes.sizeof(A32_vals))
    B32 = tc.buffer_alloc(ctx, ctypes.sizeof(B32_vals))
    C32 = tc.buffer_alloc(ctx, ctypes.sizeof(C32_vals))
    try:
        _fill_buffer(A32, A32_vals)
        _fill_buffer(B32, B32_vals)
        _fill_buffer(C32, C32_vals)
        tc.gemm(ctx, A32, B32, C32, 2, 2, 2, dtype="f32", accum="f32")
        if tc.last_backend_name() == "cuda":
            raise SystemExit("TC_DISABLE_CUDA_GEMM did not force CPU fallback")
        evidence["fallback_backend"] = tc.last_backend_name()
    finally:
        tc.buffer_free(ctx, A32)
        tc.buffer_free(ctx, B32)
        tc.buffer_free(ctx, C32)
        os.environ.pop("TC_DISABLE_CUDA_GEMM", None)

    _run_training_dispatch_smoke(ctx, evidence)
    _add_function(evidence, "lib/cuda/gemm.cpp", "cuda_gemm_sgemm")
    _add_function(evidence, "lib/cuda/gemm.cpp", "cuda_gemm_hgemm")
    if evidence["gemm_kernels"].get("cuda_gemm_bf16", {}).get("status") == "passed":
        _add_function(evidence, "lib/cuda/gemm.cpp", "cuda_gemm_bf16")
    if evidence["gemm_kernels"].get("cuda_gemm_i8", {}).get("status") == "passed":
        _add_function(evidence, "lib/cuda/gemm.cpp", "cuda_gemm_i8")
    _add_function(evidence, "lib/cuda/training.cu", "adamw_step_fp32_kernel")
    _add_function(evidence, "lib/cuda/training.cu", "adamw_step_fp16_kernel")
    _add_function(evidence, "lib/cuda/training.cu", "block_reduce_sum_f32")
    evidence["runtime_status"] = "passed"
    _write_evidence(evidence)
    print(
        "CUDA smoke OK: "
        f"{info['device_name']} cc={info['compute_capability']} "
        "f32=cublas_sgemm_managed f16=cublas_gemmex_fp16_tensorop_managed "
        f"training_ops={len(evidence['training_kernels'])}"
    )
finally:
    tc.shutdown(ctx)
PY
