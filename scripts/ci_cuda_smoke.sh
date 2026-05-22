#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${TC_CUDA_BUILD_DIR:-$ROOT/build-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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

LD_LIBRARY_PATH="$BUILD:${LD_LIBRARY_PATH:-}" \
PYTHONPATH="$ROOT/python" \
TENSORCORE_LIB="$BUILD/$shared_lib" \
TC_USE_CUDA_GEMM=1 \
"$PYTHON_BIN" - <<'PY'
import ctypes
import math
import struct

import tensorcore as tc


def _fill_buffer(buf, values):
    ctypes.memmove(tc.buffer_map(buf), values, ctypes.sizeof(values))


def _half_bits(value):
    return int.from_bytes(struct.pack("<e", float(value)), "little")


def _half_value(bits):
    return struct.unpack("<e", int(bits).to_bytes(2, "little"))[0]


ctx = tc.init()
try:
    tc.cuda_init(ctx)
    if tc.cuda_device_count() <= 0:
        raise SystemExit("CUDA smoke requires at least one visible CUDA device")
    info = tc.cuda_device_at(0)

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
        if tc.last_backend_name() != "cuda":
            raise SystemExit(f"f32 backend was {tc.last_backend_name()}, expected cuda")
        if tc.cuda_last_kernel_name() != "cublas_sgemm_managed":
            raise SystemExit(
                f"f32 kernel was {tc.cuda_last_kernel_name()}, expected cublas_sgemm_managed"
            )
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
        if tc.cuda_last_kernel_name() != "cublas_gemmex_fp16_tensorop_managed":
            raise SystemExit(
                "f16 kernel was "
                f"{tc.cuda_last_kernel_name()}, expected cublas_gemmex_fp16_tensorop_managed"
            )
    finally:
        tc.buffer_free(ctx, A16)
        tc.buffer_free(ctx, B16)
        tc.buffer_free(ctx, C16)

    print(
        "CUDA smoke OK: "
        f"{info['device_name']} cc={info['compute_capability']} "
        "f32=cublas_sgemm_managed f16=cublas_gemmex_fp16_tensorop_managed"
    )
finally:
    tc.shutdown(ctx)
PY
