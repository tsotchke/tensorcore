"""tensorcore — Python bindings.

Thin ctypes wrapper around the tensorcore C ABI. Loads libtensorcore.dylib
from a configured location and exposes the public surface.

Quick start:

    import tensorcore as tc
    import numpy as np

    ctx = tc.init()
    info = tc.device_info(ctx)
    print(f"device: {info.name}, family: Apple{info.family}")

    # fp16 GEMM
    M, N, K = 1024, 1024, 1024
    A = np.random.randn(M, K).astype(np.float16)
    B = np.random.randn(K, N).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    a, b, c = tc.buffer_alloc(ctx, A.nbytes), tc.buffer_alloc(ctx, B.nbytes), tc.buffer_alloc(ctx, C.nbytes)
    tc.buffer_write(a, A); tc.buffer_write(b, B)
    tc.gemm(ctx, a, b, c, M, N, K, dtype="f16")
    tc.buffer_read(c, C)

    print(f"max |C|: {np.abs(C).max()}")
    tc.shutdown(ctx)

For perf-critical loops, use the async variants and tc.stream_sync().
"""

import ctypes
import os
import sys
from ctypes import (
    c_int, c_uint, c_int32, c_int64, c_uint32, c_uint64, c_size_t,
    c_float, c_double, c_char_p, c_void_p, c_bool, POINTER, Structure, byref,
)

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _find_lib():
    env = os.environ.get("TENSORCORE_LIB")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "build", "libtensorcore.dylib"),
        os.path.join(here, "..", "..", "build", "libtensorcore.a"),  # static — won't load via ctypes
        "/usr/local/lib/libtensorcore.dylib",
    ]
    for p in candidates:
        if os.path.exists(p) and p.endswith(".dylib"):
            return p
    raise RuntimeError(
        "libtensorcore.dylib not found. Set TENSORCORE_LIB env var or "
        "build tensorcore as a shared library."
    )

try:
    _lib = ctypes.CDLL(_find_lib())
except Exception as e:
    print(f"[tensorcore] warning: lib not loaded ({e})", file=sys.stderr)
    _lib = None


# ---------------------------------------------------------------------------
# C ABI prototypes
# ---------------------------------------------------------------------------

TC_OK = 0
TC_DTYPE_F16 = 0
TC_DTYPE_BF16 = 1
TC_DTYPE_F32 = 2
TC_DTYPE_I8 = 3
TC_DTYPE_I32 = 4

_DTYPE_MAP = {
    "f16": TC_DTYPE_F16, "bf16": TC_DTYPE_BF16, "f32": TC_DTYPE_F32,
    "i8": TC_DTYPE_I8, "i32": TC_DTYPE_I32,
}


class TCDeviceInfo(Structure):
    _fields_ = [
        ("family",                       c_int),
        ("name",                         ctypes.c_char * 128),
        ("max_buffer_bytes",             c_uint64),
        ("recommended_working_set_bytes", c_uint64),
        ("max_threadgroup_memory",       c_uint32),
        ("max_threads_per_threadgroup",  c_uint32),
        ("thread_execution_width",       c_uint32),
        ("unified_memory",               c_bool),
        ("supports_bf16_simdgroup",      c_bool),
        ("supports_i8_simdgroup",        c_bool),
        ("supports_tensorops_m5",        c_bool),
        ("supports_fp64_native",         c_bool),
    ]


class TCGemmDesc(Structure):
    _fields_ = [
        ("M", c_int32), ("N", c_int32), ("K", c_int32),
        ("a_dtype", c_int), ("b_dtype", c_int), ("c_dtype", c_int), ("accum_dtype", c_int),
        ("transpose_a", c_bool), ("transpose_b", c_bool),
        ("alpha", c_float), ("beta", c_float),
        ("lda", c_int32), ("ldb", c_int32), ("ldc", c_int32),
    ]


if _lib is not None:
    _lib.tc_init.argtypes = [POINTER(c_void_p)];          _lib.tc_init.restype = c_int
    _lib.tc_shutdown.argtypes = [c_void_p];               _lib.tc_shutdown.restype = c_int
    _lib.tc_device_info_get.argtypes = [c_void_p, POINTER(TCDeviceInfo)]; _lib.tc_device_info_get.restype = c_int
    _lib.tc_buffer_alloc.argtypes = [c_void_p, c_size_t, POINTER(c_void_p)]; _lib.tc_buffer_alloc.restype = c_int
    _lib.tc_buffer_free.argtypes  = [c_void_p, c_void_p]; _lib.tc_buffer_free.restype  = c_int
    _lib.tc_buffer_map.argtypes   = [c_void_p, POINTER(c_void_p)]; _lib.tc_buffer_map.restype = c_int
    _lib.tc_buffer_size.argtypes  = [c_void_p];           _lib.tc_buffer_size.restype  = c_size_t
    _lib.tc_gemm.argtypes = [c_void_p, POINTER(TCGemmDesc), c_void_p, c_void_p, c_void_p]
    _lib.tc_gemm.restype  = c_int
    _lib.tc_status_string.argtypes = [c_int]; _lib.tc_status_string.restype = c_char_p
    _lib.tc_version.argtypes = []; _lib.tc_version.restype = c_char_p


# ---------------------------------------------------------------------------
# Pythonic surface
# ---------------------------------------------------------------------------

class TensorcoreError(RuntimeError):
    def __init__(self, status):
        msg = _lib.tc_status_string(status).decode() if _lib else f"status {status}"
        super().__init__(f"tensorcore error {status}: {msg}")
        self.status = status


def _check(status):
    if status != TC_OK:
        raise TensorcoreError(status)


def init():
    ctx = c_void_p()
    s = _lib.tc_init(byref(ctx))
    if s not in (TC_OK, -2):  # -2 = already_initialized
        _check(s)
    return ctx


def shutdown(ctx):
    _check(_lib.tc_shutdown(ctx))


def device_info(ctx):
    info = TCDeviceInfo()
    _check(_lib.tc_device_info_get(ctx, byref(info)))
    info.name_str = info.name.decode("utf-8", "replace")
    return info


def buffer_alloc(ctx, nbytes):
    buf = c_void_p()
    _check(_lib.tc_buffer_alloc(ctx, c_size_t(nbytes), byref(buf)))
    return buf


def buffer_free(ctx, buf):
    _check(_lib.tc_buffer_free(ctx, buf))


def buffer_map(buf):
    """Return a void* (ctypes c_void_p) to the buffer's host-visible memory.
    On Apple Silicon unified memory this is the same backing as the GPU."""
    p = c_void_p()
    _check(_lib.tc_buffer_map(buf, byref(p)))
    return p


def buffer_size(buf):
    return _lib.tc_buffer_size(buf)


def buffer_write(buf, arr):
    """Copy a numpy ndarray into the buffer."""
    import numpy as np
    p = buffer_map(buf)
    nbytes = arr.nbytes
    ctypes.memmove(p, arr.ctypes.data, nbytes)


def buffer_read(buf, arr):
    """Copy from the buffer into a numpy ndarray (preallocated)."""
    p = buffer_map(buf)
    ctypes.memmove(arr.ctypes.data, p, arr.nbytes)


def gemm(ctx, A, B, C, M, N, K, dtype="f16", accum="f32",
         alpha=1.0, beta=0.0, transpose_a=False, transpose_b=False):
    """Compute C = alpha * op(A) @ op(B) + beta * C."""
    d_in = _DTYPE_MAP[dtype]
    d_acc = _DTYPE_MAP[accum]
    desc = TCGemmDesc(
        M=M, N=N, K=K,
        a_dtype=d_in, b_dtype=d_in, c_dtype=d_in, accum_dtype=d_acc,
        transpose_a=transpose_a, transpose_b=transpose_b,
        alpha=alpha, beta=beta,
        lda=0, ldb=0, ldc=0,
    )
    _check(_lib.tc_gemm(ctx, byref(desc), A, B, C))


def version():
    return _lib.tc_version().decode() if _lib else "(unloaded)"
