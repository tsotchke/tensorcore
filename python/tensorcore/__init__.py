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
        "/opt/tensorcore/lib/libtensorcore.dylib",
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
TC_ERR_ALREADY_INITIALIZED = -2
TC_ERR_NO_DEVICE = -3
TC_DTYPE_F16 = 0
TC_DTYPE_BF16 = 1
TC_DTYPE_F32 = 2
TC_DTYPE_I8 = 3
TC_DTYPE_I32 = 4

TC_QUANT_Q4_0 = 0
TC_QUANT_Q8_0 = 1

TC_GGUF_TYPE_F32 = 0
TC_GGUF_TYPE_F16 = 1
TC_GGUF_TYPE_Q4_0 = 2
TC_GGUF_TYPE_Q4_1 = 3
TC_GGUF_TYPE_Q8_0 = 8
TC_GGUF_TYPE_BF16 = 30
TC_GGUF_TYPE_UNSUPPORTED = -1

_DTYPE_MAP = {
    "f16": TC_DTYPE_F16, "bf16": TC_DTYPE_BF16, "f32": TC_DTYPE_F32,
    "i8": TC_DTYPE_I8, "i32": TC_DTYPE_I32,
}

_QUANT_MAP = {
    "q4_0": TC_QUANT_Q4_0,
    "q8_0": TC_QUANT_Q8_0,
}

_GGUF_TYPE_NAMES = {
    TC_GGUF_TYPE_F32: "F32",
    TC_GGUF_TYPE_F16: "F16",
    TC_GGUF_TYPE_Q4_0: "Q4_0",
    TC_GGUF_TYPE_Q4_1: "Q4_1",
    TC_GGUF_TYPE_Q8_0: "Q8_0",
    TC_GGUF_TYPE_BF16: "BF16",
    TC_GGUF_TYPE_UNSUPPORTED: "unsupported",
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


class TCGGufTensorInfo(Structure):
    _fields_ = [
        ("name", c_char_p),
        ("n_dims", c_int32),
        ("dims", c_uint64 * 4),
        ("type", c_int),
        ("offset", c_uint64),
        ("n_bytes", c_size_t),
        ("data", c_void_p),
    ]


class TCGGufLoadedTensorInfo(Structure):
    _fields_ = [
        ("name", c_char_p),
        ("n_dims", c_int32),
        ("dims", c_uint64 * 4),
        ("type", c_int),
        ("offset", c_uint64),
        ("n_bytes", c_size_t),
        ("buffer", c_void_p),
    ]


class TCGGufLlamaConfig(Structure):
    _fields_ = [
        ("context_length", c_int64),
        ("embedding_length", c_int64),
        ("feed_forward_length", c_int64),
        ("block_count", c_int64),
        ("attention_head_count", c_int64),
        ("attention_head_count_kv", c_int64),
        ("rope_dimension_count", c_int64),
        ("vocab_size", c_int64),
        ("rms_norm_epsilon", c_double),
        ("rope_freq_base", c_double),
        ("rope_freq_scale", c_double),
    ]


class TCGGufQuantizedMatrixInfo(Structure):
    _fields_ = [
        ("N", c_int),
        ("K", c_int),
        ("gguf_type", c_int),
        ("quant_type", c_int),
        ("n_bytes", c_size_t),
        ("buffer", c_void_p),
    ]


if _lib is not None:
    _lib.tc_init.argtypes = [POINTER(c_void_p)];          _lib.tc_init.restype = c_int
    _lib.tc_shutdown.argtypes = [c_void_p];               _lib.tc_shutdown.restype = c_int
    _lib.tc_device_info_get.argtypes = [c_void_p, POINTER(TCDeviceInfo)]; _lib.tc_device_info_get.restype = c_int
    _lib.tc_buffer_alloc.argtypes = [c_void_p, c_size_t, POINTER(c_void_p)]; _lib.tc_buffer_alloc.restype = c_int
    _lib.tc_buffer_free.argtypes  = [c_void_p, c_void_p]; _lib.tc_buffer_free.restype  = c_int
    _lib.tc_buffer_map.argtypes   = [c_void_p, POINTER(c_void_p)]; _lib.tc_buffer_map.restype = c_int
    _lib.tc_buffer_size.argtypes  = [c_void_p];           _lib.tc_buffer_size.restype  = c_size_t
    _lib.tc_stream_create.argtypes = [c_void_p, POINTER(c_void_p)]; _lib.tc_stream_create.restype = c_int
    _lib.tc_stream_destroy.argtypes = [c_void_p, c_void_p]; _lib.tc_stream_destroy.restype = c_int
    _lib.tc_stream_sync.argtypes = [c_void_p]; _lib.tc_stream_sync.restype = c_int
    _lib.tc_gemm.argtypes = [c_void_p, POINTER(TCGemmDesc), c_void_p, c_void_p, c_void_p]
    _lib.tc_gemm.restype  = c_int
    _lib.tc_gemm_async.argtypes = [c_void_p, POINTER(TCGemmDesc), c_void_p, c_void_p, c_void_p, c_void_p]
    _lib.tc_gemm_async.restype = c_int
    _lib.tc_quantize_weights.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
    _lib.tc_quantize_weights.restype = c_int
    _lib.tc_gemv_quantized.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
    _lib.tc_gemv_quantized.restype = c_int
    _lib.tc_gemv_quantized_async.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_void_p]
    _lib.tc_gemv_quantized_async.restype = c_int
    _lib.tc_quantized_size.argtypes = [c_int, c_int, c_int]
    _lib.tc_quantized_size.restype = c_size_t
    _lib.tc_rmsnorm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_float]
    _lib.tc_rmsnorm_forward.restype = c_int
    _lib.tc_rmsnorm_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int]
    _lib.tc_rmsnorm_backward.restype = c_int
    _lib.tc_layernorm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_float]
    _lib.tc_layernorm_forward.restype = c_int
    _lib.tc_layernorm_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int]
    _lib.tc_layernorm_backward.restype = c_int
    _lib.tc_rope_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
    _lib.tc_rope_forward.restype = c_int
    _lib.tc_swiglu_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int]
    _lib.tc_swiglu_forward.restype = c_int
    _lib.tc_swiglu_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int]
    _lib.tc_swiglu_backward.restype = c_int
    _lib.tc_softmax_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int]
    _lib.tc_softmax_forward.restype = c_int
    _lib.tc_softmax_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int]
    _lib.tc_softmax_backward.restype = c_int
    _lib.tc_adamw_step.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_float, c_float, c_float, c_float, c_float, c_float, c_float]
    _lib.tc_adamw_step.restype = c_int
    _lib.tc_fused_rmsnorm_gemv.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_float]
    _lib.tc_fused_rmsnorm_gemv.restype = c_int
    _lib.tc_gguf_open.argtypes = [c_char_p, POINTER(c_void_p)]
    _lib.tc_gguf_open.restype = c_int
    _lib.tc_gguf_close.argtypes = [c_void_p]
    _lib.tc_gguf_close.restype = None
    _lib.tc_gguf_tensor_count.argtypes = [c_void_p]
    _lib.tc_gguf_tensor_count.restype = c_uint64
    _lib.tc_gguf_metadata_count.argtypes = [c_void_p]
    _lib.tc_gguf_metadata_count.restype = c_uint64
    _lib.tc_gguf_get_tensor.argtypes = [c_void_p, c_char_p, POINTER(TCGGufTensorInfo)]
    _lib.tc_gguf_get_tensor.restype = c_int
    _lib.tc_gguf_tensor_at.argtypes = [c_void_p, c_uint64, POINTER(TCGGufTensorInfo)]
    _lib.tc_gguf_tensor_at.restype = c_int
    _lib.tc_gguf_meta_get_str.argtypes = [c_void_p, c_char_p]
    _lib.tc_gguf_meta_get_str.restype = c_char_p
    _lib.tc_gguf_meta_get_i64.argtypes = [c_void_p, c_char_p, c_int64]
    _lib.tc_gguf_meta_get_i64.restype = c_int64
    _lib.tc_gguf_meta_get_f64.argtypes = [c_void_p, c_char_p, c_double]
    _lib.tc_gguf_meta_get_f64.restype = c_double
    _lib.tc_gguf_meta_array_count.argtypes = [c_void_p, c_char_p]
    _lib.tc_gguf_meta_array_count.restype = c_uint64
    _lib.tc_gguf_meta_array_get_str.argtypes = [c_void_p, c_char_p, c_uint64, POINTER(c_void_p), POINTER(c_size_t)]
    _lib.tc_gguf_meta_array_get_str.restype = c_int
    _lib.tc_gguf_meta_array_get_i64.argtypes = [c_void_p, c_char_p, c_uint64, c_int64]
    _lib.tc_gguf_meta_array_get_i64.restype = c_int64
    _lib.tc_gguf_meta_array_get_f64.argtypes = [c_void_p, c_char_p, c_uint64, c_double]
    _lib.tc_gguf_meta_array_get_f64.restype = c_double
    _lib.tc_gguf_get_llama_config.argtypes = [c_void_p, POINTER(TCGGufLlamaConfig)]
    _lib.tc_gguf_get_llama_config.restype = c_int
    _lib.tc_gguf_tensor_to_buffer.argtypes = [c_void_p, c_void_p, c_char_p, POINTER(c_void_p)]
    _lib.tc_gguf_tensor_to_buffer.restype = c_int
    _lib.tc_gguf_tensor_quantized_matrix_info.argtypes = [POINTER(TCGGufTensorInfo), POINTER(TCGGufQuantizedMatrixInfo)]
    _lib.tc_gguf_tensor_quantized_matrix_info.restype = c_int
    _lib.tc_gguf_loaded_tensor_quantized_matrix_info.argtypes = [POINTER(TCGGufLoadedTensorInfo), POINTER(TCGGufQuantizedMatrixInfo)]
    _lib.tc_gguf_loaded_tensor_quantized_matrix_info.restype = c_int
    _lib.tc_gguf_load_supported_tensors.argtypes = [c_void_p, c_void_p, POINTER(c_void_p)]
    _lib.tc_gguf_load_supported_tensors.restype = c_int
    _lib.tc_gguf_loaded_model_free.argtypes = [c_void_p, c_void_p]
    _lib.tc_gguf_loaded_model_free.restype = None
    _lib.tc_gguf_loaded_tensor_count.argtypes = [c_void_p]
    _lib.tc_gguf_loaded_tensor_count.restype = c_uint64
    _lib.tc_gguf_loaded_skipped_tensor_count.argtypes = [c_void_p]
    _lib.tc_gguf_loaded_skipped_tensor_count.restype = c_uint64
    _lib.tc_gguf_loaded_tensor_at.argtypes = [c_void_p, c_uint64, POINTER(TCGGufLoadedTensorInfo)]
    _lib.tc_gguf_loaded_tensor_at.restype = c_int
    _lib.tc_gguf_loaded_get_tensor.argtypes = [c_void_p, c_char_p, POINTER(TCGGufLoadedTensorInfo)]
    _lib.tc_gguf_loaded_get_tensor.restype = c_int
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


def _bytes(value):
    if isinstance(value, bytes):
        return value
    return str(value).encode("utf-8")


def _quant(fmt):
    if isinstance(fmt, int):
        return fmt
    key = str(fmt).lower()
    if key not in _QUANT_MAP:
        raise ValueError(f"unknown quant format: {fmt}")
    return _QUANT_MAP[key]


def _dtype(dtype):
    if isinstance(dtype, int):
        return dtype
    key = str(dtype).lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"unknown dtype: {dtype}")
    return _DTYPE_MAP[key]


def _tensor_info_dict(info):
    t = int(info.type)
    return {
        "name": info.name.decode("utf-8", "replace") if info.name else "",
        "n_dims": int(info.n_dims),
        "dims": tuple(int(info.dims[i]) for i in range(info.n_dims)),
        "type": t,
        "type_name": _GGUF_TYPE_NAMES.get(t, "unsupported"),
        "offset": int(info.offset),
        "n_bytes": int(info.n_bytes),
        "data": info.data,
    }


def _loaded_tensor_info_dict(info):
    t = int(info.type)
    return {
        "name": info.name.decode("utf-8", "replace") if info.name else "",
        "n_dims": int(info.n_dims),
        "dims": tuple(int(info.dims[i]) for i in range(info.n_dims)),
        "type": t,
        "type_name": _GGUF_TYPE_NAMES.get(t, "unsupported"),
        "offset": int(info.offset),
        "n_bytes": int(info.n_bytes),
        "buffer": info.buffer,
    }


def _tensor_info_from_dict(tensor):
    info = TCGGufTensorInfo()
    info.name = _bytes(tensor.get("name", ""))
    dims = tuple(int(d) for d in tensor.get("dims", ()))
    info.n_dims = int(tensor.get("n_dims", len(dims)))
    for i, d in enumerate(dims[:4]):
        info.dims[i] = d
    info.type = int(tensor.get("type", TC_GGUF_TYPE_UNSUPPORTED))
    info.offset = int(tensor.get("offset", 0))
    info.n_bytes = int(tensor.get("n_bytes", 0))
    info.data = tensor.get("data") or None
    return info


def _loaded_tensor_info_from_dict(tensor):
    info = TCGGufLoadedTensorInfo()
    info.name = _bytes(tensor.get("name", ""))
    dims = tuple(int(d) for d in tensor.get("dims", ()))
    info.n_dims = int(tensor.get("n_dims", len(dims)))
    for i, d in enumerate(dims[:4]):
        info.dims[i] = d
    info.type = int(tensor.get("type", TC_GGUF_TYPE_UNSUPPORTED))
    info.offset = int(tensor.get("offset", 0))
    info.n_bytes = int(tensor.get("n_bytes", 0))
    info.buffer = tensor.get("buffer") or None
    return info


def _quantized_matrix_info_dict(info):
    return {
        "N": int(info.N),
        "K": int(info.K),
        "gguf_type": int(info.gguf_type),
        "gguf_type_name": _GGUF_TYPE_NAMES.get(int(info.gguf_type), "unsupported"),
        "quant_type": int(info.quant_type),
        "n_bytes": int(info.n_bytes),
        "buffer": info.buffer,
    }


def _llama_config_dict(config):
    return {
        "context_length": int(config.context_length),
        "embedding_length": int(config.embedding_length),
        "feed_forward_length": int(config.feed_forward_length),
        "block_count": int(config.block_count),
        "attention_head_count": int(config.attention_head_count),
        "attention_head_count_kv": int(config.attention_head_count_kv),
        "rope_dimension_count": int(config.rope_dimension_count),
        "vocab_size": int(config.vocab_size),
        "rms_norm_epsilon": float(config.rms_norm_epsilon),
        "rope_freq_base": float(config.rope_freq_base),
        "rope_freq_scale": float(config.rope_freq_scale),
    }


def init():
    ctx = c_void_p()
    s = _lib.tc_init(byref(ctx))
    if s not in (TC_OK, TC_ERR_ALREADY_INITIALIZED):
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


def stream_create(ctx):
    stream = c_void_p()
    _check(_lib.tc_stream_create(ctx, byref(stream)))
    return stream


def stream_sync(stream):
    _check(_lib.tc_stream_sync(stream))


def stream_destroy(ctx, stream):
    _check(_lib.tc_stream_destroy(ctx, stream))


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
    desc = _gemm_desc(M, N, K, dtype, accum, alpha, beta, transpose_a, transpose_b)
    _check(_lib.tc_gemm(ctx, byref(desc), A, B, C))


def gemm_async(ctx, A, B, C, M, N, K, stream, dtype="f16", accum="f32",
               alpha=1.0, beta=0.0, transpose_a=False, transpose_b=False):
    """Encode C = alpha * op(A) @ op(B) + beta * C into stream."""
    desc = _gemm_desc(M, N, K, dtype, accum, alpha, beta, transpose_a, transpose_b)
    _check(_lib.tc_gemm_async(ctx, byref(desc), A, B, C, stream))


def _gemm_desc(M, N, K, dtype, accum, alpha, beta, transpose_a, transpose_b):
    d_in = _dtype(dtype)
    d_acc = _dtype(accum)
    return TCGemmDesc(
        M=M, N=N, K=K,
        a_dtype=d_in, b_dtype=d_in, c_dtype=d_in, accum_dtype=d_acc,
        transpose_a=transpose_a, transpose_b=transpose_b,
        alpha=alpha, beta=beta,
        lda=0, ldb=0, ldc=0,
    )


def quantized_size(fmt, N, K):
    """Return byte size for an [N, K] quantized weight matrix."""
    return int(_lib.tc_quantized_size(_quant(fmt), int(N), int(K)))


def quantize_weights(ctx, W_fp16, W_quant, fmt, N, K):
    """Quantize an [N, K] fp16 weight matrix into Q4_0 or Q8_0 storage."""
    _check(_lib.tc_quantize_weights(ctx, W_fp16, W_quant, _quant(fmt), int(N), int(K)))


def gemv_quantized(ctx, X, W_quant, Y, fmt, M, N, K):
    """Compute Y[M, N] = X[M, K] @ W_quant[N, K]^T."""
    _check(_lib.tc_gemv_quantized(ctx, X, W_quant, Y, _quant(fmt), int(M), int(N), int(K)))


def gemv_quantized_async(ctx, X, W_quant, Y, fmt, M, N, K, stream):
    """Encode quantized GEMV into stream."""
    _check(_lib.tc_gemv_quantized_async(
        ctx, X, W_quant, Y, _quant(fmt), int(M), int(N), int(K), stream
    ))


def rmsnorm_forward(ctx, X, gamma, Y, rstd_out, N, D, eps=1e-5):
    """Compute Llama-style RMSNorm on fp16 X[N, D]."""
    _check(_lib.tc_rmsnorm_forward(
        ctx, X, gamma, Y, rstd_out, int(N), int(D), c_float(float(eps))
    ))


def rmsnorm_backward(ctx, X, gamma, dY, rstd, dX, dgamma, N, D):
    _check(_lib.tc_rmsnorm_backward(ctx, X, gamma, dY, rstd, dX, dgamma, int(N), int(D)))


def layernorm_forward(ctx, X, gamma, beta, Y, mean_out, rstd_out, N, D, eps=1e-5):
    """Compute LayerNorm on fp16 X[N, D]."""
    _check(_lib.tc_layernorm_forward(
        ctx, X, gamma, beta, Y, mean_out, rstd_out, int(N), int(D), c_float(float(eps))
    ))


def layernorm_backward(ctx, X, gamma, dY, mean, rstd, dX, N, D):
    _check(_lib.tc_layernorm_backward(ctx, X, gamma, dY, mean, rstd, dX, int(N), int(D)))


def rope_forward(ctx, X, cos_t, sin_t, batch, heads, seq, head_dim):
    """Apply RoPE in-place to fp16 X[batch, heads, seq, head_dim]."""
    _check(_lib.tc_rope_forward(
        ctx, X, cos_t, sin_t, int(batch), int(heads), int(seq), int(head_dim)
    ))


def swiglu_forward(ctx, gate, up, out, n):
    """Compute fp16 out = silu(gate) * up."""
    _check(_lib.tc_swiglu_forward(ctx, gate, up, out, int(n)))


def swiglu_backward(ctx, gate, up, dout, dgate, dup, n):
    _check(_lib.tc_swiglu_backward(ctx, gate, up, dout, dgate, dup, int(n)))


def softmax_forward(ctx, X, Y, N, D):
    """Compute row-wise fp16 softmax for X[N, D]."""
    _check(_lib.tc_softmax_forward(ctx, X, Y, int(N), int(D)))


def softmax_backward(ctx, Y, dY, dX, N, D):
    _check(_lib.tc_softmax_backward(ctx, Y, dY, dX, int(N), int(D)))


def adamw_step(ctx, params_fp32, m_fp32, v_fp32, grads, grad_dtype, n,
               lr, beta1, beta2, eps, weight_decay, bias_correction1,
               bias_correction2):
    """Apply one AdamW optimizer step to fp32 params/moments."""
    _check(_lib.tc_adamw_step(
        ctx, params_fp32, m_fp32, v_fp32, grads, _dtype(grad_dtype), int(n),
        c_float(float(lr)), c_float(float(beta1)), c_float(float(beta2)),
        c_float(float(eps)), c_float(float(weight_decay)),
        c_float(float(bias_correction1)), c_float(float(bias_correction2))
    ))


def fused_rmsnorm_gemv(ctx, X, gamma, W, Y, M, N, K, eps=1e-5):
    """Compute Y[M, N] = RMSNorm(X[M, K], gamma[K]) @ W[K, N]."""
    _check(_lib.tc_fused_rmsnorm_gemv(
        ctx, X, gamma, W, Y, int(M), int(N), int(K), c_float(float(eps))
    ))


def gguf_open(path):
    """Open a GGUF v3 file and return an opaque handle."""
    handle = c_void_p()
    _check(_lib.tc_gguf_open(os.fsencode(path), byref(handle)))
    return handle


def gguf_close(gguf):
    _lib.tc_gguf_close(gguf)


def gguf_tensor_count(gguf):
    return int(_lib.tc_gguf_tensor_count(gguf))


def gguf_metadata_count(gguf):
    return int(_lib.tc_gguf_metadata_count(gguf))


def gguf_meta_get_str(gguf, key):
    value = _lib.tc_gguf_meta_get_str(gguf, _bytes(key))
    return value.decode("utf-8", "replace") if value else None


def gguf_meta_get_i64(gguf, key, default=0):
    return int(_lib.tc_gguf_meta_get_i64(gguf, _bytes(key), int(default)))


def gguf_meta_get_f64(gguf, key, default=0.0):
    return float(_lib.tc_gguf_meta_get_f64(gguf, _bytes(key), float(default)))


def gguf_meta_array_count(gguf, key):
    return int(_lib.tc_gguf_meta_array_count(gguf, _bytes(key)))


def gguf_meta_array_get_str(gguf, key, index):
    ptr = c_void_p()
    n = c_size_t()
    _check(_lib.tc_gguf_meta_array_get_str(gguf, _bytes(key), c_uint64(index), byref(ptr), byref(n)))
    return ctypes.string_at(ptr, n.value).decode("utf-8", "replace")


def gguf_meta_array_get_i64(gguf, key, index, default=0):
    return int(_lib.tc_gguf_meta_array_get_i64(gguf, _bytes(key), c_uint64(index), int(default)))


def gguf_meta_array_get_f64(gguf, key, index, default=0.0):
    return float(_lib.tc_gguf_meta_array_get_f64(gguf, _bytes(key), c_uint64(index), float(default)))


def gguf_get_llama_config(gguf):
    config = TCGGufLlamaConfig()
    _check(_lib.tc_gguf_get_llama_config(gguf, byref(config)))
    return _llama_config_dict(config)


def gguf_get_tensor(gguf, name):
    info = TCGGufTensorInfo()
    _check(_lib.tc_gguf_get_tensor(gguf, _bytes(name), byref(info)))
    return _tensor_info_dict(info)


def gguf_tensor_at(gguf, index):
    info = TCGGufTensorInfo()
    _check(_lib.tc_gguf_tensor_at(gguf, c_uint64(index), byref(info)))
    return _tensor_info_dict(info)


def gguf_tensor_to_buffer(ctx, gguf, name):
    """Copy a named GGUF tensor into a tensorcore buffer."""
    buf = c_void_p()
    _check(_lib.tc_gguf_tensor_to_buffer(ctx, gguf, _bytes(name), byref(buf)))
    return buf


def gguf_tensor_quantized_matrix_info(tensor):
    """Return GEMV shape/format info for a GGUF 2D Q4_0/Q8_0 tensor dict."""
    info = tensor if isinstance(tensor, TCGGufTensorInfo) else _tensor_info_from_dict(tensor)
    out = TCGGufQuantizedMatrixInfo()
    _check(_lib.tc_gguf_tensor_quantized_matrix_info(byref(info), byref(out)))
    return _quantized_matrix_info_dict(out)


def gguf_loaded_tensor_quantized_matrix_info(tensor):
    """Return GEMV shape/format info for a loaded GGUF 2D Q4_0/Q8_0 tensor dict."""
    info = tensor if isinstance(tensor, TCGGufLoadedTensorInfo) else _loaded_tensor_info_from_dict(tensor)
    out = TCGGufQuantizedMatrixInfo()
    _check(_lib.tc_gguf_loaded_tensor_quantized_matrix_info(byref(info), byref(out)))
    return _quantized_matrix_info_dict(out)


def gguf_load_supported_tensors(ctx, gguf):
    """Copy all supported GGUF tensors into tensorcore buffers."""
    model = c_void_p()
    _check(_lib.tc_gguf_load_supported_tensors(ctx, gguf, byref(model)))
    return model


def gguf_loaded_model_free(ctx, model):
    _lib.tc_gguf_loaded_model_free(ctx, model)


def gguf_loaded_tensor_count(model):
    return int(_lib.tc_gguf_loaded_tensor_count(model))


def gguf_loaded_skipped_tensor_count(model):
    return int(_lib.tc_gguf_loaded_skipped_tensor_count(model))


def gguf_loaded_tensor_at(model, index):
    info = TCGGufLoadedTensorInfo()
    _check(_lib.tc_gguf_loaded_tensor_at(model, c_uint64(index), byref(info)))
    return _loaded_tensor_info_dict(info)


def gguf_loaded_get_tensor(model, name):
    info = TCGGufLoadedTensorInfo()
    _check(_lib.tc_gguf_loaded_get_tensor(model, _bytes(name), byref(info)))
    return _loaded_tensor_info_dict(info)


def version():
    return _lib.tc_version().decode() if _lib else "(unloaded)"
