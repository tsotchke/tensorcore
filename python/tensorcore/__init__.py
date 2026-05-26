"""tensorcore — Python bindings.

Thin ctypes wrapper around the tensorcore C ABI. Loads the tensorcore native
library from a configured location and exposes the public surface.

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
import math
import os
import sys
import weakref
from ctypes import (
    c_int, c_uint, c_int32, c_int64, c_uint32, c_uint64, c_size_t,
    c_float, c_double, c_char_p, c_void_p, c_bool, POINTER, Structure, byref,
    CFUNCTYPE,
)

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _library_names():
    if sys.platform == "darwin":
        return ("libtensorcore.dylib", "libtensorcore.so")
    if sys.platform.startswith("linux"):
        return ("libtensorcore.so", "libtensorcore.dylib")
    if sys.platform.startswith("win"):
        return ("tensorcore.dll", "libtensorcore.dll")
    return ("libtensorcore.so", "libtensorcore.dylib")


def _find_lib():
    env = os.environ.get("TENSORCORE_LIB")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for name in _library_names():
        package_local = os.path.join(here, name)
        if os.path.exists(package_local):
            return package_local

    source_root = os.path.abspath(os.path.join(here, "..", ".."))
    is_source_checkout = (
        os.path.exists(os.path.join(source_root, "pyproject.toml")) and
        os.path.exists(os.path.join(source_root, "CMakeLists.txt"))
    )
    if not is_source_checkout:
        raise RuntimeError(
            "package-local tensorcore native library not found. Reinstall the "
            "tensorcore-apple wheel or set TENSORCORE_LIB explicitly."
        )

    candidate_dirs = [
        os.path.join(source_root, "build"),
        os.path.join(source_root, "build", "lib"),
        os.path.join(source_root, "build-portable-cpu"),
        "/opt/tensorcore/lib",
        "/usr/local/lib",
    ]
    for directory in candidate_dirs:
        for name in _library_names():
            p = os.path.join(directory, name)
            if os.path.exists(p):
                return p
    raise RuntimeError(
        "tensorcore native library not found. Set TENSORCORE_LIB env var or "
        "build tensorcore as a shared library."
    )

_lib = ctypes.CDLL(_find_lib())


# ---------------------------------------------------------------------------
# C ABI prototypes
# ---------------------------------------------------------------------------

TC_OK = 0
TC_ERR_NOT_INITIALIZED = -1
TC_ERR_ALREADY_INITIALIZED = -2
TC_ERR_NO_DEVICE = -3
TC_ERR_UNSUPPORTED_FAMILY = -4
TC_ERR_UNSUPPORTED_DTYPE = -5
TC_ERR_INVALID_SHAPE = -6
TC_ERR_INVALID_ARG = -7
TC_ERR_ALLOC = -8
TC_ERR_KERNEL_NOT_FOUND = -9
TC_ERR_PIPELINE = -10
TC_ERR_DISPATCH = -11
TC_ERR_INTERNAL = -99

TC_DTYPE_F16 = 0
TC_DTYPE_BF16 = 1
TC_DTYPE_F32 = 2
TC_DTYPE_I8 = 3
TC_DTYPE_I32 = 4
TC_DTYPE_F64 = 5
TC_DTYPE_SF64 = 6
TC_DTYPE_DF64 = 7
TC_DTYPE_FP24 = 8
TC_DTYPE_FP53 = 9

TC_FAMILY_UNKNOWN = 0
TC_FAMILY_APPLE7 = 7
TC_FAMILY_APPLE8 = 8
TC_FAMILY_APPLE9 = 9
TC_FAMILY_APPLE10 = 10
TC_FAMILY_APPLE11 = 11

TC_BACKEND_NONE = 0
TC_BACKEND_SIMDGROUP_MATRIX = 1
TC_BACKEND_TENSOROPS_M5 = 2
TC_BACKEND_MPS = 3
TC_BACKEND_ACCELERATE_CPU = 4
TC_BACKEND_SF64_EMULATED = 5
TC_BACKEND_OZAKI_II = 6
TC_BACKEND_PORTABLE_CPU = 7
TC_BACKEND_METAL_COMPUTE = 8
TC_BACKEND_CUDA = 9
TC_BACKEND_HIP = 10

TC_TIER_L0_DEVICE = 0
TC_TIER_L1_HOST_RAM = 1
TC_TIER_L2_REMOTE_RAM = 2
TC_TIER_L3_LOCAL_NVME = 3
TC_TIER_L4_REMOTE_NVME = 4

TC_TIER_HINT_HOT = 0
TC_TIER_HINT_WARM = 1
TC_TIER_HINT_COLD = 2
TC_TIER_HINT_ICE = 3

TC_DIST_SINGLE = 0
TC_DIST_RING = 1
TC_DIST_GLOO = 2

TC_HIP_VENDOR_UNKNOWN = 0
TC_HIP_VENDOR_INTEL = 1
TC_HIP_VENDOR_NVIDIA = 2
TC_HIP_VENDOR_AMD = 3
TC_HIP_VENDOR_ARM_MALI = 4

TC_DILOCO_COMPRESS_NONE = 0
TC_DILOCO_COMPRESS_FP16 = 1
TC_DILOCO_COMPRESS_FP8 = 2
TC_DILOCO_COMPRESS_TOPK_1PCT = 3
TC_DILOCO_COMPRESS_TOPK_01PCT = 4
TC_DILOCO_COMPRESS_LOWRANK = 5
TC_DILOCO_COMPRESS_SIGNSGD = 6

TC_DILOCO_OUTER_SGD = 0
TC_DILOCO_OUTER_NESTEROV = 1
TC_DILOCO_OUTER_ADAM = 2

TC_REDUCE_SUM = 0
TC_REDUCE_AVG = 1
TC_REDUCE_MAX = 2
TC_REDUCE_MIN = 3

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
    "f64": TC_DTYPE_F64, "sf64": TC_DTYPE_SF64, "df64": TC_DTYPE_DF64,
    "fp24": TC_DTYPE_FP24, "fp53": TC_DTYPE_FP53,
}

_DTYPE_SIZE_MAP = {
    TC_DTYPE_F16: 2, TC_DTYPE_BF16: 2, TC_DTYPE_F32: 4,
    TC_DTYPE_I8: 1, TC_DTYPE_I32: 4, TC_DTYPE_F64: 8,
    TC_DTYPE_SF64: 8, TC_DTYPE_DF64: 8, TC_DTYPE_FP24: 4,
    TC_DTYPE_FP53: 8,
}

_DIST_BACKEND_MAP = {
    "single": TC_DIST_SINGLE,
    "ring": TC_DIST_RING,
    "gloo": TC_DIST_GLOO,
}

_REDUCE_OP_MAP = {
    "sum": TC_REDUCE_SUM,
    "avg": TC_REDUCE_AVG,
    "mean": TC_REDUCE_AVG,
    "max": TC_REDUCE_MAX,
    "min": TC_REDUCE_MIN,
}

_MEMORY_TIER_MAP = {
    "l0": TC_TIER_L0_DEVICE,
    "device": TC_TIER_L0_DEVICE,
    "l1": TC_TIER_L1_HOST_RAM,
    "host": TC_TIER_L1_HOST_RAM,
    "host_ram": TC_TIER_L1_HOST_RAM,
    "l2": TC_TIER_L2_REMOTE_RAM,
    "remote_ram": TC_TIER_L2_REMOTE_RAM,
    "l3": TC_TIER_L3_LOCAL_NVME,
    "local_nvme": TC_TIER_L3_LOCAL_NVME,
    "l4": TC_TIER_L4_REMOTE_NVME,
    "remote_nvme": TC_TIER_L4_REMOTE_NVME,
}

_TIER_HINT_MAP = {
    "hot": TC_TIER_HINT_HOT,
    "warm": TC_TIER_HINT_WARM,
    "cold": TC_TIER_HINT_COLD,
    "ice": TC_TIER_HINT_ICE,
}

_DILOCO_COMPRESS_MAP = {
    "none": TC_DILOCO_COMPRESS_NONE,
    "fp16": TC_DILOCO_COMPRESS_FP16,
    "fp8": TC_DILOCO_COMPRESS_FP8,
    "topk_1pct": TC_DILOCO_COMPRESS_TOPK_1PCT,
    "topk_01pct": TC_DILOCO_COMPRESS_TOPK_01PCT,
    "topk_0_1pct": TC_DILOCO_COMPRESS_TOPK_01PCT,
    "lowrank": TC_DILOCO_COMPRESS_LOWRANK,
    "signsgd": TC_DILOCO_COMPRESS_SIGNSGD,
}

_DILOCO_OUTER_OPTIMIZER_MAP = {
    "sgd": TC_DILOCO_OUTER_SGD,
    "nesterov": TC_DILOCO_OUTER_NESTEROV,
    "adam": TC_DILOCO_OUTER_ADAM,
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


class TCGemmBatchedDesc(Structure):
    _fields_ = [
        ("base", TCGemmDesc),
        ("batch", c_int32),
        ("stride_a", c_int64),
        ("stride_b", c_int64),
        ("stride_c", c_int64),
    ]


class TCAttentionDesc(Structure):
    _fields_ = [
        ("batch", c_int32),
        ("heads", c_int32),
        ("seq_q", c_int32),
        ("seq_kv", c_int32),
        ("head_dim", c_int32),
        ("io_dtype", c_int),
        ("accum_dtype", c_int),
        ("softmax_scale", c_float),
        ("causal", c_bool),
        ("return_lse", c_bool),
        ("kv_heads", c_int32),
        ("window_size", c_int32),
        ("alibi_slopes", POINTER(c_float)),
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


class TCHipDeviceInfo(Structure):
    _fields_ = [
        ("vendor", c_int),
        ("device_name", ctypes.c_char * 128),
        ("driver_version", ctypes.c_char * 64),
        ("opencl_version", ctypes.c_char * 64),
        ("global_memory_bytes", c_uint64),
        ("local_memory_bytes", c_uint64),
        ("compute_units", c_uint32),
        ("max_workgroup_size", c_uint32),
        ("preferred_subgroup_size", c_uint32),
        ("supports_fp16", c_bool),
        ("supports_fp64", c_bool),
        ("supports_int8_dot", c_bool),
        ("unified_memory", c_bool),
    ]


class TCCudaDeviceInfo(Structure):
    _fields_ = [
        ("device_name", ctypes.c_char * 128),
        ("compute_capability", ctypes.c_char * 16),
        ("major", c_int),
        ("minor", c_int),
        ("global_memory_bytes", c_uint64),
        ("shared_memory_per_block", c_uint64),
        ("multiprocessor_count", c_uint32),
        ("max_threads_per_block", c_uint32),
        ("warp_size", c_uint32),
        ("supports_fp16", c_bool),
        ("supports_bf16", c_bool),
        ("supports_int8_tensor_core", c_bool),
        ("supports_tf32", c_bool),
        ("unified_memory", c_bool),
    ]


class TCDiLoCoConfig(Structure):
    _fields_ = [
        ("inner_steps", c_int),
        ("outer_lr", c_float),
        ("outer_momentum", c_float),
        ("outer_beta2", c_float),
        ("outer_eps", c_float),
        ("outer_optimizer", c_int),
        ("compress", c_int),
        ("async_overlap", c_bool),
        ("tolerate_dropouts", c_bool),
    ]


TCCheckpointRecomputeFn = CFUNCTYPE(c_int, c_void_p)
_CHECKPOINT_CALLBACKS = {}


if _lib is not None:
    _lib.tc_init.argtypes = [POINTER(c_void_p)];          _lib.tc_init.restype = c_int
    _lib.tc_shutdown.argtypes = [c_void_p];               _lib.tc_shutdown.restype = c_int
    _lib.tc_device_info_get.argtypes = [c_void_p, POINTER(TCDeviceInfo)]; _lib.tc_device_info_get.restype = c_int
    _lib.tc_buffer_alloc.argtypes = [c_void_p, c_size_t, POINTER(c_void_p)]; _lib.tc_buffer_alloc.restype = c_int
    _lib.tc_buffer_from_ptr.argtypes = [c_void_p, c_void_p, c_size_t, POINTER(c_void_p)]
    _lib.tc_buffer_from_ptr.restype = c_int
    _lib.tc_buffer_free.argtypes  = [c_void_p, c_void_p]; _lib.tc_buffer_free.restype  = c_int
    _lib.tc_buffer_map.argtypes   = [c_void_p, POINTER(c_void_p)]; _lib.tc_buffer_map.restype = c_int
    _lib.tc_buffer_size.argtypes  = [c_void_p];           _lib.tc_buffer_size.restype  = c_size_t
    _lib.tc_buffer_set_tier_hint.argtypes = [c_void_p, c_int]
    _lib.tc_buffer_set_tier_hint.restype = c_int
    _lib.tc_buffer_get_tier.argtypes = [c_void_p, POINTER(c_int)]
    _lib.tc_buffer_get_tier.restype = c_int
    _lib.tc_buffer_promote_async.argtypes = [c_void_p, c_int, c_void_p]
    _lib.tc_buffer_promote_async.restype = c_int
    _lib.tc_buffer_demote_async.argtypes = [c_void_p, c_int, c_void_p]
    _lib.tc_buffer_demote_async.restype = c_int
    _lib.tc_buffer_tier_sync.argtypes = [c_void_p]
    _lib.tc_buffer_tier_sync.restype = c_int
    _lib.tc_memory_tier_usage.argtypes = [c_void_p, c_int, POINTER(c_uint64), POINTER(c_uint64)]
    _lib.tc_memory_tier_usage.restype = c_int
    _lib.tc_checkpoint_register.argtypes = [c_void_p, TCCheckpointRecomputeFn, c_void_p, POINTER(c_uint64)]
    _lib.tc_checkpoint_register.restype = c_int
    _lib.tc_checkpoint_discard.argtypes = [c_uint64]
    _lib.tc_checkpoint_discard.restype = c_int
    _lib.tc_checkpoint_realize.argtypes = [c_uint64]
    _lib.tc_checkpoint_realize.restype = c_int
    _lib.tc_checkpoint_is_resident.argtypes = [c_uint64]
    _lib.tc_checkpoint_is_resident.restype = c_int
    _lib.tc_checkpoint_unregister.argtypes = [c_uint64]
    _lib.tc_checkpoint_unregister.restype = c_int
    _lib.tc_checkpoint_total_bytes_discarded.argtypes = []
    _lib.tc_checkpoint_total_bytes_discarded.restype = c_uint64
    _lib.tc_checkpoint_count_resident.argtypes = []
    _lib.tc_checkpoint_count_resident.restype = c_uint64
    _lib.tc_checkpoint_count_discarded.argtypes = []
    _lib.tc_checkpoint_count_discarded.restype = c_uint64
    _lib.tc_eshkol_init.argtypes = []
    _lib.tc_eshkol_init.restype = c_void_p
    _lib.tc_eshkol_shutdown.argtypes = [c_void_p]
    _lib.tc_eshkol_shutdown.restype = c_int
    _lib.tc_eshkol_device_name.argtypes = [c_void_p]
    _lib.tc_eshkol_device_name.restype = c_char_p
    _lib.tc_eshkol_device_family.argtypes = [c_void_p]
    _lib.tc_eshkol_device_family.restype = c_int
    _lib.tc_eshkol_device_unified_memory.argtypes = [c_void_p]
    _lib.tc_eshkol_device_unified_memory.restype = c_int
    _lib.tc_eshkol_device_supports_bf16.argtypes = [c_void_p]
    _lib.tc_eshkol_device_supports_bf16.restype = c_int
    _lib.tc_eshkol_device_supports_i8.argtypes = [c_void_p]
    _lib.tc_eshkol_device_supports_i8.restype = c_int
    _lib.tc_eshkol_device_supports_tensorops_m5.argtypes = [c_void_p]
    _lib.tc_eshkol_device_supports_tensorops_m5.restype = c_int
    _lib.tc_eshkol_buffer_alloc.argtypes = [c_void_p, c_int64]
    _lib.tc_eshkol_buffer_alloc.restype = c_void_p
    _lib.tc_eshkol_buffer_free.argtypes = [c_void_p, c_void_p]
    _lib.tc_eshkol_buffer_free.restype = c_int
    _lib.tc_eshkol_buffer_map.argtypes = [c_void_p]
    _lib.tc_eshkol_buffer_map.restype = c_void_p
    _lib.tc_eshkol_gemm.argtypes = [
        c_void_p, c_int, c_void_p, c_void_p, c_void_p,
        c_int, c_int, c_int, c_double, c_double, c_int, c_int
    ]
    _lib.tc_eshkol_gemm.restype = c_int
    _lib.tc_eshkol_attention_forward.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
        c_int, c_int, c_int, c_int, c_int, c_double, c_int
    ]
    _lib.tc_eshkol_attention_forward.restype = c_int
    _lib.tc_eshkol_last_backend.argtypes = []
    _lib.tc_eshkol_last_backend.restype = c_char_p
    _lib.tc_eshkol_last_backend_code.argtypes = []
    _lib.tc_eshkol_last_backend_code.restype = c_int
    _lib.tc_eshkol_version.argtypes = []
    _lib.tc_eshkol_version.restype = c_char_p
    _lib.tc_eshkol_status_string.argtypes = [c_int]
    _lib.tc_eshkol_status_string.restype = c_char_p
    _lib.tc_stream_create.argtypes = [c_void_p, POINTER(c_void_p)]; _lib.tc_stream_create.restype = c_int
    _lib.tc_stream_destroy.argtypes = [c_void_p, c_void_p]; _lib.tc_stream_destroy.restype = c_int
    _lib.tc_stream_sync.argtypes = [c_void_p]; _lib.tc_stream_sync.restype = c_int
    _lib.tc_gemm.argtypes = [c_void_p, POINTER(TCGemmDesc), c_void_p, c_void_p, c_void_p]
    _lib.tc_gemm.restype  = c_int
    _lib.tc_gemm_async.argtypes = [c_void_p, POINTER(TCGemmDesc), c_void_p, c_void_p, c_void_p, c_void_p]
    _lib.tc_gemm_async.restype = c_int
    _lib.tc_gemm_batched.argtypes = [
        c_void_p, POINTER(TCGemmBatchedDesc), c_void_p, c_void_p, c_void_p
    ]
    _lib.tc_gemm_batched.restype = c_int
    _lib.tc_attention_forward.argtypes = [
        c_void_p, POINTER(TCAttentionDesc), c_void_p, c_void_p, c_void_p, c_void_p, c_void_p
    ]
    _lib.tc_attention_forward.restype = c_int
    _lib.tc_attention_forward_async.argtypes = [
        c_void_p, POINTER(TCAttentionDesc), c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p
    ]
    _lib.tc_attention_forward_async.restype = c_int
    _lib.tc_attention_backward.argtypes = [
        c_void_p, POINTER(TCAttentionDesc), c_void_p, c_void_p, c_void_p, c_void_p,
        c_void_p, c_void_p, c_void_p, c_void_p, c_void_p
    ]
    _lib.tc_attention_backward.restype = c_int
    _lib.tc_conv2d_forward.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
        c_int, c_int, c_int, c_int, c_int, c_int, c_int,
        c_int, c_int, c_int, c_int, c_int, c_int,
    ]
    _lib.tc_conv2d_forward.restype = c_int
    _lib.tc_conv2d_backward_input.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
        c_int, c_int, c_int, c_int, c_int, c_int, c_int,
        c_int, c_int, c_int, c_int, c_int, c_int,
    ]
    _lib.tc_conv2d_backward_input.restype = c_int
    _lib.tc_conv2d_backward_weight.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
        c_int, c_int, c_int, c_int, c_int, c_int, c_int,
        c_int, c_int, c_int, c_int, c_int, c_int,
    ]
    _lib.tc_conv2d_backward_weight.restype = c_int
    _lib.tc_quantize_weights.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
    _lib.tc_quantize_weights.restype = c_int
    _lib.tc_gemv_quantized.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
    _lib.tc_gemv_quantized.restype = c_int
    _lib.tc_fused_rmsnorm_gemv_quantized.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_float]
    _lib.tc_fused_rmsnorm_gemv_quantized.restype = c_int
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
    _lib.tc_rope_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
    _lib.tc_rope_backward.restype = c_int
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
    _lib.tc_fused_layernorm_gemv.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_float]
    _lib.tc_fused_layernorm_gemv.restype = c_int
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
    _lib.tc_backend_name.argtypes = [c_int]
    _lib.tc_backend_name.restype = c_char_p
    _lib.tc_last_backend.argtypes = []
    _lib.tc_last_backend.restype = c_int
    _lib.tc_dtype_name.argtypes = [c_int]
    _lib.tc_dtype_name.restype = c_char_p
    _lib.tc_tensorops_gemm_kernel_name.argtypes = [POINTER(TCGemmDesc), POINTER(c_int)]
    _lib.tc_tensorops_gemm_kernel_name.restype = c_char_p
    _lib.tc_dist_init.argtypes = [c_void_p, c_int, c_int, c_int, c_char_p, POINTER(c_void_p)]
    _lib.tc_dist_init.restype = c_int
    _lib.tc_dist_finalize.argtypes = [c_void_p]
    _lib.tc_dist_finalize.restype = c_int
    _lib.tc_dist_world_size.argtypes = [c_void_p]
    _lib.tc_dist_world_size.restype = c_int
    _lib.tc_dist_rank.argtypes = [c_void_p]
    _lib.tc_dist_rank.restype = c_int
    _lib.tc_allreduce.argtypes = [c_void_p, c_void_p, c_size_t, c_int, c_int]
    _lib.tc_allreduce.restype = c_int
    _lib.tc_broadcast.argtypes = [c_void_p, c_void_p, c_size_t, c_int, c_int]
    _lib.tc_broadcast.restype = c_int
    _lib.tc_allgather.argtypes = [c_void_p, c_void_p, c_void_p, c_size_t, c_int]
    _lib.tc_allgather.restype = c_int
    _lib.tc_barrier.argtypes = [c_void_p]
    _lib.tc_barrier.restype = c_int
    _lib.tc_hip_init.argtypes = [c_void_p]
    _lib.tc_hip_init.restype = c_int
    _lib.tc_hip_device_info_get.argtypes = [c_void_p, POINTER(TCHipDeviceInfo)]
    _lib.tc_hip_device_info_get.restype = c_int
    _lib.tc_hip_device_count.argtypes = []
    _lib.tc_hip_device_count.restype = c_int
    _lib.tc_hip_device_at.argtypes = [c_int, POINTER(TCHipDeviceInfo)]
    _lib.tc_hip_device_at.restype = c_int
    _lib.tc_hip_select_device.argtypes = [c_void_p, c_int]
    _lib.tc_hip_select_device.restype = c_int
    _lib.tc_hip_last_kernel_name.argtypes = []
    _lib.tc_hip_last_kernel_name.restype = c_char_p
    _lib.tc_cuda_init.argtypes = [c_void_p]
    _lib.tc_cuda_init.restype = c_int
    _lib.tc_cuda_device_count.argtypes = []
    _lib.tc_cuda_device_count.restype = c_int
    _lib.tc_cuda_device_at.argtypes = [c_int, POINTER(TCCudaDeviceInfo)]
    _lib.tc_cuda_device_at.restype = c_int
    _lib.tc_cuda_select_device.argtypes = [c_void_p, c_int]
    _lib.tc_cuda_select_device.restype = c_int
    _lib.tc_cuda_last_kernel_name.argtypes = []
    _lib.tc_cuda_last_kernel_name.restype = c_char_p
    _lib.tc_diloco_init.argtypes = [c_void_p, POINTER(TCDiLoCoConfig), POINTER(c_void_p)]
    _lib.tc_diloco_init.restype = c_int
    _lib.tc_diloco_finalize.argtypes = [c_void_p]
    _lib.tc_diloco_finalize.restype = c_int
    _lib.tc_diloco_add_parameter.argtypes = [c_void_p, c_char_p, c_void_p, c_size_t, c_int]
    _lib.tc_diloco_add_parameter.restype = c_int
    _lib.tc_diloco_step.argtypes = [c_void_p, POINTER(c_bool)]
    _lib.tc_diloco_step.restype = c_int
    _lib.tc_diloco_apply_outer.argtypes = [c_void_p]
    _lib.tc_diloco_apply_outer.restype = c_int
    _lib.tc_diloco_outer_steps_completed.argtypes = [c_void_p]
    _lib.tc_diloco_outer_steps_completed.restype = c_uint64
    _lib.tc_diloco_inner_steps_completed.argtypes = [c_void_p]
    _lib.tc_diloco_inner_steps_completed.restype = c_uint64
    _lib.tc_diloco_last_outer_step_seconds.argtypes = [c_void_p]
    _lib.tc_diloco_last_outer_step_seconds.restype = c_double
    _lib.tc_diloco_last_outer_bytes_sent.argtypes = [c_void_p]
    _lib.tc_diloco_last_outer_bytes_sent.restype = c_double
    _lib.tc_status_string.argtypes = [c_int]; _lib.tc_status_string.restype = c_char_p
    _lib.tc_version.argtypes = []; _lib.tc_version.restype = c_char_p


# ---------------------------------------------------------------------------
# Pythonic surface
# ---------------------------------------------------------------------------

class TensorcoreError(RuntimeError):
    def __init__(self, status):
        msg = status_string(status) if _lib else f"status {status}"
        super().__init__(f"tensorcore error {status}: {msg}")
        self.status = status


def _check(status):
    if status != TC_OK:
        raise TensorcoreError(status)


def _as_handle(value):
    return getattr(value, "handle", value)


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


def _dtype_size(dtype):
    d = _dtype(dtype)
    if d not in _DTYPE_SIZE_MAP:
        raise ValueError(f"unknown dtype: {dtype}")
    return _DTYPE_SIZE_MAP[d]


def _matrix_storage_elements(rows, cols, ld):
    rows = int(rows)
    cols = int(cols)
    ld = int(ld)
    if rows <= 0 or cols <= 0 or ld < cols:
        raise ValueError(f"invalid matrix layout: rows={rows} cols={cols} ld={ld}")
    return (rows - 1) * ld + cols


def _gemm_storage_elements(M, N, K, transpose_a, transpose_b, lda=0, ldb=0, ldc=0):
    M, N, K = int(M), int(N), int(K)
    a_rows = K if transpose_a else M
    a_cols = M if transpose_a else K
    b_rows = N if transpose_b else K
    b_cols = K if transpose_b else N
    lda = int(lda) if lda else a_cols
    ldb = int(ldb) if ldb else b_cols
    ldc = int(ldc) if ldc else N
    return (
        _matrix_storage_elements(a_rows, a_cols, lda),
        _matrix_storage_elements(b_rows, b_cols, ldb),
        _matrix_storage_elements(M, N, ldc),
    )


def _dist_backend(backend):
    if isinstance(backend, int):
        return backend
    key = str(backend).lower()
    if key not in _DIST_BACKEND_MAP:
        raise ValueError(f"unknown distributed backend: {backend}")
    return _DIST_BACKEND_MAP[key]


def _reduce_op(op):
    if isinstance(op, int):
        return op
    key = str(op).lower()
    if key not in _REDUCE_OP_MAP:
        raise ValueError(f"unknown reduce op: {op}")
    return _REDUCE_OP_MAP[key]


def _memory_tier(tier):
    if isinstance(tier, int):
        return tier
    key = str(tier).lower()
    if key not in _MEMORY_TIER_MAP:
        raise ValueError(f"unknown memory tier: {tier}")
    return _MEMORY_TIER_MAP[key]


def _tier_hint(hint):
    if isinstance(hint, int):
        return hint
    key = str(hint).lower()
    if key not in _TIER_HINT_MAP:
        raise ValueError(f"unknown tier hint: {hint}")
    return _TIER_HINT_MAP[key]


def _diloco_compress(compress):
    if isinstance(compress, int):
        return compress
    key = str(compress).lower()
    if key not in _DILOCO_COMPRESS_MAP:
        raise ValueError(f"unknown DiLoCo compression: {compress}")
    return _DILOCO_COMPRESS_MAP[key]


def _diloco_outer_optimizer(optimizer):
    if isinstance(optimizer, int):
        return optimizer
    key = str(optimizer).lower()
    if key not in _DILOCO_OUTER_OPTIMIZER_MAP:
        raise ValueError(f"unknown DiLoCo outer optimizer: {optimizer}")
    return _DILOCO_OUTER_OPTIMIZER_MAP[key]


def _decode_cstr(value):
    return value.decode("utf-8") if value else None


def _decode_fixed_cstr(value):
    return bytes(value).split(b"\0", 1)[0].decode("utf-8", "replace")


def _hip_info_dict(info):
    return {
        "vendor": int(info.vendor),
        "device_name": _decode_fixed_cstr(info.device_name),
        "driver_version": _decode_fixed_cstr(info.driver_version),
        "opencl_version": _decode_fixed_cstr(info.opencl_version),
        "global_memory_bytes": int(info.global_memory_bytes),
        "local_memory_bytes": int(info.local_memory_bytes),
        "compute_units": int(info.compute_units),
        "max_workgroup_size": int(info.max_workgroup_size),
        "preferred_subgroup_size": int(info.preferred_subgroup_size),
        "supports_fp16": bool(info.supports_fp16),
        "supports_fp64": bool(info.supports_fp64),
        "supports_int8_dot": bool(info.supports_int8_dot),
        "unified_memory": bool(info.unified_memory),
    }


def _cuda_info_dict(info):
    return {
        "device_name": _decode_fixed_cstr(info.device_name),
        "compute_capability": _decode_fixed_cstr(info.compute_capability),
        "major": int(info.major),
        "minor": int(info.minor),
        "global_memory_bytes": int(info.global_memory_bytes),
        "shared_memory_per_block": int(info.shared_memory_per_block),
        "multiprocessor_count": int(info.multiprocessor_count),
        "max_threads_per_block": int(info.max_threads_per_block),
        "warp_size": int(info.warp_size),
        "supports_fp16": bool(info.supports_fp16),
        "supports_bf16": bool(info.supports_bf16),
        "supports_int8_tensor_core": bool(info.supports_int8_tensor_core),
        "supports_tf32": bool(info.supports_tf32),
        "unified_memory": bool(info.unified_memory),
    }


def status_string(status):
    """Return the C ABI status text for a tensorcore status code."""
    return _decode_cstr(_lib.tc_status_string(int(status))) or "unknown status"


def dtype_name(dtype):
    """Return the C ABI dtype name for a tensorcore dtype enum or alias."""
    return _decode_cstr(_lib.tc_dtype_name(_dtype(dtype))) or "?"


def backend_name(backend):
    """Return the C ABI backend name for a tc_backend_t value."""
    return _decode_cstr(_lib.tc_backend_name(int(backend))) or "?"


def last_backend():
    """Return the thread-local backend enum used by the most recent kernel call."""
    return int(_lib.tc_last_backend())


def last_backend_name():
    """Return the name of the thread-local backend used by the most recent kernel call."""
    return backend_name(last_backend())


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
    _check(_lib.tc_shutdown(_as_handle(ctx)))


def device_info(ctx):
    info = TCDeviceInfo()
    _check(_lib.tc_device_info_get(_as_handle(ctx), byref(info)))
    info.name_str = info.name.decode("utf-8", "replace")
    return info


def buffer_alloc(ctx, nbytes):
    buf = c_void_p()
    _check(_lib.tc_buffer_alloc(_as_handle(ctx), c_size_t(nbytes), byref(buf)))
    return buf


def buffer_from_ptr(ctx, ptr, nbytes):
    """Wrap externally owned host memory in a tc_buffer without copying it."""
    if isinstance(ptr, c_void_p):
        raw = ptr
    else:
        try:
            raw = ctypes.cast(ptr, c_void_p)
        except (ctypes.ArgumentError, TypeError):
            raw = c_void_p(int(ptr))
    buf = c_void_p()
    _check(_lib.tc_buffer_from_ptr(_as_handle(ctx), raw, c_size_t(nbytes), byref(buf)))
    return buf


def buffer_free(ctx, buf):
    _check(_lib.tc_buffer_free(_as_handle(ctx), _as_handle(buf)))


def buffer_map(buf):
    """Return a void* (ctypes c_void_p) to the buffer's host-visible memory.
    On Apple Silicon unified memory this is the same backing as the GPU."""
    p = c_void_p()
    _check(_lib.tc_buffer_map(_as_handle(buf), byref(p)))
    return p


def buffer_size(buf):
    return _lib.tc_buffer_size(_as_handle(buf))


def buffer_set_tier_hint(buf, hint):
    """Set an advisory memory-tier usage hint on a buffer."""
    _check(_lib.tc_buffer_set_tier_hint(_as_handle(buf), _tier_hint(hint)))


def buffer_get_tier(buf):
    """Return the current physical memory tier for a buffer."""
    tier = c_int(0)
    _check(_lib.tc_buffer_get_tier(_as_handle(buf), byref(tier)))
    return int(tier.value)


def buffer_promote_async(buf, target_tier=TC_TIER_L0_DEVICE, stream=None):
    """Promote a buffer toward a faster memory tier."""
    _check(_lib.tc_buffer_promote_async(
        _as_handle(buf), _memory_tier(target_tier), _as_handle(stream),
    ))


def buffer_demote_async(buf, target_tier=TC_TIER_L0_DEVICE, stream=None):
    """Demote a buffer toward a slower memory tier when the runtime supports it."""
    _check(_lib.tc_buffer_demote_async(
        _as_handle(buf), _memory_tier(target_tier), _as_handle(stream),
    ))


def buffer_tier_sync(buf):
    """Fence outstanding tier transitions for a buffer."""
    _check(_lib.tc_buffer_tier_sync(_as_handle(buf)))


def memory_tier_usage(ctx, tier=TC_TIER_L0_DEVICE):
    """Return (resident_bytes, capacity_bytes) for a memory tier."""
    resident = c_uint64(0)
    capacity = c_uint64(0)
    _check(_lib.tc_memory_tier_usage(
        _as_handle(ctx), _memory_tier(tier), byref(resident), byref(capacity),
    ))
    return int(resident.value), int(capacity.value)


def checkpoint_register(buf, recompute_fn, user_data=None):
    """Register a buffer-level activation checkpoint and return its id."""
    if recompute_fn is None:
        raise ValueError("recompute_fn is required")

    def _callback(raw_user_data):
        try:
            result = recompute_fn(raw_user_data)
            return TC_OK if result is None else int(result)
        except Exception:
            return TC_ERR_INTERNAL

    callback = TCCheckpointRecomputeFn(_callback)
    checkpoint_id = c_uint64(0)
    _check(_lib.tc_checkpoint_register(
        _as_handle(buf), callback, _as_handle(user_data), byref(checkpoint_id),
    ))
    _CHECKPOINT_CALLBACKS[int(checkpoint_id.value)] = callback
    return int(checkpoint_id.value)


def checkpoint_discard(checkpoint_id):
    _check(_lib.tc_checkpoint_discard(c_uint64(int(checkpoint_id))))


def checkpoint_realize(checkpoint_id):
    _check(_lib.tc_checkpoint_realize(c_uint64(int(checkpoint_id))))


def checkpoint_is_resident(checkpoint_id):
    return bool(_lib.tc_checkpoint_is_resident(c_uint64(int(checkpoint_id))))


def checkpoint_unregister(checkpoint_id):
    checkpoint_id = int(checkpoint_id)
    _check(_lib.tc_checkpoint_unregister(c_uint64(checkpoint_id)))
    _CHECKPOINT_CALLBACKS.pop(checkpoint_id, None)


def checkpoint_total_bytes_discarded():
    return int(_lib.tc_checkpoint_total_bytes_discarded())


def checkpoint_count_resident():
    return int(_lib.tc_checkpoint_count_resident())


def checkpoint_count_discarded():
    return int(_lib.tc_checkpoint_count_discarded())


def stream_create(ctx):
    stream = c_void_p()
    _check(_lib.tc_stream_create(_as_handle(ctx), byref(stream)))
    return stream


def stream_sync(stream):
    _check(_lib.tc_stream_sync(_as_handle(stream)))


def stream_destroy(ctx, stream):
    _check(_lib.tc_stream_destroy(_as_handle(ctx), _as_handle(stream)))


def buffer_write(buf, arr):
    """Copy a numpy ndarray into the buffer."""
    import numpy as np
    arr = np.ascontiguousarray(arr)
    p = buffer_map(buf)
    nbytes = arr.nbytes
    capacity = buffer_size(buf)
    if nbytes > capacity:
        raise ValueError(f"array has {nbytes} bytes but buffer has {capacity} bytes")
    ctypes.memmove(p, arr.ctypes.data, nbytes)


def buffer_read(buf, arr):
    """Copy from the buffer into a numpy ndarray (preallocated)."""
    import numpy as np
    p = buffer_map(buf)
    nbytes = arr.nbytes
    capacity = buffer_size(buf)
    if nbytes > capacity:
        raise ValueError(f"array has {nbytes} bytes but buffer has {capacity} bytes")
    if arr.flags.c_contiguous:
        ctypes.memmove(arr.ctypes.data, p, nbytes)
    else:
        tmp = np.empty(arr.shape, dtype=arr.dtype)
        ctypes.memmove(tmp.ctypes.data, p, nbytes)
        arr[...] = tmp


def dist_init(ctx, backend=TC_DIST_SINGLE, world_size=1, rank=0, rendezvous_url=None):
    """Create a distributed context. TC_DIST_SINGLE works as a local no-op backend."""
    dist = c_void_p()
    url = None if rendezvous_url is None else _bytes(rendezvous_url)
    _check(_lib.tc_dist_init(_as_handle(ctx), _dist_backend(backend),
                             int(world_size), int(rank), url, byref(dist)))
    return dist


def dist_finalize(dist):
    _check(_lib.tc_dist_finalize(_as_handle(dist)))


def dist_world_size(dist):
    return int(_lib.tc_dist_world_size(_as_handle(dist)))


def dist_rank(dist):
    return int(_lib.tc_dist_rank(_as_handle(dist)))


def _check_collective_buffer(buf, num_elements, dtype, multiplier=1):
    nbytes = int(num_elements) * _dtype_size(dtype) * int(multiplier)
    if int(num_elements) <= 0:
        raise ValueError("num_elements must be positive")
    capacity = buffer_size(buf)
    if nbytes > capacity:
        raise ValueError(f"collective needs {nbytes} bytes but buffer has {capacity} bytes")


def allreduce(dist, buf, num_elements, dtype="f32", op=TC_REDUCE_SUM):
    """In-place all-reduce. TC_DIST_SINGLE leaves the buffer unchanged."""
    _check_collective_buffer(buf, num_elements, dtype)
    _check(_lib.tc_allreduce(_as_handle(dist), _as_handle(buf), c_size_t(num_elements),
                             _dtype(dtype), _reduce_op(op)))


def broadcast(dist, buf, num_elements, dtype="f32", root=0):
    """Broadcast from root. TC_DIST_SINGLE leaves the buffer unchanged."""
    _check_collective_buffer(buf, num_elements, dtype)
    _check(_lib.tc_broadcast(_as_handle(dist), _as_handle(buf), c_size_t(num_elements),
                             _dtype(dtype), int(root)))


def allgather(dist, src, dst, num_elements_per_rank, dtype="f32"):
    """Gather one contribution per rank into dst."""
    world_size = dist_world_size(dist)
    _check_collective_buffer(src, num_elements_per_rank, dtype)
    _check_collective_buffer(dst, num_elements_per_rank, dtype, multiplier=world_size)
    _check(_lib.tc_allgather(_as_handle(dist), _as_handle(src), _as_handle(dst),
                             c_size_t(num_elements_per_rank), _dtype(dtype)))


def barrier(dist):
    _check(_lib.tc_barrier(_as_handle(dist)))


def hip_init(ctx):
    """Initialize the HIP/chipStar backend, if available on this host."""
    _check(_lib.tc_hip_init(_as_handle(ctx)))


def hip_device_info_get(ctx):
    """Return HIP/chipStar device metadata for an initialized HIP backend."""
    info = TCHipDeviceInfo()
    _check(_lib.tc_hip_device_info_get(_as_handle(ctx), byref(info)))
    return _hip_info_dict(info)


def hip_device_count():
    """Return the number of HIP/chipStar devices visible to tensorcore."""
    return int(_lib.tc_hip_device_count())


def hip_device_at(index):
    """Return HIP/chipStar metadata for a device index."""
    info = TCHipDeviceInfo()
    _check(_lib.tc_hip_device_at(int(index), byref(info)))
    return _hip_info_dict(info)


def hip_select_device(ctx, index):
    """Select the HIP/chipStar device used by the current tensorcore context."""
    _check(_lib.tc_hip_select_device(_as_handle(ctx), int(index)))


def hip_last_kernel_name():
    """Return the diagnostic HIP kernel name from the last HIP-dispatched call."""
    return _decode_cstr(_lib.tc_hip_last_kernel_name()) or "none"


def cuda_init(ctx):
    """Initialize the CUDA backend, if available on this host."""
    _check(_lib.tc_cuda_init(_as_handle(ctx)))


def cuda_device_count():
    """Return the number of CUDA devices visible to tensorcore."""
    return int(_lib.tc_cuda_device_count())


def cuda_device_at(index):
    """Return CUDA device metadata for a device index."""
    info = TCCudaDeviceInfo()
    _check(_lib.tc_cuda_device_at(int(index), byref(info)))
    return _cuda_info_dict(info)


def cuda_select_device(ctx, index):
    """Select the CUDA device used by the current tensorcore context."""
    _check(_lib.tc_cuda_select_device(_as_handle(ctx), int(index)))


def cuda_last_kernel_name():
    """Return the diagnostic CUDA kernel name from the last CUDA-dispatched call."""
    return _decode_cstr(_lib.tc_cuda_last_kernel_name()) or "none"


def diloco_config(inner_steps=100, outer_lr=1.0, outer_momentum=0.9,
                  outer_beta2=0.999, outer_eps=1e-8,
                  outer_optimizer=TC_DILOCO_OUTER_NESTEROV,
                  compress=TC_DILOCO_COMPRESS_NONE,
                  async_overlap=False, tolerate_dropouts=False):
    """Build a TCDiLoCoConfig from Python values and enum aliases."""
    return TCDiLoCoConfig(
        inner_steps=int(inner_steps),
        outer_lr=float(outer_lr),
        outer_momentum=float(outer_momentum),
        outer_beta2=float(outer_beta2),
        outer_eps=float(outer_eps),
        outer_optimizer=_diloco_outer_optimizer(outer_optimizer),
        compress=_diloco_compress(compress),
        async_overlap=bool(async_overlap),
        tolerate_dropouts=bool(tolerate_dropouts),
    )


def diloco_init(dist, config=None, **kwargs):
    """Create a DiLoCo context layered on an existing distributed context."""
    cfg = config if config is not None else diloco_config(**kwargs)
    handle = c_void_p()
    _check(_lib.tc_diloco_init(_as_handle(dist), byref(cfg), byref(handle)))
    return handle


def diloco_finalize(diloco):
    _check(_lib.tc_diloco_finalize(_as_handle(diloco)))


def diloco_add_parameter(diloco, name, theta_local, num_elements, dtype="f32"):
    """Register one parameter buffer with a DiLoCo context."""
    _check(_lib.tc_diloco_add_parameter(
        _as_handle(diloco), _bytes(name), _as_handle(theta_local),
        c_size_t(num_elements), _dtype(dtype),
    ))


def diloco_step(diloco):
    """Record one inner step and return whether an outer step is pending."""
    pending = c_bool(False)
    _check(_lib.tc_diloco_step(_as_handle(diloco), byref(pending)))
    return bool(pending.value)


def diloco_apply_outer(diloco):
    """Run the DiLoCo outer optimizer step."""
    _check(_lib.tc_diloco_apply_outer(_as_handle(diloco)))


def diloco_outer_steps_completed(diloco):
    return int(_lib.tc_diloco_outer_steps_completed(_as_handle(diloco)))


def diloco_inner_steps_completed(diloco):
    return int(_lib.tc_diloco_inner_steps_completed(_as_handle(diloco)))


def diloco_last_outer_step_seconds(diloco):
    return float(_lib.tc_diloco_last_outer_step_seconds(_as_handle(diloco)))


def diloco_last_outer_bytes_sent(diloco):
    return float(_lib.tc_diloco_last_outer_bytes_sent(_as_handle(diloco)))


def gemm(ctx, A, B, C, M, N, K, dtype="f16", accum="f32",
         alpha=1.0, beta=0.0, transpose_a=False, transpose_b=False,
         lda=0, ldb=0, ldc=0):
    """Compute C = alpha * op(A) @ op(B) + beta * C."""
    desc = _gemm_desc(M, N, K, dtype, accum, alpha, beta,
                      transpose_a, transpose_b, lda, ldb, ldc)
    _check(_lib.tc_gemm(_as_handle(ctx), byref(desc),
                        _as_handle(A), _as_handle(B), _as_handle(C)))


def gemm_async(ctx, A, B, C, M, N, K, stream, dtype="f16", accum="f32",
               alpha=1.0, beta=0.0, transpose_a=False, transpose_b=False,
               lda=0, ldb=0, ldc=0):
    """Encode C = alpha * op(A) @ op(B) + beta * C into stream."""
    desc = _gemm_desc(M, N, K, dtype, accum, alpha, beta,
                      transpose_a, transpose_b, lda, ldb, ldc)
    _check(_lib.tc_gemm_async(_as_handle(ctx), byref(desc),
                              _as_handle(A), _as_handle(B), _as_handle(C),
                              _as_handle(stream)))


def gemm_batched(ctx, A, B, C, batch, M, N, K, dtype="f16", accum="f32",
                 alpha=1.0, beta=0.0, transpose_a=False, transpose_b=False,
                 stride_a=0, stride_b=0, stride_c=0,
                 lda=0, ldb=0, ldc=0):
    """Compute strided batched C[b] = alpha * op(A[b]) @ op(B[b]) + beta * C[b]."""
    storage_a, storage_b, storage_c = _gemm_storage_elements(
        M, N, K, transpose_a, transpose_b, lda, ldb, ldc,
    )
    if stride_a == 0:
        stride_a = storage_a
    if stride_b == 0:
        stride_b = storage_b
    if stride_c == 0:
        stride_c = storage_c
    desc = TCGemmBatchedDesc(
        base=_gemm_desc(M, N, K, dtype, accum, alpha, beta,
                        transpose_a, transpose_b, lda, ldb, ldc),
        batch=int(batch),
        stride_a=int(stride_a),
        stride_b=int(stride_b),
        stride_c=int(stride_c),
    )
    _check(_lib.tc_gemm_batched(_as_handle(ctx), byref(desc),
                                _as_handle(A), _as_handle(B), _as_handle(C)))


def _gemm_desc(M, N, K, dtype, accum, alpha, beta, transpose_a, transpose_b,
               lda=0, ldb=0, ldc=0):
    d_in = _dtype(dtype)
    d_acc = _dtype(accum)
    return TCGemmDesc(
        M=int(M), N=int(N), K=int(K),
        a_dtype=d_in, b_dtype=d_in, c_dtype=d_in, accum_dtype=d_acc,
        transpose_a=bool(transpose_a), transpose_b=bool(transpose_b),
        alpha=float(alpha), beta=float(beta),
        lda=int(lda), ldb=int(ldb), ldc=int(ldc),
    )


def tensorops_gemm_kernel_name(dtype="f16", accum="f32"):
    """Return the Metal 4 TensorOps GEMM kernel name for a dtype combo, or None."""
    desc = _gemm_desc(1, 1, 1, dtype, accum, 1.0, 0.0, False, False)
    err = c_int(TC_OK)
    name = _lib.tc_tensorops_gemm_kernel_name(byref(desc), byref(err))
    if name:
        return _decode_cstr(name)
    if err.value == TC_ERR_UNSUPPORTED_DTYPE:
        return None
    _check(err.value)
    return None


def _attention_desc(batch, heads, seq_q, seq_kv, head_dim, dtype, accum,
                    softmax_scale, causal, return_lse, kv_heads,
                    window_size, alibi_slopes):
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(float(head_dim))
    slopes = None
    slopes_ptr = None
    if alibi_slopes is not None:
        values = [float(x) for x in alibi_slopes]
        if len(values) != int(heads):
            raise ValueError(f"alibi_slopes must contain {int(heads)} values")
        slopes = (c_float * len(values))(*values)
        slopes_ptr = ctypes.cast(slopes, POINTER(c_float))
    desc = TCAttentionDesc(
        batch=int(batch),
        heads=int(heads),
        seq_q=int(seq_q),
        seq_kv=int(seq_kv),
        head_dim=int(head_dim),
        io_dtype=_dtype(dtype),
        accum_dtype=_dtype(accum),
        softmax_scale=c_float(float(softmax_scale)),
        causal=bool(causal),
        return_lse=bool(return_lse),
        kv_heads=int(kv_heads),
        window_size=int(window_size),
        alibi_slopes=slopes_ptr,
    )
    return desc, slopes


def attention_forward(ctx, Q, K, V, O, batch, heads, seq_q, seq_kv, head_dim,
                      LSE=None, dtype="f16", accum="f32", softmax_scale=None,
                      causal=True, return_lse=False, kv_heads=0,
                      window_size=0, alibi_slopes=None):
    """Compute fused scaled-dot-product attention."""
    return_lse = bool(return_lse or LSE is not None)
    desc, slopes = _attention_desc(batch, heads, seq_q, seq_kv, head_dim,
                                   dtype, accum, softmax_scale, causal,
                                   return_lse, kv_heads, window_size,
                                   alibi_slopes)
    _check(_lib.tc_attention_forward(
        _as_handle(ctx), byref(desc), _as_handle(Q), _as_handle(K),
        _as_handle(V), _as_handle(O), _as_handle(LSE)
    ))
    _ = slopes


def attention_forward_async(ctx, Q, K, V, O, batch, heads, seq_q, seq_kv,
                            head_dim, stream, LSE=None, dtype="f16",
                            accum="f32", softmax_scale=None, causal=True,
                            return_lse=False, kv_heads=0, window_size=0,
                            alibi_slopes=None):
    """Encode fused attention into a stream."""
    return_lse = bool(return_lse or LSE is not None)
    desc, slopes = _attention_desc(batch, heads, seq_q, seq_kv, head_dim,
                                   dtype, accum, softmax_scale, causal,
                                   return_lse, kv_heads, window_size,
                                   alibi_slopes)
    _check(_lib.tc_attention_forward_async(
        _as_handle(ctx), byref(desc), _as_handle(Q), _as_handle(K),
        _as_handle(V), _as_handle(O), _as_handle(LSE), _as_handle(stream)
    ))
    _ = slopes


def attention_backward(ctx, Q, K, V, O, dO, LSE, dQ, dK, dV, batch, heads,
                       seq_q, seq_kv, head_dim, dtype="f16", accum="f32",
                       softmax_scale=None, causal=True, kv_heads=0):
    """Compute gradients for fused attention."""
    desc, slopes = _attention_desc(batch, heads, seq_q, seq_kv, head_dim,
                                   dtype, accum, softmax_scale, causal,
                                   False, kv_heads, 0, None)
    _check(_lib.tc_attention_backward(
        _as_handle(ctx), byref(desc), _as_handle(Q), _as_handle(K),
        _as_handle(V), _as_handle(O), _as_handle(dO), _as_handle(LSE),
        _as_handle(dQ), _as_handle(dK), _as_handle(dV)
    ))
    _ = slopes


def conv2d_output_shape(H, W_in, kH, kW, pad_h=0, pad_w=0, stride_h=1, stride_w=1):
    """Return (out_H, out_W) for a dilation-1 Conv2D."""
    out_H = (int(H) + 2 * int(pad_h) - int(kH)) // int(stride_h) + 1
    out_W = (int(W_in) + 2 * int(pad_w) - int(kW)) // int(stride_w) + 1
    return out_H, out_W


def conv2d_scratch_bytes(batch, in_channels, H, W_in, kH, kW,
                         pad_h=0, pad_w=0, stride_h=1, stride_w=1,
                         out_H=None, out_W=None):
    """Return fp16 im2col scratch bytes required by conv2d_forward."""
    if out_H is None or out_W is None:
        out_H, out_W = conv2d_output_shape(H, W_in, kH, kW, pad_h, pad_w,
                                           stride_h, stride_w)
    return int(batch) * int(in_channels) * int(kH) * int(kW) * int(out_H) * int(out_W) * 2


def conv2d_backward_input_scratch_bytes(batch, in_channels, H, W_in):
    """Return fp32 accumulation scratch bytes required by conv2d_backward_input."""
    return int(batch) * int(in_channels) * int(H) * int(W_in) * 4


def conv2d_forward(ctx, X, weight, bias, Y, scratch_col,
                   batch, in_channels, out_channels, H, W_in, kH, kW,
                   pad_h=0, pad_w=0, stride_h=1, stride_w=1,
                   out_H=None, out_W=None):
    """Compute fp16 Conv2D forward for NCHW input and OIHW weights."""
    if out_H is None or out_W is None:
        out_H, out_W = conv2d_output_shape(H, W_in, kH, kW, pad_h, pad_w,
                                           stride_h, stride_w)
    _check(_lib.tc_conv2d_forward(
        _as_handle(ctx), _as_handle(X), _as_handle(weight), _as_handle(bias),
        _as_handle(Y), _as_handle(scratch_col),
        int(batch), int(in_channels), int(out_channels),
        int(H), int(W_in), int(kH), int(kW),
        int(pad_h), int(pad_w), int(stride_h), int(stride_w),
        int(out_H), int(out_W)
    ))


def conv2d_backward_input(ctx, dY, weight, dX, scratch_col, scratch_dX_f32,
                          batch, in_channels, out_channels, H, W_in, kH, kW,
                          pad_h=0, pad_w=0, stride_h=1, stride_w=1,
                          out_H=None, out_W=None):
    """Compute fp16 Conv2D input gradients for NCHW dY and OIHW weights."""
    if out_H is None or out_W is None:
        out_H, out_W = conv2d_output_shape(H, W_in, kH, kW, pad_h, pad_w,
                                           stride_h, stride_w)
    _check(_lib.tc_conv2d_backward_input(
        _as_handle(ctx), _as_handle(dY), _as_handle(weight), _as_handle(dX),
        _as_handle(scratch_col), _as_handle(scratch_dX_f32),
        int(batch), int(in_channels), int(out_channels),
        int(H), int(W_in), int(kH), int(kW),
        int(pad_h), int(pad_w), int(stride_h), int(stride_w),
        int(out_H), int(out_W)
    ))


def conv2d_backward_weight(ctx, X, dY, dW, scratch_col,
                           batch, in_channels, out_channels, H, W_in, kH, kW,
                           pad_h=0, pad_w=0, stride_h=1, stride_w=1,
                           out_H=None, out_W=None):
    """Compute fp16 Conv2D weight gradients for NCHW X/dY and OIHW weights."""
    if out_H is None or out_W is None:
        out_H, out_W = conv2d_output_shape(H, W_in, kH, kW, pad_h, pad_w,
                                           stride_h, stride_w)
    _check(_lib.tc_conv2d_backward_weight(
        _as_handle(ctx), _as_handle(X), _as_handle(dY), _as_handle(dW),
        _as_handle(scratch_col),
        int(batch), int(in_channels), int(out_channels),
        int(H), int(W_in), int(kH), int(kW),
        int(pad_h), int(pad_w), int(stride_h), int(stride_w),
        int(out_H), int(out_W)
    ))


def quantized_size(fmt, N, K):
    """Return byte size for an [N, K] quantized weight matrix."""
    return int(_lib.tc_quantized_size(_quant(fmt), int(N), int(K)))


def quantize_weights(ctx, W_fp16, W_quant, fmt, N, K):
    """Quantize an [N, K] fp16 weight matrix into Q4_0 or Q8_0 storage."""
    _check(_lib.tc_quantize_weights(_as_handle(ctx), _as_handle(W_fp16),
                                    _as_handle(W_quant), _quant(fmt),
                                    int(N), int(K)))


def gemv_quantized(ctx, X, W_quant, Y, fmt, M, N, K):
    """Compute Y[M, N] = X[M, K] @ W_quant[N, K]^T."""
    _check(_lib.tc_gemv_quantized(_as_handle(ctx), _as_handle(X),
                                  _as_handle(W_quant), _as_handle(Y),
                                  _quant(fmt), int(M), int(N), int(K)))


def fused_rmsnorm_gemv_quantized(ctx, X, gamma, W_quant, Y, fmt, M, N, K, eps=1e-5):
    """Compute RMSNorm(X, gamma) followed by quantized GEMV."""
    _check(_lib.tc_fused_rmsnorm_gemv_quantized(
        _as_handle(ctx), _as_handle(X), _as_handle(gamma),
        _as_handle(W_quant), _as_handle(Y), _quant(fmt),
        int(M), int(N), int(K), c_float(float(eps))
    ))


def gemv_quantized_async(ctx, X, W_quant, Y, fmt, M, N, K, stream):
    """Encode quantized GEMV into stream."""
    _check(_lib.tc_gemv_quantized_async(
        _as_handle(ctx), _as_handle(X), _as_handle(W_quant), _as_handle(Y),
        _quant(fmt), int(M), int(N), int(K), _as_handle(stream)
    ))


def rmsnorm_forward(ctx, X, gamma, Y, rstd_out, N, D, eps=1e-5):
    """Compute Llama-style RMSNorm on fp16 X[N, D]."""
    _check(_lib.tc_rmsnorm_forward(
        _as_handle(ctx), _as_handle(X), _as_handle(gamma), _as_handle(Y),
        _as_handle(rstd_out), int(N), int(D), c_float(float(eps))
    ))


def rmsnorm_backward(ctx, X, gamma, dY, rstd, dX, dgamma, N, D):
    _check(_lib.tc_rmsnorm_backward(_as_handle(ctx), _as_handle(X),
                                    _as_handle(gamma), _as_handle(dY),
                                    _as_handle(rstd), _as_handle(dX),
                                    _as_handle(dgamma), int(N), int(D)))


def layernorm_forward(ctx, X, gamma, beta, Y, mean_out, rstd_out, N, D, eps=1e-5):
    """Compute LayerNorm on fp16 X[N, D]."""
    _check(_lib.tc_layernorm_forward(
        _as_handle(ctx), _as_handle(X), _as_handle(gamma), _as_handle(beta),
        _as_handle(Y), _as_handle(mean_out), _as_handle(rstd_out),
        int(N), int(D), c_float(float(eps))
    ))


def layernorm_backward(ctx, X, gamma, dY, mean, rstd, dX, N, D):
    _check(_lib.tc_layernorm_backward(_as_handle(ctx), _as_handle(X),
                                      _as_handle(gamma), _as_handle(dY),
                                      _as_handle(mean), _as_handle(rstd),
                                      _as_handle(dX), int(N), int(D)))


def rope_forward(ctx, X, cos_t, sin_t, batch, heads, seq, head_dim):
    """Apply RoPE in-place to fp16 X[batch, heads, seq, head_dim]."""
    _check(_lib.tc_rope_forward(
        _as_handle(ctx), _as_handle(X), _as_handle(cos_t), _as_handle(sin_t),
        int(batch), int(heads), int(seq), int(head_dim)
    ))


def rope_backward(ctx, dX, cos_t, sin_t, batch, heads, seq, head_dim):
    """Apply the inverse RoPE rotation in-place to fp16 dX gradients."""
    _check(_lib.tc_rope_backward(
        _as_handle(ctx), _as_handle(dX), _as_handle(cos_t), _as_handle(sin_t),
        int(batch), int(heads), int(seq), int(head_dim)
    ))


def swiglu_forward(ctx, gate, up, out, n):
    """Compute fp16 out = silu(gate) * up."""
    _check(_lib.tc_swiglu_forward(_as_handle(ctx), _as_handle(gate),
                                  _as_handle(up), _as_handle(out), int(n)))


def swiglu_backward(ctx, gate, up, dout, dgate, dup, n):
    _check(_lib.tc_swiglu_backward(_as_handle(ctx), _as_handle(gate),
                                   _as_handle(up), _as_handle(dout),
                                   _as_handle(dgate), _as_handle(dup), int(n)))


def softmax_forward(ctx, X, Y, N, D):
    """Compute row-wise fp16 softmax for X[N, D]."""
    _check(_lib.tc_softmax_forward(_as_handle(ctx), _as_handle(X),
                                   _as_handle(Y), int(N), int(D)))


def softmax_backward(ctx, Y, dY, dX, N, D):
    _check(_lib.tc_softmax_backward(_as_handle(ctx), _as_handle(Y),
                                    _as_handle(dY), _as_handle(dX),
                                    int(N), int(D)))


def adamw_step(ctx, params_fp32, m_fp32, v_fp32, grads, grad_dtype, n,
               lr, beta1, beta2, eps, weight_decay, bias_correction1,
               bias_correction2):
    """Apply one AdamW optimizer step to fp32 params/moments."""
    _check(_lib.tc_adamw_step(
        _as_handle(ctx), _as_handle(params_fp32), _as_handle(m_fp32),
        _as_handle(v_fp32), _as_handle(grads), _dtype(grad_dtype), int(n),
        c_float(float(lr)), c_float(float(beta1)), c_float(float(beta2)),
        c_float(float(eps)), c_float(float(weight_decay)),
        c_float(float(bias_correction1)), c_float(float(bias_correction2))
    ))


def fused_rmsnorm_gemv(ctx, X, gamma, W, Y, M, N, K, eps=1e-5):
    """Compute Y[M, N] = RMSNorm(X[M, K], gamma[K]) @ W[K, N]."""
    _check(_lib.tc_fused_rmsnorm_gemv(
        _as_handle(ctx), _as_handle(X), _as_handle(gamma), _as_handle(W),
        _as_handle(Y), int(M), int(N), int(K), c_float(float(eps))
    ))


def fused_layernorm_gemv(ctx, X, gamma, beta, W, Y, M, N, K, eps=1e-5):
    """Compute Y[M, N] = LayerNorm(X[M, K], gamma[K], beta[K]) @ W[K, N]."""
    _check(_lib.tc_fused_layernorm_gemv(
        _as_handle(ctx), _as_handle(X), _as_handle(gamma), _as_handle(beta),
        _as_handle(W), _as_handle(Y), int(M), int(N), int(K),
        c_float(float(eps))
    ))


def gguf_open(path):
    """Open a GGUF v3 file and return an opaque handle."""
    handle = c_void_p()
    _check(_lib.tc_gguf_open(os.fsencode(path), byref(handle)))
    return handle


def gguf_close(gguf):
    _lib.tc_gguf_close(_as_handle(gguf))


def gguf_tensor_count(gguf):
    return int(_lib.tc_gguf_tensor_count(_as_handle(gguf)))


def gguf_metadata_count(gguf):
    return int(_lib.tc_gguf_metadata_count(_as_handle(gguf)))


def gguf_meta_get_str(gguf, key):
    value = _lib.tc_gguf_meta_get_str(_as_handle(gguf), _bytes(key))
    return value.decode("utf-8", "replace") if value else None


def gguf_meta_get_i64(gguf, key, default=0):
    return int(_lib.tc_gguf_meta_get_i64(_as_handle(gguf), _bytes(key), int(default)))


def gguf_meta_get_f64(gguf, key, default=0.0):
    return float(_lib.tc_gguf_meta_get_f64(_as_handle(gguf), _bytes(key), float(default)))


def gguf_meta_array_count(gguf, key):
    return int(_lib.tc_gguf_meta_array_count(_as_handle(gguf), _bytes(key)))


def gguf_meta_array_get_str(gguf, key, index):
    ptr = c_void_p()
    n = c_size_t()
    _check(_lib.tc_gguf_meta_array_get_str(_as_handle(gguf), _bytes(key),
                                           c_uint64(index), byref(ptr), byref(n)))
    return ctypes.string_at(ptr, n.value).decode("utf-8", "replace")


def gguf_meta_array_get_i64(gguf, key, index, default=0):
    return int(_lib.tc_gguf_meta_array_get_i64(_as_handle(gguf), _bytes(key),
                                               c_uint64(index), int(default)))


def gguf_meta_array_get_f64(gguf, key, index, default=0.0):
    return float(_lib.tc_gguf_meta_array_get_f64(_as_handle(gguf), _bytes(key),
                                                 c_uint64(index), float(default)))


def gguf_get_llama_config(gguf):
    config = TCGGufLlamaConfig()
    _check(_lib.tc_gguf_get_llama_config(_as_handle(gguf), byref(config)))
    return _llama_config_dict(config)


def gguf_get_tensor(gguf, name):
    info = TCGGufTensorInfo()
    _check(_lib.tc_gguf_get_tensor(_as_handle(gguf), _bytes(name), byref(info)))
    return _tensor_info_dict(info)


def gguf_tensor_at(gguf, index):
    info = TCGGufTensorInfo()
    _check(_lib.tc_gguf_tensor_at(_as_handle(gguf), c_uint64(index), byref(info)))
    return _tensor_info_dict(info)


def gguf_tensor_to_buffer(ctx, gguf, name):
    """Copy a named GGUF tensor into a tensorcore buffer."""
    buf = c_void_p()
    _check(_lib.tc_gguf_tensor_to_buffer(_as_handle(ctx), _as_handle(gguf),
                                         _bytes(name), byref(buf)))
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
    _check(_lib.tc_gguf_load_supported_tensors(_as_handle(ctx), _as_handle(gguf),
                                               byref(model)))
    return model


def gguf_loaded_model_free(ctx, model):
    _lib.tc_gguf_loaded_model_free(_as_handle(ctx), _as_handle(model))


def gguf_loaded_tensor_count(model):
    return int(_lib.tc_gguf_loaded_tensor_count(_as_handle(model)))


def gguf_loaded_skipped_tensor_count(model):
    return int(_lib.tc_gguf_loaded_skipped_tensor_count(_as_handle(model)))


def gguf_loaded_tensor_at(model, index):
    info = TCGGufLoadedTensorInfo()
    _check(_lib.tc_gguf_loaded_tensor_at(_as_handle(model), c_uint64(index), byref(info)))
    return _loaded_tensor_info_dict(info)


def gguf_loaded_get_tensor(model, name):
    info = TCGGufLoadedTensorInfo()
    _check(_lib.tc_gguf_loaded_get_tensor(_as_handle(model), _bytes(name), byref(info)))
    return _loaded_tensor_info_dict(info)


class Context:
    """Owned tensorcore context for Python scripts.

    The raw-handle API remains available; this wrapper just gives predictable
    cleanup and accepts Buffer/Stream objects in the operation methods.
    """

    def __init__(self):
        self.handle = init()
        self._buffers = weakref.WeakSet()
        self._streams = weakref.WeakSet()
        self._loaded_models = weakref.WeakSet()
        self._dist_contexts = weakref.WeakSet()
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _remember_buffer(self, buf):
        self._buffers.add(buf)

    def _forget_buffer(self, buf):
        self._buffers.discard(buf)

    def _remember_stream(self, stream):
        self._streams.add(stream)

    def _forget_stream(self, stream):
        self._streams.discard(stream)

    def _remember_loaded_model(self, model):
        self._loaded_models.add(model)

    def _forget_loaded_model(self, model):
        self._loaded_models.discard(model)

    def _remember_dist_context(self, dist):
        self._dist_contexts.add(dist)

    def _forget_dist_context(self, dist):
        self._dist_contexts.discard(dist)

    def close(self):
        if self._closed:
            return
        for stream in list(self._streams):
            stream.close()
        for dist in list(self._dist_contexts):
            dist.close()
        for model in list(self._loaded_models):
            model.close()
        for buf in list(self._buffers):
            buf.close()
        shutdown(self.handle)
        self.handle = None
        self._closed = True

    def device_info(self):
        return device_info(self)

    def memory_tier_usage(self, tier=TC_TIER_L0_DEVICE):
        return memory_tier_usage(self, tier)

    def last_backend(self):
        return last_backend()

    def last_backend_name(self):
        return last_backend_name()

    def buffer(self, nbytes):
        return Buffer(self, nbytes)

    def buffer_from_ptr(self, ptr, nbytes):
        return Buffer(self, handle=buffer_from_ptr(self, ptr, nbytes))

    def buffer_from_array(self, arr):
        return self.buffer(arr.nbytes).write(arr)

    def stream(self):
        return Stream(self)

    def dist(self, backend=TC_DIST_SINGLE, world_size=1, rank=0, rendezvous_url=None):
        return DistContext(self, backend, world_size, rank, rendezvous_url)

    def hip_init(self):
        return hip_init(self)

    def hip_device_info(self):
        return hip_device_info_get(self)

    def hip_select_device(self, index):
        return hip_select_device(self, index)

    def cuda_init(self):
        return cuda_init(self)

    def cuda_select_device(self, index):
        return cuda_select_device(self, index)

    def gemm(self, A, B, C, M, N, K, **kwargs):
        return gemm(self, A, B, C, M, N, K, **kwargs)

    def gemm_async(self, A, B, C, M, N, K, stream, **kwargs):
        return gemm_async(self, A, B, C, M, N, K, stream, **kwargs)

    def gemm_batched(self, A, B, C, batch, M, N, K, **kwargs):
        return gemm_batched(self, A, B, C, batch, M, N, K, **kwargs)

    def attention_forward(self, Q, K, V, O, batch, heads, seq_q, seq_kv, head_dim, **kwargs):
        return attention_forward(self, Q, K, V, O, batch, heads, seq_q, seq_kv, head_dim, **kwargs)

    def attention_forward_async(self, Q, K, V, O, batch, heads, seq_q, seq_kv, head_dim, stream, **kwargs):
        return attention_forward_async(self, Q, K, V, O, batch, heads, seq_q, seq_kv,
                                       head_dim, stream, **kwargs)

    def attention_backward(self, Q, K, V, O, dO, LSE, dQ, dK, dV,
                           batch, heads, seq_q, seq_kv, head_dim, **kwargs):
        return attention_backward(self, Q, K, V, O, dO, LSE, dQ, dK, dV,
                                  batch, heads, seq_q, seq_kv, head_dim, **kwargs)

    def conv2d_forward(self, X, weight, bias, Y, scratch_col,
                       batch, in_channels, out_channels, H, W_in, kH, kW, **kwargs):
        return conv2d_forward(self, X, weight, bias, Y, scratch_col,
                              batch, in_channels, out_channels, H, W_in, kH, kW, **kwargs)

    def conv2d_backward_input(self, dY, weight, dX, scratch_col, scratch_dX_f32,
                              batch, in_channels, out_channels, H, W_in, kH, kW, **kwargs):
        return conv2d_backward_input(self, dY, weight, dX, scratch_col, scratch_dX_f32,
                                     batch, in_channels, out_channels, H, W_in, kH, kW, **kwargs)

    def conv2d_backward_weight(self, X, dY, dW, scratch_col,
                               batch, in_channels, out_channels, H, W_in, kH, kW, **kwargs):
        return conv2d_backward_weight(self, X, dY, dW, scratch_col,
                                      batch, in_channels, out_channels, H, W_in, kH, kW, **kwargs)

    def quantize_weights(self, W_fp16, W_quant, fmt, N, K):
        return quantize_weights(self, W_fp16, W_quant, fmt, N, K)

    def gemv_quantized(self, X, W_quant, Y, fmt, M, N, K):
        return gemv_quantized(self, X, W_quant, Y, fmt, M, N, K)

    def fused_rmsnorm_gemv_quantized(self, X, gamma, W_quant, Y, fmt, M, N, K, eps=1e-5):
        return fused_rmsnorm_gemv_quantized(self, X, gamma, W_quant, Y,
                                            fmt, M, N, K, eps)

    def gemv_quantized_async(self, X, W_quant, Y, fmt, M, N, K, stream):
        return gemv_quantized_async(self, X, W_quant, Y, fmt, M, N, K, stream)

    def rmsnorm_forward(self, X, gamma, Y, rstd_out, N, D, eps=1e-5):
        return rmsnorm_forward(self, X, gamma, Y, rstd_out, N, D, eps)

    def rmsnorm_backward(self, X, gamma, dY, rstd, dX, dgamma, N, D):
        return rmsnorm_backward(self, X, gamma, dY, rstd, dX, dgamma, N, D)

    def layernorm_forward(self, X, gamma, beta, Y, mean_out, rstd_out, N, D, eps=1e-5):
        return layernorm_forward(self, X, gamma, beta, Y, mean_out, rstd_out, N, D, eps)

    def layernorm_backward(self, X, gamma, dY, mean, rstd, dX, N, D):
        return layernorm_backward(self, X, gamma, dY, mean, rstd, dX, N, D)

    def rope_forward(self, X, cos_t, sin_t, batch, heads, seq, head_dim):
        return rope_forward(self, X, cos_t, sin_t, batch, heads, seq, head_dim)

    def rope_backward(self, dX, cos_t, sin_t, batch, heads, seq, head_dim):
        return rope_backward(self, dX, cos_t, sin_t, batch, heads, seq, head_dim)

    def swiglu_forward(self, gate, up, out, n):
        return swiglu_forward(self, gate, up, out, n)

    def swiglu_backward(self, gate, up, dout, dgate, dup, n):
        return swiglu_backward(self, gate, up, dout, dgate, dup, n)

    def softmax_forward(self, X, Y, N, D):
        return softmax_forward(self, X, Y, N, D)

    def softmax_backward(self, Y, dY, dX, N, D):
        return softmax_backward(self, Y, dY, dX, N, D)

    def adamw_step(self, params_fp32, m_fp32, v_fp32, grads, grad_dtype, n,
                   lr, beta1, beta2, eps, weight_decay, bias_correction1,
                   bias_correction2):
        return adamw_step(self, params_fp32, m_fp32, v_fp32, grads, grad_dtype, n,
                          lr, beta1, beta2, eps, weight_decay, bias_correction1,
                          bias_correction2)

    def fused_rmsnorm_gemv(self, X, gamma, W, Y, M, N, K, eps=1e-5):
        return fused_rmsnorm_gemv(self, X, gamma, W, Y, M, N, K, eps)

    def fused_layernorm_gemv(self, X, gamma, beta, W, Y, M, N, K, eps=1e-5):
        return fused_layernorm_gemv(self, X, gamma, beta, W, Y, M, N, K, eps)

    def open_gguf(self, path):
        return GgufFile(path)

    def load_supported_tensors(self, gguf):
        return LoadedModel(self, gguf)


class Buffer:
    """Owned tc_buffer wrapper."""

    def __init__(self, ctx, nbytes=None, handle=None, owned=True):
        if handle is None and nbytes is None:
            raise ValueError("Buffer requires nbytes or an existing handle")
        self.ctx = ctx
        self.handle = handle if handle is not None else buffer_alloc(ctx, nbytes)
        self.owned = owned
        if hasattr(ctx, "_remember_buffer"):
            ctx._remember_buffer(self)

    def __bool__(self):
        return self.handle is not None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        if self.handle is not None:
            if self.owned:
                buffer_free(self.ctx, self.handle)
            self.handle = None
        if hasattr(self.ctx, "_forget_buffer"):
            self.ctx._forget_buffer(self)

    def map(self):
        return buffer_map(self)

    def size(self):
        return buffer_size(self)

    @property
    def nbytes(self):
        return self.size()

    def set_tier_hint(self, hint):
        buffer_set_tier_hint(self, hint)
        return self

    def get_tier(self):
        return buffer_get_tier(self)

    def promote_async(self, target_tier=TC_TIER_L0_DEVICE, stream=None):
        buffer_promote_async(self, target_tier, stream)
        return self

    def demote_async(self, target_tier=TC_TIER_L0_DEVICE, stream=None):
        buffer_demote_async(self, target_tier, stream)
        return self

    def tier_sync(self):
        buffer_tier_sync(self)
        return self

    def write(self, arr):
        buffer_write(self, arr)
        return self

    def read(self, arr):
        buffer_read(self, arr)
        return arr

    def to_numpy(self, shape, dtype):
        import numpy as np
        arr = np.empty(shape, dtype=dtype)
        self.read(arr)
        return arr


class Stream:
    """Owned tc_stream wrapper."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.handle = stream_create(ctx)
        if hasattr(ctx, "_remember_stream"):
            ctx._remember_stream(self)

    def __bool__(self):
        return self.handle is not None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def sync(self):
        stream_sync(self)

    def close(self):
        if self.handle is not None:
            stream_destroy(self.ctx, self.handle)
            self.handle = None
        if hasattr(self.ctx, "_forget_stream"):
            self.ctx._forget_stream(self)


class DistContext:
    """Owned tc_dist_ctx wrapper."""

    def __init__(self, ctx, backend=TC_DIST_SINGLE, world_size=1, rank=0,
                 rendezvous_url=None):
        self.ctx = ctx
        self.handle = dist_init(ctx, backend, world_size, rank, rendezvous_url)
        self._diloco_contexts = weakref.WeakSet()
        if hasattr(ctx, "_remember_dist_context"):
            ctx._remember_dist_context(self)

    def __bool__(self):
        return self.handle is not None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def world_size(self):
        return dist_world_size(self)

    @property
    def rank(self):
        return dist_rank(self)

    def allreduce(self, buf, num_elements, dtype="f32", op=TC_REDUCE_SUM):
        allreduce(self, buf, num_elements, dtype, op)
        return buf

    def broadcast(self, buf, num_elements, dtype="f32", root=0):
        broadcast(self, buf, num_elements, dtype, root)
        return buf

    def allgather(self, src, dst, num_elements_per_rank, dtype="f32"):
        allgather(self, src, dst, num_elements_per_rank, dtype)
        return dst

    def barrier(self):
        barrier(self)

    def _remember_diloco(self, diloco):
        self._diloco_contexts.add(diloco)

    def _forget_diloco(self, diloco):
        self._diloco_contexts.discard(diloco)

    def diloco(self, config=None, **kwargs):
        return DiLoCoContext(self, config=config, **kwargs)

    def close(self):
        if self.handle is not None:
            for diloco in list(self._diloco_contexts):
                diloco.close()
            dist_finalize(self)
            self.handle = None
        if hasattr(self.ctx, "_forget_dist_context"):
            self.ctx._forget_dist_context(self)


class DiLoCoContext:
    """Owned tc_diloco_ctx wrapper layered on a DistContext."""

    def __init__(self, dist, config=None, **kwargs):
        self.dist = dist
        self.handle = diloco_init(dist, config=config, **kwargs)
        self._closed = False
        if hasattr(dist, "_remember_diloco"):
            dist._remember_diloco(self)

    def __bool__(self):
        return self.handle is not None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def add_parameter(self, name, theta_local, num_elements, dtype="f32"):
        diloco_add_parameter(self, name, theta_local, num_elements, dtype)
        return self

    def step(self):
        return diloco_step(self)

    def apply_outer(self):
        diloco_apply_outer(self)
        return self

    @property
    def outer_steps_completed(self):
        return diloco_outer_steps_completed(self)

    @property
    def inner_steps_completed(self):
        return diloco_inner_steps_completed(self)

    @property
    def last_outer_step_seconds(self):
        return diloco_last_outer_step_seconds(self)

    @property
    def last_outer_bytes_sent(self):
        return diloco_last_outer_bytes_sent(self)

    def close(self):
        if not self._closed and self.handle is not None:
            diloco_finalize(self)
            self.handle = None
            self._closed = True
        if hasattr(self.dist, "_forget_diloco"):
            self.dist._forget_diloco(self)


class GgufFile:
    """Owned GGUF file handle."""

    def __init__(self, path):
        self.path = os.fspath(path)
        self.handle = gguf_open(path)
        self._closed = False

    def __bool__(self):
        return self.handle is not None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        if not self._closed and self.handle is not None:
            gguf_close(self.handle)
            self.handle = None
            self._closed = True

    def tensor_count(self):
        return gguf_tensor_count(self)

    def metadata_count(self):
        return gguf_metadata_count(self)

    def get_tensor(self, name):
        return gguf_get_tensor(self, name)

    def tensor_at(self, index):
        return gguf_tensor_at(self, index)

    def meta_get_str(self, key):
        return gguf_meta_get_str(self, key)

    def meta_get_i64(self, key, default=0):
        return gguf_meta_get_i64(self, key, default)

    def meta_get_f64(self, key, default=0.0):
        return gguf_meta_get_f64(self, key, default)

    def meta_array_count(self, key):
        return gguf_meta_array_count(self, key)

    def meta_array_get_str(self, key, index):
        return gguf_meta_array_get_str(self, key, index)

    def llama_config(self):
        return gguf_get_llama_config(self)

    def tensor_to_buffer(self, ctx, name):
        handle = gguf_tensor_to_buffer(ctx, self, name)
        return Buffer(ctx, handle=handle, owned=True)

    def load_supported_tensors(self, ctx):
        return LoadedModel(ctx, self)


class LoadedModel:
    """Owned tc_gguf_loaded_model wrapper."""

    def __init__(self, ctx, gguf):
        self.ctx = ctx
        self.handle = gguf_load_supported_tensors(ctx, gguf)
        self._closed = False
        if hasattr(ctx, "_remember_loaded_model"):
            ctx._remember_loaded_model(self)

    def __bool__(self):
        return self.handle is not None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        if not self._closed and self.handle is not None:
            gguf_loaded_model_free(self.ctx, self.handle)
            self.handle = None
            self._closed = True
        if hasattr(self.ctx, "_forget_loaded_model"):
            self.ctx._forget_loaded_model(self)

    def tensor_count(self):
        return gguf_loaded_tensor_count(self)

    def skipped_tensor_count(self):
        return gguf_loaded_skipped_tensor_count(self)

    def tensor_at(self, index):
        return LoadedTensor(self, gguf_loaded_tensor_at(self, index))

    def get_tensor(self, name):
        return LoadedTensor(self, gguf_loaded_get_tensor(self, name))

    def quantized_matrix(self, name):
        return QuantizedMatrix(self, name)


class LoadedTensor(dict):
    """Loaded tensor metadata with a strong reference to its owning model."""

    def __init__(self, model, info):
        super().__init__(info)
        self.model = model

    def _check_alive(self):
        if getattr(self.model, "_closed", False):
            raise RuntimeError("loaded tensor buffer is no longer valid; owning model is closed")

    def __getitem__(self, key):
        if key == "buffer":
            self._check_alive()
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "buffer":
            self._check_alive()
        return super().get(key, default)

    @property
    def buffer(self):
        return self["buffer"]


class QuantizedMatrix:
    """Loaded GGUF Q4_0/Q8_0 matrix ready for tc_gemv_quantized."""

    def __init__(self, model, name):
        self.model = model
        self.name = str(name)
        self.tensor = model.get_tensor(name)
        self.info = gguf_loaded_tensor_quantized_matrix_info(self.tensor)
        self.N = self.info["N"]
        self.K = self.info["K"]
        self.quant_type = self.info["quant_type"]
        self.gguf_type = self.info["gguf_type"]
        self.n_bytes = self.info["n_bytes"]
        self.buffer = self.info["buffer"]

    def _check_alive(self):
        if getattr(self.model, "_closed", False):
            raise RuntimeError("quantized matrix buffer is no longer valid; owning model is closed")

    def output(self, M=1):
        self._check_alive()
        return Buffer(self.model.ctx, int(M) * self.N * 2)

    def gemv(self, X, Y, M=1, ctx=None):
        self._check_alive()
        run_ctx = self.model.ctx if ctx is None else ctx
        gemv_quantized(run_ctx, X, self.buffer, Y, self.quant_type,
                       int(M), self.N, self.K)
        return Y

    def gemv_async(self, X, Y, stream, M=1, ctx=None):
        self._check_alive()
        run_ctx = self.model.ctx if ctx is None else ctx
        gemv_quantized_async(run_ctx, X, self.buffer, Y, self.quant_type,
                             int(M), self.N, self.K, stream)
        return Y


def version():
    return _lib.tc_version().decode() if _lib else "(unloaded)"
