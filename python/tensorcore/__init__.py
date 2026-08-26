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
TC_ERR_ABI_MISMATCH = -12
TC_ERR_BUSY = -13
TC_ERR_AUTH = -14
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

TC_RUNTIME_CAPABILITIES_ABI_VERSION_1 = 1
TC_RUNTIME_CAPABILITIES_ABI_VERSION_CURRENT = 1

TC_CAPABILITY_GEMM_F32 = 1 << 0
TC_CAPABILITY_DISTRIBUTED_SINGLE = 1 << 1
TC_CAPABILITY_DISTRIBUTED_GLOO = 1 << 2
TC_CAPABILITY_DILOCO = 1 << 3
TC_CAPABILITY_DILOCO_SPARSE_GLOO = 1 << 4
TC_CAPABILITY_REMOTE_TENSOR = 1 << 5
TC_CAPABILITY_DILOCO_ASYNC_SNAPSHOT_SAFE = 1 << 6
TC_CAPABILITY_DILOCO_CHECKPOINT_RESUME = 1 << 7
TC_CAPABILITY_DILOCO_ELASTIC_MEMBERSHIP = 1 << 8
TC_CAPABILITY_DILOCO_FP16_WIRE = 1 << 9
TC_CAPABILITY_TRANSPORT_IDENTITY_AUTH = 1 << 10
TC_CAPABILITY_DILOCO_CAPABILITY_QUERY = 1 << 11
TC_CAPABILITY_DILOCO_STATE_ABI_V2 = 1 << 12
TC_CAPABILITY_GEMM_F16 = 1 << 13
TC_CAPABILITY_GEMM_BF16 = 1 << 14
TC_CAPABILITY_GEMM_I8 = 1 << 15
TC_CAPABILITY_V1_KNOWN_MASK = (
    TC_CAPABILITY_GEMM_F32 |
    TC_CAPABILITY_DISTRIBUTED_SINGLE |
    TC_CAPABILITY_DISTRIBUTED_GLOO |
    TC_CAPABILITY_DILOCO |
    TC_CAPABILITY_DILOCO_SPARSE_GLOO |
    TC_CAPABILITY_REMOTE_TENSOR |
    TC_CAPABILITY_DILOCO_ASYNC_SNAPSHOT_SAFE |
    TC_CAPABILITY_DILOCO_CHECKPOINT_RESUME |
    TC_CAPABILITY_DILOCO_ELASTIC_MEMBERSHIP |
    TC_CAPABILITY_DILOCO_FP16_WIRE |
    TC_CAPABILITY_TRANSPORT_IDENTITY_AUTH |
    TC_CAPABILITY_DILOCO_CAPABILITY_QUERY |
    TC_CAPABILITY_DILOCO_STATE_ABI_V2 |
    TC_CAPABILITY_GEMM_F16 |
    TC_CAPABILITY_GEMM_BF16 |
    TC_CAPABILITY_GEMM_I8
)

TC_BACKEND_MASK_SIMDGROUP_MATRIX = 1 << TC_BACKEND_SIMDGROUP_MATRIX
TC_BACKEND_MASK_TENSOROPS_M5 = 1 << TC_BACKEND_TENSOROPS_M5
TC_BACKEND_MASK_MPS = 1 << TC_BACKEND_MPS
TC_BACKEND_MASK_ACCELERATE_CPU = 1 << TC_BACKEND_ACCELERATE_CPU
TC_BACKEND_MASK_PORTABLE_CPU = 1 << TC_BACKEND_PORTABLE_CPU
TC_BACKEND_MASK_METAL_COMPUTE = 1 << TC_BACKEND_METAL_COMPUTE
TC_BACKEND_MASK_CUDA = 1 << TC_BACKEND_CUDA
TC_BACKEND_MASK_HIP = 1 << TC_BACKEND_HIP

TC_TRANSPORT_AUTH_ABI_VERSION_1 = 1
TC_TRANSPORT_AUTH_ABI_VERSION_CURRENT = 1
TC_TRANSPORT_AUTH_MIN_SECRET_BYTES = 32
TC_TRANSPORT_AUTH_MAX_IDENTITY_BYTES = 127

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

TC_DILOCO_ASYNC_IDLE = 0
TC_DILOCO_ASYNC_RUNNING = 1
TC_DILOCO_ASYNC_READY = 2
TC_DILOCO_ASYNC_FAILED = 3

TC_DILOCO_STATE_ABI_VERSION_1 = 1
TC_DILOCO_STATE_ABI_VERSION_2 = 2
TC_DILOCO_STATE_ABI_VERSION_MIN = TC_DILOCO_STATE_ABI_VERSION_1
TC_DILOCO_STATE_ABI_VERSION_CURRENT = TC_DILOCO_STATE_ABI_VERSION_2
TC_DILOCO_STATE_V1_HEADER_SIZE = 32
TC_DILOCO_STATE_V2_HEADER_SIZE = 64
TC_DILOCO_STATE_V2_DIGEST_BYTES = 32

TC_DILOCO_CAPABILITIES_ABI_VERSION_1 = 1
TC_DILOCO_CAPABILITIES_ABI_VERSION_CURRENT = TC_DILOCO_CAPABILITIES_ABI_VERSION_1

TC_DILOCO_STATE_FEATURE_OUTER_ANCHOR = 1 << 0
TC_DILOCO_STATE_FEATURE_OUTER_MOMENTS = 1 << 1
TC_DILOCO_STATE_FEATURE_ERROR_FEEDBACK = 1 << 2
TC_DILOCO_STATE_FEATURE_COUNTERS = 1 << 3
TC_DILOCO_STATE_FEATURE_PENDING_ROUND = 1 << 4
TC_DILOCO_STATE_FEATURE_TOPOLOGY_EPOCH = 1 << 5
TC_DILOCO_STATE_FEATURE_MEMBERSHIP_EPOCH = 1 << 6
TC_DILOCO_STATE_FEATURE_PAYLOAD_SHA256 = 1 << 7
TC_DILOCO_STATE_FEATURE_DETERMINISTIC_LAYOUT = 1 << 8
TC_DILOCO_STATE_FEATURE_ATOMIC_RESTORE = 1 << 9

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


class TCRuntimeCapabilities(Structure):
    _fields_ = [
        ("struct_size", c_uint32),
        ("abi_version", c_uint32),
        ("runtime_version_major", c_uint32),
        ("runtime_version_minor", c_uint32),
        ("runtime_version_patch", c_uint32),
        ("reserved0", c_uint32),
        ("known_capability_mask", c_uint64),
        ("available_capability_mask", c_uint64),
        ("compiled_backend_mask", c_uint64),
        ("available_backend_mask", c_uint64),
        ("reserved", c_uint64 * 4),
    ]


TC_RUNTIME_CAPABILITIES_V1_MIN_SIZE = (
    TCRuntimeCapabilities.available_capability_mask.offset + ctypes.sizeof(c_uint64)
)


class TCTransportAuthKey(Structure):
    _fields_ = [
        ("identity", c_char_p),
        ("key_id", c_uint64),
        ("secret", c_void_p),
        ("secret_bytes", c_size_t),
    ]


class TCTransportAuthConfig(Structure):
    _fields_ = [
        ("abi_version", c_uint32),
        ("local_identity", c_char_p),
        ("active_key_id", c_uint64),
        ("keys", POINTER(TCTransportAuthKey)),
        ("key_count", c_size_t),
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


class TCDiLoCoCapabilities(Structure):
    _fields_ = [
        ("struct_size", c_uint32),
        ("abi_version", c_uint32),
        ("runtime_version_major", c_uint32),
        ("runtime_version_minor", c_uint32),
        ("runtime_version_patch", c_uint32),
        ("reserved0", c_uint32),
        ("state_abi_version_min", c_uint32),
        ("state_abi_version_max", c_uint32),
        ("state_abi_version_current", c_uint32),
        ("state_header_size", c_uint32),
        ("state_payload_digest_bytes", c_uint32),
        ("reserved1", c_uint32),
        ("outer_optimizer_supported_mask", c_uint64),
        ("outer_optimizer_serializable_mask", c_uint64),
        ("compress_supported_mask", c_uint64),
        ("compress_serializable_mask", c_uint64),
        ("compress_sparse_wire_mask", c_uint64),
        ("compress_dense_fp32_wire_mask", c_uint64),
        ("state_feature_mask", c_uint64),
        ("async_overlap_supported", c_uint32),
        ("async_state_serializable", c_uint32),
        ("tolerate_dropouts_supported", c_uint32),
        ("reserved2", c_uint32),
        ("reserved", c_uint64 * 4),
    ]


TC_DILOCO_CAPABILITIES_V1_MIN_SIZE = (
    TCDiLoCoCapabilities.state_feature_mask.offset + ctypes.sizeof(c_uint64)
)


TCCheckpointRecomputeFn = CFUNCTYPE(c_int, c_void_p)
_CHECKPOINT_CALLBACKS = {}


if _lib is not None:
    _lib.tc_init.argtypes = [POINTER(c_void_p)];          _lib.tc_init.restype = c_int
    _lib.tc_shutdown.argtypes = [c_void_p];               _lib.tc_shutdown.restype = c_int
    _lib.tc_device_info_get.argtypes = [c_void_p, POINTER(TCDeviceInfo)]; _lib.tc_device_info_get.restype = c_int
    _lib.tc_runtime_capabilities_get.argtypes = [
        c_void_p, c_uint32, POINTER(TCRuntimeCapabilities), c_size_t,
    ]
    _lib.tc_runtime_capabilities_get.restype = c_int
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
    # Poincaré ball ops (see include/tensorcore/poincare.h).
    _lib.tc_poincare_mobius_add.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_float, c_int, c_int]
    _lib.tc_poincare_mobius_add.restype = c_int
    _lib.tc_poincare_distance.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_float, c_int, c_int]
    _lib.tc_poincare_distance.restype = c_int
    _lib.tc_poincare_conformal_factor.argtypes = [c_void_p, c_void_p, c_void_p, c_float, c_int, c_int]
    _lib.tc_poincare_conformal_factor.restype = c_int
    _lib.tc_poincare_exp_map_zero.argtypes = [c_void_p, c_void_p, c_void_p, c_float, c_int, c_int]
    _lib.tc_poincare_exp_map_zero.restype = c_int
    _lib.tc_poincare_log_map_zero.argtypes = [c_void_p, c_void_p, c_void_p, c_float, c_int, c_int]
    _lib.tc_poincare_log_map_zero.restype = c_int
    _lib.tc_poincare_exp_map.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_float, c_int, c_int]
    _lib.tc_poincare_exp_map.restype = c_int
    _lib.tc_poincare_log_map.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_float, c_int, c_int]
    _lib.tc_poincare_log_map.restype = c_int
    _lib.tc_poincare_parallel_transport.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_float, c_int, c_int]
    _lib.tc_poincare_parallel_transport.restype = c_int
    # Lorentz / hyperboloid ops (see include/tensorcore/lorentz.h).
    # Raw single-vector ABI: float* + size_t, no buffers; suitable for
    # one-off math (embedding conversions, geodesic computations) and as
    # the inner kernel for batched wrappers.
    _f32p = POINTER(c_float)
    _lib.tc_lorentz_minkowski.argtypes = [_f32p, _f32p, c_size_t]
    _lib.tc_lorentz_minkowski.restype = c_float
    _lib.tc_lorentz_project.argtypes = [_f32p, c_size_t, c_float]
    _lib.tc_lorentz_project.restype = None
    _lib.tc_lorentz_tangent_project.argtypes = [_f32p, _f32p, c_size_t]
    _lib.tc_lorentz_tangent_project.restype = None
    _lib.tc_lorentz_inner_product.argtypes = [_f32p, _f32p, _f32p, c_size_t]
    _lib.tc_lorentz_inner_product.restype = c_float
    _lib.tc_lorentz_add.argtypes = [_f32p, _f32p, _f32p, _f32p, c_size_t]
    _lib.tc_lorentz_add.restype = None
    _lib.tc_lorentz_scale.argtypes = [_f32p, c_float, _f32p, c_size_t]
    _lib.tc_lorentz_scale.restype = None
    _lib.tc_lorentz_exp.argtypes = [_f32p, _f32p, _f32p, c_size_t, c_float]
    _lib.tc_lorentz_exp.restype = None
    _lib.tc_lorentz_log.argtypes = [_f32p, _f32p, _f32p, c_size_t, c_float]
    _lib.tc_lorentz_log.restype = None
    _lib.tc_lorentz_distance.argtypes = [_f32p, _f32p, c_size_t, c_float]
    _lib.tc_lorentz_distance.restype = c_float
    _lib.tc_lorentz_parallel_transport.argtypes = [_f32p, _f32p, _f32p, _f32p, c_size_t, c_float]
    _lib.tc_lorentz_parallel_transport.restype = None
    _lib.tc_lorentz_to_poincare.argtypes = [_f32p, _f32p, c_size_t, c_float]
    _lib.tc_lorentz_to_poincare.restype = None
    _lib.tc_poincare_to_lorentz.argtypes = [_f32p, _f32p, c_size_t, c_float]
    _lib.tc_poincare_to_lorentz.restype = None
    # Sphere ops (see include/tensorcore/sphere.h). Raw-vector ABI;
    # serves as one-off math + the inner kernel for product manifolds.
    _lib.tc_sphere_inner.argtypes = [_f32p, _f32p, c_size_t]
    _lib.tc_sphere_inner.restype = c_float
    _lib.tc_sphere_project.argtypes = [_f32p, c_size_t, c_float]
    _lib.tc_sphere_project.restype = None
    _lib.tc_sphere_tangent_project.argtypes = [_f32p, _f32p, c_size_t]
    _lib.tc_sphere_tangent_project.restype = None
    _lib.tc_sphere_inner_product.argtypes = [_f32p, _f32p, _f32p, c_size_t]
    _lib.tc_sphere_inner_product.restype = c_float
    _lib.tc_sphere_add.argtypes = [_f32p, _f32p, _f32p, _f32p, c_size_t]
    _lib.tc_sphere_add.restype = None
    _lib.tc_sphere_scale.argtypes = [_f32p, c_float, _f32p, c_size_t]
    _lib.tc_sphere_scale.restype = None
    _lib.tc_sphere_exp.argtypes = [_f32p, _f32p, _f32p, c_size_t, c_float]
    _lib.tc_sphere_exp.restype = None
    _lib.tc_sphere_log.argtypes = [_f32p, _f32p, _f32p, c_size_t, c_float]
    _lib.tc_sphere_log.restype = None
    _lib.tc_sphere_distance.argtypes = [_f32p, _f32p, c_size_t, c_float]
    _lib.tc_sphere_distance.restype = c_float
    _lib.tc_sphere_parallel_transport.argtypes = [_f32p, _f32p, _f32p, _f32p, c_size_t, c_float]
    _lib.tc_sphere_parallel_transport.restype = None
    _lib.tc_sphere_slerp.argtypes = [_f32p, _f32p, c_float, _f32p, c_size_t, c_float]
    _lib.tc_sphere_slerp.restype = None
    # Flat torus ops (see include/tensorcore/torus.h).
    _lib.tc_torus_project.argtypes = [_f32p, c_size_t, c_float]
    _lib.tc_torus_project.restype = None
    _lib.tc_torus_exp.argtypes = [_f32p, _f32p, _f32p, c_size_t, c_float]
    _lib.tc_torus_exp.restype = None
    _lib.tc_torus_log.argtypes = [_f32p, _f32p, _f32p, c_size_t, c_float]
    _lib.tc_torus_log.restype = None
    _lib.tc_torus_distance.argtypes = [_f32p, _f32p, c_size_t, c_float]
    _lib.tc_torus_distance.restype = c_float
    _lib.tc_torus_parallel_transport.argtypes = [
        _f32p, _f32p, _f32p, _f32p, c_size_t, c_float,
    ]
    _lib.tc_torus_parallel_transport.restype = None
    # Lie group ops (see include/tensorcore/lie_groups.h).
    _lib.tc_su2_exp.argtypes = [c_float, c_float, c_float, _f32p]
    _lib.tc_su2_exp.restype = None
    _lib.tc_su2_log.argtypes = [_f32p, _f32p, _f32p, _f32p]
    _lib.tc_su2_log.restype = None
    _lib.tc_su2_mul.argtypes = [_f32p, _f32p, _f32p]
    _lib.tc_su2_mul.restype = None
    _lib.tc_so3_exp.argtypes = [c_float, c_float, c_float, _f32p]
    _lib.tc_so3_exp.restype = None
    _lib.tc_so3_log.argtypes = [_f32p, _f32p, _f32p, _f32p]
    _lib.tc_so3_log.restype = None
    _lib.tc_so3_mul.argtypes = [_f32p, _f32p, _f32p]
    _lib.tc_so3_mul.restype = None
    _lib.tc_su2_to_so3.argtypes = [_f32p, _f32p]
    _lib.tc_su2_to_so3.restype = None
    # Product manifold (see include/tensorcore/product_manifold.h).
    # Factor struct mirrors tc_factor_t in C: { kind, intrinsic_dim, curvature }.
    class _TCFactor(Structure):
        _fields_ = [("kind", c_int32),
                    ("intrinsic_dim", c_int32),
                    ("curvature", c_float)]
    _lib._tc_factor_struct = _TCFactor
    _lib.tc_factor_ambient_dim.argtypes = [POINTER(_TCFactor)]
    _lib.tc_factor_ambient_dim.restype = c_int32
    _lib.tc_product_ambient_dim.argtypes = [POINTER(_TCFactor), c_int32]
    _lib.tc_product_ambient_dim.restype = c_int32
    _lib.tc_product_project.argtypes = [POINTER(_TCFactor), c_int32, _f32p]
    _lib.tc_product_project.restype = None
    _lib.tc_product_exp.argtypes = [POINTER(_TCFactor), c_int32, _f32p, _f32p, _f32p]
    _lib.tc_product_exp.restype = None
    _lib.tc_product_log.argtypes = [POINTER(_TCFactor), c_int32, _f32p, _f32p, _f32p]
    _lib.tc_product_log.restype = None
    _lib.tc_product_distance.argtypes = [POINTER(_TCFactor), c_int32, _f32p, _f32p]
    _lib.tc_product_distance.restype = c_float
    _lib.tc_product_parallel_transport.argtypes = [POINTER(_TCFactor), c_int32, _f32p, _f32p, _f32p, _f32p]
    _lib.tc_product_parallel_transport.restype = None
    # Quantum gates + state-vector apply (see include/tensorcore/quantum_gates.h).
    # State is a fp32 array of 2*2^n_qubits floats (interleaved complex).
    # Gate type values match QGTL's gate_type_t for interop.
    _lib.tc_gate_matrix_1q.argtypes = [c_int, _f32p, _f32p]
    _lib.tc_gate_matrix_1q.restype = None
    _lib.tc_gate_matrix_2q.argtypes = [c_int, _f32p, _f32p]
    _lib.tc_gate_matrix_2q.restype = None
    _lib.tc_gate_matrix_3q.argtypes = [c_int, _f32p, _f32p]
    _lib.tc_gate_matrix_3q.restype = None
    _lib.tc_qstate_zero.argtypes = [_f32p, c_int]
    _lib.tc_qstate_zero.restype = None
    _lib.tc_qstate_apply_1q_unitary.argtypes = [_f32p, c_int, c_int, _f32p]
    _lib.tc_qstate_apply_1q_unitary.restype = None
    _lib.tc_qstate_apply_2q_unitary.argtypes = [_f32p, c_int, c_int, c_int, _f32p]
    _lib.tc_qstate_apply_2q_unitary.restype = None
    _lib.tc_qstate_apply_3q_unitary.argtypes = [
        _f32p, c_int, c_int, c_int, c_int, _f32p,
    ]
    _lib.tc_qstate_apply_3q_unitary.restype = None
    _lib.tc_qstate_apply_gate.argtypes = [_f32p, c_int, c_int, POINTER(c_int), _f32p]
    _lib.tc_qstate_apply_gate.restype = None
    _lib.tc_qstate_prob_one.argtypes = [_f32p, c_int, c_int]
    _lib.tc_qstate_prob_one.restype = c_float
    _lib.tc_qstate_norm_sq.argtypes = [_f32p, c_int]
    _lib.tc_qstate_norm_sq.restype = c_float
    _lib.tc_qstate_inner.argtypes = [_f32p, _f32p, c_int, _f32p, _f32p]
    _lib.tc_qstate_inner.restype = None
    class _TCPauliTerm(Structure):
        _fields_ = [("n_paulis", c_int32),
                    ("axes", POINTER(c_int32)),
                    ("qubits", POINTER(c_int32)),
                    ("coef", c_float)]
    _lib._tc_pauli_term_struct = _TCPauliTerm
    _lib.tc_qstate_trotter_step.argtypes = [
        _f32p, c_int, POINTER(_TCPauliTerm), c_int, c_float, c_int,
    ]
    _lib.tc_qstate_trotter_step.restype = None
    _lib.tc_quantum_geometric_tensor.argtypes = [
        _f32p, POINTER(_f32p), c_int, c_int, _f32p,
    ]
    _lib.tc_quantum_geometric_tensor.restype = None
    # Density matrices / open systems (see include/tensorcore/density_matrix.h).
    _lib.tc_dmstate_zero.argtypes = [_f32p, c_int]
    _lib.tc_dmstate_zero.restype = None
    _lib.tc_dmstate_from_pure.argtypes = [_f32p, _f32p, c_int]
    _lib.tc_dmstate_from_pure.restype = None
    _lib.tc_dmstate_apply_1q_unitary.argtypes = [_f32p, c_int, c_int, _f32p]
    _lib.tc_dmstate_apply_1q_unitary.restype = None
    _lib.tc_dmstate_apply_2q_unitary.argtypes = [
        _f32p, c_int, c_int, c_int, _f32p,
    ]
    _lib.tc_dmstate_apply_2q_unitary.restype = None
    _lib.tc_dmstate_partial_trace.argtypes = [_f32p, c_int, c_int, _f32p]
    _lib.tc_dmstate_partial_trace.restype = None
    _lib.tc_dmstate_apply_kraus_1q.argtypes = [
        _f32p, c_int, c_int, _f32p, c_int,
    ]
    _lib.tc_dmstate_apply_kraus_1q.restype = None
    _lib.tc_dmstate_trace.argtypes = [_f32p, c_int, _f32p, _f32p]
    _lib.tc_dmstate_trace.restype = None
    _lib.tc_dmstate_purity.argtypes = [_f32p, c_int]
    _lib.tc_dmstate_purity.restype = c_float
    _lib.tc_dmstate_lindblad_step.argtypes = [
        _f32p, c_int, POINTER(_TCPauliTerm), c_int, _f32p,
        POINTER(c_int), c_int, c_float, c_int,
    ]
    _lib.tc_dmstate_lindblad_step.restype = None
    # Phase attention + Born-rule (see include/tensorcore/phase_attention.h).
    _lib.tc_phase_attention_combine.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int]
    _lib.tc_phase_attention_combine.restype = c_int
    _lib.tc_born_rule_output.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int]
    _lib.tc_born_rule_output.restype = c_int
    # RiemannianAdam (see include/tensorcore/riemannian_adam.h).
    _lib.tc_riemannian_adam_step_poincare.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float]
    _lib.tc_riemannian_adam_step_poincare.restype = c_int
    _lib.tc_riemannian_adam_step_sphere.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_float, c_float, c_float, c_float, c_float, c_float, c_float]
    _lib.tc_riemannian_adam_step_sphere.restype = c_int
    _lib.tc_riemannian_adam_step_euclidean.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_float, c_float, c_float, c_float, c_float, c_float, c_float]
    _lib.tc_riemannian_adam_step_euclidean.restype = c_int
    _lib.tc_riemannian_adam_step_poincare_reference.argtypes = list(
        _lib.tc_riemannian_adam_step_poincare.argtypes)
    _lib.tc_riemannian_adam_step_poincare_reference.restype = c_int
    _lib.tc_riemannian_adam_step_sphere_reference.argtypes = list(
        _lib.tc_riemannian_adam_step_sphere.argtypes)
    _lib.tc_riemannian_adam_step_sphere_reference.restype = c_int
    _lib.tc_riemannian_adam_step_euclidean_reference.argtypes = list(
        _lib.tc_riemannian_adam_step_euclidean.argtypes)
    _lib.tc_riemannian_adam_step_euclidean_reference.restype = c_int
    _lib.tc_riemannian_offmanifold_count.argtypes = []
    _lib.tc_riemannian_offmanifold_count.restype = c_uint64
    _lib.tc_riemannian_offmanifold_counts.argtypes = [POINTER(c_uint64), POINTER(c_uint64)]
    _lib.tc_riemannian_offmanifold_counts.restype = None
    _lib.tc_riemannian_offmanifold_reset.argtypes = []
    _lib.tc_riemannian_offmanifold_reset.restype = None
    # 2:4 structured-sparse GEMM (see include/tensorcore/sparse_gemm.h).
    _lib.tc_sparse_24_prune.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int]
    _lib.tc_sparse_24_prune.restype = c_int
    _lib.tc_sparse_24_check.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int]
    _lib.tc_sparse_24_check.restype = c_int
    _lib.tc_sparse_24_gemm.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_float, c_float]
    _lib.tc_sparse_24_gemm.restype = c_int
    _lib.tc_sparse_24_available.argtypes = []
    _lib.tc_sparse_24_available.restype = c_int
    _lib.tc_gguf_open.argtypes = [c_char_p, POINTER(c_void_p)]
    # Remote tensor-fetch transport (Kimi inference weight paging,
    # see include/tensorcore/remote_tensor.h).
    _lib.tc_remote_init.argtypes = [c_void_p, c_int, c_char_p, POINTER(c_void_p)]
    _lib.tc_remote_init.restype = c_int
    _lib.tc_remote_init_authenticated.argtypes = [
        c_void_p, c_int, c_char_p, POINTER(TCTransportAuthConfig), POINTER(c_void_p)]
    _lib.tc_remote_init_authenticated.restype = c_int
    _lib.tc_remote_shutdown.argtypes = [c_void_p]
    _lib.tc_remote_shutdown.restype = c_int
    _lib.tc_remote_register_tensor.argtypes = [c_void_p, c_char_p, c_void_p, c_size_t]
    _lib.tc_remote_register_tensor.restype = c_int
    _lib.tc_remote_unregister_tensor.argtypes = [c_void_p, c_char_p]
    _lib.tc_remote_unregister_tensor.restype = c_int
    _lib.tc_remote_registered_count.argtypes = [c_void_p]
    _lib.tc_remote_registered_count.restype = c_size_t
    _lib.tc_remote_connect.argtypes = [c_void_p, c_char_p]
    _lib.tc_remote_connect.restype = c_int
    _lib.tc_remote_connect_authenticated.argtypes = [
        c_void_p, c_char_p, c_char_p, POINTER(c_int)]
    _lib.tc_remote_connect_authenticated.restype = c_int
    _lib.tc_remote_auth_rotate.argtypes = [c_void_p, POINTER(TCTransportAuthConfig)]
    _lib.tc_remote_auth_rotate.restype = c_int
    _lib.tc_remote_peer_identity.argtypes = [c_void_p, c_int]
    _lib.tc_remote_peer_identity.restype = c_char_p
    _lib.tc_remote_peer_key_id.argtypes = [c_void_p, c_int]
    _lib.tc_remote_peer_key_id.restype = c_uint64
    _lib.tc_remote_auth_failure_count.argtypes = [c_void_p]
    _lib.tc_remote_auth_failure_count.restype = c_uint64
    _lib.tc_remote_tensor_fetch.argtypes = [c_void_p, c_int, c_char_p, c_void_p, c_size_t]
    _lib.tc_remote_tensor_fetch.restype = c_int
    _lib.tc_remote_total_bytes_served.argtypes = [c_void_p]
    _lib.tc_remote_total_bytes_served.restype = c_uint64
    _lib.tc_remote_total_bytes_fetched.argtypes = [c_void_p]
    _lib.tc_remote_total_bytes_fetched.restype = c_uint64
    _lib.tc_remote_fetch_count.argtypes = [c_void_p]
    _lib.tc_remote_fetch_count.restype = c_uint64
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
    _lib.tc_dist_init_authenticated.argtypes = [
        c_void_p, c_int, c_int, c_int, c_char_p, POINTER(c_char_p), c_size_t,
        POINTER(TCTransportAuthConfig), POINTER(c_void_p)]
    _lib.tc_dist_init_authenticated.restype = c_int
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
    _lib.tc_cuda_is_active.argtypes = []
    _lib.tc_cuda_is_active.restype = c_int
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
    _lib.tc_diloco_async_poll.argtypes = [
        c_void_p, POINTER(c_int), POINTER(c_int), POINTER(c_uint64),
    ]
    _lib.tc_diloco_async_poll.restype = c_int
    _lib.tc_diloco_async_wait.argtypes = [c_void_p]
    _lib.tc_diloco_async_wait.restype = c_int
    _lib.tc_diloco_async_commit.argtypes = [c_void_p]
    _lib.tc_diloco_async_commit.restype = c_int
    _lib.tc_diloco_capability_query.argtypes = [
        c_void_p, c_uint32, POINTER(TCDiLoCoCapabilities), c_size_t,
    ]
    _lib.tc_diloco_capability_query.restype = c_int
    _lib.tc_diloco_state_set_epochs.argtypes = [c_void_p, c_uint64, c_uint64]
    _lib.tc_diloco_state_set_epochs.restype = c_int
    _lib.tc_diloco_state_get_epochs.argtypes = [
        c_void_p, POINTER(c_uint64), POINTER(c_uint64),
    ]
    _lib.tc_diloco_state_get_epochs.restype = c_int
    _lib.tc_diloco_state_size.argtypes = [c_void_p, c_uint32, POINTER(c_size_t)]
    _lib.tc_diloco_state_size.restype = c_int
    _lib.tc_diloco_state_serialize.argtypes = [
        c_void_p, c_uint32, c_void_p, c_size_t, POINTER(c_size_t),
    ]
    _lib.tc_diloco_state_serialize.restype = c_int
    _lib.tc_diloco_state_deserialize.argtypes = [
        c_void_p, c_uint32, c_void_p, c_size_t,
    ]
    _lib.tc_diloco_state_deserialize.restype = c_int
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

    # ---------------------------------------------------------------------
    # Phase 4 substrate ABI (metric / geodesic / holonomic / shard).
    # Raw fp32-pointer surfaces; the Pythonic wrappers below give NumPy
    # callers an array-in / array-out path that hides the ctypes plumbing.
    # ---------------------------------------------------------------------
    _f32p = POINTER(c_float)

    # tc_metric_fn callback type: void(*)(const float*, int, float*, void*).
    _TCMetricFn = ctypes.CFUNCTYPE(None, _f32p, c_int, _f32p, c_void_p)
    _lib._tc_metric_fn_t = _TCMetricFn
    _lib.tc_metric_apply.argtypes = [_TCMetricFn, c_void_p, _f32p, c_int, _f32p, _f32p]
    _lib.tc_metric_apply.restype = c_float
    _lib.tc_metric_inverse.argtypes = [_f32p, c_int, _f32p]
    _lib.tc_metric_inverse.restype = c_int
    _lib.tc_metric_inverse_apply.argtypes = [_TCMetricFn, c_void_p, _f32p, c_int, _f32p, _f32p]
    _lib.tc_metric_inverse_apply.restype = c_float
    _lib.tc_metric_christoffel.argtypes = [_TCMetricFn, c_void_p, _f32p, c_int, c_float, _f32p]
    _lib.tc_metric_christoffel.restype = c_int
    # Stock metrics — bind as raw symbols so we can hand them straight back
    # to apply/christoffel without paying for a Python callback round-trip.
    _lib.tc_metric_euclidean.argtypes = [_f32p, c_int, _f32p, c_void_p]
    _lib.tc_metric_euclidean.restype = None
    _lib.tc_metric_poincare.argtypes = [_f32p, c_int, _f32p, c_void_p]
    _lib.tc_metric_poincare.restype = None
    _lib.tc_metric_sphere_stereographic.argtypes = [_f32p, c_int, _f32p, c_void_p]
    _lib.tc_metric_sphere_stereographic.restype = None

    # Geodesic ODE solver.
    _lib.tc_geodesic_step.argtypes = [_TCMetricFn, c_void_p, c_int, c_float, c_float,
                                       _f32p, _f32p, _f32p, _f32p]
    _lib.tc_geodesic_step.restype = c_int
    _lib.tc_geodesic_integrate.argtypes = [_TCMetricFn, c_void_p, c_int, c_float, c_int, c_float,
                                            _f32p, _f32p, _f32p, _f32p]
    _lib.tc_geodesic_integrate.restype = c_int

    # Holonomic gates.
    _lib.tc_holonomic_compose_su2.argtypes = [_f32p, c_int32, _f32p]
    _lib.tc_holonomic_compose_su2.restype = c_int
    _lib.tc_holonomic_berry_phase.argtypes = [_f32p, _f32p]
    _lib.tc_holonomic_berry_phase.restype = None

    # Quantum attention (B3): Born-rule overlap + softmax + apply + entanglement.
    _lib.tc_quantum_attention_score.argtypes = [_f32p, _f32p, c_int]
    _lib.tc_quantum_attention_score.restype = c_float
    _lib.tc_quantum_attention_softmax.argtypes = [_f32p, _f32p, c_int, c_int, c_float]
    _lib.tc_quantum_attention_softmax.restype = None
    _lib.tc_quantum_attention_apply.argtypes = [_f32p, _f32p, _f32p, c_int, c_int, c_int]
    _lib.tc_quantum_attention_apply.restype = None
    _lib.tc_quantum_entanglement_entropy.argtypes = [_f32p, c_int, c_int]
    _lib.tc_quantum_entanglement_entropy.restype = c_float

    # Remote shard (owner-routed tensor sharding over the mesh transport).
    class _TCShardPlan(Structure):
        _fields_ = [("n_peers",        c_int32),
                    ("rows",           c_int32),
                    ("cols",           c_int32),
                    ("dtype",          c_int)]   # tc_coll_dtype_t
    _lib._tc_shard_plan_struct = _TCShardPlan
    _lib.tc_remote_shard_owner.argtypes = [POINTER(_TCShardPlan), c_int32]
    _lib.tc_remote_shard_owner.restype = c_int32
    _lib.tc_remote_shard_local_range.argtypes = [POINTER(_TCShardPlan), c_int32,
                                                   POINTER(c_int32), POINTER(c_int32)]
    _lib.tc_remote_shard_local_range.restype = None
    _lib.tc_remote_shard_register.argtypes = [c_void_p, POINTER(_TCShardPlan), c_char_p, c_void_p]
    _lib.tc_remote_shard_register.restype = c_int
    _lib.tc_remote_shard_get.argtypes = [c_void_p, POINTER(_TCShardPlan), c_char_p,
                                           c_int32, c_int32, c_void_p, c_void_p]
    _lib.tc_remote_shard_get.restype = c_int
    _lib.tc_remote_shard_publish_put.argtypes = [c_void_p, POINTER(_TCShardPlan), c_char_p,
                                                   c_int32, c_int32, c_void_p]
    _lib.tc_remote_shard_publish_put.restype = c_int
    _lib.tc_remote_shard_drain_puts.argtypes = [c_void_p, POINTER(_TCShardPlan), c_char_p,
                                                  c_void_p, POINTER(c_int32)]
    _lib.tc_remote_shard_drain_puts.restype = c_int

    # Mesh-collective group lifecycle (already shipped, but never bound here).
    _lib.tc_mesh_group_init.argtypes = [c_void_p, c_int32, c_int32, POINTER(c_char_p),
                                          POINTER(c_void_p)]
    _lib.tc_mesh_group_init.restype = c_int
    _lib.tc_mesh_group_init_authenticated.argtypes = [
        c_void_p, c_int32, c_int32, POINTER(c_char_p), POINTER(c_char_p),
        POINTER(TCTransportAuthConfig), POINTER(c_void_p)]
    _lib.tc_mesh_group_init_authenticated.restype = c_int
    _lib.tc_mesh_group_shutdown.argtypes = [c_void_p]
    _lib.tc_mesh_group_shutdown.restype = c_int
    _lib.tc_mesh_allreduce.argtypes = [c_void_p, c_void_p, c_size_t, c_int, c_int]
    _lib.tc_mesh_allreduce.restype = c_int
    _lib.tc_mesh_allreduce_tagged.argtypes = [
        c_void_p, c_uint64, c_void_p, c_size_t, c_int, c_int]
    _lib.tc_mesh_allreduce_tagged.restype = c_int
    _lib.tc_mesh_broadcast.argtypes = [c_void_p, c_void_p, c_size_t, c_int, c_int32]
    _lib.tc_mesh_broadcast.restype = c_int
    _lib.tc_mesh_broadcast_tagged.argtypes = [
        c_void_p, c_uint64, c_void_p, c_size_t, c_int, c_int32]
    _lib.tc_mesh_broadcast_tagged.restype = c_int
    _lib.tc_mesh_allgather.argtypes = [c_void_p, c_void_p, c_size_t, c_void_p, c_int]
    _lib.tc_mesh_allgather.restype = c_int
    _lib.tc_mesh_allgather_tagged.argtypes = [
        c_void_p, c_uint64, c_void_p, c_size_t, c_void_p, c_int]
    _lib.tc_mesh_allgather_tagged.restype = c_int
    _lib.tc_mesh_collective_release_tag.argtypes = [c_void_p, c_uint64]
    _lib.tc_mesh_collective_release_tag.restype = c_int
    _lib.tc_mesh_total_bytes.argtypes = [c_void_p]
    _lib.tc_mesh_total_bytes.restype = c_uint64
    _lib.tc_mesh_retained_snapshot_count.argtypes = [c_void_p]
    _lib.tc_mesh_retained_snapshot_count.restype = c_size_t
    _lib.tc_mesh_allreduce_algorithm.argtypes = [c_void_p]
    _lib.tc_mesh_allreduce_algorithm.restype = c_char_p


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


def _transport_auth_config(local_identity, active_key_id, keys):
    """Build a temporary ctypes auth config.

    ``keys`` contains ``(identity, key_id, secret_bytes)`` tuples or mappings
    with those fields. Native constructors deep-copy the resulting keyring,
    so the returned keepalive is needed only for the duration of the call.
    """
    entries = list(keys)
    if not entries:
        raise ValueError("transport auth keyring must not be empty")
    key_array = (TCTransportAuthKey * len(entries))()
    identities = []
    secret_buffers = []
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            identity = entry["identity"]
            key_id = entry["key_id"]
            secret = entry["secret"]
        else:
            identity, key_id, secret = entry
        identity_bytes = _bytes(identity)
        secret_bytes = bytes(secret)
        if len(secret_bytes) < 32:
            raise ValueError("transport auth secrets must contain at least 32 bytes")
        secret_buffer = (ctypes.c_ubyte * len(secret_bytes)).from_buffer_copy(secret_bytes)
        identities.append(identity_bytes)
        secret_buffers.append(secret_buffer)
        key_array[index] = TCTransportAuthKey(
            identity_bytes, c_uint64(int(key_id)),
            ctypes.cast(secret_buffer, c_void_p), c_size_t(len(secret_bytes)))
    local_identity_bytes = _bytes(local_identity)
    auth = TCTransportAuthConfig(
        c_uint32(TC_TRANSPORT_AUTH_ABI_VERSION_CURRENT),
        local_identity_bytes,
        c_uint64(int(active_key_id)),
        key_array,
        c_size_t(len(entries)),
    )
    return auth, (key_array, identities, secret_buffers, local_identity_bytes)


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


def runtime_capabilities(ctx, abi_version=TC_RUNTIME_CAPABILITIES_ABI_VERSION_CURRENT):
    """Return the versioned runtime capability record for ``ctx``.

    Unknown ABI versions raise ``TensorcoreError`` with
    ``TC_ERR_ABI_MISMATCH``. Use :func:`capability_available` instead of
    testing the available mask alone so unknown future bits fail closed.
    """
    capabilities = TCRuntimeCapabilities()
    _check(_lib.tc_runtime_capabilities_get(
        _as_handle(ctx),
        c_uint32(int(abi_version)),
        byref(capabilities),
        c_size_t(ctypes.sizeof(capabilities)),
    ))
    return capabilities


def capability_available(capabilities, capability):
    """Return true only when ``capability`` is both known and available."""
    bit = int(capability)
    return (
        bit != 0 and
        (int(capabilities.known_capability_mask) & bit) == bit and
        (int(capabilities.available_capability_mask) & bit) == bit
    )


def cuda_is_active():
    """Return whether CUDA is active under the current runtime policy."""
    return bool(_lib.tc_cuda_is_active())


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


def dist_init_authenticated(ctx, backend, world_size, rank, rendezvous_url,
                            rank_identities, local_identity, active_key_id, keys):
    """Create a mutually authenticated Gloo context with rank identity binding."""
    identities = [_bytes(identity) for identity in rank_identities]
    identity_array = (c_char_p * len(identities))(*identities)
    auth, keepalive = _transport_auth_config(local_identity, active_key_id, keys)
    dist = c_void_p()
    _check(_lib.tc_dist_init_authenticated(
        _as_handle(ctx), _dist_backend(backend), int(world_size), int(rank),
        _bytes(rendezvous_url), identity_array, c_size_t(len(identities)),
        byref(auth), byref(dist)))
    _ = keepalive
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
    """Run or launch the DiLoCo outer optimizer step."""
    _check(_lib.tc_diloco_apply_outer(_as_handle(diloco)))


def diloco_async_poll(diloco):
    """Return ``(state, worker_status, round_id)`` without blocking."""
    state = c_int(TC_DILOCO_ASYNC_IDLE)
    worker_status = c_int(TC_OK)
    round_id = c_uint64(0)
    _check(_lib.tc_diloco_async_poll(
        _as_handle(diloco), byref(state), byref(worker_status), byref(round_id),
    ))
    return int(state.value), int(worker_status.value), int(round_id.value)


def diloco_async_wait(diloco):
    """Wait for private async work; raise if the worker failed."""
    _check(_lib.tc_diloco_async_wait(_as_handle(diloco)))


def diloco_async_commit(diloco):
    """Commit a ready async round at the caller's parameter boundary."""
    _check(_lib.tc_diloco_async_commit(_as_handle(diloco)))


def diloco_capabilities(
        diloco=None, abi_version=TC_DILOCO_CAPABILITIES_ABI_VERSION_CURRENT):
    """Return build-level or context-refined DiLoCo capabilities.

    Pass ``None`` for the build-level contract. A bound :class:`DiLoCoContext`
    additionally reports which compression modes are sparse on its transport.
    """
    capabilities = TCDiLoCoCapabilities()
    handle = c_void_p() if diloco is None else _as_handle(diloco)
    _check(_lib.tc_diloco_capability_query(
        handle,
        c_uint32(int(abi_version)),
        byref(capabilities),
        c_size_t(ctypes.sizeof(capabilities)),
    ))
    return capabilities


def diloco_outer_optimizer_is_serializable(capabilities, optimizer):
    value = _diloco_outer_optimizer(optimizer)
    if value < 0 or value >= 64:
        return False
    bit = 1 << value
    return (
        (int(capabilities.outer_optimizer_supported_mask) & bit) == bit and
        (int(capabilities.outer_optimizer_serializable_mask) & bit) == bit
    )


def diloco_compress_is_serializable(capabilities, compress):
    value = _diloco_compress(compress)
    if value < 0 or value >= 64:
        return False
    bit = 1 << value
    return (
        (int(capabilities.compress_supported_mask) & bit) == bit and
        (int(capabilities.compress_serializable_mask) & bit) == bit
    )


def diloco_state_feature_available(capabilities, feature):
    bit = int(feature)
    return bit != 0 and (int(capabilities.state_feature_mask) & bit) == bit


def diloco_state_set_epochs(diloco, topology_epoch, membership_epoch):
    """Bind DiLoCo checkpoint state to authoritative topology epochs."""
    _check(_lib.tc_diloco_state_set_epochs(
        _as_handle(diloco), c_uint64(topology_epoch), c_uint64(membership_epoch),
    ))


def diloco_state_get_epochs(diloco):
    """Return ``(topology_epoch, membership_epoch)``."""
    topology_epoch = c_uint64(0)
    membership_epoch = c_uint64(0)
    _check(_lib.tc_diloco_state_get_epochs(
        _as_handle(diloco), byref(topology_epoch), byref(membership_epoch),
    ))
    return int(topology_epoch.value), int(membership_epoch.value)


def diloco_state_serialize(
        diloco, abi_version=TC_DILOCO_STATE_ABI_VERSION_CURRENT):
    """Return a versioned checkpoint blob for DiLoCo-owned state."""
    size = c_size_t(0)
    _check(_lib.tc_diloco_state_size(
        _as_handle(diloco), c_uint32(abi_version), byref(size),
    ))
    output = (ctypes.c_ubyte * size.value)()
    written = c_size_t(0)
    _check(_lib.tc_diloco_state_serialize(
        _as_handle(diloco), c_uint32(abi_version),
        ctypes.cast(output, c_void_p), size, byref(written),
    ))
    return bytes(output[:written.value])


def diloco_state_deserialize(
        diloco, state, abi_version=TC_DILOCO_STATE_ABI_VERSION_CURRENT):
    """Restore DiLoCo-owned state after caller-owned model buffers."""
    payload = bytes(state)
    if not payload:
        raise ValueError("DiLoCo state blob must not be empty")
    source = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
    _check(_lib.tc_diloco_state_deserialize(
        _as_handle(diloco), c_uint32(abi_version),
        ctypes.cast(source, c_void_p), c_size_t(len(payload)),
    ))


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


def sparse_24_prune(ctx, W, dtype, rows, cols):
    """Apply 2:4 sparsity mask in place: each 4-block keeps top-2 |·|, zeros others."""
    _check(_lib.tc_sparse_24_prune(_as_handle(ctx), _as_handle(W),
                                     int(dtype), int(rows), int(cols)))


def sparse_24_check(ctx, W, dtype, rows, cols):
    """Verify W satisfies the 2:4 pattern. Returns True if valid, False otherwise."""
    s = _lib.tc_sparse_24_check(_as_handle(ctx), _as_handle(W),
                                  int(dtype), int(rows), int(cols))
    return s == 0


def sparse_24_gemm(ctx, A, B, C, M, N, K, a_dtype, b_dtype, c_dtype,
                    alpha=1.0, beta=0.0):
    """C = alpha * A @ B + beta * C, where B must be 2:4-pruned (call
    sparse_24_prune first). Uses cusparseLt tensor cores on Ampere+ when
    available; dense fallback (still correct) elsewhere."""
    _check(_lib.tc_sparse_24_gemm(_as_handle(ctx), _as_handle(A), _as_handle(B),
                                    _as_handle(C), int(M), int(N), int(K),
                                    int(a_dtype), int(b_dtype), int(c_dtype),
                                    c_float(float(alpha)), c_float(float(beta))))


def sparse_24_available():
    """Return True if the host has cusparseLt + Ampere+ hardware (real 2× speedup)."""
    return bool(_lib.tc_sparse_24_available())
TC_REMOTE_ROLE_WEIGHT_SERVER  = 0
TC_REMOTE_ROLE_COMPUTE_CLIENT = 1


def remote_init(ctx, role, bind_url):
    """Initialize a tensorcore remote-tensor endpoint.
       role = TC_REMOTE_ROLE_WEIGHT_SERVER or TC_REMOTE_ROLE_COMPUTE_CLIENT.
       bind_url = "tcp://0.0.0.0:port" for server, None for client.
       Returns an opaque handle (c_void_p)."""
    h = c_void_p()
    bu = None if bind_url is None else bind_url.encode('utf-8')
    _check(_lib.tc_remote_init(_as_handle(ctx), int(role), bu, byref(h)))
    return h


def remote_init_authenticated(ctx, role, bind_url, local_identity,
                              active_key_id, keys):
    """Initialize a mutually authenticated remote-tensor endpoint."""
    auth, keepalive = _transport_auth_config(local_identity, active_key_id, keys)
    h = c_void_p()
    bu = None if bind_url is None else _bytes(bind_url)
    _check(_lib.tc_remote_init_authenticated(
        _as_handle(ctx), int(role), bu, byref(auth), byref(h)))
    _ = keepalive
    return h


def remote_shutdown(handle):
    _check(_lib.tc_remote_shutdown(handle))


def remote_register_tensor(handle, name, ptr, nbytes):
    _check(_lib.tc_remote_register_tensor(
        handle, name.encode('utf-8'), c_void_p(ptr), int(nbytes)))


def remote_unregister_tensor(handle, name):
    _check(_lib.tc_remote_unregister_tensor(handle, name.encode('utf-8')))


def remote_registered_count(handle):
    return int(_lib.tc_remote_registered_count(handle))


def remote_connect(handle, peer_url):
    """Connect a client to a server. Returns peer_id (int)."""
    pid = _lib.tc_remote_connect(handle, peer_url.encode('utf-8'))
    if pid < 0:
        raise TensorcoreError(-1)
    return int(pid)


def remote_connect_authenticated(handle, peer_url, expected_peer_identity):
    """Connect and mutually authenticate an explicitly named server."""
    peer_id = c_int(-1)
    _check(_lib.tc_remote_connect_authenticated(
        _as_handle(handle), _bytes(peer_url), _bytes(expected_peer_identity),
        byref(peer_id)))
    return int(peer_id.value)


def remote_auth_rotate(handle, local_identity, active_key_id, keys):
    """Atomically replace the keyring used by future remote handshakes."""
    auth, keepalive = _transport_auth_config(local_identity, active_key_id, keys)
    _check(_lib.tc_remote_auth_rotate(_as_handle(handle), byref(auth)))
    _ = keepalive


def remote_peer_identity(handle, peer_id):
    identity = _lib.tc_remote_peer_identity(_as_handle(handle), int(peer_id))
    return None if identity is None else identity.decode("utf-8")


def remote_peer_key_id(handle, peer_id):
    return int(_lib.tc_remote_peer_key_id(_as_handle(handle), int(peer_id)))


def remote_auth_failure_count(handle):
    return int(_lib.tc_remote_auth_failure_count(_as_handle(handle)))


def remote_tensor_fetch(handle, peer_id, name, dst_ptr, nbytes):
    """Synchronous fetch of bytes for `name` from `peer_id` into dst_ptr.
       dst_ptr is a raw host pointer (use ctypes to get .data_ptr() of
       a numpy/torch array). nbytes must match the server-registered size."""
    _check(_lib.tc_remote_tensor_fetch(
        handle, int(peer_id), name.encode('utf-8'),
        c_void_p(dst_ptr), int(nbytes)))


def remote_total_bytes_served(handle):
    return int(_lib.tc_remote_total_bytes_served(handle))

def remote_total_bytes_fetched(handle):
    return int(_lib.tc_remote_total_bytes_fetched(handle))

def remote_fetch_count(handle):
    return int(_lib.tc_remote_fetch_count(handle))



def riemannian_adam_step_poincare(ctx, params, grads, m, v, N, D, c,
                                    lr, beta1, beta2, eps, weight_decay,
                                    bias_correction1, bias_correction2):
    """One RiemannianAdam step on the Poincaré ball."""
    _check(_lib.tc_riemannian_adam_step_poincare(
        _as_handle(ctx), _as_handle(params), _as_handle(grads),
        _as_handle(m), _as_handle(v), int(N), int(D),
        c_float(float(c)), c_float(float(lr)), c_float(float(beta1)),
        c_float(float(beta2)), c_float(float(eps)), c_float(float(weight_decay)),
        c_float(float(bias_correction1)), c_float(float(bias_correction2))))


def riemannian_adam_step_sphere(ctx, params, grads, m, v, N, D,
                                 lr, beta1, beta2, eps, weight_decay,
                                 bias_correction1, bias_correction2):
    """One RiemannianAdam step on the unit sphere."""
    _check(_lib.tc_riemannian_adam_step_sphere(
        _as_handle(ctx), _as_handle(params), _as_handle(grads),
        _as_handle(m), _as_handle(v), int(N), int(D),
        c_float(float(lr)), c_float(float(beta1)), c_float(float(beta2)),
        c_float(float(eps)), c_float(float(weight_decay)),
        c_float(float(bias_correction1)), c_float(float(bias_correction2))))


def riemannian_adam_step_euclidean(ctx, params, grads, m, v, N, D,
                                    lr, beta1, beta2, eps, weight_decay,
                                    bias_correction1, bias_correction2):
    """RiemannianAdam Euclidean step (≡ plain AdamW)."""
    _check(_lib.tc_riemannian_adam_step_euclidean(
        _as_handle(ctx), _as_handle(params), _as_handle(grads),
        _as_handle(m), _as_handle(v), int(N), int(D),
        c_float(float(lr)), c_float(float(beta1)), c_float(float(beta2)),
        c_float(float(eps)), c_float(float(weight_decay)),
        c_float(float(bias_correction1)), c_float(float(bias_correction2))))


def riemannian_offmanifold_count():
    """Return the process-global total of Poincare entry/retraction clamps."""
    return int(_lib.tc_riemannian_offmanifold_count())


def riemannian_offmanifold_counts():
    """Return ``(entry_clamps, retraction_clamps)`` since the last reset."""
    entry = c_uint64(0)
    retract = c_uint64(0)
    _lib.tc_riemannian_offmanifold_counts(byref(entry), byref(retract))
    return int(entry.value), int(retract.value)


def riemannian_offmanifold_reset():
    """Reset both process-global off-manifold diagnostic counters."""
    _lib.tc_riemannian_offmanifold_reset()


def phase_attention_combine(ctx, inner_products, phase_diffs, distances,
                              weights, gammas, lambdas, scores_out,
                              N_pairs, M):
    """Phase-attention score combine: Σ_m w_m·ip_m·cos(Δφ_m+γ_m)·exp(−λ_m·d_m)."""
    _check(_lib.tc_phase_attention_combine(
        _as_handle(ctx), _as_handle(inner_products), _as_handle(phase_diffs),
        _as_handle(distances), _as_handle(weights), _as_handle(gammas),
        _as_handle(lambdas), _as_handle(scores_out), int(N_pairs), int(M)))


def born_rule_output(ctx, h_amp, s_amp, e_amp, probs_out, N, V):
    """Born-rule output: P(w) = |h+s+e|² / Z. s_amp/e_amp may be None to omit."""
    _check(_lib.tc_born_rule_output(
        _as_handle(ctx), _as_handle(h_amp),
        _as_handle(s_amp) if s_amp is not None else c_void_p(0),
        _as_handle(e_amp) if e_amp is not None else c_void_p(0),
        _as_handle(probs_out), int(N), int(V)))


def poincare_mobius_add(ctx, X, Y, out, c, N, D):
    """Möbius addition X ⊕_c Y on Poincaré ball, [N,D] fp32 in/out."""
    _check(_lib.tc_poincare_mobius_add(_as_handle(ctx), _as_handle(X),
                                        _as_handle(Y), _as_handle(out),
                                        c_float(float(c)), int(N), int(D)))


def poincare_distance(ctx, X, Y, dist_out, c, N, D):
    """Geodesic distance d_c(x,y). dist_out is [N] fp32."""
    _check(_lib.tc_poincare_distance(_as_handle(ctx), _as_handle(X),
                                       _as_handle(Y), _as_handle(dist_out),
                                       c_float(float(c)), int(N), int(D)))


def poincare_conformal_factor(ctx, X, lam_out, c, N, D):
    """Conformal factor λ_x = 2/(1−c‖x‖²). lam_out is [N] fp32."""
    _check(_lib.tc_poincare_conformal_factor(_as_handle(ctx), _as_handle(X),
                                              _as_handle(lam_out),
                                              c_float(float(c)), int(N), int(D)))


def poincare_exp_map_zero(ctx, V, out, c, N, D):
    """exp_0^c(v). [N,D] fp32 in/out."""
    _check(_lib.tc_poincare_exp_map_zero(_as_handle(ctx), _as_handle(V),
                                          _as_handle(out),
                                          c_float(float(c)), int(N), int(D)))


def poincare_log_map_zero(ctx, X, out, c, N, D):
    """log_0^c(x). [N,D] fp32 in/out."""
    _check(_lib.tc_poincare_log_map_zero(_as_handle(ctx), _as_handle(X),
                                          _as_handle(out),
                                          c_float(float(c)), int(N), int(D)))


def poincare_exp_map(ctx, X, V, out, c, N, D):
    """General exp map: exp_x^c(v)."""
    _check(_lib.tc_poincare_exp_map(_as_handle(ctx), _as_handle(X),
                                     _as_handle(V), _as_handle(out),
                                     c_float(float(c)), int(N), int(D)))


def poincare_log_map(ctx, X, Y, out, c, N, D):
    """General log map: log_x^c(y)."""
    _check(_lib.tc_poincare_log_map(_as_handle(ctx), _as_handle(X),
                                     _as_handle(Y), _as_handle(out),
                                     c_float(float(c)), int(N), int(D)))


def poincare_parallel_transport(ctx, V, X, Y, out, c, N, D):
    """Parallel transport of tangent v from x to y on the Poincaré ball."""
    _check(_lib.tc_poincare_parallel_transport(_as_handle(ctx), _as_handle(V),
                                                 _as_handle(X), _as_handle(Y),
                                                 _as_handle(out),
                                                 c_float(float(c)), int(N), int(D)))


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

    def runtime_capabilities(self, abi_version=TC_RUNTIME_CAPABILITIES_ABI_VERSION_CURRENT):
        return runtime_capabilities(self, abi_version)

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

    def async_poll(self):
        return diloco_async_poll(self)

    def async_wait(self):
        diloco_async_wait(self)
        return self

    def async_commit(self):
        diloco_async_commit(self)
        return self

    def capabilities(self, abi_version=TC_DILOCO_CAPABILITIES_ABI_VERSION_CURRENT):
        return diloco_capabilities(self, abi_version)

    def set_state_epochs(self, topology_epoch, membership_epoch):
        diloco_state_set_epochs(self, topology_epoch, membership_epoch)
        return self

    @property
    def state_epochs(self):
        return diloco_state_get_epochs(self)

    def serialize_state(self, abi_version=TC_DILOCO_STATE_ABI_VERSION_CURRENT):
        return diloco_state_serialize(self, abi_version)

    def deserialize_state(
            self, state, abi_version=TC_DILOCO_STATE_ABI_VERSION_CURRENT):
        diloco_state_deserialize(self, state, abi_version)
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


# ---------------------------------------------------------------------------
# Phase 5 NumPy-friendly substrate surface.
#
# These wrappers exist so sibling repos (qLLM, Noesis, moonlab, QGTL) can
# call the tensorcore math substrate from Python WITHOUT writing raw
# ctypes glue. Inputs are NumPy float32 arrays (or anything np.asarray
# can convert); outputs are fresh NumPy arrays. The wrappers handle
# shape, dtype coercion, and contiguous-buffer marshalling — the
# underlying C kernels are the same single-source-of-truth math.
#
# All functions are no-NumPy-import-cost: numpy is imported on first call.
# ---------------------------------------------------------------------------


def _np():
    import numpy as np  # noqa: WPS433
    return np


def _f32_buf(arr_or_seq):
    """Coerce to a contiguous float32 NumPy array (copy if needed) and
    return (array, void_pointer_into_it). The caller MUST keep the
    array alive while it uses the pointer."""
    np = _np()
    a = np.ascontiguousarray(np.asarray(arr_or_seq, dtype=np.float32))
    return a, a.ctypes.data_as(POINTER(c_float))


def _f32_out(shape):
    """Allocate a fresh contiguous float32 array of the given shape."""
    np = _np()
    return np.zeros(shape, dtype=np.float32)


# ---- Lorentz / hyperboloid ----

def lorentz_exp(base, tangent, curvature=1.0):
    """exp_base(tangent) on the Lorentz hyperboloid of curvature `c`."""
    base_a, base_p = _f32_buf(base)
    tan_a,  tan_p  = _f32_buf(tangent)
    n = base_a.size
    out = _f32_out(base_a.shape)
    _lib.tc_lorentz_exp(base_p, tan_p, out.ctypes.data_as(POINTER(c_float)),
                         c_size_t(n), c_float(float(curvature)))
    return out


def lorentz_log(base, point, curvature=1.0):
    """log_base(point) → tangent on the Lorentz hyperboloid."""
    base_a, base_p = _f32_buf(base)
    pt_a,   pt_p   = _f32_buf(point)
    n = base_a.size
    out = _f32_out(base_a.shape)
    _lib.tc_lorentz_log(base_p, pt_p, out.ctypes.data_as(POINTER(c_float)),
                         c_size_t(n), c_float(float(curvature)))
    return out


def lorentz_distance(p, q, curvature=1.0):
    """Geodesic distance on the Lorentz hyperboloid."""
    p_a, p_p = _f32_buf(p)
    q_a, q_p = _f32_buf(q)
    return float(_lib.tc_lorentz_distance(p_p, q_p, c_size_t(p_a.size),
                                            c_float(float(curvature))))


# ---- Sphere ----

def sphere_exp(base, tangent, radius=1.0):
    base_a, base_p = _f32_buf(base)
    tan_a,  tan_p  = _f32_buf(tangent)
    n = base_a.size
    out = _f32_out(base_a.shape)
    _lib.tc_sphere_exp(base_p, tan_p, out.ctypes.data_as(POINTER(c_float)),
                        c_size_t(n), c_float(float(radius)))
    return out


def sphere_log(base, point, radius=1.0):
    base_a, base_p = _f32_buf(base)
    pt_a,   pt_p   = _f32_buf(point)
    n = base_a.size
    out = _f32_out(base_a.shape)
    _lib.tc_sphere_log(base_p, pt_p, out.ctypes.data_as(POINTER(c_float)),
                        c_size_t(n), c_float(float(radius)))
    return out


def sphere_distance(p, q, radius=1.0):
    p_a, p_p = _f32_buf(p)
    q_a, q_p = _f32_buf(q)
    return float(_lib.tc_sphere_distance(p_p, q_p, c_size_t(p_a.size),
                                            c_float(float(radius))))


def sphere_slerp(p, q, t, radius=1.0):
    p_a, p_p = _f32_buf(p)
    q_a, q_p = _f32_buf(q)
    out = _f32_out(p_a.shape)
    _lib.tc_sphere_slerp(p_p, q_p, c_float(float(t)),
                          out.ctypes.data_as(POINTER(c_float)),
                          c_size_t(p_a.size), c_float(float(radius)))
    return out


# ---- Torus ----

def torus_exp(base, tangent, radius=1.0):
    base_a, base_p = _f32_buf(base)
    tan_a,  tan_p  = _f32_buf(tangent)
    out = _f32_out(base_a.shape)
    _lib.tc_torus_exp(base_p, tan_p, out.ctypes.data_as(POINTER(c_float)),
                       c_size_t(base_a.size), c_float(float(radius)))
    return out


def torus_log(base, point, radius=1.0):
    base_a, base_p = _f32_buf(base)
    pt_a,   pt_p   = _f32_buf(point)
    out = _f32_out(base_a.shape)
    _lib.tc_torus_log(base_p, pt_p, out.ctypes.data_as(POINTER(c_float)),
                       c_size_t(base_a.size), c_float(float(radius)))
    return out


def torus_distance(p, q, radius=1.0):
    p_a, p_p = _f32_buf(p)
    q_a, q_p = _f32_buf(q)
    return float(_lib.tc_torus_distance(p_p, q_p, c_size_t(p_a.size),
                                          c_float(float(radius))))


def torus_project(point, radius=1.0):
    """Wrap torus coordinates into the fundamental interval [0, 2πr)."""
    point_a, point_p = _f32_buf(point)
    _lib.tc_torus_project(point_p, c_size_t(point_a.size),
                           c_float(float(radius)))
    return point_a


def torus_parallel_transport(base_from, base_to, tangent, radius=1.0):
    """Parallel transport a tangent vector on the flat torus."""
    from_a, from_p = _f32_buf(base_from)
    to_a, to_p = _f32_buf(base_to)
    tangent_a, tangent_p = _f32_buf(tangent)
    if to_a.size != from_a.size or tangent_a.size != from_a.size:
        raise ValueError("torus points and tangent must have equal sizes")
    out = _f32_out(from_a.shape)
    _lib.tc_torus_parallel_transport(
        from_p, to_p, tangent_p, out.ctypes.data_as(POINTER(c_float)),
        c_size_t(from_a.size), c_float(float(radius)))
    return out


# ---- Lie groups (SU(2) / SO(3)) ----

def su2_exp(a, b, c):
    """exp(i (a σx + b σy + c σz)) → 2×2 SU(2) unitary as 8 floats
    (row-major interleaved complex)."""
    U = _f32_out((8,))
    _lib.tc_su2_exp(c_float(float(a)), c_float(float(b)), c_float(float(c)),
                     U.ctypes.data_as(POINTER(c_float)))
    return U


def su2_log(U):
    """log_{SU(2)}(U) → (a, b, c) algebra vector."""
    U_a, U_p = _f32_buf(U)
    a = c_float(0.0); b = c_float(0.0); cc = c_float(0.0)
    _lib.tc_su2_log(U_p, ctypes.byref(a), ctypes.byref(b), ctypes.byref(cc))
    return (a.value, b.value, cc.value)


def su2_mul(U, V):
    U_a, U_p = _f32_buf(U)
    V_a, V_p = _f32_buf(V)
    out = _f32_out((8,))
    _lib.tc_su2_mul(U_p, V_p, out.ctypes.data_as(POINTER(c_float)))
    return out


def so3_exp(wx, wy, wz):
    R = _f32_out((9,))
    _lib.tc_so3_exp(c_float(float(wx)), c_float(float(wy)), c_float(float(wz)),
                     R.ctypes.data_as(POINTER(c_float)))
    return R.reshape(3, 3)


def so3_log(R):
    R_a, R_p = _f32_buf(R)
    wx = c_float(0.0); wy = c_float(0.0); wz = c_float(0.0)
    _lib.tc_so3_log(R_p, ctypes.byref(wx), ctypes.byref(wy), ctypes.byref(wz))
    return (wx.value, wy.value, wz.value)


def so3_mul(R1, R2):
    """Compose two 3×3 SO(3) rotations."""
    R1_a, R1_p = _f32_buf(R1)
    R2_a, R2_p = _f32_buf(R2)
    if R1_a.size != 9 or R2_a.size != 9:
        raise ValueError("SO(3) matrices must contain 9 float values")
    out = _f32_out((9,))
    _lib.tc_so3_mul(R1_p, R2_p, out.ctypes.data_as(POINTER(c_float)))
    return out.reshape(3, 3)


def su2_to_so3(U):
    U_a, U_p = _f32_buf(U)
    R = _f32_out((9,))
    _lib.tc_su2_to_so3(U_p, R.ctypes.data_as(POINTER(c_float)))
    return R.reshape(3, 3)


# ---- Quantum gates / state-vector apply ----

def qstate_zero(n_qubits):
    """|0...0⟩ on n_qubits as a 2 * 2^n_qubits float32 buffer (interleaved
    complex)."""
    n = int(n_qubits)
    state = _f32_out((2 * (1 << n),))
    _lib.tc_qstate_zero(state.ctypes.data_as(POINTER(c_float)), c_int(n))
    return state


def qstate_apply_1q(state, n_qubits, target_qubit, gate_matrix):
    """In-place: apply a 1-qubit unitary (gate_matrix: 8 floats) to `state`
    at `target_qubit`. `state` must be the same buffer used elsewhere
    (this function mutates it). Returns `state` for chaining."""
    np = _np()
    s = np.ascontiguousarray(np.asarray(state, dtype=np.float32))
    g_a, g_p = _f32_buf(gate_matrix)
    _lib.tc_qstate_apply_1q_unitary(s.ctypes.data_as(POINTER(c_float)),
                                     c_int(int(n_qubits)),
                                     c_int(int(target_qubit)), g_p)
    return s


def qstate_apply_2q(state, n_qubits, qubit_a, qubit_b, gate_matrix):
    np = _np()
    s = np.ascontiguousarray(np.asarray(state, dtype=np.float32))
    g_a, g_p = _f32_buf(gate_matrix)
    _lib.tc_qstate_apply_2q_unitary(s.ctypes.data_as(POINTER(c_float)),
                                     c_int(int(n_qubits)),
                                     c_int(int(qubit_a)), c_int(int(qubit_b)),
                                     g_p)
    return s


def qstate_apply_3q(state, n_qubits, qubit_a, qubit_b, qubit_c, gate_matrix):
    np = _np()
    s = np.ascontiguousarray(np.asarray(state, dtype=np.float32))
    g_a, g_p = _f32_buf(gate_matrix)
    if g_a.size != 128:
        raise ValueError("a 3-qubit unitary must contain 128 float values")
    _lib.tc_qstate_apply_3q_unitary(
        s.ctypes.data_as(POINTER(c_float)), c_int(int(n_qubits)),
        c_int(int(qubit_a)), c_int(int(qubit_b)), c_int(int(qubit_c)), g_p)
    return s


def qstate_prob_one(state, n_qubits, qubit):
    s_a, s_p = _f32_buf(state)
    return float(_lib.tc_qstate_prob_one(s_p, c_int(int(n_qubits)),
                                            c_int(int(qubit))))


def qstate_norm_sq(state, n_qubits):
    s_a, s_p = _f32_buf(state)
    return float(_lib.tc_qstate_norm_sq(s_p, c_int(int(n_qubits))))


def qstate_inner(a, b, n_qubits):
    """Return the complex Hilbert-space inner product ⟨a|b⟩."""
    a_a, a_p = _f32_buf(a)
    b_a, b_p = _f32_buf(b)
    expected = 2 * (1 << int(n_qubits))
    if a_a.size != expected or b_a.size != expected:
        raise ValueError("state buffers do not match n_qubits")
    out_re = c_float(0.0)
    out_im = c_float(0.0)
    _lib.tc_qstate_inner(a_p, b_p, c_int(int(n_qubits)),
                          ctypes.byref(out_re), ctypes.byref(out_im))
    return complex(out_re.value, out_im.value)


def gate_matrix_1q(gate_type):
    """Materialise a 1-qubit gate by tc_gate_type_t enum value
    (X=1, Y=2, Z=3, H=4, S=5, T=6, RX=10, RY=11, RZ=12, ...
    — see include/tensorcore/quantum_gates.h for the full table).
    Returns 8 floats (2×2 row-major interleaved complex)."""
    out = _f32_out((8,))
    _lib.tc_gate_matrix_1q(c_int(int(gate_type)), None,
                            out.ctypes.data_as(POINTER(c_float)))
    return out


def gate_matrix_2q(gate_type, params=None):
    """Materialise a 2-qubit gate as 32 interleaved-complex floats."""
    params_a = None
    params_p = None
    if params is not None:
        params_a, params_p = _f32_buf(params)
    out = _f32_out((32,))
    _lib.tc_gate_matrix_2q(c_int(int(gate_type)), params_p,
                            out.ctypes.data_as(POINTER(c_float)))
    _ = params_a
    return out


def gate_matrix_3q(gate_type, params=None):
    """Materialise a 3-qubit gate as 128 interleaved-complex floats."""
    params_a = None
    params_p = None
    if params is not None:
        params_a, params_p = _f32_buf(params)
    out = _f32_out((128,))
    _lib.tc_gate_matrix_3q(c_int(int(gate_type)), params_p,
                            out.ctypes.data_as(POINTER(c_float)))
    _ = params_a
    return out


def quantum_geometric_tensor(state, derivatives, n_qubits):
    """Return the complex quantum geometric tensor for state derivatives."""
    np = _np()
    state_a, state_p = _f32_buf(state)
    expected = 2 * (1 << int(n_qubits))
    if state_a.size != expected:
        raise ValueError("state buffer does not match n_qubits")
    derivative_arrays = [
        np.ascontiguousarray(np.asarray(item, dtype=np.float32)).reshape(-1)
        for item in derivatives
    ]
    if any(item.size != expected for item in derivative_arrays):
        raise ValueError("derivative buffers do not match n_qubits")
    pointers = (POINTER(c_float) * len(derivative_arrays))(*[
        item.ctypes.data_as(POINTER(c_float)) for item in derivative_arrays
    ])
    out = _f32_out((len(derivative_arrays), len(derivative_arrays), 2))
    _lib.tc_quantum_geometric_tensor(
        state_p, pointers, c_int(int(n_qubits)), c_int(len(derivative_arrays)),
        out.ctypes.data_as(POINTER(c_float)))
    return out[..., 0] + 1j * out[..., 1]


# ---- Density matrices / open quantum systems ----

def dmstate_zero(n_qubits):
    """Return |0…0⟩⟨0…0| as an interleaved-complex density matrix."""
    n = int(n_qubits)
    dim = 1 << n
    rho = _f32_out((2 * dim * dim,))
    _lib.tc_dmstate_zero(rho.ctypes.data_as(POINTER(c_float)), c_int(n))
    return rho


def dmstate_from_pure(state, n_qubits):
    """Construct a density matrix ρ = |ψ⟩⟨ψ| from a pure state."""
    state_a, state_p = _f32_buf(state)
    n = int(n_qubits)
    if state_a.size != 2 * (1 << n):
        raise ValueError("state buffer does not match n_qubits")
    rho = _f32_out((2 * (1 << n) * (1 << n),))
    _lib.tc_dmstate_from_pure(rho.ctypes.data_as(POINTER(c_float)), state_p,
                               c_int(n))
    return rho


def dmstate_trace(rho, n_qubits):
    rho_a, rho_p = _f32_buf(rho)
    out_re = c_float(0.0)
    out_im = c_float(0.0)
    _lib.tc_dmstate_trace(rho_p, c_int(int(n_qubits)),
                           ctypes.byref(out_re), ctypes.byref(out_im))
    return complex(out_re.value, out_im.value)


def dmstate_purity(rho, n_qubits):
    rho_a, rho_p = _f32_buf(rho)
    return float(_lib.tc_dmstate_purity(rho_p, c_int(int(n_qubits))))


def dmstate_partial_trace(rho, n_qubits, trace_qubit):
    rho_a, rho_p = _f32_buf(rho)
    n = int(n_qubits)
    if n < 2:
        raise ValueError("partial trace requires at least two qubits")
    dim = 1 << (n - 1)
    out = _f32_out((2 * dim * dim,))
    _lib.tc_dmstate_partial_trace(
        rho_p, c_int(n), c_int(int(trace_qubit)),
        out.ctypes.data_as(POINTER(c_float)))
    return out


# ---- Phase 4: Riemannian metric tensor ----

def metric_apply(metric_fn, point, v, w, user_ptr=None):
    """g_{ij}(point) v^i w^j. `metric_fn` is either a stock symbol
    (e.g. metric_euclidean()) or a Python callable wrapped via
    metric_python_callback()."""
    p_a, p_p = _f32_buf(point)
    v_a, v_p = _f32_buf(v)
    w_a, w_p = _f32_buf(w)
    d = p_a.size
    user = c_void_p(0) if user_ptr is None else c_void_p(int(user_ptr))
    return float(_lib.tc_metric_apply(metric_fn, user, p_p, c_int(int(d)),
                                        v_p, w_p))


def metric_inverse(g):
    """Invert a d×d symmetric positive-definite matrix `g` (NumPy array,
    flattened or shaped (d, d)). Returns (g_inv, return_code); rc=0 OK."""
    np = _np()
    g_a = np.ascontiguousarray(np.asarray(g, dtype=np.float32))
    if g_a.ndim == 2:
        d = g_a.shape[0]
    else:
        d = int(round(g_a.size ** 0.5))
    out = _f32_out((d, d))
    rc = _lib.tc_metric_inverse(g_a.ctypes.data_as(POINTER(c_float)),
                                  c_int(d), out.ctypes.data_as(POINTER(c_float)))
    return out, int(rc)


def metric_christoffel(metric_fn, point, h=1e-3, user_ptr=None):
    """Numerical Christoffel symbols Γ^k_{ij}(point), shape (d, d, d)."""
    p_a, p_p = _f32_buf(point)
    d = p_a.size
    out = _f32_out((d, d, d))
    user = c_void_p(0) if user_ptr is None else c_void_p(int(user_ptr))
    rc = _lib.tc_metric_christoffel(metric_fn, user, p_p, c_int(d),
                                      c_float(float(h)),
                                      out.ctypes.data_as(POINTER(c_float)))
    if rc != 0:
        raise RuntimeError(f"tc_metric_christoffel returned {rc} "
                           f"(metric likely singular at point)")
    return out


def metric_euclidean():
    """Stock Euclidean metric callback (g_{ij} = δ_{ij})."""
    return ctypes.cast(_lib.tc_metric_euclidean, _lib._tc_metric_fn_t)


def metric_poincare(curvature):
    """Stock Poincaré-ball metric callback. Returns (fn, holder) — keep
    `holder` alive so its address (passed via user_ptr) stays valid."""
    c_val = c_float(float(curvature))
    fn = ctypes.cast(_lib.tc_metric_poincare, _lib._tc_metric_fn_t)
    return fn, c_val


def metric_sphere_stereographic(radius):
    """Stock stereographic-sphere metric callback. Returns (fn, holder)."""
    r_val = c_float(float(radius))
    fn = ctypes.cast(_lib.tc_metric_sphere_stereographic, _lib._tc_metric_fn_t)
    return fn, r_val


# ---- Phase 4: Geodesic ODE solver ----

def geodesic_integrate(metric_fn, pos0, vel0, dt, n_steps,
                       h_christoffel=1e-3, user_ptr=None):
    """Integrate γ̈ + Γγ̇γ̇ = 0 for n_steps of length dt starting from
    (pos0, vel0). Returns (pos_final, vel_final) as NumPy float32 arrays."""
    p_a, p_p = _f32_buf(pos0)
    v_a, v_p = _f32_buf(vel0)
    d = p_a.size
    pos_out = _f32_out(p_a.shape)
    vel_out = _f32_out(v_a.shape)
    user = c_void_p(0) if user_ptr is None else c_void_p(int(user_ptr))
    rc = _lib.tc_geodesic_integrate(metric_fn, user, c_int(d),
                                      c_float(float(dt)), c_int(int(n_steps)),
                                      c_float(float(h_christoffel)),
                                      p_p, v_p,
                                      pos_out.ctypes.data_as(POINTER(c_float)),
                                      vel_out.ctypes.data_as(POINTER(c_float)))
    if rc != 0:
        raise RuntimeError(f"tc_geodesic_integrate returned {rc} "
                           f"(metric singular at some substep)")
    return pos_out, vel_out


# ---- Phase 4: Holonomic gates ----

def holonomic_compose_su2(generators):
    """Compose a sequence of SU(2) generators (shape (N, 3) or flat 3N)
    into one loop unitary. Returns 8 floats (interleaved complex)."""
    np = _np()
    gens = np.ascontiguousarray(np.asarray(generators, dtype=np.float32).reshape(-1, 3))
    n_segs = gens.shape[0]
    U = _f32_out((8,))
    rc = _lib.tc_holonomic_compose_su2(gens.ctypes.data_as(POINTER(c_float)),
                                         c_int32(n_segs),
                                         U.ctypes.data_as(POINTER(c_float)))
    if rc != 0:
        raise RuntimeError(f"tc_holonomic_compose_su2 returned {rc}")
    return U


def holonomic_berry_phase(U):
    """Extract the SU(2) Berry phase (signed rotation angle) from a
    composed loop unitary. Returns a float in (-π, π]."""
    U_a, U_p = _f32_buf(U)
    out = c_float(0.0)
    _lib.tc_holonomic_berry_phase(U_p, ctypes.byref(out))
    return float(out.value)


# ---- Tier B3: Quantum attention + entanglement ----

def quantum_attention_score(state_q, state_k, n_qubits):
    """Born-rule overlap |⟨ψ_Q|ψ_K⟩|² between two n-qubit pure states.
    Both states MUST already be normalised."""
    q_a, q_p = _f32_buf(state_q)
    k_a, k_p = _f32_buf(state_k)
    return float(_lib.tc_quantum_attention_score(q_p, k_p, c_int(int(n_qubits))))


def quantum_attention_softmax(scores, n_q, n_k, temperature=1.0):
    """Per-row softmax over an [N_q, N_k] score matrix. Returns attn
    weights summing to 1 per row."""
    s_a, s_p = _f32_buf(scores)
    out = _f32_out((n_q, n_k))
    _lib.tc_quantum_attention_softmax(s_p,
                                        out.ctypes.data_as(POINTER(c_float)),
                                        c_int(int(n_q)), c_int(int(n_k)),
                                        c_float(float(temperature)))
    return out


def quantum_attention_apply(attn, values, n_q, n_k, d_v):
    """Weighted sum out[q, :] = Σ_k attn[q, k] · values[k, :]."""
    a_a, a_p = _f32_buf(attn)
    v_a, v_p = _f32_buf(values)
    out = _f32_out((n_q, d_v))
    _lib.tc_quantum_attention_apply(a_p, v_p,
                                      out.ctypes.data_as(POINTER(c_float)),
                                      c_int(int(n_q)), c_int(int(n_k)),
                                      c_int(int(d_v)))
    return out


def quantum_entanglement_entropy(state, n_qubits, qubit_keep):
    """Single-qubit von Neumann entropy S(ρ_q) for an n-qubit pure
    state, in bits. 0 for product, 1 for maximally entangled."""
    s_a, s_p = _f32_buf(state)
    return float(_lib.tc_quantum_entanglement_entropy(s_p,
                                                       c_int(int(n_qubits)),
                                                       c_int(int(qubit_keep))))


# ---- Phase 4: Remote shard ----

# tc_coll_dtype_t — mirror enum from include/tensorcore/mesh_collective.h
TC_COLL_DTYPE_F32 = 0


def shard_plan(n_peers, rows, cols, dtype=TC_COLL_DTYPE_F32):
    """Build a tc_shard_plan_t struct (passed to the shard ops)."""
    return _lib._tc_shard_plan_struct(int(n_peers), int(rows), int(cols),
                                       int(dtype))


def shard_owner(plan, row):
    return int(_lib.tc_remote_shard_owner(ctypes.byref(plan), c_int32(int(row))))


def shard_local_range(plan, my_rank):
    lo = c_int32(0); hi = c_int32(0)
    _lib.tc_remote_shard_local_range(ctypes.byref(plan), c_int32(int(my_rank)),
                                       ctypes.byref(lo), ctypes.byref(hi))
    return int(lo.value), int(hi.value)


def shard_register(group_handle, plan, name, my_buf):
    """Register the caller's local row-block on the mesh transport. The
    buffer is read zero-copy on the server side — caller MUST keep it
    alive (e.g. hold a reference) until the next register call or
    group shutdown."""
    np = _np()
    buf = np.ascontiguousarray(np.asarray(my_buf, dtype=np.float32))
    rc = _lib.tc_remote_shard_register(_as_handle(group_handle),
                                         ctypes.byref(plan),
                                         name.encode("utf-8") if isinstance(name, str) else name,
                                         buf.ctypes.data_as(c_void_p))
    if rc != 0:
        raise RuntimeError(f"tc_remote_shard_register returned {rc}")
    return buf  # caller holds the lifetime via the returned reference


def shard_publish_put(group_handle, plan, name, row_start, row_end, src):
    """Non-owner publishes a row-range push targeting whichever rank owns
    [row_start, row_end). All target rows MUST share one owner; for
    multi-owner ranges call publish_put once per owner-block."""
    np = _np()
    buf = np.ascontiguousarray(np.asarray(src, dtype=np.float32))
    rc = _lib.tc_remote_shard_publish_put(_as_handle(group_handle),
                                            ctypes.byref(plan),
                                            name.encode("utf-8") if isinstance(name, str) else name,
                                            c_int32(int(row_start)),
                                            c_int32(int(row_end)),
                                            buf.ctypes.data_as(c_void_p))
    if rc != 0:
        raise RuntimeError(f"tc_remote_shard_publish_put returned {rc}")
    return buf  # caller holds the lifetime


def shard_drain_puts(group_handle, plan, name, owner_mut_buf):
    """Owner polls every peer for pending puts, applies them into
    owner_mut_buf in place. Returns the number of puts applied this
    round (0 if no peers had pending puts)."""
    np = _np()
    buf = np.ascontiguousarray(np.asarray(owner_mut_buf, dtype=np.float32))
    applied = c_int32(0)
    rc = _lib.tc_remote_shard_drain_puts(_as_handle(group_handle),
                                           ctypes.byref(plan),
                                           name.encode("utf-8") if isinstance(name, str) else name,
                                           buf.ctypes.data_as(c_void_p),
                                           ctypes.byref(applied))
    if rc != 0:
        raise RuntimeError(f"tc_remote_shard_drain_puts returned {rc}")
    return int(applied.value)


def shard_get(group_handle, plan, name, row_start, row_end, local_buf=None):
    """Fetch rows [row_start, row_end) into a fresh NumPy array of shape
    (row_end - row_start, plan.cols). Provide `local_buf` for the
    local-fast-path optimisation; pass None to skip it (slightly slower
    for ranges that include locally-owned rows)."""
    np = _np()
    n_rows = int(row_end) - int(row_start)
    dst = np.zeros((n_rows, int(plan.cols)), dtype=np.float32)
    local_p = c_void_p(0)
    local_ref = None
    if local_buf is not None:
        local_ref = np.ascontiguousarray(np.asarray(local_buf, dtype=np.float32))
        local_p = local_ref.ctypes.data_as(c_void_p)
    rc = _lib.tc_remote_shard_get(_as_handle(group_handle),
                                    ctypes.byref(plan),
                                    name.encode("utf-8") if isinstance(name, str) else name,
                                    c_int32(int(row_start)), c_int32(int(row_end)),
                                    local_p, dst.ctypes.data_as(c_void_p))
    if rc != 0:
        raise RuntimeError(f"tc_remote_shard_get returned {rc}")
    return dst


# ---- Mesh group lifecycle ----

def mesh_group_init(ctx, n_peers, my_rank, peer_urls):
    """Initialise a mesh group bound to peer_urls[my_rank], connected
    to all other peer_urls. Returns the group handle (c_void_p)."""
    n = int(n_peers)
    url_array = (c_char_p * n)()
    for i, u in enumerate(peer_urls):
        url_array[i] = u.encode("utf-8") if isinstance(u, str) else u
    out = c_void_p()
    _check(_lib.tc_mesh_group_init(_as_handle(ctx), c_int32(n),
                                     c_int32(int(my_rank)), url_array,
                                     ctypes.byref(out)))
    return out


def mesh_group_init_authenticated(ctx, n_peers, my_rank, peer_urls,
                                  peer_identities, local_identity,
                                  active_key_id, keys):
    """Initialize an authenticated mesh with stable identity-per-rank binding."""
    n = int(n_peers)
    urls = [_bytes(value) for value in peer_urls]
    identities = [_bytes(value) for value in peer_identities]
    if len(urls) != n or len(identities) != n:
        raise ValueError("peer_urls and peer_identities must have n_peers entries")
    url_array = (c_char_p * n)(*urls)
    identity_array = (c_char_p * n)(*identities)
    auth, keepalive = _transport_auth_config(local_identity, active_key_id, keys)
    out = c_void_p()
    _check(_lib.tc_mesh_group_init_authenticated(
        _as_handle(ctx), c_int32(n), c_int32(int(my_rank)), url_array,
        identity_array, byref(auth), byref(out)))
    _ = keepalive
    return out


def mesh_group_shutdown(group_handle):
    _check(_lib.tc_mesh_group_shutdown(_as_handle(group_handle)))


def mesh_total_bytes(group_handle):
    return int(_lib.tc_mesh_total_bytes(_as_handle(group_handle)))


def mesh_retained_snapshot_count(group_handle):
    """Return the bounded transient snapshot count for a mesh group."""
    return int(_lib.tc_mesh_retained_snapshot_count(_as_handle(group_handle)))


def mesh_allreduce_algorithm(group_handle):
    """Return the selected mesh reduction topology."""
    return _lib.tc_mesh_allreduce_algorithm(_as_handle(group_handle)).decode()


def version():
    return _lib.tc_version().decode() if _lib else "(unloaded)"
