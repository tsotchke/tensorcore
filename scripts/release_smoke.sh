#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-"$ROOT/build"}"
PREFIX="${PREFIX:-/private/tmp/tensorcore-install}"
# Set TENSORCORE_RELEASE_SMOKE_EVIDENCE_PATH= to disable evidence output.
RELEASE_SMOKE_EVIDENCE_PATH="${TENSORCORE_RELEASE_SMOKE_EVIDENCE_PATH-"$BUILD_DIR/release_smoke_runtime_evidence.json"}"
if [ -z "${PY_PREFIX:-}" ]; then
    PY_PREFIX="$(mktemp -d /private/tmp/tensorcore-py-install.XXXXXX)"
fi
if [ -z "${WHEEL_DIR:-}" ]; then
    WHEEL_DIR="$(mktemp -d /private/tmp/tensorcore-wheels.XXXXXX)"
fi
if [ -z "${WHEEL_PREFIX:-}" ]; then
    WHEEL_PREFIX="$(mktemp -d /private/tmp/tensorcore-wheel-install.XXXXXX)"
fi
REQUIRE_GPU="${REQUIRE_GPU:-0}"
REQUIRE_METAL4_TENSOROPS="${REQUIRE_METAL4_TENSOROPS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_OK=0
CONSUMER_DIR=""
RELEASE_SMOKE_PHASE="init"
RELEASE_SMOKE_STATUS="running"
RELEASE_SMOKE_EXIT_STATUS=""
EXPECTED_VERSION="$("$PYTHON_BIN" - "$ROOT/pyproject.toml" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
if not match:
    raise SystemExit("project.version not found in pyproject.toml")
print(match.group(1))
PY
)"
export EXPECTED_VERSION

TC_SDK_VERSION="$(xcrun --show-sdk-version 2>/dev/null || true)"
if [ -z "$TC_SDK_VERSION" ]; then
    TC_SDK_VERSION="0.0"
fi
TC_SDK_SUPPORTS_METAL4="$("$PYTHON_BIN" - "$TC_SDK_VERSION" <<'PY'
import sys


def parse(version):
    parts = []
    for item in version.split("."):
        try:
            parts.append(int(item))
        except ValueError:
            parts.append(0)
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts[:2])


print("1" if parse(sys.argv[1]) >= (26, 0) else "0")
PY
)"

TESTS_STATUS="not_run"
TESTS_MODE="not_run"
WHEEL_TAG_STATUS="not_run"
WHEEL_PLATFORM_TAG=""
INSTALLED_WHEEL_SMOKE_STATUS="not_run"
INSTALLED_WHEEL_SMOKE_MODE="not_run"
CMAKE_CONSUMER_STATUS="not_run"
CMAKE_SHARED_CONSUMER_STATUS="not_run"
CMAKE_STATIC_CONSUMER_STATUS="not_run"
PKG_CONFIG_CONSUMER_STATUS="not_run"
AUTOTUNE_STATUS="not_run"
GEMM_128_TILE_STATUS="not_run"
GEMM_ASYNC_STATUS="not_run"
if [ "$TC_SDK_SUPPORTS_METAL4" = "1" ]; then
    METAL4_TENSOROPS_COMPILE_STATUS="pending_build"
    METAL4_TENSOROPS_RUNTIME_STATUS="skipped_no_m5"
    METAL4_TENSOROPS_REASON="SDK ${TC_SDK_VERSION} supports Metal 4 TensorOps sources; build has not completed yet"
else
    METAL4_TENSOROPS_COMPILE_STATUS="skipped_sdk_too_old"
    METAL4_TENSOROPS_RUNTIME_STATUS="skipped_not_compiled"
    METAL4_TENSOROPS_REASON="SDK ${TC_SDK_VERSION} is below the SDK 26.0 requirement for Metal 4 mpp::tensor_ops"
fi
METAL4_TENSOROPS_RUNTIME_COVERED="0"
METAL4_TENSOROPS_RUNTIME_OUTPUT=""
WHEEL_PATH=""

release_smoke_on_exit() {
    local status=$?
    if [ "$status" -ne 0 ]; then
        set +e
        RELEASE_SMOKE_STATUS="failed"
        RELEASE_SMOKE_EXIT_STATUS="$status"
        if [ "$TC_SDK_SUPPORTS_METAL4" = "1" ] &&
           [ "$RELEASE_SMOKE_PHASE" = "build" ] &&
           [ "$METAL4_TENSOROPS_COMPILE_STATUS" != "compiled" ]; then
            METAL4_TENSOROPS_COMPILE_STATUS="failed"
            METAL4_TENSOROPS_REASON="SDK ${TC_SDK_VERSION} supports Metal 4 TensorOps sources, but the build failed before compile evidence was proven"
        fi
        write_runtime_evidence >/dev/null 2>&1
    fi
    if [ -n "$CONSUMER_DIR" ]; then
        rm -rf "$CONSUMER_DIR"
    fi
}
trap release_smoke_on_exit EXIT

write_runtime_evidence() {
    if [ -z "$RELEASE_SMOKE_EVIDENCE_PATH" ]; then
        return
    fi

    mkdir -p "$(dirname "$RELEASE_SMOKE_EVIDENCE_PATH")"
    export ROOT BUILD_DIR PREFIX PY_PREFIX WHEEL_DIR WHEEL_PREFIX PYTHON_BIN EXPECTED_VERSION
    export GPU_OK TESTS_STATUS TESTS_MODE WHEEL_PATH WHEEL_TAG_STATUS WHEEL_PLATFORM_TAG
    export INSTALLED_WHEEL_SMOKE_STATUS INSTALLED_WHEEL_SMOKE_MODE
    export CMAKE_CONSUMER_STATUS CMAKE_SHARED_CONSUMER_STATUS CMAKE_STATIC_CONSUMER_STATUS
    export PKG_CONFIG_CONSUMER_STATUS AUTOTUNE_STATUS
    export GEMM_128_TILE_STATUS GEMM_ASYNC_STATUS
    export TC_SDK_VERSION METAL4_TENSOROPS_COMPILE_STATUS
    export METAL4_TENSOROPS_RUNTIME_STATUS METAL4_TENSOROPS_RUNTIME_COVERED
    export METAL4_TENSOROPS_REASON METAL4_TENSOROPS_RUNTIME_OUTPUT
    export RELEASE_SMOKE_PHASE RELEASE_SMOKE_STATUS RELEASE_SMOKE_EXIT_STATUS
    "$PYTHON_BIN" - "$RELEASE_SMOKE_EVIDENCE_PATH" <<'PY'
import datetime
import json
import os
import pathlib
import re
import sys


def env(name, default=""):
    return os.environ.get(name, default)


def passed(status):
    return status == "passed"


def optional_passed(status):
    if status.startswith("skipped"):
        return None
    return passed(status)


def function_line(root, rel_path, name):
    path = pathlib.Path(root) / rel_path
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return 1

    py_def = re.compile(rf"^\s*def\s+{re.escape(name)}\s*\(")
    if path.suffix == ".py":
        if "." in name:
            class_name, method_name = name.rsplit(".", 1)
            class_def = re.compile(rf"^(\s*)class\s+{re.escape(class_name)}\b")
            method_def = re.compile(rf"^\s+def\s+{re.escape(method_name)}\s*\(")
            in_class = False
            class_indent = 0
            for index, line in enumerate(lines, start=1):
                match = class_def.search(line)
                if match:
                    in_class = True
                    class_indent = len(match.group(1))
                    continue
                if not in_class:
                    continue
                if line.strip() and len(line) - len(line.lstrip()) <= class_indent:
                    in_class = False
                    continue
                if method_def.search(line):
                    return index
            return 1
        for index, line in enumerate(lines, start=1):
            if py_def.search(line):
                return index
        return 1

    objc_class = re.compile(rf"^\s*@(interface|implementation)\s+{re.escape(name)}\b")
    for index, line in enumerate(lines, start=1):
        if objc_class.search(line):
            return index

    c_ref = re.compile(rf"\b{re.escape(name)}\s*\(")
    control_prefixes = ("if", "for", "while", "switch", "return")
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if py_def.search(line):
            return index

        if not c_ref.search(line):
            continue
        prefix = stripped.split(name, 1)[0].strip()
        if "=" in prefix or prefix.startswith(control_prefixes):
            continue

        # C/C++ declarations often span several lines. Treat the occurrence as
        # a definition only if the signature opens a body before it terminates.
        for lookahead in lines[index - 1:min(len(lines), index + 12)]:
            if "{" in lookahead:
                return index
            if ";" in lookahead:
                break
    return 1


def add_function_call(files, root, rel_path, name):
    line = function_line(root, rel_path, name)
    entry = files.setdefault(rel_path, {"executed_lines": [], "functions": {}})
    if line not in entry["executed_lines"]:
        entry["executed_lines"].append(line)
    entry["functions"][name] = {
        "start_line": line,
        "executed_lines": [line],
    }


def add_function_calls(files, root, rel_path, names):
    for name in names:
        add_function_call(files, root, rel_path, name)


evidence_path = pathlib.Path(sys.argv[1])
root = env("ROOT")
wheel_path = env("WHEEL_PATH")
wheel_name = pathlib.Path(wheel_path).name if wheel_path else ""
platform_tag = env("WHEEL_PLATFORM_TAG")
if not platform_tag and wheel_name.endswith(".whl") and "-" in wheel_name:
    platform_tag = wheel_name[:-4].split("-")[-1]

checks = {
    "tests": {
        "status": env("TESTS_STATUS"),
        "passed": passed(env("TESTS_STATUS")),
        "mode": env("TESTS_MODE"),
        "gpu_device_available": env("GPU_OK") == "1",
    },
    "wheel_tag": {
        "status": env("WHEEL_TAG_STATUS"),
        "inspected": passed(env("WHEEL_TAG_STATUS")),
        "wheel_path": wheel_path,
        "wheel_name": wheel_name,
        "platform_tag": platform_tag,
    },
    "installed_wheel_smoke": {
        "status": env("INSTALLED_WHEEL_SMOKE_STATUS"),
        "passed": passed(env("INSTALLED_WHEEL_SMOKE_STATUS")),
        "mode": env("INSTALLED_WHEEL_SMOKE_MODE"),
    },
    "consumers": {
        "cmake": {
            "status": env("CMAKE_CONSUMER_STATUS"),
            "passed": passed(env("CMAKE_CONSUMER_STATUS")),
            "shared_consumer_status": env("CMAKE_SHARED_CONSUMER_STATUS"),
            "static_consumer_status": env("CMAKE_STATIC_CONSUMER_STATUS"),
        },
        "pkg_config": {
            "status": env("PKG_CONFIG_CONSUMER_STATUS"),
            "passed": optional_passed(env("PKG_CONFIG_CONSUMER_STATUS")),
        },
    },
    "autotune_cache": {
        "status": env("AUTOTUNE_STATUS"),
        "passed": passed(env("AUTOTUNE_STATUS")),
    },
    "gemm_env_variants": {
        "use_128_tile": {
            "status": env("GEMM_128_TILE_STATUS"),
            "passed": passed(env("GEMM_128_TILE_STATUS")),
        },
        "use_async": {
            "status": env("GEMM_ASYNC_STATUS"),
            "passed": passed(env("GEMM_ASYNC_STATUS")),
        },
    },
    "metal4_tensorops": {
        "sdk_version": env("TC_SDK_VERSION"),
        "compile_status": env("METAL4_TENSOROPS_COMPILE_STATUS"),
        "runtime_compile_status": env("METAL4_TENSOROPS_COMPILE_STATUS"),
        "runtime_status": env("METAL4_TENSOROPS_RUNTIME_STATUS"),
        "runtime_covered": env("METAL4_TENSOROPS_RUNTIME_COVERED") == "1",
        "reason": env("METAL4_TENSOROPS_REASON"),
        "runtime_output": env("METAL4_TENSOROPS_RUNTIME_OUTPUT"),
    },
}

coverage_files = {}
native_full_tests = checks["tests"]["passed"] and checks["tests"]["mode"] == "full"
python_smoke = (
    checks["installed_wheel_smoke"]["passed"] and
    checks["installed_wheel_smoke"]["mode"] == "python_tests"
)
wheel_import_smoke = checks["installed_wheel_smoke"]["passed"]
cmake_consumer_smoke = checks["consumers"]["cmake"]["passed"]
pkg_config_consumer_smoke = checks["consumers"]["pkg_config"]["passed"] is True
packaging_consumer_smoke = (
    wheel_import_smoke and
    checks["wheel_tag"]["inspected"] and
    cmake_consumer_smoke and
    pkg_config_consumer_smoke
)
autotune_cache_smoke = checks["autotune_cache"]["passed"]
gemm_128_tile_smoke = checks["gemm_env_variants"]["use_128_tile"]["passed"]
gemm_async_smoke = checks["gemm_env_variants"]["use_async"]["passed"]
public_integration_smoke = (
    native_full_tests and
    packaging_consumer_smoke and
    autotune_cache_smoke and
    gemm_128_tile_smoke and
    gemm_async_smoke
)
checks["packaging_and_consumers"] = {
    "runtime_status": "passed" if packaging_consumer_smoke else "failed",
    "runtime_covered": packaging_consumer_smoke,
}
checks["public_integration"] = {
    "runtime_status": (
        "passed" if public_integration_smoke else
        "skipped_no_gpu" if not checks["tests"]["gpu_device_available"] else
        "failed"
    ),
    "runtime_covered": public_integration_smoke,
}

if native_full_tests:
    native_coverage = {
        "lib/core/status.c": [
            "tc_status_string",
        ],
        "lib/core/device.mm": [
            "load_metallib",
            "tc_init",
            "tc_shutdown",
            "tc_device_family_from_mtl",
            "tc_last_backend",
            "tc_backend_name",
            "tc_version",
            "tc_device_info_get",
        ],
        "lib/core/buffer_pool.mm": [
            "TCBufferPool",
            "bucket_for",
            "bytes_for_bucket",
            "tc_buffer_pool_create",
            "tc_buffer_pool_destroy",
            "tc_buffer_pool_alloc",
            "tc_buffer_pool_free",
        ],
        "lib/core/pipeline_cache.mm": [
            "TCPipelineCache",
            "tc_pipeline_cache_create",
            "tc_pipeline_cache_destroy",
            "tc_pipeline_get",
        ],
        "include/tensorcore/dtype.h": [
            "tc_dtype_size",
        ],
        "lib/ops/gemm.mm": [
            "checked_mul",
            "checked_add",
            "validate",
            "matrix_bytes",
            "validate_gemm_buffers",
            "batched_matrix_bytes",
            "kernel_for",
            "resolve_pipeline",
            "tc_gemm",
            "tc_gemm_async",
            "tc_gemm_batched",
        ],
        "lib/fallback/mps_gemm.mm": [
            "to_mps_dtype",
            "bf16_to_f32",
            "f32_to_bf16",
            "bf16_via_fp32",
            "i8_via_fp32",
            "tc_mps_gemm",
        ],
        "lib/fallback/accelerate_gemm.c": [
            "tc_accelerate_gemm_f32",
        ],
        "lib/ops/attention.mm": [
            "checked_mul",
            "attention_tensor_bytes",
            "lse_tensor_bytes",
            "common_attention_shape",
            "kernel_name_for",
            "resolve_pipeline",
            "make_forward_plan",
            "encode_forward",
            "validate_backward_buffers",
            "tc_attention_forward",
            "tc_attention_forward_async",
            "tc_attention_backward",
        ],
        "lib/ops/conv.mm": [
            "conv_dims_valid",
            "conv_bytes",
            "validate_conv_common",
            "tc_conv2d_forward",
            "tc_conv2d_backward_input",
            "tc_conv2d_backward_weight",
        ],
        "lib/ops/quantized.mm": [
            "checked_mul",
            "fp16_matrix_bytes",
            "validate_quantize_buffers",
            "tc_quantized_size",
            "validate_gemv_quantized_buffers",
            "tc_quantize_weights",
            "gemv_quant_encode",
            "tc_gemv_quantized",
            "tc_gemv_quantized_async",
        ],
        "lib/ops/training.mm": [
            "pso_for",
            "threads_for_d",
            "tc_rmsnorm_forward",
            "tc_rmsnorm_backward",
            "tc_layernorm_forward",
            "tc_layernorm_backward",
            "tc_rope_forward",
            "tc_swiglu_forward",
            "tc_swiglu_backward",
            "tc_softmax_forward",
            "tc_softmax_backward",
            "tc_fused_rmsnorm_gemv",
            "tc_adamw_step",
        ],
        "lib/tensorops/tensorops_select.c": [
            "tc_tensorops_gemm_kernel_name",
        ],
        "lib/distributed/ring_local.mm": [
            "sock_send_all",
            "sock_recv_all",
            "tc_dist_ring_pair_make",
            "tc_dist_ring_local_allreduce_ex",
        ],
        "lib/io/gguf.c": [
            "rd_bytes",
            "rd_u32",
            "rd_u64",
            "rd_str",
            "rd_str_dup_n",
            "rd_str_dup",
            "gguf_scalar_size",
            "rd_value",
            "map_ggml_type",
            "type_bytes",
            "tc_gguf_open",
            "tc_gguf_close",
            "tc_gguf_tensor_count",
            "tc_gguf_metadata_count",
            "tc_gguf_get_tensor",
            "tc_gguf_tensor_at",
            "tc_gguf_meta_get_str",
            "tc_gguf_meta_get_i64",
            "tc_gguf_meta_get_f64",
            "tc_gguf_meta_array_count",
            "tc_gguf_meta_array_get_str",
            "tc_gguf_meta_array_get_i64",
            "tc_gguf_meta_array_get_f64",
            "find_kv",
            "tc_gguf_get_llama_config",
            "tc_gguf_tensor_to_buffer",
            "loaded_tensor_to_info",
            "tc_gguf_tensor_info_to_buffer",
            "gguf_type_to_quant",
            "gguf_quantized_matrix_info_common",
            "tc_gguf_tensor_quantized_matrix_info",
            "tc_gguf_loaded_tensor_quantized_matrix_info",
            "tc_gguf_load_supported_tensors",
            "tc_gguf_loaded_model_free",
            "tc_gguf_loaded_tensor_count",
            "tc_gguf_loaded_skipped_tensor_count",
            "tc_gguf_loaded_tensor_at",
            "tc_gguf_loaded_get_tensor",
            "scalar_at_i64",
            "scalar_at_f64",
        ],
    }
    for rel_path, names in native_coverage.items():
        add_function_calls(coverage_files, root, rel_path, names)

if wheel_import_smoke:
    add_function_calls(coverage_files, root, "python/tensorcore/__init__.py", [
        "_find_lib",
        "version",
    ])

if python_smoke:
    add_function_calls(coverage_files, root, "python/tensorcore/__init__.py", [
        "init",
        "shutdown",
        "device_info",
        "buffer_alloc",
        "buffer_free",
        "buffer_map",
        "buffer_size",
        "stream_create",
        "stream_sync",
        "stream_destroy",
        "buffer_write",
        "buffer_read",
        "_check",
        "_as_handle",
        "_bytes",
        "_dtype",
        "_quant",
        "gemm",
        "gemm_async",
        "gemm_batched",
        "_gemm_desc",
        "_attention_desc",
        "attention_forward",
        "attention_forward_async",
        "attention_backward",
        "conv2d_output_shape",
        "conv2d_scratch_bytes",
        "conv2d_backward_input_scratch_bytes",
        "conv2d_forward",
        "conv2d_backward_input",
        "conv2d_backward_weight",
        "quantized_size",
        "quantize_weights",
        "gemv_quantized",
        "gemv_quantized_async",
        "rmsnorm_forward",
        "rmsnorm_backward",
        "layernorm_forward",
        "layernorm_backward",
        "swiglu_forward",
        "swiglu_backward",
        "rope_forward",
        "softmax_forward",
        "softmax_backward",
        "fused_rmsnorm_gemv",
        "adamw_step",
        "gguf_open",
        "gguf_close",
        "gguf_tensor_count",
        "gguf_metadata_count",
        "gguf_meta_get_str",
        "gguf_meta_get_i64",
        "gguf_meta_get_f64",
        "gguf_meta_array_count",
        "gguf_meta_array_get_str",
        "gguf_meta_array_get_i64",
        "gguf_meta_array_get_f64",
        "gguf_get_tensor",
        "gguf_tensor_at",
        "gguf_tensor_to_buffer",
        "gguf_tensor_quantized_matrix_info",
        "gguf_loaded_tensor_quantized_matrix_info",
        "gguf_load_supported_tensors",
        "gguf_loaded_model_free",
        "gguf_loaded_tensor_count",
        "gguf_loaded_skipped_tensor_count",
        "gguf_loaded_tensor_at",
        "gguf_loaded_get_tensor",
        "gguf_get_llama_config",
        "_tensor_info_dict",
        "_tensor_info_from_dict",
        "_loaded_tensor_info_dict",
        "_loaded_tensor_info_from_dict",
        "_quantized_matrix_info_dict",
        "_llama_config_dict",
        "TensorcoreError.__init__",
        "Context.__init__",
        "Context.__enter__",
        "Context.__exit__",
        "Context._remember_buffer",
        "Context._forget_buffer",
        "Context._remember_stream",
        "Context._forget_stream",
        "Context._remember_loaded_model",
        "Context._forget_loaded_model",
        "Context.close",
        "Context.device_info",
        "Context.buffer",
        "Context.buffer_from_array",
        "Context.stream",
        "Context.gemm",
        "Context.gemm_async",
        "Context.gemm_batched",
        "Context.attention_forward",
        "Context.attention_forward_async",
        "Context.attention_backward",
        "Context.conv2d_forward",
        "Context.conv2d_backward_input",
        "Context.conv2d_backward_weight",
        "Context.quantize_weights",
        "Context.gemv_quantized",
        "Context.gemv_quantized_async",
        "Context.rmsnorm_forward",
        "Context.rmsnorm_backward",
        "Context.layernorm_forward",
        "Context.layernorm_backward",
        "Context.rope_forward",
        "Context.swiglu_forward",
        "Context.swiglu_backward",
        "Context.softmax_forward",
        "Context.softmax_backward",
        "Context.adamw_step",
        "Context.fused_rmsnorm_gemv",
        "Context.open_gguf",
        "Context.load_supported_tensors",
        "Buffer.__init__",
        "Buffer.__enter__",
        "Buffer.__exit__",
        "Buffer.close",
        "Buffer.map",
        "Buffer.size",
        "Buffer.nbytes",
        "Buffer.write",
        "Buffer.read",
        "Buffer.to_numpy",
        "Stream.__init__",
        "Stream.__enter__",
        "Stream.__exit__",
        "Stream.sync",
        "Stream.close",
        "GgufFile.__init__",
        "GgufFile.__enter__",
        "GgufFile.__exit__",
        "GgufFile.close",
        "GgufFile.tensor_count",
        "GgufFile.metadata_count",
        "GgufFile.get_tensor",
        "GgufFile.tensor_at",
        "GgufFile.meta_get_str",
        "GgufFile.meta_get_i64",
        "GgufFile.meta_get_f64",
        "GgufFile.meta_array_count",
        "GgufFile.meta_array_get_str",
        "GgufFile.llama_config",
        "GgufFile.tensor_to_buffer",
        "GgufFile.load_supported_tensors",
        "LoadedModel.__init__",
        "LoadedModel.__enter__",
        "LoadedModel.__exit__",
        "LoadedModel.close",
        "LoadedModel.tensor_count",
        "LoadedModel.skipped_tensor_count",
        "LoadedModel.tensor_at",
        "LoadedModel.get_tensor",
        "LoadedModel.quantized_matrix",
        "LoadedTensor.__init__",
        "LoadedTensor._check_alive",
        "LoadedTensor.__getitem__",
        "LoadedTensor.get",
        "LoadedTensor.buffer",
        "QuantizedMatrix.__init__",
        "QuantizedMatrix._check_alive",
        "QuantizedMatrix.output",
        "QuantizedMatrix.gemv",
        "QuantizedMatrix.gemv_async",
    ])

if autotune_cache_smoke:
    add_function_calls(coverage_files, root, "lib/core/autotune.cpp", [
        "tc_autotune_cache_dir",
        "tc_autotune_load_cache",
        "tc_autotune_run_sweep",
        "tc_autotune_save_cache",
    ])

if gemm_128_tile_smoke:
    add_function_calls(coverage_files, root, "lib/ops/gemm.mm", [
        "use_128_tile",
    ])

if gemm_async_smoke:
    add_function_calls(coverage_files, root, "lib/ops/gemm.mm", [
        "use_async_kernel",
    ])

if cmake_consumer_smoke or pkg_config_consumer_smoke:
    add_function_calls(coverage_files, root, "lib/core/device.mm", [
        "tc_version",
    ])
    add_function_calls(coverage_files, root, "include/tensorcore/dtype.h", [
        "tc_dtype_size",
    ])
    add_function_calls(coverage_files, root, "lib/core/dtype.c", [
        "tc_dtype_name",
    ])
    add_function_calls(coverage_files, root, "lib/ops/quantized.mm", [
        "tc_quantized_size",
    ])
    add_function_calls(coverage_files, root, "lib/io/gguf.c", [
        "gguf_quantized_matrix_info_common",
        "tc_gguf_tensor_quantized_matrix_info",
    ])

if cmake_consumer_smoke:
    add_function_calls(coverage_files, root, "lib/core/status.c", [
        "tc_status_string",
    ])

for entry in coverage_files.values():
    entry["executed_lines"].sort()

public_core_required_files = [
    "lib/core/device.mm",
    "lib/ops/gemm.mm",
    "lib/ops/attention.mm",
    "lib/ops/conv.mm",
    "lib/ops/training.mm",
    "lib/ops/quantized.mm",
    "lib/io/gguf.c",
    "lib/tensorops/tensorops_select.c",
    "python/tensorcore/__init__.py",
]
public_core_missing = sorted(
    rel_path for rel_path in public_core_required_files
    if rel_path not in coverage_files
)
public_core_covered = checks["tests"]["gpu_device_available"] and not public_core_missing
checks["public_core_paths"] = {
    "runtime_status": (
        "passed" if public_core_covered else
        "skipped_no_gpu" if not checks["tests"]["gpu_device_available"] else
        "failed"
    ),
    "runtime_covered": public_core_covered,
    "required_files": public_core_required_files,
    "missing_files": public_core_missing,
}

artifact = {
    "schema": "tensorcore.release_smoke.runtime_evidence.v1",
    "meta": {
        "format": 3,
        "source": "tensorcore_release_smoke",
    },
    "files": coverage_files,
    "status": env("RELEASE_SMOKE_STATUS", "passed"),
    "generated_at": datetime.datetime.now(datetime.timezone.utc)
    .isoformat()
    .replace("+00:00", "Z"),
    "run": {
        "phase": env("RELEASE_SMOKE_PHASE"),
        "exit_status": env("RELEASE_SMOKE_EXIT_STATUS"),
    },
    "project": {
        "root": env("ROOT"),
        "version": env("EXPECTED_VERSION"),
    },
    "paths": {
        "build_dir": env("BUILD_DIR"),
        "prefix": env("PREFIX"),
        "python_prefix": env("PY_PREFIX"),
        "wheel_dir": env("WHEEL_DIR"),
        "wheel_prefix": env("WHEEL_PREFIX"),
    },
    "python": {
        "executable": env("PYTHON_BIN"),
        "version": sys.version.split()[0],
    },
    "summary": {
        "tests_passed": checks["tests"]["passed"],
        "wheel_tag_inspected": checks["wheel_tag"]["inspected"],
        "installed_wheel_smoke_passed": checks["installed_wheel_smoke"]["passed"],
        "cmake_consumers_passed": checks["consumers"]["cmake"]["passed"],
        "pkg_config_consumer_passed": checks["consumers"]["pkg_config"]["passed"],
        "packaging_and_consumers_passed": checks["packaging_and_consumers"]["runtime_covered"],
        "public_core_paths_passed": checks["public_core_paths"]["runtime_covered"],
        "autotune_cache_passed": checks["autotune_cache"]["passed"],
        "gemm_128_tile_passed": checks["gemm_env_variants"]["use_128_tile"]["passed"],
        "gemm_async_passed": checks["gemm_env_variants"]["use_async"]["passed"],
        "public_integration_runtime_passed": checks["public_integration"]["runtime_covered"],
        "metal4_tensorops_compile_passed": checks["metal4_tensorops"]["compile_status"] == "compiled",
        "metal4_tensorops_runtime_passed": checks["metal4_tensorops"]["runtime_status"] == "passed",
    },
    "checks": checks,
}

tmp_path = evidence_path.with_name(f".{evidence_path.name}.tmp")
tmp_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
tmp_path.replace(evidence_path)
PY
}

echo "[tensorcore] configure"
RELEASE_SMOKE_PHASE="configure"
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

echo "[tensorcore] build"
RELEASE_SMOKE_PHASE="build"
cmake --build "$BUILD_DIR"
if [ "$TC_SDK_SUPPORTS_METAL4" = "1" ]; then
    METAL4_TENSOROPS_COMPILE_STATUS="compiled"
    METAL4_TENSOROPS_REASON="SDK ${TC_SDK_VERSION} compiled Metal 4 TensorOps sources; runtime probe not covered on this host"
fi

echo "[tensorcore] public export surface"
RELEASE_SMOKE_PHASE="public_exports"
BUILD_DIR="$BUILD_DIR" "$ROOT/scripts/check_public_exports.sh"

echo "[tensorcore] test"
RELEASE_SMOKE_PHASE="test"
if "$BUILD_DIR/tests/test_device"; then
    GPU_OK=1
    ctest --test-dir "$BUILD_DIR" --output-on-failure
    echo "[tensorcore] GEMM env variants"
    cmake -E env TC_METALLIB="$BUILD_DIR/tensorcore.metallib" TC_USE_128_TILE=1 \
        "$BUILD_DIR/tests/test_gemm_f16"
    cmake -E env TC_METALLIB="$BUILD_DIR/tensorcore.metallib" TC_USE_128_TILE=1 \
        "$BUILD_DIR/tests/test_gemm_f32"
    GEMM_128_TILE_STATUS="passed"
    cmake -E env TC_METALLIB="$BUILD_DIR/tensorcore.metallib" TC_USE_ASYNC=1 \
        "$BUILD_DIR/tests/test_gemm_f16"
    GEMM_ASYNC_STATUS="passed"
    echo "[tensorcore] Metal 4 TensorOps runtime probe"
    set +e
    TENSOROPS_RUNTIME_OUTPUT="$(
        cmake -E env TC_METALLIB="$BUILD_DIR/tensorcore.metallib" \
            "$BUILD_DIR/tests/test_tensorops_runtime" 2>&1
    )"
    TENSOROPS_RUNTIME_RC=$?
    set -e
    printf "%s\n" "$TENSOROPS_RUNTIME_OUTPUT"
    METAL4_TENSOROPS_RUNTIME_OUTPUT="$TENSOROPS_RUNTIME_OUTPUT"
    if [ "$TENSOROPS_RUNTIME_RC" -ne 0 ]; then
        METAL4_TENSOROPS_RUNTIME_STATUS="failed"
        METAL4_TENSOROPS_RUNTIME_COVERED="0"
        METAL4_TENSOROPS_REASON="TensorOps runtime probe failed with exit ${TENSOROPS_RUNTIME_RC}"
        TESTS_STATUS="failed"
        write_runtime_evidence
        exit 1
    elif printf "%s\n" "$TENSOROPS_RUNTIME_OUTPUT" | grep -q 'tensorops_runtime_status=passed'; then
        METAL4_TENSOROPS_RUNTIME_STATUS="passed"
        METAL4_TENSOROPS_RUNTIME_COVERED="1"
        METAL4_TENSOROPS_REASON="Metal 4 TensorOps GEMM runtime probe used TC_BACKEND_TENSOROPS_M5"
    elif printf "%s\n" "$TENSOROPS_RUNTIME_OUTPUT" | grep -q 'tensorops_runtime_status=skipped_no_m5'; then
        METAL4_TENSOROPS_RUNTIME_STATUS="skipped_no_m5"
        METAL4_TENSOROPS_RUNTIME_COVERED="0"
        METAL4_TENSOROPS_REASON="Host GPU does not report supports_tensorops_m5"
    elif printf "%s\n" "$TENSOROPS_RUNTIME_OUTPUT" | grep -q 'tensorops_runtime_status=skipped_no_gpu'; then
        METAL4_TENSOROPS_RUNTIME_STATUS="skipped_no_gpu"
        METAL4_TENSOROPS_RUNTIME_COVERED="0"
        METAL4_TENSOROPS_REASON="No usable Metal device for TensorOps runtime probe"
    else
        METAL4_TENSOROPS_RUNTIME_STATUS="failed"
        METAL4_TENSOROPS_RUNTIME_COVERED="0"
        METAL4_TENSOROPS_REASON="TensorOps runtime probe did not emit a recognized status"
        TESTS_STATUS="failed"
        write_runtime_evidence
        exit 1
    fi
    TESTS_MODE="full"
else
    if [ "$REQUIRE_GPU" = "1" ]; then
        echo "Metal device smoke failed and REQUIRE_GPU=1 was set." >&2
        exit 1
    fi
    echo "No usable Metal device in this environment; skipping GPU tests."
    ctest --test-dir "$BUILD_DIR" --output-on-failure -R 'distributed_ring'
    TESTS_MODE="no_gpu_distributed_ring_only"
    GEMM_128_TILE_STATUS="skipped_no_gpu"
    GEMM_ASYNC_STATUS="skipped_no_gpu"
fi
TESTS_STATUS="passed"

echo "[tensorcore] install"
RELEASE_SMOKE_PHASE="install"
cmake --install "$BUILD_DIR" --prefix "$PREFIX"

echo "[tensorcore] python syntax"
RELEASE_SMOKE_PHASE="python_syntax"
"$PYTHON_BIN" -m py_compile \
    "$ROOT/setup.py" \
    "$ROOT/python/tensorcore/__init__.py" \
    "$ROOT/python/tests/test_basic.py"

echo "[tensorcore] python native loader policy"
RELEASE_SMOKE_PHASE="python_loader_policy"
"$PYTHON_BIN" - "$ROOT" "$BUILD_DIR/libtensorcore.dylib" <<'PY'
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

root = pathlib.Path(sys.argv[1])
dylib = pathlib.Path(sys.argv[2])
tmp = pathlib.Path(tempfile.mkdtemp(prefix="tensorcore-loader.", dir="/private/tmp"))
try:
    pkg = tmp / "tensorcore"
    pkg.mkdir()
    shutil.copy2(root / "python" / "tensorcore" / "__init__.py", pkg / "__init__.py")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp)
    env.pop("TENSORCORE_LIB", None)
    missing = subprocess.run(
        [sys.executable, "-c", "import tensorcore"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if missing.returncode == 0 or "package-local libtensorcore.dylib not found" not in missing.stderr:
        raise SystemExit("installed-package import did not reject missing native dylib")

    env["TENSORCORE_LIB"] = str(dylib)
    explicit = subprocess.run(
        [sys.executable, "-c", "import tensorcore as tc; print(tc.version())"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if explicit.returncode != 0:
        raise SystemExit(explicit.stderr)
    print(explicit.stdout.strip())
finally:
    shutil.rmtree(tmp, ignore_errors=True)
PY

echo "[tensorcore] python wheel"
RELEASE_SMOKE_PHASE="python_wheel"
mkdir -p "$WHEEL_DIR"
TENSORCORE_NATIVE_DIR="$PREFIX/lib" \
    "$PYTHON_BIN" -m pip wheel "$ROOT" --no-build-isolation -w "$WHEEL_DIR"

WHEEL_PATH="$("$PYTHON_BIN" - "$WHEEL_DIR" <<'PY'
import pathlib
import sys

wheels = sorted(pathlib.Path(sys.argv[1]).glob("tensorcore_apple-*.whl"))
if not wheels:
    raise SystemExit("no tensorcore_apple wheel was built")
print(wheels[-1])
PY
)"
"$PYTHON_BIN" - "$WHEEL_PATH" <<'PY'
import sys
import zipfile

required = {
    "tensorcore/libtensorcore.dylib",
    "tensorcore/tensorcore.metallib",
}
with zipfile.ZipFile(sys.argv[1]) as zf:
    names = set(zf.namelist())
missing = [
    suffix for suffix in sorted(required)
    if not any(name == suffix or name.endswith(f"/purelib/{suffix}") for name in names)
]
if missing:
    raise SystemExit(f"wheel missing native artifacts: {missing}")
PY
"$PYTHON_BIN" - "$WHEEL_PATH" <<'PY'
import pathlib
import re
import subprocess
import sys
import tempfile
import zipfile

ARCH_TAGS = {
    "arm64": {"arm64"},
    "x86_64": {"x86_64"},
    "universal2": {"arm64", "x86_64"},
}


def normalize(version):
    major, minor = version
    if major > 10:
        return major, 0
    return major, minor


def platform_tags(platform):
    tags = []
    for tag in platform.split("."):
        match = re.fullmatch(r"macosx_(\d+)_(\d+)_(.+)", tag)
        if match:
            tags.append((tag, (int(match.group(1)), int(match.group(2))), match.group(3)))
    if not tags:
        raise SystemExit(f"wheel is not tagged for macOS: {platform}")
    return tags


def dylib_macos_version(path):
    out = subprocess.check_output(["otool", "-l", str(path)], text=True)
    match = re.search(r"\bminos\s+(\d+)\.(\d+)", out)
    if not match:
        match = re.search(
            r"cmd LC_VERSION_MIN_MACOSX(?:.|\n)*?\n\s*version\s+(\d+)\.(\d+)",
            out,
        )
    if not match:
        raise SystemExit(f"could not determine minimum macOS version for {path}")
    return normalize((int(match.group(1)), int(match.group(2))))


wheel_path = pathlib.Path(sys.argv[1])
platform = wheel_path.name[:-4].split("-")[-1]
with tempfile.TemporaryDirectory(prefix="tensorcore-wheel-native.", dir="/private/tmp") as td:
    with zipfile.ZipFile(wheel_path) as zf:
        members = [
            name for name in zf.namelist()
            if name == "tensorcore/libtensorcore.dylib"
            or name.endswith("/tensorcore/libtensorcore.dylib")
        ]
        if not members:
            raise SystemExit("wheel missing tensorcore/libtensorcore.dylib")
        dylib = pathlib.Path(zf.extract(members[0], td))

    archs = set(subprocess.check_output(["lipo", "-archs", str(dylib)], text=True).split())
    minos = dylib_macos_version(dylib)
    for tag, tag_version, arch_tag in platform_tags(platform):
        required = ARCH_TAGS.get(arch_tag)
        if required is None:
            raise SystemExit(f"unsupported macOS wheel architecture tag: {tag}")
        if not required.issubset(archs):
            raise SystemExit(
                f"{wheel_path.name} tag {tag} requires {sorted(required)}, "
                f"but dylib contains {sorted(archs)}"
            )
        if minos > tag_version:
            raise SystemExit(
                f"{wheel_path.name} tag {tag} advertises macOS {tag_version[0]}.{tag_version[1]}, "
                f"but dylib requires {minos[0]}.{minos[1]}"
            )
    print(f"{wheel_path.name}: dylib archs={','.join(sorted(archs))} minos={minos[0]}.{minos[1]}")
PY
WHEEL_TAG_STATUS="passed"
WHEEL_PLATFORM_TAG="${WHEEL_PATH%.whl}"
WHEEL_PLATFORM_TAG="${WHEEL_PLATFORM_TAG##*-}"

PY_VER="$("$PYTHON_BIN" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
echo "[tensorcore] python wheel install"
RELEASE_SMOKE_PHASE="python_wheel_install"
"$PYTHON_BIN" -m pip install "$WHEEL_PATH" --no-deps --prefix "$WHEEL_PREFIX"
WHEEL_SITE="$WHEEL_PREFIX/lib/$PY_VER/site-packages"
TENSORCORE_LIB= TC_METALLIB= "$PYTHON_BIN" - "$WHEEL_SITE" <<'PY'
import os
import sys

sys.path.insert(0, sys.argv[1])
import tensorcore as tc

expected = os.environ["EXPECTED_VERSION"]
lib = os.path.realpath(tc._find_lib())
expected_suffix = os.path.join("tensorcore", "libtensorcore.dylib")
if not lib.endswith(expected_suffix):
    raise SystemExit(f"package-local dylib was not selected: {lib}")
metallib = os.path.join(os.path.dirname(lib), "tensorcore.metallib")
if not os.path.exists(metallib):
    raise SystemExit(f"package-local metallib missing: {metallib}")
assert tc.version().startswith(f"tensorcore {expected}"), tc.version()
print(tc.version())
PY

echo "[tensorcore] python editable install"
RELEASE_SMOKE_PHASE="python_editable_install"
"$PYTHON_BIN" -m pip install -e "$ROOT" --no-build-isolation --prefix "$PY_PREFIX"
PY_SITE="$PY_PREFIX/lib/$PY_VER/site-packages"
TENSORCORE_LIB="$PREFIX/lib/libtensorcore.dylib" "$PYTHON_BIN" - "$PY_SITE" <<'PY'
import site
import os
import sys

site.addsitedir(sys.argv[1])
import tensorcore as tc

expected = os.environ["EXPECTED_VERSION"]
assert tc.version().startswith(f"tensorcore {expected}"), tc.version()
print(tc.version())
PY

echo "[tensorcore] installed wheel python smoke"
RELEASE_SMOKE_PHASE="installed_wheel_python_smoke"
if [ "$GPU_OK" = "1" ]; then
    TENSORCORE_LIB= TC_METALLIB= \
    TENSORCORE_TEST_INSTALLED=1 \
    PYTHONPATH="$WHEEL_SITE" \
        "$PYTHON_BIN" "$ROOT/python/tests/test_basic.py"
else
    TENSORCORE_LIB= TC_METALLIB= \
    PYTHONPATH="$WHEEL_SITE" \
        "$PYTHON_BIN" -c 'import tensorcore as tc; print(tc.version())'
fi
INSTALLED_WHEEL_SMOKE_STATUS="passed"
if [ "$GPU_OK" = "1" ]; then
    INSTALLED_WHEEL_SMOKE_MODE="python_tests"
else
    INSTALLED_WHEEL_SMOKE_MODE="import_version"
fi

echo "[tensorcore] out-of-tree CMake consumer"
RELEASE_SMOKE_PHASE="cmake_consumer"
CONSUMER_DIR="$(mktemp -d /private/tmp/tensorcore-consumer.XXXXXX)"
cat > "$CONSUMER_DIR/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.20)
project(tensorcore_consumer LANGUAGES C)

find_package(tensorcore CONFIG REQUIRED)

add_executable(consumer main.c)
target_link_libraries(consumer PRIVATE tensorcore::tensorcore_shared)

add_executable(static_consumer static_main.c)
target_link_libraries(static_consumer PRIVATE tensorcore::tensorcore)
tensorcore_copy_metallib(static_consumer)
CMAKE
cat > "$CONSUMER_DIR/main.c" <<'C'
#include <stdio.h>
#include <string.h>
#include "tensorcore/tensorcore.h"

int main(void) {
    if (tc_dtype_size(TC_DTYPE_F16) != 2 ||
        strcmp(tc_dtype_name(TC_DTYPE_F32), "f32") != 0) {
        return 1;
    }

    tc_gguf_tensor_info t = {0};
    t.n_dims = 2;
    t.dims[0] = 32;
    t.dims[1] = 1;
    t.type = TC_GGUF_TYPE_Q4_0;
    t.n_bytes = tc_quantized_size(TC_QUANT_Q4_0, 1, 32);

    tc_gguf_quantized_matrix_info q = {0};
    tc_status_t s = tc_gguf_tensor_quantized_matrix_info(&t, &q);
    if (s != TC_OK || q.N != 1 || q.K != 32 || q.quant_type != TC_QUANT_Q4_0) {
        return 1;
    }

    printf("%s\n", tc_version());
    return 0;
}
C
cat > "$CONSUMER_DIR/static_main.c" <<'C'
#include <stdio.h>
#include "tensorcore/tensorcore.h"

int main(void) {
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init: %s\n", tc_status_string(s));
        return 1;
    }
    tc_shutdown(ctx);
    printf("%s\n", tc_version());
    return 0;
}
C
cmake -S "$CONSUMER_DIR" -B "$CONSUMER_DIR/build" \
    -DCMAKE_PREFIX_PATH="$PREFIX"
cmake --build "$CONSUMER_DIR/build"
"$CONSUMER_DIR/build/consumer"
CMAKE_SHARED_CONSUMER_STATUS="passed"
if [ "$GPU_OK" = "1" ]; then
    AUTOTUNE_HOME="$CONSUMER_DIR/autotune-home"
    mkdir -p "$AUTOTUNE_HOME"
    TC_AUTOTUNE=1 HOME="$AUTOTUNE_HOME" "$CONSUMER_DIR/build/static_consumer"
    TC_AUTOTUNE=1 HOME="$AUTOTUNE_HOME" "$CONSUMER_DIR/build/static_consumer"
    CMAKE_STATIC_CONSUMER_STATUS="passed"
    AUTOTUNE_STATUS="passed"
else
    CMAKE_STATIC_CONSUMER_STATUS="skipped_no_gpu"
    AUTOTUNE_STATUS="skipped_no_gpu"
fi
CMAKE_CONSUMER_STATUS="passed"

echo "[tensorcore] pkg-config consumer"
RELEASE_SMOKE_PHASE="pkg_config_consumer"
if command -v pkg-config >/dev/null 2>&1; then
    CC_BIN="${CC:-cc}"
    PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" \
        "$CC_BIN" "$CONSUMER_DIR/main.c" \
        $(PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" pkg-config --cflags --libs tensorcore) \
        -o "$CONSUMER_DIR/pkg-consumer"
    "$CONSUMER_DIR/pkg-consumer"
    PKG_CONFIG_CONSUMER_STATUS="passed"
else
    echo "pkg-config not found; skipping pkg-config consumer smoke."
    PKG_CONFIG_CONSUMER_STATUS="skipped_pkg_config_unavailable"
fi

if [ "$REQUIRE_METAL4_TENSOROPS" = "1" ]; then
    RELEASE_SMOKE_PHASE="require_metal4_tensorops"
    if [ "$METAL4_TENSOROPS_COMPILE_STATUS" != "compiled" ] ||
       [ "$METAL4_TENSOROPS_RUNTIME_STATUS" != "passed" ]; then
        write_runtime_evidence
        echo "Metal 4 TensorOps runtime evidence required but not passed: ${METAL4_TENSOROPS_COMPILE_STATUS}/${METAL4_TENSOROPS_RUNTIME_STATUS}" >&2
        exit 1
    fi
fi

RELEASE_SMOKE_PHASE="complete"
RELEASE_SMOKE_STATUS="passed"
RELEASE_SMOKE_EXIT_STATUS="0"
write_runtime_evidence
echo "[tensorcore] release smoke OK"
