#!/usr/bin/env python3
"""Compare public C struct layout against the Python ctypes declarations."""

from __future__ import annotations

import ctypes
import os
import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = r'''
#include <stddef.h>
#include <stdio.h>
#include "tensorcore/tensorcore.h"

#define SIZE(type) printf("sizeof." #type "=%zu\n", sizeof(type))
#define FIELD(type, field) printf("offsetof." #type "." #field "=%zu\n", offsetof(type, field))

int main(void) {
    SIZE(tc_device_info);
    FIELD(tc_device_info, family);
    FIELD(tc_device_info, name);
    FIELD(tc_device_info, max_buffer_bytes);
    FIELD(tc_device_info, recommended_working_set_bytes);
    FIELD(tc_device_info, max_threadgroup_memory);
    FIELD(tc_device_info, max_threads_per_threadgroup);
    FIELD(tc_device_info, thread_execution_width);
    FIELD(tc_device_info, unified_memory);
    FIELD(tc_device_info, supports_bf16_simdgroup);
    FIELD(tc_device_info, supports_i8_simdgroup);
    FIELD(tc_device_info, supports_tensorops_m5);
    FIELD(tc_device_info, supports_fp64_native);

    SIZE(tc_gemm_desc);
    FIELD(tc_gemm_desc, M);
    FIELD(tc_gemm_desc, N);
    FIELD(tc_gemm_desc, K);
    FIELD(tc_gemm_desc, a_dtype);
    FIELD(tc_gemm_desc, b_dtype);
    FIELD(tc_gemm_desc, c_dtype);
    FIELD(tc_gemm_desc, accum_dtype);
    FIELD(tc_gemm_desc, transpose_a);
    FIELD(tc_gemm_desc, transpose_b);
    FIELD(tc_gemm_desc, alpha);
    FIELD(tc_gemm_desc, beta);
    FIELD(tc_gemm_desc, lda);
    FIELD(tc_gemm_desc, ldb);
    FIELD(tc_gemm_desc, ldc);

    SIZE(tc_gemm_batched_desc);
    FIELD(tc_gemm_batched_desc, base);
    FIELD(tc_gemm_batched_desc, batch);
    FIELD(tc_gemm_batched_desc, stride_a);
    FIELD(tc_gemm_batched_desc, stride_b);
    FIELD(tc_gemm_batched_desc, stride_c);

    SIZE(tc_attention_desc);
    FIELD(tc_attention_desc, batch);
    FIELD(tc_attention_desc, heads);
    FIELD(tc_attention_desc, seq_q);
    FIELD(tc_attention_desc, seq_kv);
    FIELD(tc_attention_desc, head_dim);
    FIELD(tc_attention_desc, io_dtype);
    FIELD(tc_attention_desc, accum_dtype);
    FIELD(tc_attention_desc, softmax_scale);
    FIELD(tc_attention_desc, causal);
    FIELD(tc_attention_desc, return_lse);
    FIELD(tc_attention_desc, kv_heads);
    FIELD(tc_attention_desc, window_size);
    FIELD(tc_attention_desc, alibi_slopes);

    SIZE(tc_hip_device_info);
    FIELD(tc_hip_device_info, vendor);
    FIELD(tc_hip_device_info, device_name);
    FIELD(tc_hip_device_info, driver_version);
    FIELD(tc_hip_device_info, opencl_version);
    FIELD(tc_hip_device_info, global_memory_bytes);
    FIELD(tc_hip_device_info, local_memory_bytes);
    FIELD(tc_hip_device_info, compute_units);
    FIELD(tc_hip_device_info, max_workgroup_size);
    FIELD(tc_hip_device_info, preferred_subgroup_size);
    FIELD(tc_hip_device_info, supports_fp16);
    FIELD(tc_hip_device_info, supports_fp64);
    FIELD(tc_hip_device_info, supports_int8_dot);
    FIELD(tc_hip_device_info, unified_memory);

    SIZE(tc_diloco_config);
    FIELD(tc_diloco_config, inner_steps);
    FIELD(tc_diloco_config, outer_lr);
    FIELD(tc_diloco_config, outer_momentum);
    FIELD(tc_diloco_config, outer_beta2);
    FIELD(tc_diloco_config, outer_eps);
    FIELD(tc_diloco_config, outer_optimizer);
    FIELD(tc_diloco_config, compress);
    FIELD(tc_diloco_config, async_overlap);
    FIELD(tc_diloco_config, tolerate_dropouts);

    SIZE(tc_gguf_tensor_info);
    FIELD(tc_gguf_tensor_info, name);
    FIELD(tc_gguf_tensor_info, n_dims);
    FIELD(tc_gguf_tensor_info, dims);
    FIELD(tc_gguf_tensor_info, type);
    FIELD(tc_gguf_tensor_info, offset);
    FIELD(tc_gguf_tensor_info, n_bytes);
    FIELD(tc_gguf_tensor_info, data);

    SIZE(tc_gguf_loaded_tensor_info);
    FIELD(tc_gguf_loaded_tensor_info, name);
    FIELD(tc_gguf_loaded_tensor_info, n_dims);
    FIELD(tc_gguf_loaded_tensor_info, dims);
    FIELD(tc_gguf_loaded_tensor_info, type);
    FIELD(tc_gguf_loaded_tensor_info, offset);
    FIELD(tc_gguf_loaded_tensor_info, n_bytes);
    FIELD(tc_gguf_loaded_tensor_info, buffer);

    SIZE(tc_gguf_llama_config);
    FIELD(tc_gguf_llama_config, context_length);
    FIELD(tc_gguf_llama_config, embedding_length);
    FIELD(tc_gguf_llama_config, feed_forward_length);
    FIELD(tc_gguf_llama_config, block_count);
    FIELD(tc_gguf_llama_config, attention_head_count);
    FIELD(tc_gguf_llama_config, attention_head_count_kv);
    FIELD(tc_gguf_llama_config, rope_dimension_count);
    FIELD(tc_gguf_llama_config, vocab_size);
    FIELD(tc_gguf_llama_config, rms_norm_epsilon);
    FIELD(tc_gguf_llama_config, rope_freq_base);
    FIELD(tc_gguf_llama_config, rope_freq_scale);

    SIZE(tc_gguf_quantized_matrix_info);
    FIELD(tc_gguf_quantized_matrix_info, N);
    FIELD(tc_gguf_quantized_matrix_info, K);
    FIELD(tc_gguf_quantized_matrix_info, gguf_type);
    FIELD(tc_gguf_quantized_matrix_info, quant_type);
    FIELD(tc_gguf_quantized_matrix_info, n_bytes);
    FIELD(tc_gguf_quantized_matrix_info, buffer);
    return 0;
}
'''


def c_layout() -> dict[str, int]:
    cc = os.environ.get("CC", "cc")
    with tempfile.TemporaryDirectory(prefix="tensorcore-abi-layout.") as tmp:
        tmp_path = pathlib.Path(tmp)
        source = tmp_path / "layout.c"
        exe = tmp_path / "layout"
        source.write_text(SOURCE, encoding="utf-8")
        subprocess.run(
            [cc, "-std=c11", "-I", str(ROOT / "include"), str(source), "-o", str(exe)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output = subprocess.check_output([str(exe)], text=True)

    layout: dict[str, int] = {}
    for line in output.splitlines():
        key, value = line.split("=", 1)
        layout[key] = int(value)
    return layout


def python_layout() -> dict[str, int]:
    sys.path.insert(0, str(ROOT / "python"))
    import tensorcore as tc

    specs = {
        "tc_device_info": (tc.TCDeviceInfo, [
            "family", "name", "max_buffer_bytes", "recommended_working_set_bytes",
            "max_threadgroup_memory", "max_threads_per_threadgroup",
            "thread_execution_width", "unified_memory", "supports_bf16_simdgroup",
            "supports_i8_simdgroup", "supports_tensorops_m5", "supports_fp64_native",
        ]),
        "tc_gemm_desc": (tc.TCGemmDesc, [
            "M", "N", "K", "a_dtype", "b_dtype", "c_dtype", "accum_dtype",
            "transpose_a", "transpose_b", "alpha", "beta", "lda", "ldb", "ldc",
        ]),
        "tc_gemm_batched_desc": (tc.TCGemmBatchedDesc, [
            "base", "batch", "stride_a", "stride_b", "stride_c",
        ]),
        "tc_attention_desc": (tc.TCAttentionDesc, [
            "batch", "heads", "seq_q", "seq_kv", "head_dim", "io_dtype",
            "accum_dtype", "softmax_scale", "causal", "return_lse",
            "kv_heads", "window_size", "alibi_slopes",
        ]),
        "tc_hip_device_info": (tc.TCHipDeviceInfo, [
            "vendor", "device_name", "driver_version", "opencl_version",
            "global_memory_bytes", "local_memory_bytes", "compute_units",
            "max_workgroup_size", "preferred_subgroup_size", "supports_fp16",
            "supports_fp64", "supports_int8_dot", "unified_memory",
        ]),
        "tc_diloco_config": (tc.TCDiLoCoConfig, [
            "inner_steps", "outer_lr", "outer_momentum", "outer_beta2",
            "outer_eps", "outer_optimizer", "compress", "async_overlap",
            "tolerate_dropouts",
        ]),
        "tc_gguf_tensor_info": (tc.TCGGufTensorInfo, [
            "name", "n_dims", "dims", "type", "offset", "n_bytes", "data",
        ]),
        "tc_gguf_loaded_tensor_info": (tc.TCGGufLoadedTensorInfo, [
            "name", "n_dims", "dims", "type", "offset", "n_bytes", "buffer",
        ]),
        "tc_gguf_llama_config": (tc.TCGGufLlamaConfig, [
            "context_length", "embedding_length", "feed_forward_length", "block_count",
            "attention_head_count", "attention_head_count_kv", "rope_dimension_count",
            "vocab_size", "rms_norm_epsilon", "rope_freq_base", "rope_freq_scale",
        ]),
        "tc_gguf_quantized_matrix_info": (tc.TCGGufQuantizedMatrixInfo, [
            "N", "K", "gguf_type", "quant_type", "n_bytes", "buffer",
        ]),
    }

    layout: dict[str, int] = {}
    for c_name, (py_type, fields) in specs.items():
        layout[f"sizeof.{c_name}"] = ctypes.sizeof(py_type)
        for field in fields:
            layout[f"offsetof.{c_name}.{field}"] = getattr(py_type, field).offset
    return layout


def main() -> int:
    expected = c_layout()
    actual = python_layout()
    mismatches = []

    for key in sorted(set(expected) | set(actual)):
        c_value = expected.get(key)
        py_value = actual.get(key)
        if c_value != py_value:
            mismatches.append(f"{key}: C={c_value} Python={py_value}")

    if mismatches:
        print("Python ctypes ABI layout mismatch:", file=sys.stderr)
        for mismatch in mismatches:
            print(f"  {mismatch}", file=sys.stderr)
        return 1

    print(f"python ABI layout OK: {len(expected)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
