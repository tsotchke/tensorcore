# Integrating tensorcore

This is the short path for using tensorcore from another local project while
the API is still pre-1.0.

## Build and install

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
cmake --install build --prefix /opt/tensorcore
```

The install contains:

- `include/tensorcore/*.h`
- `lib/libtensorcore.a`
- `lib/libtensorcore.dylib`
- `lib/tensorcore.metallib`
- `lib/cmake/tensorcore/tensorcoreConfig.cmake`
- `lib/pkgconfig/tensorcore.pc`

At runtime, tensorcore loads kernels from this order:

1. `TC_METALLIB` environment override
2. `tensorcore.metallib` next to the loaded dylib or executable
3. `../lib/tensorcore.metallib` relative to the loaded dylib or executable
4. the build-tree metallib path baked into local builds
5. `default.metallib` in the current working directory

## CMake consumer

```cmake
find_package(tensorcore CONFIG REQUIRED)

add_executable(my_inference main.c)
target_link_libraries(my_inference PRIVATE tensorcore::tensorcore_shared)
```

Use `tensorcore::tensorcore` for the static library. Use
`tensorcore::tensorcore_shared` when you want the installed dylib, which is also
what the Python binding loads.

If tensorcore is installed somewhere non-standard:

```sh
cmake -B build -DCMAKE_PREFIX_PATH=/opt/tensorcore
```

## C API starting point

```c
#include "tensorcore/tensorcore.h"

tc_context* ctx = NULL;
tc_status_t s = tc_init(&ctx);
if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
    return 1;
}

tc_device_info info;
tc_device_info_get(ctx, &info);

/* Allocate tc_buffer objects and call tc_gemm, tc_attention_forward,
 * tc_quantize_weights, tc_gemv_quantized, or tc_gguf_tensor_to_buffer. */

tc_shutdown(ctx);
```

## pkg-config consumer

For Makefiles or direct compiler commands:

```sh
export PKG_CONFIG_PATH=/opt/tensorcore/lib/pkgconfig
cc main.c $(pkg-config --cflags --libs tensorcore) -o my_inference
```

The `tensorcore.pc` file embeds an rpath to the installed library directory so
small local tools can run without extra `DYLD_LIBRARY_PATH` setup.

## Python consumer

Install the pure-Python binding from this checkout:

```sh
python3 -m pip install -e . --no-build-isolation
```

The Python package does not vendor the native library. Install tensorcore with
CMake first, then point the binding at the installed dylib when it is not in a
standard prefix:

```sh
export TENSORCORE_LIB=/opt/tensorcore/lib/libtensorcore.dylib
export TC_METALLIB=/opt/tensorcore/lib/tensorcore.metallib
python3 -c 'import tensorcore as tc; print(tc.version())'
```

For local development without install, run from the checkout after building;
`python/tensorcore/__init__.py` searches `build/libtensorcore.dylib`. For
installed builds, `TC_METALLIB` is optional when `tensorcore.metallib` sits next
to `libtensorcore.dylib`.

The Python binding exposes the same GGUF descriptor helper:

```python
loaded = tc.gguf_load_supported_tensors(ctx, gguf)
proj = tc.gguf_loaded_get_tensor(loaded, "blk.0.attn_q.weight")
qproj = tc.gguf_loaded_tensor_quantized_matrix_info(proj)
tc.gemv_quantized(ctx, x, qproj["buffer"], y,
                  qproj["quant_type"], 1, qproj["N"], qproj["K"])
```

## Current integration targets

Start with narrow replacements:

- GEMM callsites: `tc_gemm` / `tc_gemm_async`
- small-batch inference projection GEMVs: `tc_quantize_weights` +
  `tc_gemv_quantized`
- LLM elementwise/fusion callsites: `tc_rmsnorm_forward`,
  `tc_rope_forward`, `tc_swiglu_forward`, and `tc_fused_rmsnorm_gemv`
- GGUF tensor loading: `tc_gguf_open` + `tc_gguf_load_supported_tensors`
- attention experiments: `tc_attention_forward`

Avoid treating tensorcore as a complete model runtime yet. Tokenization,
sampling, KV-cache orchestration, model-specific graph scheduling, and full
fine-tuning integration are still active work.

## GGUF bulk load pattern

```c
tc_gguf_file* gguf = NULL;
tc_gguf_loaded_model* model = NULL;

tc_gguf_open("model.gguf", &gguf);
tc_gguf_load_supported_tensors(ctx, gguf, &model);

tc_gguf_loaded_tensor_info tok_embeddings;
tc_gguf_loaded_get_tensor(model, "token_embd.weight", &tok_embeddings);

/* tok_embeddings.buffer is a tc_buffer* ready for tensorcore kernels. */

tc_gguf_loaded_model_free(ctx, model);
tc_gguf_close(gguf);
```

`tc_gguf_load_supported_tensors` skips unknown GGUF encodings instead of
failing the whole load. Check `tc_gguf_loaded_skipped_tensor_count(model)` and
decide whether your runtime can proceed.

For quantized 2D weights, use the descriptor helper before calling GEMV. GGUF
matrix tensors store `dim[0]=K, dim[1]=N`; `tc_gemv_quantized` takes `[N, K]`.

```c
tc_gguf_loaded_tensor_info proj;
tc_gguf_loaded_get_tensor(model, "blk.0.attn_q.weight", &proj);

tc_gguf_quantized_matrix_info qproj;
tc_gguf_loaded_tensor_quantized_matrix_info(&proj, &qproj);

tc_gemv_quantized(ctx, x, qproj.buffer, y,
                  qproj.quant_type, 1, qproj.N, qproj.K);
```

GGUF metadata helpers cover the fields most model runtimes need:

```c
tc_gguf_llama_config cfg;
tc_gguf_get_llama_config(gguf, &cfg);

int64_t hidden = tc_gguf_meta_get_i64(gguf, "llama.embedding_length", 0);
double eps = tc_gguf_meta_get_f64(
    gguf, "llama.attention.layer_norm_rms_epsilon", 1e-5);

uint64_t vocab = tc_gguf_meta_array_count(gguf, "tokenizer.ggml.tokens");
const char* token = NULL;
size_t token_len = 0;
tc_gguf_meta_array_get_str(
    gguf, "tokenizer.ggml.tokens", 0, &token, &token_len);
```

String array values are pointer+length pairs into the mapped GGUF file; they
are valid until `tc_gguf_close`.
