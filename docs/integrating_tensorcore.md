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

The repository ships the same out-of-tree fixture used by release CI:

```sh
cmake -S examples/native_sdk_consumer -B /tmp/tc-consumer \
  -DCMAKE_PREFIX_PATH=/opt/tensorcore
cmake --build /tmp/tc-consumer
DYLD_LIBRARY_PATH=/opt/tensorcore/lib /tmp/tc-consumer/consumer_shared
/tmp/tc-consumer/consumer_static
DYLD_LIBRARY_PATH=/opt/tensorcore/lib /tmp/tc-consumer/consumer_cxx
```

Set `TC_CONSUMER_RUN_INIT=1` on `consumer_shared` or `consumer_static`
when you also want to prove runtime initialization on the current host.

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

## Minimum-viable integration

The smallest useful "hook tensorcore into my project" is one GEMM call.
That's worth proving before you wire up the rest of the surface.

```c
#include "tensorcore/tensorcore.h"
#include <stdio.h>

int main(void) {
    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) return 1;

    tc_device_info info;
    tc_device_info_get(ctx, &info);
    printf("tensorcore on %s (Apple%d)\n", info.name, (int)info.family);

    tc_buffer *A, *B, *C;
    tc_buffer_alloc(ctx, 256 * 256 * 2, &A);
    tc_buffer_alloc(ctx, 256 * 256 * 2, &B);
    tc_buffer_alloc(ctx, 256 * 256 * 2, &C);

    tc_gemm_desc d = {0};
    d.M = d.N = d.K = 256;
    d.a_dtype = d.b_dtype = d.c_dtype = TC_DTYPE_F16;
    d.accum_dtype = TC_DTYPE_F32;
    d.alpha = 1.0f;
    d.beta = 0.0f;

    tc_gemm(ctx, &d, A, B, C);
    printf("backend: %s\n", tc_backend_name(tc_last_backend()));

    tc_buffer_free(ctx, A);
    tc_buffer_free(ctx, B);
    tc_buffer_free(ctx, C);
    tc_shutdown(ctx);
    return 0;
}
```

Build:

```sh
cc tc_smoke.c $(pkg-config --cflags --libs tensorcore) -o tc_smoke
./tc_smoke
```

You should see `backend: simdgroup_matrix`. If it says anything else, see
[troubleshooting.md](troubleshooting.md).

## Error handling, the boring-but-load-bearing version

Every entry point returns `tc_status_t`. Wrap it:

```c
#define TC_CHECK(call) do {                                       \
    tc_status_t _s = (call);                                      \
    if (_s != TC_OK) {                                            \
        fprintf(stderr, "%s:%d: %s -> %s (%d)\n",                 \
                __FILE__, __LINE__, #call,                        \
                tc_status_string(_s), (int)_s);                   \
        abort();                                                  \
    }                                                             \
} while (0)
```

Common codes you should be ready to handle:

- `TC_ERR_NOT_INITIALIZED` — `tc_init` not called, or called and then
  shut down.
- `TC_ERR_ALREADY_INITIALIZED` — second `tc_init`. Idempotent; not an
  error in your code unless you assumed it succeeded.
- `TC_ERR_UNSUPPORTED_FAMILY` — chip too old for the requested path. Drop
  to fp32 or report cleanly.
- `TC_ERR_UNSUPPORTED_DTYPE` — kernel doesn't accept that dtype combo.
  Check the table in [dtypes.md](dtypes.md).
- `TC_ERR_INVALID_SHAPE` — descriptor shape doesn't match buffer sizes.
- `TC_ERR_INVALID_ARG` — generic input validation failure; check NULL
  pointers in the call.
- `TC_ERR_KERNEL_NOT_FOUND` — the metallib doesn't have the function the
  dispatch expects. Usually a stale / missing metallib; see
  [troubleshooting.md](troubleshooting.md).

## Picking an integration footprint

There are three reasonable footprints depending on what your project
already has:

### Narrow: keep your model code, swap individual ops

Best for projects that already have a working Metal pipeline and just want
faster GEMM or quantized GEMV.

- Drop `tc_gemm` into your matmul site.
- Optionally drop `tc_gemv_quantized` into your Q4_0 path.
- Keep using your KV-cache, sampling, tokenization.

You'll start to question whether to widen the footprint once you see how
much code in your pipeline is GEMM-shaped.

### Medium: take the LLM hot path

Best for inference projects that don't have a heavy investment in
hand-tuned Metal yet.

- Use `tc_gguf_load_supported_tensors` to load the model.
- Use `tc_fused_rmsnorm_gemv` + `tc_rope_forward` + `tc_gemv_quantized` +
  `tc_attention_forward` + `tc_swiglu_forward` to drive the decode step.
- Stream the GEMVs with `tc_gemv_quantized_async` against a single
  `tc_stream` for the 2.1× tok/s win.
- Keep tokenization, sampling, KV-cache in your code.

### Wide: take the training step

Best for projects that want a Llama-class training loop on Apple Silicon
and don't already have one.

- Use `tc_gemm`, `tc_attention_forward[/_backward]`, `tc_rmsnorm_*`,
  `tc_rope_forward`, `tc_swiglu_*`, `tc_softmax_*`, `tc_adamw_step` for
  the per-step graph.
- Use `tc_dist_init(TC_DIST_SINGLE, ...)` and `tc_allreduce` for a
  one-process loop. Portable CPU builds can switch that same call site to
  `TC_DIST_GLOO` for TCP collectives today; the multi-Mac TB5 upgrade in
  v0.5 is another backend swap.
- Keep your data loader, your scheduler, your checkpoint format.

`tests/test_transformer_block.c` and `tests/test_e2e_training.c` are
worth reading as the smallest existing examples that exercise this full
shape.

## Threading

- One `tc_context` per process is the easy mode and probably what you want.
- The C API is reentrant per-context. Two threads using the same context
  must serialize access to the *same* `tc_buffer` (no atomic guarantee
  there); two threads using two different buffers concurrently is fine.
- The pipeline cache and the buffer pool are internally guarded.
- `tc_last_backend` is thread-local — it reports the most recent call on
  the calling thread, not globally.
- The Python binding doesn't release the GIL inside dispatches; if you
  want concurrent Python threads, build them around separate contexts.

## CI / build matrix considerations

- **Hardware:** tensorcore needs Apple Silicon. GitHub's macOS-14 / macOS-15
  runners expose a virtual GPU that's enough to run `tc_init` and the
  Accelerate fallback, but `simdgroup_matrix` kernels will report as
  `mps` or `accelerate_cpu`. Use a self-hosted M-series runner for
  perf-sensitive CI.
- **SDK version:** `xcrun --show-sdk-version` decides whether the Metal 4
  path is included. Pin your CI image's Xcode version or the binary's
  surface will drift across runs.
- **macOS version:** the runtime `supports_tensorops_m5` flag depends on
  M5-class hardware reporting Metal 4 support at runtime. Build with
  SDK 26.0+ and pin the macOS version on M5-class runners so that
  runtime capability reporting stays stable.

## Versioning

`TENSORCORE_VERSION_MAJOR / _MINOR / _PATCH` live in
`include/tensorcore/tensorcore.h`. The `CMakeLists.txt` `project(...
VERSION ...)` field is the authoritative build version (it drives the
`pkg-config` file). They are kept in sync at tag time.

Public ABI is stable within a minor version. Major version bumps may
move enum values; minor / patch bumps will not (new dtypes append; status
codes append; new fields go at the *end* of descriptor structs).

`find_package(tensorcore CONFIG)` enforces `SameMajorVersion` compatibility
in the generated `tensorcoreConfigVersion.cmake`.

## Going further

- Build a real inference loop: [gguf.md](gguf.md) + [quantized.md](quantized.md)
  + [attention.md](attention.md).
- Build a real training step: [training_kernels.md](training_kernels.md)
  + [gemm.md](gemm.md) + [attention.md](attention.md).
- Understand which path served your call: [family_gating.md](family_gating.md)
  + `tc_last_backend()`.
- Diagnose a failure: [troubleshooting.md](troubleshooting.md).
