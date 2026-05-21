# Extending tensorcore — adding a new kernel

The mechanical version of [CONTRIBUTING.md § Adding a kernel](../CONTRIBUTING.md#adding-a-kernel),
with the why behind each step and a worked example.

## What you're touching

To add a new GPU primitive, you write code in five places:

| Layer | File | Why |
|---|---|---|
| 1. Metal kernel | `kernels/metal/<name>.metal` | The actual GPU code |
| 2. Build | `CMakeLists.txt` | So the metallib includes it |
| 3. Public ABI | `include/tensorcore/<group>.h` | Stable C declaration |
| 4. Host dispatch | `lib/ops/<group>.mm` | Encode buffers + dispatch |
| 5. Test | `tests/test_<group>.c` + `tests/CMakeLists.txt` | Numerical contract |

Optionally also:

| Layer | File | Why |
|---|---|---|
| 6. Bench | `bench/bench_<op>.c` + `bench/CMakeLists.txt` | Perf number |
| 7. Python | `python/tensorcore/__init__.py` + `python/tests/test_basic.py` | If wrappers wanted |
| 8. Docs | `docs/` | Reference + per-op page |

The work-shape pattern doesn't vary; the kernel itself is the hard part.

## Worked example: a hypothetical `tc_gelu_forward`

GELU is `y = x * 0.5 * (1 + erf(x / sqrt(2)))`, the activation used by
GPT-2/3 and BERT. Elementwise; small kernel. Good practice example.

### 1. The Metal kernel

```metal
/* kernels/metal/gelu.metal */
#include <metal_stdlib>
using namespace metal;

kernel void tc_gelu_forward_f16(
    device const half* X     [[buffer(0)]],
    device       half* Y     [[buffer(1)]],
    constant   uint& n       [[buffer(2)]],
    uint tid                 [[thread_position_in_grid]])
{
    if (tid >= n) return;
    const float x = (float)X[tid];
    const float c = 0.7978845608f;          /* sqrt(2/pi) */
    const float t = c * (x + 0.044715f * x * x * x);
    const float g = 0.5f * x * (1.f + tanh(t));
    Y[tid] = (half)g;
}
```

Conventions to follow:

- **One thread per element** for elementwise ops; `tid >= n` guard.
- **fp32 internal math** even for fp16 IO. The cast at load/store is the
  precision boundary.
- **Function constants** for compile-time switches (dtype, transpose,
  etc.) — not preprocessor.
- **Threadgroup memory** only when the algorithm needs cross-thread
  communication (reductions, matmul tiles). For elementwise: none.

### 2. Add to the build

```cmake
# CMakeLists.txt, in TC_METAL_SOURCES list
set(TC_METAL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/gemm_simdgroup.metal
    # ...
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/gelu.metal       # <-- new
)
```

If your kernel uses Metal 4 features (`mpp::tensor_ops`), gate it on
`if(TC_HAVE_METAL4) ... endif()`. If it uses private AIR `__asm`, gate
it on `if(TC_SDK_VERSION VERSION_LESS "26.0") ... endif()`.

### 3. The public ABI

```c
/* include/tensorcore/training.h (or a new include/tensorcore/activations.h) */

tc_status_t tc_gelu_forward(tc_context* ctx,
                            const tc_buffer* X,    /* [n] fp16 */
                            tc_buffer*       Y,    /* [n] fp16 */
                            int n);
```

Conventions:

- **Opaque struct pointers** (`tc_context*`, `tc_buffer*`).
- **`tc_status_t` return** for every op (no exceptions).
- **Shape parameters as `int`** unless you need 64-bit (then `int64_t`).
- **`const` for read-only buffers**, non-const for write outputs.
- **New descriptor fields go at the end** of existing descriptor structs
  to preserve ABI.

If you're adding a new op family (new noun), make a new `.h` file. If
you're adding to an existing family (new verb on existing noun), add to
the existing `.h` and update the umbrella `tensorcore.h`:

```c
/* include/tensorcore/tensorcore.h */
#include "tensorcore/activations.h"   /* <-- if new family */
```

### 4. Host dispatch

```objc++
/* lib/ops/training.mm (or new lib/ops/activations.mm) */

#include "tensorcore/training.h"
#include "core/internal.h"

tc_status_t tc_gelu_forward(tc_context* ctx,
                            const tc_buffer* X,
                            tc_buffer*       Y,
                            int n) {
    /* 1. Validate */
    if (!ctx || !X || !Y || n <= 0) return TC_ERR_INVALID_ARG;

    NSError* err = nil;
    id<MTLComputePipelineState> pso =
        tc_pipeline_get(ctx, @"tc_gelu_forward_f16", &err);
    if (!pso) return TC_ERR_PIPELINE;

    /* 2. Get a command buffer */
    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    /* 3. Encode */
    [enc setComputePipelineState:pso];
    [enc setBuffer:X->mtl offset:0 atIndex:0];
    [enc setBuffer:Y->mtl offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];

    const NSUInteger tg_size = MIN(pso.maxTotalThreadsPerThreadgroup,
                                   (NSUInteger)256);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    [enc endEncoding];

    /* 4. Commit + wait */
    [cb commit];
    [cb waitUntilCompleted];

    return TC_OK;
}
```

For an op with a backend choice (multiple kernel paths), wrap the choice
in a small selector function (see `lib/ops/gemm.mm::kernel_for` as the
canonical pattern) and call `tc_set_last_backend(...)` from each branch
so `tc_last_backend()` reports the path that ran.

### 5. The test

```c
/* tests/test_gelu.c */
#include "tensorcore/tensorcore.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

/* fp16 conversion helpers (any test file has them; copy or factor out) */
static uint16_t f32_to_f16(float x) { /* ... */ }
static float f16_to_f32(uint16_t h) { /* ... */ }

static double rms_scaled(const uint16_t* y, const float* ref, int n) {
    double sum_err = 0.0;
    double sum_ref = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = (double)f16_to_f32(y[i]) - (double)ref[i];
        sum_err += d * d;
        sum_ref += (double)ref[i] * (double)ref[i];
    }
    return sqrt(sum_err) / (sqrt(sum_ref) + 1e-30);
}

int main(void) {
    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) return 1;

    const int N = 4096;
    tc_buffer *Xb, *Yb;
    tc_buffer_alloc(ctx, N * sizeof(uint16_t), &Xb);
    tc_buffer_alloc(ctx, N * sizeof(uint16_t), &Yb);

    uint16_t* Xp = NULL;  tc_buffer_map(Xb, (void**)&Xp);
    uint16_t* Yp = NULL;  tc_buffer_map(Yb, (void**)&Yp);

    srand(0xC0FFEE);
    float* ref = malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        float x = ((float)rand() / RAND_MAX) * 4.f - 2.f;  /* [-2, 2] */
        Xp[i] = f32_to_f16(x);
        const float c = 0.7978845608f;
        const float t = c * (x + 0.044715f * x * x * x);
        ref[i] = 0.5f * x * (1.f + tanhf(t));
    }

    tc_status_t s = tc_gelu_forward(ctx, Xb, Yb, N);
    if (s != TC_OK) {
        fprintf(stderr, "tc_gelu_forward failed: %s\n", tc_status_string(s));
        return 1;
    }

    const double err = rms_scaled(Yp, ref, N);
    printf("test_gelu: rms_scaled=%.3e\n", err);
    if (err > 5e-3) return 1;   /* tolerance per docs/numerics.md */

    free(ref);
    tc_buffer_free(ctx, Xb);
    tc_buffer_free(ctx, Yb);
    tc_shutdown(ctx);
    return 0;
}
```

Register:

```cmake
# tests/CMakeLists.txt
add_executable(test_gelu test_gelu.c)
target_link_libraries(test_gelu PRIVATE tensorcore)
target_include_directories(test_gelu PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_gelu COMMAND test_gelu)
```

### 6. Optional: bench

```c
/* bench/bench_gelu.c */
/* Same harness pattern as bench_gemm.c: 3 warmup, 10 measured, report median */
```

### 7. Optional: Python binding

```python
# python/tensorcore/__init__.py

# 1. ctypes signature near the other declarations
if _lib is not None:
    _lib.tc_gelu_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int]
    _lib.tc_gelu_forward.restype = c_int

# 2. Wrapper function
def gelu_forward(ctx, X, Y, n):
    _check(_lib.tc_gelu_forward(_as_handle(ctx), _as_handle(X),
                                 _as_handle(Y), int(n)))

# 3. Context method (optional ergonomic addition)
class Context:
    def gelu_forward(self, X, Y, n):
        return gelu_forward(self.handle, X, Y, n)
```

Then in `python/tests/test_basic.py` add a small test that allocates a
buffer, fills it, calls `tc.gelu_forward`, and compares to a NumPy
reference.

### 8. Docs

Add a section to `docs/training_kernels.md` (or wherever the op fits)
documenting the signature, expected dtype, and contracts. If it's a
brand-new family, add a new doc and link it from `docs/README.md`.

Update [CHANGELOG.md](../CHANGELOG.md) under "Unreleased".

## Anti-patterns to avoid

- **Putting compile-time switches in the preprocessor.** Use function
  constants instead. Multiple `#define`-specialized variants of the same
  kernel are an anti-pattern; one source, multiple specializations via
  function constants is correct.
- **Calling `[cb waitUntilCompleted]` in async paths.** That defeats the
  whole point of the async API; instead encode into the stream's pending
  command buffer and let `tc_stream_sync` wait.
- **Skipping the public-export check.** If your new kernel uses helper
  functions, mark them `static` or they leak into the dylib symbol table
  and break `scripts/check_public_exports.sh`.
- **Allocating in the hot path.** Inside a dispatch handler, *don't*
  call `tc_buffer_alloc`. The buffer pool is fast but it's still a mutex.
  Allocate scratch buffers once at init and pass them through.
- **fp16 accumulators**. The library's contract is "fp32 accumulation
  inside the kernel, fp16 IO". Don't write `half sum = ...` then a long
  inner loop; write `float sum = ...` and cast at the end.

## Adding a backend target

Different from adding a kernel. A backend target is an alternative
dispatch path the dispatch layer chooses among (`SIMDGROUP_MATRIX`,
`TENSOROPS_M5`, `MPS`, `ACCELERATE_CPU`, ...). To add one:

1. Add a new `TC_BACKEND_*` enum value at the end of `tc_backend_t` in
   `include/tensorcore/gemm.h`.
2. Update `tc_backend_name()` in `lib/core/device.mm` to render the new
   name (lowercase string, matching the convention).
3. Implement your dispatch in a new file under `lib/ops/` or
   `lib/<area>/`.
4. Call `tc_set_last_backend(TC_BACKEND_YOURS)` from your dispatch site
   before commit / stream handoff.
5. Wire it into the selector function (`kernel_for` in `lib/ops/gemm.mm`,
   or the equivalent for whatever op family).

See [docs/architecture.md § Fallback ladder](architecture.md) and
`lib/fallback/mps_gemm.mm` as the canonical fallback implementation.

## Adding a dtype

Different from adding a kernel. To add a new dtype to `tc_dtype_t`:

1. **Add to the end of the enum** in `include/tensorcore/dtype.h`. Don't
   renumber.
2. Update `tc_dtype_size()` (inline in the header) and `tc_dtype_name()`
   (in `lib/core/dtype.c`) to handle the new value.
3. If your dtype is meant for matrix work, update the dispatch logic in
   `lib/ops/gemm.mm::kernel_for` to either pick a matching kernel or
   fall back to an existing path.
4. Add a `test_gemm_<dtype>.c` exercising the new dtype against an
   appropriate reference.
5. Update [docs/dtypes.md](dtypes.md) — the dtype table is the single
   source of truth for "what dtypes does the library support."

`TC_DTYPE_SF64` / `_DF64` / `_FP24` / `_FP53` are reserved for the v0.4
consolidation; if your new dtype is one of those, you'll be moving the
existing eshkol-platform implementations into tensorcore rather than
writing fresh ones. See [docs/precision_emulation.md](precision_emulation.md).

## Adding a Python helper

Pure-Python ergonomics on top of the existing ABI. Pattern:

```python
class Context:
    def my_helper(self, ...):
        """One-line docstring that explains the convenience the helper provides."""
        # call self.method_x, self.method_y, return the composed result
```

Keep wrappers in the Python file; don't add C ABI just to make Python
shorter. The C ABI is the contract.

## When to file vs implement

- **Bug**: file via the bug-report issue template, then fix-forward.
- **New feature**: file via the feature-request template first if the
  surface is non-obvious; for narrow additions (a single new elementwise
  kernel that mirrors an existing one) just submit a PR.
- **Performance regression**: file via the regression template with
  before/after numbers — those drive the priority.

See [.github/ISSUE_TEMPLATE/](../.github/ISSUE_TEMPLATE/) for the
templates.

## See also

- [CONTRIBUTING.md](../CONTRIBUTING.md) — the high-level contributor
  guide.
- [docs/architecture.md](architecture.md) — internal layering.
- [docs/numerics.md](numerics.md) — the tolerance contract every test
  enforces.
- [docs/kernels.md](kernels.md) — per-existing-kernel walkthrough.
