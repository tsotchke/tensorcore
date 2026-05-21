# bench/

TFLOPS and tokens/sec harnesses for the load-bearing kernels. Each is a
single `.c` file that links only `libtensorcore.dylib` and reports
median over a fixed number of iterations after warmup.

```sh
./build/bench/bench_gemm           # GEMM TFLOPS sweep
./build/bench/bench_attention       # FlashAttention TFLOPS
./build/bench/bench_inference_7b    # 7B Q4_0 decode latency
```

All three follow the same pattern: 3 warmup calls, 10-20 measured calls,
report median TFLOPS / tok-s with the backend that served it. See
[../docs/benchmarks.md](../docs/benchmarks.md) for current measured numbers.

## `bench_gemm.c`

GEMM TFLOPS sweep across square shapes 256-4096 and dtypes fp16 / fp32.
Prints one line per (dtype, shape) cell:

```
f16  M=4096 N=4096 K=4096   simdgroup_matrix    median=7.69 ms   17.88 TFLOPS
```

The reported backend should be `simdgroup_matrix` on every M-series
chip; if it's `mps` or `accelerate_cpu`, you found a kernel-coverage
gap or a path that fell back unexpectedly. See
[../docs/family_gating.md](../docs/family_gating.md).

Try the experimental large tile:

```sh
TC_USE_128_TILE=1 ./build/bench/bench_gemm
```

(v0.1 regresses on M2; v0.2 retunes.)

## `bench_attention.c`

FlashAttention forward TFLOPS at D=64 fp16, causal=true, across common
transformer shapes (B=1, H ∈ {8,16,32}, S ∈ {512, 1024, 2048, 4096}).

```
B=1 H=32 S=4096 D=64 causal=1   simdgroup_matrix   median=19.4 ms   7.07 TFLOPS
```

D=128 isn't in the sweep yet; the v0.1 D=128 kernel is correctness-
validated but tile-tuning is a v0.2 item — see
[../docs/attention.md](../docs/attention.md).

## `bench_inference_7b.c`

The synthetic 7B Q4_0 decode harness. Allocates random Q4_0 weights at
the 7B Llama shape (32 layers × hidden=4096 × mlp=11008) and times the
per-token GEMV work, excluding attention/RoPE/RMSnorm — pure GEMV
throughput.

```
=== tensorcore 7B Q4_0 decode latency bench ===
hidden=4096 heads=32 head_dim=128 mlp_dim=11008 layers=32

Q4_0 weight footprint per 7B model: 3.39 GB
Results (20 tokens x 5 repeats, 32 layers, Q4_0 GEMVs only):
  median tok/s   : 186.3
  median weight bw: 632.2 GB/s
```

Reference: llama.cpp on M2 Ultra reports ~55-65 tok/s. The current
async-batched harness is ~3-3.5× ahead of that on pure GEMV throughput.
End-to-end inference (attention + softmax + RoPE + RMSnorm on top) is a
v0.2 integration target.

Switch to the v1 Q4_0 kernel for comparison:

```sh
TC_Q4_USE_V1=1 ./build/bench/bench_inference_7b
```

See [../docs/quantized.md](../docs/quantized.md) for the v1 → v2 kernel
design diff.

## Adding a bench

Pattern (from `bench_gemm.c`):

```c
#include "tensorcore/tensorcore.h"
#include <stdio.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    tc_context* ctx;  tc_init(&ctx);

    /* 1. Allocate inputs once */
    /* 2. Warm up: 3 calls (lets pipeline cache + buffer pool settle) */
    /* 3. Time N calls; record times[i] */
    /* 4. qsort and report times[N/2] (median) */
    /* 5. Report TFLOPS = (2*M*N*K / time) * 1e-12 */
    /* 6. Print tc_backend_name(tc_last_backend()) per cell */

    tc_shutdown(ctx);
    return 0;
}
```

Register in `bench/CMakeLists.txt`:

```cmake
add_executable(bench_mything bench_mything.c)
target_link_libraries(bench_mything PRIVATE tensorcore)
target_include_directories(bench_mything PRIVATE ${CMAKE_SOURCE_DIR}/include)
```

## What each harness measures

| Bench | Counts | Excludes |
|---|---|---|
| `bench_gemm` | `2 * M * N * K` FLOPs per call | warmup, dispatch overhead amortized in median |
| `bench_attention` | `2 * B * H * S² * D` FLOPs per call (counts both halves of causal mask in the score-FLOPs total — the reported TFLOPS is "effective" user-perceived) | same |
| `bench_inference_7b` | weight bytes read per token (memory-bound metric) | attention, softmax, RoPE, RMSnorm |

These are *per-call* numbers; nothing is amortized across calls except
warmup. For end-to-end inference, expect lower throughput once attention
+ KV-cache + sampling are in the loop.

## See also

- [../docs/benchmarks.md](../docs/benchmarks.md) — current measured
  numbers with chip + shape + backend.
- [../docs/observability.md § Backend tracing](../docs/observability.md)
  — how to interpret the backend field in bench output.
