# Benchmarks

This page collects measured numbers, the harnesses that produce them, and
how to reproduce them on your hardware. Reference points come from the
public literature (Apple's reported peaks, MLX on equivalent hardware,
llama.cpp, the metal-flash-attention project).

The numbers below were taken on **Apple M2 Ultra (76-core GPU, 144 GB
working set)** with `[tensorcore] family=Apple8 bf16_sg=no i8_sg=no
tensorops_m5=no` — i.e. the simdgroup_matrix path, no Metal 4. The
library binary is the same one tested by `ctest --test-dir build` and
the numbers reproduce inside ±5%.

## GEMM, square shapes (measured on this checkout)

| Shape | dtype | Time | TFLOPS | Backend |
|---|---|---:|---:|---|
| 256³ | F16 | 0.28 ms | 0.12 | simdgroup_matrix |
| 512³ | F16 | 0.33 ms | 0.82 | simdgroup_matrix |
| 1024³ | F16 | 0.45 ms | 4.80 | simdgroup_matrix |
| 2048³ | F16 | 1.54 ms | 11.19 | simdgroup_matrix |
| **4096³** | **F16** | **7.69 ms** | **17.88** | simdgroup_matrix |
| 256³ | F32 | 0.38 ms | 0.09 | simdgroup_matrix |
| 512³ | F32 | 0.48 ms | 0.56 | simdgroup_matrix |
| 1024³ | F32 | 1.47 ms | 1.46 | simdgroup_matrix |
| 2048³ | F32 | 7.61 ms | 2.26 | simdgroup_matrix |
| **4096³** | **F32** | **55.89 ms** | **2.46** | simdgroup_matrix |

## Portable CPU backend

The portable CPU backend delegates fp32 and fp16 GEMM to CBLAS when
available (Accelerate on macOS, OpenBLAS / MKL / Netlib on Linux). With
CBLAS, throughput is competitive with native BLAS code; without it,
the triple-loop reference still runs (correctness-first, ~1 GFLOPS).
Detected at CMake time:

```
-- tensorcore: CBLAS via Accelerate                 (on macOS)
-- tensorcore: CBLAS via system BLAS (OpenBLAS ...)  (on Linux with OpenBLAS)
-- tensorcore: no CBLAS found; CPU GEMM uses the triple-loop reference
```

Build:

```sh
cmake -B build-cpu -DTC_ENABLE_METAL=OFF -DTC_BUILD_BENCH=ON
cmake --build build-cpu -j
OPENBLAS_NUM_THREADS=44 TC_BENCH_DTYPES=f32 TC_BENCH_SIZES=1024,2048,4096 \
  TC_BENCH_WARMUP=1 TC_BENCH_ITERS=3 ./build-cpu/bench/bench_gemm
```

### Measured: old-donkey (88-core Xeon E5-2699 v4, OpenBLAS)

| Shape | dtype | Threads | Time | Throughput | vs reference |
|---|---|---:|---:|---:|---:|
| 1024³ | F32 | 1 (OpenBLAS) | 28.5 ms | **80 GFLOPS** | 120× |
| 1024³ | F32 | 44 (1 socket) | 2.23 ms | **0.96 TFLOPS** | **1455×** |
| 2048³ | F32 | 44 (1 socket) | 14.26 ms | **1.20 TFLOPS** | — |
| 4096³ | F32 | 44 (1 socket) | 102.75 ms | **1.34 TFLOPS** | — |

For reference, M2 Ultra GPU fp32 at 4096³ is 2.46 TFLOPS. **old-donkey
through tensorcore is now ~55% of M2 Ultra's fp32 throughput** — a
legitimately useful CPU compute peer, not just a memory tier.

NUMA pinning matters: 44 threads = one socket is the sweet spot on
dual-socket old-donkey. 88 threads (both sockets) tanks due to cross-
socket cache traffic. Use `OPENBLAS_NUM_THREADS=44` or
`numactl --cpunodebind=0 --membind=0` for strict pinning.

fp16 GEMM currently goes through a dequant -> sgemm -> requant path.
The dequant/requant passes use thread-local fp32 scratch and OpenMP when
available, but the path is still bounded by conversion bandwidth rather
than the host BLAS peak.

The CPU build reports `family=Apple0` (= `TC_FAMILY_UNKNOWN`) and
`device=portable-cpu`; this is documented behavior, not a misfire. See
[family_gating.md](family_gating.md).

Reference points:
- MLX on M3 Max: ~13.3 TFLOPS fp16 (philipturner/metal-benchmarks).
- M2 Ultra fp16 theoretical peak: ~27 TFLOPS at 76-core × 1.4 GHz.
- H100 fp16 tensor-core peak: ~1500 TFLOPS (per-chip; not per-watt).

The v0.2 target is **>20 TFLOPS fp16 @ 4096³ on M2 Ultra** (~75% of peak)
via double-buffered K-loads, async dispatch, and the 128×128 tile retune.
We are currently at **~66% of peak** on the default 64×64 tile.

Small-shape numbers (256-512) are warmup- and dispatch-overhead-dominated;
treat 1024+ as the load-bearing data. The async batched stream amortizes
small-shape dispatch overhead — see the inference numbers below.

## FlashAttention forward (measured on this checkout)

| Shape (B × H × S × D, causal=1) | dtype | Time | TFLOPS | Backend |
|---|---|---:|---:|---|
| 1 × 8 × 512 × 64 | F16 | 0.62 ms | 0.87 | simdgroup_matrix |
| 1 × 8 × 1024 × 64 | F16 | 1.05 ms | 2.04 | simdgroup_matrix |
| 1 × 8 × 2048 × 64 | F16 | 2.25 ms | 3.82 | simdgroup_matrix |
| 1 × 16 × 2048 × 64 | F16 | 3.27 ms | 5.26 | simdgroup_matrix |
| 1 × 16 × 4096 × 64 | F16 | 10.55 ms | 6.51 | simdgroup_matrix |
| **1 × 32 × 4096 × 64** | **F16** | **19.43 ms** | **7.07** | simdgroup_matrix |

Reference: Apple's `metal-flash-attention` (MFA) project reports ~9-10
TFLOPS at the same D=64 shape, fully hand-tuned. We close that gap in
v0.2 with their patterns (Br=64 D=128 via aliased TG memory, K-block
early exit).

D=128 lands in `bench/bench_attention.c` once the v0.2 D=128 tile retune
ships — today the kernel is correctness-validated but the throughput-tuned
configuration only exists for D=64.

## Quantized inference — synthetic 7B Q4_0 decode (measured)

```
hidden=4096 heads=32 head_dim=128 mlp_dim=11008 layers=32
Q4_0 weight footprint per token: ~3.39 GB
```

| Mode | tok/s | GB/s effective | % of theoretical bw |
|---|---:|---:|---:|
| **async batched stream** | **186.3** | **632.2** | **~79%** |

Reference: llama.cpp on M2 Ultra reports ~55-65 tok/s at the same shape.
Theoretical ceiling (M2 Ultra LPDDR5 at ~800 GB/s) is ~220 tok/s. The
async-batched harness is at ~85% of theoretical, **clearing llama.cpp by
3-3.5×** on pure Q4_0 GEMV throughput.

Important caveat: the bench measures Q4_0 GEMV only — it excludes
attention, softmax, RoPE, and RMSnorm. End-to-end inference will land
lower; the real comparison to llama.cpp is "the GEMV core is now faster,
the rest of the loop is still active integration work."

Bench: `bench/bench_inference_7b.c`. Doesn't load a real model — allocates
random Q4_0 weights at the 7B llama shape (32 layers × hidden=4096 ×
mlp=11008) and times the per-token GEMV work across 20 tokens × 5 repeats.

## How to reproduce

### Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Smoke + correctness

```sh
ctest --test-dir build --output-on-failure
```

22/22 should pass on Apple M2 Ultra in ~3-5 s. The full CTest suite is
the 20-test correctness surface plus two executable example smokes:

```
1/22 Test  #1: test_device ........................ Passed
2/22 Test  #2: test_gemm_f32 ...................... Passed
3/22 Test  #3: test_gemm_f16 ...................... Passed
4/22 Test  #4: test_gemm_bf16 ..................... Passed
5/22 Test  #5: test_gemm_i8 ....................... Passed
6/22 Test  #6: test_attention_correctness ......... Passed
7/22 Test  #7: test_attention_backward ............ Passed
8/22 Test  #8: test_training_kernels .............. Passed
9/22 Test  #9: test_transformer_block ............. Passed
10/22 Test #10: test_e2e_training .................. Passed
11/22 Test #11: test_conv2d ......................... Passed
12/22 Test #12: test_distributed_ring ............... Passed
13/22 Test #13: test_quantized ...................... Passed
14/22 Test #14: test_fused_norm_gemv ................ Passed
15/22 Test #15: test_distributed_ring_fork .......... Passed
16/22 Test #16: test_gguf ........................... Passed
17/22 Test #17: test_tensorops_select ............... Passed
18/22 Test #18: test_tensorops_runtime .............. Passed
19/22 Test #19: test_buffer_pool .................... Passed
20/22 Test #20: python_basic ........................ Passed
21/22 Test #21: example_decode_step ................. Passed
22/22 Test #22: example_training_step ............... Passed

100% tests passed, 0 tests failed out of 22
Total Test time (real) =   3.16 sec
```

### GEMM TFLOPS

```sh
./build/bench/bench_gemm
```

Sweeps 256..4096 across fp16 / fp32 and prints median throughput plus
backend. GPU-scale results print as TFLOPS; tiny or CPU-scale results
print as GFLOPS. Reproduces the table above inside ±5%.

Use the same binary for bounded smoke or CPU runs:

```sh
TC_BENCH_DTYPES=f32 TC_BENCH_SIZES=256,512 TC_BENCH_WARMUP=1 TC_BENCH_ITERS=3 \
  ./build/bench/bench_gemm
```

### Attention TFLOPS

```sh
./build/bench/bench_attention
```

Sweeps common transformer shapes at D=64 fp16 with causal masking, batch
of 1, head counts {8, 16, 32}. Prints median TFLOPS.

### Inference (synthetic Q4_0 7B)

```sh
./build/bench/bench_inference_7b
```

Prints token-rate and effective-bandwidth numbers. The current default
uses the async-batched stream path; sync-per-call is the fallback when
no stream is supplied.

### Try the 128×128 tile

```sh
TC_USE_128_TILE=1 ./build/bench/bench_gemm
```

v0.1 regresses (~10 TFLOPS @ 4096³ vs ~18 for the 64×64 tile). v0.2 retunes.

### Try the older Q4_0 kernel

```sh
TC_Q4_USE_V1=1 ./build/bench/bench_inference_7b
```

For comparison against the v0.1.6+ default (gemm_quantized_v2.metal). The
v1 kernel is the original "1 simdgroup per output cell" path; v2 packs
more output cells per simdgroup.

## How the numbers are taken

All bench harnesses follow the same pattern:

1. Allocate inputs once, fill with non-zero data (values don't matter for
   timing; we want predictable memory traffic).
2. Warmup: 3 calls (lets the pipeline cache + buffer pool stabilize).
3. Measure: 10-20 calls timed with `clock_gettime(CLOCK_MONOTONIC)`
   between sync points.
4. Report median time, derive TFLOPS as `2 * M * N * K / time` (counting
   one multiply-add as 2 FLOPs).

The inference bench is the exception: 20 tokens × 5 repeats × 32 layers,
all GEMVs batched into a single stream, with the sync at the end of the
20-token sequence. This is exactly the loop a real decode runtime would
execute (minus attention/norms).

## Per-watt comparison

The single number that captures the structural advantage:

| Workload | M3 Max | H100 | Ratio |
|---|---|---|---|
| 7B fp16 llama decode | ~50-80 tok/s @ ~30 W | similar tok/s @ ~350 W | **~5-10× tokens / joule** |
| Idle power | ~5 W | ~50-70 W | ~10× |
| Cold start | ~0 ms | seconds (CUDA init, model load) | unbounded |

This is silicon physics, not software. The unified memory, the integrated
NPU, the lack of PCIe — all combined make Apple Silicon structurally
better at inference per watt than any discrete GPU. tensorcore's job is
to make sure the software stack doesn't squander that advantage; on the
Q4_0 7B GEMV path we're now ~3× ahead of llama.cpp's published numbers
on the same chip.

## What we don't claim

- We don't beat H100 on per-chip raw fp16 TFLOPS. Won't happen this year
  no matter what we ship.
- We don't have a "head-to-head training benchmark" against an H100
  cluster yet — v0.5 distributed first.
- The Q4_0 decode numbers are *synthetic* (random weights, no real model
  semantics); they measure the GEMV throughput, not end-to-end generation
  quality. End-to-end inference adds attention, softmax, RoPE, and RMSnorm
  to the per-token cost; the real comparison vs llama.cpp on a complete
  decode loop is a v0.2 task.

## Where to find the harness source

| Bench | Source |
|---|---|
| GEMM TFLOPS | `bench/bench_gemm.c` |
| FlashAttention TFLOPS | `bench/bench_attention.c` |
| 7B Q4_0 decode latency | `bench/bench_inference_7b.c` |

All three are small (~150 lines each), depend only on the public C ABI,
and link `libtensorcore.dylib` directly. Adapt freely.

## Per-chip projections

Rough TFLOPS ceilings the v0.2-v0.3 kernels should hit at fp16:

| Chip | Cores | Clock | Theoretical fp16 | v0.2 target (75% of peak) |
|---|---:|---:|---:|---:|
| M1 Max (Apple7) | 32 | 1.3 GHz | ~10 TFLOPS | ~7.5 |
| M2 Max (Apple8) | 38 | 1.4 GHz | ~13 TFLOPS | ~10 |
| **M2 Ultra (Apple8)** | **76** | **1.4 GHz** | **~27 TFLOPS** | **~20 (measured 17.88 today)** |
| M3 Max (Apple9) | 40 | 1.4 GHz | ~14 TFLOPS | ~10.5 |
| M4 Max (Apple10) | 40 | 1.5 GHz | ~16 TFLOPS | ~12 |
| M5 Max (Apple11) | 40 | 1.5 GHz | ~80-110 TFLOPS (TensorOps) | depends on TensorOps perf |
| M5 Ultra (Apple11) | 80 | 1.5 GHz | ~160-220 TFLOPS (TensorOps) | pending M5 hardware |

The M5 jump is real but speculative until we measure: Apple reports a 4×
speedup on small-shape matmul from the "neural accelerators." The number
above assumes the speedup transfers to the moderate-shape (4096³) case.
We'll know in v0.3.
