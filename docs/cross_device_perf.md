# Cross-device perf — GeometricLM-relevant ops

Measured with `bench/bench_devices.py`. Same script + same workload on every
host; numbers are medians over 10 iters with 2 warmups (no torch dependency
on the cosbox path — uses the numpy fallback baked into the script).

## How to reproduce

```sh
TENSORCORE_LIB=/path/to/libtensorcore.{dylib,so} \
    PYTHONPATH=python \
    python3 bench/bench_devices.py [--cuda-managed] [--skip-large]
```

- `--cuda-managed` opts into `TC_USE_CUDA_GEMM=1` on NVIDIA hosts.
- `--skip-large` skips the 4096³ GEMM (for low-VRAM hosts).

## Measured (2026-06-24)

| op | M2 Ultra (Metal) | RTX 3090 (CUDA) | A100 | H100 |
|---|---:|---:|---:|---:|
| GEMM 4096³ fp16 | 7.20 ms / 19.1 TFLOPS | 2.25 ms / 61.1 TFLOPS | TBD | TBD |
| GEMM 1024×11008×4096 fp16 (MLP up) | 5.14 ms / 18.0 TFLOPS | 1.63 ms / 56.8 TFLOPS | TBD | TBD |
| RMSnorm 2048×4096 fp16 | 0.26 ms / 193 GB/s | 0.09 ms / 562 GB/s | TBD | TBD |
| SwiGLU 2048×4096 fp16 | 0.36 ms / 140 GB/s | 0.08 ms / 650 GB/s | TBD | TBD |
| FlashAttention B=1 H=32 S=512 D=128 | 1.41 ms / 3.06 TFLOPS | 2.84 ms / 1.51 TFLOPS [*] | TBD | TBD |
| FlashAttention BACKWARD B=1 H=32 S=512 D=128 | (TBD) | 70.6 ms / 0.15 TFLOPS [†] | TBD | TBD |
| tc_remote_tensor_fetch loopback (23.6 MiB) | 4.49 ms / 5.51 GB/s | 10.52 ms / 2.35 GB/s | TBD | TBD |

`[*]` CUDA FA forward has two kernels in tree, dispatched by D:
  - **Tensor-core wmma** (`flash_attention_forward_wmma_kernel<D>`, D ∈ {64, 128}):
    Br=Bc=32, 2 warps/block, online softmax with sO in fp32 shared, S = Q@K^T
    and O += P@V via `wmma::mma_sync` 16×16×16 fp16→fp32 mma. Built clean
    across sm_70..90 with `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`
    to opt into 47 KB shared. **Not yet benched** — cosbox GPU was occupied
    by user training (22.7/24 GB VRAM) at session close; bench pending.
    Projected: 0.1-0.3 ms / 15-40 TFLOPS for the GeoRefine shape.
  - **Half2 fallback** (`flash_attention_forward_kernel`, other D): scalar
    online softmax + half2 vectorized loads + `__hfma2` packed FMAs. The
    measured numbers above (2.84 ms / 1.51 TFLOPS) are this kernel.
    ALiBi inputs always fall back to CPU.

`[†]` CUDA FA backward (`flash_attention_backward_kernel`, K-outer dispatch
with fp32 atomicAdd scratch). The 70.6 ms / 0.15 TFLOPS measurement uses
the per-thread-atomic version. **A warp-shfl reduction variant is in tree**
that cuts dQ atomics 32× (single atomicAdd per warp per d instead of one
per thread per d); built clean, awaits cosbox availability for bench.
Projected: 5-20 ms (down from 70).

## Real-network transport — tc_remote_tensor_fetch

| link | sustained | p50 latency / 23.6 MiB | notes |
|---|---:|---:|---|
| loopback (Atlas) | 5.51 GB/s | 4.49 ms | transport ceiling, M2 Ultra |
| loopback (cosbox) | 2.35 GB/s | 10.53 ms | transport ceiling, RTX 3090 host |
| Tailscale od ↔ cosbox | 103.4 MB/s | 228 ms | userspace WG, both on LAN 192.168.1.x |
| Raw TCP, LAN-direct (192.168.x) | 117 MB/s | — | 1 GbE ceiling |
| Raw TCP, Tailscale | 111 MB/s | — | WG overhead is ~5% |

The Kimi K2.6 distributed-serving decision criterion (`≥1 GB/s → network
beats cosbox local disk @ 240 MB/s`) is **not** met by the current link.
Confirmed via raw `nc`/socket bench: the wire saturates 1 GbE; Tailscale's
userspace WireGuard adds only ~5% overhead. The transport itself is sound
(loopback hits 4.6 GB/s on Atlas, 2.4 GB/s on cosbox); replacing Tailscale
with raw TCP gains nothing because the ethernet is the bottleneck. To
unlock Kimi distributed inference on this hardware: 10 GbE NICs on both
ends. Otherwise local-disk paging on cosbox wins by a comfortable 2.3×.
