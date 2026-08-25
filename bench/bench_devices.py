#!/usr/bin/env python3
"""bench_devices.py — measured-latency table across Metal / 3090 / A100 / H100.

Produces the table GeoRefine P4 needs: tokens/sec and TFLOPS per device
for the GeometricLM-relevant ops. Same script runs on every host; output
format is identical so the cross-device table is just a paste of the
runs concatenated.

Usage:
    # On each host (Atlas / cosbox / GCP A100 / GCP H100):
    TENSORCORE_LIB=/path/to/libtensorcore.{dylib,so} \\
        python3 bench/bench_devices.py [--cuda-managed]
    # then collect the printed table from each host into the README.

What we measure (all fp16 IO, matches Llama-3 8B hidden=4096):
  - GEMM 4096³                      (the GeometricLM-block bottleneck)
  - GEMM 1024×4096×11008            (the SwiGLU MLP gate/up projection)
  - RMSnorm 4096-wide, 2048 tokens  (memory-BW bound)
  - SwiGLU 4096×11008, 2048 tokens  (memory-BW bound)
  - FlashAttention B=1 H=32 S=512 D=128
  - tc_remote_tensor_fetch loopback (transport ceiling on this host)

For each op we report median wall time + derived TFLOPS / GB/s.

The script auto-detects the available backend at run-time:
  - Apple: Metal via tc_gemm (default)
  - CUDA host (cosbox/GCP): set --cuda-managed to opt into the
    managed-memory CUDA path (TC_USE_CUDA_GEMM=1)
  - Otherwise: portable CPU path

GeoRefine reference for FLOPS counts:
    GEMM(M,N,K)        = 2*M*N*K
    RMSnorm(N,D)       ~ 3*N*D   (sum² + scale + rsqrt amortized)
    SwiGLU(N,D)        ~ 3*N*D
    FlashAttention     ~ 4*B*H*S*S*D + 2*B*H*S*D (Q@K + softmax + V dot)
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Callable, Tuple


def _setup_env(cuda_managed: bool) -> None:
    if cuda_managed:
        os.environ.setdefault("TC_USE_CUDA_GEMM", "1")


def _now() -> float:
    return time.perf_counter()


def _bench(name: str, op: Callable, work_per_iter: float, *,
           iters: int = 10, warmup: int = 2, divisor: float = 1e12) -> Tuple[float, float]:
    """Run `op` `iters` times; return (avg_ms, perf).

    Times the whole tight loop and averages: this matches steady-state
    throughput correctly even when the backend has async launch overlap
    (CUDA), where per-iter timestamps would be misleading.

    `divisor`: 1e12 -> TFLOPS for flops; 1e9 -> GB/s for bytes.
    """
    for _ in range(warmup):
        op()
    t0 = _now()
    for _ in range(iters):
        op()
    dt = (_now() - t0) / iters
    perf = work_per_iter / dt / divisor
    return dt * 1000.0, perf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-managed", action="store_true",
                         help="opt into CUDA managed-memory path on NVIDIA hosts")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--skip-large", action="store_true",
                         help="skip the GEMM 4096³ test (low-RAM hosts)")
    args = parser.parse_args()
    _setup_env(args.cuda_managed)

    try:
        import tensorcore as tc
    except ImportError as exc:
        print(f"ERROR: tensorcore import failed: {exc}")
        return 1
    import numpy as np
    try:
        import torch
        def _randn(*shape, dtype):
            np_dtype = {F16: np.float16, torch.float32: np.float32}[dtype]
            return torch.from_numpy(np.random.randn(*shape).astype(np_dtype))
        def _ones(*shape, dtype):
            np_dtype = {F16: np.float16, torch.float32: np.float32}[dtype]
            return torch.from_numpy(np.ones(shape, dtype=np_dtype))
        F16 = torch.float16
        def _np(x): return x.numpy().copy()
    except ImportError:
        # numpy-only fallback for hosts without torch.
        F16 = np.float16
        class _Arr:
            def __init__(self, a): self.a = a
            def numpy(self): return self.a
        def _randn(*shape, dtype):
            return _Arr(np.random.randn(*shape).astype(dtype))
        def _ones(*shape, dtype):
            return _Arr(np.ones(shape, dtype=dtype))
        def _np(x): return x.numpy().copy()

    ctx = tc.Context()
    info = ctx.device_info()
    device_name = info.name.decode("utf-8") if isinstance(info.name, bytes) else str(info.name)
    print(f"=== tensorcore device: {device_name} ===")
    if args.cuda_managed:
        print(f"=== TC_USE_CUDA_GEMM=1 (CUDA managed-memory path) ===")
    print()

    results = []

    def add(name: str, ms: float, perf: float, perf_unit: str):
        results.append((name, ms, perf, perf_unit))
        print(f"  {name:<48s} {ms:8.3f} ms   {perf:7.2f} {perf_unit}")

    # ---- 1. GEMM 4096^3 fp16 ----
    if not args.skip_large:
        M = N = K = 4096
        A = _randn(M, K, dtype=F16)
        B = _randn(K, N, dtype=F16)
        bA = ctx.buffer_from_array(A.numpy().copy())
        bB = ctx.buffer_from_array(B.numpy().copy())
        bC = tc.Buffer(ctx, nbytes=M * N * 2)
        def op_gemm():
            tc.gemm(ctx, bA, bB, bC, M, N, K, dtype="f16", accum="f32")
        ms, tflops = _bench("GEMM 4096³ fp16", op_gemm, 2.0 * M * N * K, iters=args.iters)
        add("GEMM 4096³ fp16", ms, tflops, "TFLOPS")

    # ---- 2. GEMM 1024×4096×11008 fp16 (SwiGLU up-proj) ----
    M, N, K = 1024, 11008, 4096
    A = _randn(M, K, dtype=F16)
    B = _randn(K, N, dtype=F16)
    bA = ctx.buffer_from_array(A.numpy().copy())
    bB = ctx.buffer_from_array(B.numpy().copy())
    bC = tc.Buffer(ctx, nbytes=M * N * 2)
    def op_gemm_mlp():
        tc.gemm(ctx, bA, bB, bC, M, N, K, dtype="f16", accum="f32")
    ms, tflops = _bench("GEMM 1024×11008×4096 fp16 (MLP up)", op_gemm_mlp,
                         2.0 * M * N * K, iters=args.iters)
    add("GEMM 1024×11008×4096 fp16", ms, tflops, "TFLOPS")

    # ---- 3. RMSnorm 2048 × 4096 fp16 ----
    N, D = 2048, 4096
    X = _randn(N, D, dtype=F16)
    gamma = _ones(D, dtype=F16)
    bX = ctx.buffer_from_array(X.numpy().copy())
    bg = ctx.buffer_from_array(gamma.numpy().copy())
    bY = tc.Buffer(ctx, nbytes=N * D * 2)
    br = tc.Buffer(ctx, nbytes=N * 4)
    def op_rmsnorm():
        tc.rmsnorm_forward(ctx, bX, bg, bY, br, N, D, 1e-5)
    bytes_per_iter = 3.0 * N * D * 2  # 2 reads (X, gamma) + 1 write (Y)
    ms, gbps = _bench("RMSnorm 2048×4096 fp16", op_rmsnorm,
                       bytes_per_iter, iters=args.iters, divisor=1e9)
    add("RMSnorm 2048×4096 fp16", ms, gbps, "GB/s (bandwidth)")

    # ---- 4. SwiGLU 2048 × 4096 fp16 ----
    n = 2048 * 4096
    gate = _randn(n, dtype=F16)
    up = _randn(n, dtype=F16)
    bg = ctx.buffer_from_array(gate.numpy().copy())
    bu = ctx.buffer_from_array(up.numpy().copy())
    bo = tc.Buffer(ctx, nbytes=n * 2)
    def op_swiglu():
        tc.swiglu_forward(ctx, bg, bu, bo, n)
    bytes_swiglu = 3.0 * n * 2  # gate + up read, out write
    ms, gbps = _bench("SwiGLU 2048×4096 fp16", op_swiglu,
                       bytes_swiglu, iters=args.iters, divisor=1e9)
    add("SwiGLU 2048×4096 fp16", ms, gbps, "GB/s (bandwidth)")

    # ---- 5. FlashAttention B=1 H=32 S=512 D=128 ----
    B_a, H, S, D = 1, 32, 512, 128
    Q = _randn(B_a, H, S, D, dtype=F16)
    K_ = _randn(B_a, H, S, D, dtype=F16)
    V = _randn(B_a, H, S, D, dtype=F16)
    bQ = ctx.buffer_from_array(Q.numpy().copy())
    bK = ctx.buffer_from_array(K_.numpy().copy())
    bV = ctx.buffer_from_array(V.numpy().copy())
    bO = tc.Buffer(ctx, nbytes=B_a * H * S * D * 2)
    def op_attn():
        tc.attention_forward(ctx, bQ, bK, bV, bO, B_a, H, S, S, D)
    flops_attn = 4.0 * B_a * H * S * S * D + 2.0 * B_a * H * S * D
    ms, tflops = _bench("FlashAttention B=1 H=32 S=512 D=128 fp16",
                         op_attn, flops_attn, iters=args.iters)
    add("FlashAttention B=1 H=32 S=512 D=128", ms, tflops, "TFLOPS")

    # ---- 6. tc_remote_tensor_fetch loopback ceiling (Kimi K2.6 transport) ----
    try:
        srv = tc.remote_init(ctx, tc.TC_REMOTE_ROLE_WEIGHT_SERVER,
                              "tcp://127.0.0.1:49999")
        cli = tc.remote_init(ctx, tc.TC_REMOTE_ROLE_COMPUTE_CLIENT, None)
        expert_bytes = 24731648
        import numpy as np
        expert = np.frombuffer(np.random.bytes(expert_bytes), dtype=np.uint8).copy()
        tc.remote_register_tensor(srv, "bench/expert", expert.ctypes.data, expert_bytes)
        peer = tc.remote_connect(cli, "tcp://127.0.0.1:49999")
        dst = np.zeros(expert_bytes, dtype=np.uint8)
        # warmup
        tc.remote_tensor_fetch(cli, peer, "bench/expert", dst.ctypes.data, expert_bytes)
        N_iters = 8
        t0 = _now()
        for _ in range(N_iters):
            tc.remote_tensor_fetch(cli, peer, "bench/expert", dst.ctypes.data, expert_bytes)
        dt = _now() - t0
        gbps = (N_iters * expert_bytes) / dt / 1e9
        ms = dt / N_iters * 1000
        add(f"tc_remote_tensor_fetch loopback (23.6 MiB)", ms, gbps, "GB/s")
        tc.remote_shutdown(cli)
        tc.remote_shutdown(srv)
    except Exception as exc:
        print(f"  remote_tensor_fetch loopback: skipped ({exc!r})")

    # ---- summary table for cross-device pasting ----
    print()
    print(f"=== summary ({device_name}) ===")
    print(f"| op | ms | perf |")
    print(f"|---|---:|---:|")
    for name, ms, perf, unit in results:
        print(f"| {name} | {ms:.2f} | {perf:.2f} {unit} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
