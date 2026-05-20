#!/usr/bin/env python3
"""Smoke test for the tensorcore Python binding.

Builds tensorcore as a shared library, then runs a small fp16 GEMM through
the Python API and compares against numpy.
"""

import os
import sys

# Make our local checkout importable.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import numpy as np
import tensorcore as tc

def main():
    print(f"tensorcore: {tc.version()}")
    ctx = tc.init()
    info = tc.device_info(ctx)
    print(f"device   : {info.name_str}")
    print(f"family   : Apple{info.family}")
    print(f"unified  : {info.unified_memory}")
    print(f"bf16 sg  : {info.supports_bf16_simdgroup}")
    print(f"i8   sg  : {info.supports_i8_simdgroup}")
    print(f"tensorops: {info.supports_tensorops_m5}")
    print()

    M, N, K = 256, 256, 256
    np.random.seed(0xCA75)
    A = np.random.randn(M, K).astype(np.float16)
    B = np.random.randn(K, N).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    a = tc.buffer_alloc(ctx, A.nbytes)
    b = tc.buffer_alloc(ctx, B.nbytes)
    c = tc.buffer_alloc(ctx, C.nbytes)
    tc.buffer_write(a, A)
    tc.buffer_write(b, B)

    tc.gemm(ctx, a, b, c, M, N, K, dtype="f16")
    tc.buffer_read(c, C)

    C_ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
    err = np.abs(C.astype(np.float32) - C_ref.astype(np.float32))
    rms = np.sqrt((err * err).mean())
    ref_rms = np.sqrt((C_ref.astype(np.float32) ** 2).mean())
    scaled = rms / (ref_rms + 1e-9)
    print(f"GEMM fp16 {M}x{N}x{K}:  max_abs={err.max():.3e}  scaled_rms={scaled:.3e}  "
          f"{'OK' if scaled < 1e-2 else 'FAIL'}")

    tc.buffer_free(ctx, a)
    tc.buffer_free(ctx, b)
    tc.buffer_free(ctx, c)
    tc.shutdown(ctx)
    return 0 if scaled < 1e-2 else 5

if __name__ == "__main__":
    sys.exit(main())
