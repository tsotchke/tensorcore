"""2:4 structured-sparse GEMM — torch.autograd glue over tensorcore C kernels.

For GeometricLM's T8 sparsity track: take a dense weight matrix, prune to
the 2:4 pattern (each 4-element block along the K axis keeps top-2 |·|),
then dispatch the forward GEMM into cusparseLt's tensor-core path on
Ampere+ for a 2× speedup. Backward goes back through the dense path
(gradient of sparse weights isn't directly accelerated by cusparseLt 2:4).

Public API:

    from tensorcore_torch import sparse_24 as sp

    # Prune weights in-place; returns the same tensor for chaining.
    W = sp.prune_2_4(W)
    assert sp.is_2_4(W), "pruning failed"

    # Drop-in for F.linear when the weight is 2:4-pruned:
    y = sp.sparse_linear(x, W, bias=b)

    # Or directly:
    Y = sp.sparse_gemm(A, B)   # A @ B where B is 2:4-pruned

    # Hardware probe — caller may keep dense path when cusparseLt absent:
    if sp.tensor_core_available():
        ...

Numerics: fp16/bf16/fp32 supported. On hosts without cusparseLt, the
fallback is a bit-correct dense GEMM (zeros contribute zero), so model
code stays portable.
"""

from __future__ import annotations

import threading
from typing import Optional

import torch

try:
    import tensorcore as _tc
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tensorcore_torch.sparse_24 requires the `tensorcore` ctypes package."
    ) from exc


_TC_DTYPE = {
    torch.float32: 0,   # TC_DTYPE_F32
    torch.float16: 2,   # TC_DTYPE_F16
    torch.bfloat16: 1,  # TC_DTYPE_BF16
}


_ctx_lock = threading.Lock()
_ctx = None


def _get_ctx():
    global _ctx
    if _ctx is None:
        with _ctx_lock:
            if _ctx is None:
                _ctx = _tc.Context()
    return _ctx


def _dtype_code(t: torch.Tensor) -> int:
    if t.dtype not in _TC_DTYPE:
        raise TypeError(f"sparse_24 supports fp32/fp16/bf16; got {t.dtype}")
    return _TC_DTYPE[t.dtype]


def tensor_core_available() -> bool:
    """True iff the runtime has cusparseLt + Ampere+ hardware (real 2× speedup)."""
    return _tc.sparse_24_available()


def prune_2_4(W: torch.Tensor) -> torch.Tensor:
    """Apply the 2:4 sparsity mask in place. Each 4-element block along
    the last (K) dim keeps the two largest |·| entries; zeros the others.

    Returns the same tensor (for chaining). cols must be a multiple of 4.
    """
    ctx = _get_ctx()
    if W.dim() < 2:
        raise ValueError(f"prune_2_4 expects W.dim() >= 2; got {W.shape}")
    if W.shape[-1] % 4 != 0:
        raise ValueError(f"prune_2_4: last dim must be multiple of 4; got {W.shape[-1]}")
    flat = W.detach().contiguous()
    rows = int(flat.numel() // flat.shape[-1])
    cols = int(flat.shape[-1])
    arr = flat.view(rows, cols)
    arr_np = arr.numpy()
    buf = ctx.buffer_from_array(arr_np)
    _tc.sparse_24_prune(ctx, buf, _dtype_code(flat), rows, cols)
    # Read back into the original tensor (in-place).
    import numpy as np
    np_dtype = {0: np.float32, 1: np.float32, 2: np.float16}[_dtype_code(flat)]
    if W.dtype == torch.bfloat16:
        # numpy lacks native bf16; round-trip through fp32 then convert back.
        new_arr = buf.to_numpy((rows, cols), np.float32)
        W.copy_(torch.from_numpy(new_arr).view_as(W).to(W.dtype))
    else:
        new_arr = buf.to_numpy((rows, cols), np_dtype)
        W.copy_(torch.from_numpy(new_arr).view_as(W).to(W.dtype))
    return W


def is_2_4(W: torch.Tensor) -> bool:
    """Verify W satisfies the 2:4 pattern."""
    ctx = _get_ctx()
    if W.dim() < 2 or W.shape[-1] % 4 != 0:
        return False
    flat = W.detach().contiguous().view(-1, W.shape[-1])
    arr = flat.numpy() if W.dtype != torch.bfloat16 else flat.to(torch.float32).numpy()
    buf = ctx.buffer_from_array(arr)
    dtype_code = _dtype_code(flat) if W.dtype != torch.bfloat16 else _TC_DTYPE[torch.float32]
    return _tc.sparse_24_check(ctx, buf, dtype_code,
                                int(flat.shape[0]), int(flat.shape[1]))


def sparse_gemm(A: torch.Tensor, B: torch.Tensor,
                 alpha: float = 1.0, beta: float = 0.0,
                 out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """C = alpha * A @ B + beta * C, with B already 2:4-pruned along K.

    A: [M, K], B: [K, N]. Returns [M, N].
    """
    ctx = _get_ctx()
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError("sparse_gemm needs 2-D A and B")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"sparse_gemm shape mismatch: A {A.shape}, B {B.shape}")
    if A.shape[1] % 4 != 0:
        raise ValueError(f"sparse_gemm K must be multiple of 4; got {A.shape[1]}")
    M, K = int(A.shape[0]), int(A.shape[1])
    N = int(B.shape[1])
    if out is None:
        out = torch.empty((M, N), dtype=A.dtype, device="cpu")
    A_c = A.detach().contiguous()
    B_c = B.detach().contiguous()
    out_c = out.detach().contiguous()

    bA = ctx.buffer_from_array(A_c.numpy())
    bB = ctx.buffer_from_array(B_c.numpy())
    bC = ctx.buffer_from_array(out_c.numpy()) if beta != 0.0 else _tc.Buffer(
        ctx, nbytes=M * N * out_c.element_size())

    _tc.sparse_24_gemm(ctx, bA, bB, bC, M, N, K,
                        _dtype_code(A), _dtype_code(B), _dtype_code(out),
                        alpha, beta)

    import numpy as np
    np_dtype = {torch.float32: np.float32, torch.float16: np.float16,
                torch.bfloat16: np.float32}[out.dtype]
    arr = bC.to_numpy((M, N), np_dtype)
    return torch.from_numpy(arr).clone().to(out.dtype)


def sparse_linear(x: torch.Tensor, weight: torch.Tensor,
                   bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Drop-in for nn.Linear when `weight` is 2:4-pruned.

    Equivalent to F.linear(x, weight, bias) = x @ weight.T + bias.
    Internally transposes weight so the K axis is sparse-aligned per
    the 2:4 cusparseLt contract.
    """
    if weight.dim() != 2:
        raise ValueError("sparse_linear: weight must be 2-D")
    # F.linear: y = x @ W.T → with W [out, in], W.T is [in, out].
    W_T = weight.t().contiguous()
    y = sparse_gemm(x, W_T)
    if bias is not None:
        y = y + bias
    return y


__all__ = [
    "tensor_core_available",
    "prune_2_4",
    "is_2_4",
    "sparse_gemm",
    "sparse_linear",
]
