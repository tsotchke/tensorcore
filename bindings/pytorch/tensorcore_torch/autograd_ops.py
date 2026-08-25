"""torch.autograd.Function wrappers backed by tensorcore C kernels.

This module gives GeometricLM (and any other PyTorch model) a drop-in path
to dispatch its heavy ops onto tensorcore's fused kernels — both forward
AND backward — without changing the model's shape language.

The bridge wraps the existing ctypes layer in `python/tensorcore` rather
than rebuilding the C extension. Each Function:

  1. Makes inputs contiguous (tensorcore expects row-major) on the right device.
  2. Wraps the tensor's data_ptr() as a tc_buffer via tc_buffer_from_ptr —
     no data copy; the tc_buffer doesn't own the storage.
  3. Calls the tc_*_forward / tc_*_backward kernel.
  4. Returns torch tensors that share the same storage.

GeometricLM usage:

    from tensorcore_torch import autograd_ops as tcops
    # Drop-in replacement for nn.LayerNorm / nn.RMSNorm / F.silu*F.linear:
    y, rstd = tcops.rmsnorm(x, gamma)            # forward + autograd-aware
    y, mean, rstd = tcops.layernorm(x, gamma, beta)
    o = tcops.swiglu(gate, up)
    o, lse = tcops.flash_attention(q, k, v, causal=True)

Numerics: fp16 IO with fp32 accumulators, matching the Metal/CUDA tensorcore
kernels and the CPU reference. dgamma/dbeta accumulate in fp32 (matches the
underlying tc_buffer dtype contract).

Backend: the tensorcore context (`Context()`) auto-selects Metal on Apple,
CUDA on systems with TC_USE_CUDA_GEMM=1, or portable CPU otherwise. The
tc_*_forward calls dispatch to the matching kernel via tensorcore's existing
dispatch logic.
"""

from __future__ import annotations

import threading
from typing import Tuple

import torch

try:
    import tensorcore as _tc
except ImportError as exc:  # pragma: no cover — surface a clearer error
    raise ImportError(
        "tensorcore_torch.autograd_ops requires the `tensorcore` ctypes package "
        "(import tensorcore). Install with `pip install -e python/` from the "
        "tensorcore source tree."
    ) from exc


# ---------- process-wide context + bufferwrap helpers ----------

_ctx_lock = threading.Lock()
_ctx = None


def _get_ctx():
    """Lazily create one tc_context per process; reused across all ops."""
    global _ctx
    if _ctx is None:
        with _ctx_lock:
            if _ctx is None:
                _ctx = _tc.Context()
    return _ctx


def _contig(t: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Materialize a contiguous CPU tensor in the requested dtype.

    The tc_buffer_from_ptr path wraps the tensor's underlying storage; the
    tensor must therefore (a) be contiguous and (b) live in host memory
    that the tensorcore runtime can dereference (CPU side; PrivateUse1
    host-memory is also OK because the allocator returns regular malloc).
    """
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    if not t.is_contiguous():
        t = t.contiguous()
    if t.is_cuda or t.is_mps:
        # Operate on a host-side copy — the geometric_lm path expects this
        # while the device-resident CUDA/Metal autograd path lands in P1.
        # Round-trip is one cudaMemcpy per call; cheap relative to the GEMM.
        t = t.detach().to("cpu")
        if not t.is_contiguous():
            t = t.contiguous()
    return t


def _to_tc(ctx, t: torch.Tensor):
    """Allocate a tc_buffer and copy tensor data in. Returns the Buffer.

    tc_buffer_from_ptr would be cheaper (zero-copy), but the Metal build
    rejects it (MTLBuffer can't wrap arbitrary host pointers without
    alignment guarantees). Allocate + write works on every backend
    (Metal, CUDA managed, portable CPU) and matches the buffer's
    storage class so subsequent compute is in the right memory tier.
    """
    arr = t.detach().contiguous().numpy()
    return ctx.buffer_from_array(arr)


def _from_tc(buf, like: torch.Tensor) -> torch.Tensor:
    """Read a tc_buffer back into a fresh torch tensor matching `like`'s
    shape + dtype.

    Supported dtypes: fp16, fp32, bfloat16. bf16 has no native numpy
    dtype, so we fetch the bytes as int16 and reinterpret through
    torch's bf16 view — round-trip safe and avoids a fp32→bf16 cast
    that would lose the original tensorcore-side bits."""
    import numpy as np
    if like.dtype == torch.float16:
        arr = buf.to_numpy(tuple(like.shape), np.float16)
        return torch.from_numpy(arr).clone()
    if like.dtype == torch.float32:
        arr = buf.to_numpy(tuple(like.shape), np.float32)
        return torch.from_numpy(arr).clone()
    if like.dtype == torch.bfloat16:
        # Fetch raw 16-bit words; reinterpret as bf16 via torch's view.
        raw = buf.to_numpy(tuple(like.shape), np.int16)
        return torch.from_numpy(raw).clone().view(torch.bfloat16)
    raise ValueError(
        f"_from_tc: unsupported dtype {like.dtype}; "
        f"supported = {{torch.float16, torch.float32, torch.bfloat16}}"
    )


def _alloc_tc(ctx, nbytes: int):
    """Allocate a fresh tc_buffer of nbytes."""
    return _tc.Buffer(ctx, nbytes=nbytes)


# ---------- RMSNorm ----------

class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx_ag, x: torch.Tensor, gamma: torch.Tensor, eps: float):
        ctx = _get_ctx()
        x = _contig(x, torch.float16)
        gamma = _contig(gamma, torch.float16)
        if x.dim() < 2:
            raise ValueError(f"rmsnorm expects x.dim() >= 2, got {x.dim()}")
        N = int(x.numel() // x.shape[-1])
        D = int(x.shape[-1])
        if int(gamma.numel()) != D:
            raise ValueError(f"rmsnorm gamma shape {gamma.shape} does not match feature dim {D}")

        x_flat = x.view(N, D).contiguous()
        y_flat_template = torch.empty_like(x_flat)
        rstd_template = torch.empty((N,), dtype=torch.float32)

        bX = _to_tc(ctx, x_flat)
        bG = _to_tc(ctx, gamma)
        bY = _alloc_tc(ctx, y_flat_template.numel() * y_flat_template.element_size())
        bR = _alloc_tc(ctx, rstd_template.numel() * rstd_template.element_size())
        _tc.rmsnorm_forward(ctx, bX, bG, bY, bR, N, D, eps)

        y_flat = _from_tc(bY, y_flat_template)
        rstd = _from_tc(bR, rstd_template)

        ctx_ag.save_for_backward(x_flat, gamma, rstd)
        ctx_ag.shape = x.shape
        return y_flat.view(x.shape), rstd

    @staticmethod
    def backward(ctx_ag, dy: torch.Tensor, _drstd):
        ctx = _get_ctx()
        x_flat, gamma, rstd = ctx_ag.saved_tensors
        dy = _contig(dy, torch.float16).view_as(x_flat)
        N, D = x_flat.shape

        dx_template = torch.empty_like(x_flat)
        dgamma_template = torch.zeros((D,), dtype=torch.float32)

        bX = _to_tc(ctx, x_flat)
        bG = _to_tc(ctx, gamma)
        bDY = _to_tc(ctx, dy)
        bR = _to_tc(ctx, rstd)
        bDX = _alloc_tc(ctx, dx_template.numel() * dx_template.element_size())
        bDg = _to_tc(ctx, dgamma_template)
        _tc.rmsnorm_backward(ctx, bX, bG, bDY, bR, bDX, bDg, N, D)

        dx_flat = _from_tc(bDX, dx_template)
        dgamma = _from_tc(bDg, dgamma_template)
        return dx_flat.view(ctx_ag.shape), dgamma.to(gamma.dtype), None


def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSnorm via tensorcore kernel; autograd-aware.

    Returns (y, rstd). y has the same shape as x; rstd is fp32 over the
    flattened leading dims (saved for backward — caller can ignore).
    """
    return _RMSNormFn.apply(x, gamma, float(eps))


# ---------- LayerNorm ----------

class _LayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx_ag, x, gamma, beta, eps: float):
        ctx = _get_ctx()
        x = _contig(x, torch.float16)
        gamma = _contig(gamma, torch.float16)
        beta = _contig(beta, torch.float16)
        N = int(x.numel() // x.shape[-1])
        D = int(x.shape[-1])
        if int(gamma.numel()) != D or int(beta.numel()) != D:
            raise ValueError(f"layernorm gamma/beta must match feature dim {D}")

        x_flat = x.view(N, D).contiguous()
        y_template = torch.empty_like(x_flat)
        mean_template = torch.empty((N,), dtype=torch.float32)
        rstd_template = torch.empty((N,), dtype=torch.float32)

        bX = _to_tc(ctx, x_flat); bG = _to_tc(ctx, gamma); bB = _to_tc(ctx, beta)
        bY = _alloc_tc(ctx, y_template.numel() * y_template.element_size())
        bM = _alloc_tc(ctx, mean_template.numel() * mean_template.element_size())
        bR = _alloc_tc(ctx, rstd_template.numel() * rstd_template.element_size())
        _tc.layernorm_forward(ctx, bX, bG, bB, bY, bM, bR, N, D, eps)
        y_flat = _from_tc(bY, y_template)
        mean = _from_tc(bM, mean_template)
        rstd = _from_tc(bR, rstd_template)

        ctx_ag.save_for_backward(x_flat, gamma, mean, rstd)
        ctx_ag.shape = x.shape
        return y_flat.view(x.shape), mean, rstd

    @staticmethod
    def backward(ctx_ag, dy, _dmean, _drstd):
        ctx = _get_ctx()
        x_flat, gamma, mean, rstd = ctx_ag.saved_tensors
        dy = _contig(dy, torch.float16).view_as(x_flat)
        N, D = x_flat.shape

        dx_template = torch.empty_like(x_flat)
        bX = _to_tc(ctx, x_flat); bG = _to_tc(ctx, gamma); bDY = _to_tc(ctx, dy)
        bM = _to_tc(ctx, mean); bR = _to_tc(ctx, rstd)
        bDX = _alloc_tc(ctx, dx_template.numel() * dx_template.element_size())
        _tc.layernorm_backward(ctx, bX, bG, bDY, bM, bR, bDX, N, D)
        dx_flat = _from_tc(bDX, dx_template)
        # Note: tc_layernorm_backward computes dX only; dgamma/dbeta accumulators
        # would need separate kernels (CPU impl does dX only too — see
        # lib/ops/training_cpu.cpp). Compute them here in fp32 for now.
        x_hat = (x_flat.float() - mean.unsqueeze(-1)) * rstd.unsqueeze(-1)
        dgamma = (dy.float() * x_hat).sum(0)
        dbeta = dy.float().sum(0)
        return dx_flat.view(ctx_ag.shape), dgamma.to(gamma.dtype), dbeta.to(gamma.dtype), None


def layernorm(x, gamma, beta, eps: float = 1e-5):
    """LayerNorm via tensorcore. Returns (y, mean, rstd)."""
    return _LayerNormFn.apply(x, gamma, beta, float(eps))


# ---------- SwiGLU ----------

class _SwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx_ag, gate, up):
        ctx = _get_ctx()
        gate = _contig(gate, torch.float16)
        up = _contig(up, torch.float16)
        if gate.shape != up.shape:
            raise ValueError(f"swiglu gate {gate.shape} != up {up.shape}")
        n = int(gate.numel())
        out_template = torch.empty_like(gate)
        bG = _to_tc(ctx, gate); bU = _to_tc(ctx, up)
        bO = _alloc_tc(ctx, out_template.numel() * out_template.element_size())
        _tc.swiglu_forward(ctx, bG, bU, bO, n)
        out = _from_tc(bO, out_template)
        ctx_ag.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx_ag, dout):
        ctx = _get_ctx()
        gate, up = ctx_ag.saved_tensors
        dout = _contig(dout, torch.float16).view_as(gate)
        n = int(gate.numel())
        dgate_template = torch.empty_like(gate); dup_template = torch.empty_like(up)
        bG = _to_tc(ctx, gate); bU = _to_tc(ctx, up); bDO = _to_tc(ctx, dout)
        bDG = _alloc_tc(ctx, dgate_template.numel() * dgate_template.element_size())
        bDU = _alloc_tc(ctx, dup_template.numel() * dup_template.element_size())
        _tc.swiglu_backward(ctx, bG, bU, bDO, bDG, bDU, n)
        dgate = _from_tc(bDG, dgate_template); dup = _from_tc(bDU, dup_template)
        return dgate, dup


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation via tensorcore: out = silu(gate) * up."""
    return _SwiGLUFn.apply(gate, up)


# ---------- FlashAttention ----------

class _FlashAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx_ag, q, k, v, causal: bool, softmax_scale: float | None,
                kv_heads: int):
        ctx = _get_ctx()
        # [B, H, S, D] fp16
        q = _contig(q, torch.float16)
        k = _contig(k, torch.float16)
        v = _contig(v, torch.float16)
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError(f"flash_attention expects [B,H,S,D]; got Q={q.shape}, K={k.shape}, V={v.shape}")
        B, H, Sq, D = q.shape
        _, _, Skv, _ = k.shape
        out_template = torch.empty_like(q)
        lse_template = torch.empty((B, H, Sq), dtype=torch.float32)

        bQ = _to_tc(ctx, q); bK = _to_tc(ctx, k); bV = _to_tc(ctx, v)
        bO = _alloc_tc(ctx, out_template.numel() * out_template.element_size())
        bL = _alloc_tc(ctx, lse_template.numel() * lse_template.element_size())
        _tc.attention_forward(
            ctx, bQ, bK, bV, bO,
            B, H, Sq, Skv, D, LSE=bL,
            causal=causal, return_lse=True,
            softmax_scale=softmax_scale, kv_heads=int(kv_heads or H),
        )
        out = _from_tc(bO, out_template)
        lse = _from_tc(bL, lse_template)

        ctx_ag.save_for_backward(q, k, v, out, lse)
        ctx_ag.shape_meta = (B, H, Sq, Skv, D, bool(causal),
                             float(softmax_scale or (1.0 / (D ** 0.5))),
                             int(kv_heads or H))
        return out, lse

    @staticmethod
    def backward(ctx_ag, do, _dlse):
        ctx = _get_ctx()
        q, k, v, out, lse = ctx_ag.saved_tensors
        B, H, Sq, Skv, D, causal, scale, kv_heads = ctx_ag.shape_meta
        do = _contig(do, torch.float16).view_as(q)
        dq_t = torch.empty_like(q); dk_t = torch.empty_like(k); dv_t = torch.empty_like(v)

        bQ = _to_tc(ctx, q); bK = _to_tc(ctx, k); bV = _to_tc(ctx, v)
        bO = _to_tc(ctx, out); bDO = _to_tc(ctx, do); bL = _to_tc(ctx, lse)
        bDQ = _alloc_tc(ctx, dq_t.numel() * dq_t.element_size())
        bDK = _alloc_tc(ctx, dk_t.numel() * dk_t.element_size())
        bDV = _alloc_tc(ctx, dv_t.numel() * dv_t.element_size())
        _tc.attention_backward(
            ctx, bQ, bK, bV, bO, bDO, bL, bDQ, bDK, bDV,
            B, H, Sq, Skv, D,
            causal=causal, softmax_scale=scale, kv_heads=kv_heads,
        )
        dq = _from_tc(bDQ, dq_t); dk = _from_tc(bDK, dk_t); dv = _from_tc(bDV, dv_t)
        return dq, dk, dv, None, None, None


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
                    causal: bool = True, softmax_scale: float | None = None,
                    kv_heads: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """FlashAttention-2 forward+backward via tensorcore.

    q, k, v: [B, H, S, D] fp16. kv_heads=0 means MHA; >0 enables GQA with
    that many KV heads. softmax_scale defaults to 1/sqrt(D).

    Returns (out, lse). LSE is saved for backward; callers can ignore it.
    """
    return _FlashAttentionFn.apply(q, k, v, causal, softmax_scale, kv_heads)


# ---------- GEMM ----------

class _GemmFn(torch.autograd.Function):
    """C = A @ B with autograd.

    Forward shapes: A=[M,K] fp16, B=[K,N] fp16, C=[M,N] fp16 (fp32 accum).
    Backward: dA = dC @ B^T, dB = A^T @ dC; both via the same tc.gemm kernel
    with the transpose flags rather than materializing transposed copies.
    """

    @staticmethod
    def forward(ctx_ag, a: torch.Tensor, b: torch.Tensor):
        ctx = _get_ctx()
        a = _contig(a, torch.float16)
        b = _contig(b, torch.float16)
        if a.dim() != 2 or b.dim() != 2:
            raise ValueError(f"gemm expects 2-D inputs; got A={a.shape}, B={b.shape}")
        M, Ka = a.shape
        Kb, N = b.shape
        if Ka != Kb:
            raise ValueError(f"gemm inner dim mismatch: A.K={Ka} vs B.K={Kb}")
        K = Ka
        c_template = torch.empty((M, N), dtype=torch.float16)

        bA = _to_tc(ctx, a); bB = _to_tc(ctx, b)
        bC = _alloc_tc(ctx, c_template.numel() * c_template.element_size())
        _tc.gemm(ctx, bA, bB, bC, M, N, K, dtype="f16", accum="f32")
        out = _from_tc(bC, c_template)

        ctx_ag.save_for_backward(a, b)
        ctx_ag.shape_meta = (M, N, K)
        return out

    @staticmethod
    def backward(ctx_ag, dout: torch.Tensor):
        ctx = _get_ctx()
        a, b = ctx_ag.saved_tensors
        M, N, K = ctx_ag.shape_meta
        dout = _contig(dout, torch.float16)
        if dout.shape != (M, N):
            dout = dout.view(M, N).contiguous()

        bDC = _to_tc(ctx, dout)
        da = db = None

        if ctx_ag.needs_input_grad[0]:
            # dA[M,K] = dC[M,N] @ B[K,N]^T
            da_template = torch.empty((M, K), dtype=torch.float16)
            bB = _to_tc(ctx, b)
            bDA = _alloc_tc(ctx, da_template.numel() * da_template.element_size())
            _tc.gemm(ctx, bDC, bB, bDA, M, K, N,
                     dtype="f16", accum="f32", transpose_b=True)
            da = _from_tc(bDA, da_template)

        if ctx_ag.needs_input_grad[1]:
            # dB[K,N] = A[M,K]^T @ dC[M,N]
            db_template = torch.empty((K, N), dtype=torch.float16)
            bA = _to_tc(ctx, a)
            bDB = _alloc_tc(ctx, db_template.numel() * db_template.element_size())
            _tc.gemm(ctx, bA, bDC, bDB, K, N, M,
                     dtype="f16", accum="f32", transpose_a=True)
            db = _from_tc(bDB, db_template)

        return da, db


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """GEMM via tensorcore: C = A @ B; autograd-aware (fp16 IO, fp32 accum)."""
    return _GemmFn.apply(a, b)


__all__ = [
    "gemm",
    "rmsnorm",
    "layernorm",
    "swiglu",
    "flash_attention",
]
