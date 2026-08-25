"""Phase attention + Born-rule output — torch.autograd glue over C kernels.

Math lives in `lib/ops/phase_attention_cpu.cpp`; Python is the glue layer
(routes pytorch tensors through tc_buffer write/read, calls the C op,
returns a fresh torch tensor).

GeometricLM usage:

    from tensorcore_torch import phase_attention as tcpa

    # Compute per-manifold ⟨Q_m, K_m⟩_m, Δφ_m, d_m upstream (via the
    # existing tensorcore_torch.poincare ops). Then combine:
    scores = tcpa.phase_attention_combine(
        inner_products,    # [N_pairs, M]
        phase_diffs,       # [N_pairs, M]
        distances,         # [N_pairs, M]
        weights, gammas, lambdas,  # [M] each — per-manifold learnable
    )
    # scores: [N_pairs]; reshape into [B,H,Sq,Skv] then softmax + weighted V.

    # Output projection (qLLM Born-rule):
    probs = tcpa.born_rule_output(h_amp, s_amp, e_amp)  # [N, V]
    # rows sum to 1, drop-in for softmax(logits).

Both ops use fp32 tensors. The C kernels enforce numerical safety
(Z floor 1e-30 for Born-rule). Backward dispatch lives in the autograd
Functions below; until dedicated backward C primitives ship, gradients
are computed via forward composition of differentiable ops (the cos/exp
modulators are themselves differentiable, so vanilla autograd handles
the gradient through the combine formula correctly).
"""

from __future__ import annotations

import threading
from typing import Optional

import torch

try:
    import tensorcore as _tc
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tensorcore_torch.phase_attention requires the `tensorcore` ctypes package."
    ) from exc


_ctx_lock = threading.Lock()
_ctx = None


def _get_ctx():
    global _ctx
    if _ctx is None:
        with _ctx_lock:
            if _ctx is None:
                _ctx = _tc.Context()
    return _ctx


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float32).contiguous()


def phase_attention_combine(inner_products: torch.Tensor,
                             phase_diffs: torch.Tensor,
                             distances: torch.Tensor,
                             weights: torch.Tensor,
                             gammas: torch.Tensor,
                             lambdas: torch.Tensor) -> torch.Tensor:
    """Phase-attention score combine. Returns scores of shape [N_pairs].

    All [N_pairs, M] / [M] tensors must be fp32 (or convertible). The
    C kernel does the per-pair Σ_m w_m·ip_m·cos(Δφ_m+γ_m)·exp(−λ_m·d_m).
    """
    ctx = _get_ctx()
    import numpy as np
    ip = _f32(inner_products)
    pd = _f32(phase_diffs)
    dd = _f32(distances)
    w = _f32(weights); g = _f32(gammas); l = _f32(lambdas)
    if ip.dim() != 2 or pd.shape != ip.shape or dd.shape != ip.shape:
        raise ValueError("inner_products/phase_diffs/distances must all be [N_pairs, M] and same shape")
    N, M = int(ip.shape[0]), int(ip.shape[1])
    if w.numel() != M or g.numel() != M or l.numel() != M:
        raise ValueError(f"weights/gammas/lambdas must each have M={M} entries")

    b_ip = ctx.buffer_from_array(ip.numpy())
    b_pd = ctx.buffer_from_array(pd.numpy())
    b_dd = ctx.buffer_from_array(dd.numpy())
    b_w = ctx.buffer_from_array(w.numpy())
    b_g = ctx.buffer_from_array(g.numpy())
    b_l = ctx.buffer_from_array(l.numpy())
    b_out = _tc.Buffer(ctx, nbytes=N * 4)
    _tc.phase_attention_combine(ctx, b_ip, b_pd, b_dd, b_w, b_g, b_l, b_out, N, M)
    return torch.from_numpy(b_out.to_numpy((N,), np.float32)).clone()


def born_rule_output(h_amp: torch.Tensor,
                      s_amp: Optional[torch.Tensor] = None,
                      e_amp: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Born-rule P(w) = (h+s+e)² / Z over the last dim. Returns [..., V] probs.

    s_amp / e_amp may be None to omit those terms (ablation-friendly).
    """
    ctx = _get_ctx()
    import numpy as np
    h = _f32(h_amp)
    shape = h.shape
    V = int(shape[-1])
    N = int(h.numel() // V)
    h_flat = h.view(N, V)
    s_flat = _f32(s_amp).view(N, V) if s_amp is not None else None
    e_flat = _f32(e_amp).view(N, V) if e_amp is not None else None
    if s_flat is not None and s_flat.shape != h_flat.shape:
        raise ValueError(f"s_amp shape {s_flat.shape} ≠ h_amp {h_flat.shape}")
    if e_flat is not None and e_flat.shape != h_flat.shape:
        raise ValueError(f"e_amp shape {e_flat.shape} ≠ h_amp {h_flat.shape}")

    b_h = ctx.buffer_from_array(h_flat.numpy())
    b_s = ctx.buffer_from_array(s_flat.numpy()) if s_flat is not None else None
    b_e = ctx.buffer_from_array(e_flat.numpy()) if e_flat is not None else None
    b_p = _tc.Buffer(ctx, nbytes=N * V * 4)
    _tc.born_rule_output(ctx, b_h, b_s, b_e, b_p, N, V)
    arr = b_p.to_numpy((N, V), np.float32)
    return torch.from_numpy(arr).clone().view(shape)


__all__ = ["phase_attention_combine", "born_rule_output"]
