"""RiemannianAdam optimizer for PyTorch, backed by tensorcore C kernels.

Drop-in for torch.optim.Adam / AdamW when the parameter lives on a
Riemannian manifold (Poincaré ball, unit sphere) or just R^D. Matches
the GeoRefine `geometric_lm.py` RiemannianAdam: tangent-space gradient
projection + exp-map retraction + first-moment parallel transport,
per-manifold.

Manifold tag is stored in param.tc_manifold (set by the model when the
parameter is constructed) and read by the optimizer step:

    p = nn.Parameter(torch.randn(N, D) * 0.01)
    p.tc_manifold = ("poincare", curvature_c)   # or ("sphere",) or ("euclidean",)

    optimizer = RiemannianAdam([p], lr=1e-3, betas=(0.9, 0.999),
                               weight_decay=0.01)
    optimizer.step()

Each step copies fp32 tensor data into a tc_buffer, runs the fused C
kernel (tc_riemannian_adam_step_*), and copies the new params + moments
back. Once GeometricLM moves to tensorcore-allocated buffers, this glue
disappears (the C op can run in-place on the persistent buffer).

The "math lives in C, Python is glue" rule applies; nothing here
re-implements the manifold formulas.
"""

from __future__ import annotations

import threading
from typing import Iterable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer

try:
    import tensorcore as _tc
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tensorcore_torch.riemannian_adam requires the `tensorcore` ctypes package."
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


def _flat(t: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Reshape t to (N, D) as a host-resident fp32 view. Last dim is the
    manifold dim.

    The `.cpu()` is load-bearing, not defensive. `ctx.buffer_from_array`
    takes a numpy array, and `Tensor.numpy()` raises

        TypeError: can't convert cuda:0 device type tensor to numpy

    so without it every CUDA parameter throws before reaching a kernel --
    which is exactly what happened to the `tcr` arms of the geometry A/B.
    It is a no-op for tensors already on the host.
    """
    t = t.detach().to(torch.float32).contiguous().cpu()
    if t.dim() < 1:
        raise ValueError("RiemannianAdam params must have at least 1 dim")
    D = int(t.shape[-1])
    N = int(t.numel() // D)
    return t.view(N, D), N, D


class RiemannianAdam(Optimizer):
    """Per-parameter manifold-aware Adam, dispatched into tensorcore.

    The manifold for each param is taken from `param.tc_manifold`:
        ("poincare", c)   — Poincaré ball with curvature c (default c=1)
        ("sphere",)        — unit sphere (each row of param is unit-norm)
        ("euclidean",)     — plain R^D / AdamW
    Params without `tc_manifold` are treated as Euclidean (matches Adam).
    """

    def __init__(self, params: Iterable, lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        if lr <= 0.0:
            raise ValueError(f"invalid lr: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ctx = _get_ctx()
        import numpy as np

        for group in self.param_groups:
            lr = group["lr"]; b1, b2 = group["betas"]
            eps = group["eps"]; wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1
                t_step = state["step"]
                bc1 = 1.0 - b1 ** t_step
                bc2 = 1.0 - b2 ** t_step

                manifold = getattr(p, "tc_manifold", ("euclidean",))
                kind = manifold[0]

                # Materialize host-side fp32 copies for the C kernel. Every
                # one of these must be on the host: buffer_from_array goes
                # through numpy, which refuses CUDA tensors.
                params_flat, N, D = _flat(p)
                grad_flat = p.grad.detach().to(torch.float32).contiguous().cpu().view(N, D)
                m_flat = state["exp_avg"].detach().cpu().view(N, D)
                v_flat = state["exp_avg_sq"].detach().cpu().view(N, D)

                b_p = ctx.buffer_from_array(params_flat.numpy())
                b_g = ctx.buffer_from_array(grad_flat.numpy())
                b_m = ctx.buffer_from_array(m_flat.numpy())
                b_v = ctx.buffer_from_array(v_flat.numpy())

                if kind == "poincare":
                    c = float(manifold[1]) if len(manifold) > 1 else 1.0
                    _tc.riemannian_adam_step_poincare(
                        ctx, b_p, b_g, b_m, b_v, N, D, c,
                        lr, b1, b2, eps, wd, bc1, bc2)
                elif kind == "sphere":
                    _tc.riemannian_adam_step_sphere(
                        ctx, b_p, b_g, b_m, b_v, N, D,
                        lr, b1, b2, eps, wd, bc1, bc2)
                elif kind == "euclidean":
                    _tc.riemannian_adam_step_euclidean(
                        ctx, b_p, b_g, b_m, b_v, N, D,
                        lr, b1, b2, eps, wd, bc1, bc2)
                else:
                    raise ValueError(f"unknown tc_manifold kind: {kind!r}")

                # Copy back into the existing tensors (in-place).
                new_p = torch.from_numpy(b_p.to_numpy((N, D), np.float32))
                new_m = torch.from_numpy(b_m.to_numpy((N, D), np.float32))
                new_v = torch.from_numpy(b_v.to_numpy((N, D), np.float32))
                p.copy_(new_p.view_as(p).to(p.dtype))
                state["exp_avg"].copy_(new_m.view_as(state["exp_avg"]))
                state["exp_avg_sq"].copy_(new_v.view_as(state["exp_avg_sq"]))

        return loss


__all__ = ["RiemannianAdam"]
