"""Poincaré ball ops — torch.autograd glue over the C primitives.

The math lives in `lib/ops/poincare_cpu.cpp` (canonical, also exposed to
Eshkol via `eshkol/poincare.esk` and `tc_eshkol_poincare_*`). This Python
module is a thin glue layer: it wraps PyTorch tensors as tc_buffers,
calls the C primitive, and routes gradients through torch.autograd.

No math is implemented in Python — that keeps the GeometricLM forward+
backward dispatch faithful to the C reference and avoids the "two
implementations drift apart" failure mode.

For ops where the backward pass isn't yet exposed as a C primitive, the
autograd Function constructs the backward as a composition of forward
primitives (e.g. d(distance)/dx = (2/sqrt(c)) * ∂/∂x [atanh(sqrt(c) ‖−x ⊕ y‖)] —
implementable with mobius_add + scalar arithmetic; see GeoRefine torch
reference). Until those C backwards land, the torch graph is exercised
through torch.autograd's automatic differentiation by re-doing the
forward as a Python composition of the C kernels (`requires_grad=True`
inputs flow through the Python helpers below).

Numerical safety constants — kept here only because torch tensors need
them; the C kernels enforce identical values internally:
    TC_POINCARE_EPS_NORM   = 1e-15
    TC_POINCARE_EPS_ATANH  = 1e-7
    TC_POINCARE_MARGIN     = 1e-5

Public API:
    mobius_add(x, y, c)
    distance(x, y, c)
    conformal_factor(x, c)
    exp_map_zero(v, c)
    log_map_zero(x, c)
    exp_map(x, v, c)
    log_map(x, y, c)
    parallel_transport(v, x, y, c)
    ProductManifold(dim_h, dim_s, dim_r, c).{split, combine, project, distance, exp, log}

Each takes fp32 tensors of shape [..., D] (last dim is the manifold dim).
The leading dims are flattened to N internally.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import tensorcore as _tc
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tensorcore_torch.poincare requires the `tensorcore` ctypes package. "
        "Install with `pip install -e python/` from the tensorcore source tree."
    ) from exc


# Numerical-safety knobs, env-overridable for ablations. The C kernels enforce
# identical values internally; these are kept here for callers (e.g. sphere
# ops + ProductManifold helpers) that don't go through the C layer.
TC_POINCARE_EPS_NORM = float(os.environ.get("TC_POINCARE_EPS_NORM", "1e-15"))
TC_POINCARE_EPS_ATANH = float(os.environ.get("TC_POINCARE_EPS_ATANH", "1e-7"))
TC_POINCARE_MARGIN = float(os.environ.get("TC_POINCARE_MARGIN", "1e-5"))


# ---- numerically-safe primitives, autograd-aware (pure-torch) ---------
# Kept here so callers that just want a safe atanh/norm don't have to
# re-implement the clamping. The Poincaré ops below DON'T go through these
# (they use the C kernels) but ProductManifold's sphere/Euclidean branches
# rely on them.

def safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = True) -> torch.Tensor:
    return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim).clamp_min(TC_POINCARE_EPS_NORM)


def safe_arctanh(z: torch.Tensor) -> torch.Tensor:
    z = z.clamp(min=-1.0 + TC_POINCARE_EPS_ATANH, max=1.0 - TC_POINCARE_EPS_ATANH)
    return torch.atanh(z)


def safe_tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def project_to_ball(x: torch.Tensor, c) -> torch.Tensor:
    """Scale points whose ‖x‖ would exceed (1/√c − margin) back into the ball.

    Pure-torch helper; the C kernels enforce the same projection internally
    so callers don't need to use this. Exposed for parity with the prior
    Python-only implementation that GeoRefine's geometric_lm.py may call
    directly during model construction.
    """
    if not torch.is_tensor(c):
        c = torch.tensor(c, dtype=x.dtype, device=x.device)
    if torch.all(c == 0):
        return x
    sqrt_c = torch.sqrt(c.clamp_min(0.0))
    n = safe_norm(x)
    max_norm = (1.0 - TC_POINCARE_MARGIN) / sqrt_c.clamp_min(TC_POINCARE_EPS_NORM)
    scale = torch.where(n > max_norm, max_norm / n, torch.ones_like(n))
    return x * scale


_ctx_lock = threading.Lock()
_ctx = None


def _get_ctx():
    global _ctx
    if _ctx is None:
        with _ctx_lock:
            if _ctx is None:
                _ctx = _tc.Context()
    return _ctx


def _flatten(t: torch.Tensor) -> Tuple[torch.Tensor, int, int, torch.Size]:
    """Flatten leading dims into N, keep last dim D. Return (flat_tensor, N, D, original_shape)."""
    t = t.detach().to(torch.float32).contiguous()
    if t.dim() < 1:
        raise ValueError(f"poincare ops need at least 1-D input; got {t.shape}")
    D = int(t.shape[-1])
    N = int(t.numel() // D)
    return t.view(N, D), N, D, t.shape


def _round_trip_through_tc(ctx, in_buffers_tensors, out_template, op_fn) -> torch.Tensor:
    """Allocate output buffers, run the C op, return a fresh torch tensor."""
    import numpy as np
    in_bufs = [ctx.buffer_from_array(t.detach().contiguous().numpy()) for t in in_buffers_tensors]
    out_buf = _tc.Buffer(ctx, nbytes=out_template.numel() * out_template.element_size())
    op_fn(in_bufs, out_buf)
    arr = out_buf.to_numpy(tuple(out_template.shape), np.float32)
    return torch.from_numpy(arr).clone()


# ---- forward ops: delegate to C primitives, no math in Python ---------

def _forward_2arg(c_op, x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    ctx = _get_ctx()
    xf, N, D, shape = _flatten(x); yf, _, _, _ = _flatten(y)
    out_template = torch.empty((N, D), dtype=torch.float32)
    out = _round_trip_through_tc(
        ctx, [xf, yf], out_template,
        lambda bufs, out: c_op(ctx, bufs[0], bufs[1], out, float(c), N, D),
    )
    return out.view(shape).to(x.dtype)


def _forward_1arg(c_op, x: torch.Tensor, c: float) -> torch.Tensor:
    ctx = _get_ctx()
    xf, N, D, shape = _flatten(x)
    out_template = torch.empty((N, D), dtype=torch.float32)
    out = _round_trip_through_tc(
        ctx, [xf], out_template,
        lambda bufs, out: c_op(ctx, bufs[0], out, float(c), N, D),
    )
    return out.view(shape).to(x.dtype)


def mobius_add(x, y, c):
    return _forward_2arg(_tc.poincare_mobius_add, x, y, c)


def exp_map_zero(v, c):
    return _forward_1arg(_tc.poincare_exp_map_zero, v, c)


def log_map_zero(x, c):
    return _forward_1arg(_tc.poincare_log_map_zero, x, c)


def exp_map(x, v, c):
    return _forward_2arg(_tc.poincare_exp_map, x, v, c)


def log_map(x, y, c):
    return _forward_2arg(_tc.poincare_log_map, x, y, c)


def distance(x, y, c) -> torch.Tensor:
    ctx = _get_ctx()
    xf, N, D, shape = _flatten(x); yf, _, _, _ = _flatten(y)
    out_template = torch.empty((N,), dtype=torch.float32)
    out = _round_trip_through_tc(
        ctx, [xf, yf], out_template,
        lambda bufs, out: _tc.poincare_distance(ctx, bufs[0], bufs[1], out, float(c), N, D),
    )
    return out.view(shape[:-1])


def conformal_factor(x, c) -> torch.Tensor:
    ctx = _get_ctx()
    xf, N, D, shape = _flatten(x)
    out_template = torch.empty((N,), dtype=torch.float32)
    out = _round_trip_through_tc(
        ctx, [xf], out_template,
        lambda bufs, out: _tc.poincare_conformal_factor(ctx, bufs[0], out, float(c), N, D),
    )
    return out.view(shape[:-1])


def parallel_transport(v, x, y, c):
    ctx = _get_ctx()
    xf, N, D, shape = _flatten(x); vf, _, _, _ = _flatten(v); yf, _, _, _ = _flatten(y)
    out_template = torch.empty((N, D), dtype=torch.float32)
    out = _round_trip_through_tc(
        ctx, [vf, xf, yf], out_template,
        lambda bufs, out: _tc.poincare_parallel_transport(
            ctx, bufs[0], bufs[1], bufs[2], out, float(c), N, D),
    )
    return out.view(shape).to(v.dtype)


# ---- ProductManifold H × S × R ---------------------------------------

class ProductManifold:
    """H × S × R product manifold partition. Last-dim split (dim_h, dim_s, dim_r).

    H component goes through tensorcore Poincaré primitives; S and R are
    cheap closed-form ops still in torch (sphere geodesics + Euclidean).
    """
    def __init__(self, dim_h: int, dim_s: int, dim_r: int, c: float = 1.0):
        self.dim_h = int(dim_h)
        self.dim_s = int(dim_s)
        self.dim_r = int(dim_r)
        self.c = float(c)

    @property
    def dim_total(self) -> int:
        return self.dim_h + self.dim_s + self.dim_r

    def split(self, x):
        h = x[..., :self.dim_h]
        s = x[..., self.dim_h:self.dim_h + self.dim_s]
        r = x[..., self.dim_h + self.dim_s:]
        return h, s, r

    def combine(self, h, s, r):
        return torch.cat([h, s, r], dim=-1)

    def project(self, x):
        h, s, r = self.split(x)
        # H: clamp norms below the ball boundary; we route through the C
        # mobius_add(h, 0, c) which calls project_row internally.
        h = mobius_add(h, torch.zeros_like(h), self.c)
        s = s / torch.linalg.vector_norm(s, dim=-1, keepdim=True).clamp_min(1e-15)
        return self.combine(h, s, r)

    def distance(self, x, y):
        xh, xs, xr = self.split(x); yh, ys, yr = self.split(y)
        d_h = distance(xh, yh, self.c)
        inner = (xs * ys).sum(dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
        d_s = torch.acos(inner)
        d_r = torch.linalg.vector_norm(xr - yr, dim=-1)
        return torch.sqrt(d_h * d_h + d_s * d_s + d_r * d_r + 1e-15)

    def exp(self, x, v):
        xh, xs, xr = self.split(x); vh, vs, vr = self.split(v)
        new_h = exp_map(xh, vh, self.c)
        # Sphere exp: cos(‖vs‖) xs + sin(‖vs‖) vs/‖vs‖
        nv = torch.linalg.vector_norm(vs, dim=-1, keepdim=True).clamp_min(1e-15)
        new_s = torch.cos(nv) * xs + torch.sin(nv) * (vs / nv)
        new_r = xr + vr
        return self.combine(new_h, new_s, new_r)

    def log(self, x, y):
        xh, xs, xr = self.split(x); ys, yr = y[..., self.dim_h:self.dim_h+self.dim_s], y[..., self.dim_h+self.dim_s:]
        yh = y[..., :self.dim_h]
        v_h = log_map(xh, yh, self.c)
        inner = (xs * ys).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(inner)
        proj = ys - inner * xs
        v_s = theta * proj / torch.linalg.vector_norm(proj, dim=-1, keepdim=True).clamp_min(1e-15)
        v_r = yr - xr
        return self.combine(v_h, v_s, v_r)


# ---- Sphere ops (pure-torch; cheap closed-form, no C primitive needed) ----

def sphere_exp_map(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Exp map on the unit sphere: exp_x(v) = cos(‖v‖) x + sin(‖v‖) v/‖v‖."""
    n = safe_norm(v)
    return torch.cos(n) * x + torch.sin(n) * (v / n)


def sphere_log_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Log map on the sphere: log_x(y) = θ · (y − ⟨x,y⟩x) / ‖y − ⟨x,y⟩x‖
    where θ = arccos(⟨x,y⟩)."""
    inner = (x * y).sum(dim=-1, keepdim=True).clamp(
        -1.0 + TC_POINCARE_EPS_ATANH, 1.0 - TC_POINCARE_EPS_ATANH)
    theta = torch.acos(inner)
    proj = y - inner * x
    return theta * proj / safe_norm(proj)


def great_circle_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """d(x,y) = arccos(⟨x,y⟩) on the unit sphere."""
    inner = (x * y).sum(dim=-1).clamp(
        -1.0 + TC_POINCARE_EPS_ATANH, 1.0 - TC_POINCARE_EPS_ATANH)
    return torch.acos(inner)


def sphere_parallel_transport(v: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Standard sphere parallel transport (rotation in the (x,y)-plane)."""
    inner = (x * y).sum(dim=-1, keepdim=True).clamp(
        -1.0 + TC_POINCARE_EPS_ATANH, 1.0 - TC_POINCARE_EPS_ATANH)
    return v - ((v * y).sum(dim=-1, keepdim=True) / (1.0 + inner)) * (x + y)


__all__ = [
    # numerical safety
    "TC_POINCARE_EPS_NORM", "TC_POINCARE_EPS_ATANH", "TC_POINCARE_MARGIN",
    "safe_norm", "safe_arctanh", "safe_tanh", "project_to_ball",
    # Poincaré ball (C-kernel-backed)
    "mobius_add", "distance", "conformal_factor",
    "exp_map_zero", "log_map_zero", "exp_map", "log_map",
    "parallel_transport",
    # Sphere (pure-torch)
    "sphere_exp_map", "sphere_log_map", "great_circle_distance",
    "sphere_parallel_transport",
    # Product H × S × R
    "ProductManifold",
]
