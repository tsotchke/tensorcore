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

import ctypes
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
_offmanifold_step_lock = threading.Lock()
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


def _offmanifold_count() -> int:
    """Process-global tally of Poincaré clamps, or -1 if the engine predates it.

    The C engine clamps 1 − c‖x‖² up to 1e-15 when a row leaves the ball, which
    keeps the arithmetic finite while driving the tangent rescale 1/λ_x² to
    ~2.5e-31 — the row's gradient is annihilated, not projected. Reading this
    is the difference between "the geometry did nothing" and "the geometry was
    never evaluated inside its domain", which look identical in a loss curve.
    """
    try:
        fn = _tc._lib.tc_riemannian_offmanifold_count
    except AttributeError:
        return -1
    fn.restype = ctypes.c_ulonglong
    return int(fn())


def _wrappable(t: torch.Tensor) -> bool:
    """Can the C engine address this tensor's memory in place?

    Needs to be device memory, fp32, and contiguous — the kernels index rows
    as `params + n*D`, so a strided or half-precision tensor cannot be written
    through. Anything failing this drops to the host copy path rather than
    being silently mis-indexed.
    """
    return t.is_cuda and t.dtype == torch.float32 and t.is_contiguous()


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
        # Baseline for the off-manifold tally. The C counter is monotonic and
        # process-global, so the delta across a step is this optimizer's share
        # as long as one optimizer is stepping — which is the case in training.
        # Reading rather than resetting keeps it to one C call per step.
        #
        # `offmanifold_available` is not decoration. An engine built before
        # this counter existed — or one whose export list omits the symbol,
        # which is exactly how the first attempt at this wire shipped dead —
        # cannot answer the question, and "cannot answer" must not be
        # rendered as "zero clamps". A false all-clear is worse than no
        # reading, because it is the reading a caller stops checking.
        _probe = _offmanifold_count()
        self.offmanifold_available = _probe >= 0
        self.offmanifold_steps = 0    # steps during which at least one clamp fired
        self.offmanifold_rows = 0     # total rows clamped since construction
        self.offmanifold_counter_resets = 0

    @torch.no_grad()
    def step(self, closure=None):
        # The C counters are process-global. Serialize Python optimizer steps
        # and measure a local before/after interval so one optimizer cannot
        # claim clamps produced by another optimizer that stepped earlier.
        with _offmanifold_step_lock:
            return self._step_serialized(closure)

    def _step_serialized(self, closure=None):
        offmanifold_before = _offmanifold_count()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ctx = _get_ctx()
        import numpy as np

        # The kernels launch on CUDA's default stream, while torch may have
        # produced these gradients on another. One sync per step (not per
        # parameter) guarantees the backward pass has landed before the
        # optimizer reads it; the kernels themselves end in
        # cudaDeviceSynchronize, so the write-back direction is already
        # ordered.
        if torch.cuda.is_available() and any(
                p.is_cuda for g in self.param_groups for p in g["params"]):
            torch.cuda.current_stream().synchronize()

        try:
            cuda_ok = bool(_tc._lib.tc_cuda_is_active())
        except Exception:
            cuda_ok = False

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

                m_t = state["exp_avg"]
                v_t = state["exp_avg_sq"]
                # `cuda_ok` is not redundant with `_wrappable`. A tensor can be
                # device-resident while tensorcore's CUDA path is unavailable
                # (runtime not initialised, GEMM policy disabled). Wrapping a
                # device pointer in that state routes GPU addresses into the
                # OpenMP host reference. The C side now refuses that outright,
                # but refusing means an exception mid-training; checking here
                # degrades to the copy path instead, which is slow but runs.
                zero_copy = (cuda_ok and _wrappable(p) and _wrappable(m_t)
                             and _wrappable(v_t) and p.grad.is_cuda)

                if zero_copy:
                    # Hand the C engine the tensors' own device pointers. The
                    # kernels update params and moments IN PLACE, so there is
                    # no allocation, no transfer, and nothing to copy back.
                    #
                    # This is the whole ballgame. The kernels were never the
                    # cost -- the binding was. Going through numpy meant, per
                    # parameter per step: 4 device-to-host copies, 4
                    # cudaMallocManaged plus host-to-device, then 3 reads back
                    # and 3 copies to the GPU. The trainer tags EVERY
                    # parameter with a manifold, so all 38 took that path,
                    # output head (32.8M) and embedding (16.4M) included, when
                    # only 2 are curved -- roughly 1.8 GB of PCIe traffic per
                    # step for the output head alone and ~152 synchronizing
                    # allocations per step overall. Measured 0.65 s/step of
                    # optimizer overhead, ~6x slower than plain AdamW.
                    D = int(p.shape[-1])
                    N = int(p.numel() // D)
                    g = p.grad.detach()
                    if g.dtype != torch.float32 or not g.is_contiguous():
                        # Grads are read-only here, so a device-side
                        # contiguous fp32 copy is harmless and still never
                        # touches the host.
                        g = g.to(torch.float32).contiguous()
                    nbytes = N * D * 4
                    b_p = ctx.buffer_from_ptr(p.data_ptr(), nbytes)
                    b_g = ctx.buffer_from_ptr(g.data_ptr(), nbytes)
                    b_m = ctx.buffer_from_ptr(m_t.data_ptr(), nbytes)
                    b_v = ctx.buffer_from_ptr(v_t.data_ptr(), nbytes)
                else:
                    # Host path: CPU tensors, or a layout the kernel cannot
                    # address in place (non-contiguous, non-fp32).
                    params_flat, N, D = _flat(p)
                    grad_flat = p.grad.detach().to(torch.float32).contiguous().cpu().view(N, D)
                    m_flat = m_t.detach().cpu().view(N, D)
                    v_flat = v_t.detach().cpu().view(N, D)

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

                if zero_copy:
                    continue   # kernels wrote through the tensors' own memory

                # Host path only: copy back into the existing tensors.
                new_p = torch.from_numpy(b_p.to_numpy((N, D), np.float32))
                new_m = torch.from_numpy(b_m.to_numpy((N, D), np.float32))
                new_v = torch.from_numpy(b_v.to_numpy((N, D), np.float32))
                p.copy_(new_p.view_as(p).to(p.dtype))
                state["exp_avg"].copy_(new_m.view_as(state["exp_avg"]))
                state["exp_avg_sq"].copy_(new_v.view_as(state["exp_avg_sq"]))

        # Did any curved row leave its manifold this step? One C read, after
        # the whole group, so the cost does not scale with parameter count.
        now = _offmanifold_count()
        if offmanifold_before >= 0 and now >= 0:
            if now < offmanifold_before:
                # A caller used the public process-global reset during this
                # step. Do not manufacture a wrapped unsigned delta; the next
                # step starts from the new counter epoch.
                self.offmanifold_counter_resets += 1
                delta = 0
            else:
                delta = now - offmanifold_before
            if delta > 0:
                self.offmanifold_steps += 1
                self.offmanifold_rows += delta
                # Warn on the first occurrence and then on powers of two, so a
                # persistent fault stays visible without flooding a training log.
                n = self.offmanifold_steps
                if n & (n - 1) == 0:
                    import warnings
                    warnings.warn(
                        f"RiemannianAdam: {delta} parameter row(s) left the "
                        f"Poincare ball this step ({self.offmanifold_rows} rows "
                        f"over {n} step(s)). The conformal denominator was "
                        f"clamped, so those rows' gradients are scaled by "
                        f"~2.5e-31 -- they are not learning. Check the "
                        f"initialisation scale and the learning rate on the "
                        f"hyperbolic factor.",
                        RuntimeWarning, stacklevel=2)

        return loss


__all__ = ["RiemannianAdam"]
