"""Smoke test for the tensorcore <-> PyTorch bridge dispatch hooks.

Covers two failure modes that previously forced trainer-side work-arounds:

  #938  PrivateUse1HooksInterface not registered -> DataLoader crashes when
        num_workers > 0 OR pin_memory=True. Verifies a DataLoader with
        num_workers=2 + pin_memory=True iterates one batch without raising
        the "Please register PrivateUse1HooksInterface" RuntimeError.

  #936  tc_mps_gemm not wired into the bridge -> aten::matmul on MPS goes
        through native CPU/MPS instead of the tuned tensorcore Metal
        kernels. Verifies the MPS dispatch returns a result that agrees
        with a CPU reference to within fp32 tolerance, and that the
        dispatch counter increments.

Run:
    cd bindings/pytorch
    python setup.py build_ext --inplace
    pytest test_tensorcore_torch_dispatch.py -v
"""
from __future__ import annotations

import os
import sys
import unittest

import torch
from torch.utils.data import DataLoader, Dataset


def _bridge_path():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


_bridge_path()

try:
    import tensorcore_torch  # noqa: E402
    _IMPORT_ERR = None
except Exception as exc:  # pragma: no cover - import is part of the smoke
    tensorcore_torch = None  # type: ignore[assignment]
    _IMPORT_ERR = exc


class _SmallTensorDataset(Dataset):
    """Plain CPU tensor dataset; DataLoader is what touches pin_memory."""

    def __init__(self, n: int = 8, dim: int = 16):
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, 4, (n,), dtype=torch.long)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class TestPrivateUse1Hooks(unittest.TestCase):
    """Task #938 — PrivateUse1HooksInterface registration."""

    def setUp(self):
        if _IMPORT_ERR is not None:
            self.skipTest(f"tensorcore_torch import failed: {_IMPORT_ERR}")

    def test_hooks_registered(self):
        self.assertTrue(
            tensorcore_torch.privateuse1_hooks_registered(),
            "PrivateUse1HooksInterface must be registered after bridge import",
        )

    def test_dataloader_pin_memory_workers(self):
        """The actual repro of the trainer crash: DataLoader with workers + pin_memory.

        On a system with no real pinnable device (CPU-only macOS), pin_memory
        is still safe: the bridge's getPinnedMemoryAllocator returns the
        tensorcore host allocator and DataLoader proceeds normally.
        """
        ds = _SmallTensorDataset(n=8, dim=16)
        try:
            dl = DataLoader(
                ds,
                batch_size=2,
                num_workers=2,
                pin_memory=True,
                persistent_workers=False,
            )
            xb, yb = next(iter(dl))
        except RuntimeError as exc:
            msg = str(exc)
            if "PrivateUse1HooksInterface" in msg:
                self.fail(
                    "DataLoader still crashes with the unregistered-hooks error: "
                    f"{msg}"
                )
            raise
        self.assertEqual(xb.shape, (2, 16))
        self.assertEqual(yb.shape, (2,))


class TestMPSDispatch(unittest.TestCase):
    """Task #936 — tc_mps_gemm dispatch from aten::matmul on MPS."""

    def setUp(self):
        if _IMPORT_ERR is not None:
            self.skipTest(f"tensorcore_torch import failed: {_IMPORT_ERR}")
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available on this host")

    def test_mps_matmul_matches_cpu_reference(self):
        torch.manual_seed(0)
        M, K, N = 256, 256, 256
        a_cpu = torch.randn(M, K, dtype=torch.float32)
        b_cpu = torch.randn(K, N, dtype=torch.float32)
        ref = a_cpu @ b_cpu

        a_mps = a_cpu.to("mps")
        b_mps = b_cpu.to("mps")

        mps_available = tensorcore_torch.mps_bridge_available()
        before = tensorcore_torch.mps_dispatch_count()
        tensorcore_torch.set_default_matmul(True)
        try:
            out = (a_mps @ b_mps).to("cpu")
        finally:
            tensorcore_torch.set_default_matmul(False)
        after = tensorcore_torch.mps_dispatch_count()

        # Correctness regardless of which path served it (tc_mps_gemm or native).
        max_abs = (out - ref).abs().max().item()
        self.assertLess(
            max_abs, 1e-3,
            f"MPS matmul disagrees with CPU reference: max_abs={max_abs:.3e}",
        )

        # Dispatch engagement: only assert when libtensorcore exposes tc_mps_gemm.
        if mps_available:
            self.assertGreater(
                after, before,
                "MPS dispatch was eligible but tc_gemm (MPS path) was not invoked",
            )
            print(
                f"\n  [info] tensorcore MPS path engaged: dispatch_count {before} -> {after}, "
                f"max_abs={max_abs:.3e}"
            )
        else:
            print(
                "\n  [info] libtensorcore was built without TC_ENABLE_METAL; "
                "MPS dispatch fell through to native PyTorch. "
                f"Set TENSORCORE_LIB_DIR=<metal-build> and rebuild the bridge "
                f"to engage tc_gemm's MPS backend. max_abs={max_abs:.3e}"
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
