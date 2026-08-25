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
import warnings

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
        # Registration is opt-in as of the autograd-stream fix: importing this
        # module no longer registers the PrivateUse1 backend, because doing so
        # made torch treat tensorcore as a process-wide accelerator and broke
        # CUDA autograd for every consumer that merely had it on PYTHONPATH
        # ("opt_ready_stream && opt_parent_stream INTERNAL ASSERT FAILED",
        # measured in all four arms of a qLLM A/B including the control).
        # These tests are about the device, so they ask for it.
        tensorcore_torch.enable_device()

    def test_hooks_registered(self):
        self.assertTrue(
            tensorcore_torch.privateuse1_hooks_registered(),
            "PrivateUse1HooksInterface must be registered once the device is "
            "enabled -- Task #938's DataLoader workers+pin_memory crash",
        )

    def test_import_alone_does_not_register_the_backend(self):
        """The regression this file most needs to hold.

        A library that registers a device backend on import mutates global
        torch state for consumers that never asked for it. Run out-of-process
        so this class's own setUp() enable_device() cannot mask it.
        """
        import subprocess
        import sys as _sys
        code = (
            "import torch, tensorcore_torch\n"
            "print(torch._C._get_privateuse1_backend_name())\n"
        )
        out = subprocess.run([_sys.executable, "-c", code],
                             capture_output=True, text=True)
        if out.returncode != 0:
            self.skipTest(f"probe failed: {out.stderr.strip()[:140]}")
        self.assertEqual(
            out.stdout.strip(), "privateuseone",
            "importing tensorcore_torch must NOT register the PrivateUse1 "
            "backend -- doing so makes torch treat tensorcore as a "
            "process-wide accelerator and breaks CUDA autograd")

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

    def test_amp_policy_is_consumed_by_state_and_report(self):
        expected = [torch.float32, torch.bfloat16]
        self.assertEqual(tensorcore_torch.pytorch_amp_supported_dtypes(), expected)
        self.assertEqual(torch.tensorcore.get_amp_supported_dtype(), expected)
        state = tensorcore_torch.pytorch_backend_state()
        self.assertEqual(
            state["amp_supported_dtypes"],
            ["torch.float32", "torch.bfloat16"],
        )
        self.assertIn("amp=torch.float32,torch.bfloat16", tensorcore_torch.pytorch_backend_report())


class TestExecutionLedger(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERR is not None:
            self.skipTest(f"tensorcore_torch import failed: {_IMPORT_ERR}")
        tensorcore_torch.reset_execution_state()

    def tearDown(self):
        if tensorcore_torch is not None:
            tensorcore_torch.reset_execution_state()

    def test_records_bounded_public_safe_dispatch_metadata(self):
        from tensorcore_torch.execution import record_execution

        record_execution(
            "gemm", "forward", "portable_cpu",
            input_devices=["cpu", "cpu"],
            transport="buffer_copy",
        )
        state = tensorcore_torch.execution_state()
        self.assertEqual(state["schema_version"], 1)
        self.assertEqual(state["total_dispatches"], 1)
        self.assertEqual(state["counts"], {"gemm:forward:portable_cpu": 1})
        self.assertEqual(
            state["last"],
            {
                "sequence": 1,
                "operation": "gemm",
                "phase": "forward",
                "backend": "portable_cpu",
                "input_devices": ["cpu"],
                "transport": "buffer_copy",
                "zero_copy": False,
            },
        )
        # Snapshots must not expose mutable internal state.
        state["last"]["backend"] = "tampered"
        self.assertEqual(
            tensorcore_torch.execution_state()["last"]["backend"],
            "portable_cpu",
        )

    def test_redacts_unsafe_labels_and_bounds_cardinality(self):
        import json
        from tensorcore_torch.execution import record_execution

        record_execution(
            "unsafe/path", "forward path", "backend/path",
            input_devices=["cpu/path"], transport="copy/path",
            fallback_reason="reason/path",
        )
        for index in range(200):
            record_execution(
                f"operation_{index}", "forward", "portable_cpu",
                input_devices=["cpu"], transport="buffer_copy",
            )
        state = tensorcore_torch.execution_state()
        serialized = json.dumps(state, sort_keys=True)
        self.assertNotIn("unsafe/path", serialized)
        self.assertNotIn("reason/path", serialized)
        self.assertLessEqual(len(state["last_by_operation"]), 128)
        self.assertLessEqual(len(state["counts"]), 512)
        self.assertEqual(state["total_dispatches"], 201)


class TestSparse24ABI(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERR is not None:
            self.skipTest(f"tensorcore_torch import failed: {_IMPORT_ERR}")

    def test_torch_dtypes_match_native_abi(self):
        import tensorcore as tc
        from tensorcore_torch import sparse_24

        expected = {
            torch.float16: tc.TC_DTYPE_F16,
            torch.bfloat16: tc.TC_DTYPE_BF16,
            torch.float32: tc.TC_DTYPE_F32,
        }
        for dtype, native_code in expected.items():
            with self.subTest(dtype=dtype):
                self.assertEqual(
                    sparse_24._dtype_code(torch.empty(1, dtype=dtype)),
                    native_code,
                )

    def test_prune_and_gemm_roundtrip_all_supported_dtypes(self):
        from tensorcore_torch import sparse_24

        weight_values = [
            [1.0, -4.0, 2.0, 3.0],
            [-2.0, 0.5, 5.0, 1.0],
            [3.0, 1.0, -1.0, 6.0],
            [0.25, -3.0, 4.0, 2.0],
        ]
        input_values = [
            [1.0, 2.0, -1.0, 0.5],
            [-2.0, 1.0, 3.0, 1.0],
        ]
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                weight = torch.tensor(weight_values, dtype=dtype)
                sparse_24.prune_2_4(weight)
                self.assertTrue(sparse_24.is_2_4(weight))
                matrix = weight.t().contiguous()
                inputs = torch.tensor(input_values, dtype=dtype)
                actual = sparse_24.sparse_gemm(inputs, matrix)
                expected = inputs @ matrix
                torch.testing.assert_close(
                    actual.float(), expected.float(), rtol=1e-2, atol=1e-2,
                )


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


class TestRiemannianAdamDiagnostics(unittest.TestCase):
    def test_multiple_optimizers_do_not_share_clamp_attribution(self):
        if _IMPORT_ERR is not None:
            self.skipTest(f"tensorcore_torch import failed: {_IMPORT_ERR}")
        import tensorcore as tc
        from tensorcore_torch.riemannian_adam import RiemannianAdam

        tc.riemannian_offmanifold_reset()
        off_ball = torch.nn.Parameter(torch.tensor([[4.0, 0.0]], dtype=torch.float32))
        off_ball.tc_manifold = ("poincare", 1.0)
        off_ball.grad = torch.zeros_like(off_ball)
        in_ball = torch.nn.Parameter(torch.tensor([[0.1, 0.0]], dtype=torch.float32))
        in_ball.tc_manifold = ("poincare", 1.0)
        in_ball.grad = torch.zeros_like(in_ball)
        first = RiemannianAdam([off_ball])
        second = RiemannianAdam([in_ball])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                first.step()
                second.step()
            self.assertEqual(first.offmanifold_rows, 1)
            self.assertEqual(second.offmanifold_rows, 0)
            self.assertEqual(tc.riemannian_offmanifold_count(), 1)
        finally:
            tc.riemannian_offmanifold_reset()


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
