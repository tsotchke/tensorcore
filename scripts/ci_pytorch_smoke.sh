#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TENSORCORE_LIB_DIR="${TENSORCORE_LIB_DIR:-${ROOT}/build-portable-cpu-current}"
REQUIRE_PYTORCH="${REQUIRE_PYTORCH:-0}"
REQUIRE_PYTORCH_BACKEND="${REQUIRE_PYTORCH_BACKEND:-0}"

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import torch  # noqa: F401
PY
then
    if [[ "${REQUIRE_PYTORCH}" == "1" ]]; then
        echo "PyTorch is required but is not importable with ${PYTHON_BIN}" >&2
        exit 1
    fi
    echo "PyTorch not importable with ${PYTHON_BIN}; skipping PyTorch bridge smoke"
    exit 0
fi

if [[ ! -f "${TENSORCORE_LIB_DIR}/libtensorcore.dylib" &&
      ! -f "${TENSORCORE_LIB_DIR}/libtensorcore.so" ]]; then
    echo "No libtensorcore.{dylib,so} in ${TENSORCORE_LIB_DIR}" >&2
    echo "Build tensorcore first or set TENSORCORE_LIB_DIR." >&2
    exit 1
fi
TENSORCORE_LIB_DIR="$(cd "${TENSORCORE_LIB_DIR}" && pwd)"

(
    cd "${ROOT}/bindings/pytorch"
    TENSORCORE_LIB_DIR="${TENSORCORE_LIB_DIR}" \
        "${PYTHON_BIN}" setup.py build_ext --inplace --force
)

PYTHONPATH="${ROOT}/bindings/pytorch:${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" \
TENSORCORE_LIB_DIR="${TENSORCORE_LIB_DIR}" \
REQUIRE_PYTORCH_BACKEND="${REQUIRE_PYTORCH_BACKEND}" \
"${PYTHON_BIN}" - <<'PY'
import os
import json
import sys
import torch

# Pre-initialize the C ABI through the ctypes wrapper before importing the
# PyTorch bridge. The extension must tolerate tc_init returning
# TC_ERR_ALREADY_INITIALIZED with a valid context.
lib_dir = os.environ["TENSORCORE_LIB_DIR"]
for lib_name in ("libtensorcore.dylib", "libtensorcore.so"):
    candidate = os.path.join(lib_dir, lib_name)
    if os.path.exists(candidate):
        os.environ["TENSORCORE_LIB"] = candidate
        break
else:
    raise AssertionError(f"no tensorcore shared library found in {lib_dir}")

import tensorcore as tc
ctx = tc.init()
if not ctx:
    raise AssertionError("tensorcore ctypes pre-init returned a null context")

import tensorcore_torch as tct

if not tct.pytorch_backend_registered():
    raise AssertionError("tensorcore_torch did not register the PyTorch PrivateUse1 backend module")
if not hasattr(torch, "tensorcore"):
    raise AssertionError("torch.tensorcore runtime module is not registered")
if sys.modules.get("torch.tensorcore") is not torch.tensorcore:
    raise AssertionError("torch.tensorcore is not present in sys.modules")
if not torch.tensorcore.is_available():
    raise AssertionError("torch.tensorcore reports unavailable")
if torch.tensorcore.device_count() != 1:
    raise AssertionError("torch.tensorcore should expose one logical device")
if torch.tensorcore.current_device() != 0:
    raise AssertionError("torch.tensorcore current_device should be 0")
if torch.device("tensorcore").type != "tensorcore":
    raise AssertionError("torch.device did not recognize the tensorcore backend name")
if torch.device("tensorcore:0").index != 0:
    raise AssertionError("torch.device did not recognize tensorcore:0")
if not hasattr(torch.Tensor, "is_tensorcore"):
    raise AssertionError("PrivateUse1 tensor helpers were not generated")

state = tct.pytorch_backend_state()
if state.get("backend_name") != "tensorcore":
    raise AssertionError(f"unexpected backend state name: {state}")
if state.get("privateuse1_backend_name") != "tensorcore":
    raise AssertionError(f"unexpected PrivateUse1 state name: {state}")
if state.get("extension_privateuse1_backend_name") != "tensorcore":
    raise AssertionError(f"unexpected extension PrivateUse1 name: {state}")
if state.get("registered") is not True:
    raise AssertionError(f"backend state should report registered: {state}")
if state.get("torch_module_registered") is not True:
    raise AssertionError(f"backend state should report torch module registered: {state}")
if state.get("generated_tensor_methods") is not True:
    raise AssertionError(f"backend state should report generated tensor methods: {state}")
if state.get("is_available") is not True:
    raise AssertionError(f"backend state should report available runtime shim: {state}")
if state.get("device_count") != 1 or state.get("current_device") != 0:
    raise AssertionError(f"backend state device fields mismatch: {state}")
if state.get("supports_device_allocation") is not False:
    raise AssertionError(f"backend state should report allocation unsupported: {state}")
if state.get("allocator_status") != "not_implemented":
    raise AssertionError(f"backend state allocation status mismatch: {state}")
if state.get("factory_kernels") is not False or state.get("storage_kernels") is not False:
    raise AssertionError(f"backend state should report missing factory/storage kernels: {state}")
if state.get("matmul_extension_loaded") is not True:
    raise AssertionError(f"backend state should report matmul extension loaded: {state}")
if torch.tensorcore.backend_state() != state:
    raise AssertionError("torch.tensorcore.backend_state does not match package state")
report = tct.pytorch_backend_report()
if "allocation=not_implemented" not in report or "registered=True" not in report:
    raise AssertionError(f"backend report missing expected fields: {report}")
if torch.tensorcore.backend_report() != report:
    raise AssertionError("torch.tensorcore.backend_report does not match package report")
print("tensorcore PyTorch backend state:", json.dumps(state, sort_keys=True))


def assert_close(actual, expected, *, rtol=1e-5, atol=1e-5):
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def expect_raises(fn, needle):
    try:
        fn()
    except Exception as exc:
        if needle not in str(exc):
            raise AssertionError(f"expected {needle!r} in {exc!r}") from exc
        return
    raise AssertionError(f"expected exception containing {needle!r}")


torch.manual_seed(7)

A = torch.randn(3, 4, dtype=torch.float32)
B = torch.randn(4, 5, dtype=torch.float32)
expected = A @ B
out = tct.matmul(A, B)
assert_close(out, expected)
if tct.last_backend_name() != "portable_cpu":
    raise AssertionError(f"unexpected backend after fp32 matmul: {tct.last_backend_name()}")

A_nc = torch.randn(3, 8, dtype=torch.float32)[:, ::2]
B_nc = torch.randn(10, 4, dtype=torch.float32).t()
assert not A_nc.is_contiguous()
assert not B_nc.is_contiguous()
assert_close(tct.matmul(A_nc, B_nc), A_nc @ B_nc)

Ab = torch.randn(4, 3, dtype=torch.float32).to(torch.bfloat16)
Bb = torch.randn(3, 2, dtype=torch.float32).to(torch.bfloat16)
expected_b = (Ab.float() @ Bb.float()).to(torch.bfloat16)
assert_close(tct.matmul_bf16(Ab, Bb), expected_b, rtol=0.0, atol=0.0)

K0 = tct.matmul(torch.empty(3, 0), torch.empty(0, 5))
assert K0.shape == (3, 5)
assert torch.count_nonzero(K0).item() == 0

M0 = tct.matmul(torch.empty(0, 4), torch.empty(4, 5))
N0 = tct.matmul(torch.empty(3, 4), torch.empty(4, 0))
assert M0.shape == (0, 5)
assert N0.shape == (3, 0)

expect_raises(lambda: tct.matmul(A, B.to(torch.bfloat16)), "share dtype")
expect_raises(lambda: tct.matmul(torch.randn(2, 3), torch.randn(4, 2)), "shape mismatch")

previous = tct.set_default_matmul(True)
try:
    routed = torch.matmul(A, B)
    assert_close(routed, expected)
    if not tct.default_matmul_enabled():
        raise AssertionError("dispatcher flag not enabled")

    grad_a = A.detach().clone().requires_grad_(True)
    grad_b = B.detach().clone().requires_grad_(True)
    loss = torch.matmul(grad_a, grad_b).sum()
    loss.backward()
    if grad_a.grad is None or grad_b.grad is None:
        raise AssertionError("autograd fallback did not produce gradients")
finally:
    tct.set_default_matmul(previous)

if tct.privateuse1_backend_name() != "tensorcore":
    raise AssertionError("PrivateUse1 backend name mismatch")

backend_required = os.environ.get("REQUIRE_PYTORCH_BACKEND") == "1"
try:
    allocated = torch.empty((1,), device="tensorcore")
except Exception as exc:
    if backend_required:
        raise AssertionError(
            "torch.empty(device='tensorcore') failed; full tensorcore device "
            "allocation requires PrivateUse1 allocator/storage/factory kernels"
        ) from exc
    print(
        "torch.empty(device='tensorcore') unavailable as expected without "
        "PrivateUse1 allocator/storage/factory kernels"
    )
else:
    if allocated.device.type != "tensorcore":
        raise AssertionError(f"unexpected allocated device: {allocated.device}")

print("tensorcore PyTorch bridge smoke OK")
PY
