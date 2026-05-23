#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TENSORCORE_LIB_DIR="${TENSORCORE_LIB_DIR:-${ROOT}/build-portable-cpu-current}"
REQUIRE_PYTORCH="${REQUIRE_PYTORCH:-0}"

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

PYTHONPATH="${ROOT}/bindings/pytorch${PYTHONPATH:+:${PYTHONPATH}}" \
TENSORCORE_LIB_DIR="${TENSORCORE_LIB_DIR}" \
"${PYTHON_BIN}" - <<'PY'
import torch
import tensorcore_torch as tct


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

print("tensorcore PyTorch bridge smoke OK")
PY
