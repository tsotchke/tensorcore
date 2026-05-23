"""Python usability shim for the tensorcore PyTorch bridge.

Importing this package registers PyTorch's PrivateUse1 backend name as
``tensorcore`` and installs a small ``torch.tensorcore`` runtime module. That
is intentionally narrower than a full PyTorch device backend: tensor factory
APIs such as ``torch.empty(..., device="tensorcore")`` still require native
PrivateUse1 allocator/storage/factory kernels and are expected to fail until
that lower-level backend work exists.

The existing extension API remains available directly from this package:
``tensorcore_torch.matmul`` and ``tensorcore_torch.set_default_matmul`` are
re-exported from the compiled bridge.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Optional

import torch

_BACKEND_NAME = "tensorcore"
_DEFAULT_PRIVATEUSE1_NAME = "privateuseone"


def _privateuse1_backend_name() -> Optional[str]:
    getter = getattr(getattr(torch, "_C", None), "_get_privateuse1_backend_name", None)
    if getter is None:
        return None
    return str(getter())


def _ensure_privateuse1_name() -> bool:
    current = _privateuse1_backend_name()
    if current == _BACKEND_NAME:
        return True
    if current not in (None, _DEFAULT_PRIVATEUSE1_NAME):
        return False

    rename = getattr(torch.utils, "rename_privateuse1_backend", None)
    if rename is None:
        return False

    try:
        rename(_BACKEND_NAME)
    except RuntimeError:
        if _privateuse1_backend_name() != _BACKEND_NAME:
            raise
    return _privateuse1_backend_name() == _BACKEND_NAME


def _device_index(device: Any = None) -> int:
    if device is None:
        return 0
    parsed = torch.device(device)
    if parsed.type != _BACKEND_NAME:
        raise ValueError(f"expected a {_BACKEND_NAME!r} device, got {device!r}")
    return 0 if parsed.index is None else int(parsed.index)


def _check_device(device: Any = None) -> int:
    index = _device_index(device)
    if index != 0:
        raise ValueError("tensorcore exposes one logical PrivateUse1 device: tensorcore:0")
    return index


def _torch_backend_module() -> Optional[types.ModuleType]:
    module = getattr(torch, _BACKEND_NAME, None)
    if isinstance(module, types.ModuleType):
        return module
    return None


def _torch_backend_module_registered() -> bool:
    module = _torch_backend_module()
    return module is not None and sys.modules.get(f"torch.{_BACKEND_NAME}") is module


def _new_backend_module() -> types.ModuleType:
    module = types.ModuleType("torch.tensorcore")
    module.__doc__ = (
        "Runtime shim for the tensorcore PrivateUse1 PyTorch backend. "
        "This registers the backend name and common runtime helpers, but "
        "does not provide allocator/storage/factory kernels for tensors "
        "created directly on device='tensorcore'."
    )

    def is_available() -> bool:
        return True

    def device_count() -> int:
        return 1

    def current_device() -> int:
        return 0

    def set_device(device: Any) -> None:
        _check_device(device)

    def get_device_name(device: Any = None) -> str:
        _check_device(device)
        return _BACKEND_NAME

    def synchronize(device: Any = None) -> None:
        _check_device(device)

    def _is_in_bad_fork() -> bool:
        return False

    def manual_seed_all(seed: int) -> None:
        del seed

    def get_rng_state(device: Any = None) -> torch.Tensor:
        _check_device(device)
        return torch.empty(0, dtype=torch.uint8)

    def set_rng_state(new_state: torch.Tensor, device: Any = None) -> None:
        _check_device(device)
        if not isinstance(new_state, torch.Tensor):
            raise TypeError("new_state must be a torch.Tensor")

    def get_amp_supported_dtype() -> List[torch.dtype]:
        return [torch.float32, torch.bfloat16]

    def supports_device_allocation() -> bool:
        return False

    def backend_state() -> Dict[str, Any]:
        owner = sys.modules.get("tensorcore_torch")
        state = getattr(owner, "pytorch_backend_state", None)
        if state is None:
            return {
                "backend_name": _BACKEND_NAME,
                "registered": False,
                "allocator_status": "initializing",
            }
        return state()

    def backend_report() -> str:
        owner = sys.modules.get("tensorcore_torch")
        report = getattr(owner, "pytorch_backend_report", None)
        if report is None:
            return "tensorcore PyTorch backend initializing"
        return report()

    module.is_available = is_available
    module.device_count = device_count
    module.current_device = current_device
    module.set_device = set_device
    module.get_device_name = get_device_name
    module.synchronize = synchronize
    module._is_in_bad_fork = _is_in_bad_fork
    module.manual_seed_all = manual_seed_all
    module.get_rng_state = get_rng_state
    module.set_rng_state = set_rng_state
    module.get_amp_supported_dtype = get_amp_supported_dtype
    module.supports_device_allocation = supports_device_allocation
    module.backend_state = backend_state
    module.backend_report = backend_report
    return module


def _ensure_generated_methods() -> None:
    if hasattr(torch.Tensor, f"is_{_BACKEND_NAME}"):
        return

    generate = getattr(torch.utils, "generate_methods_for_privateuse1_backend", None)
    if generate is not None:
        generate()


def _ensure_torch_backend_module() -> bool:
    if not _ensure_privateuse1_name():
        return False

    existing = getattr(torch, _BACKEND_NAME, None)
    if existing is None:
        backend_module = _new_backend_module()
        register_device_module = getattr(torch, "_register_device_module", None)
        if register_device_module is None:
            setattr(torch, _BACKEND_NAME, backend_module)
            sys.modules[f"torch.{_BACKEND_NAME}"] = backend_module
        else:
            register_device_module(_BACKEND_NAME, backend_module)
    else:
        sys.modules.setdefault(f"torch.{_BACKEND_NAME}", existing)

    _ensure_generated_methods()
    return True


_PYTORCH_BACKEND_REGISTERED = _ensure_torch_backend_module()

from ._C import (  # noqa: E402
    default_matmul_enabled,
    is_matmul_eligible,
    last_backend_name,
    matmul,
    matmul_bf16,
    matmul_eligibility,
    privateuse1_backend_name,
    set_default_matmul,
)

__all__ = [
    "default_matmul_enabled",
    "is_matmul_eligible",
    "last_backend_name",
    "matmul",
    "matmul_bf16",
    "matmul_eligibility",
    "privateuse1_backend_name",
    "pytorch_backend_report",
    "pytorch_backend_registered",
    "pytorch_backend_state",
    "set_default_matmul",
]


def pytorch_backend_registered() -> bool:
    """Return whether import registered ``torch.tensorcore`` in this process."""
    return _PYTORCH_BACKEND_REGISTERED


def pytorch_backend_state() -> Dict[str, Any]:
    """Return a structured snapshot of the tensorcore PyTorch bridge state.

    This intentionally distinguishes the import-time PrivateUse1 registration
    shim from a future full tensorcore tensor backend. Today matmul dispatch is
    backed by the extension, while direct ``device="tensorcore"`` allocation
    remains unavailable until allocator/storage/factory kernels land.
    """
    module = _torch_backend_module()
    supports_device_allocation = False
    is_available = False
    device_count = 0
    current_device: Optional[int] = None
    amp_supported_dtypes: List[str] = []
    matmul_dispatch_probe: Dict[str, Any] = {
        "eligible": False,
        "reason": "unprobed",
    }

    if module is not None:
        is_available_fn = getattr(module, "is_available", None)
        device_count_fn = getattr(module, "device_count", None)
        current_device_fn = getattr(module, "current_device", None)
        allocation_fn = getattr(module, "supports_device_allocation", None)
        amp_dtype_fn = getattr(module, "get_amp_supported_dtype", None)
        try:
            is_available = bool(is_available_fn()) if is_available_fn is not None else False
            device_count = int(device_count_fn()) if device_count_fn is not None else 0
            current_device = int(current_device_fn()) if current_device_fn is not None else None
            supports_device_allocation = (
                bool(allocation_fn()) if allocation_fn is not None else False
            )
            if amp_dtype_fn is not None:
                amp_supported_dtypes = [str(dtype) for dtype in amp_dtype_fn()]
            matmul_dispatch_probe = matmul_eligibility(
                torch.empty((1, 1), dtype=torch.float32),
                torch.empty((1, 1), dtype=torch.float32),
            )
        except Exception:
            is_available = False
            device_count = 0
            current_device = None
            supports_device_allocation = False
            amp_supported_dtypes = []
            matmul_dispatch_probe = {
                "eligible": False,
                "reason": "probe_error",
            }

    allocator_status = "available" if supports_device_allocation else "not_implemented"
    if not _PYTORCH_BACKEND_REGISTERED:
        allocator_status = "unregistered"

    return {
        "backend_name": _BACKEND_NAME,
        "privateuse1_backend_name": _privateuse1_backend_name(),
        "extension_privateuse1_backend_name": privateuse1_backend_name(),
        "registered": bool(_PYTORCH_BACKEND_REGISTERED),
        "torch_module_registered": _torch_backend_module_registered(),
        "generated_tensor_methods": hasattr(torch.Tensor, f"is_{_BACKEND_NAME}"),
        "is_available": is_available,
        "device_count": device_count,
        "current_device": current_device,
        "supports_device_allocation": supports_device_allocation,
        "allocator_status": allocator_status,
        "factory_kernels": supports_device_allocation,
        "storage_kernels": supports_device_allocation,
        "matmul_extension_loaded": callable(matmul),
        "matmul_dispatch_probe": matmul_dispatch_probe,
        "default_matmul_enabled": bool(default_matmul_enabled()),
        "last_backend_name": last_backend_name(),
        "amp_supported_dtypes": amp_supported_dtypes,
    }


def pytorch_backend_report() -> str:
    """Return a compact human-readable tensorcore PyTorch backend report."""
    state = pytorch_backend_state()
    return (
        "tensorcore PyTorch backend: "
        f"registered={state['registered']} "
        f"privateuse1={state['privateuse1_backend_name']} "
        f"module={state['torch_module_registered']} "
        f"tensor_methods={state['generated_tensor_methods']} "
        f"allocation={state['allocator_status']} "
        f"dispatch_probe={state['matmul_dispatch_probe']['reason']} "
        f"matmul_extension={state['matmul_extension_loaded']} "
        f"default_matmul={state['default_matmul_enabled']} "
        f"last_backend={state['last_backend_name']}"
    )


if _torch_backend_module_registered():
    torch.tensorcore.backend_state = pytorch_backend_state
    torch.tensorcore.backend_report = pytorch_backend_report
