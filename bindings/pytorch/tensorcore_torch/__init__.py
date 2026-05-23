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
from typing import Any, List, Optional

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
    last_backend_name,
    matmul,
    matmul_bf16,
    privateuse1_backend_name,
    set_default_matmul,
)

__all__ = [
    "default_matmul_enabled",
    "last_backend_name",
    "matmul",
    "matmul_bf16",
    "privateuse1_backend_name",
    "pytorch_backend_registered",
    "set_default_matmul",
]


def pytorch_backend_registered() -> bool:
    """Return whether import registered ``torch.tensorcore`` in this process."""
    return _PYTORCH_BACKEND_REGISTERED
