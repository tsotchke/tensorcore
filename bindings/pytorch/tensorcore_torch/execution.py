"""Process-local execution provenance for TensorCore-backed PyTorch ops.

The ledger records only bounded diagnostic metadata: operation, phase, native
backend, source device classes, and the binding transport used to reach the C
runtime. It never stores tensors, pointers, shapes, paths, or user data.
"""

from __future__ import annotations

import copy
import re
import threading
from typing import Any, Iterable


_SCHEMA_VERSION = 1
_MAX_OPERATION_KEYS = 128
_MAX_COUNT_KEYS = 512
_MAX_DEVICE_CLASSES = 8
_SAFE_LABEL = re.compile(r"[A-Za-z0-9_.:-]{1,64}\Z")
_lock = threading.Lock()
_sequence = 0
_counts: dict[str, int] = {}
_last: dict[str, Any] | None = None
_last_by_operation: dict[str, dict[str, Any]] = {}


def _safe_label(value: object, default: str) -> str:
    label = str(value or default)
    return label if _SAFE_LABEL.fullmatch(label) else default


def _normalized_devices(devices: Iterable[str]) -> list[str]:
    values = sorted({_safe_label(device, "unknown") for device in devices})
    values = values[:_MAX_DEVICE_CLASSES]
    return values or ["unknown"]


def record_execution(
    operation: str,
    phase: str,
    backend: str,
    *,
    input_devices: Iterable[str],
    transport: str,
    zero_copy: bool = False,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Record one successful native dispatch and return its public-safe row."""
    operation = _safe_label(operation, "unknown")
    phase = _safe_label(phase, "unknown")
    backend = _safe_label(backend, "none")
    transport = _safe_label(transport, "unknown")
    devices = _normalized_devices(input_devices)
    fallback = _safe_label(fallback_reason, "redacted") if fallback_reason else None

    global _sequence, _last
    with _lock:
        _sequence += 1
        if (
            operation not in _last_by_operation
            and len(_last_by_operation) >= _MAX_OPERATION_KEYS - 1
        ):
            operation = "overflow"
        key = f"{operation}:{phase}:{backend}"
        if key not in _counts and len(_counts) >= _MAX_COUNT_KEYS - 1:
            key = "overflow"
        _counts[key] = _counts.get(key, 0) + 1
        row: dict[str, Any] = {
            "sequence": _sequence,
            "operation": operation,
            "phase": phase,
            "backend": backend,
            "input_devices": devices,
            "transport": transport,
            "zero_copy": bool(zero_copy),
        }
        if fallback:
            row["fallback_reason"] = fallback
        _last = row
        _last_by_operation[operation] = row
        return copy.deepcopy(row)


def execution_state() -> dict[str, Any]:
    """Return a stable snapshot suitable for runtime evidence and diagnostics."""
    with _lock:
        return {
            "schema_version": _SCHEMA_VERSION,
            "sequence": _sequence,
            "total_dispatches": sum(_counts.values()),
            "counts": dict(sorted(_counts.items())),
            "last": copy.deepcopy(_last),
            "last_by_operation": {
                key: copy.deepcopy(_last_by_operation[key])
                for key in sorted(_last_by_operation)
            },
        }


def reset_execution_state() -> None:
    """Clear the process-local diagnostic ledger."""
    global _sequence, _last
    with _lock:
        _sequence = 0
        _counts.clear()
        _last = None
        _last_by_operation.clear()


__all__ = ["execution_state", "record_execution", "reset_execution_state"]
