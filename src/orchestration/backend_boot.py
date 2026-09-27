"""Process boot identity for restart vs runtime-loss recovery evidence."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_BOOT_ID: Optional[str] = None
_BOOT_MONO: Optional[float] = None
_BOOT_AT: Optional[datetime] = None


def init_backend_boot() -> str:
    """Record a fresh boot marker for this process (idempotent per process)."""
    global _BOOT_ID, _BOOT_MONO, _BOOT_AT
    if _BOOT_ID is None:
        _BOOT_ID = uuid.uuid4().hex
        _BOOT_MONO = time.monotonic()
        _BOOT_AT = datetime.now(timezone.utc)
    return _BOOT_ID


def get_backend_boot_id() -> str:
    if _BOOT_ID is None:
        init_backend_boot()
    assert _BOOT_ID is not None
    return _BOOT_ID


def get_backend_uptime_seconds() -> float:
    if _BOOT_MONO is None:
        init_backend_boot()
    assert _BOOT_MONO is not None
    return time.monotonic() - _BOOT_MONO


def classify_recovery_cause(recorded_boot_id: Optional[str]) -> str:
    """backend_restart when run was stamped under a prior boot, else runtime_task_lost."""
    if not recorded_boot_id:
        return "runtime_task_lost"
    if recorded_boot_id != get_backend_boot_id():
        return "backend_restart"
    return "runtime_task_lost"


def build_recovery_metadata(recorded_boot_id: Optional[str]) -> Dict[str, Any]:
    cause = classify_recovery_cause(recorded_boot_id)
    return {
        "cause": cause,
        "current_boot_id": get_backend_boot_id(),
        "recorded_boot_id": recorded_boot_id,
        "backend_uptime_seconds": round(get_backend_uptime_seconds(), 1),
        "backend_started_at": _BOOT_AT.isoformat() if _BOOT_AT else None,
    }
