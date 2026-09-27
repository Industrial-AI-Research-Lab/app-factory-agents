"""Shared helpers for ``metadata.external_mcp.headers`` runtime fields."""

from __future__ import annotations

from typing import Any, Dict, Optional


def runtime_headers_dict(runtime: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Convert ``headers`` (list of ``{name, value}`` or flat dict) to a header map."""
    if not isinstance(runtime, dict):
        return None
    raw = runtime.get("headers")
    if isinstance(raw, dict):
        out = {str(k): str(v) for k, v in raw.items() if str(k).strip()}
        return out or None
    if isinstance(raw, list):
        out: Dict[str, str] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if name:
                out[name] = str(item.get("value") or "")
        return out or None
    return None
