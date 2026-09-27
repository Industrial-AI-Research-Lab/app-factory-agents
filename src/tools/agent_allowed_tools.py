"""Merge agent ``allowed_tools`` and ``allowed_mcp_tools`` for runtime enforcement."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, List, Mapping, Optional

import yaml

from tools.mcp_tool_ids import parse_mcp_public_tool_id

logger = logging.getLogger(__name__)

# Must stay in allowed_tools (never MCP list) — see agents.delegation_policy.
DELEGATE_TO_AGENT_TOOL = "delegate_to_agent"

_TOOLS_YAML = os.path.join(os.path.dirname(__file__), "..", "config", "tools.yaml")

# Fallback when tools.yaml is unavailable (tests, minimal env).
_MANUAL_BUILTIN_TOOL_IDS = frozenset(
    {
        "attachment_fetch",
        "attachment_list",
        "attachment_presign_get",
        "attachment_presign_put",
        "attachment_view",
        "bash",
        "context_read",
        "context_write",
        "create",
        "delegate_to_agent",
        "deploy_from_artifacts",
        "edit",
        "glob",
        "grep",
        "read",
    }
)


class _ToolsYamlUnavailable(OSError):
    """Transient ``tools.yaml`` read failure — must not be cached."""


@lru_cache(maxsize=1)
def _known_builtin_tool_ids_from_yaml() -> frozenset[str]:
    """Load builtin wire names from ``tools.yaml`` (cached on success only)."""
    ids: set[str] = set(_MANUAL_BUILTIN_TOOL_IDS)
    try:
        with open(_TOOLS_YAML, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise _ToolsYamlUnavailable(str(exc)) from exc
    for tool in data.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("source") == "mcp_server":
            continue
        storage_id = str(tool.get("_id") or "").strip()
        if storage_id:
            ids.add(storage_id)
        schema = tool.get("schema") if isinstance(tool.get("schema"), dict) else {}
        function = (
            schema.get("function") if isinstance(schema.get("function"), dict) else {}
        )
        wire = str(function.get("name") or "").strip()
        if wire:
            ids.add(wire)
    return frozenset(ids)


def known_builtin_tool_ids() -> frozenset[str]:
    """Builtin wire names from ``tools.yaml`` (source != mcp_server)."""
    try:
        return _known_builtin_tool_ids_from_yaml()
    except _ToolsYamlUnavailable as exc:
        logger.warning(
            "[ALLOWLIST] source=tools.yaml — failed to load builtin ids: %s",
            exc,
        )
        return frozenset(_MANUAL_BUILTIN_TOOL_IDS)


def _is_mcp_allowlist_ref(tool_id: str) -> bool:
    """True when ``tool_id`` belongs in ``allowed_mcp_tools`` (public id or path-A wire)."""
    from config.configuration_resolution import is_wire_configuration_name

    tid = str(tool_id or "").strip()
    if not tid or tid in known_builtin_tool_ids():
        return False
    if is_mcp_public_tool_id(tid):
        return True
    # Path-A MCP wires: ``{abbr}_{method}`` (always contains ``_``).
    return is_wire_configuration_name(tid) and "_" in tid


def is_mcp_public_tool_id(tool_id: str) -> bool:
    """True when ``tool_id`` uses the MCP public id shape ``server.tool``."""
    return parse_mcp_public_tool_id(str(tool_id or "").strip()) is not None


def _dedupe_preserve_order(tool_ids: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for raw in tool_ids:
        tid = str(raw or "").strip()
        if not tid or tid in seen:
            continue
        seen.add(tid)
        out.append(tid)
    return out


def normalize_agent_tool_allowlists(
    allowed_tools: Optional[List[str]],
    allowed_mcp_tools: Optional[List[str]],
) -> tuple[List[str], List[str]]:
    """Partition tool ids into built-in vs MCP lists (canonical storage shape).

    - MCP-shaped ids (``server.tool``) and path-A wire names land in ``allowed_mcp_tools``.
    - ``delegate_to_agent`` always stays in ``allowed_tools``.
    - Built-ins accidentally placed in ``allowed_mcp_tools`` move to ``allowed_tools``.
    """
    regular: List[str] = []
    mcp: List[str] = []

    def add_regular(tid: str) -> None:
        if tid not in regular:
            regular.append(tid)

    def add_mcp(tid: str) -> None:
        if tid not in mcp:
            mcp.append(tid)

    for raw in list(allowed_mcp_tools or []):
        tid = str(raw or "").strip()
        if not tid:
            continue
        if tid == DELEGATE_TO_AGENT_TOOL:
            add_regular(tid)
        elif _is_mcp_allowlist_ref(tid):
            add_mcp(tid)
        else:
            add_regular(tid)

    for raw in list(allowed_tools or []):
        tid = str(raw or "").strip()
        if not tid:
            continue
        if tid == DELEGATE_TO_AGENT_TOOL:
            add_regular(tid)
        elif _is_mcp_allowlist_ref(tid):
            add_mcp(tid)
        else:
            add_regular(tid)

    return _dedupe_preserve_order(regular), _dedupe_preserve_order(mcp)


def clear_agent_mcp_allowlist(doc: Mapping[str, Any]) -> dict:
    """Remove all MCP tool ids from both allow-lists (explicit API clear)."""
    regular: List[str] = []
    for raw in list(doc.get("allowed_tools") or []):
        tid = str(raw or "").strip()
        if not tid or is_mcp_public_tool_id(tid):
            continue
        if tid not in regular:
            regular.append(tid)
    out = dict(doc)
    out["allowed_tools"] = _dedupe_preserve_order(regular)
    out["allowed_mcp_tools"] = []
    return out


def apply_agent_tool_allowlist_normalization(doc: Mapping[str, Any]) -> dict:
    """Return a copy of an agent config doc with canonical allow-lists."""
    reg, mcp = normalize_agent_tool_allowlists(
        doc.get("allowed_tools"),
        doc.get("allowed_mcp_tools"),
    )
    out = dict(doc)
    out["allowed_tools"] = reg
    out["allowed_mcp_tools"] = mcp
    return out


def effective_allowed_tool_ids(
    config: Mapping[str, Any] | Any,
    *,
    allowed_tools: Optional[List[str]] = None,
    allowed_mcp_tools: Optional[List[str]] = None,
) -> List[str]:
    """Union of regular and MCP allow-lists (backward compatible with legacy single list)."""
    if allowed_tools is None or allowed_mcp_tools is None:
        if isinstance(config, Mapping):
            allowed_tools = (
                list(allowed_tools)
                if allowed_tools is not None
                else list(config.get("allowed_tools") or [])
            )
            allowed_mcp_tools = (
                list(allowed_mcp_tools)
                if allowed_mcp_tools is not None
                else list(config.get("allowed_mcp_tools") or [])
            )
        else:
            allowed_tools = list(
                allowed_tools
                if allowed_tools is not None
                else getattr(config, "allowed_tools", []) or []
            )
            allowed_mcp_tools = list(
                allowed_mcp_tools
                if allowed_mcp_tools is not None
                else getattr(config, "allowed_mcp_tools", []) or []
            )
    reg, mcp = normalize_agent_tool_allowlists(allowed_tools, allowed_mcp_tools)
    return _dedupe_preserve_order(reg + mcp)


def split_allowed_tools_for_ui(
    allowed_tools: List[str],
    allowed_mcp_tools: Optional[List[str]],
    *,
    known_mcp_tool_ids: Optional[set[str]] = None,
) -> tuple[List[str], List[str]]:
    """Split legacy combined ``allowed_tools`` for the agent settings UI."""
    reg, mcp = normalize_agent_tool_allowlists(allowed_tools, allowed_mcp_tools)
    if mcp or not known_mcp_tool_ids:
        return reg, mcp
    mcp_out = [t for t in (allowed_tools or []) if t in known_mcp_tool_ids]
    reg_out = [t for t in (allowed_tools or []) if t not in known_mcp_tool_ids]
    return _dedupe_preserve_order(reg_out), _dedupe_preserve_order(mcp_out)
