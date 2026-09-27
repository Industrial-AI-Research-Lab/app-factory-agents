"""Wire ``name`` generation for MCP tools (path A: ``name`` = OpenAI wire, ``rpc_name`` = MCP RPC)."""

from __future__ import annotations

import re
from typing import Iterable

from tools.mcp_tool_ids import OPENAI_FUNCTION_NAME_MAX_LEN, clamp_openai_function_name

_SERVER_ABBR_STOP_WORDS = frozenset({
    "api",
    "mcp",
    "management",
    "server",
    "service",
})
_SERVER_ABBR_MAX_WORDS = 3
_LLM_FUNCTION_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


class LlmFunctionNameError(ValueError):
    """Invalid or duplicate ``llm_function_name``."""


def _builtin_tool_ids() -> frozenset[str]:
    from tools.agent_allowed_tools import known_builtin_tool_ids

    return known_builtin_tool_ids()


def pascal_to_snake(segment: str) -> str:
    """``GetProjectTerritoryByProjectId`` → ``get_project_territory_by_project_id``."""
    text = str(segment or "").strip()
    if not text:
        return ""
    spaced = re.sub(r"(?<!^)(?=[A-Z])", "_", text)
    return re.sub(r"_+", "_", spaced).strip("_").lower()


def rpc_method_snake(rpc_name: str) -> str:
    """Last ``-`` segment of MCP RPC ``name``, PascalCase → snake_case."""
    raw = str(rpc_name or "").strip()
    if not raw:
        return ""
    method = raw.rsplit("-", 1)[-1]
    return pascal_to_snake(method)


def _server_slug_words(server_id: str) -> list[str]:
    words = [
        w
        for w in str(server_id or "").strip().lower().split("-")
        if w and w not in _SERVER_ABBR_STOP_WORDS
    ]
    if words:
        return words
    return [w for w in str(server_id or "").strip().lower().split("-") if w]


def propose_server_abbr_base(server_id: str) -> str:
    """Deterministic short server code from ``server-name`` (layer 2)."""
    words = _server_slug_words(server_id)
    if not words:
        return "mcp"
    picked = words[:_SERVER_ABBR_MAX_WORDS]
    return "".join(w[0] for w in picked)


def resolve_server_abbrs(server_ids: Iterable[str]) -> dict[str, str]:
    """Map each server id to a unique abbr per batch (``usp``, ``usp2``, …)."""
    ordered = sorted({str(s or "").strip() for s in server_ids if str(s or "").strip()})
    base_groups: dict[str, list[str]] = {}
    bases: dict[str, str] = {}
    for sid in ordered:
        base = propose_server_abbr_base(sid)
        bases[sid] = base
        base_groups.setdefault(base, []).append(sid)

    resolved: dict[str, str] = {}
    for base, sids in base_groups.items():
        for index, sid in enumerate(sids):
            resolved[sid] = base if index == 0 else f"{base}{index + 1}"
    return resolved


def preserved_wire_name_for_identity(
    existing_doc: dict | None,
    server_id: str,
    rpc_name: str,
) -> str | None:
    """Return stored wire ``name`` when ``mcp_server`` and ``rpc_name`` are unchanged."""
    from tools.mcp_tool_ids import mcp_rpc_name_from_doc

    if not isinstance(existing_doc, dict):
        return None
    sid = str(server_id or "").strip()
    rpc = str(rpc_name or "").strip()
    if not sid or not rpc:
        return None
    if str(existing_doc.get("mcp_server") or "").strip() != sid:
        return None
    if mcp_rpc_name_from_doc(existing_doc) != rpc:
        return None
    wire = str(existing_doc.get("name") or "").strip()
    legacy = str(existing_doc.get("llm_function_name") or "").strip()
    if wire and wire != rpc and _LLM_FUNCTION_NAME_RE.match(wire):
        return wire
    return legacy or None


def preserved_llm_function_name_for_identity(
    existing_doc: dict | None,
    server_id: str,
    rpc_name: str,
) -> str | None:
    """Backward-compatible alias for discovery during path B → A migration."""
    return preserved_wire_name_for_identity(existing_doc, server_id, rpc_name)


def _unique_llm_wire(base: str, existing_names: set[str]) -> str:
    candidate = clamp_openai_function_name(str(base or "").strip())
    if not candidate:
        raise LlmFunctionNameError("llm_function_name base is empty")
    if candidate not in existing_names:
        return candidate
    suffix = 2
    while suffix < 1000:
        alt = clamp_openai_function_name(f"{candidate}_{suffix}")
        if alt not in existing_names:
            return alt
        suffix += 1
    raise LlmFunctionNameError(f"cannot uniquify llm_function_name: {candidate!r}")


def propose_llm_function_name(
    server_abbr: str,
    rpc_name: str,
    existing_names: set[str] | frozenset[str] | None = None,
) -> str:
    """Build ``{server_abbr}_{snake_method}`` unique within ``existing_names``."""
    abbr = str(server_abbr or "").strip().lower()
    method = rpc_method_snake(rpc_name)
    if not abbr or not method:
        raise LlmFunctionNameError(
            f"cannot propose llm_function_name from server_abbr={server_abbr!r} rpc_name={rpc_name!r}"
        )
    base = f"{abbr}_{method}"
    taken = set(existing_names or ()) | _builtin_tool_ids()
    wire = _unique_llm_wire(base, taken)
    return validate_llm_function_name(wire)


def validate_llm_function_name(value: str) -> str:
    """Enforce OpenAI Chat Completions function name rules."""
    raw = str(value or "").strip()
    if not raw:
        raise LlmFunctionNameError("llm_function_name must be non-empty")
    if len(raw) > OPENAI_FUNCTION_NAME_MAX_LEN:
        raise LlmFunctionNameError(
            f"llm_function_name must be at most {OPENAI_FUNCTION_NAME_MAX_LEN} characters"
        )
    if not _LLM_FUNCTION_NAME_RE.match(raw):
        raise LlmFunctionNameError(
            "llm_function_name must match [a-zA-Z0-9_-]+"
        )
    return raw


def assign_wire_name_to_mcp_doc(
    doc: dict,
    *,
    existing_names: set[str],
    server_abbr_map: dict[str, str] | None = None,
    preserve_existing: bool = True,
) -> str:
    """Set path-A fields: wire ``name`` and ``rpc_name`` (MCP RPC contract name)."""
    from tools.mcp_tool_ids import mcp_rpc_name_from_doc

    if doc.get("source") != "mcp_server":
        return str(doc.get("name") or "").strip()

    server_id = str(doc.get("mcp_server") or "").strip()
    rpc_name = str(doc.get("rpc_name") or "").strip() or mcp_rpc_name_from_doc(doc)
    if not server_id or not rpc_name:
        return str(doc.get("name") or "").strip()

    doc["rpc_name"] = rpc_name

    current_wire = str(doc.get("name") or "").strip()
    legacy_wire = str(doc.get("llm_function_name") or "").strip()
    # A preserved wire name equal to a builtin tool id would shadow that builtin
    # at dispatch for the whole tenant (AppFactory-268) — regenerate instead.
    if preserve_existing and current_wire and current_wire != rpc_name:
        if _LLM_FUNCTION_NAME_RE.match(current_wire):
            wire = clamp_openai_function_name(current_wire)
            if wire not in _builtin_tool_ids():
                doc["name"] = wire
                doc.pop("llm_function_name", None)
                existing_names.add(wire)
                return wire
    if preserve_existing and legacy_wire:
        wire = clamp_openai_function_name(legacy_wire)
        if wire not in _builtin_tool_ids():
            doc["name"] = wire
            doc.pop("llm_function_name", None)
            existing_names.add(wire)
            return wire

    abbr_map = server_abbr_map or resolve_server_abbrs([server_id])
    server_abbr = abbr_map.get(server_id) or propose_server_abbr_base(server_id)
    wire = propose_llm_function_name(server_abbr, rpc_name, existing_names)
    doc["name"] = wire
    doc.pop("llm_function_name", None)
    existing_names.add(wire)
    return wire


def assign_llm_function_name_to_mcp_doc(
    doc: dict,
    *,
    existing_names: set[str],
    server_abbr_map: dict[str, str] | None = None,
    preserve_existing: bool = True,
) -> str:
    """Backward-compatible alias: writes wire ``name`` (path A), not ``llm_function_name``."""
    return assign_wire_name_to_mcp_doc(
        doc,
        existing_names=existing_names,
        server_abbr_map=server_abbr_map,
        preserve_existing=preserve_existing,
    )
