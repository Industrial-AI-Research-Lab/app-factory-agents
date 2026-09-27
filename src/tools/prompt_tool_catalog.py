from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ALLOWED_TOOLS_CATALOG_PLACEHOLDER = "{{ALLOWED_TOOLS_CATALOG}}"
TOOL_CATALOG_INTRO_LINE = "The following tools are available (from configuration):"
MAX_TOOL_DESCRIPTION_LEN = 300


def apply_tool_catalog_placeholder(system_prompt: str, catalog_text: str) -> str:
    """Insert ``catalog_text`` at the placeholder, or strip placeholder + intro when catalog is empty.

    When ``catalog_text`` is empty/whitespace, removes any line containing the placeholder
    and the immediately preceding line if it matches ``TOOL_CATALOG_INTRO_LINE`` (after strip),
    so YAML prompts do not leave a dangling header above an empty list.
    """
    ph = ALLOWED_TOOLS_CATALOG_PLACEHOLDER
    if ph not in system_prompt:
        return system_prompt
    if (catalog_text or "").strip():
        return system_prompt.replace(ph, catalog_text)
    lines = system_prompt.splitlines(keepends=True)
    out: List[str] = []
    for line in lines:
        if ph in line:
            if out and out[-1].rstrip("\r\n").strip() == TOOL_CATALOG_INTRO_LINE:
                out.pop()
            continue
        out.append(line)
    return "".join(out)


def _normalize_description_line(line: str) -> str:
    """Collapse redundant horizontal whitespace without destroying line breaks or leading indent."""
    s = line.replace("\t", " ")
    # Multiple spaces/tabs only between two non-whitespace chars (keeps ``  - item`` indents).
    s = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", s)
    return s.rstrip()


def _truncate_description(text: str, max_len: int = MAX_TOOL_DESCRIPTION_LEN) -> str:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(_normalize_description_line(line) for line in raw.split("\n"))
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip("\n")
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3].rstrip() + "..."


def tool_doc_allowed_for_agent(doc: Dict[str, Any], agent_id: Optional[str]) -> bool:
    """Deprecated: tool documents no longer gate on agent lists (``allowed_tools`` only).

    Kept for backward compatibility in tests; always returns True.
    """
    return True


async def build_tool_catalog_text(
    allowed_tool_ids: List[str],
    storage: Any = None,
    tenant_id: Optional[str] = None,
    registry_tools: Optional[List[Dict[str, Any]]] = None,
    agent_context: str = "unknown",
    agent_id: Optional[str] = None,
) -> str:
    """Build human-readable tool catalog block for prompts.

    Registry rows (``registry_tools``) are the primary source for allowed tool ids;
    Mongo bulk ``get_tool_configurations`` runs only for ids not found there, and
    never overwrites registry entries.

    If storage reads fail while tool ids are still unresolved, returns ``""`` so the
    caller can keep the base system prompt without a misleading catalog (stays aligned
    with schema loading, which does not depend on this helper's error paths).

    Output format per tool:
    - `tool_id` (Tool Name): description
    """
    if not allowed_tool_ids:
        return ""

    allowed_set = frozenset(allowed_tool_ids)
    tool_map: Dict[str, Dict[str, Any]] = {}

    # Refs whose effective doc is disabled: both schema builders drop those after
    # the same ranking, so listing them here would promise the model a tool that
    # dispatch refuses — and a later pass must not resolve them from another doc.
    suppressed: set[str] = set()

    # Primary: in-memory registry (same source as get_schemas_for_agent when registry is used).
    # Visibility and key matching MUST mirror the schema path (ToolRegistry):
    # allowed_tools references builtins by wire name while rows key on the
    # storage UUID, and builtins are tenant-global — resolving by id-only or
    # tenant-strict rules here made the prompt claim "(not configured)" for
    # tools the agent could actually call.
    if registry_tools:
        from config.tool_configuration_schema import tool_doc_tenant_rank
        from tools.tool_registry import ToolRegistry

        ranked: Dict[str, int] = {}
        for t in registry_tools:
            if not isinstance(t, dict):
                continue
            if not ToolRegistry._tool_matches_tenant(t, tenant_id):
                continue
            rank = tool_doc_tenant_rank(t, tenant_id=tenant_id)
            for key in ToolRegistry._tool_allowed_keys(t):
                if key not in allowed_set:
                    continue
                if key in ranked and rank <= ranked[key]:
                    continue
                tool_map[key] = t
                ranked[key] = rank
        suppressed = {k for k, doc in tool_map.items() if not doc.get("enabled", True)}
        for key in suppressed:
            tool_map.pop(key, None)

    missing_after_registry = {
        tid
        for tid in allowed_tool_ids
        if tid not in tool_map and tid not in suppressed
    }
    storage_error = False

    # Bulk Mongo read only for tool_ids not covered by registry (one round-trip, no overwrite of registry).
    if missing_after_registry and storage and hasattr(storage, "get_tool_configurations"):
        try:
            from storage.tool_doc_storage import get_all_tool_configurations_for_registry

            docs = await get_all_tool_configurations_for_registry(
                storage,
                enabled_only=True,
                tenant_id=tenant_id,
            )
            from tools.mcp_tool_ids import mcp_public_tool_id_from_doc

            from config.tool_configuration_schema import (
                tool_doc_tenant_rank,
                tool_wire_name_from_doc,
            )

            db_ranked: Dict[str, int] = {}
            for d in docs:
                if not isinstance(d, dict) or not isinstance(d.get("_id"), str):
                    continue
                tid = d["_id"]
                if d.get("source") == "mcp_server":
                    pub = mcp_public_tool_id_from_doc(d)
                    if pub:
                        tid = pub
                # Same key set as the registry pass: allowed ids may be the
                # storage id OR the wire/stored name (builtins).
                keys = {tid}
                wire = tool_wire_name_from_doc(d)
                if wire:
                    keys.add(wire)
                stored_name = str(d.get("name") or "").strip()
                if stored_name:
                    keys.add(stored_name)
                rank = tool_doc_tenant_rank(d, tenant_id=tenant_id)
                for key in keys:
                    if key not in missing_after_registry:
                        continue
                    if key in db_ranked and rank <= db_ranked[key]:
                        continue
                    tool_map[key] = d
                    db_ranked[key] = rank
            db_suppressed = {
                k for k in db_ranked if not tool_map[k].get("enabled", True)
            }
            for key in db_suppressed:
                tool_map.pop(key, None)
            suppressed |= db_suppressed
        except Exception as e:
            storage_error = True
            logger.warning(
                "[TOOL_PROMPT_CATALOG] agent_context=%s tenant_id=%s — failed to load tool catalog from storage: %s",
                agent_context,
                tenant_id,
                e,
            )

    still_missing = [
        tid
        for tid in allowed_tool_ids
        if tid not in tool_map and tid not in suppressed
    ]
    if still_missing and storage and hasattr(storage, "get_tool_configuration"):
        for tid in still_missing:
            if tid in tool_map:
                continue
            try:
                from tools.mcp_tool_ids import resolve_mcp_tool_doc

                doc = await resolve_mcp_tool_doc(storage, tenant_id, tid)
                if doc is None:
                    doc = await storage.get_tool_configuration(tid)
            except Exception as e:
                storage_error = True
                doc = None
                logger.warning(
                    "[TOOL_PROMPT_CATALOG] agent_context=%s tool_id=%s — get_tool_configuration failed: %s",
                    agent_context,
                    tid,
                    e,
                )
            if isinstance(doc, dict):
                tool_map[tid] = doc

    unresolved = [
        tid
        for tid in allowed_tool_ids
        if tid not in tool_map and tid not in suppressed
    ]
    if unresolved and storage_error:
        logger.warning(
            "[TOOL_PROMPT_CATALOG] agent_context=%s — omitting tool catalog (storage error, "
            "unresolved_ids=%s) to avoid prompt/schema desync",
            agent_context,
            unresolved,
        )
        return ""

    lines: List[str] = []
    for tool_id in allowed_tool_ids:
        if tool_id in suppressed:
            continue
        doc = tool_map.get(tool_id)
        if not doc and storage and hasattr(storage, "get_tool_configuration"):
            try:
                from tools.mcp_tool_ids import resolve_mcp_tool_doc

                doc = await resolve_mcp_tool_doc(storage, tenant_id, tool_id)
                if doc is None:
                    doc = await storage.get_tool_configuration(tool_id)
            except Exception as e:
                storage_error = True
                doc = None
                logger.warning(
                    "[TOOL_PROMPT_CATALOG] agent_context=%s tool_id=%s — get_tool_configuration failed: %s",
                    agent_context,
                    tool_id,
                    e,
                )

        if not doc:
            if storage_error:
                logger.warning(
                    "[TOOL_PROMPT_CATALOG] agent_context=%s tool_id=%s — omitting catalog "
                    "(storage error mid-build)",
                    agent_context,
                    tool_id,
                )
                return ""
            logger.warning(
                "[TOOL_PROMPT_CATALOG] agent_context=%s tool_id=%s — missing tool_configuration",
                agent_context,
                tool_id,
            )
            lines.append(f"- `{tool_id}` (unknown): (not configured)")
            continue

        name = str(doc.get("name") or tool_id)
        from config.tool_configuration_schema import tool_description_from_doc, tool_wire_name_from_doc

        description = _truncate_description(tool_description_from_doc(doc) or "(no description)")
        from tools.agent_tool_schemas import existing_function_name

        registry_tool_id = doc.get("tool_id")
        wire_name = existing_function_name(doc.get("schema"))
        if not wire_name:
            if isinstance(registry_tool_id, str) and registry_tool_id.strip():
                wire_name = registry_tool_id.strip()
            else:
                wire_name = tool_wire_name_from_doc(doc) or tool_id
        lines.append(f"- `{wire_name}` ({name}): {description}")

    return "\n".join(lines)

