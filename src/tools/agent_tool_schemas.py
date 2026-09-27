"""Tool schema loading helpers.

Primary source is MongoDB ``tool_configurations`` via
``get_tool_schemas_for_agent``.
``get_tool_schemas`` is a legacy compatibility helper that reads schemas from
``config/tools.yaml`` (no hardcoded in-code dict).
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Dict, Iterable, List, Optional

import yaml

logger = logging.getLogger(__name__)

ToolSchema = Dict[str, object]


def existing_function_name(schema: object) -> Optional[str]:
    """Return ``function.name`` from an already-wrapped OpenAI tool schema, if present."""
    if not isinstance(schema, dict) or schema.get("type") != "function":
        return None
    fn = schema.get("function")
    if not isinstance(fn, dict):
        return None
    name = fn.get("name")
    if not name:
        return None
    return str(name)


def ensure_openai_function_schema(
    schema: object,
    *,
    function_name: str,
    description: str = "",
) -> ToolSchema:
    """Wrap bare JSON Schema parameters into OpenAI function-calling tool shape."""
    name = str(function_name or "").strip()
    desc = str(description or "")
    if not isinstance(schema, dict):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
        out = deepcopy(schema)
        fn = out["function"]
        if name:
            fn["name"] = name
        if desc and not fn.get("description"):
            fn["description"] = desc
        return out
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": schema,
        },
    }


_TOOLS_YAML = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "tools.yaml")


def get_tool_schemas(tool_ids: Iterable[str]) -> List[ToolSchema]:
    """Legacy helper: load schemas from tools.yaml for specified tool ids."""
    wanted = set(tool_ids or [])
    if not wanted:
        return []
    try:
        with open(_TOOLS_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.error("[TOOL_SCHEMAS] Failed to read tools.yaml: %s", e)
        return []

    tools = data.get("tools", []) if isinstance(data, dict) else []
    schemas: List[ToolSchema] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_id = tool.get("_id")
        schema = tool.get("schema")
        if tool_id in wanted and isinstance(schema, dict):
            schemas.append(deepcopy(schema))
    return schemas


def _doc_matches_tenant(doc: dict, tenant_id: Optional[str]) -> bool:
    if doc.get("source") != "mcp_server":
        return True
    if not tenant_id:
        return False
    doc_tenant = str(doc.get("tenant_id") or "__root__")
    # Shared platform offering (ADR-0013): visible to every tenant, same rule as
    # dispatch; select_effective_tool_docs_for_allow prefers the tenant's fork.
    if doc_tenant == "__system__":
        return True
    return doc_tenant == str(tenant_id)


async def get_tool_schemas_for_agent(
    agent_id: str,
    storage,
    allowed_tool_ids: Optional[List[str]] = None,
    *,
    tenant_id: Optional[str] = None,
) -> List[ToolSchema]:
    """Load tool schemas from MongoDB gated strictly by ``allowed_tool_ids``.

    ``agent_id`` is retained for call-site compatibility and logging only.
    Missing/empty ``allowed_tool_ids`` is deny-by-default and returns ``[]``.
    """
    if not storage:
        logger.error("[TOOL_SCHEMAS] Storage is required for DB-backed schemas")
        return []

    try:
        from storage.tool_doc_storage import get_all_tool_configurations_for_registry

        docs = await get_all_tool_configurations_for_registry(
            storage,
            enabled_only=False,
            tenant_id=tenant_id,
        )
    except Exception as e:
        logger.error("[TOOL_SCHEMAS] DB read failed: %s", e)
        return []

    if not docs:
        logger.warning("[TOOL_SCHEMAS] tool_count=0 in DB")
        return []

    from tools.mcp_tool_ids import mcp_public_tool_id_from_doc
    from config.tool_configuration_schema import (
        drop_disabled_effective_docs,
        render_tool_openai_schema,
        select_effective_tool_docs_for_allow,
    )

    if not allowed_tool_ids:
        logger.debug(
            "[TOOL_SCHEMAS] agent_id=%s empty allowed_tool_ids -> [] (strict)",
            agent_id,
        )
        return []

    allow = frozenset(allowed_tool_ids)
    selected_docs = drop_disabled_effective_docs(
        select_effective_tool_docs_for_allow(
            [doc for doc in docs if isinstance(doc, dict) and _doc_matches_tenant(doc, tenant_id)],
            allow,
            tenant_id=tenant_id,
        )
    )
    schemas: List[ToolSchema] = []
    seen_public: set[str] = set()
    for doc in selected_docs:
        pub = mcp_public_tool_id_from_doc(doc) if isinstance(doc, dict) else None
        if pub and pub in seen_public:
            continue
        if pub:
            seen_public.add(pub)

        out = render_tool_openai_schema(doc)
        schemas.append(out)

    logger.debug("[TOOL_SCHEMAS] Loaded %d schemas for agent '%s' from DB", len(schemas), agent_id)
    return schemas
