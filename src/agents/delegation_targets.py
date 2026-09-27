"""Resolve configured delegation targets to real project-agent instances."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from config.agent_delegation_identity import (
    agent_identities_equal,
    agent_identity_from_runtime,
    agent_identity_matches_reference,
)
from config.configuration_resolution import agent_display_name_from_doc
from schemas.configuration_schemas import entity_short_description_from_doc

from .delegation_policy import (
    ALL_DELEGATION_TARGETS_TOKEN,
    DELEGATE_TO_AGENT_TOOL,
    SYSTEM_TENANT_ID,
)

logger = logging.getLogger(__name__)


def candidate_wire_name(candidate: Any, *, parent_tenant: str = "") -> str:
    """Return the runtime wire name, never the opaque MongoDB storage id."""
    _, name = agent_identity_from_runtime(
        candidate,
        parent_tenant=parent_tenant or None,
    )
    return name


def configured_tenant_id(agent: Any) -> str:
    config = getattr(agent, "config", None)
    if isinstance(config, dict) and config.get("tenant_id"):
        return str(config["tenant_id"])
    tenant_id = getattr(agent, "tenant_id", None)
    return str(tenant_id) if tenant_id else ""


def parent_tenant_id(agent: Any) -> str:
    shared_context = getattr(agent, "shared_context", None)
    tenant_id = getattr(shared_context, "tenant_id", None)
    if tenant_id:
        return str(tenant_id)
    return configured_tenant_id(agent)


def target_display_name(target: Any) -> str:
    """One name for a pool agent as a delegation target: the target catalog lists it, delegation events record it."""
    get_display_name = getattr(target, "get_display_name", None)
    if callable(get_display_name):
        return get_display_name()
    config = getattr(target, "config", None)
    return agent_display_name_from_doc(
        config if isinstance(config, dict) else {},
        runtime_tenant_id=configured_tenant_id(target) or None,
    )


def find_project_agent(parent: Any, requested_agent_id: str) -> Any | None:
    """Resolve one wire-name target reference against the project-local pool."""
    agent_pool = getattr(parent, "agent_pool", None)
    if not isinstance(agent_pool, Iterable):
        return None

    requested = str(requested_agent_id or "").strip()
    tenant_id = parent_tenant_id(parent)

    for candidate in agent_pool:
        if str(getattr(candidate, "agent_id", "") or "") == requested:
            return candidate

    candidates: list[Any] = []
    for candidate in agent_pool:
        config = getattr(candidate, "config", None)
        if not isinstance(config, dict):
            config = {}
        if agent_identity_matches_reference(
            config,
            requested,
            parent_tenant=tenant_id,
            runtime_agent=candidate,
        ):
            candidates.append(candidate)

    return _prefer_target_for_parent_tenant(parent, candidates)


def resolve_project_delegation_targets(parent: Any) -> list[dict[str, str]]:
    """Return DB-backed wire targets that the current project agent may call."""
    configured = getattr(parent, "allowed_delegation_targets", None)
    agent_pool = getattr(parent, "agent_pool", None)
    if not isinstance(configured, list) or not isinstance(agent_pool, Iterable):
        return []

    requested_targets = [
        str(target).strip()
        for target in configured
        if str(target or "").strip()
    ]
    tenant_id = parent_tenant_id(parent)
    candidates: list[tuple[str, Any]] = []
    if ALL_DELEGATION_TARGETS_TOKEN in requested_targets:
        grouped: dict[str, list[Any]] = {}
        for item in agent_pool:
            wire_name = candidate_wire_name(item, parent_tenant=tenant_id)
            if wire_name:
                grouped.setdefault(wire_name, []).append(item)
        for items in grouped.values():
            preferred = _prefer_target_for_parent_tenant(parent, items)
            if preferred is not None:
                candidates.append((ALL_DELEGATION_TARGETS_TOKEN, preferred))
    else:
        for requested in requested_targets:
            target = find_project_agent(parent, requested)
            if target is None:
                logger.warning(
                    "[DELEGATION] parent=%s configured_target=%s "
                    "reason=not_found_in_project_pool — omitted from LLM target catalog",
                    getattr(parent, "agent_id", "unknown"),
                    requested,
                )
                continue
            candidates.append((requested, target))

    resolved: list[dict[str, str]] = []
    seen: set[str] = set()
    parent_id = str(getattr(parent, "agent_id", "") or "")
    parent_identity = agent_identity_from_runtime(
        parent,
        parent_tenant=tenant_id or None,
    )
    for requested, target in candidates:
        target_identity = agent_identity_from_runtime(
            target,
            parent_tenant=tenant_id or None,
        )
        exact_id = target_identity[1]
        if not exact_id or exact_id in seen:
            continue
        if agent_identities_equal(
            parent_identity[0],
            parent_identity[1],
            target_identity[0],
            target_identity[1],
            scope_tenant=tenant_id or None,
        ):
            continue

        target_tenant = configured_tenant_id(target)
        if target_tenant != SYSTEM_TENANT_ID and (
            not tenant_id or not target_tenant or target_tenant != tenant_id
        ):
            logger.warning(
                "[DELEGATION] parent=%s configured_target=%s resolved_target=%s "
                "parent_tenant=%s target_tenant=%s reason=tenant_scope "
                "— omitted from LLM target catalog",
                parent_id or "unknown",
                requested,
                exact_id,
                tenant_id or "unknown",
                target_tenant or "unknown",
            )
            continue

        config = getattr(target, "config", None)
        config = config if isinstance(config, dict) else {}
        resolved.append(
            {
                "agent_id": exact_id,
                "name": str(target_display_name(target) or exact_id),
                "description": entity_short_description_from_doc(config),
            }
        )
        seen.add(exact_id)

    return resolved


async def resolve_a2a_delegation_targets(parent: Any) -> list[dict[str, str]]:
    """Return registered a2a servers this agent may delegate to, as catalog entries.

    An a2a server is a valid target only when its name is listed *explicitly* in
    ``allowed_delegation_targets``: the ``*`` wildcard expands to pool agents only, so
    turning on ``*`` never silently exposes every external agent. A name that already
    resolves to a project-pool agent is left to ``resolve_project_delegation_targets``
    (pool wins), so an a2a server never shadows a same-named pool agent. Tenant scope is
    enforced by the tenant-scoped ``get_a2a_server_by_name`` lookup itself.
    """
    configured = getattr(parent, "allowed_delegation_targets", None)
    if not isinstance(configured, list):
        return []
    shared_context = getattr(parent, "shared_context", None)
    storage = getattr(shared_context, "storage", None)
    if storage is None or not callable(getattr(storage, "get_a2a_server_by_name", None)):
        return []
    tenant_id = parent_tenant_id(parent)
    if not tenant_id:
        return []

    resolved: list[dict[str, str]] = []
    seen: set[str] = set()
    for target in configured:
        name = str(target or "").strip()
        if not name or name == ALL_DELEGATION_TARGETS_TOKEN or name in seen:
            continue
        # Pool agents take precedence: a name resolvable in the project pool is the
        # pool resolver's job, not ours — checking here avoids a duplicate enum entry.
        if find_project_agent(parent, name) is not None:
            continue
        try:
            server = await storage.get_a2a_server_by_name(name, tenant_id)
        except Exception as exc:
            logger.warning(
                "[DELEGATION] parent=%s a2a_target=%s reason=lookup_failed err=%s "
                "— omitted from LLM target catalog",
                getattr(parent, "agent_id", "unknown"),
                name,
                exc,
            )
            continue
        if not server or server.get("enabled") is False:
            continue
        summary = server.get("cached_agent_card_summary") or {}
        card_description = str(server.get("description") or "").strip()
        description = "external a2a agent"
        if card_description:
            description = f"{description} — {card_description}"
        resolved.append(
            {
                # The enum value the LLM passes back; must round-trip through
                # get_a2a_server_by_name, so it is the exact name we resolved with.
                "agent_id": name,
                "name": str(summary.get("name") or server.get("name") or name),
                "description": description,
                "kind": "a2a",
            }
        )
        seen.add(name)
    return resolved


def format_delegation_targets_prompt(targets: list[dict[str, str]]) -> str:
    if not targets:
        return ""
    lines = [
        "Allowed delegation targets (authoritative, resolved from the current project agent pool):",
        "Use only the exact `agent_id` values listed below when calling `delegate_to_agent`.",
    ]
    for target in targets:
        line = f"- `{target['agent_id']}` — {target['name']}"
        if target.get("description"):
            line += f": {target['description']}"
        lines.append(line)
    return "\n".join(lines)


def bind_delegation_targets_to_tool_schemas(
    tools: list[dict[str, Any]],
    targets: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Bind exact target wire names to a per-invocation delegation schema copy."""
    target_ids = [target["agent_id"] for target in targets if target.get("agent_id")]
    bound: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or function.get("name") != DELEGATE_TO_AGENT_TOOL:
            bound.append(tool)
            continue
        if not target_ids:
            continue

        delegate_tool = deepcopy(tool)
        delegate_function = delegate_tool.setdefault("function", {})
        parameters = delegate_function.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
            delegate_function["parameters"] = parameters
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            properties = {}
            parameters["properties"] = properties
        agent_id_schema = properties.get("agent_id")
        if not isinstance(agent_id_schema, dict):
            agent_id_schema = {"type": "string"}
            properties["agent_id"] = agent_id_schema
        agent_id_schema["enum"] = target_ids
        agent_id_schema["description"] = (
            "Exact target agent id. Must be one of: " + ", ".join(target_ids)
        )
        bound.append(delegate_tool)
    return bound


def _prefer_target_for_parent_tenant(parent: Any, candidates: list[Any]) -> Any | None:
    if not candidates:
        return None
    tenant_id = parent_tenant_id(parent)
    if tenant_id:
        for candidate in candidates:
            if configured_tenant_id(candidate) == tenant_id:
                return candidate
    for candidate in candidates:
        if configured_tenant_id(candidate) == SYSTEM_TENANT_ID:
            return candidate
    return candidates[0]
