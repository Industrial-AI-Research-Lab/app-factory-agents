from __future__ import annotations

from typing import Any, Optional

from config.agent_delegation_identity import (
    bare_delegation_name,
    base_agent_id,
)

DELEGATE_TO_AGENT_TOOL = "delegate_to_agent"
ALL_DELEGATION_TARGETS_TOKEN = "*"
SYSTEM_TENANT_ID = "__system__"


def delegation_activation_error(agent: Any, task: Any) -> Optional[str]:
    """Return why delegate_to_agent is inactive for this parent agent/task."""
    allowed_tools = getattr(agent, "allowed_tools", []) or []
    if DELEGATE_TO_AGENT_TOOL not in allowed_tools:
        return "delegate_to_agent is not in parent agent allowed_tools"

    if not isinstance(task, dict) or task.get("can_delegate") is not True:
        return "Current task does not allow delegation"

    if not normalize_delegation_targets(getattr(agent, "allowed_delegation_targets", None)):
        return "allowed_delegation_targets is not configured"

    return None


def delegation_target_error(agent: Any, target_base_id: str) -> Optional[str]:
    """Return why the requested target is not allowed by parent target policy."""
    normalized = normalize_delegation_targets(
        getattr(agent, "allowed_delegation_targets", None),
        tenant_id=_agent_tenant_id(agent),
    )
    if not normalized:
        return "allowed_delegation_targets is not configured"
    if ALL_DELEGATION_TARGETS_TOKEN in normalized:
        return None

    target_base_id = base_agent_id(target_base_id)
    tenant_id = _agent_tenant_id(agent)
    if not delegation_target_matches(normalized, target_base_id, tenant_id):
        return f"Agent '{target_base_id}' is not in allowed_delegation_targets"
    return None


def delegation_target_matches(
    normalized_targets: set[str],
    target_base_id: str,
    tenant_id: Optional[str] = None,
) -> bool:
    """Return whether ``target_base_id`` is allowed by configured delegation targets."""
    if not target_base_id:
        return False
    if ALL_DELEGATION_TARGETS_TOKEN in normalized_targets:
        return True
    bare_target = bare_delegation_name(target_base_id, tenant_id)
    return bare_target in normalized_targets


def _agent_tenant_id(agent: Any) -> Optional[str]:
    shared_context = getattr(agent, "shared_context", None)
    tenant_id = getattr(shared_context, "tenant_id", None)
    if tenant_id:
        return str(tenant_id)

    config = getattr(agent, "config", None)
    if isinstance(config, dict) and config.get("tenant_id"):
        return str(config["tenant_id"])

    tenant_id = getattr(agent, "tenant_id", None)
    if tenant_id:
        return str(tenant_id)
    return None


def normalize_delegation_targets(
    targets: Any,
    *,
    tenant_id: Optional[str] = None,
) -> set[str]:
    """Normalize configured delegation target names for policy checks."""
    if not isinstance(targets, list):
        return set()
    normalized: set[str] = set()
    for target in targets:
        text = str(target or "").strip()
        if not text:
            continue
        if text == ALL_DELEGATION_TARGETS_TOKEN:
            normalized.add(text)
            continue
        normalized.add(bare_delegation_name(text, tenant_id))
    return normalized
