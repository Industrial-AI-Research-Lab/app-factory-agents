"""Helpers for choosing specialized agents from a mixed pool."""

from __future__ import annotations

from typing import Any, Iterable, Optional


def _agent_id_prefix(agent: Any) -> str:
    agent_id = getattr(agent, "agent_id", "") or ""
    return agent_id.split("@")[0]


def select_critic_agent(agents: Iterable[Any]) -> Optional[Any]:
    """
    Select the dedicated auction critic from an agent pool.

    Multiple runtime agents currently share the ``critic`` enum value
    (for example ``human_expert`` and ``critic_expert``), so selecting
    the first ``agent_type == "critic"`` is not stable enough.
    """
    agents = list(agents)

    for agent in agents:
        if _agent_id_prefix(agent) == "critic_expert":
            return agent

    for agent in agents:
        if getattr(getattr(agent, "agent_type", None), "value", "") == "critic":
            return agent

    return None
