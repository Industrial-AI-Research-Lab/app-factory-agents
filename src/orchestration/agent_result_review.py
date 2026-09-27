"""Typed contract shared by persistent agent-result approval gates."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


AGENT_RESULT_REVIEW_TYPE = "agent_result_review"


class AgentResultDecision(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"


def build_agent_result_interaction_schema() -> Dict[str, Any]:
    """Return the frontend/backend contract for reviewing an agent result."""
    return {
        "type": AGENT_RESULT_REVIEW_TYPE,
        "version": 1,
        "decision_field": "decision",
        "decisions": [decision.value for decision in AgentResultDecision],
        "feedback_required_for": [AgentResultDecision.REVISE.value],
    }


def decision_from_approval(approval: Any) -> Optional[AgentResultDecision]:
    """Read an exact typed decision; never infer intent from feedback text."""
    if not isinstance(approval, dict):
        return None
    response = approval.get("interaction_response")
    if not isinstance(response, dict):
        resolution = (approval.get("data") or {}).get("resolution")
        if isinstance(resolution, dict):
            response = resolution.get("interaction_response")
    raw_decision = response.get("decision") if isinstance(response, dict) else None
    try:
        return AgentResultDecision(raw_decision)
    except (TypeError, ValueError):
        return None
