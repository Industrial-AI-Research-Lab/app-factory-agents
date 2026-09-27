"""Deterministic parent selection for persisted HITL approval messages."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable


@dataclass(frozen=True)
class ApprovalAttachment:
    """The graph parent selected for one approval message."""

    parent_id: str
    correlation: str
    basis: str


def resolve_approval_attachment(
    approval: dict,
    *,
    approval_time: datetime,
    attempt_windows: list[dict],
    structural_parent_id: str | None,
    structural_correlation: str = "exact",
    run_parent_id: str,
    run_correlation: str = "exact",
    is_descendant: Callable[[str, str], bool],
) -> ApprovalAttachment:
    """Resolve an approval without selecting a future or arbitrary attempt.

    A persisted workflow/phase reference narrows the candidate set and is also
    the fallback for gates that intentionally have no agent attempt.  A match
    based only on timestamps is explicitly inferred because the approval does
    not persist an attempt identity.
    """
    data = approval.get("data") if isinstance(approval.get("data"), dict) else {}
    run_id = approval.get("run_id")
    candidates = [
        window for window in attempt_windows
        if window.get("run_id") == run_id
        and window.get("started_at") is not None
    ]
    if data.get("task_id") is not None:
        candidates = [window for window in candidates if window.get("task_id") == data["task_id"]]
    if data.get("agent_id") is not None:
        candidates = [window for window in candidates if window.get("agent_id") == data["agent_id"]]
    if structural_parent_id:
        candidates = [
            window for window in candidates
            if is_descendant(str(window["node_id"]), structural_parent_id)
        ]

    containing = [
        window for window in candidates
        if window["started_at"] <= approval_time
        and (window.get("completed_at") is None or approval_time <= window["completed_at"])
    ]
    if len(containing) == 1:
        return ApprovalAttachment(containing[0]["node_id"], "inferred", "attempt_time_window")
    if len(containing) > 1:
        return _fallback(structural_parent_id, structural_correlation, run_parent_id, run_correlation,
                         "ambiguous_time_fallback")

    preceding = [window for window in candidates if window["started_at"] <= approval_time]
    if preceding:
        latest_start = max(window["started_at"] for window in preceding)
        nearest = [window for window in preceding if window["started_at"] == latest_start]
        if len(nearest) == 1:
            return ApprovalAttachment(nearest[0]["node_id"], "inferred", "latest_preceding_attempt")

    return _fallback(structural_parent_id, structural_correlation, run_parent_id, run_correlation,
                     "workflow_target" if structural_parent_id else "run_fallback")


def _fallback(
    structural_parent_id: str | None,
    structural_correlation: str,
    run_parent_id: str,
    run_correlation: str,
    basis: str,
) -> ApprovalAttachment:
    # The parent may still be a known structural/run node, but an ambiguous
    # time window does not identify a unique attempt.  Correlation describes
    # attachment confidence (and controls the edge style), so never advertise
    # this fallback as exact.
    if basis == "ambiguous_time_fallback":
        correlation = "inferred"
    elif structural_parent_id:
        correlation = structural_correlation
    else:
        correlation = run_correlation
    if structural_parent_id:
        return ApprovalAttachment(structural_parent_id, correlation, basis)
    return ApprovalAttachment(run_parent_id, correlation, basis)
