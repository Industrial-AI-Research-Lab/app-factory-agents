"""Normalize persisted typed A2A progress into deterministic trace facts."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any


PROGRESS_SCHEMA = "coscientist-a2a-progress-v1"
DELEGATION_STARTED = "delegation.started"
DELEGATION_COMPLETED = "delegation.completed"
TOOL_STARTED = "tool.started"
TOOL_COMPLETED = "tool.completed"
TOOL_FAILED = "tool.failed"


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def normalize_progress(event: dict) -> dict | None:
    """Return one safe typed progress row, or None for unsupported records."""
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    progress = data.get("progress") if isinstance(data.get("progress"), dict) else {}
    sequence = progress.get("sequence")
    if (
        progress.get("schema_version") != PROGRESS_SCHEMA
        or not _text(progress.get("event_id"))
        or not _text(progress.get("run_id"))
        or not _text(progress.get("type"))
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence <= 0
    ):
        return None
    return {
        "outer_event_id": event.get("event_id"),
        "outer_run_id": event.get("run_id"),
        "outer_node_id": data.get("node_id"),
        "timestamp": event.get("timestamp"),
        "event_id": progress["event_id"],
        "remote_run_id": progress["run_id"],
        "sequence": sequence,
        "type": progress["type"],
        "agent": _text(progress.get("agent")),
        "tool_name": _text(progress.get("tool_name")),
    }


def _key(row: dict) -> tuple[str | None, str | None, str, str, str]:
    return (
        row.get("outer_run_id"),
        row.get("outer_node_id"),
        row["remote_run_id"],
        row["agent"],
        row["tool_name"],
    )


def _fact(start: dict | None, terminal: dict | None, *, kind: str, status: str) -> dict:
    anchor = start or terminal
    assert anchor is not None
    base = {
        "outer_run_id": anchor.get("outer_run_id"),
        "outer_node_id": anchor.get("outer_node_id"),
        "remote_run_id": anchor["remote_run_id"],
        "sequence": (start or terminal)["sequence"],
        "status": status,
        # A persisted start is already an exact fact about a running action.
        # Only a terminal record without its matching start needs inference.
        "correlation": "exact" if start else "inferred",
    }
    if kind == "delegation":
        return {
            **base,
            "parent_agent": anchor["agent"],
            "child_agent": anchor["tool_name"],
            "start_event_id": start.get("event_id") if start else None,
            "terminal_event_id": terminal.get("event_id") if terminal else None,
        }
    return {
        **base,
        "agent": anchor["agent"],
        "tool_name": anchor["tool_name"],
        "start_event_id": start.get("event_id") if start else None,
        "terminal_event_id": terminal.get("event_id") if terminal else None,
    }


def pair_a2a_progress(events: list[dict]) -> dict[str, list[dict]]:
    """Pair typed inner A2A lifecycles by durable sequence and structured IDs."""
    normalized = [row for event in events if (row := normalize_progress(event))]
    normalized.sort(
        key=lambda row: (
            str(row.get("outer_run_id") or ""),
            str(row.get("outer_node_id") or ""),
            row["remote_run_id"],
            row["sequence"],
            row["event_id"],
        )
    )
    open_delegations: dict[tuple[str | None, str | None, str, str, str], deque[dict]] = defaultdict(deque)
    open_tools: dict[tuple[str | None, str | None, str, str, str], deque[dict]] = defaultdict(deque)
    delegations: list[dict] = []
    tools: list[dict] = []

    for row in normalized:
        kind = row["type"]
        if not row["agent"] or not row["tool_name"]:
            continue
        key = _key(row)
        if kind == DELEGATION_STARTED:
            open_delegations[key].append(row)
        elif kind == DELEGATION_COMPLETED:
            start = open_delegations[key].popleft() if open_delegations[key] else None
            delegations.append(_fact(start, row, kind="delegation", status="completed"))
        elif kind == TOOL_STARTED:
            open_tools[key].append(row)
        elif kind in {TOOL_COMPLETED, TOOL_FAILED}:
            start = open_tools[key].popleft() if open_tools[key] else None
            tools.append(_fact(start, row, kind="tool", status="failed" if kind == TOOL_FAILED else "completed"))

    for starts in open_delegations.values():
        delegations.extend(_fact(start, None, kind="delegation", status="running") for start in starts)
    for starts in open_tools.values():
        tools.extend(_fact(start, None, kind="tool", status="running") for start in starts)

    def ordering(fact: dict) -> tuple:
        return (
            str(fact.get("outer_run_id") or ""),
            str(fact.get("outer_node_id") or ""),
            fact["remote_run_id"],
            fact["sequence"],
            str(fact.get("start_event_id") or fact.get("terminal_event_id") or ""),
        )
    delegations.sort(key=ordering)
    tools.sort(key=ordering)
    return {"delegations": delegations, "tools": tools}


def has_published_inner_trace(events: list[dict], outer_run_id: str, outer_node_id: str) -> bool:
    """Whether a wrapper published at least one valid typed inner progress event."""
    return any(
        row is not None
        and row.get("outer_run_id") == outer_run_id
        and row.get("outer_node_id") == outer_node_id
        for event in events
        if (row := normalize_progress(event))
    )
