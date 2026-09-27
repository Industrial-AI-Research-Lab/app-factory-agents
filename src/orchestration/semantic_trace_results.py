"""Pure helpers for safe semantic Trace result nodes."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def when(row: dict) -> datetime | None:
    value = row.get("created_at") or row.get("timestamp")
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def select_agent_result(attempt: dict, rows: list[dict], *, closed_attempt_count: int) -> dict | None:
    """Return exactly one stored assistant answer within a closed attempt window."""
    start, end = attempt.get("started_at"), attempt.get("completed_at")
    if start is None or end is None:
        return None
    candidates = []
    for row in rows:
        data = row.get("data") if isinstance(row.get("data"), dict) else {}
        if row.get("run_id") != attempt.get("run_id") or data.get("task_id") != attempt.get("task_id"):
            continue
        timestamp = when(row)
        if data.get("agent_id") != attempt.get("agent_id") or timestamp is None or not start <= timestamp <= end:
            continue
        stored_attempt = data.get("attempt")
        if stored_attempt is None and closed_attempt_count != 1:
            continue
        if stored_attempt is not None and stored_attempt != attempt.get("attempt"):
            continue
        candidates.append(row)
    return {"id": candidates[0]["id"], "correlation": "exact"} if len(candidates) == 1 else None


def add_result(graph: Any, parent_id: str, run_id: str | None, stable_id: str, *, source: str,
               source_message_id: str | None, status: str = "completed", sequence: int | None = None,
               correlation: str = "exact") -> str:
    node_id = f"result:{stable_id}"
    recorded = source_message_id is not None
    graph.node(node_id=node_id, node_type="result", parent_id=parent_id, run_id=run_id,
               title="Result", summary="available" if recorded else "not recorded",
               details={"display_status": status, "result_source": source if recorded else "not_recorded",
                        "source_message_id": source_message_id, "payload_available": recorded},
               correlation=correlation if recorded else "inferred", sequence=sequence)
    graph.edge(parent_id, node_id, "produced", correlation if recorded else "inferred")
    return node_id
