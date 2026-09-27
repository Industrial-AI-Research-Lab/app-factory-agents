"""Read-only data access and serialization for the semantic Trace canvas."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from orchestration.full_trace_projector import project_full_trace


logger = logging.getLogger(__name__)

EVENT_LIMIT = 5000
LLM_LIMIT = 5000
AGENT_RESULT_LIMIT = 5000
SNAPSHOT_LIMIT = 1000
ARTIFACT_LIMIT = 1000
DEPLOYMENT_LIMIT = 1000
LEDGER_LIMIT = 10000
PAYLOAD_VALUE_LIMIT = 64 * 1024
PAYLOAD_PAIR_LIMIT = 128 * 1024


class InvalidTracePayloadSelection(ValueError):
    """The requested payload node is not a visible semantic tool node."""


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value) if value is not None else None


def _data(row: dict) -> dict:
    return row.get("data") if isinstance(row.get("data"), dict) else {}


def _bounded(value: Any, limit: int = PAYLOAD_VALUE_LIMIT) -> Any:
    """Return the stored value or a bounded preview, preserving archive refs."""
    if isinstance(value, dict) and value.get("kind") == "archive_ref":
        return value
    try:
        raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        raw_bytes = len(raw.encode("utf-8"))
    except Exception:
        raw = str(value)
        raw_bytes = len(raw.encode("utf-8", errors="replace"))
    if raw_bytes <= limit:
        return value
    return {
        "preview": raw.encode("utf-8")[:limit].decode("utf-8", errors="ignore"),
        "truncated": True,
        "original_bytes": raw_bytes,
    }


def _payload_bytes(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
    except Exception:
        return len(str(value).encode("utf-8", errors="replace"))


def _window_items(result: Any) -> tuple[list[dict], bool]:
    if isinstance(result, dict):
        return list(result.get("items") or []), bool(result.get("limit_reached"))
    return list(result or []), False


def _event_needs_run_attribution(row: dict) -> bool:
    """Keep unscoped project lifecycle records at project level."""
    data = _data(row)
    event_type = str(row.get("event_type") or "")
    return bool(
        data.get("task_id")
        or data.get("agent_id")
        or data.get("workflow_node_id")
        or data.get("phase")
        or data.get("snapshot_id")
        or event_type.startswith(("task_", "auction", "a2a_", "agent.delegation"))
        or event_type in {"rollback", "workflow_rollback"}
    )


def _with_effective_run_scope(
    rows: list[dict], sole_run_id: str | None, *, source: str,
    live_run_ids: set[str] | None = None,
) -> tuple[list[dict], int, int]:
    """Copy rows with exact, sole-run inferred, or unresolved run attribution.

    Explicit references to non-live runs stay unresolved so a deleted branch is
    never recreated.  Only genuinely unscoped rows may use the sole-live-run
    fallback.
    """
    normalized: list[dict] = []
    inferred = unresolved = 0
    for row in rows:
        if source == "events" and not _event_needs_run_attribution(row):
            normalized.append(row)
            continue
        data = _data(row)
        copied = {**row, "data": dict(data)}
        stored_run_id = row.get("run_id") or data.get("run_id")
        if stored_run_id:
            normalized_run_id = str(stored_run_id)
            copied["run_id"] = normalized_run_id
            if live_run_ids is not None and normalized_run_id not in live_run_ids:
                copied["_trace_run_correlation"] = "unresolved"
                unresolved += 1
            else:
                copied["_trace_run_correlation"] = "exact"
        elif sole_run_id:
            copied["run_id"] = sole_run_id
            copied["_trace_run_correlation"] = "inferred"
            inferred += 1
        else:
            copied["_trace_run_correlation"] = "unresolved"
            unresolved += 1
        normalized.append(copied)
    return normalized, inferred, unresolved


class ExecutionTraceBuilder:
    """Fetch bounded stored records and project only meaningful execution facts."""

    def __init__(self, *, storage, message_store, llm_calls_store, artifact_store=None, now_fn=None):
        self.storage = storage
        self.message_store = message_store
        self.llm_calls_store = llm_calls_store
        self.artifact_store = artifact_store
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    async def build(
        self, project: dict, run_id: str | None, include_payloads: bool,
        payload_node_id: str | None = None,
    ) -> dict:
        if include_payloads and not payload_node_id:
            raise InvalidTracePayloadSelection("include_payloads requires payload_node_id")
        if payload_node_id and not include_payloads:
            raise InvalidTracePayloadSelection("payload_node_id requires include_payloads=true")

        project_id = project.get("project_id")
        live_runs = [row for row in await self.storage.list_live_runs_for_trace(project_id) if not row.get("deleted_at")]
        live_ids = {str(row["run_id"]) for row in live_runs if row.get("run_id")}
        scope_lookup = getattr(self.storage, "list_run_scopes_for_trace", None)
        run_scope_lookup_failed = False
        try:
            if scope_lookup:
                await scope_lookup(project_id)
        except Exception:
            run_scope_lookup_failed = True
            logger.warning(
                "[TRACE_VIEW] project_id=%s source=run_scope status=partial reason=read_failed",
                project_id,
            )
        sole_live_run_id = next(iter(live_ids)) if len(live_ids) == 1 else None
        safe_sole_run_id = sole_live_run_id if sole_live_run_id and not run_scope_lookup_failed else None
        if run_id is not None and run_id not in live_ids:
            raise InvalidTracePayloadSelection("run_id is not a live project run")
        runs = [row for row in live_runs if run_id is None or row.get("run_id") == run_id]
        include_unscoped = bool(run_id and run_id == safe_sole_run_id)

        input_lookup = getattr(self.message_store, "get_trace_input_message", None)
        input_call = input_lookup(project_id) if input_lookup else asyncio.sleep(0, result=None)
        artifact_call = self.artifact_store.list_trace_metadata(project_id, limit=ARTIFACT_LIMIT) if self.artifact_store else asyncio.sleep(0, result=[])
        deployment_lookup = getattr(self.storage, "list_trace_deployments", None)
        deployment_call = deployment_lookup(project_id, limit=DEPLOYMENT_LIMIT) if deployment_lookup else self.storage.get_project_deployments(project_id)
        names = ("events", "messages", "agent_llm_calls", "a2a_task_state", "snapshots", "deployments", "artifacts", "agent_results", "input_message")
        snapshot_ids: list[str] = []
        if run_id is None:
            source_calls = (
                self.storage.list_trace_events(project_id, run_id=None, limit=EVENT_LIMIT),
                self.message_store.list_trace_messages(project_id, run_id=None, limit=LEDGER_LIMIT, include_payloads=False),
                self.llm_calls_store.list_trace_summaries(project_id, run_id=None, limit=LLM_LIMIT),
                self.storage.list_trace_a2a_state(project_id, run_id=None, limit=EVENT_LIMIT),
                self.storage.list_trace_snapshots(project_id, limit=SNAPSHOT_LIMIT),
                deployment_call,
                artifact_call,
                self.message_store.list_trace_agent_results(project_id, run_id=None, limit=AGENT_RESULT_LIMIT),
                input_call,
            )
            gathered = await asyncio.gather(*source_calls, return_exceptions=True)
        else:
            try:
                events_value = await self.storage.list_trace_events(
                    project_id, run_id=run_id, limit=EVENT_LIMIT, include_unscoped=include_unscoped,
                )
            except Exception as exc:
                events_value = exc
            event_rows, _ = _window_items(events_value) if not isinstance(events_value, Exception) else ([], False)
            snapshot_ids = [
                str(_data(event)["snapshot_id"])
                for event in event_rows
                if event.get("event_type") == "snapshot_created" and _data(event).get("snapshot_id")
            ]
            source_calls = (
                self.message_store.list_trace_messages(
                    project_id, run_id=run_id, limit=LEDGER_LIMIT, include_payloads=False,
                    include_unscoped=include_unscoped,
                ),
                self.llm_calls_store.list_trace_summaries(
                    project_id, run_id=run_id, limit=LLM_LIMIT, include_unscoped=include_unscoped,
                ),
                self.storage.list_trace_a2a_state(
                    project_id, run_id=run_id, limit=EVENT_LIMIT, include_unscoped=include_unscoped,
                ),
                self.storage.list_trace_snapshots(
                    project_id, run_id=run_id, snapshot_ids=snapshot_ids, limit=SNAPSHOT_LIMIT,
                    include_unscoped=include_unscoped,
                ),
                deployment_call,
                artifact_call,
                self.message_store.list_trace_agent_results(
                    project_id, run_id=run_id, limit=AGENT_RESULT_LIMIT, include_unscoped=include_unscoped,
                ),
                input_call,
            )
            gathered = [events_value, *await asyncio.gather(*source_calls, return_exceptions=True)]
        sources: dict[str, list[dict] | dict | None] = {}
        warnings: list[str] = []
        partial = False
        if run_scope_lookup_failed:
            warnings.append("run lineage unavailable")
            partial = True
        for name, value in zip(names, gathered):
            if isinstance(value, Exception):
                sources[name] = [] if name != "input_message" else None
                warnings.append(f"{name} unavailable")
                partial = True
                logger.warning("[TRACE_VIEW] project_id=%s run_id=%s source=%s status=partial reason=read_failed", project_id, run_id or "all", name)
                continue
            if name == "input_message":
                sources[name] = value
                continue
            rows, capped = _window_items(value)
            sources[name] = rows
            if capped:
                warnings.append(f"{name} latest window reached")
                partial = True
                logger.warning("[TRACE_VIEW] project_id=%s run_id=%s source=%s status=partial reason=limit_reached", project_id, run_id or "all", name)

        inferred_records = unresolved_records = 0
        for name in ("events", "messages", "agent_llm_calls", "a2a_task_state", "snapshots", "agent_results"):
            rows = sources.get(name)
            if not isinstance(rows, list):
                continue
            scoped, inferred, unresolved = _with_effective_run_scope(
                rows, safe_sole_run_id, source=name, live_run_ids=live_ids,
            )
            sources[name] = scoped
            inferred_records += inferred
            unresolved_records += unresolved
        if inferred_records:
            logger.info(
                "[TRACE_VIEW] project_id=%s source=run_scope status=complete reason=sole_live_run_inferred count=%s",
                project_id, inferred_records,
            )
        if unresolved_records:
            warnings.append("Unscoped trace records cannot be assigned to a run")
            partial = True
            logger.warning(
                "[TRACE_VIEW] project_id=%s source=run_scope status=partial reason=ambiguous count=%s",
                project_id, unresolved_records,
            )

        if run_id is not None and snapshot_ids and "snapshots unavailable" not in warnings:
            returned_snapshot_ids = {
                str(snapshot.get("id") or snapshot.get("snapshot_id"))
                for snapshot in sources["snapshots"] or []
                if snapshot.get("id") or snapshot.get("snapshot_id")
            }
            if set(snapshot_ids) - returned_snapshot_ids:
                warnings.append("snapshots referenced by events unavailable")
                partial = True
                logger.warning(
                    "[TRACE_VIEW] project_id=%s run_id=%s source=snapshots status=partial reason=referenced_snapshot_missing",
                    project_id, run_id,
                )

        result = project_full_trace(
            project=project,
            runs=runs,
            events=sources["events"] or [],
            messages=sources["messages"] or [],
            llm_calls=sources["agent_llm_calls"] or [],
            a2a_state=sources["a2a_task_state"] or [],
            snapshots=sources["snapshots"] or [],
            artifacts=sources["artifacts"] or [], deployments=sources["deployments"] or [],
            agent_results=sources["agent_results"] or [],
            input_message=sources["input_message"] if isinstance(sources["input_message"], dict) else None,
            warnings=warnings,
            partial=partial,
        )
        if payload_node_id:
            await self._attach_payloads(result, project_id, payload_node_id)
        diagnostics = result["completeness"].setdefault("diagnostics", {})
        diagnostics.update({
            "run_scope_inferred_records": inferred_records,
            "run_scope_unresolved_records": unresolved_records,
        })
        result["generated_at"] = _iso(self.now_fn())
        logger.info(
            "[TRACE_VIEW] project_id=%s run_id=%s nodes=%s edges=%s status=%s",
            project_id, run_id or "all", len(result["nodes"]), len(result["edges"]), result["completeness"]["status"],
        )
        return result

    async def _attach_payloads(self, result: dict, project_id: str, payload_node_id: str) -> None:
        node = next((row for row in result["nodes"] if row["id"] == payload_node_id
                     and row["type"] in {"tool", "tool_call", "tool_result", "result"}), None)
        if node is None:
            raise InvalidTracePayloadSelection("payload node is not in current trace")
        if node["type"] == "result" and node["details"].get("result_source") == "assistant_message":
            source_id = node["details"].get("source_message_id")
            if not source_id:
                return
            try:
                row = await self.message_store.get_trace_agent_result(project_id, source_id)
                if row and "content" in row:
                    node["details"]["payload"] = _bounded(row["content"])
                return
            except Exception:
                result["completeness"]["status"] = "partial"
                result["completeness"]["warnings"].append("agent_result_payload unavailable")
                logger.warning("[TRACE_VIEW] project_id=%s source=agent_result_payload status=partial reason=read_failed", project_id)
                return
        if node["type"] in {"tool_call", "tool_result"}:
            message_ids = [node["details"].get("source_message_id")]
            message_ids = [message_id for message_id in message_ids if message_id]
        else:
            message_ids = [message_id for message_id in node["details"].get("ledger_message_ids", []) if message_id]
        if not message_ids:
            return
        try:
            payload_rows = await self.message_store.get_trace_payloads(project_id, message_ids)
            used = 0
            payloads: dict[str, Any] = {}
            fields = ("arguments",) if node["type"] == "tool_call" else ("result",) if node["type"] == "tool_result" else ("arguments", "result")
            for row in payload_rows:
                for field in fields:
                    if field not in _data(row):
                        continue
                    bounded = _bounded(_data(row)[field])
                    if used + _payload_bytes(bounded) > PAYLOAD_PAIR_LIMIT:
                        bounded = _bounded(_data(row)[field], max(1, PAYLOAD_PAIR_LIMIT - used))
                    payloads[field] = bounded
                    used += _payload_bytes(bounded)
            node["details"]["payloads"] = payloads
        except Exception:
            result["completeness"]["status"] = "partial"
            result["completeness"]["warnings"].append("messages_payload unavailable")
            logger.warning("[TRACE_VIEW] project_id=%s source=messages_payload status=partial reason=read_failed", project_id)
