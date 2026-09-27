"""Pure read-model projection for the semantic execution Trace canvas."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from orchestration.semantic_a2a_progress import has_published_inner_trace, normalize_progress, pair_a2a_progress
from orchestration.semantic_trace_agent_names import agent_labeler
from orchestration.semantic_trace_delegations import add_delegation_branches, collect_delegations
from orchestration.semantic_trace_terminals import add_error, add_terminal_results
from orchestration.semantic_trace_results import add_result, select_agent_result
from orchestration.semantic_trace_output import add_workflow_output
from orchestration.semantic_trace_approvals import resolve_approval_attachment
from storage.message_store import pair_tool_records
SEMANTIC_NODE_TYPES = frozenset({"project", "run", "phase", "workflow_node", "task", "auction", "auction_bid", "input", "agent_attempt", "llm_call", "tool", "tool_call", "tool_result", "delegation", "approval", "snapshot", "rollback", "error", "result", "output"})
AUCTION_BID_EVENTS = frozenset({"auction_bid_completed", "auction_bid_timeout", "auction_bid_failed"})
AUCTION_TERMINAL_EVENTS = frozenset({"auction_completed", "auction_no_bids"})
TASK_FINAL_EVENTS = frozenset({"task_completed", "task_failed"})
SELECTION_MODES = frozenset({"auction", "direct", "resume"})
LLM_ATTEMPT_TIME_SKEW = timedelta(seconds=5)


def _data(row: dict) -> dict:
    return row.get("data") if isinstance(row.get("data"), dict) else {}


def _text(value: Any, limit: int = 240) -> str | None:
    if value is None:
        return None
    value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return value if len(value) <= limit else value[:limit] + "…"


def _time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.min.replace(tzinfo=timezone.utc)


def _when(row: dict) -> datetime:
    return _time(row.get("timestamp") or row.get("created_at") or row.get("started_at"))


def _event_order(row: dict) -> tuple[datetime, str]:
    """Return deterministic durable source order without inventing sequence."""
    return (_when(row), str(row.get("event_id") or row.get("id") or ""))


def _lifecycle_identity(row: dict) -> tuple[Any, ...]:
    """Use explicit attempt when persisted, otherwise an event-scoped identity."""
    data = _data(row)
    group = (row.get("run_id"), data.get("task_id"), data.get("agent_id"))
    attempt = data.get("attempt")
    if attempt is not None:
        return (*group, "attempt", str(attempt))
    return (*group, "event", str(row.get("event_id") or row.get("id") or ""))


def _row_correlation(row: dict, default: str = "exact") -> str:
    correlation = str(row.get("_trace_run_correlation") or default)
    return correlation if correlation in {"exact", "inferred", "unresolved"} else default


def _combine_correlation(*values: str) -> str:
    if "unresolved" in values:
        return "unresolved"
    if "inferred" in values:
        return "inferred"
    return "exact"


def _status(value: Any, default: str = "running") -> str:
    if value in {"completed", "complete", "success", "succeeded", "approved"}:
        return "completed"
    if value in {"failed", "failure", "error", "cancelled", "canceled", "rejected"}:
        return "failed"
    if value in {"running", "started", "pending"}:
        return "running"
    return default


def _selection_mode(data: dict) -> str | None:
    value = data.get("selection_mode") or data.get("_trace_selection_mode")
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if value in SELECTION_MODES else None


def _llm_display_status(call: dict) -> str:
    """Return an honest terminal status from an already persisted LLM capture."""
    raw_status = call.get("status")
    explicit = _status(raw_status.strip().lower() if isinstance(raw_status, str) else raw_status, "")
    if explicit:
        return explicit
    response = call.get("response") if isinstance(call.get("response"), dict) else {}
    if response.get("error"):
        return "failed"
    if call.get("completed_at"):
        return "completed"
    return "unknown"


def _a2a_status(state: dict) -> str:
    """Map persisted A2A state to a truthful trace status.

    A2A closes its state document before retaining the actual outcome in
    ``final_status``.  That outcome is authoritative; a bare ``closed``
    state is only considered completed when a completion timestamp exists.
    """
    if not isinstance(state, dict):
        return "unknown"
    final_status = state.get("final_status")
    if final_status is not None:
        normalized = str(final_status).strip().lower()
        if normalized in {"completed", "complete", "success", "succeeded"} or normalized.endswith("_completed"):
            return "completed"
        if normalized in {"failed", "failure", "error", "cancelled", "canceled"} \
                or normalized.endswith(("_failed", "_error")):
            return "failed"
        return "unknown"
    if str(state.get("status", "")).strip().lower() == "closed":
        return "completed" if state.get("completed_at") else "unknown"
    return _status(state.get("status"), "running")


def _canonicalize_task_lifecycle(rows: list[dict]) -> list[dict]:
    """Normalize lifecycle rows while preserving legacy attempt evidence.

    Explicit ``data.attempt`` values are exact.  Rows without that field get
    an internal event-scoped identity; they are never assigned a synthetic
    attempt number.  Outer wrapper rows are discarded only when their
    duplication of a numbered start is temporally unambiguous.
    """
    groups: dict[tuple[Any, Any, Any], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        kind = row.get("event_type")
        if kind not in {"task_attempt", *TASK_FINAL_EVENTS}:
            continue
        data = _data(row)
        group = (row.get("run_id"), data.get("task_id"), data.get("agent_id"))
        groups[group][kind].append(row)

    retained_starts: dict[tuple[Any, Any, Any], list[dict]] = defaultdict(list)
    wrapper_attempts: dict[tuple[Any, Any, Any], set[str]] = defaultdict(set)
    wrapper_selection_modes: dict[tuple[Any, ...], str] = {}
    reused_attempt_start_ids: set[int] = set()
    for group, by_kind in groups.items():
        starts = sorted(by_kind.get("task_attempt", []), key=_event_order)
        explicit = [row for row in starts if _data(row).get("attempt") is not None]
        latest_start_by_attempt: dict[str, dict] = {}
        for start in starts:
            attempt = _data(start).get("attempt")
            if attempt is not None:
                key = str(attempt)
                previous = latest_start_by_attempt.get(key)
                if previous is not None:
                    previous_order, start_order = _event_order(previous), _event_order(start)
                    closed_between = any(
                        previous_order < _event_order(terminal) < start_order
                        for final_kind in TASK_FINAL_EVENTS
                        for terminal in by_kind.get(final_kind, [])
                    )
                    resumed = _selection_mode(_data(start)) == "resume"
                    if not closed_between and not resumed:
                        continue
                    # The same ordinal after a terminal boundary, or an
                    # explicit legacy resume marker, is a new execution with
                    # ambiguous source numbering. Preserve it under an event
                    # identity instead of discarding it as an outer/inner
                    # wrapper pair.
                    reused_attempt_start_ids.update({id(previous), id(start)})
                retained_starts[group].append(start)
                latest_start_by_attempt[key] = start
                continue
            later_explicit = [candidate for candidate in explicit
                              if _when(candidate) >= _when(start)]
            between = [terminal for final_kind in TASK_FINAL_EVENTS for terminal in by_kind.get(final_kind, [])
                       if _when(start) < _when(terminal)
                       <= _when(later_explicit[0])] if later_explicit else []
            prior_explicit_terminal = any(
                _when(terminal) < _when(start)
                for final_kind in TASK_FINAL_EVENTS
                for terminal in by_kind.get(final_kind, [])
                if _data(terminal).get("attempt") is not None
            )
            if later_explicit and not between and not prior_explicit_terminal:
                wrapper_attempts[group].add(str(_data(later_explicit[0]).get("attempt")))
                mode = _selection_mode(_data(start))
                if mode:
                    wrapper_selection_modes.setdefault(_lifecycle_identity(later_explicit[0]), mode)
                continue
            retained_starts[group].append(start)

    explicit_terminal_attempts: dict[tuple[Any, Any, Any], set[str]] = defaultdict(set)
    explicit_terminal_kinds: dict[tuple[Any, Any, Any], set[tuple[str, str]]] = defaultdict(set)
    for group, by_kind in groups.items():
        for kind in TASK_FINAL_EVENTS:
            for terminal in by_kind.get(kind, []):
                attempt = _data(terminal).get("attempt")
                if attempt is not None:
                    explicit_terminal_attempts[group].add(str(attempt))
                    explicit_terminal_kinds[group].add((kind, str(attempt)))

    start_identity: dict[int, tuple[Any, ...]] = {}
    reused_terminal_identity: dict[int, tuple[Any, ...]] = {}
    for group, starts in retained_starts.items():
        ordered_starts = sorted(starts, key=_event_order)
        terminals = sorted(
            [terminal for final_kind in TASK_FINAL_EVENTS for terminal in groups[group].get(final_kind, [])],
            key=_event_order,
        )
        for index, start in enumerate(ordered_starts):
            if id(start) in reused_attempt_start_ids:
                identity = (*group, "event", str(start.get("event_id") or start.get("id") or ""))
            else:
                identity = _lifecycle_identity(start)
            start_identity[id(start)] = identity

            if id(start) not in reused_attempt_start_ids:
                continue
            next_start = ordered_starts[index + 1] if index + 1 < len(ordered_starts) else None
            for terminal in terminals:
                if _event_order(terminal) <= _event_order(start):
                    continue
                if next_start is not None and _event_order(terminal) >= _event_order(next_start):
                    break
                reused_terminal_identity[id(terminal)] = identity
    terminal_correlation: dict[int, str] = {}
    terminal_for_start: dict[tuple[Any, ...], dict] = {}
    closed_inferred: set[tuple[Any, ...]] = set()
    seen_explicit: set[tuple[Any, Any, Any, Any, str]] = set()
    seen_derived: set[tuple[Any, ...]] = set()
    result: list[dict] = []

    for row in rows:
        kind = row.get("event_type")
        if kind not in {"task_attempt", *TASK_FINAL_EVENTS}:
            result.append(row)
            continue
        data = _data(row)
        group = (row.get("run_id"), data.get("task_id"), data.get("agent_id"))
        attempt = data.get("attempt")
        if kind == "task_attempt":
            if id(row) not in start_identity:
                continue
            copied = dict(row)
            identity = start_identity[id(row)]
            copied_data = {**data, "_trace_lifecycle_identity": identity}
            if id(row) in reused_attempt_start_ids:
                copied_data.update({"_trace_correlation": "inferred", "_trace_reused_attempt": True})
            if not _selection_mode(copied_data):
                propagated_mode = wrapper_selection_modes.get(identity)
                if propagated_mode:
                    copied_data["_trace_selection_mode"] = propagated_mode
            copied["data"] = copied_data
            result.append(copied)
            continue

        reused_identity = reused_terminal_identity.get(id(row))
        if reused_identity is not None:
            terminal_correlation[id(row)] = "inferred"
            terminal_for_start.setdefault(reused_identity, row)
            copied = dict(row)
            copied["data"] = {
                **data,
                "_trace_lifecycle_identity": reused_identity,
                "_trace_correlation": "inferred",
            }
            result.append(copied)
            continue

        if attempt is not None:
            identity = (*group, str(attempt), kind)
            if identity in seen_explicit:
                continue
            seen_explicit.add(identity)
            terminal_correlation[id(row)] = "exact"
            result.append(row)
            continue

        starts = sorted(retained_starts.get(group, []), key=_event_order)
        explicit_kinds = explicit_terminal_kinds.get(group, set())
        independent_starts = [start for start in starts if _data(start).get("attempt") is None]
        wrapper_terminal = any(
            (kind, attempt_id) in explicit_kinds
            for attempt_id in wrapper_attempts.get(group, set())
        )
        if wrapper_terminal and (not independent_starts or _when(row) < _when(min(independent_starts, key=_event_order))):
            continue

        candidates = []
        for start in starts:
            identity = start_identity[id(start)]
            if _when(start) > _when(row) or identity in closed_inferred:
                continue
            start_attempt = _data(start).get("attempt")
            if start_attempt is not None and str(start_attempt) in explicit_terminal_attempts.get(group, set()):
                continue
            candidates.append(start)

        selected = candidates[0] if len(candidates) == 1 else None
        if kind == "task_failed" and candidates:
            # BaseAgent emits the terminal failure without an attempt after
            # retry exhaustion.  When no numbered terminal closed an attempt,
            # prefer a single legacy retry; otherwise close the latest
            # numbered attempt without fabricating an attempt number.
            independent_candidates = [
                candidate for candidate in candidates if _data(candidate).get("attempt") is None
            ]
            explicit_starts = [start for start in starts if _data(start).get("attempt") is not None]
            if len(independent_candidates) == 1:
                selected = independent_candidates[0]
            elif not explicit_terminal_attempts.get(group) and explicit_starts:
                selected = max(explicit_starts, key=_event_order)
                terminal_correlation[id(row)] = "inferred"
        if selected is None and kind == "task_failed" and not candidates:
            explicit_starts = [start for start in starts if _data(start).get("attempt") is not None]
            if explicit_starts:
                selected = max(explicit_starts, key=_event_order)
                terminal_correlation[id(row)] = "inferred"
        if selected is not None:
            identity = start_identity[id(selected)]
            terminal_correlation.setdefault(id(row), "inferred")
            closed_inferred.add(identity)
            terminal_for_start[identity] = row
            derived_key = (*identity, kind)
            if derived_key in seen_derived:
                continue
            seen_derived.add(derived_key)
            copied = dict(row)
            copied["data"] = {**data, "_trace_lifecycle_identity": identity,
                              "_trace_correlation": terminal_correlation[id(row)]}
            result.append(copied)
            continue

        copied = dict(row)
        copied["data"] = {**data, "_trace_correlation": "unresolved"}
        result.append(copied)

    # Add the selected terminal event ID to each retained start so status and
    # result projection can use the same identity without re-running matching.
    for row in result:
        if row.get("event_type") != "task_attempt":
            continue
        identity = _data(row).get("_trace_lifecycle_identity")
        terminal = terminal_for_start.get(tuple(identity) if isinstance(identity, (tuple, list)) else identity)
        if terminal:
            row["data"] = {**_data(row), "_trace_terminal_event_id": terminal.get("event_id")}
    return result


class _Graph:
    def __init__(self):
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self.by_id: dict[str, dict] = {}
        self.edge_ids: set[str] = set()

    def node(self, *, node_id: str, node_type: str, parent_id: str | None, run_id: str | None,
             title: str, summary: str | None = None, details: dict | None = None,
             correlation: str = "exact", sequence: int | None = None,
             sort_key: tuple[datetime, str] | None = None) -> str:
        if node_type not in SEMANTIC_NODE_TYPES:
            raise ValueError(f"non-semantic node type: {node_type}")
        if node_id in self.by_id:
            return node_id
        node = {
            "id": node_id, "type": node_type, "parent_id": parent_id, "run_id": run_id,
            "sequence": sequence, "title": title, "summary": summary,
            "details": details or {}, "counts": {}, "correlation": correlation,
            "_trace_sort_key": sort_key or (datetime.max.replace(tzinfo=timezone.utc), ""),
        }
        sort_time = sort_key[0] if sort_key else None
        min_time = datetime.min.replace(tzinfo=timezone.utc)
        max_time = datetime.max.replace(tzinfo=timezone.utc)
        if sort_time and min_time < sort_time < max_time:
            node["started_at"] = sort_time.isoformat()
        self.nodes.append(node)
        self.by_id[node_id] = node
        return node_id

    def edge(self, source: str, target: str, edge_type: str, correlation: str = "exact") -> None:
        if source not in self.by_id or target not in self.by_id:
            return
        source_corr = self.by_id[source]["correlation"]
        target_corr = self.by_id[target]["correlation"]
        if "unresolved" in {source_corr, target_corr}:
            correlation = "unresolved"
        elif "inferred" in {source_corr, target_corr} and correlation == "exact":
            correlation = "inferred"
        edge_id = f"{edge_type}:{source}:{target}"
        if edge_id not in self.edge_ids:
            self.edges.append({"id": edge_id, "source": source, "target": target,
                               "type": edge_type, "correlation": correlation})
            self.edge_ids.add(edge_id)

    def remove_edge(self, source: str, target: str, edge_type: str) -> None:
        edge_id = f"{edge_type}:{source}:{target}"
        if edge_id in self.edge_ids:
            self.edges[:] = [edge for edge in self.edges if edge["id"] != edge_id]
            self.edge_ids.remove(edge_id)


def project_semantic_trace(*, project: dict, runs: list[dict], events: list[dict], messages: list[dict],
                           llm_calls: list[dict], a2a_state: list[dict], agent_results: list[dict] | None = None,
                           snapshots: list[dict] | None = None,
                           artifacts: list[dict] | None = None, deployments: list[dict] | None = None,
                           input_message: dict | None = None,
                           warnings: list[str] | tuple[str, ...] = (), partial: bool = False) -> dict:
    """Build nodes, edges, stats, and completeness without querying or mutating storage."""
    graph = _Graph()
    project_id = str(project.get("project_id") or "project")
    project_node = graph.node(node_id=f"project:{project_id}", node_type="project", parent_id=None, run_id=None,
                              title=project.get("title") or "Project", details={"project_id": project_id})
    run_nodes: dict[str, str] = {}
    live_runs = [run for run in runs if run.get("run_id") and not run.get("deleted_at")]
    known_run_ids = {str(run.get("run_id")) for run in live_runs}
    pending_runs = sorted(live_runs, key=lambda row: (str(row.get("created_at") or ""), str(row.get("run_id") or "")))
    ordered_runs: list[dict] = []
    while pending_runs:
        ready = [run for run in pending_runs
                 if not run.get("parent_run_id") or str(run.get("parent_run_id")) in run_nodes
                 or str(run.get("parent_run_id")) not in known_run_ids]
        if not ready:
            # A cycle or malformed legacy parent cannot become a branch; keep
            # the stable input order and mark the unresolved parent below.
            ready = [pending_runs[0]]
        for run in ready:
            pending_runs.remove(run)
            ordered_runs.append(run)
            run_nodes[str(run.get("run_id"))] = f"run:{project_id}:{run.get('run_id')}"
    run_nodes.clear()
    for run in ordered_runs:
        run_id = str(run.get("run_id") or "")
        parent_run = run_nodes.get(str(run.get("parent_run_id") or ""))
        node_id = f"run:{project_id}:{run_id}"
        graph.node(node_id=node_id, node_type="run", parent_id=parent_run or project_node, run_id=run_id,
                   title="Run", summary=run.get("run_status"), details={"active": bool(run.get("active")), "run_id": run_id},
                   correlation="exact" if parent_run or not run.get("parent_run_id") else "inferred")
        graph.edge(parent_run or project_node, node_id, "forked_from" if parent_run else "contains")
        run_nodes[run_id] = node_id
    unresolved_history: str | None = None

    def parent_for_run(run_id: str | None, correlation: str = "exact") -> str:
        nonlocal unresolved_history, partial
        if correlation != "unresolved" and (not run_id or str(run_id) in run_nodes):
            return run_nodes.get(str(run_id or ""), project_node)
        if unresolved_history is None:
            unresolved_history = graph.node(
                node_id=f"unresolved:{project_id}", node_type="workflow_node", parent_id=project_node, run_id=None,
                title="Unresolved history", details={"reason": "run is unavailable or ambiguous"}, correlation="unresolved",
            )
            graph.edge(project_node, unresolved_history, "contains", "unresolved")
        if run_id:
            all_warnings.add("Unresolved history belongs to a non-live run")
            partial = True
        return unresolved_history
    prompt = project.get("user_prompt") or (input_message or {}).get("content")
    prompt_details = prompt if isinstance(prompt, str) else (
        json.dumps(prompt, ensure_ascii=False, default=str) if prompt is not None else None
    )
    input_id = graph.node(node_id=f"input:{project_id}", node_type="input", parent_id=project_node, run_id=None,
                          title="Input", summary=_text(prompt), details={
                              "source": "project.user_prompt" if project.get("user_prompt") else "user_message",
                              "prompt": prompt_details,
                          })
    graph.edge(project_node, input_id, "contains")
    event_rows = _canonicalize_task_lifecycle(sorted(events, key=_event_order))
    delegations = collect_delegations(event_rows, project_id=project_id, when=_when, order=_event_order,
                                      row_correlation=_row_correlation)
    delegated_tasks = {(delegation["run_id"], str(delegation["task_id"])) for delegation in delegations}
    message_rows = sorted(messages, key=lambda row: (row.get("sequence", 0), str(row.get("id") or "")))
    all_warnings = set(warnings)
    partial = partial or bool(warnings)
    llm_correlation_diagnostics = {
        "llm_exact_matches": 0,
        "llm_inferred_matches": 0,
        "llm_temporal_unresolved": 0,
        "llm_ambiguous": 0,
    }
    tasks: dict[tuple[str | None, str], str] = {}
    phases: dict[tuple[str | None, str], str] = {}
    workflow_nodes: dict[tuple[str | None, str], str] = {}
    for event in event_rows:
        data, run_id = _data(event), event.get("run_id")
        event_correlation = _row_correlation(event)
        phase_name = data.get("phase") or data.get("workflow_phase")
        workflow_name = data.get("workflow_node_id")
        run_parent = parent_for_run(run_id, event_correlation)
        phase_id = run_parent
        if phase_name:
            phase_id = f"phase:{run_id or 'project'}:{phase_name}"
            if (run_id, str(phase_name)) not in phases:
                correlation = "unresolved" if run_parent == unresolved_history else event_correlation
                graph.node(node_id=phase_id, node_type="phase", parent_id=run_parent, run_id=run_id,
                           title=str(phase_name), details={"workflow_node_id": data.get("workflow_node_id")},
                           correlation=correlation, sequence=data.get("sequence"), sort_key=_event_order(event))
                graph.edge(run_parent, phase_id, "contains", correlation)
                phases[(run_id, str(phase_name))] = phase_id
        if workflow_name:
            workflow_id = f"workflow-node:{run_id or 'project'}:{workflow_name}"
            if (run_id, str(workflow_name)) not in workflow_nodes:
                correlation = "unresolved" if run_parent == unresolved_history else event_correlation
                graph.node(node_id=workflow_id, node_type="workflow_node", parent_id=phase_id, run_id=run_id,
                           title=str(workflow_name), details={"workflow_node_id": workflow_name},
                           correlation=correlation, sequence=data.get("sequence"), sort_key=_event_order(event))
                graph.edge(phase_id, workflow_id, "contains", correlation)
                workflow_nodes[(run_id, str(workflow_name))] = workflow_id
            phase_id = workflow_nodes[(run_id, str(workflow_name))]
        task_id = data.get("task_id")
        if task_id and (run_id, str(task_id)) not in tasks and (run_id, str(task_id)) not in delegated_tasks:
            task_node = f"task:{run_id or 'project'}:{task_id}"
            correlation = "unresolved" if run_parent == unresolved_history else event_correlation
            task_description = data.get("task_description")
            graph.node(node_id=task_node, node_type="task", parent_id=phase_id, run_id=run_id,
                       title=str(task_description or task_id),
                       details={"task_id": task_id, "display_name_source": "task_description" if task_description else "task_id",
                                "technical_id": not bool(task_description)},
                       correlation=correlation, sequence=data.get("sequence"), sort_key=_event_order(event))
            graph.edge(phase_id, task_node, "contains", correlation)
            tasks[(run_id, str(task_id))] = task_node
    attempts: dict[tuple[str | None, str | None, str | None], str] = {}
    attempts_by_identity: dict[tuple[Any, ...], str] = {}
    attempt_identity_counts: defaultdict[tuple[str | None, str | None, str | None], int] = defaultdict(int)
    attempt_windows: list[dict] = []
    attempt_windows_by_node: dict[str, dict] = {}
    completed_attempts: list[dict] = []
    latest_by_run: dict[str | None, str] = {}
    snapshot_runs = {
        str(_data(event).get("snapshot_id")): event.get("run_id")
        for event in event_rows
        if event.get("event_type") == "snapshot_created" and _data(event).get("snapshot_id")
    }
    for snapshot in snapshots or []:
        snapshot_id = str(snapshot.get("id") or snapshot.get("snapshot_id") or "")
        if not snapshot_id:
            continue
        run_id = snapshot.get("run_id") or snapshot_runs.get(snapshot_id)
        source_correlation = _row_correlation(snapshot)
        parent = parent_for_run(run_id, source_correlation)
        correlation = "unresolved" if not run_id or parent == unresolved_history else source_correlation
        if correlation == "unresolved":
            all_warnings.add("Snapshot run correlation unavailable")
            partial = True
        node_id = f"snapshot:{run_id or 'project'}:{snapshot_id}"
        graph.node(node_id=node_id, node_type="snapshot", parent_id=parent, run_id=run_id,
                   title="Snapshot", summary=_text(snapshot.get("label") or snapshot.get("phase")),
                   details={"snapshot_id": snapshot_id, "event_id": snapshot.get("event_id")},
                   correlation=correlation, sequence=snapshot.get("sequence"), sort_key=_event_order(snapshot))
        graph.edge(parent, node_id, "contains", correlation)
    project_context = project.get("context") if isinstance(project.get("context"), dict) else {}
    for artifact in list(artifacts or project.get("artifacts") or project_context.get("artifacts") or []):
        if not isinstance(artifact, dict):
            continue
        artifact_id = str(artifact.get("artifact_id") or artifact.get("id") or artifact.get("path") or "")
        if not artifact_id:
            continue
        node_id = f"result:artifact:{project_id}:{artifact_id}"
        graph.node(node_id=node_id, node_type="result", parent_id=project_node, run_id=None, title="Result",
                   summary=_text(artifact.get("path") or artifact.get("name") or "Artifact"),
                   details={"display_status": "completed", "result_source": "artifact", "artifact_id": artifact_id,
                            "artifact_path": artifact.get("path") or artifact.get("name"),
                            "payload_available": False}, sequence=artifact.get("sequence"))
        graph.edge(project_node, node_id, "produced")
    for deployment in list(deployments or project.get("deployments") or project_context.get("deployments") or []):
        if not isinstance(deployment, dict):
            continue
        deployment_id = str(deployment.get("deployment_id") or deployment.get("id") or "")
        if not deployment_id:
            continue
        node_id = f"result:deployment:{project_id}:{deployment_id}"
        status = _status(deployment.get("status") or deployment.get("deploy_status"), "completed")
        graph.node(node_id=node_id, node_type="result", parent_id=project_node, run_id=None, title="Result",
                   summary=_text(deployment.get("status") or deployment.get("deploy_status") or "Deployment"),
                   details={"display_status": status, "result_source": "deployment", "deployment_id": deployment_id,
                            "deployment_url": deployment.get("url"),
                            "payload_available": False}, sequence=deployment.get("sequence"))
        graph.edge(project_node, node_id, "produced")
    auction_events: list[dict] = []
    for event in event_rows:
        kind, data, run_id = event.get("event_type"), _data(event), event.get("run_id")
        if kind in {"project_reverted", "rollback", "workflow_rollback"}:
            rollback_id = f"rollback:{run_id or 'project'}:{event.get('event_id') or 'event'}"
            correlation = _row_correlation(event)
            parent = parent_for_run(run_id, correlation)
            correlation = "unresolved" if parent == unresolved_history else correlation
            graph.node(node_id=rollback_id, node_type="rollback", parent_id=parent, run_id=run_id, title="Rollback",
                       details={"target_sequence": data.get("target_sequence"), "target_snapshot_id": data.get("snapshot_id")},
                       correlation=correlation, sequence=data.get("sequence"), sort_key=_event_order(event))
            graph.edge(parent, rollback_id, "contains", correlation)
            target = next((node["id"] for node in graph.nodes if node["type"] == "snapshot" and node["details"].get("snapshot_id") == data.get("snapshot_id")), None)
            if target:
                graph.edge(rollback_id, target, "reverted_to")
            else:
                all_warnings.add("Rollback target unavailable")
                partial = True
        if isinstance(kind, str) and kind.startswith("auction"):
            auction_events.append(event)

    def attach_start(node_id: str, parent_id: str | None, correlation: str) -> None:
        graph.edge(parent_id or input_id, node_id, "starts", correlation)

    def record_attempt_window(node_id: str, run_id: str | None, task_id: str | None,
                              agent_id: str | None, started_at: datetime,
                              event_id: str | None = None,
                              completed_at: datetime | None = None) -> None:
        if started_at == datetime.min.replace(tzinfo=timezone.utc):
            return
        window = {"node_id": node_id, "run_id": run_id, "task_id": task_id,
                  "agent_id": agent_id, "started_at": started_at,
                  "completed_at": completed_at, "event_id": event_id}
        attempt_windows_by_node[node_id] = window
        attempt_windows.append(window)

    def add_attempt(run_id: str | None, task_id: str | None, agent_id: str | None, event_id: str,
                    agent_name: str, *, parent_id: str | None = None, a2a: bool = False,
                    sequence: int | None = None, status: str = "running", correlation: str = "exact",
                    selection_mode: str = "unknown",
                    sort_key: tuple[datetime, str] | None = None,
                    started_at: datetime | None = None,
                    completed_at: datetime | None = None) -> str:
        node_id = f"{'a2a-agent' if a2a else 'agent'}:{run_id or 'project'}:{task_id or 'none'}:{agent_id or agent_name}:{event_id}"
        resolved_parent = (
            parent_id
            or tasks.get((run_id, str(task_id or "")))
            or parent_for_run(run_id, correlation)
        )
        graph.node(node_id=node_id, node_type="agent_attempt", parent_id=resolved_parent, run_id=run_id,
                   title=agent_name, summary="A2A agent" if a2a else None,
                   details={"agent_id": agent_id, "agent_name": agent_name, "execution_kind": "a2a" if a2a else "local",
                            "display_status": status, "selection_mode": selection_mode,
                            "llm_rounds": []}, correlation=correlation, sequence=sequence,
                   sort_key=sort_key)
        attach_start(node_id, resolved_parent, correlation)
        attempts[(run_id, task_id, agent_id)] = node_id
        attempt_identity_counts[(run_id, task_id, agent_id)] += 1
        if sequence is not None:
            attempts_by_identity[(run_id, task_id, agent_id, str(sequence))] = node_id
        latest_by_run[run_id] = node_id
        if started_at is not None:
            record_attempt_window(node_id, run_id, task_id, agent_id, started_at, event_id, completed_at)
        return node_id

    event_by_id = {str(event.get("event_id")): event for event in event_rows if event.get("event_id")}
    terminal_events = defaultdict(list)
    for event in event_rows:
        kind, data = event.get("event_type"), _data(event)
        if kind in {"task_completed", "task_failed"}:
            terminal_events[(event.get("run_id"), data.get("task_id"), data.get("agent_id"), data.get("attempt"))].append(event)
    for event in event_rows:
        if event.get("event_type") != "task_attempt":
            continue
        data, run_id = _data(event), event.get("run_id")
        run_correlation = _row_correlation(event)
        identity = data.get("_trace_lifecycle_identity")
        key = (run_id, data.get("task_id"), data.get("agent_id"), data.get("attempt"))
        terminal = event_by_id.get(data.get("_trace_terminal_event_id"))
        if terminal is None and data.get("attempt") is not None and not data.get("_trace_reused_attempt"):
            terminal = next((row for row in terminal_events[key] if _when(row) >= _when(event)), None)
        status = "completed" if terminal and terminal.get("event_type") == "task_completed" else "failed" if terminal else "running"
        attempt_correlation = _combine_correlation(
            run_correlation,
            data.get("_trace_correlation") or (
                "exact" if data.get("attempt") is not None else "inferred"
            ),
        )
        selection_mode = _selection_mode(data) or "unknown"
        node_id = add_attempt(run_id, data.get("task_id"), data.get("agent_id"), str(event.get("event_id") or "attempt"),
                    _text(data.get("agent_display_name") or data.get("agent_id") or "Agent") or "Agent",
                    parent_id=tasks.get((run_id, str(data.get("task_id") or "")))
                    or parent_for_run(run_id, run_correlation),
                    sequence=data.get("sequence"), status=status,
                    correlation=attempt_correlation, selection_mode=selection_mode,
                    sort_key=_event_order(event), started_at=_when(event),
                    completed_at=_when(terminal) if terminal else None)
        if isinstance(identity, list):
            identity = tuple(identity)
        if isinstance(identity, tuple):
            attempts_by_identity[identity] = node_id
        if terminal is not None:
            terminal_data = _data(terminal)
            terminal_data["_trace_attempt_node_id"] = node_id
            if terminal_data.get("_trace_correlation") is None:
                terminal_data["_trace_correlation"] = attempt_correlation
        if terminal and terminal.get("event_type") == "task_completed":
            completed_attempts.append({"node_id": node_id, "run_id": run_id, "task_id": data.get("task_id"),
                                       "agent_id": data.get("agent_id"), "attempt": data.get("attempt"),
                                       "started_at": _when(event), "completed_at": _when(terminal),
                                       "sequence": terminal.get("sequence"),
                                       "terminal_event_id": terminal.get("event_id")})

    def add_auction(completed: dict, bid_events: list[dict], correlation: str) -> None:
        data, run_id = _data(completed), completed.get("run_id")
        task_id = data.get("task_id")
        raw_auction_id = str(data.get("auction_id") or completed.get("event_id") or "")
        correlation = _combine_correlation(correlation, _row_correlation(completed))
        parent = tasks.get((run_id, str(task_id or ""))) or parent_for_run(run_id, correlation)
        correlation = "unresolved" if parent == unresolved_history else correlation
        auction_id = f"auction:{run_id or 'project'}:{raw_auction_id}"
        auction_outcome = "no_bids" if completed.get("event_type") == "auction_no_bids" else "completed"
        graph.node(node_id=auction_id, node_type="auction", parent_id=parent, run_id=run_id, title="Auction",
                   details={"reason": data.get("reason") or "Reason not recorded", "auction_id": raw_auction_id,
                            "outcome": auction_outcome, "agent_count": data.get("agent_count")},
                   correlation=correlation, sequence=data.get("sequence"), sort_key=_event_order(completed))
        graph.edge(parent, auction_id, "contains", correlation)
        selected_agent = (data.get("winner_id") or data.get("winner_agent_id") or data.get("selected_agent_id")
                          or data.get("selected_agent")) if auction_outcome == "completed" else None
        aggregate_bids = data.get("bids") if isinstance(data.get("bids"), list) else []
        rows = bid_events or [{"event_id": f"{completed.get('event_id')}:bid:{index}", "run_id": run_id, "data": bid}
                              for index, bid in enumerate(aggregate_bids) if isinstance(bid, dict)]
        for index, bid_event in enumerate(rows):
            bid = _data(bid_event)
            agent_id = bid.get("agent_id") or bid.get("agent")
            if not agent_id:
                continue
            bid_outcome = {"auction_bid_timeout": "timed_out", "auction_bid_failed": "failed"}.get(
                bid_event.get("event_type"), "completed")
            reason = bid.get("reasoning") or bid.get("reason")
            if not reason and bid_outcome == "timed_out":
                reason = "Bid evaluation timed out"
            elif not reason and bid_outcome == "failed":
                reason = "Bid evaluation failed"
            error_preview = _text(bid.get("error"))
            bid_id = f"auction-bid:{run_id or 'project'}:{raw_auction_id}:{agent_id}:{index}"
            graph.node(node_id=bid_id, node_type="auction_bid", parent_id=auction_id, run_id=run_id,
                       title=str(agent_id), summary=_text(reason),
                       details={"agent_id": agent_id, "fit_score": bid.get("fit_score"),
                                "reason": _text(reason) or "Reason not recorded", "outcome": bid_outcome,
                                "error_preview": error_preview, "critic_validation": bid.get("critic_validation")},
                       correlation=_combine_correlation(correlation, _row_correlation(bid_event)),
                       sequence=bid.get("sequence") or data.get("sequence"),
                       sort_key=_event_order(bid_event))
            graph.edge(auction_id, bid_id, "contains", _combine_correlation(correlation, _row_correlation(bid_event)))
            attempt = attempts.get((run_id, task_id, agent_id))
            if selected_agent and str(agent_id) == str(selected_agent) and attempt:
                graph.edge(bid_id, attempt, "selected", correlation)

    plain_by_task: dict[tuple[str | None, str | None], list[dict]] = defaultdict(list)
    for event in auction_events:
        if event.get("event_type") in AUCTION_BID_EVENTS | AUCTION_TERMINAL_EVENTS:
            plain_by_task[(event.get("run_id"), _data(event).get("task_id"))].append(event)
    for (run_id, task_id), rows in plain_by_task.items():
        bids, previous = [], datetime.min.replace(tzinfo=timezone.utc)
        for event in rows:
            if event.get("event_type") in AUCTION_BID_EVENTS:
                bids.append(event)
                continue
            window = [bid for bid in bids if previous < _when(bid) <= _when(event)]
            add_auction(event, window, "exact" if _data(event).get("auction_id") else "inferred")
            previous = _when(event)
        unmatched = [bid for bid in bids if _when(bid) > previous]
        if unmatched:
            first = unmatched[0]
            unresolved = {"event_id": f"unresolved:{first.get('event_id')}", "run_id": run_id,
                          "data": {"task_id": task_id, "reason": "Auction completion unavailable"}}
            add_auction(unresolved, unmatched, "unresolved")
            all_warnings.add("Auction completion unavailable")
            partial = True
    plain_tasks = set(plain_by_task)
    for event in auction_events:
        if "." not in str(event.get("event_type") or ""):
            continue
        if (event.get("run_id"), _data(event).get("task_id")) not in plain_tasks:
            add_auction(event, [], "inferred")

    auction_evidence = {
        (event.get("run_id"), _data(event).get("task_id"))
        for event in auction_events
        if event.get("event_type") in AUCTION_BID_EVENTS | AUCTION_TERMINAL_EVENTS
    }
    for event in event_rows:
        if event.get("event_type") != "task_attempt":
            continue
        data = _data(event)
        if _selection_mode(data) != "auction":
            continue
        task_id = data.get("task_id")
        if task_id and (event.get("run_id"), task_id) not in auction_evidence:
            all_warnings.add("Auction selection not recorded")
            partial = True

    closed_counts = defaultdict(int)
    for attempt in completed_attempts:
        closed_counts[(attempt["run_id"], attempt["task_id"], attempt["agent_id"])] += 1
    for attempt in completed_attempts:
        match = select_agent_result(
            attempt, agent_results or [],
            closed_attempt_count=closed_counts[(attempt["run_id"], attempt["task_id"], attempt["agent_id"])],
        )
        add_result(graph, attempt["node_id"], attempt["run_id"], attempt["node_id"],
                   source="assistant_message", source_message_id=match["id"] if match else None,
                   sequence=attempt["sequence"], correlation=match["correlation"] if match else "inferred")
        # A task_completed terminal is already authoritative for lifecycle
        # completion; some built-in agents intentionally do not emit an
        # assistant result message.  add_terminal_results upgrades this
        # placeholder to source=task_terminal, so do not report a retention
        # gap in that normal case.
        if not match and not attempt.get("terminal_event_id"):
            all_warnings.add("Agent result not recorded")
            partial = True

    def attempt_for_time(run_id: str | None, task_id: str | None, agent_id: str | None,
                         at: datetime) -> str | None:
        candidates = [row for row in attempt_windows
                      if row["run_id"] == run_id and row["task_id"] == task_id and row["agent_id"] == agent_id
                      and row["started_at"] <= at and (row["completed_at"] is None or at <= row["completed_at"])]
        return candidates[0]["node_id"] if len(candidates) == 1 else None

    def nearest_attempt_for_time(run_id: str | None, task_id: str | None, agent_id: str | None,
                                 at: datetime) -> tuple[str | None, str | None]:
        """Return a unique attempt within the bounded timestamp skew.

        Exact containment remains authoritative.  This fallback handles clock
        skew at capture boundaries, but rejects ties so retries are never
        guessed from an ambiguous gap.
        """
        candidates = [row for row in attempt_windows
                      if row["run_id"] == run_id and row["task_id"] == task_id and row["agent_id"] == agent_id]
        if not candidates:
            return None, None
        containing = [row for row in candidates
                      if row["started_at"] <= at
                      and (row.get("completed_at") is None or at <= row["completed_at"])]
        if len(containing) != 0:
            return (None, None) if len(containing) != 1 else (containing[0]["node_id"], "attempt_time_window")

        def distance(window: dict) -> timedelta:
            if at < window["started_at"]:
                return window["started_at"] - at
            completed_at = window.get("completed_at")
            return at - completed_at if completed_at is not None else timedelta.max

        distances = [(window, distance(window)) for window in candidates]
        nearest_distance = min((value for _, value in distances), default=timedelta.max)
        if nearest_distance > LLM_ATTEMPT_TIME_SKEW:
            return None, None
        nearest = [window for window, value in distances if value == nearest_distance]
        if len(nearest) != 1:
            return None, None
        return nearest[0]["node_id"], "nearest_attempt_within_skew"

    agent_label = agent_labeler(event_rows, project_id)
    delegated_agents = add_delegation_branches(
        graph, delegations, attempt_windows=attempt_windows, attempts=attempts, run_parent=parent_for_run,
        combine_correlation=_combine_correlation, text_for=_text, name_for=agent_label, warnings=all_warnings,
    )

    wrappers: dict[tuple[str | None, str | None], str] = {}
    a2a_state_by_key: dict[tuple[str | None, str | None], dict] = {}
    for event in event_rows:
        if event.get("event_type") != "a2a_agent_started":
            continue
        data, run_id, node_key = _data(event), event.get("run_id"), _data(event).get("node_id")
        name = _text(data.get("agent_name") or data.get("server_id") or "A2A") or "A2A"
        wrapper = add_attempt(run_id, str(node_key or "a2a"), data.get("server_id"), str(event.get("event_id") or "start"),
                              name, a2a=True, correlation=_row_correlation(event),
                              started_at=_when(event))
        wrappers[(run_id, node_key)] = wrapper
    for state in a2a_state:
        key = (state.get("run_id"), state.get("node_id"))
        status = _a2a_status(state)
        a2a_state_by_key[key] = state
        if key not in wrappers:
            wrappers[key] = add_attempt(state.get("run_id"), str(state.get("node_id") or "a2a"), state.get("server_id"),
                                        "state", _text(state.get("server_id") or "A2A") or "A2A", a2a=True,
                                        status=status, correlation=_combine_correlation(_row_correlation(state), "inferred"),
                                        started_at=_when(state),
                                        completed_at=_time(state.get("completed_at")) if state.get("completed_at") else None)
        wrapper = graph.by_id[wrappers[key]]
        wrapper["details"].update({
            "status": state.get("status"),
            "final_status": state.get("final_status"),
            "completed_at": state.get("completed_at"),
            "closed_at": state.get("closed_at"),
        })
        wrapper["details"]["display_status"] = status
        window = attempt_windows_by_node.get(wrappers[key])
        if window and state.get("completed_at"):
            window["completed_at"] = _time(state.get("completed_at"))

    inner_facts = pair_a2a_progress(event_rows)
    normalized = [row for event in event_rows if (row := normalize_progress(event))]
    inner_agents: dict[tuple[str | None, str | None, str, str], str] = {}
    delegation_child: dict[tuple[str | None, str | None, str, str], str] = {}

    def inner_agent(fact: dict, agent_name: str) -> str:
        key = (fact.get("outer_run_id"), fact.get("outer_node_id"), fact["remote_run_id"], agent_name)
        if key in inner_agents:
            return inner_agents[key]
        parent = delegation_child.get(key) or wrappers.get((fact.get("outer_run_id"), fact.get("outer_node_id")))
        source_event = event_by_id.get(fact.get("start_event_id") or fact.get("terminal_event_id"), {})
        node_id = add_attempt(fact.get("outer_run_id"), fact["remote_run_id"], agent_name,
                              f"inner-{agent_name}-{fact['sequence']}", agent_name, parent_id=parent, a2a=True,
                              sequence=fact["sequence"], correlation="exact",
                              started_at=_when(source_event) if source_event else _time(fact.get("timestamp")))
        inner_agents[key] = node_id
        return node_id

    for fact in inner_facts["delegations"]:
        parent = inner_agent(fact, fact["parent_agent"])
        delegation_id = f"delegation:{fact['outer_run_id']}:{fact['remote_run_id']}:{fact['start_event_id'] or fact['terminal_event_id']}"
        graph.node(node_id=delegation_id, node_type="delegation", parent_id=parent, run_id=fact.get("outer_run_id"),
                   title=fact["child_agent"], summary="A2A delegation",
                   details={"parent_agent": fact["parent_agent"], "child_agent": fact["child_agent"], "display_status": fact["status"]},
                   correlation=fact["correlation"], sequence=fact["sequence"],
                   sort_key=_event_order(event_by_id.get(fact.get("start_event_id") or fact.get("terminal_event_id"), {})))
        graph.edge(parent, delegation_id, "delegates", fact["correlation"])
        child_key = (fact.get("outer_run_id"), fact.get("outer_node_id"), fact["remote_run_id"], fact["child_agent"])
        delegation_child[child_key] = delegation_id
        child = inner_agent(fact, fact["child_agent"])
        previous_parent = graph.by_id[child]["parent_id"]
        graph.by_id[child]["parent_id"] = delegation_id
        if previous_parent:
            graph.remove_edge(previous_parent, child, "starts")
        graph.edge(delegation_id, child, "starts", fact["correlation"])
    for row in normalized:
        if row.get("agent"):
            inner_agent(row, row["agent"])
    for fact in inner_facts["tools"]:
        parent = inner_agent(fact, fact["agent"])
        tool_id = f"a2a-tool:{fact['outer_run_id']}:{fact['remote_run_id']}:{fact['agent']}:{fact['tool_name']}:{fact['start_event_id'] or fact['terminal_event_id']}"
        graph.node(node_id=tool_id, node_type="tool", parent_id=parent, run_id=fact.get("outer_run_id"), title=fact["tool_name"],
                   summary="A2A tool", details={"agent_name": fact["agent"], "tool_name": fact["tool_name"],
                   "display_status": fact["status"], "ledger_message_ids": [], "payload_available": False},
                   correlation=fact["correlation"], sequence=fact["sequence"],
                   sort_key=_event_order(event_by_id.get(fact.get("start_event_id") or fact.get("terminal_event_id"), {})))
        graph.edge(parent, tool_id, "uses", fact["correlation"])
        if fact["status"] == "completed":
            add_result(graph, tool_id, fact.get("outer_run_id"), tool_id,
                       source="not_recorded", source_message_id=None,
                       sequence=fact["sequence"], correlation=fact["correlation"])
        if fact["status"] == "failed":
            add_error(graph, tool_id, fact.get("outer_run_id"), f"a2a-progress:{fact['terminal_event_id']}", "Tool failed", fact["correlation"], fact["sequence"])
    for key, wrapper in wrappers.items():
        if not has_published_inner_trace(event_rows, key[0], key[1]):
            all_warnings.add("A2A inner trace not published")
            partial = True

    # The outer A2A lifecycle is the source of truth for the wrapper terminal
    # state. Typed progress describes its inner work but cannot replace the
    # persisted completion and output of the delegated agent itself.
    for event in event_rows:
        if event.get("event_type") != "a2a_agent_completed":
            continue
        data = _data(event)
        run_id = event.get("run_id")
        wrapper = wrappers.get((run_id, data.get("node_id")))
        if not wrapper:
            continue
        state = a2a_state_by_key.get((run_id, data.get("node_id")))
        if state and state.get("final_status") is not None:
            # Persisted A2A final_status is the terminal source of truth.
            continue
        graph.by_id[wrapper]["details"]["display_status"] = "completed"
        add_result(graph, wrapper, run_id, f"{run_id or 'project'}:a2a:{event.get('event_id') or 'completed'}",
                   source="not_recorded", source_message_id=None, sequence=data.get("attempt"), correlation="exact")

    ledger_ids = set()
    technical_calls: dict[tuple[str | None, str | None, str | None, str], tuple[str, datetime]] = {}
    for pair in pair_tool_records(message_rows):
        call, result = pair.get("call"), pair.get("result")
        anchor = call or result
        if not anchor:
            continue
        data, run_id = _data(anchor), anchor.get("run_id")
        ledger_ids.add(str(data.get("tool_call_id") or ""))
        run_correlation = _row_correlation(anchor)
        exact_attempt = (attempt_for_time(run_id, data.get("task_id"), data.get("agent_id"), _when(anchor))
                         or delegated_agents.get((run_id, data.get("task_id"), data.get("agent_id"))))
        task_parent = tasks.get((run_id, str(data.get("task_id") or "")))
        parent = exact_attempt or task_parent or parent_for_run(run_id, run_correlation)
        name = _text(data.get("name") or _data(result or {}).get("name") or "Tool") or "Tool"
        status = _status((result or {}).get("status"), "completed" if result else "running")
        correlation = _combine_correlation(run_correlation, "exact" if call and exact_attempt else "inferred")
        tool_id = f"tool:{run_id or 'project'}:{call.get('id') if call else result.get('id')}"
        message_ids = [row.get("id") for row in (call, result) if row and row.get("id")]
        graph.node(node_id=tool_id, node_type="tool", parent_id=parent, run_id=run_id, title=name, summary="Tool",
                   details={"agent_name": _text(agent_label(data.get("agent_id"), data.get("agent_display_name"))),
                            "tool_name": name,
                            "display_status": status, "ledger_message_ids": message_ids, "payload_available": bool(message_ids)},
                   correlation=correlation, sequence=anchor.get("sequence"), sort_key=_event_order(anchor))
        graph.edge(parent, tool_id, "uses", correlation)
        technical_parent = task_parent or parent
        call_node = None
        if call:
            call_node = f"tool-call:{run_id or 'project'}:{call.get('id')}"
            call_data = _data(call)
            graph.node(node_id=call_node, node_type="tool_call", parent_id=technical_parent, run_id=run_id,
                       title=name, summary="Tool Call", details={
                           "tool_call_id": call_data.get("tool_call_id"),
                           "source_message_id": call.get("id"),
                           "run_id": run_id,
                           "task_id": call_data.get("task_id"),
                           "agent_id": call_data.get("agent_id"),
                           "agent_display_name": _text(call_data.get("agent_display_name")),
                           "workflow_node_id": call_data.get("workflow_node_id"),
                           "sequence": call.get("sequence"),
                           "created_at": _when(call).isoformat() if _when(call) != datetime.min.replace(tzinfo=timezone.utc) else None,
                           "display_status": _status(call.get("status"), "completed" if result else "running"),
                           "payload_available": bool(call.get("id")),
                       }, correlation=correlation,
                       sequence=call.get("sequence"), sort_key=_event_order(call))
            graph.edge(technical_parent, call_node, "contains", correlation)
            technical_calls[(run_id, data.get("task_id"), data.get("agent_id"), str(data.get("tool_call_id") or ""))] = (call_node, _when(call))
        if result:
            technical_result = f"tool-result:{run_id or 'project'}:{result.get('id')}"
            result_data = _data(result)
            graph.node(node_id=technical_result, node_type="tool_result", parent_id=call_node or technical_parent, run_id=run_id,
                       title=name, summary="Tool Result", details={
                           "tool_call_id": result_data.get("tool_call_id"),
                           "source_message_id": result.get("id"),
                           "run_id": run_id,
                           "task_id": result_data.get("task_id"),
                           "agent_id": result_data.get("agent_id"),
                           "agent_display_name": _text(result_data.get("agent_display_name")),
                           "workflow_node_id": result_data.get("workflow_node_id"),
                           "sequence": result.get("sequence"),
                           "created_at": _when(result).isoformat() if _when(result) != datetime.min.replace(tzinfo=timezone.utc) else None,
                           "display_status": _status(result.get("status"), "completed"),
                           "payload_available": bool(result.get("id")),
                           "result_recorded": True,
                       }, correlation=correlation,
                       sequence=result.get("sequence"), sort_key=_event_order(result))
            graph.edge(call_node or technical_parent, technical_result, "returned", correlation)
        if result and status != "failed":
            result_id = f"result:{tool_id}:{result.get('id')}"
            graph.node(node_id=result_id, node_type="result", parent_id=tool_id, run_id=run_id,
                       title="Result", summary="available",
                       details={"display_status": status, "result_source": "tool_ledger",
                                "source_message_id": result.get("id"), "payload_available": True,
                                "ledger_message_ids": [result.get("id")]},
                       correlation=correlation, sequence=result.get("sequence"), sort_key=_event_order(result))
            graph.edge(tool_id, result_id, "produced", correlation)
        if status == "failed":
            add_error(graph, tool_id, run_id, f"ledger:{anchor.get('id')}", "Tool failed", correlation, anchor.get("sequence"))

    llm_tool_requests: dict[tuple[str | None, str | None, str | None, str], list[tuple[str, datetime]]] = defaultdict(list)
    for call in sorted(llm_calls, key=lambda row: (row.get("turn_index") is None, row.get("turn_index") or 0, _when(row), str(row.get("_id") or row.get("call_id") or ""))):
        run_id = call.get("run_id")
        run_correlation = _row_correlation(call)
        attempt_identity = (run_id, call.get("task_id"), call.get("agent_id"))
        attempt = attempt_for_time(run_id, call.get("task_id"), call.get("agent_id"), _when(call))
        correlation_basis = "attempt_time_window" if attempt else None
        if attempt is None and attempt_identity in delegated_agents:
            attempt, correlation_basis = delegated_agents[attempt_identity], "delegation_id"
        if attempt is None:
            attempt, correlation_basis = nearest_attempt_for_time(
                run_id, call.get("task_id"), call.get("agent_id"), _when(call),
            )
        attempt_confidence = "exact" if correlation_basis in {"attempt_time_window", "delegation_id"} else "inferred"
        call_id = str(call.get("_id") or call.get("call_id") or "")
        llm_id = f"llm:{run_id or 'project'}:{call.get('task_id') or 'none'}:{call.get('agent_id') or 'none'}:{call_id}"
        parent = attempt or tasks.get((run_id, str(call.get("task_id") or ""))) or parent_for_run(run_id, run_correlation)
        graph.node(node_id=llm_id, node_type="llm_call", parent_id=parent, run_id=run_id, title="LLM Turn",
                   summary=str(call.get("model") or ""), details={"call_id": call_id, "turn_index": call.get("turn_index"),
                                                                    "display_status": _llm_display_status(call),
                                                                    **({"correlation_basis": correlation_basis} if correlation_basis else {})},
                   correlation=_combine_correlation(run_correlation, attempt_confidence if attempt else "inferred"),
                   sequence=call.get("turn_index"), sort_key=_event_order(call))
        graph.edge(parent, llm_id, "contains", _combine_correlation(run_correlation, attempt_confidence if attempt else "inferred"))
        has_attempt_identity = attempt_identity in attempts
        if attempt:
            if attempt_confidence == "exact":
                llm_correlation_diagnostics["llm_exact_matches"] += 1
            else:
                llm_correlation_diagnostics["llm_inferred_matches"] += 1
        elif has_attempt_identity:
            identity_attempt_count = attempt_identity_counts.get(attempt_identity, 0)
            if identity_attempt_count <= 1:
                # A visible capture with one known lifecycle attempt is not a
                # missing source record.  Keep it under the task and expose
                # the timing uncertainty through diagnostics instead of
                # raising the project-wide completeness banner.
                llm_correlation_diagnostics["llm_temporal_unresolved"] += 1
            else:
                llm_correlation_diagnostics["llm_ambiguous"] += 1
                all_warnings.add("LLM capture attempt correlation ambiguous")
                partial = True
        if attempt:
            graph.by_id[attempt]["details"]["llm_rounds"].append({"call_id": call_id, "turn_index": call.get("turn_index")})
        uses = ((call.get("response") or {}).get("tool_uses") or [])
        for use in uses:
            if not isinstance(use, dict):
                continue
            tool_use_id = str(use.get("id") or "")
            if tool_use_id:
                llm_tool_requests[(run_id, call.get("task_id"), call.get("agent_id"), tool_use_id)].append((llm_id, _when(call)))
        if any(str(use.get("id") or "") not in ledger_ids for use in uses if isinstance(use, dict)):
            all_warnings.add("LLM tool request has no ledger execution")
            partial = True

    # Ledger truth prevents a model-declared tool use from becoming a synthetic
    # Tool Call.  For a real call, link only the unique closest preceding LLM
    # turn with the same run/task/agent/call identifier.
    for key, (tool_node, called_at) in technical_calls.items():
        candidates = [(node_id, at) for node_id, at in llm_tool_requests.get(key, []) if at <= called_at]
        if not candidates:
            continue
        latest_time = max(at for _, at in candidates)
        closest = [node_id for node_id, at in candidates if at == latest_time]
        if len(closest) == 1:
            old_parent = graph.by_id[tool_node]["parent_id"]
            graph.by_id[tool_node]["parent_id"] = closest[0]
            if old_parent:
                graph.remove_edge(old_parent, tool_node, "contains")
            graph.edge(closest[0], tool_node, "contains")
            graph.edge(closest[0], tool_node, "called")

    def is_descendant(node_id: str, ancestor_id: str) -> bool:
        current = graph.by_id.get(node_id)
        seen: set[str] = set()
        while current and current["id"] not in seen:
            current_id = current["id"]
            if current_id == ancestor_id:
                return True
            seen.add(current_id)
            parent_id = current.get("parent_id")
            current = graph.by_id.get(parent_id) if parent_id else None
        return False

    for message in message_rows:
        if message.get("type") != "approval":
            continue
        data, run_id = _data(message), message.get("run_id")
        source_correlation = _row_correlation(message)
        run_parent = parent_for_run(run_id, source_correlation)
        target = data.get("refine_target_node_id") or data.get("workflow_node_id")
        structural_parent = workflow_nodes.get((run_id, str(target))) if target else None
        if structural_parent is None and data.get("phase"):
            structural_parent = phases.get((run_id, str(data["phase"])))
        structural_correlation = (
            graph.by_id[structural_parent]["correlation"]
            if structural_parent in graph.by_id else source_correlation
        )
        attachment = resolve_approval_attachment(
            message,
            approval_time=_when(message),
            attempt_windows=attempt_windows,
            structural_parent_id=structural_parent,
            structural_correlation=structural_correlation,
            run_parent_id=run_parent,
            run_correlation=source_correlation,
            is_descendant=is_descendant,
        )
        parent, correlation = attachment.parent_id, attachment.correlation
        if attachment.basis == "ambiguous_time_fallback":
            all_warnings.add("Approval attempt correlation ambiguous")
            partial = True
        if parent == unresolved_history:
            correlation = "unresolved"
        approval_id = str(data.get("approval_id") or message.get("id"))
        status = data.get("approval_status") or message.get("status") or "pending"
        node_id = f"approval:{run_id or 'project'}:{approval_id}"
        graph.node(node_id=node_id, node_type="approval", parent_id=parent, run_id=run_id, title="HITL", summary=_text(status),
                   details={"approval_id": approval_id, "gate_node_id": data.get("gate_node_id"),
                            "refine_target_node_id": data.get("refine_target_node_id"),
                            "correlation_basis": attachment.basis,
                            "display_status": _status(status, "running"), "status": status},
                   correlation=correlation, sequence=message.get("sequence"),
                   sort_key=_event_order(message))
        graph.edge(parent, node_id, "awaits_approval", correlation)

    add_terminal_results(
        graph, project, runs, event_rows, attempts, latest_by_run, input_id, all_warnings,
        _data, _text, _status, wrappers, attempts_by_identity, tasks,
    )
    add_workflow_output(
        graph, project, input_id, project_id,
        list(artifacts or project.get("artifacts") or project_context.get("artifacts") or []),
        list(deployments or project.get("deployments") or project_context.get("deployments") or []),
        _status, _text,
    )
    for node in graph.nodes:
        if node["type"] != "agent_attempt":
            continue
        node["counts"] = {
            "llm_calls": sum(child["type"] == "llm_call" and child["parent_id"] == node["id"] for child in graph.nodes),
            "tool_calls": sum(child["type"] == "tool" and child["parent_id"] == node["id"] for child in graph.nodes),
        }
    for node in graph.nodes:
        if node["type"] == "agent_attempt":
            node["details"]["llm_rounds"] = sorted(
                node["details"].get("llm_rounds", []),
                key=lambda row: (row.get("turn_index") is None, row.get("turn_index"), row["call_id"]),
            )
        else:
            node["details"].pop("llm_rounds", None)
    graph.nodes.sort(key=lambda node: (
        node["type"] != "input", node.get("sequence") is None, node.get("sequence") or 0,
        node.get("_trace_sort_key") or (datetime.max.replace(tzinfo=timezone.utc), ""), node["id"],
    ))
    for node in graph.nodes:
        node.pop("_trace_sort_key", None)
    graph.edges.sort(key=lambda edge: edge["id"])
    warnings_out = sorted(all_warnings)
    def count(node_type: str) -> int:
        return sum(node["type"] == node_type for node in graph.nodes)

    return {
        "project_id": project_id,
        "nodes": graph.nodes, "edges": graph.edges,
        "stats": {"runs": count("run"), "phases": count("phase") + count("workflow_node"),
                  "tasks": count("task"), "auctions": count("auction"), "auction_bids": count("auction_bid"),
                  "agents": count("agent_attempt"), "agent_attempts": count("agent_attempt"),
                  "llm_calls": count("llm_call"), "tools": count("tool"),
                  "tool_calls": count("tool_call"), "tool_results": count("tool_result"),
                  "approvals": count("approval"), "snapshots": count("snapshot"),
                  "rollbacks": count("rollback"), "delegations": count("delegation"),
                  "errors": count("error"), "results": count("result"), "outputs": count("output")},
        "completeness": {"status": "partial" if partial or warnings_out else "complete", "warnings": warnings_out,
                         "diagnostics": llm_correlation_diagnostics,
                         "discarded_revert_branches_available": False},
    }
