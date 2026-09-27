"""Terminal result and error projection for the semantic Trace."""
from __future__ import annotations

from typing import Any, Callable


def add_error(graph: Any, parent: str, run_id: str | None, source_id: str, title: str,
              correlation: str, sequence: int | None, details: dict[str, Any] | None = None) -> str:
    error_details = {"source_id": source_id, "display_status": "failed"}
    if details:
        error_details.update(details)
    node_id = f"error:{run_id or 'project'}:{source_id}"
    graph.node(
        node_id=node_id, node_type="error", parent_id=parent, run_id=run_id, title="Error", summary=title,
        details=error_details, correlation=correlation, sequence=sequence,
    )
    graph.edge(parent, node_id, "failed", correlation)
    return node_id


def add_terminal_results(
    graph: Any, project: dict, runs: list[dict], events: list[dict], attempts: dict, latest_by_run: dict,
    input_id: str, warnings: set[str], data_for: Callable[[dict], dict], text_for: Callable[[Any], str | None],
    status_for: Callable[[Any, str], str], wrappers: dict[tuple[str | None, str | None], str],
    attempts_by_identity: dict[tuple[Any, ...], str] | None = None,
    task_parents: dict[tuple[str | None, str], str] | None = None,
) -> None:
    """Attach recorded errors and only terminal failed results to their real location."""
    terminal_results: set[str | None] = set()
    last_error: dict[str | None, tuple[str, str]] = {}
    errors_by_identity: dict[tuple[Any, ...], str] = {}
    task_errors_by_label: dict[tuple[str | None, str], list[str]] = {}
    failed_results_by_error: dict[str, str] = {}
    project_failure_events = [event for event in events if event.get("event_type") == "project_failed"]
    ordered_events = [event for event in events if event.get("event_type") != "project_failed"] + project_failure_events
    for event in ordered_events:
        kind, data, run_id = event.get("event_type"), data_for(event), event.get("run_id")
        if kind == "task_completed":
            identity = (run_id, data.get("task_id"), data.get("agent_id"), str(data.get("attempt")))
            unresolved = data.get("_trace_correlation") == "unresolved"
            parent = (
                task_parents.get((run_id, str(data.get("task_id"))))
                if unresolved and task_parents and data.get("task_id") is not None else None
            ) or (
                data.get("_trace_attempt_node_id")
                or (attempts_by_identity.get(tuple(data.get("_trace_lifecycle_identity")))
                    if attempts_by_identity and isinstance(data.get("_trace_lifecycle_identity"), (tuple, list)) else None)
                or (attempts_by_identity.get(identity) if attempts_by_identity and data.get("attempt") is not None else None)
            ) or attempts.get((run_id, data.get("task_id"), data.get("agent_id"))) or latest_by_run.get(run_id) or input_id
            correlation = "unresolved" if unresolved else data.get("_trace_correlation") or "exact"
            if unresolved:
                warnings.add("Task terminal correlation unresolved")
            existing = next(
                (node for node in graph.nodes if node["type"] == "result" and node["parent_id"] == parent
                 and node.get("details", {}).get("result_source") in {"assistant_message", "not_recorded", "task_terminal"}),
                None,
            )
            if existing:
                existing["details"]["terminal_event_id"] = event.get("event_id")
                existing["details"]["display_status"] = "completed"
                if existing["details"].get("result_source") == "not_recorded":
                    existing["details"]["result_source"] = "task_terminal"
                    existing["summary"] = text_for(data.get("task_description") or "Task completed")
                continue
            result_id = f"result:{run_id or 'project'}:{event.get('event_id')}"
            graph.node(
                node_id=result_id, node_type="result", parent_id=parent, run_id=run_id, title="Result",
                summary=text_for(data.get("task_description") or "Task completed"),
                details={"display_status": "completed"}, sequence=data.get("sequence"), correlation=correlation,
            )
            graph.edge(parent, result_id, "produced", correlation)
            continue
        if kind not in {"project_failed", "a2a_agent_failed", "task_failed", "task_error"}:
            continue
        wrapper = wrappers.get((run_id, data.get("node_id"))) if kind == "a2a_agent_failed" else None
        parent = wrapper
        identity = (run_id, data.get("task_id"), data.get("agent_id"), str(data.get("attempt")))
        parent = parent or (
            data.get("_trace_attempt_node_id")
            or (attempts_by_identity.get(tuple(data.get("_trace_lifecycle_identity")))
                if attempts_by_identity and isinstance(data.get("_trace_lifecycle_identity"), (tuple, list)) else None)
            or (attempts_by_identity.get(identity) if attempts_by_identity and data.get("attempt") is not None else None)
        ) or attempts.get((run_id, data.get("task_id"), data.get("agent_id"))) or latest_by_run.get(run_id) or input_id
        correlation = data.get("_trace_correlation") or ("exact" if parent != input_id else "unresolved")
        if parent == input_id:
            warnings.add("Error location not recorded")
        if wrapper:
            graph.by_id[parent]["details"]["display_status"] = "failed"
        if kind in {"task_failed", "task_error"}:
            identity = (
                "final" if kind == "task_failed" else "retry",
                run_id,
                data.get("task_id"),
                data.get("agent_id"),
                str(data.get("agent_retry") if kind == "task_error" and data.get("agent_retry") is not None
                    else data.get("attempt")),
            )
            existing_error = errors_by_identity.get(identity)
            if existing_error:
                details = graph.by_id[existing_error].setdefault("details", {})
                source_ids = details.setdefault("source_event_ids", [])
                event_id = event.get("event_id")
                if event_id and event_id not in source_ids:
                    source_ids.append(event_id)
                last_error[run_id] = (existing_error, correlation)
                continue
        failure_title = text_for(
            data.get("error_type") or data.get("error") or data.get("reason") or "Workflow failed"
        ) or "Workflow failed"
        error_details = {"source_event_ids": [event.get("event_id")]} if event.get("event_id") else {}
        if data.get("reason"):
            error_details["reason"] = text_for(data.get("reason"))
        if kind == "project_failed":
            candidates = task_errors_by_label.get((run_id, failure_title), [])
            existing_error = candidates[0] if len(candidates) == 1 else None
            if existing_error:
                error = existing_error
                details = graph.by_id[error].setdefault("details", {})
                source_ids = details.setdefault("source_event_ids", [])
                event_id = event.get("event_id")
                if event_id and event_id not in source_ids:
                    source_ids.append(event_id)
                if data.get("reason") and not details.get("reason"):
                    details["reason"] = text_for(data.get("reason"))
            else:
                error = add_error(
                    graph, parent, run_id, str(event.get("event_id") or kind), failure_title,
                    correlation, data.get("sequence"), error_details,
                )
        else:
            error = add_error(
                graph, parent, run_id, str(event.get("event_id") or kind),
                failure_title, correlation, data.get("sequence"), error_details,
            )
        if kind in {"task_failed", "task_error"}:
            errors_by_identity[identity] = error
            if kind == "task_failed":
                task_errors_by_label.setdefault((run_id, failure_title), []).append(error)
        last_error[run_id] = (error, correlation)
        if kind != "project_failed":
            continue
        result_id = failed_results_by_error.get(error)
        if result_id is None:
            result_id = f"result:{run_id or 'project'}:failed:{error}"
            graph.node(
                node_id=result_id, node_type="result", parent_id=error, run_id=run_id, title="Result", summary="Failed",
                details={"display_status": "failed"}, correlation=correlation, sequence=data.get("sequence"),
            )
            graph.edge(error, result_id, "failed", correlation)
            failed_results_by_error[error] = result_id
        terminal_results.add(run_id)

    for run in runs:
        run_id = run.get("run_id")
        if run_id in terminal_results or status_for(run.get("run_status"), "running") != "failed":
            continue
        error, correlation = last_error.get(run_id, (None, None))
        if error is None:
            parent = latest_by_run.get(run_id) or input_id
            correlation = "exact" if parent != input_id else "unresolved"
            if parent == input_id:
                warnings.add("Error location not recorded")
            error = add_error(graph, parent, run_id, "run-status", "Workflow failed", correlation, None)
        result_id = f"result:{run_id}:failed:run-status"
        graph.node(
            node_id=result_id, node_type="result", parent_id=error, run_id=run_id, title="Result", summary="Failed",
            details={"display_status": "failed"}, correlation=correlation,
        )
        graph.edge(error, result_id, "failed", correlation)
