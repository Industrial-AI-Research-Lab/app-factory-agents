"""Read-only projection of one final workflow output."""
from __future__ import annotations

from typing import Any, Callable


def _order(row: dict) -> tuple[float, str, str]:
    try:
        sequence = float(row.get("sequence"))
    except (TypeError, ValueError):
        sequence = -1
    return sequence, str(row.get("created_at") or row.get("updated_at") or ""), str(
        row.get("deployment_id") or row.get("artifact_id") or row.get("id") or row.get("path") or ""
    )


def add_workflow_output(
    graph: Any, project: dict, input_id: str, project_id: str, artifacts: list[dict], deployments: list[dict],
    status_for: Callable[[Any, str], str], text_for: Callable[[Any], str | None],
) -> None:
    """Add the single, API-backed aggregate outcome for a project's workflow."""
    status = status_for(project.get("status"), "running")
    output_id = f"output:{project_id}"
    details: dict[str, Any] = {"payload_available": False}
    sequence = None
    if status == "failed":
        summary, display_status = "Workflow failed", "failed"
        details["output_source"] = "failure"
    else:
        completed_deployments = [
            row for row in deployments
            if status_for(row.get("status") or row.get("deploy_status"), "running") == "completed"
        ]
        if completed_deployments:
            deployment = max(completed_deployments, key=_order)
            summary = text_for(deployment.get("url") or deployment.get("name") or "Deployment")
            display_status, sequence = "completed", deployment.get("sequence")
            details.update({
                "output_source": "deployment",
                "deployment_id": deployment.get("deployment_id") or deployment.get("id"),
                "deployment_url": deployment.get("url"),
            })
        elif artifacts:
            artifact = max(artifacts, key=_order)
            summary = text_for(artifact.get("path") or artifact.get("name") or "Artifact")
            display_status, sequence = "completed", artifact.get("sequence")
            details.update({
                "output_source": "artifact",
                "artifact_id": artifact.get("artifact_id") or artifact.get("id"),
                "artifact_path": artifact.get("path") or artifact.get("name"),
            })
        elif status == "completed":
            summary, display_status = "Not recorded", "unknown"
            details["output_source"] = "not_recorded"
        else:
            summary, display_status = "Pending", "running"
            details["output_source"] = "pending"
    details["display_status"] = display_status
    graph.node(
        node_id=output_id, node_type="output", parent_id=input_id, run_id=None, title="Output", summary=summary,
        details=details, correlation="inferred", sequence=sequence,
    )
    graph.edge(input_id, output_id, "produced", "inferred")
