from copy import deepcopy
import logging


logger = logging.getLogger(__name__)


async def publish_workflow_approval(
    shared_context,
    message_store,
    approval,
    *,
    approval_id,
    project_id,
    run_id,
    gate_node_id
):
    try:
        record = (
            await message_store.get_approval_by_id(approval_id)
            if message_store is not None
            else approval
        )
        data = record.get("data") if isinstance(record, dict) else None
        valid = (
            isinstance(data, dict)
            and record.get("status") == "approved"
            and record.get("project_id") == project_id
            and record.get("run_id") == run_id
            and (data.get("approval_id") or record.get("approval_id")) == approval_id
            and data.get("gate_node_id") == gate_node_id
        )
        if not valid:
            raise ValueError(
                "Approved record does not match the current project, run and gate"
            )
        projection = {
            "project_id": record["project_id"],
            "run_id": record.get("run_id"),
            "approval_id": data.get("approval_id") or record["approval_id"],
            "gate_node_id": data["gate_node_id"],
            "status": record["status"],
            "message_id": record.get("id") or record.get("message_id"),
            "data": deepcopy(data),
        }
        await shared_context.record_workflow_approval(projection)
    except Exception as exc:
        logger.warning(
            "[APPROVAL] project_id=%s run_id=%s approval_id=%s gate=%s — projection failed: %s",
            project_id,
            run_id,
            approval_id,
            gate_node_id,
            exc,
        )
        return {
            "status": "failed",
            "error_type": "workflow_approval_unavailable",
            "error": str(exc),
            "approval_id": approval_id,
            "node_id": gate_node_id,
        }
    logger.info(
        "[APPROVAL] project_id=%s run_id=%s approval_id=%s gate=%s — approved context saved",
        project_id,
        run_id,
        approval_id,
        gate_node_id,
    )
    result = {"status": "approved"}
    reviewed = data.get("agent_result")
    if isinstance(reviewed, dict) and "output" in reviewed:
        result["output"] = deepcopy(reviewed["output"])
    return result
