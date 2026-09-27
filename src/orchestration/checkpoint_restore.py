"""Authorize and coordinate checkpoint restore without rewriting source journals."""

import asyncio
import logging

from integrations.checkpoint_client import CheckpointError
from orchestration.checkpoints import resolve_client
from storage.checkpoint_store import CheckpointStore
from storage.checkpoint_restore_store import CheckpointRestoreStore, RestoreError

logger = logging.getLogger(__name__)


async def restore_checkpoint(orch, project_id, run_id, point_id, tenant_id):
    store = CheckpointStore(orch.storage, orch.message_store)
    operations = CheckpointRestoreStore(orch.storage)
    async with orch.projects.get_lock(project_id):
        project = await orch.storage.load_project(project_id)
        source = await orch.storage.get_run(run_id)
        binding = await store.get_binding(run_id)
        point = await store.get_point(project_id, run_id, point_id)
        if (
            not project
            or project.get("tenant_id") != tenant_id
            or not source
            or source.get("project_id") != project_id
            or source.get("deleted_at")
            or not binding
            or binding.get("project_id") != project_id
            or binding.get("tenant_id") != tenant_id
            or not point
        ):
            raise RestoreError("checkpoint_not_found", 404)
        task = orch.project_tasks.get(project_id)
        if (task and not task.done()) or orch.projects.workflow_finalize_pending(
            project_id
        ):
            raise RestoreError("checkpoint_workflow_busy")
        current = await operations.get(project_id)
        previous_operation = None
        if current and current["state"] != "failed":
            restored_run = await orch.storage.get_run(current["run_id"])
            if (
                current["state"] == "dispatched"
                and restored_run
                and restored_run.get("run_status")
                in ("completed", "failed", "cancelled")
            ):
                previous_operation = current["run_id"]
            elif current["restored_from"] != {"run_id": run_id, "point_id": point_id}:
                raise RestoreError(
                    "checkpoint_restore_pending", run_id=current["run_id"]
                )
            else:
                from orchestration.checkpoint_runtime import recover_restore

                await recover_restore(orch, project_id, current)
                return response(await operations.get(project_id))
        if await orch.message_store.get_pending_approvals_for_project(
            project_id, limit=1
        ):
            raise RestoreError("checkpoint_approval_pending")
        if await orch.storage.db.a2a_task_state.find_one(
            {"project_id": project_id, "status": {"$ne": "closed"}}
        ):
            raise RestoreError("checkpoint_a2a_pending")
        active = await orch.storage.get_active_run(project_id)
        if active and active.get("run_status") not in (
            "completed",
            "failed",
            "cancelled",
        ):
            raise RestoreError("checkpoint_run_nonterminal")
        try:
            client = await resolve_client(orch.storage, binding)
        except CheckpointError as exc:
            raise RestoreError(exc.code, exc.status_code) from exc
        op = await operations.claim(
            project_id, run_id, point_id, previous_operation=previous_operation
        )
        try:
            context_id = await client.restore(point["point_id"])
            op = await operations.transition(
                project_id,
                op,
                "request-started",
                "external-restored",
                context_id=context_id,
            )
        except BaseException as exc:
            # Only an explicit adapter rejection proves restore did not happen.
            definite = isinstance(exc, CheckpointError) and not exc.outcome_unknown
            state = "failed" if definite else "outcome-unknown"
            try:
                await operations.transition(project_id, op, "request-started", state)
            except Exception:
                logger.warning(
                    "[CHECKPOINT] project_id=%s run_id=%s — failure state write failed; durable request-started remains closed",
                    project_id,
                    op["run_id"],
                )
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise RestoreError(
                exc.code if definite else "checkpoint_restore_outcome_unknown",
                exc.status_code if definite else 502,
                op["run_id"],
            ) from exc
        from orchestration.checkpoint_runtime import recover_restore

        await recover_restore(orch, project_id, op)
        return response(await operations.get(project_id))


def response(op):
    return {
        "run_id": op["run_id"],
        "restored_from": op["restored_from"],
        "status": op["state"],
    }
