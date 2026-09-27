"""Recover confirmed remote contexts and transfer the project runtime once."""

from copy import deepcopy
import asyncio
import logging

from context.shared_context import SharedContext
from orchestration.checkpoints import resolve_client
from storage.checkpoint_store import CheckpointStore
from storage.checkpoint_restore_store import CheckpointRestoreStore, RestoreError
from telemetry.run_context import current_run_traceparent
from telemetry.run_scope import run_trace_scope
from telemetry.tracer import create_task_with_context

logger = logging.getLogger(__name__)


async def recover_restore(orch, project_id, op):
    """Called under the project lock; never repeats remote restore or submission."""
    operations = CheckpointRestoreStore(orch.storage)
    if op["state"] in ("request-started", "outcome-unknown"):
        raise RestoreError("checkpoint_restore_outcome_unknown", 502, op["run_id"])
    if op["state"] == "failed":
        return False
    if op["state"] == "dispatched":
        run = await orch.storage.get_run(op["run_id"])
        if run and run.get("run_status") in ("completed", "failed", "cancelled"):
            return False
        if await operations.has_continuation(project_id, op["run_id"]):
            return None
        cursor = await orch.storage.get_open_a2a_task_state(project_id)
        if not cursor or cursor.get("run_id") != op["run_id"]:
            raise RestoreError("checkpoint_dispatch_outcome_unknown", 502, op["run_id"])
        # Existing A2A reconciliation handles confirmed tasks / uncertain send.
        return None
    store = CheckpointStore(orch.storage, orch.message_store)
    binding = await store.get_binding(op["run_id"])
    if not binding:
        source = await store.get_binding(op["restored_from"]["run_id"])
        if not source:
            raise RestoreError("checkpoint_source_unavailable", 409, op["run_id"])
        binding = {
            **deepcopy(source),
            "run_id": op["run_id"],
            "context_id": op["context_id"],
        }
        await store.bind_run(binding)
    if op["state"] == "external-restored":
        try:
            client = await resolve_client(orch.storage, binding)
            async with run_trace_scope(
                orch.storage, orch.tracer, project_id, op["run_id"]
            ):
                await client.register_run(
                    binding["context_id"],
                    op["run_id"],
                    current_run_traceparent(orch.tracer),
                )
            op = await operations.transition(
                project_id, op, "external-restored", "ready"
            )
        except Exception as exc:
            raise RestoreError(
                "checkpoint_registration_pending", 502, op["run_id"]
            ) from exc
    # Each ready recovery constructs from the immutable platform inputs. Context
    # storage is project-scoped; it is not a historical scientific snapshot.
    project_doc = await orch.storage.load_project(project_id)
    previous = orch.active_projects.get(project_id) or {}
    inputs = binding["inputs"]
    sc = SharedContext(
        project_id,
        orch.storage,
        run_id=op["run_id"],
        message_store=orch.message_store,
        run_config=inputs.get("context", {}).get("run_config"),
    )
    sc._cache.update(deepcopy(inputs.get("context") or {}))
    sc._loaded = True
    old_sc = previous.get("shared_context")
    for name in (
        "_ephemeral_api_key",
        "_ephemeral_fallback_models",
        "_model_override",
        "_force_model_override",
        "_reasoning_effort_override",
        "_temperature_override",
        "_tenant_id",
    ):
        if old_sc is not None and hasattr(old_sc, name):
            setattr(sc, name, getattr(old_sc, name))
    sc._tenant_id = binding["tenant_id"]
    runtime = inputs.get("runtime") or {}
    for field, attribute in (
        ("model_id", "_model_override"),
        ("force_model", "_force_model_override"),
        ("reasoning_effort", "_reasoning_effort_override"),
        ("temperature", "_temperature_override"),
    ):
        if runtime.get(field) is not None:
            setattr(sc, attribute, runtime[field])
    project = {
        **project_doc,
        **deepcopy(inputs.get("runtime") or {}),
        "run_id": op["run_id"],
        "shared_context": sc,
        "status": "running",
        "user_prompt": sc.get("user_prompt"),
        "_checkpoint_binding": binding,
        "workflow_epoch": previous.get("workflow_epoch", 0),
    }
    if previous.get("resolved_agent_pool"):
        project["resolved_agent_pool"] = previous["resolved_agent_pool"]
    elif hasattr(orch.projects, "_resolve_agent_pool_for_project"):
        project[
            "resolved_agent_pool"
        ] = await orch.projects._resolve_agent_pool_for_project(
            orch.agent_pool, binding["tenant_id"]
        )
    orch.active_projects[project_id] = project
    orch._reinstate_runtime(project_id, sc)
    await sc._sync_to_db()
    await orch.storage.set_active_run(project_id, op["run_id"])
    await orch.storage.db.projects.update_one(
        {"project_id": project_id},
        {
            "$set": {
                **deepcopy(runtime),
                "status": "running",
                "user_prompt": sc.get("user_prompt"),
            }
        },
    )
    await orch.storage.update_run_status(op["run_id"], "running")
    await operations.transition(project_id, op, "ready", "dispatched")

    async def run():
        try:
            await orch.event_emitter.emit(
                "checkpoint_restored",
                op["run_id"],
                {"project_id": project_id, "restored_from": op["restored_from"]},
            )
        except Exception as exc:
            logger.warning(
                "[CHECKPOINT] project_id=%s run_id=%s — restore notification failed: %s",
                project_id,
                op["run_id"],
                type(exc).__name__,
            )
        await orch.workflow_engine.execute(
            project_id, binding["workflow_def"], start_node_id=binding["node_id"]
        )

    task = create_task_with_context(run())
    try:
        orch.register_workflow_task(project_id, task)
    except Exception as exc:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise RestoreError(
            "checkpoint_dispatch_outcome_unknown", 502, op["run_id"]
        ) from exc
    return True
