"""
Main Orchestrator

Thin coordinator that delegates to specialized managers:
- ProjectManager: Project lifecycle
- ApprovalManager: Approval gates
- PhaseRunner: Phase execution
- TaskExecutor: Task execution
- RevertManager: Snapshot/revert operations
"""

from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import asyncio
import json
import logging
import os

from agents.generic_agent import GenericAgent
from agents.run_rehydration import (
    attempt_has_outcome_unknown,
    find_interrupted_attempt,
    last_completed_node_attempt,
)
from context.shared_context import SharedContext
from orchestration.snapshot_manager import SnapshotManager
from orchestration.auction import TaskAuction
from orchestration.approval_manager import ApprovalManager
from orchestration.project_manager import ProjectManager
from orchestration.phase_runner import PhaseRunner
from orchestration.task_executor import TaskExecutor
from orchestration.revert_manager import RevertManager
from orchestration.workflow_engine import WorkflowEngine
from orchestration.backend_boot import build_recovery_metadata
from orchestration.workflow_task_lifecycle import (
    EXTERNAL_TOOL_OUTCOME_UNKNOWN,
    ResumeBlockedLookupError,
    TOOL_INTERRUPTED_BEFORE_RESULT,
)
from orchestration.a2a_cancellation_wal import A2ACancellationWal
from integrations import A2AClientFactory
from a2a.utils.errors import TaskNotFoundError
from config.seed import seed_workflows
from orchestration.project_workflow import resolve_project_workflow
from telemetry.run_scope import run_trace_scope
from telemetry.tracer import get_tracer, create_task_with_context
from schemas import (
    TaskSchema,
    ResultSchema,
    ApprovalStatus,
    ApprovalType,
    ApprovalSchema,
    EventSchema,
)
from deploy.build_helpers import check_artifacts_ready

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main orchestration engine for AppFactory.
    
    Thin coordinator that delegates to specialized managers.
    """

    _A2A_CANCELLATION_FOREGROUND_TIMEOUT_SECONDS = 1.0
    _A2A_CANCELLATION_RETRY_BACKOFF_CAP_SECONDS = 300
    
    def __init__(
        self,
        storage_backend,
        llm_client,
        tool_registry,
        event_emitter,
        container_manager=None,
        deploy_service: Optional[Any] = None,
        message_store=None,
        artifact_store=None,
        mcp_executor: Optional[Any] = None,
        a2a_cancellation_wal: A2ACancellationWal | None = None,
    ):
        self.storage = storage_backend
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.event_emitter = event_emitter
        self.container_manager = container_manager
        self.deploy_service = deploy_service
        self.message_store = message_store
        self.artifact_store = artifact_store
        # Optional MCP executor used for tool execution in helper agents
        self.mcp_executor = mcp_executor
        self.a2a_cancellation_wal = a2a_cancellation_wal or A2ACancellationWal(
            os.getenv(
                "A2A_CANCELLATION_WAL_PATH",
                Path.cwd() / ".AppFactory-data" / "a2a-cancellation.sqlite3",
            )
        )
        # Agent pool (pre-defined agents)
        self.agent_pool: List[Any] = []

        # Deploy idempotency: approval_ids for which we already ran _execute_deploy (avoid double run from side_effects + workflow_engine)
        self._deploy_executed_approval_ids: Set[str] = set()

        # Live `_await_and_execute` tasks keyed by deploy approval_id. Used
        # by `_respawn_deploy_executor` to skip re-spawning when an executor
        # is already awaiting the same approval (happy path: no restart
        # between seed and approve). Cleaned up by the task itself in its
        # finally clause. `_execute_deploy` is already idempotent per
        # approval_id (line ~1063), so a spurious duplicate is at worst a
        # wasted await — but tracking avoids the waste entirely.
        self._deploy_executor_tasks: Dict[str, asyncio.Task] = {}
        
        # Auction system
        self.auction = TaskAuction(timeout=10)
        self.auction.set_event_emitter(event_emitter)
        
        # Tracer for telemetry
        self.tracer = get_tracer()
        
        # Snapshot manager
        self.snapshot_manager = SnapshotManager(storage_backend, container_manager, event_emitter, message_store)
        
        # Initialize specialized managers
        self.projects = ProjectManager(storage_backend, llm_client, event_emitter, container_manager, message_store)
        self.approvals = ApprovalManager(
            storage_backend, event_emitter, self.snapshot_manager, message_store,
            container_manager=container_manager, artifact_store=self.artifact_store
        )
        self.phases = PhaseRunner(self.auction, storage_backend, event_emitter, self.tracer)
        self.tasks = TaskExecutor(
            self.auction, event_emitter, self.snapshot_manager, self.tracer,
            artifact_store=self.artifact_store,
            message_store=message_store,
            storage_backend=storage_backend,
        )
        self.reverts = RevertManager(
            storage_backend, event_emitter, self.snapshot_manager, container_manager,
            message_store, artifact_store=self.artifact_store
        )
        
        # Workflow engine (DAG executor)
        self.workflow_engine = WorkflowEngine(self)
        # Cancellation requests that could not reach MongoDB must outlive the
        # workflow task which observed them. They remain process-managed until
        # MongoDB accepts the durable outbox entry or the remote task is known
        # terminal.
        self._a2a_cancellation_delivery_tasks: Dict[tuple, asyncio.Task] = {}
    
    # === Delegate properties for backward compatibility ===
    @property
    def active_projects(self) -> Dict[str, Dict]:
        return self.projects.active_projects
    
    @property
    def pending_approvals(self) -> Dict[str, Dict]:
        return self.approvals.pending_approvals
    
    @property
    def project_tasks(self) -> Dict[str, asyncio.Task]:
        return self.projects.project_tasks

    def is_workflow_live(self, project_id: str) -> bool:
        """True if a workflow runtime task is registered and still running.

        `ensure_workflow_running` returns False for BOTH "already running"
        and "nothing to restore"; callers that must tell those apart (the
        answer route's resumed flag) check liveness with this.
        """
        task = self.project_tasks.get(project_id)
        return task is not None and not task.done()

    @property
    def project_locks(self) -> Dict[str, asyncio.Lock]:
        return self.projects.project_locks
    
    # === Atomic State Update Methods (Bulletproof Persistence) ===
    async def update_project_phase(self, project_id: str, phase: str) -> None:
        """Update project phase atomically (memory + DB)."""
        await self.projects.update_project_phase(project_id, phase)
    
    async def update_project_status(self, project_id: str, status: str) -> None:
        """Update project status atomically (memory + DB)."""
        await self.projects.update_project_status(project_id, status)
    
    # === Agent Registration ===
    def register_agent(self, agent):
        """Add agent to the pool"""
        agent.inject_dependencies(
            shared_context=None,
            tool_registry=self.tool_registry,
            llm_client=self.llm_client,
            event_emitter=self.event_emitter
        )
        self.agent_pool.append(agent)
        for registered_agent in self.agent_pool:
            try:
                registered_agent.agent_pool = self.agent_pool
            except Exception:
                pass
        
        if agent.agent_type.value == "critic":
            self.auction.critic_agent = agent
    
    # === State Recovery ===
    async def recover_state_from_db(self):
        """Recover running projects from MongoDB. Approvals are now derived from messages on-demand."""
        await self.projects.recover_state_from_db(self.agent_pool)
        
        # Note: approval recovery removed - messages are now the single source of truth
        # Approvals are loaded on-demand when projects are accessed
        
        logger.info(f"🎉 State recovery complete: {len(self.active_projects)} projects (approvals load on-demand from messages)")
    
    # === Project Lifecycle (delegates to ProjectManager) ===
    # TODO: legacy proxy; current title generation logic is evolving to use run_config in ProjectManager.
    # Remove this method if no callers remain.
    async def generate_project_title(
            self,
            user_prompt: str,
            api_key_override: Optional[str] = None,
            run_config: Optional[dict] = None,
    ) -> str:
        return await self.projects.generate_project_title(
            user_prompt,
            api_key_override=api_key_override,
            run_config=run_config,
        )

    async def start_project(
        self,
        user_prompt: str,
        project_name: Optional[str] = None,
        approval_mode: Optional[str] = None,
        api_key_override: Optional[str] = None,
        model_override: Optional[str] = None,
        fallback_models_override: Optional[List[str]] = None,
        force_model: bool = False,
        workflow_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        run_config_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        created_by: Optional[str] = None,
    ) -> str:
        return await self.projects.start_project(
            user_prompt=user_prompt,
            agent_pool=self.agent_pool,
            project_name=project_name,
            approval_mode=approval_mode,
            api_key_override=api_key_override,
            model_override=model_override,
            fallback_models_override=fallback_models_override,
            force_model=force_model,
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            run_config_id=run_config_id,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            created_by=created_by,
        )
    
    async def cancel_project(
        self,
        project_id: str,
        reason: str = "Cancelled by user",
        *,
        already_locked: bool = False,
    ):
        try:
            await self.request_active_a2a_task_cancellation(project_id, reason=reason)
        except Exception:
            logger.exception(
                "[A2A_RECOVER] project_id=%s action=project_cancel_remote status=failed",
                project_id,
            )
        finally:
            await self.projects.cancel_project(
                project_id, reason, already_locked=already_locked
            )

    async def _cancel_project_holding_lock(
        self, project_id: str, reason: str = "Cancelled by user"
    ):
        """For revert paths that already hold ``projects.get_lock(project_id)``."""
        await self.cancel_project(project_id, reason, already_locked=True)

    async def request_active_a2a_task_cancellation(
        self, project_id: str, *, reason: str
    ) -> bool:
        """Request cancellation even when the workflow task is no longer in memory."""
        try:
            cursor = await self.storage.get_active_a2a_task_state(project_id)
        except Exception as exc:
            logger.warning(
                "[A2A_RECOVER] project_id=%s action=project_cancel_cursor_lookup "
                "status=failed error=%s",
                project_id,
                exc,
            )
            return False
        if not cursor or not cursor.get("task_id"):
            return False

        try:
            await self.storage.mark_a2a_task_cancellation_pending(
                project_id=project_id,
                node_id=cursor["node_id"],
                run_id=cursor.get("run_id"),
                reason=reason,
            )
        except Exception as exc:
            logger.warning(
                "[A2A_RECOVER] project_id=%s action=project_cancel_persist "
                "status=failed error=%s",
                project_id,
                exc,
            )
        try:
            return await self._cancel_a2a_cursor(cursor, project_id=project_id)
        except Exception:
            logger.exception(
                "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                "action=project_cancel_remote status=failed",
                project_id,
                cursor.get("node_id"),
                cursor.get("task_id"),
            )
            return False

    async def request_a2a_cancellation_delivery(
        self,
        *,
        a2a_client,
        project_id: str,
        run_id: str | None,
        node_id: str,
        tenant_id: str,
        server_id: str,
        task_id: str,
        context_id: str | None,
        reason: str,
    ) -> bool:
        """Record and request cancellation without trapping a workflow in retry.

        A cancelled workflow gets one short foreground opportunity to create a
        durable outbox cursor. If MongoDB is unavailable, a tracked delivery
        task takes over; it is deliberately not a child of the workflow task.
        """
        request = {
            "project_id": project_id,
            "run_id": run_id,
            "node_id": node_id,
            "tenant_id": tenant_id,
            "server_id": server_id,
            "task_id": task_id,
            "context_id": context_id,
            "reason": reason,
        }
        try:
            await self.a2a_cancellation_wal.record_intent(**request)
        except Exception:
            logger.exception(
                "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                "action=record_cancellation_wal status=failed",
                project_id,
                node_id,
                task_id,
            )
            return False
        try:
            intent = await asyncio.wait_for(
                self.storage.persist_a2a_task_cancellation_intent(**request),
                timeout=getattr(
                    self, "_A2A_CANCELLATION_FOREGROUND_TIMEOUT_SECONDS", 1.0
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._enqueue_a2a_cancellation_delivery(request, a2a_client)
            logger.warning(
                "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                "action=handoff_cancellation_delivery status=memory_pending error=%s",
                project_id,
                node_id,
                task_id,
                exc,
            )
            return False

        if (intent or {}).get("status") == "closed":
            await self._mark_a2a_cancellation_wal_closed(request)
            return True
        await self._mark_a2a_cancellation_wal_mongo_pending(request)
        try:
            return await self._cancel_a2a_cursor(
                {**request, **(intent or {})},
                project_id=project_id,
                a2a_client=a2a_client,
                operation_timeout=getattr(
                    self, "_A2A_CANCELLATION_FOREGROUND_TIMEOUT_SECONDS", 1.0
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                "action=foreground_cancellation status=durable_pending error=%s",
                project_id,
                node_id,
                task_id,
                exc,
            )
            return False

    def _enqueue_a2a_cancellation_delivery(
        self, request: Dict[str, Any], a2a_client
    ) -> asyncio.Task:
        """Create at most one in-memory delivery task per remote A2A task."""
        key = (
            request["project_id"],
            request.get("run_id"),
            request["node_id"],
            request["task_id"],
        )
        tasks = getattr(self, "_a2a_cancellation_delivery_tasks", None)
        if tasks is None:
            tasks = {}
            self._a2a_cancellation_delivery_tasks = tasks
        existing = tasks.get(key)
        if existing is not None and not existing.done():
            return existing

        task = asyncio.create_task(
            self._deliver_undurable_a2a_cancellation(request, a2a_client),
            name=f"a2a-cancellation-delivery:{request['project_id']}:{request['task_id']}",
        )
        tasks[key] = task

        def _forget(completed_task: asyncio.Task) -> None:
            if tasks.get(key) is completed_task:
                tasks.pop(key, None)

        task.add_done_callback(_forget)
        return task

    async def _deliver_undurable_a2a_cancellation(
        self, request: Dict[str, Any], a2a_client
    ) -> None:
        """Deliver a WAL-backed cancellation after the workflow task has exited."""
        attempt = 0
        final_status = (
            request.get("final_status")
            if request.get("state") == "remote_terminal"
            else None
        )
        while True:
            attempt += 1
            if final_status is None:
                final_status = await self._cancel_undurable_a2a_task(request, a2a_client)
                if final_status:
                    await self._mark_a2a_cancellation_wal_remote_terminal(
                        request, final_status=final_status
                    )
            try:
                intent = await self.storage.persist_a2a_task_cancellation_intent(
                    **request
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                backoff_seconds = min(
                    2 ** (attempt - 1),
                    getattr(
                        self, "_A2A_CANCELLATION_RETRY_BACKOFF_CAP_SECONDS", 300
                    ),
                )
                retry_status = (
                    "retrying_backoff_capped"
                    if backoff_seconds
                    == getattr(
                        self, "_A2A_CANCELLATION_RETRY_BACKOFF_CAP_SECONDS", 300
                    )
                    else "retrying"
                )
                logger.warning(
                    "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                    "action=deliver_cancellation_intent status=%s attempt=%d "
                    "backoff_seconds=%s error=%s",
                    request["project_id"],
                    request["node_id"],
                    request["task_id"],
                    retry_status,
                    attempt,
                    backoff_seconds,
                    exc,
                )
                await asyncio.sleep(backoff_seconds)
                continue

            if (intent or {}).get("status") == "closed":
                await self._mark_a2a_cancellation_wal_closed(request)
                return
            await self._mark_a2a_cancellation_wal_mongo_pending(request)
            if final_status:
                try:
                    await self.storage.close_a2a_task_state(
                        project_id=request["project_id"],
                        node_id=request["node_id"],
                        run_id=request.get("run_id"),
                        final_status=final_status,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                        "action=close_terminal_cancellation status=durable_pending",
                        request["project_id"],
                        request["node_id"],
                        request["task_id"],
                    )
                    return
                await self._mark_a2a_cancellation_wal_closed(request)
                return
            try:
                closed = await self._cancel_a2a_cursor(
                    {**request, **(intent or {})},
                    project_id=request["project_id"],
                    a2a_client=a2a_client,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                    "action=deliver_cancellation status=durable_pending",
                    request["project_id"],
                    request["node_id"],
                    request["task_id"],
                )
            else:
                if not closed:
                    logger.warning(
                        "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                        "action=deliver_cancellation status=remote_retry_pending",
                        request["project_id"],
                        request["node_id"],
                        request["task_id"],
                    )
            return

    async def _cancel_undurable_a2a_task(
        self, request: Dict[str, Any], a2a_client
    ) -> str | None:
        """Return a confirmed terminal remote state, if cancellation reached one."""
        async with run_trace_scope(
            self.storage,
            self.tracer,
            request["project_id"],
            request.get("run_id"),
        ):
            try:
                client = a2a_client or A2AClientFactory.get_client(self.storage)
                result = await client.tasks_cancel(
                    request["server_id"], request["tenant_id"], request["task_id"]
                )
            except TaskNotFoundError:
                logger.info(
                    "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                    "action=cancel_undurable_task status=not_found",
                    request["project_id"], request["node_id"], request["task_id"],
                )
                return "a2a_task_not_found_during_cancellation"
            except Exception as exc:
                logger.warning(
                    "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                    "action=cancel_undurable_task status=retry_pending error=%s",
                    request["project_id"], request["node_id"], request["task_id"], exc,
                )
                return None

            state = (result or {}).get("status", {}).get("state")
            if not state or state in {"TASK_STATE_SUBMITTED", "TASK_STATE_WORKING"}:
                logger.warning(
                    "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                    "action=cancel_undurable_task status=retry_pending remote_state=%s",
                    request["project_id"], request["node_id"], request["task_id"], state,
                )
                return None
            logger.info(
                "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                "action=cancel_undurable_task status=terminal remote_state=%s",
                request["project_id"], request["node_id"], request["task_id"], state,
            )
            return state

    async def recover_a2a_cancellation_deliveries(self) -> int:
        """Resume cancellation requests recorded before a process restart."""
        try:
            intents = await self.a2a_cancellation_wal.list_open_intents()
        except Exception:
            logger.exception("[A2A_RECOVER] action=load_cancellation_wal status=failed")
            return 0
        for intent in intents:
            self._enqueue_a2a_cancellation_delivery(intent, a2a_client=None)
        logger.info(
            "[A2A_RECOVER] action=load_cancellation_wal status=scheduled count=%d",
            len(intents),
        )
        return len(intents)

    async def _mark_a2a_cancellation_wal_mongo_pending(
        self, request: Dict[str, Any]
    ) -> None:
        wal = getattr(self, "a2a_cancellation_wal", None)
        if wal is None:
            return
        try:
            await wal.mark_mongo_pending(request)
        except Exception:
            logger.exception(
                "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                "action=mark_cancellation_wal_mongo_pending status=failed",
                request["project_id"], request["node_id"], request["task_id"],
            )

    async def _mark_a2a_cancellation_wal_remote_terminal(
        self, request: Dict[str, Any], *, final_status: str
    ) -> None:
        wal = getattr(self, "a2a_cancellation_wal", None)
        if wal is None:
            return
        try:
            await wal.mark_remote_terminal(
                request, final_status=final_status
            )
        except Exception:
            logger.exception(
                "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                "action=mark_cancellation_wal_remote_terminal status=failed",
                request["project_id"], request["node_id"], request["task_id"],
            )

    async def _mark_a2a_cancellation_wal_closed(
        self, request: Dict[str, Any]
    ) -> None:
        wal = getattr(self, "a2a_cancellation_wal", None)
        if wal is None:
            return
        try:
            await wal.mark_closed(request)
        except Exception:
            logger.exception(
                "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                "action=mark_cancellation_wal_closed status=failed",
                request["project_id"], request["node_id"], request["task_id"],
            )

    async def stop_a2a_cancellation_delivery_tasks(self) -> None:
        """Stop managed deliveries before their A2A and storage dependencies close."""
        tasks = list(getattr(self, "_a2a_cancellation_delivery_tasks", {}).values())
        active_tasks = [task for task in tasks if not task.done()]
        if not active_tasks:
            return
        for task in active_tasks:
            task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.gather(*active_tasks, return_exceptions=True)),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[A2A_RECOVER] action=stop_cancellation_deliveries status=cancel_timeout count=%d",
                len(active_tasks),
            )

    async def _cancel_a2a_cursor(
        self,
        cursor: Dict[str, Any],
        *,
        project_id: str,
        a2a_client=None,
        operation_timeout: float | None = None,
    ) -> bool:
        """Cancel one durable cursor; leave it pending until a terminal reply."""
        task_id = cursor.get("task_id")
        if not task_id:
            return False
        if operation_timeout is not None:
            return await asyncio.wait_for(
                self._cancel_a2a_cursor(
                    cursor, project_id=project_id, a2a_client=a2a_client,
                ),
                timeout=operation_timeout,
            )
        async with run_trace_scope(
            self.storage, self.tracer, project_id, cursor.get("run_id")
        ):
            return await self._cancel_a2a_cursor_body(
                cursor,
                project_id=project_id,
                a2a_client=a2a_client,
            )

    async def _cancel_a2a_cursor_body(
        self,
        cursor: Dict[str, Any],
        *,
        project_id: str,
        a2a_client=None,
    ) -> bool:
        """Perform remote cancellation and close its durable cursor."""
        task_id = cursor.get("task_id")
        node_id = cursor.get("node_id")
        run_id = cursor.get("run_id")
        server_id = cursor.get("server_id")
        tenant_id = cursor.get("tenant_id")
        try:
            client = a2a_client or A2AClientFactory.get_client(self.storage)
            result = await client.tasks_cancel(server_id, tenant_id, task_id)
        except TaskNotFoundError:
            state = "a2a_task_not_found_during_cancellation"
        except Exception as exc:
            logger.warning(
                "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                "action=cancel_pending_cursor status=retry_pending error=%s",
                project_id,
                node_id,
                task_id,
                exc,
            )
            return False
        else:
            state = (result or {}).get("status", {}).get("state")
            if not state or state in {"TASK_STATE_SUBMITTED", "TASK_STATE_WORKING"}:
                logger.warning(
                    "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                    "action=cancel_pending_cursor status=retry_pending remote_state=%s",
                    project_id,
                    node_id,
                    task_id,
                    state,
                )
                return False

        await self.storage.close_a2a_task_state(
            project_id=project_id,
            node_id=node_id,
            run_id=run_id,
            final_status=state,
        )
        await self._mark_a2a_cancellation_wal_closed(cursor)
        logger.info(
            "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
            "action=cancel_pending_cursor status=closed remote_state=%s",
            project_id,
            node_id,
            task_id,
            state,
        )
        return True
    
    def register_workflow_task(self, project_id: str, task: asyncio.Task):
        self.projects.register_workflow_task(project_id, task)
    
    # === Helper methods ===
    def _is_cancelled(self, project_id: str) -> bool:
        return self.projects.is_cancelled(project_id)
    
    def _reinstate_runtime(self, project_id: str, sc: SharedContext) -> None:
        self.projects.reinstate_runtime(project_id, sc, self.agent_pool)
    
    # === Main Workflow ===
    async def get_resume_blocked_reason(self, project_id: str) -> str | None:
        """Return durable run-doc reason when POST /resume must be refused."""
        try:
            from storage.checkpoint_restore_store import CheckpointRestoreStore
            blocked = await CheckpointRestoreStore(self.storage).blocked_reason(project_id)
            if blocked:
                return blocked
            run = await self.storage.get_active_run(project_id)
        except Exception as exc:
            logger.warning(
                "[REHYDRATE] project_id=%s resume_blocked lookup failed: %s",
                project_id,
                exc,
            )
            raise ResumeBlockedLookupError(str(exc)) from exc
        if not run:
            return None
        reason = run.get("resume_blocked_reason")
        return str(reason) if reason else None

    async def _resume_blocked_for_project(self, project_id: str) -> bool:
        """Fail-closed: True when any spawn path must refuse resume."""
        try:
            blocked = await self.get_resume_blocked_reason(project_id)
        except ResumeBlockedLookupError as exc:
            logger.warning(
                "[REHYDRATE] project_id=%s resume safety lookup failed — not spawning: %s",
                project_id,
                exc,
            )
            return True
        if blocked:
            logger.warning(
                "[REHYDRATE] project_id=%s resume blocked (%s) — not spawning",
                project_id,
                blocked,
            )
            return True
        return False

    async def ensure_workflow_running(
        self, project_id: str, *, project_doc: dict | None = None
    ) -> bool:
        """
        Rehydrate the workflow runtime for a project after a backend restart.

        Idempotent — safe to call repeatedly from page-load, approve, and
        backend-startup paths. The workflow runtime is in-memory only; backend
        restarts orphan it. This coordinator restores the singleton runtime
        invariant: at most one runtime per project, restored from whichever
        durable cursor exists — a parked gate's pending approval first, else
        the tool journal's interrupted attempt (ADR-0009).

        Returns True if a runtime was spawned this call, False if no action
        was taken (already running, nothing to restore from, or terminal
        status).
        """
        # Fast-path bail-out: if a runtime is already registered for this
        # project, skip the lock entirely. The locked block below re-checks.
        existing = self.project_tasks.get(project_id)
        if existing is not None and not existing.done():
            return False

        if not self.message_store:
            return False

        # Serialize concurrent rehydration callers per project. Without this,
        # GET /projects/{id} (page-load) and POST /approve/{type} firing in
        # the same window each pass the singleton guard and both register a
        # runtime — the second overwrites the first, but the first keeps
        # running, so both walk past the same gate and execute the next
        # phase twice. Uses the same lock as revert_project and
        # _execute_deploy so the locking story stays consistent.
        async with self.projects.get_lock(project_id):
            if self.projects.workflow_finalize_pending(project_id):
                logger.info(
                    "[REHYDRATE] project_id=%s workflow finalize pending — not spawning",
                    project_id,
                )
                return False

            existing = self.project_tasks.get(project_id)
            if existing is not None and not existing.done():
                return False

            from orchestration.checkpoint_runtime import recover_restore
            from storage.checkpoint_restore_store import CheckpointRestoreStore, RestoreError
            operation = await CheckpointRestoreStore(self.storage).get(project_id)
            if operation and operation.get("state") == "dispatched":
                active = await self.storage.get_active_run(project_id)
                if active and active.get("run_id") != operation["run_id"]:
                    operation = None
            if operation and operation.get("state") != "failed":
                try:
                    recovered = await recover_restore(self, project_id, operation)
                except RestoreError as exc:
                    logger.warning("[CHECKPOINT] project_id=%s run_id=%s — recovery blocked: %s", project_id, exc.run_id, exc.code)
                    return False
                if recovered is not None:
                    return recovered
                # A caller may still hold the terminal source-project read from
                # before activation. Continue against the restored Run's state.
                project_doc = await self.storage.load_project(project_id)

            if await self._resume_blocked_for_project(project_id):
                return False

            if project_doc:
                terminal = project_doc.get("status")
                if terminal in ("failed", "cancelled", "completed"):
                    await self._maybe_heal_run_when_project_terminal(
                        project_id,
                        project_doc=project_doc,
                        write_heal=False,
                    )
                    return False

            try:
                pending = await self.message_store.get_pending_approvals_for_project(project_id, limit=1)
            except Exception as e:
                logger.warning("[REHYDRATE] %s: failed to query pending approvals: %s", project_id, e)
                return False
            if not pending:
                # No parked gate. Two other durable cursors can explain a mid-run
                # restart, tried in order: the tool journal's interrupted attempt
                # (ADR-0009), then — if that found nothing to resume — an open
                # a2a_task_state cursor left by an A2A node that began durable polling
                # (AppFactory-280 Step 4). Both are lazy/on-demand, same as the gate
                # check above: nothing is scanned unless ensure_workflow_running
                # is actually called.
                if await self._maybe_resume_interrupted_run(
                    project_id, project_doc=project_doc
                ):
                    return True
                return await self._maybe_resume_interrupted_a2a(project_id)

            approval_doc = pending[0]
            if (approval_doc.get("data") or {}).get("delegation_gate"):
                # Mid-agent gated-delegation pause, not a graph gate. The agent's
                # in-memory execution state cannot be rehydrated (consistent with the
                # "mid-phase parks are not rehydrated" policy above), and the synthetic
                # gate_node_id has no matching workflow node — rehydrating would crash
                # the FSM on an unknown start node. Nothing to resume here.
                logger.info(
                    "[REHYDRATE] %s: top pending approval is a delegation gate; skipping",
                    project_id,
                )
                return False
            approval_id = approval_doc.get("approval_id") or (approval_doc.get("data") or {}).get("approval_id")
            gate_node_id = (approval_doc.get("data") or {}).get("gate_node_id")
            if not approval_id or not gate_node_id:
                logger.warning(
                    "[REHYDRATE] %s: pending approval missing approval_id (%s) or gate_node_id (%s)",
                    project_id, approval_id, gate_node_id,
                )
                return False

            project = self.active_projects.get(project_id)
            if not project:
                try:
                    project = await self.projects.load_project_to_active(project_id, self.agent_pool)
                except Exception as e:
                    logger.error("[REHYDRATE] %s: load_project_to_active failed: %s", project_id, e)
                    return False
                if not project:
                    return False

            if project.get("status") in ("completed", "failed", "cancelled"):
                return False

            # Stamp run_id from the pending approval so the runtime emits events
            # under the correct run scope. The active_projects dict may have come
            # from a prior load that didn't carry run_id.
            run_id = approval_doc.get("run_id") or project.get("run_id")
            if run_id:
                project["run_id"] = run_id

            # Hydrate the in-memory pending_approvals dict before the FSM polls
            # it via wait_for_approval. Without this the runtime would poll for
            # an approval that exists in the message store but not in memory and
            # eventually time out (contract item: inbox-before-await).
            if approval_id not in self.approvals.pending_approvals:
                self.approvals.pending_approvals[approval_id] = approval_doc
                logger.info(
                    "[REHYDRATE] %s: hydrated approval %s (gate=%s, status=%s)",
                    project_id, approval_id, gate_node_id, approval_doc.get("status"),
                )

            # One-shot rehydration marker: tell the spawned FSM which approval
            # to observe on its first gate hit. Without this, a reject()/approve()
            # call landing AFTER spawn but BEFORE the FSM reaches the gate flips
            # status, and request_approval's PENDING/APPROVED-only dedup misses
            # the just-rejected case → FSM mints a duplicate, user sees a fresh
            # card, rejected edge isn't walked on the first click. The marker is
            # popped in _handle_approval_gate; loop-back gate revisits (gate →
            # rejected edge → upstream phase → gate) see no marker and fall
            # through to normal request_approval, so dedup on revisits is
            # unchanged.
            project["_rehydration_approval_id"] = approval_id

            try:
                workflow_def = await self._load_workflow_definition(project)
            except Exception as e:
                logger.warning("[REHYDRATE] %s: workflow_def load failed: %s", project_id, e)
                return False

            async def _run():
                try:
                    logger.info(
                        "[REHYDRATE] %s: FSM resumed at gate=%s (approval=%s)",
                        project_id, gate_node_id, approval_id,
                    )
                    await self.workflow_engine.execute(project_id, workflow_def, start_node_id=gate_node_id)
                except Exception as e:
                    import traceback
                    logger.error(
                        "[REHYDRATE] %s: FSM crashed: %s\n%s",
                        project_id, e, traceback.format_exc(),
                    )
                    raise

            task = create_task_with_context(_run())
            self.register_workflow_task(project_id, task)
            return True

    async def _recover_hanging_tool_calls(
        self,
        project_id: str,
        run_id: str,
        hanging: list,
        *,
        recorded_boot_id: str | None = None,
    ) -> bool:
        """Close interrupted (non-ask_human) tool calls with an error result.

        Returns False if any close failed — resuming on a journal we couldn't
        make consistent (a still-hanging call) would just re-park next scan.
        """
        recovery = build_recovery_metadata(recorded_boot_id)
        for pair in hanging:
            cd = (pair.get("call") or {}).get("data") or {}
            call_id = cd.get("tool_call_id")
            if not call_id:
                continue
            try:
                await self.message_store.append_tool_result(
                    project_id,
                    tool_call_id=call_id,
                    name=cd.get("name"),
                    result={
                        "status": "error",
                        "error": TOOL_INTERRUPTED_BEFORE_RESULT,
                        "outcome_unknown": True,
                        "recovery": recovery,
                    },
                    status="error",
                    run_id=run_id,
                    agent_id=cd.get("agent_id"),
                    task_id=cd.get("task_id"),
                )
            except Exception as e:
                logger.warning(
                    "[REHYDRATE] %s: failed to close hanging call %s (%s): %s",
                    project_id, call_id, cd.get("name"), e,
                )
                return False
        logger.info(
            "[REHYDRATE] project_id=%s run_id=%s recovery cause=%s boot=%s",
            project_id,
            run_id,
            recovery.get("cause"),
            recovery.get("current_boot_id"),
        )
        return True

    async def _maybe_heal_partial_terminal_persist(
        self, project_id: str, run: Dict[str, Any]
    ) -> None:
        """Self-heal when run doc is terminal but project document is not."""
        run_status = run.get("run_status")
        run_id = run.get("run_id")
        if run_status not in ("failed", "cancelled", "completed"):
            return
        if run_status == "cancelled":
            if await self.projects._durable_project_status(project_id, "initialized"):
                try:
                    await self.storage.update_run_status(run_id, "running")
                    logger.info(
                        "[REHYDRATE] project_id=%s run_id=%s revert artifact — "
                        "restored run_status=running (project initialized)",
                        project_id,
                        run_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "[REHYDRATE] project_id=%s run_id=%s revert run restore failed: %s",
                        project_id,
                        run_id,
                        exc,
                    )
                return
        if await self.projects._durable_project_status(project_id, run_status):
            proj = self.active_projects.get(project_id)
            if proj and proj.get("status") != run_status:
                proj["status"] = run_status
            return
        logger.warning(
            "[REHYDRATE] project_id=%s run_id=%s partial persist — run=%s, healing project doc",
            project_id,
            run_id,
            run_status,
        )
        persisted = False
        if project_id not in self.active_projects:
            try:
                await self.projects.load_project_to_active(project_id, self.agent_pool)
            except Exception as exc:
                logger.warning(
                    "[REHYDRATE] project_id=%s load_project_to_active before partial heal: %s",
                    project_id,
                    exc,
                )
        try:
            await self.update_project_status(project_id, run_status)
            persisted = await self.projects._durable_project_status(
                project_id, run_status
            )
        except Exception as exc:
            logger.error(
                "[REHYDRATE] project_id=%s failed to heal partial persist via update_project_status: %s",
                project_id,
                exc,
            )
        if not persisted:
            try:
                db_project = await self.storage.load_project(project_id)
                if db_project:
                    payload = dict(db_project)
                    payload["status"] = run_status
                    await self.storage.save_project(project_id, payload)
                    persisted = await self.projects._durable_project_status(
                        project_id, run_status
                    )
                    if persisted:
                        logger.warning(
                            "[REHYDRATE] project_id=%s run_id=%s partial persist healed via storage fallback",
                            project_id,
                            run_id,
                        )
            except Exception as exc:
                logger.error(
                    "[REHYDRATE] project_id=%s storage fallback for partial heal failed: %s",
                    project_id,
                    exc,
                )
        if not persisted:
            logger.error(
                "[REHYDRATE] project_id=%s run_id=%s partial persist heal failed",
                project_id,
                run_id,
            )

    async def _maybe_heal_run_when_project_terminal(
        self,
        project_id: str,
        *,
        run_id: str | None = None,
        project_doc: dict | None = None,
        write_heal: bool = False,
    ) -> str | None:
        """Detect terminal project doc; optionally heal split-brain run doc.

        write_heal=False on GET/rehydrate paths (AppFactory-292); True only after
        this PR's own partial-persist paths.
        """
        if project_doc is None:
            try:
                project_doc = await self.storage.load_project(project_id)
            except Exception as exc:
                logger.warning(
                    "[REHYDRATE] project_id=%s load_project for terminal check failed: %s",
                    project_id,
                    exc,
                )
                project_doc = None

        terminal_status: str | None = None
        if project_doc:
            status = project_doc.get("status")
            if status in ("failed", "cancelled", "completed"):
                terminal_status = status
        if not terminal_status:
            return None

        proj = self.active_projects.get(project_id)
        if proj and proj.get("status") != terminal_status:
            proj["status"] = terminal_status

        if not write_heal:
            return terminal_status

        try:
            run = await self.storage.get_active_run(project_id)
        except Exception as exc:
            logger.warning(
                "[REHYDRATE] project_id=%s active-run lookup for reverse heal failed: %s",
                project_id,
                exc,
            )
            return terminal_status
        if not run:
            return terminal_status
        active_run_id = run.get("run_id") or run_id
        run_status = run.get("run_status")
        if not active_run_id or run_status in ("failed", "cancelled", "completed"):
            return terminal_status
        try:
            await self.storage.update_run_status(active_run_id, terminal_status)
            logger.warning(
                "[REHYDRATE] project_id=%s run_id=%s reverse partial persist — "
                "project=%s, healing run doc",
                project_id,
                active_run_id,
                terminal_status,
            )
        except Exception as exc:
            logger.error(
                "[REHYDRATE] project_id=%s run_id=%s reverse partial heal failed: %s",
                project_id,
                active_run_id,
                exc,
            )
        return terminal_status

    async def _fail_project_outcome_unknown(
        self,
        project_id: str,
        run_id: str,
        attempt: Dict[str, Any],
        *,
        emit_event: bool = True,
    ) -> None:
        """Terminal fail when a side-effecting tool's outcome cannot be determined."""
        if await self.projects._durable_project_status(project_id, "failed"):
            proj = self.active_projects.get(project_id)
            if proj and proj.get("status") != "failed":
                proj["status"] = "failed"
            logger.info(
                "[REHYDRATE] project_id=%s project doc already failed — skip outcome_unknown fail",
                project_id,
            )
            await self._maybe_heal_run_when_project_terminal(
                project_id, run_id=run_id, write_heal=True
            )
            return
        persisted = False
        if project_id not in self.active_projects:
            try:
                await self.projects.load_project_to_active(project_id, self.agent_pool)
            except Exception as exc:
                logger.warning(
                    "[REHYDRATE] project_id=%s load_project_to_active before outcome_unknown fail: %s",
                    project_id,
                    exc,
                )
        try:
            await self.update_project_status(project_id, "failed")
            persisted = await self.projects._durable_project_status(project_id, "failed")
        except Exception as exc:
            logger.error(
                "[REHYDRATE] project_id=%s failed to persist outcome_unknown status: %s",
                project_id,
                exc,
            )
        if not persisted and run_id:
            try:
                persisted = await self.projects._persist_terminal_status(
                    project_id,
                    "failed",
                    run_id=run_id,
                )
                if persisted:
                    logger.warning(
                        "[REHYDRATE] project_id=%s run_id=%s outcome_unknown persisted via storage fallback",
                        project_id,
                        run_id,
                    )
                else:
                    logger.error(
                        "[REHYDRATE] project_id=%s run_id=%s outcome_unknown partial persist: "
                        "terminal transition failed",
                        project_id,
                        run_id,
                    )
            except Exception as exc:
                logger.error(
                    "[REHYDRATE] project_id=%s storage fallback for outcome_unknown failed: %s",
                    project_id,
                    exc,
                )
        if not persisted:
            if run_id:
                try:
                    await self.storage.set_run_resume_blocked(
                        run_id, EXTERNAL_TOOL_OUTCOME_UNKNOWN
                    )
                    logger.warning(
                        "[REHYDRATE] project_id=%s run_id=%s outcome_unknown persist "
                        "failed — resume blocked on run doc",
                        project_id,
                        run_id,
                    )
                except Exception as exc:
                    logger.error(
                        "[REHYDRATE] project_id=%s set resume_blocked err=%s",
                        project_id,
                        exc,
                    )
            logger.error(
                "[REHYDRATE] project_id=%s run_id=%s outcome_unknown terminal state not persisted",
                project_id,
                run_id,
            )
            return
        if run_id:
            try:
                await self.storage.clear_run_resume_blocked(run_id)
            except Exception as exc:
                logger.warning(
                    "[REHYDRATE] project_id=%s clear resume_blocked err=%s",
                    project_id,
                    exc,
                )
        await self._maybe_heal_run_when_project_terminal(
            project_id, run_id=run_id, write_heal=True
        )
        if self.message_store is not None and persisted:
            try:
                await self.message_store.append_system_message(
                    project_id,
                    "error",
                    EXTERNAL_TOOL_OUTCOME_UNKNOWN,
                    run_id=run_id,
                )
            except Exception as exc:
                logger.warning(
                    "[REHYDRATE] project_id=%s append outcome_unknown chat message err=%s",
                    project_id,
                    exc,
                )
        if not emit_event:
            return
        try:
            await self.event_emitter.emit(
                EventSchema.PROJECT_FAILED,
                run_id,
                {
                    "project_id": project_id,
                    "error": EXTERNAL_TOOL_OUTCOME_UNKNOWN,
                    "task_id": attempt.get("task_id"),
                },
            )
        except Exception as exc:
            logger.warning(
                "[REHYDRATE] project_id=%s emit project_failed outcome_unknown failed: %s",
                project_id,
                exc,
            )

    async def _maybe_resume_interrupted_run(
        self, project_id: str, *, project_doc: dict | None = None
    ) -> bool:
        """Resume a run interrupted mid-phase, from the tool journal.

        Called under the project lock from ensure_workflow_running when no
        gate is parked. A hanging ask_human call parks the project (its human
        answer is the continuation signal); any other hanging call is treated
        as outcome_unknown and fails the project without resume. Closed pairs
        respawn the FSM at the interrupted node with a resume marker the phase
        adopts. Anything unmappable degrades to doing nothing.
        """
        try:
            run = await self.storage.get_active_run(project_id)
        except Exception as e:
            logger.warning("[REHYDRATE] %s: active-run lookup failed: %s", project_id, e)
            return False
        if not run:
            return False
        durable_terminal = await self._maybe_heal_run_when_project_terminal(
            project_id, run_id=run.get("run_id"), project_doc=project_doc
        )
        if durable_terminal:
            if (
                durable_terminal == "failed"
                and run.get("run_status") == "running"
            ):
                await self._maybe_heal_run_when_project_terminal(
                    project_id,
                    run_id=run.get("run_id"),
                    project_doc=project_doc,
                    write_heal=True,
                )
            return False
        try:
            run = await self.storage.get_active_run(project_id)
        except Exception as e:
            logger.warning("[REHYDRATE] %s: active-run re-fetch failed: %s", project_id, e)
            return False
        if not run:
            return False
        if run.get("run_status") != "running":
            await self._maybe_heal_partial_terminal_persist(project_id, run)
            return False
        run_id = run.get("run_id")

        completed = set(run.get("completed_attempts") or [])
        try:
            attempt = await find_interrupted_attempt(
                self.message_store,
                project_id,
                run_id,
                completed_task_ids=completed,
            )
        except Exception as e:
            logger.warning("[REHYDRATE] %s: journal scan failed: %s", project_id, e)
            return False
        if not attempt:
            # A completed marker makes find_interrupted_attempt return None. If
            # that finished attempt is a gated phase whose approval row never
            # landed (crash in the marker→request_approval window), advance it to
            # its gate instead of stranding the run — resuming the phase itself
            # would re-run its side effects.
            return await self._maybe_create_gate_for_completed_phase(
                project_id, run_id, completed
            )

        if attempt["hanging"]:
            hanging_names = [
                ((p.get("call") or {}).get("data") or {}).get("name")
                for p in attempt["hanging"]
            ]
            # ask_human is the ONLY tool whose hanging call has an external
            # resume signal that survives a restart: its answer route
            # (human_input.py) writes the missing result and re-triggers this
            # resume. Park and wait for it.
            if any(name == "ask_human" for name in hanging_names):
                logger.info(
                    "[REHYDRATE] %s: parked on hanging call(s) %s (task %s) — waiting for result",
                    project_id, hanging_names, attempt["task_id"],
                )
                return False

            # Any OTHER hanging call is an ordinary tool the process died inside
            # of (hard-kill between the pre-execution journal write and the
            # result). Close with outcome_unknown and fail the project — do not
            # resume, to avoid replaying side-effecting MCP writes.
            if not await self._recover_hanging_tool_calls(
                project_id,
                run_id,
                attempt["hanging"],
                recorded_boot_id=run.get("backend_boot_id"),
            ):
                return False
            logger.warning(
                "[REHYDRATE] %s: recovered %d interrupted tool call(s) %s (task %s) — "
                "outcome unknown, not resuming to avoid duplicate side effects",
                project_id,
                len(attempt["hanging"]),
                hanging_names,
                attempt["task_id"],
            )
            await self._fail_project_outcome_unknown(project_id, run_id, attempt)
            return False

        # Journal marker survives even when terminal persist failed on a prior
        # GET — without this guard a second request sees hanging=[] and resumes.
        if attempt_has_outcome_unknown(attempt):
            logger.warning(
                "[REHYDRATE] %s: attempt %s has outcome_unknown closed tool(s) — "
                "not resuming to avoid duplicate side effects",
                project_id,
                attempt["task_id"],
            )
            await self._fail_project_outcome_unknown(
                project_id, run_id, attempt, emit_event=False
            )
            return False

        node_id = attempt.get("workflow_node_id")
        if not node_id:
            logger.warning(
                "[REHYDRATE] %s: interrupted task %s carries no workflow_node_id "
                "(records predate the node-id field) — cannot map to a node, not resuming",
                project_id, attempt["task_id"],
            )
            return False

        project = self.active_projects.get(project_id)
        if not project:
            try:
                project = await self.projects.load_project_to_active(project_id, self.agent_pool)
            except Exception as e:
                logger.error("[REHYDRATE] %s: load_project_to_active failed: %s", project_id, e)
                return False
            if not project:
                return False
        if project.get("status") in ("completed", "failed", "cancelled"):
            return False

        # Continued journal writes must land in the interrupted run, not a
        # fresh one. The runner takes its run_id from shared_context (not the
        # project dict), so both get stamped.
        project["run_id"] = run_id
        sc = project.get("shared_context")
        if sc is not None:
            try:
                sc.run_id = run_id
            except Exception:
                pass

        try:
            workflow_def = await self._load_workflow_definition(project)
        except Exception as e:
            logger.warning("[REHYDRATE] %s: workflow_def load failed: %s", project_id, e)
            return False
        node_ids = {n.get("id") for n in (workflow_def or {}).get("nodes", [])}
        if node_id not in node_ids:
            logger.warning(
                "[REHYDRATE] %s: node %s not in workflow — not resuming",
                project_id, node_id,
            )
            return False

        project["_resume_attempt"] = {
            "task_id": attempt["task_id"],
            "agent_id": attempt["agent_id"],
            "node_id": node_id,
        }

        async def _run():
            try:
                logger.info(
                    "[REHYDRATE] %s: FSM resumed mid-phase at node=%s (task=%s, %d closed pairs)",
                    project_id, node_id, attempt["task_id"], len(attempt["pairs"]),
                )
                await self.workflow_engine.execute(project_id, workflow_def, start_node_id=node_id)
            except Exception as e:
                import traceback
                logger.error(
                    "[REHYDRATE] %s: resumed FSM crashed: %s\n%s",
                    project_id, e, traceback.format_exc(),
                )
                raise

        task = create_task_with_context(_run())
        self.register_workflow_task(project_id, task)
        return True

    async def _maybe_create_gate_for_completed_phase(
        self, project_id: str, run_id: str, completed_task_ids: set
    ) -> bool:
        """Re-enter the FSM at the gate of a completed-but-ungated gated phase.

        A gated phase writes its completion marker one node before its approval
        row is created; a crash in that window leaves the phase done, no gate, no
        runtime — the project silently stalls (no card, nothing advances). Spawn
        the FSM at the gate node so it creates the approval card, WITHOUT re-running
        the phase. Idempotent: bails if the gate already has an approval row.
        A completed map node is resumed at the node after it, gate or not.
        """
        node_info = await last_completed_node_attempt(
            self.message_store, project_id, run_id, completed_task_ids
        )
        if not node_info:
            return False
        phase_node_id = node_info.get("workflow_node_id")
        if not phase_node_id:
            return False

        project = self.active_projects.get(project_id)
        if not project:
            try:
                project = await self.projects.load_project_to_active(project_id, self.agent_pool)
            except Exception as e:
                logger.error("[REHYDRATE] %s: load_project_to_active failed: %s", project_id, e)
                return False
            if not project:
                return False
        if project.get("status") in ("completed", "failed", "cancelled"):
            return False

        try:
            workflow_def = await self._load_workflow_definition(project)
        except Exception as e:
            logger.warning("[REHYDRATE] %s: workflow_def load failed: %s", project_id, e)
            return False

        # completed_attempts holds gated-phase and map ids, but the workflow may
        # have been edited between runs — re-check that this phase still leads into
        # an approval_gate before creating one. Static methods on the class (not
        # self.workflow_engine, which tests mock) so the real edge logic runs.
        completed_node = next(
            (n for n in workflow_def.get("nodes", []) if n.get("id") == phase_node_id),
            {},
        )
        if completed_node.get("type") != "map" and not (
            WorkflowEngine._phase_is_followed_by_approval_gate(phase_node_id, workflow_def)
        ):
            return False
        gate_node_id = WorkflowEngine._find_default_edge(
            phase_node_id, workflow_def.get("edges", [])
        )
        if not gate_node_id:
            return False

        # Don't re-create a gate that already exists in ANY state. request_approval
        # only dedups PENDING/APPROVED via its in-memory map (empty after a restart)
        # and its DB check is pending-only, so an already-approved/rejected gate
        # would otherwise be duplicated here.
        try:
            approvals = await self.message_store.get_messages(
                project_id, run_id=run_id, only_types=["approval"],
            )
        except Exception as e:
            logger.warning("[REHYDRATE] %s: approval lookup failed: %s", project_id, e)
            return False
        if any(
            (a.get("data") or {}).get("gate_node_id") == gate_node_id
            for a in (approvals or [])
        ):
            return False

        # Continued gate/journal writes must land in this run, not a fresh one
        # (mirror the mid-phase resume path above).
        project["run_id"] = run_id
        sc = project.get("shared_context")
        if sc is not None:
            try:
                sc.run_id = run_id
            except Exception:
                pass

        async def _run():
            try:
                logger.info(
                    "[REHYDRATE] %s: node %s completed before the restart — "
                    "re-entering at %s",
                    project_id, phase_node_id, gate_node_id,
                )
                await self.workflow_engine.execute(
                    project_id, workflow_def, start_node_id=gate_node_id
                )
            except Exception as e:
                import traceback
                logger.error(
                    "[REHYDRATE] %s: gate re-entry FSM crashed: %s\n%s",
                    project_id, e, traceback.format_exc(),
                )
                raise

        task = create_task_with_context(_run())
        self.register_workflow_task(project_id, task)
        return True

    async def _maybe_resume_interrupted_a2a(self, project_id: str) -> bool:
        """Resume a run interrupted while an A2A node had a durable task cursor.

        Third durable-cursor fallback in ensure_workflow_running, tried after
        the gate and the tool journal both find nothing. An open a2a_task_state
        row means _a2a_submit_and_poll persisted a correlation id before the
        process died — there's nothing to re-execute ("adoption over re-run",
        same principle as ADR-0009). Two open statuses, two resume shapes
        (AppFactory-280 finding #2):
          - "in_flight" (task_id confirmed): resume polling the SAME task_id.
          - "pending_submit" (task_id never confirmed): resume by re-submitting
            with the SAME message_id instead of minting a new one.
        Either way _run_a2a_agent_node's `_resume_a2a` marker handles the branch,
        so this method only needs to pick which field the marker carries.
        """
        try:
            cursor = await self.storage.get_open_a2a_task_state(project_id)
        except Exception as e:
            logger.warning("[A2A_RECOVER] %s: a2a_task_state lookup failed: %s", project_id, e)
            return False
        if not cursor:
            return False

        node_id = cursor.get("node_id")
        run_id = cursor.get("run_id")
        task_id = cursor.get("task_id")
        message_id = cursor.get("message_id")
        if not node_id or not run_id or not (task_id or message_id):
            logger.warning(
                "[A2A_RECOVER] %s: open a2a_task_state cursor missing node_id/run_id "
                "and both task_id/message_id (%s) — not resuming",
                project_id, cursor,
            )
            return False

        project = self.active_projects.get(project_id)
        if not project:
            try:
                project = await self.projects.load_project_to_active(project_id, self.agent_pool)
            except Exception as e:
                logger.error("[A2A_RECOVER] %s: load_project_to_active failed: %s", project_id, e)
                return False
            if not project:
                return False
        if project.get("status") in ("completed", "failed", "cancelled"):
            return False

        # Continued journal/event writes must land in the interrupted run, not a
        # fresh one (mirrors the mid-phase resume path above).
        project["run_id"] = run_id
        sc = project.get("shared_context")
        if sc is not None:
            try:
                sc.run_id = run_id
            except Exception:
                pass

        try:
            workflow_def = await self._load_workflow_definition(project)
        except Exception as e:
            logger.warning("[A2A_RECOVER] %s: workflow_def load failed: %s", project_id, e)
            return False
        node_ids = {n.get("id") for n in (workflow_def or {}).get("nodes", [])}
        if node_id not in node_ids:
            logger.warning(
                "[A2A_RECOVER] %s: a2a node %s not in workflow — not resuming",
                project_id, node_id,
            )
            return False
        
        # One-shot marker: _run_a2a_agent_node pops this on its first hit. Shape
        # mirrors which field the cursor actually has — task_id means poll the
        # existing task; message_id-only means re-submit reusing that id (the
        # node's own message-building runs again in that case, unlike the
        # task_id case which skips straight to polling). A third shape (AppFactory-281):
        # a human answered an input_required question — mark_a2a_task_human_answered
        # already moved the cursor back to "in_flight" and stashed the answer text,
        # so get_open_a2a_task_state above found it via the same query as any other
        # in_flight cursor, zero query changes needed. The answer rides the marker so
        # _run_a2a_agent_node sends it (continuing the SAME task_id) before resuming
        # the normal poll loop.
        #
        # A fourth shape (AppFactory-281 P1 review fix, bug #3): answer_dispatch_
        # started_at set means the PREVIOUS attempt crashed strictly between
        # "about to send the answer" (mark_a2a_task_answer_dispatching) and
        # "confirmed sent" (mark_a2a_task_answer_delivered) — genuinely ambiguous
        # whether the adapter ever received it. Blindly resending here is exactly
        # the bug the reviewer found (the answer going out twice for one human
        # answer). reconcile_before_resend tells _run_a2a_agent_node to check
        # tasks/get FIRST and only (re)send if the task is still at
        # input_required, instead of assuming the worst or the best.
        #
        # answered_question_message_id (P1 bug #3 follow-up): the adapter's own
        # message_id for the SPECIFIC question this pending_human_answer targets
        # (kept unchanged on the cursor by mark_a2a_task_human_answered — "audit
        # trail" field). Lets the reconciliation tell "still the same unanswered
        # question" apart from "task is still input_required but already asking
        # a NEW question after accepting this answer" — state alone can't, since
        # one task may ask several questions in a row.
        pending_human_answer = cursor.get("pending_human_answer")
        answer_dispatch_started_at = cursor.get("answer_dispatch_started_at")
        answer_message_id = cursor.get("answer_message_id")
        answered_question_message_id = cursor.get("question_message_id")
        project["_resume_a2a"] = (
            {
                "node_id": node_id, "task_id": task_id, "human_answer": pending_human_answer,
                "answer_message_id": answer_message_id, "reconcile_before_resend": True,
                "answered_question_message_id": answered_question_message_id,
            }
            if task_id and pending_human_answer and answer_dispatch_started_at
            else {
                "node_id": node_id, "task_id": task_id, "human_answer": pending_human_answer,
                "answer_message_id": answer_message_id,
            }
            if task_id and pending_human_answer
            else {"node_id": node_id, "task_id": task_id} if task_id
            else {"node_id": node_id, "message_id": message_id}
        )

        async def _run():
            try:
                logger.info(
                    "[A2A_RECOVER] %s: FSM resumed at a2a node=%s (%s) after restart",
                    project_id, node_id,
                    f"task={task_id}" if task_id else f"message_id={message_id}",
                )
                await self.workflow_engine.execute(project_id, workflow_def, start_node_id=node_id)
            except Exception as e:
                import traceback
                logger.error(
                    "[A2A_RECOVER] %s: a2a-resumed FSM crashed: %s\n%s",
                    project_id, e, traceback.format_exc(),
                )
                raise

        task = create_task_with_context(_run())
        self.register_workflow_task(project_id, task)
        return True

    async def reconcile_open_a2a_cursors(self) -> Dict[str, int]:
        """Startup-wide sweep: resume every project with an open a2a_task_state
        cursor, not just the one a caller already happens to ask about.

        AppFactory-280 Issue 1: ensure_workflow_running (and its a2a arm above)
        only ever runs for a project_id an HTTP handler already knows —
        page-load, approve, revert. A project nobody touches after a restart
        stayed running forever even if the external task had long finished.
        Called once from api/main.py's startup lifespan; deliberately reuses
        ensure_workflow_running per project instead of duplicating its
        locking/idempotency/terminal-status logic.
        """
        summary = {"resumed": 0, "skipped": 0, "failed": 0}
        try:
            project_ids = await self.storage.list_open_a2a_project_ids()
        except Exception as e:
            logger.error("[A2A_RECOVER] action=startup_sweep status=list_failed error=%s", e)
            return summary

        for project_id in project_ids:
            try:
                spawned = await self.ensure_workflow_running(project_id)
            except Exception as e:
                summary["failed"] += 1
                logger.exception(
                    "[A2A_RECOVER] project_id=%s action=startup_sweep status=failed error=%s",
                    project_id, e,
                )
                continue
            if spawned:
                summary["resumed"] += 1
                logger.info(
                    "[A2A_RECOVER] project_id=%s action=startup_sweep status=resumed", project_id,
                )
            else:
                summary["skipped"] += 1
                logger.info(
                    "[A2A_RECOVER] project_id=%s action=startup_sweep status=skipped", project_id,
                )

        logger.info(
            "[A2A_RECOVER] action=startup_sweep_summary status=ok resumed=%d skipped=%d failed=%d",
            summary["resumed"], summary["skipped"], summary["failed"],
        )
        return summary

    async def reconcile_pending_a2a_cancellations(self) -> Dict[str, int]:
        """Retry only cancellation requests left durable by a local failure.

        This deliberately does not resume workflows or send messages. A project
        may already be cancelled or failed; the remaining responsibility is to
        stop its confirmed remote task and close the cursor once confirmed.
        """
        summary = {"closed": 0, "pending": 0, "failed": 0}
        try:
            cursors = await self.storage.list_a2a_tasks_pending_cancellation()
        except Exception as exc:
            logger.error(
                "[A2A_RECOVER] action=cancellation_sweep status=list_failed error=%s",
                exc,
            )
            return summary

        for cursor in cursors:
            project_id = cursor.get("project_id")
            try:
                cancelled = await self._cancel_a2a_cursor(cursor, project_id=project_id)
            except Exception:
                summary["failed"] += 1
                logger.exception(
                    "[A2A_RECOVER] project_id=%s action=cancellation_sweep status=failed",
                    project_id,
                )
            else:
                summary["closed" if cancelled else "pending"] += 1

        logger.info(
            "[A2A_RECOVER] action=cancellation_sweep status=ok closed=%d pending=%d failed=%d",
            summary["closed"],
            summary["pending"],
            summary["failed"],
        )
        return summary

    async def run_workflow(self, project_id: str):
        """Execute complete workflow for a project."""
        project = self.active_projects.get(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        if project.get("status") == "failed":
            await self.event_emitter.emit(EventSchema.PROJECT_FAILED, project.get("run_id"), {
                "project_id": project_id,
                "error": "Project cannot start: container not ready"
            })
            return {"status": "failed", "reason": "container_not_ready"}
        
        shared_context = project["shared_context"]

        async with run_trace_scope(
            self.storage, self.tracer, project_id, project.get("run_id")
        ):
            with self.tracer.start_span(
                "project.execution",
                attributes={
                    "project.id": project_id,
                    "user.prompt": project.get("user_prompt", ""),
                    "project.title": project.get("title", ""),
                    "approval.mode": project.get("approval_mode", "human"),
                }
            ) as root_span:
                try:
                    if root_span:
                        project["root_span"] = root_span
                except Exception as e:
                    self.tracer.set_error(root_span, e)
                    raise

                # Create initial snapshots
                await self._create_initial_snapshots(project_id, project, shared_context)

                # Load workflow definition from MongoDB (or fallback)
                workflow_def = await self._load_workflow_definition(project)

                # Execute DAG via workflow engine
                result = await self.workflow_engine.execute(project_id, workflow_def)

                if result and result.get("status") == "completed":
                    self.tracer.set_success(root_span)
                return result
    
    async def _create_initial_snapshots(self, project_id: str, project: Dict, shared_context):
        """Create initial snapshots for a new workflow run."""
        user_prompt = project.get("user_prompt", "")
        if user_prompt:
            # NOTE: Initial user message is written by SharedContext.initialize()
            # Do NOT call add_conversation_message here - that would be a duplicate write
            try:
                existing_user_snap = await self.storage.get_latest_user_snapshot(project_id)
                if not existing_user_snap:
                    idx = await self.message_store.get_latest_sequence(project_id) if self.message_store else 0
                    await self.snapshot_manager.create_snapshot(
                        project_id, shared_context, snap_type="user_message",
                        label="User message", phase="initialization",
                        meta={"tags": ["user_action", "user_message"], "conversation_index": idx, "input_prefill": user_prompt},
                    )
            except Exception as e:
                logger.error("[INIT] user_message snapshot creation failed: %s", str(e))
        
        try:
            await self.snapshot_manager.create_snapshot(
                project_id, shared_context, snap_type="project_started",
                label="project started", phase="initialization",
                meta={"tags": ["system_checkpoint"]},
            )
        except Exception:
            pass
    
    @staticmethod
    def _project_tenant_id(project: Dict) -> str:
        tenant_id = str(project.get("tenant_id") or "").strip()
        if tenant_id:
            return tenant_id
        sc = project.get("shared_context")
        if isinstance(sc, SharedContext):
            return str(sc.tenant_id or "").strip()
        return ""

    async def _resolve_workflow_for_project(
        self,
        project: Dict,
        workflow_id: str,
    ) -> Optional[dict]:
        return await resolve_project_workflow(
            self.storage, workflow_id, self._project_tenant_id(project)
        )

    async def _resolve_agent_config(
        self,
        wire_name: str,
        tenant_id: Optional[str],
    ) -> Optional[dict]:
        tenant = str(tenant_id or "").strip()
        if not tenant:
            return None
        resolved = await self.storage.resolve_agent_configuration(tenant, wire_name)
        if resolved is not None:
            return resolved.document
        return None

    async def _load_workflow_definition(self, project: Dict) -> dict:
        """Load workflow definition from MongoDB, seed if empty, or use fallback."""
        cached = project.get("_checkpoint_binding")
        if cached and cached.get("run_id") == project.get("run_id"):
            return cached["workflow_def"]
        run = await self.storage.get_active_run(project["project_id"]) if project.get("project_id") else None
        if run and run.get("restored_from") and run.get("run_id") == project.get("run_id"):
            from storage.checkpoint_store import CheckpointStore
            binding = await CheckpointStore(self.storage, self.message_store).get_binding(project["run_id"])
            if binding:
                project["_checkpoint_binding"] = binding
                return binding["workflow_def"]
        workflow_id = project.get("workflow_id", "default_build")
        try:
            wf = await self._resolve_workflow_for_project(project, workflow_id)
            if wf:
                return wf
            # Try seeding from JSON on first access
            await seed_workflows(self.storage)
            wf = await self._resolve_workflow_for_project(project, workflow_id)
            if wf:
                return wf
        except Exception as e:
            logger.warning("[WORKFLOW] Failed to load workflow '%s' from DB: %s", workflow_id, e)
        
        logger.info("[WORKFLOW] Using hardcoded fallback for '%s'", workflow_id)
        return self._get_fallback_build_workflow()

    @staticmethod
    def _get_fallback_build_workflow() -> dict:
        """Hardcoded fallback — identical to the default_build seed."""
        return {
            "_id": "default_build",
            "name": "Build Workflow (fallback)",
            "nodes": [
                {"id": "start", "type": "start"},
                {"id": "req_gather", "type": "phase", "task_type": "requirements_gathering", "description": "Gather project requirements from user", "agent_selection": "auction", "phase_label": "requirements"},
                {"id": "human_expert", "type": "phase", "task_type": "answer_questions", "description": "Answer requirements questions automatically", "agent_selection": "direct", "agent_type": "human_expert", "phase_label": "requirements"},
                {"id": "req_finalizer", "type": "phase", "task_type": "finalize_requirements", "description": "Create final requirements document", "agent_selection": "direct", "agent_type": "requirements_finalizer", "phase_label": "requirements"},
                {"id": "gate_req", "type": "approval_gate", "label": "Review requirements"},
                {"id": "planning", "type": "phase", "task_type": "planning", "description": "Create hierarchical task breakdown", "agent_selection": "auction", "phase_label": "planning"},
                {"id": "gate_plan", "type": "approval_gate", "label": "Review plan", "interaction_schema": {"type": "plan_review", "version": 1, "decision_field": "decision", "fields": [{"name": "remove_task_ids", "type": "task_multi_select", "label": "Tasks to remove", "source": "context_snapshot.plan.tasks"}, {"name": "feedback_text", "type": "textarea", "label": "Requested changes"}]}},
                {"id": "execution", "type": "execution", "phase_label": "execution"},
                {"id": "gate_output", "type": "approval_gate", "label": "Review output"},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"from": "start", "to": "req_gather"},
                {"from": "req_gather", "to": "human_expert"},
                {"from": "human_expert", "to": "req_finalizer"},
                {"from": "req_finalizer", "to": "gate_req"},
                {"from": "gate_req", "to": "planning", "condition": "approved"},
                {"from": "gate_req", "to": "req_gather", "condition": "rejected"},
                {"from": "planning", "to": "gate_plan"},
                {"from": "gate_plan", "to": "execution", "condition": "approved"},
                {"from": "execution", "to": "gate_output"},
                {"from": "gate_output", "to": "end", "condition": "approved"},
            ],
            "is_default": True,
        }

    async def _sync_container_to_repo(self, project_id: str):
        """Sync container worktree back to host repo."""
        try:
            if self.container_manager:
                with self.tracer.start_span("container.apply_environment", attributes={"project.id": project_id}) as span:
                    try:
                        res = await self.container_manager.apply_environment(project_id)
                        logger.info(f"[SYNC] apply_environment for {project_id}: {res}")
                        
                        # Also run checkout to ensure host repo has latest from container-use branch
                        checkout_res = await self.container_manager.checkout_environment(project_id)
                        logger.info(f"[SYNC] checkout_environment for {project_id}: {checkout_res}")
                        
                        self.tracer.set_success(span)
                    except Exception as e:
                        logger.warning(f"[SYNC] apply_environment failed: {e}")
                        self.tracer.set_error(span, e)
        except Exception as e:
            logger.warning(f"[SYNC] _sync_container_to_repo failed: {e}")
    
    async def _prepare_final_artifacts(self, project_id: str, project: Dict, shared_context) -> List[Dict]:
        """Prepare and dedupe final artifacts."""
        try:
            raw_artifacts: List[Dict[str, Any]] = []

            # Seed from ArtifactStore (source of truth) instead of shared_context cache.
            if self.artifact_store:
                try:
                    raw_artifacts = await self.artifact_store.get_all_files(project_id)
                except Exception as e:
                    logger.warning(f"[HARVEST] Failed to load artifacts from ArtifactStore: {e}")

            # Dedupe by path
            latest_by_path: Dict[str, Any] = {}
            for art in raw_artifacts:
                p = (art.get("path") or "").strip()
                if not p:
                    continue
                prev = latest_by_path.get(p)
                if not prev:
                    latest_by_path[p] = art
                else:
                    prev_len = len(prev.get("content") or "")
                    cur_len = len(art.get("content") or "")
                    if cur_len > prev_len:
                        latest_by_path[p] = art
            
            deduped = list(latest_by_path.values())
            
            # ALWAYS harvest from repo - it's the source of truth for current file state
            # After refinement, shared_context.artifacts may be stale while container has updated files
            if self.container_manager:
                deduped = await self._harvest_from_repo(project_id, deduped)
            
            return deduped[:50]
        except Exception:
            if self.artifact_store:
                try:
                    return await self.artifact_store.get_all_files(project_id)
                except Exception:
                    pass
            return []
    
    async def _harvest_from_repo(self, project_id: str, deduped: List[Dict]) -> List[Dict]:
        """Harvest artifacts directly from container (source of truth).
        
        We read from container instead of host repo because:
        - Host repo may have git merge conflicts (our snapshots vs container-use branch)
        - Container always has the latest file state
        """
        try:
            # First try to read directly from container (preferred - no conflict issues)
            container_arts = await self._harvest_from_container(project_id)
            if container_arts:
                # Merge container files with existing artifacts, preferring container
                merged = {a.get("path"): a for a in deduped}
                for ca in container_arts:
                    pth = ca.get("path")
                    if pth:
                        merged[pth] = ca  # Container is source of truth
                return list(merged.values())
        except Exception:
            pass
        
        # Fallback to host repo if container unavailable
        # Read from container-use branch (source of truth), not main
        try:
            status = await self.container_manager.get_container_status(project_id)
            repo_path = status.get("repo_path")
            logger.info(f"[HARVEST] get_container_status returned repo_path={repo_path}")
        except Exception as e:
            logger.warning(f"[HARVEST] get_container_status failed: {e}")
            return deduped
        
        if not repo_path:
            logger.warning(f"[HARVEST] No repo_path for {project_id}")
            return deduped
        
        try:
            import subprocess
            base = Path(repo_path)
            
            # Find and checkout the container-use branch (has latest files)
            if base.exists() and (base / ".git").exists():
                # List container-use branches
                result = subprocess.run(
                    ["git", "branch", "-a", "--list", "*container-use/*"],
                    cwd=str(base), capture_output=True, text=True
                )
                cu_branches = [b.strip().lstrip("* ") for b in result.stdout.strip().splitlines() if b.strip()]
                logger.info(f"[HARVEST] Found container-use branches: {cu_branches}")
                
                if cu_branches:
                    # Checkout the latest container-use branch to read files
                    latest_branch = cu_branches[-1]
                    checkout_result = subprocess.run(["git", "checkout", latest_branch], cwd=str(base), capture_output=True, text=True)
                    logger.info(f"[HARVEST] Checkout {latest_branch}: exit={checkout_result.returncode}")
            
            from config.artifacts import should_include_file, has_conflict_markers, MAX_FILES_HARD_LIMIT
            
            host_arts = []
            if base.exists():
                for p in base.rglob("*"):
                    try:
                        if p.is_file():
                            rel = str(p.relative_to(base)).replace("\\", "/")
                            if not should_include_file(rel):
                                continue
                            try:
                                txt = p.read_text(encoding="utf-8", errors="replace")
                            except Exception:
                                txt = p.read_text(errors="replace")
                            # Log what we found
                            if rel == "index.html":
                                preview = txt[:100].replace('\n', ' ')
                                logger.info(f"[HARVEST] Read index.html: {preview}...")
                            if has_conflict_markers(txt):
                                continue
                            host_arts.append({"type": "code_file", "path": rel, "content": txt, "metadata": {"source": "repo"}})
                            if len(host_arts) >= MAX_FILES_HARD_LIMIT:
                                break
                    except Exception:
                        continue
            
            # Merge - repo is SOURCE OF TRUTH, always prefer repo content
            merged = {a.get("path"): a for a in deduped}
            for ha in host_arts:
                pth = ha.get("path")
                if not pth:
                    continue
                # Always prefer repo content (it has the latest from container-use branch)
                if ha.get("content"):
                    merged[pth] = ha
            logger.info(f"[HARVEST] Final artifact count: {len(merged)}")
            return list(merged.values())
        except Exception:
            return deduped
    
    async def _harvest_from_container(self, project_id: str) -> List[Dict]:
        """Read files directly from container filesystem."""
        try:
            # List files in container workdir
            files_result = await self.container_manager.list_files_in_container(project_id, ".")
            if not files_result:
                return []
            
            exts = (".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".md", ".json", ".txt")
            container_arts = []
            
            for f in files_result:
                # f might be a string path or a dict with path info
                path = f if isinstance(f, str) else f.get("path", f.get("name", ""))
                if not path:
                    continue
                
                # Skip hidden dirs and common non-source dirs
                if path.startswith(".git/") or "/node_modules/" in path or path.startswith("node_modules/"):
                    continue
                
                # Check extension
                ext = Path(path).suffix.lower()
                if ext not in exts:
                    continue
                
                # Read file content from container
                try:
                    content = await self.container_manager.read_file_from_container(project_id, path)
                    if content:
                        container_arts.append({
                            "type": "code_file",
                            "path": path,
                            "content": content,
                            "metadata": {"source": "container"}
                        })
                        if len(container_arts) >= 50:
                            break
                except Exception:
                    continue
            
            return container_arts
        except Exception:
            return []
    
    # === Approval Delegates ===
    async def approve(
        self,
        approval_id: str,
        feedback: Optional[str] = None,
        interaction_response: Optional[Dict] = None,
        expected_data: Optional[Dict] = None,
    ) -> bool:
        """Approve a pending gate.

        The workflow runtime parked at the corresponding gate (rehydrated by
        ensure_workflow_running on page-load or pre-approve) wakes when the
        approval status flips and follows the edge defined in the workflow
        definition. No Python-side dispatch table runs here.
        """
        project_id = None
        approval = self.approvals.get_pending(approval_id)
        if approval:
            project_id = approval.get("project_id")

        project_state = self.active_projects.get(project_id) if project_id else None

        return await self.approvals.approve(
            approval_id,
            feedback,
            project_state,
            interaction_response,
            expected_data,
        )

    async def reject(
        self,
        approval_id: str,
        reason: str,
        interaction_response: Optional[Dict] = None,
        expected_data: Optional[Dict] = None,
    ) -> bool:
        return await self.approvals.reject(
            approval_id,
            reason,
            interaction_response,
            expected_data,
        )

    def update_approval_activity(self, approval_id: str):
        self.approvals.update_approval_activity(approval_id)

    async def _seed_deploy_approval(self, project_id: str, proj: Dict):
        """Seed a deploy approval and wire up the post-approval executor.

        Used by the user-initiated `deploy` intent in intent_router for
        already-completed projects (workflow has already exited; we want a
        fresh deploy without re-running everything). The workflow's automatic
        gate_output → deploy edge does NOT come through here — that path is
        driven entirely by WorkflowEngine._run_deploy_node, which has its own
        request_approval + wait_for_approval + _execute_deploy sequence.

        Restart safety: the in-memory `_await_and_execute` task is lost on
        backend restart but the approval row remains in Mongo. The approve
        routes call `_respawn_deploy_executor` before flipping status so
        post-restart approve still triggers the deploy.
        """
        prod_enabled = os.getenv("DEPLOY_AGENT_PROD_ENABLED", "false").lower() == "true"
        local_enabled = os.getenv("DEPLOY_AGENT_LOCAL_ENABLED", "false").lower() == "true"

        if not (prod_enabled or local_enabled):
            return

        base_slug = (project_id.split("-")[0] or project_id).lower()
        deploy_data = {
            "gate_node_id": "deploy",
            "gate_label": "Deploy application",
            "deploy_slug": base_slug,
            "target_namespace": "AppFactory-apps",
            "deploy_mode": "prod" if prod_enabled else "local",
        }
        approval_id = await self.approvals.request_approval(
            project_id, ApprovalType.DEPLOY.value, deploy_data, run_id=proj.get("run_id")
        )

        self._spawn_deploy_executor(project_id, approval_id, proj)

    def _spawn_deploy_executor(self, project_id: str, approval_id: str, proj: Dict) -> None:
        """Spawn the `_await_and_execute` task that runs once the user approves.

        Shared between `_seed_deploy_approval` (initial seed) and
        `_respawn_deploy_executor` (post-restart re-spawn) so the body lives
        in one place. The task removes its entry from
        `_deploy_executor_tasks` in its `finally` so the dict reflects only
        live awaiters.
        """
        async def _await_and_execute():
            try:
                approval = await self.approvals.wait_for_approval(approval_id)
                if not approval or approval.get("status") in ("timeout", "rejected"):
                    return
                result = await self._execute_deploy(project_id, proj, approval)
                if isinstance(result, dict) and result.get("status") == "failed":
                    logger.warning(
                        "[DEPLOY] _execute_deploy returned failed for project_id=%s reason=%s recoverable=%s",
                        project_id,
                        result.get("reason"),
                        result.get("recoverable"),
                    )
            except Exception as e:
                import traceback
                logger.error(
                    "[DEPLOY] post-seed executor failed for %s: %s\n%s",
                    project_id, e, traceback.format_exc(),
                )
            finally:
                self._deploy_executor_tasks.pop(approval_id, None)

        task = create_task_with_context(_await_and_execute())
        self._deploy_executor_tasks[approval_id] = task

    async def _respawn_deploy_executor(
        self,
        project_id: str,
        approval_id: str,
        approval_doc: Dict,
    ) -> None:
        """Ensure a live executor awaits the given deploy approval.

        Called by the approve routes before flipping approval status. If an
        executor is already alive (no restart since the original
        `_seed_deploy_approval` ran), this is a no-op. After a backend
        restart the original task is gone; this re-spawns one bound to a
        freshly-loaded project so `_execute_deploy` actually runs when the
        user clicks approve.

        Only deploy approvals (`gate_node_id == "deploy"`) routed through
        `_seed_deploy_approval` need this; the workflow's automatic
        `gate_output → deploy` path is rehydrated by
        `ensure_workflow_running` instead, which re-enters
        `WorkflowEngine._run_deploy_node` (it has its own wait/execute).
        """
        existing = self._deploy_executor_tasks.get(approval_id)
        if existing is not None and not existing.done():
            return

        proj = self.active_projects.get(project_id)
        if not proj:
            try:
                proj = await self.projects.load_project_to_active(
                    project_id, self.agent_pool
                )
            except Exception as e:
                logger.error(
                    "[DEPLOY] respawn: load_project_to_active failed for %s: %s",
                    project_id, e,
                )
                return
        if not proj:
            return

        # Stamp run_id from the approval so deploy emits land on the right
        # run. The canonical loader doesn't carry run_id; matches the
        # equivalent stamp in `ensure_workflow_running`.
        run_id = approval_doc.get("run_id")
        if run_id:
            proj["run_id"] = run_id
            sc = proj.get("shared_context")
            if sc is not None:
                try:
                    sc.run_id = run_id
                except Exception:
                    pass

        # Make sure the approval is in the in-memory pending_approvals dict
        # so `wait_for_approval` (in-memory poll) can see status changes.
        # Restart-loaded approvals would otherwise live only in Mongo.
        if approval_id not in self.approvals.pending_approvals:
            self.approvals.pending_approvals[approval_id] = approval_doc

        self._spawn_deploy_executor(project_id, approval_id, proj)

    def _parse_deploy_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract deploy_status, needs_delegation, recommendation, error from deploy agent result."""
        out = ResultSchema.get_output(result) or {}
        raw = out.get("final_output") if isinstance(out, dict) else None
        logger.info(f"[DEPLOY] _parse_deploy_result: output type={type(out).__name__}, has final_output={raw is not None}")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                res = {"deploy_status": "failed", "error": raw[:500] if raw else "no output", "needs_delegation": False, "recommendation": ""}
                logger.warning(f"[DEPLOY] _parse_deploy_result: JSON parse failed, returning failed, error={res.get('error', '')[:200]}")
                return res
        elif isinstance(out, dict) and "deploy_status" in out:
            parsed = out
        else:
            logger.warning(f"[DEPLOY] _parse_deploy_result: no deploy JSON in output")
            return {"deploy_status": "failed", "error": "no deploy JSON", "needs_delegation": False, "recommendation": ""}
        res = {
            "deploy_status": parsed.get("deploy_status"),
            "needs_delegation": parsed.get("needs_delegation") is True,
            "recommendation": parsed.get("recommendation") or "",
            "error": parsed.get("error") or "",
        }
        logger.info(f"[DEPLOY] _parse_deploy_result: parsed status={res['deploy_status']}, needs_delegation={res['needs_delegation']}, recommendation_len={len(res['recommendation'])}")
        return res

    async def _run_coding_fix(self, project_id: str, sc: SharedContext, fix_description: str) -> bool:
        """Run coding_agent once to fix artifacts; returns True if run completed successfully."""
        logger.info(f"[DEPLOY] _run_coding_fix: project_id={project_id}, fix_desc_len={len(fix_description or '')}")
        cfg = await self._resolve_agent_config("coding_agent", sc.tenant_id)
        if not cfg:
            logger.warning(f"[DEPLOY] coding_agent config not found, skip delegation for project_id={project_id}")
            return False
        coding_agent = GenericAgent(cfg)
        coding_agent.inject_dependencies(
            shared_context=None,
            tool_registry=self.tool_registry,
            llm_client=self.llm_client,
            event_emitter=self.event_emitter,
            mcp_executor=self.mcp_executor,
        )
        coding_agent.shared_context = sc
        task = TaskSchema.create(
            task_id=f"deploy_fix_{project_id}",
            project_id=project_id,
            task_type="fix",
            description=fix_description,
            context={"source": "deploy", "fix_request": fix_description},
        )
        try:
            res = await coding_agent.execute_task(task)
            ok = ResultSchema.is_completed(res)
            logger.info(f"[DEPLOY] _run_coding_fix: project_id={project_id}, completed={ok}")
            return ok
        except Exception as e:
            logger.error(f"[DEPLOY] Coding fix failed for project_id={project_id}: {e}")
            return False
    
    async def _execute_deploy(self, project_id: str, proj: Dict, approval: Dict):
        """Execute deploy after deploy approval. Idempotent per approval_id to avoid double execution."""
        # A standalone approval executor does not pass through WorkflowEngine.
        async with run_trace_scope(
            self.storage, self.tracer, project_id, proj.get("run_id")
        ):
            return await self._execute_deploy_in_run(project_id, proj, approval)

    async def _execute_deploy_in_run(self, project_id: str, proj: Dict, approval: Dict):
        approval_id = ApprovalSchema.get_id(approval)
        if approval_id:
            lock = self.projects.get_lock(project_id)
            async with lock:
                if approval_id in self._deploy_executed_approval_ids:
                    logger.info(f"[DEPLOY] Deploy already executed for project_id={project_id} approval_id={approval_id}, skipping")
                    return
                if len(self._deploy_executed_approval_ids) >= 1000:
                    self._deploy_executed_approval_ids.clear()
                    logger.info(f"[DEPLOY] Cleared deploy idempotency set (max size reached)")
                self._deploy_executed_approval_ids.add(approval_id)  # атомарно: check+set под локом

        logger.info(f"[DEPLOY] _execute_deploy called for {project_id}")
        sc = proj.get("shared_context")
        if not isinstance(sc, SharedContext):
            logger.warning(f"[DEPLOY] shared_context is not SharedContext instance: {type(sc)}")
            return
        if not self.deploy_service:
            logger.warning("[DEPLOY] deploy_service is None, cannot deploy")
            return
        
        data = approval.get("data", {})
        deploy_slug = data.get("deploy_slug")
        target_namespace = data.get("target_namespace", "AppFactory-apps")
        
        prod_enabled = os.getenv("DEPLOY_AGENT_PROD_ENABLED", "false").lower() == "true"
        local_enabled = os.getenv("DEPLOY_AGENT_LOCAL_ENABLED", "false").lower() == "true"
        logger.info(f"[DEPLOY] prod_enabled={prod_enabled}, local_enabled={local_enabled}, slug={deploy_slug}")
        
        if not prod_enabled and not local_enabled:
            logger.warning(f"[DEPLOY] Neither prod nor local deploy enabled for {project_id}, skipping")
            return
        
        if prod_enabled:
            # Precheck: validate artifacts exist before spending a deploy attempt.
            # Uses the shared check_artifacts_ready helper so precheck and build
            # path never diverge on the readiness criteria.
            if self.artifact_store:
                try:
                    precheck = await check_artifacts_ready(project_id, self.artifact_store)
                except Exception as e:
                    logger.warning(
                        "[DEPLOY] precheck_failed project_id=%s error=%s — skipping precheck",
                        project_id, e,
                    )
                    precheck = None
                if precheck is not None and not precheck["ready"]:
                    logger.warning(
                        "[DEPLOY] precheck_ready=false project_id=%s artifacts=%d reason=%s",
                        project_id, precheck["count"], precheck["reason"],
                    )
                    if approval_id:
                        self._deploy_executed_approval_ids.discard(approval_id)
                    return {
                        "status": "failed",
                        "reason": "artifacts_missing",
                        "recoverable": True,
                    }
                elif precheck is not None:
                    logger.info(
                        "[DEPLOY] precheck_ready=true project_id=%s artifacts=%d",
                        project_id, precheck["count"],
                    )
            else:
                logger.warning(
                    "[DEPLOY] artifact_store_unavailable project_id=%s — skipping precheck",
                    project_id,
                )

            # Use GenericAgent with deploy config (tools: deploy_from_artifacts, detect_stack, analyze_and_repair).
            # Create a fresh instance per deploy so shared_context is not shared across projects.
            cfg = await self._resolve_agent_config("deploy_agent", sc.tenant_id)
            deploy_agent = None
            if cfg:
                deploy_agent = GenericAgent(cfg)
                deploy_agent.inject_dependencies(
                    shared_context=None,
                    tool_registry=self.tool_registry,
                    llm_client=self.llm_client,
                    event_emitter=self.event_emitter,
                    mcp_executor=None,
                )
            if deploy_agent:
                deploy_agent.shared_context = sc
                deploy_agent.deploy_service = self.deploy_service
                deploy_agent.current_task = None
                task = TaskSchema.create(
                    task_id=f"deploy_{project_id}",
                    project_id=project_id,
                    task_type="deploy",
                    description="Deploy application to production",
                    context={
                        "deploy_mode": "prod",
                        "deploy_slug": deploy_slug,
                        "target_namespace": target_namespace,
                    },
                )
                deploy_agent.current_task = task
                logger.info(f"[DEPLOY] Using GenericAgent (deploy) for {project_id}")
                max_attempts = 3
                last_result = None
                try:
                    for attempt in range(1, max_attempts + 1):
                        task["context"] = {
                            "deploy_mode": "prod",
                            "deploy_slug": deploy_slug,
                            "target_namespace": target_namespace,
                            "attempt": attempt,
                        }
                        deploy_agent.current_task = task
                        logger.info(f"[DEPLOY] Deploy attempt {attempt}/{max_attempts} for {project_id}, calling execute_task")
                        result = await deploy_agent.execute_task(task)
                        last_result = result
                        parsed = self._parse_deploy_result(result)
                        status = parsed.get("deploy_status") or ""
                        logger.info(f"[DEPLOY] Deploy attempt {attempt} for {project_id}: status={status}")
                        if status == "succeeded":
                            logger.info(f"[DEPLOY] Deploy succeeded for {project_id} on attempt {attempt}")
                            break
                        needs_delegation = parsed.get("needs_delegation") is True
                        logger.info(f"[DEPLOY] After attempt {attempt}: status={status}, needs_delegation={needs_delegation}, attempt<max={attempt < max_attempts}")
                        if needs_delegation and attempt < max_attempts:
                            err = parsed.get("error") or ""
                            rec = parsed.get("recommendation") or ""
                            fix_desc = f"Fix build/deploy error and update artifacts. Error: {err}. Recommendation: {rec}"
                            logger.info(f"[DEPLOY] Invoking _run_coding_fix for {project_id} (attempt {attempt})")
                            coding_ok = await self._run_coding_fix(project_id, sc, fix_desc)
                            logger.info(f"[DEPLOY] Coding fix run for {project_id}: ok={coding_ok}")
                        else:
                            logger.info(f"[DEPLOY] Stopping deploy loop for {project_id}: needs_delegation={needs_delegation}, attempt={attempt}, max_attempts={max_attempts}")
                            break
                    logger.info(f"[DEPLOY] Deploy finished for {project_id}: {last_result.get('status') if isinstance(last_result, dict) else last_result}")
                except Exception as e:
                    logger.error(f"[DEPLOY] Deploy failed for {project_id}: {e}")
                    import traceback
                    logger.error(f"[DEPLOY] Traceback: {traceback.format_exc()}")
            else:
                logger.warning(f"[DEPLOY] deploy_agent config not found in DB, skipping deploy for {project_id}")
        elif local_enabled:
            logger.info(f"[DEPLOY] Using local deploy for {project_id}")
            try:
                result = await self.deploy_service.deploy_static_demo_local(
                    project_id=project_id, shared_context=sc,
                    deploy_slug=deploy_slug, target_namespace=target_namespace,
                    cluster_target=data.get("cluster_target"),
                )
                logger.info(f"[DEPLOY] deploy_static_demo_local finished for {project_id}: {result}")
            except Exception as e:
                logger.error(f"[DEPLOY] deploy_static_demo_local failed for {project_id}: {e}")
                import traceback
                logger.error(f"[DEPLOY] Traceback: {traceback.format_exc()}")
    
    # === Revert Delegates ===
    async def list_snapshots(self, project_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        return await self.snapshot_manager.list_snapshots(project_id, limit)
    
    async def revert_project(self, project_id: str, target: Dict[str, Any], resume: bool = False) -> Dict[str, Any]:
        lock = self.projects.get_lock(project_id)
        async with lock:
            from storage.checkpoint_restore_store import CheckpointRestoreStore
            await CheckpointRestoreStore(self.storage).assert_mutation_allowed(project_id)
            proj = self.active_projects.get(project_id)
            if not proj:
                raise RuntimeError("Project not found")

            result = await self.reverts.revert_project(
                project_id, target, proj,
                reinstate_runtime_fn=lambda pid, sc: self._reinstate_runtime(pid, sc),
                cancel_project_fn=self._cancel_project_holding_lock,
                approval_manager=self.approvals,
                wait_for_task_completion_fn=self.projects.wait_for_task_completion,
            )

        # FSM spawn is delegated to the rehydration coordinator — single
        # owner pattern, same as approve/reject endpoints. `resume` is
        # accepted for HTTP back-compat but is now a no-op: ensure_running
        # is idempotent and always runs on revert.
        await self.ensure_workflow_running(project_id)
        return result

    async def revert_to_message_sequence(self, project_id: str, target_sequence: int) -> Dict[str, Any]:
        """PRIMARY: Revert to a specific message sequence using unified message system."""
        lock = self.projects.get_lock(project_id)
        async with lock:
            from storage.checkpoint_restore_store import CheckpointRestoreStore
            await CheckpointRestoreStore(self.storage).assert_mutation_allowed(project_id)
            proj = self.active_projects.get(project_id)
            if not proj:
                raise RuntimeError("Project not found")

            result = await self.reverts.revert_to_message_sequence(
                project_id, target_sequence, proj,
                cancel_project_fn=self._cancel_project_holding_lock,
                approval_manager=self.approvals,
                reinstate_runtime_fn=lambda pid, sc: self._reinstate_runtime(pid, sc),
                wait_for_task_completion_fn=self.projects.wait_for_task_completion,
            )

        await self.ensure_workflow_running(project_id)
        return result

    async def revert_to_latest_user_message(self, project_id: str) -> Dict[str, Any]:
        lock = self.projects.get_lock(project_id)
        async with lock:
            from storage.checkpoint_restore_store import CheckpointRestoreStore
            await CheckpointRestoreStore(self.storage).assert_mutation_allowed(project_id)
            proj = self.active_projects.get(project_id)
            if not proj:
                raise RuntimeError("Project not found")

            result = await self.reverts.revert_to_latest_user_message(
                project_id, proj,
                reinstate_runtime_fn=lambda pid, sc: self._reinstate_runtime(pid, sc),
                cancel_project_fn=self._cancel_project_holding_lock,
                approval_manager=self.approvals,
                wait_for_task_completion_fn=self.projects.wait_for_task_completion,
            )

        await self.ensure_workflow_running(project_id)
        return result
    
