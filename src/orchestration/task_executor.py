"""
Task Executor

Handles execution of plan tasks during the execution phase:
- Execute all tasks in the plan
- Execute single task via auction
- Task-level snapshots and events
"""

from typing import Dict, List, Any
import asyncio
import logging

from schemas import TaskSchema, ResultSchema, EventSchema
from telemetry.tracer import get_tracer

logger = logging.getLogger(__name__)


class TaskExecutor:
    """
    Executes plan tasks during the execution phase.
    
    Uses auction system to assign tasks to best-fit agents.
    """
    
    def __init__(
        self,
        auction,
        event_emitter,
        snapshot_manager,
        tracer=None,
        artifact_store=None,
        message_store=None,
        storage_backend=None,
    ):
        self.auction = auction
        self.event_emitter = event_emitter
        self.snapshot_manager = snapshot_manager
        self.tracer = tracer or get_tracer()
        self.artifact_store = artifact_store
        self.message_store = message_store
        self.storage = storage_backend

    @staticmethod
    def _trace_attempt_number(task: Dict) -> int:
        """Normalize the ordinal for this executor-owned task invocation."""
        raw_attempt = task.get(TaskSchema.TRACE_ATTEMPT) or task.get("attempt") or 1
        try:
            return max(1, int(raw_attempt))
        except (TypeError, ValueError):
            return 1

    async def _claim_trace_attempt(self, shared_context, task: Dict, agent) -> int:
        """Claim one durable attempt ordinal before emitting the lifecycle start."""
        run_id = getattr(shared_context, "run_id", None)
        task_id = TaskSchema.get_id(task)
        agent_id = getattr(agent, "agent_id", None)
        if self.storage is None or not all(
            isinstance(value, str) and value for value in (run_id, task_id, agent_id)
        ):
            return self._trace_attempt_number(task)
        try:
            claimed = await self.storage.claim_task_attempt(run_id, task_id, agent_id)
        except Exception as exc:
            logger.warning(
                "[ATTEMPT] run_id=%s task_id=%s agent_id=%s status=claim_failed error=%s",
                run_id,
                task_id,
                agent_id,
                exc,
            )
            return self._trace_attempt_number(task)
        if isinstance(claimed, int) and claimed > 0:
            task[TaskSchema.TRACE_ATTEMPT] = claimed
            return claimed
        logger.warning(
            "[ATTEMPT] run_id=%s task_id=%s agent_id=%s status=claim_unavailable",
            run_id,
            task_id,
            agent_id,
        )
        return self._trace_attempt_number(task)

    @staticmethod
    def _is_failed_task_result(task_result: dict) -> bool:
        if task_result.get("status") in ("failed", "cancelled"):
            return True
        agent_result = task_result.get("result")
        if not isinstance(agent_result, dict):
            return "result" in task_result
        return ResultSchema.is_failed(agent_result) or not ResultSchema.is_completed(agent_result)

    async def _emit_plan_task_terminal(
        self,
        shared_context,
        project_id: str,
        task: Dict,
        *,
        terminal: str,
        agent: Any = None,
        reason: str | None = None,
        agent_result: Dict | None = None,
    ) -> bool:
        """Emit task_completed / task_failed for plan tasks (UI terminal contract).

        Returns True when persisted, False after retries exhausted. Emit failure
        must not change agent execution outcome — callers keep success/fail as-is.
        """
        event_type = (
            EventSchema.TASK_COMPLETED
            if terminal == "completed"
            else EventSchema.TASK_FAILED
        )
        payload: Dict[str, Any] = {
            "project_id": project_id,
            "task_id": TaskSchema.get_id(task),
            "task_description": task.get("description")
            or task.get("task_description")
            or TaskSchema.get_description(task),
            "task_type": TaskSchema.get_type(task),
        }
        if agent is not None:
            payload["agent_id"] = getattr(agent, "agent_id", None)
            if hasattr(agent, "get_display_name"):
                payload["agent_display_name"] = agent.get_display_name()
            payload["attempt"] = self._trace_attempt_number(task)
        if reason:
            payload["reason"] = reason
        if terminal == "failed" and isinstance(agent_result, dict):
            err = agent_result.get("error")
            if err:
                payload["error"] = err
            err_type = agent_result.get("error_type")
            if err_type:
                payload["error_type"] = err_type
            gen = agent_result.get(ResultSchema.GENERATION_OUTCOME) or agent_result.get(
                "generation_outcome"
            )
            if gen:
                payload["generation_outcome"] = gen
        last_err = None
        for attempt in (1, 2):
            try:
                await self.event_emitter.emit(
                    event_type,
                    getattr(shared_context, "run_id", None),
                    payload,
                )
                return True
            except Exception as e:
                last_err = e
                if attempt == 1:
                    logger.warning(
                        "[EXECUTE] emit %s retry project_id=%s task_id=%s err=%s",
                        event_type,
                        project_id,
                        TaskSchema.get_id(task),
                        e,
                    )
        logger.error(
            "[EXECUTE] emit %s failed project_id=%s task_id=%s err=%s",
            event_type,
            project_id,
            TaskSchema.get_id(task),
            last_err,
        )
        return False
    
    async def execute_tasks(
        self,
        project_id: str,
        shared_context,
        agents: List[Any],
        token: Any,
        is_cancelled_fn: callable,
    ) -> Dict:
        """
        Execute all tasks in the plan.
        
        Plan is now a FLAT list, no hierarchy to flatten.
        Re-fetches plan on each iteration to pick up refinements.
        """
        # Track completed tasks
        completed_task_ids = set()
        results = []
        
        # Log initial plan state for diagnostics
        initial_plan = shared_context.get("plan", {})
        initial_tasks = initial_plan.get("tasks", [])
        logger.info(
            "[EXECUTE] execute_tasks project_id=%s plan_keys=%s task_count=%d",
            project_id, list(initial_plan.keys()), len(initial_tasks),
        )
        if not initial_tasks:
            logger.warning(
                "[EXECUTE] project_id=%s plan has 0 tasks — execution will be a no-op. "
                "Check that planner agent has output_save_key configured and plan was saved to shared_context.",
                project_id,
            )
        
        # Execute tasks - re-fetch plan on each iteration to pick up refinements
        while True:
            if is_cancelled_fn():
                return {"status": "cancelled", "task_results": results}
            
            # Re-fetch plan (may have been updated via /refine endpoint)
            plan = shared_context.get("plan", {})
            all_tasks = plan.get("tasks", [])
            
            # Find next task to execute
            next_task = None
            for task in all_tasks:
                task_id = task.get("task_id")
                if task_id not in completed_task_ids:
                    next_task = task
                    break
            
            # No more tasks
            if not next_task:
                logger.info("[EXECUTE] project_id=%s all %d tasks completed", project_id, len(completed_task_ids))
                print(f"✅ All {len(completed_task_ids)} tasks completed")
                break
            
            if is_cancelled_fn():
                return {"status": "cancelled", "task_results": results}
            
            # Execute task with per-task tracing + events
            task_index = len(completed_task_ids) + 1
            task_total = len(all_tasks)
            print(f"📋 Executing task {task_index}/{task_total}: {next_task.get('description', '')[:60]}...")
            
            task_type = next_task.get("task_type", "unknown")
            with self.tracer.start_span(
                f"execution.task.{task_type}",
                attributes={
                    "project.id": project_id,
                    "task.id": next_task.get("task_id"),
                    "task.type": task_type,
                    "task.description": (next_task.get("description", "") or "")[:200],
                    "task.index": task_index,
                    "task.total": task_total,
                },
            ) as exec_span:
                try:
                    try:
                        await self.event_emitter.emit("execution_task_started", getattr(shared_context, "run_id", None), {
                            "project_id": project_id,
                            "task_id": next_task.get("task_id"),
                            "task_description": next_task.get("description"),
                            "index": task_index,
                            "total": task_total,
                        })
                    except Exception:
                        pass

                    task_result = await self.execute_single_task(
                        project_id, next_task, shared_context, agents, token, is_cancelled_fn
                    )

                    # Mark span based on result status
                    result_obj = task_result.get("result") if isinstance(task_result, dict) else None
                    status = None
                    try:
                        if result_obj and ResultSchema.is_completed(result_obj):
                            status = "completed"
                            self.tracer.set_success(exec_span)
                        elif result_obj and ResultSchema.is_failed(result_obj):
                            status = "failed"
                            self.tracer.set_error(exec_span, RuntimeError("task_failed"))
                        else:
                            status = "unknown"
                    except Exception:
                        status = "unknown"

                    results.append(task_result)
                    completed_task_ids.add(next_task.get("task_id"))

                    try:
                        await self.event_emitter.emit("execution_task_finished", getattr(shared_context, "run_id", None), {
                            "project_id": project_id,
                            "task_id": next_task.get("task_id"),
                            "task_description": next_task.get("description"),
                            "status": status,
                            "agent_id": task_result.get("agent_id") if isinstance(task_result, dict) else None,
                            "index": task_index,
                            "total": task_total,
                        })
                    except Exception:
                        pass

                    if isinstance(task_result, dict) and self._is_failed_task_result(task_result):
                        stop_status = (
                            "cancelled"
                            if task_result.get("status") == "cancelled"
                            else "failed"
                        )
                        logger.warning(
                            "[EXECUTE] project_id=%s stopping after %s task_id=%s",
                            project_id,
                            stop_status,
                            next_task.get("task_id"),
                        )
                        return {"status": stop_status, "task_results": results}
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    try:
                        self.tracer.set_error(exec_span, e)
                    except Exception:
                        pass
                    task_id = next_task.get("task_id")
                    exc_msg = str(e) or type(e).__name__
                    is_cancel = is_cancelled_fn()
                    reason = "cancelled" if is_cancel else exc_msg
                    await self._emit_plan_task_terminal(
                        shared_context,
                        project_id,
                        next_task,
                        terminal="failed",
                        reason=reason,
                    )
                    failed_result = {
                        "task_id": task_id,
                        "status": "cancelled" if is_cancel else "failed",
                        "reason": reason,
                        "error": exc_msg,
                    }
                    results.append(failed_result)
                    completed_task_ids.add(task_id)
                    try:
                        await self.event_emitter.emit("execution_task_finished", getattr(shared_context, "run_id", None), {
                            "project_id": project_id,
                            "task_id": task_id,
                            "task_description": next_task.get("description"),
                            "status": "failed",
                            "agent_id": None,
                            "index": task_index,
                            "total": task_total,
                        })
                    except Exception:
                        pass
                    logger.warning(
                        "[EXECUTE] project_id=%s stopping after exception task_id=%s error=%s",
                        project_id,
                        task_id,
                        exc_msg,
                    )
                    return {
                        "status": "cancelled" if is_cancel else "failed",
                        "task_results": results,
                    }
        
        return {"task_results": results}

    async def execute_single_task(
        self,
        project_id: str,
        task: Dict,
        shared_context,
        agents: List[Any],
        token: Any,
        is_cancelled_fn: callable,
    ) -> Dict:
        """Execute one task via auction"""
        if is_cancelled_fn():
            await self._emit_plan_task_terminal(
                shared_context,
                project_id,
                task,
                terminal="failed",
                reason="cancelled",
            )
            return {"status": "cancelled", "task_id": task.get("task_id"), "reason": "cancelled"}
        
        # Add project context
        task["project_id"] = project_id
        
        if is_cancelled_fn():
            await self._emit_plan_task_terminal(
                shared_context,
                project_id,
                task,
                terminal="failed",
                reason="cancelled",
            )
            return {"status": "cancelled", "task_id": task.get("task_id"), "reason": "cancelled"}
        
        # Run auction
        try:
            logger.info(
                "[EXECUTE] auction.start project_id=%s task_id=%s desc=%s",
                project_id,
                TaskSchema.get_id(task),
                (TaskSchema.get_description(task) or "")[:120],
            )
        except Exception:
            pass
        
        # Snapshot: auction started (pre-bid state)
        try:
            await self.snapshot_manager.create_snapshot(
                project_id,
                shared_context,
                snap_type="auction_started",
                label=f"auction started: {TaskSchema.get_id(task)}",
                phase="execution",
                event_id=TaskSchema.get_id(task),
                meta={"description": TaskSchema.get_description(task)},
            )
        except Exception:
            pass
        
        # Set critic agent if available for execution phase
        try:
            critic_candidates = [
                a for a in agents
                if getattr(getattr(a, "agent_type", None), "value", "") == "critic"
            ]
            filtered_out = [
                getattr(a, "agent_id", "unknown")
                for a in critic_candidates
                if not getattr(a, "can_bid_on_phase", lambda _phase: True)("execution")
            ]
            critic = next(
                (
                    a for a in critic_candidates
                    if getattr(a, "can_bid_on_phase", lambda _phase: True)("execution")
                ),
                None,
            )
            if filtered_out:
                logger.info(
                    "[AUCTION] phase=execution critics_filtered=%s",
                    ",".join(filtered_out),
                )
            if critic:
                self.auction.critic_agent = critic
                logger.info(
                    "[AUCTION] phase=execution critic_selected=%s",
                    getattr(critic, "agent_id", "unknown"),
                )
            else:
                self.auction.critic_agent = None
                logger.info("[AUCTION] phase=execution no_critic_selected")
        except Exception:
            pass
        
        # Defensive: timebox the auction to prevent silent hangs
        try:
            auction_result = await asyncio.wait_for(
                self.auction.run_auction(task, agents, token, phase="execution", run_id=getattr(shared_context, "run_id", None)),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            logger.error("[EXECUTE] auction.timeout project_id=%s task_id=%s", project_id, TaskSchema.get_id(task))
            await self._emit_plan_task_terminal(
                shared_context,
                project_id,
                task,
                terminal="failed",
                reason="Auction timed out",
            )
            return {
                "task_id": task["task_id"],
                "status": "failed",
                "reason": "Auction timed out",
            }
        except asyncio.CancelledError:
            await self._emit_plan_task_terminal(
                shared_context,
                project_id,
                task,
                terminal="failed",
                reason="cancelled",
            )
            raise
        try:
            await self.snapshot_manager.create_snapshot(
                project_id,
                shared_context,
                snap_type="auction_completed",
                label=f"auction completed: {TaskSchema.get_id(task)}",
                phase="execution",
                event_id=TaskSchema.get_id(task),
                meta={"winner": (auction_result.get("winning_bid", {}) or {}).get("agent_id")},
            )
        except Exception:
            pass
        
        try:
            logger.info(
                "[EXECUTE] auction.end project_id=%s task_id=%s winner=%s conf=%s",
                project_id,
                TaskSchema.get_id(task),
                (auction_result.get("winning_bid", {}) or {}).get("agent_id"),
                (auction_result.get("winning_bid", {}) or {}).get("fit_score"),
            )
        except Exception:
            pass

        if not auction_result["winner"]:
            await self._emit_plan_task_terminal(
                shared_context,
                project_id,
                task,
                terminal="failed",
                reason="No agent could handle task",
            )
            return {
                "task_id": task["task_id"],
                "status": "failed",
                "reason": "No agent could handle task"
            }
        
        if is_cancelled_fn():
            await self._emit_plan_task_terminal(
                shared_context,
                project_id,
                task,
                terminal="failed",
                reason="cancelled",
            )
            return {"status": "cancelled", "task_id": task.get("task_id"), "reason": "cancelled"}
        
        # Execute
        agent = auction_result["winner"]
        try:
            logger.info(
                "[EXECUTE] agent.start project_id=%s task_id=%s agent=%s",
                project_id,
                TaskSchema.get_id(task),
                getattr(agent, "agent_id", None),
            )
        except Exception:
            pass
        
        try:
            logger.info("[EXECUTE] emit.task_assigned.start project_id=%s task_id=%s", project_id, TaskSchema.get_id(task))
            await asyncio.wait_for(self.event_emitter.emit(EventSchema.TASK_ASSIGNED, getattr(shared_context, "run_id", None), {
                "project_id": project_id,
                "task_id": TaskSchema.get_id(task),
                "task_description": TaskSchema.get_description(task),
                "agent_id": agent.agent_id,
                "agent_display_name": getattr(agent, "get_display_name", lambda: None)(),
                "winner_fit_score": auction_result.get("winning_bid", {}).get("fit_score")
            }), timeout=10.0)
            logger.info("[EXECUTE] emit.task_assigned.end project_id=%s task_id=%s", project_id, TaskSchema.get_id(task))
        except Exception:
            pass
        
        # Snapshot: task assigned
        try:
            await self.snapshot_manager.create_snapshot(
                project_id,
                shared_context,
                snap_type="task_assigned",
                label=f"task assigned: {TaskSchema.get_id(task)}",
                phase="execution",
                event_id=TaskSchema.get_id(task),
                meta={"agent": getattr(agent, "agent_id", None)},
            )
        except Exception:
            pass
        
        print(f"🏁 Assigned task to {agent.agent_id}; starting execution...")
        
        # Snapshot: task start checkpoint (for task-level reversion)
        try:
            await self.snapshot_manager.create_snapshot(
                project_id,
                shared_context,
                snap_type="task_start",
                label=f"task start: {TaskSchema.get_id(task)}",
                phase="execution",
                event_id=TaskSchema.get_id(task),
                meta={"tags": ["task", "task_start", f"task_id:{TaskSchema.get_id(task)}"]},
            )
        except Exception:
            pass
        
        attempt_number = await self._claim_trace_attempt(shared_context, task, agent)
        try:
            agent.current_task = task
        except Exception:
            pass
        # Uniform "agent X is now active" signal — see phase_runner._emit_agent_invocation
        # for the rationale. Without this the UI sees no event between the
        # tasks_assigned snapshot and the first agent.streaming.* (which can
        # be 30s+ away with buffered-thinking models).
        try:
            await self.event_emitter.emit("task_attempt", getattr(shared_context, "run_id", None), {
                "project_id": project_id,
                "agent_id": getattr(agent, "agent_id", None),
                "agent_display_name": (
                    agent.get_display_name() if hasattr(agent, "get_display_name") else None
                ),
                "task_id": TaskSchema.get_id(task),
                "task_description": task.get("description") or task.get("task_description"),
                "task_type": TaskSchema.get_type(task),
                "selection_mode": "auction",
                "attempt": attempt_number,
            })
        except Exception as e:
            print(f"⚠️ emit task_attempt failed: {e}")
        try:
            result = await agent.execute_task(task)
        except asyncio.CancelledError:
            await self._emit_plan_task_terminal(
                shared_context,
                project_id,
                task,
                terminal="failed",
                agent=agent,
                reason="cancelled",
            )
            raise
        except Exception as exc:
            exc_msg = str(exc) or type(exc).__name__
            reason = "cancelled" if is_cancelled_fn() else exc_msg
            await self._emit_plan_task_terminal(
                shared_context,
                project_id,
                task,
                terminal="failed",
                agent=agent,
                reason=reason,
            )
            return {
                "task_id": task.get("task_id"),
                "agent_id": getattr(agent, "agent_id", None),
                "status": "cancelled" if is_cancelled_fn() else "failed",
                "reason": reason,
                "error": exc_msg,
            }
        print(f"✅ Execution finished for {task.get('task_id')} by {agent.agent_id}")

        if isinstance(result, dict) and ResultSchema.is_completed(result):
            if not await self._emit_plan_task_terminal(
                shared_context, project_id, task, terminal="completed", agent=agent
            ):
                logger.warning(
                    "[EXECUTE] project_id=%s task_id=%s agent completed but task_completed emit failed — "
                    "execution outcome preserved; UI may backfill via execution_task_finished",
                    project_id,
                    TaskSchema.get_id(task),
                )
        else:
            reason = None
            if isinstance(result, dict) and ResultSchema.is_failed(result):
                reason = (
                    result.get("error")
                    or result.get("reason")
                    or ResultSchema.get_reasoning(result)
                    or str(result.get(ResultSchema.STATUS))
                )
            elif not isinstance(result, dict):
                reason = "invalid_result"
            else:
                reason = "unknown_status"
            await self._emit_plan_task_terminal(
                shared_context,
                project_id,
                task,
                terminal="failed",
                agent=agent,
                reason=reason,
                agent_result=result if isinstance(result, dict) else None,
            )
        
        # Snapshot: task completed
        try:
            await self.snapshot_manager.create_snapshot(
                project_id,
                shared_context,
                snap_type="task_completed",
                label=f"task completed: {TaskSchema.get_id(task)}",
                phase="execution",
                event_id=TaskSchema.get_id(task),
                meta={"agent": getattr(agent, "agent_id", None), "status": (result or {}).get("status")},
            )
        except Exception:
            pass

        try:
            status = None
            if isinstance(result, dict):
                if ResultSchema.is_completed(result):
                    status = "completed"
                elif ResultSchema.is_failed(result):
                    status = "failed"
            logger.info(
                "[EXECUTE] agent.end project_id=%s task_id=%s agent=%s status=%s",
                project_id,
                TaskSchema.get_id(task),
                getattr(agent, "agent_id", None),
                status,
            )
        except Exception:
            pass
        
        return {
            "task_id": task["task_id"],
            "agent_id": agent.agent_id,
            "result": result
        }
