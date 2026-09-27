"""
Phase Runner

Handles phase execution for the orchestration workflow:
- Run phase via auction
- Run phase with specific agent (direct)
- Emit phase transition events
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import asyncio
import uuid
import logging

from config.agent_delegation_identity import (
    agent_identity_matches_reference,
    base_agent_id,
)
from config.configuration_reference_validation import (
    normalize_workflow_node_agent_type_ref,
)
from config.configuration_resolution import agent_wire_name_from_doc
from schemas import TaskSchema, EventSchema, ResultSchema
from telemetry.tracer import get_tracer

logger = logging.getLogger(__name__)


def _trace_attempt_number(task: Dict[str, Any]) -> int:
    """Return the execution attempt ordinal owned by the phase runner."""
    raw = task.get(TaskSchema.TRACE_ATTEMPT) or task.get("attempt") or 1
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 1
    return value if value > 0 else 1


def _direct_agent_matches(
    agent: Any,
    target: str,
    *,
    tenant_id: Optional[str] = None,
) -> bool:
    """Match workflow direct ``agent_type`` to a runtime agent by configuration name."""
    ref = str(target or "").strip()
    if not ref:
        return False
    if base_agent_id(getattr(agent, "agent_id", "")) == ref:
        return True
    cfg = getattr(agent, "config", None)
    if isinstance(cfg, dict) and agent_identity_matches_reference(
        cfg,
        ref,
        parent_tenant=tenant_id,
    ):
        return True
    if isinstance(cfg, dict):
        wire = agent_wire_name_from_doc(cfg, runtime_tenant_id=tenant_id)
        if wire and wire == ref:
            return True
    return False


def _agent_type_value(agent: Any) -> Optional[str]:
    at = getattr(agent, "agent_type", None)
    if at is None:
        return None
    value = getattr(at, "value", at)
    return str(value) if value is not None else None


# Types that may map to multiple wire names — enum ref alone is never enough.
_AMBIGUOUS_DIRECT_TYPE_REFS = frozenset({"planner"})


def _resolve_direct_agent(
    agents: List[Any],
    target: str,
    *,
    tenant_id: Optional[str] = None,
) -> Optional[Any]:
    """Resolve workflow direct ref to a runtime agent (wire name first, then unique type)."""
    ref = str(target or "").strip()
    if not ref:
        return None
    for agent in agents:
        if _direct_agent_matches(agent, ref, tenant_id=tenant_id):
            return agent
    # ponytail: type fallback only when the pool has exactly one agent of that type;
    # upgrade path: require wire name when multiple agents share a type (planner collision).
    type_matches = [a for a in agents if _agent_type_value(a) == ref]
    if ref not in _AMBIGUOUS_DIRECT_TYPE_REFS and len(type_matches) == 1:
        return type_matches[0]
    return None


class PhaseRunner:
    """
    Runs workflow phases (requirements, planning, execution).
    
    Uses auction system to find best agent for each phase.
    """
    
    def __init__(
        self,
        auction,
        storage_backend,
        event_emitter,
        tracer=None,
    ):
        self.auction = auction
        self.storage = storage_backend
        self.event_emitter = event_emitter
        self.tracer = tracer or get_tracer()

    async def _emit_agent_invocation(
        self, project_id, agent, task, run_id=None, *, selection_mode="unknown", attempt=None
    ):
        """Emit task_attempt right before invoking an agent.

        This is the UNIFORM "agent X is now active" signal — fires for both
        auction-selected agents and sequentially-invoked workflow agents
        (gatherer / human_expert / finalizer / output / etc.). Without this,
        the UI has no way to distinguish "system is silent because upstream
        model is buffering its reasoning" from "system is hung" — see
        project fec87717-... where the user observed 31s+ gaps between
        agents with zero visible feedback. The UI's agentsPending state
        keys off this event to render "{Agent} is processing..." until the
        first agent.streaming.* arrives.

        project_id is REQUIRED — emitter.py routes events to SSE subscribers
        by data["project_id"]. Without it the event lands in the "global"
        bucket and the per-project UI subscriber never receives it (the
        first version of this helper omitted project_id; the diag on
        project e7b5e041-... showed zero task_attempt events reaching the
        UI even though the emit completed without error).
        """
        try:
            shared_context = getattr(agent, "shared_context", None)
            resolved_run_id = run_id or getattr(shared_context, "run_id", None)
            if not resolved_run_id:
                logger.warning(
                    "[PHASE_RUNNER] project_id=%s agent=%s event=task_attempt run_id=missing",
                    project_id, getattr(agent, "agent_id", "?"),
                )
            await self.event_emitter.emit("task_attempt", resolved_run_id, {
                "project_id": project_id,
                "agent_id": getattr(agent, "agent_id", None),
                "agent_display_name": (
                    agent.get_display_name() if hasattr(agent, "get_display_name") else None
                ),
                "task_id": TaskSchema.get_id(task),
                "task_description": task.get("description") or task.get("task_description"),
                "task_type": TaskSchema.get_type(task),
                "selection_mode": selection_mode,
                "attempt": attempt if attempt is not None else _trace_attempt_number(task),
            })
        except Exception as e:
            logger.warning(
                "[PHASE_RUNNER] emit task_attempt failed agent=%s task=%s err=%s",
                getattr(agent, "agent_id", "?"), TaskSchema.get_id(task), e,
            )

    async def _claim_trace_attempt(self, run_id: Any, task: Dict[str, Any], agent: Any) -> int:
        """Claim one durable attempt ordinal before the agent receives the task."""
        task_id = TaskSchema.get_id(task)
        agent_id = getattr(agent, "agent_id", None)
        if not all(isinstance(value, str) and value for value in (run_id, task_id, agent_id)):
            return _trace_attempt_number(task)
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
            return _trace_attempt_number(task)
        if isinstance(claimed, int) and claimed > 0:
            task[TaskSchema.TRACE_ATTEMPT] = claimed
            return claimed
        logger.warning(
            "[ATTEMPT] run_id=%s task_id=%s agent_id=%s status=claim_unavailable",
            run_id,
            task_id,
            agent_id,
        )
        return _trace_attempt_number(task)

    async def _emit_task_terminal(
        self,
        project_id,
        run_id,
        agent,
        task,
        *,
        result=None,
        error=None,
        attempt=None,
    ):
        """Persist a compact terminal lifecycle event for a phase task.

        PhaseRunner invokes ``execute_task`` directly, so it owns the terminal
        event for phase attempts. Observability failures remain best-effort and
        must never replace the agent's actual result or exception.
        """
        payload = {
            "project_id": project_id,
            "task_id": TaskSchema.get_id(task),
            "task_description": task.get("description") or task.get("task_description"),
            "task_type": TaskSchema.get_type(task),
            "agent_id": getattr(agent, "agent_id", None),
            "agent_display_name": (
                agent.get_display_name() if hasattr(agent, "get_display_name") else None
            ),
            "attempt": attempt if attempt is not None else _trace_attempt_number(task),
        }
        if error is not None:
            event_type = EventSchema.TASK_FAILED
            reason = str(error) or type(error).__name__
        elif isinstance(result, dict) and ResultSchema.is_completed(result):
            event_type = EventSchema.TASK_COMPLETED
            reason = None
        else:
            event_type = EventSchema.TASK_FAILED
            if not isinstance(result, dict):
                reason = "invalid_result"
            else:
                raw_status = ResultSchema.get_status(result)
                normalized_status = getattr(raw_status, "value", raw_status)
                status_text = str(normalized_status or "").strip().lower()
                reason = "cancelled" if status_text in {"cancelled", "canceled"} else (
                    result.get("reason")
                    or ResultSchema.get_reasoning(result)
                    or status_text
                    or "unknown_status"
                )
        if reason:
            payload["reason"] = str(reason)[:400]
        try:
            await self.event_emitter.emit(event_type, run_id, payload)
            return True
        except Exception as exc:
            logger.warning(
                "[PHASE_RUNNER] emit %s failed project_id=%s task_id=%s err=%s",
                event_type,
                project_id,
                TaskSchema.get_id(task),
                exc,
            )
            return False
    
    async def run_phase(
        self,
        project_id: str,
        phase_type: str,
        description: str,
        agents: List[Any],
        token: Any,
        is_cancelled_fn: callable,
        task_metadata: Optional[Dict[str, Any]] = None,
        auction_phase: Optional[str] = None,
        delegation_reviewers: Any = None,
        resume_attempt: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run a single phase (requirements, planning, etc.).

        Uses auction to find best agent for the phase.

        ``resume_attempt`` ({task_id, agent_id, ...}) continues an attempt
        interrupted by a backend restart: the recorded agent is reused
        without an auction and the recorded task_id keeps journal writes in
        the same attempt. If the recorded agent left the pool, fall back to
        a fresh auction under a fresh task id — reusing the id under a
        different agent would splice two agents' transcripts into one
        attempt.
        """
        with self.tracer.start_span(
            f"phase.{phase_type}",
            attributes={
                "project.id": project_id,
                "phase.name": phase_type,
                "phase.description": description,
                "AppFactory.phase.type": phase_type,
                "AppFactory.phase.is_first": phase_type == "requirements_gathering",
                "AppFactory.phase.is_last": phase_type == "output_review"
            }
        ) as phase_span:
            try:
                if is_cancelled_fn():
                    return {"status": "cancelled"}

                agent = None
                if resume_attempt:
                    agent = next(
                        (
                            a for a in agents
                            if getattr(a, "agent_id", None) == resume_attempt.get("agent_id")
                        ),
                        None,
                    )
                    if agent is not None:
                        logger.info(
                            "[REHYDRATE] project_id=%s resuming task %s with agent %s (no auction)",
                            project_id, resume_attempt.get("task_id"), agent.agent_id,
                        )
                    else:
                        logger.error(
                            "[REHYDRATE] project_id=%s recorded resume agent %s is no "
                            "longer in the pool (config removed/renamed between crash "
                            "and resume) — re-running task %s from scratch via a fresh "
                            "auction. The interrupted attempt's prior tool activity is "
                            "visible in the agent's context, but any external side "
                            "effects it caused (commits, deploys) may re-execute.",
                            project_id, resume_attempt.get("agent_id"),
                            resume_attempt.get("task_id"),
                        )

                # Use TaskSchema for consistent format
                task = TaskSchema.create(
                    task_id=(
                        resume_attempt["task_id"] if agent is not None
                        else str(uuid.uuid4())
                    ),
                    project_id=project_id,
                    task_type=phase_type,
                    description=description,
                    is_first_task=phase_type == "requirements_gathering",
                    **(task_metadata or {}),
                )
                if agent is not None:
                    task["resume_from_journal"] = True

                if is_cancelled_fn():
                    return {"status": "cancelled"}

                phase_label = auction_phase or (
                    "requirements"
                    if phase_type == "requirements_gathering"
                    else phase_type
                )

                if agent is None:
                    # Set critic agent if available for the current phase
                    try:
                        critic_candidates = [
                            a for a in agents
                            if getattr(getattr(a, "agent_type", None), "value", "") == "critic"
                        ]
                        filtered_out = [
                            getattr(a, "agent_id", "unknown")
                            for a in critic_candidates
                            if not getattr(a, "can_bid_on_phase", lambda _phase: True)(phase_label)
                        ]
                        critic = next(
                            (
                                a for a in critic_candidates
                                if getattr(a, "can_bid_on_phase", lambda _phase: True)(phase_label)
                            ),
                            None,
                        )
                        if filtered_out:
                            logger.info(
                                "[AUCTION] phase=%s critics_filtered=%s",
                                phase_label,
                                ",".join(filtered_out),
                            )
                        if critic:
                            self.auction.critic_agent = critic
                            logger.info(
                                "[AUCTION] phase=%s critic_selected=%s",
                                phase_label,
                                getattr(critic, "agent_id", "unknown"),
                            )
                        else:
                            self.auction.critic_agent = None
                            logger.info(
                                "[AUCTION] phase=%s no_critic_selected",
                                phase_label,
                            )
                    except Exception:
                        pass

                    # Run auction with timeout
                    try:

                        # TODO: plumb run_id from caller (workflow_engine knows it).
                        # For now auction events emit with run_id=None — they show
                        # in every run view (lenient run_id filter), not just the
                        # active one. Acceptable tradeoff vs. invasive signature
                        # changes across 5 run_phase callers.
                        auction_result = await asyncio.wait_for(
                            self.auction.run_auction(task, agents, token, phase=phase_label, run_id=None),
                            timeout=120.0
                        )
                    except asyncio.TimeoutError:
                        logger.error("[EXECUTE] auction.timeout project_id=%s task_id=%s", project_id, TaskSchema.get_id(task))
                        return {
                            "task_id": task["task_id"],
                            "status": "failed",
                            "reason": "Auction timed out"
                        }
                    except Exception as e:
                        logger.error("[EXECUTE] auction.exception project_id=%s task_id=%s err=%s", project_id, TaskSchema.get_id(task), str(e))
                        return {
                            "task_id": task["task_id"],
                            "status": "failed",
                            "reason": f"Auction error: {e}"
                        }

                    try:
                        logger.info(
                            "[EXECUTE] after.auction_result project_id=%s task_id=%s winner=%s conf=%s",
                            project_id,
                            TaskSchema.get_id(task),
                            (auction_result.get("winning_bid", {}) or {}).get("agent_id"),
                            (auction_result.get("winning_bid", {}) or {}).get("fit_score"),
                        )
                    except Exception:
                        pass

                    if not auction_result or not auction_result.get("winner"):
                        raise RuntimeError(f"No agent could handle phase: {phase_type}")

                    # Add auction results to span
                    if phase_span:
                        phase_span.set_attribute("agent.selected", auction_result["winner"].agent_id)
                        phase_span.set_attribute("auction.fit_score", auction_result.get("winning_bid", {}).get("fit_score", 0))

                    # Execute with winning agent
                    agent = auction_result["winner"]
                elif phase_span:
                    phase_span.set_attribute("agent.selected", agent.agent_id)
                    phase_span.set_attribute("phase.resumed_from_journal", True)

                if is_cancelled_fn():
                    return {"status": "cancelled"}
                    
                run_id = getattr(getattr(agent, "shared_context", None), "run_id", None)
                attempt_number = await self._claim_trace_attempt(run_id, task, agent)
                try:
                    agent.current_task = task
                except Exception:
                    pass
                prior_reviewers = getattr(agent, "delegation_reviewers", None)
                agent.delegation_reviewers = delegation_reviewers
                agent._delegation_trajectory = []
                try:
                    selection_mode = (
                        "resume" if resume_attempt and task.get("resume_from_journal") else "auction"
                    )
                    await self._emit_agent_invocation(
                        project_id, agent, task, selection_mode=selection_mode,
                        attempt=attempt_number,
                    )
                    result = await agent.execute_task(task)
                except asyncio.CancelledError:
                    await self._emit_task_terminal(
                        project_id,
                        getattr(getattr(agent, "shared_context", None), "run_id", None),
                        agent,
                        task,
                        error=RuntimeError("cancelled"),
                        attempt=attempt_number,
                    )
                    raise
                except Exception as exc:
                    await self._emit_task_terminal(
                        project_id,
                        getattr(getattr(agent, "shared_context", None), "run_id", None),
                        agent,
                        task,
                        error=exc,
                        attempt=attempt_number,
                    )
                    raise
                else:
                    await self._emit_task_terminal(
                        project_id,
                        getattr(getattr(agent, "shared_context", None), "run_id", None),
                        agent,
                        task,
                        result=result,
                        attempt=attempt_number,
                    )
                finally:
                    if (
                        delegation_reviewers is not None
                        and hasattr(delegation_reviewers, "cleanup")
                    ):
                        delegation_reviewers.cleanup()
                    agent.delegation_reviewers = prior_reviewers

                self.tracer.set_success(phase_span)
                return result
            except Exception as e:
                self.tracer.set_error(phase_span, e)
                raise
    
    async def resolve_direct_agent(
        self,
        agent_type: str,
        agents: List[Any],
        tenant_id: Optional[str] = None,
    ) -> Optional[Any]:
        resolved_agent_type = str(agent_type or "").strip()
        if resolved_agent_type and tenant_id and self.storage is not None:
            try:
                from config.configuration_reference_validation import (
                    known_tenant_ids_from_storage,
                )

                tenant_ids = await known_tenant_ids_from_storage(self.storage)
                resolved_agent_type = await normalize_workflow_node_agent_type_ref(
                    resolved_agent_type,
                    tenant_id=tenant_id,
                    storage=self.storage,
                    known_tenant_ids=tenant_ids,
                )
            except Exception as exc:
                logger.warning(
                    "[PHASE_RUNNER] agent_type=%s tenant_id=%s — resolution failed: %s",
                    agent_type,
                    tenant_id,
                    exc,
                )
        return _resolve_direct_agent(agents, resolved_agent_type, tenant_id=tenant_id)

    async def run_phase_direct(
        self,
        project_id: str,
        task: Dict[str, Any],
        agent_type: str,
        agents: List[Any],
        phase: Optional[str] = None,
        tenant_id: Optional[str] = None,
        delegation_reviewers: Any = None,
        resume_attempt: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run a phase with a specific agent type (no auction).

        Args:
            project_id: Project ID
            task: Task to execute
            agent_type: Bare agent name from workflow node (e.g., "human_expert")
            agents: List of available agents
            tenant_id: Project tenant for configuration identity resolution
            delegation_reviewers: Optional AppFactory-154 reviewer seam for this run
            resume_attempt: Restart-resume marker ({task_id, agent_id, ...});
                adopted only after agent resolution, and only when the
                resolved agent IS the recorded one — the node's agent_type
                may have been edited between crash and resume, and a reused
                task id under a different agent would splice two agents'
                transcripts into one attempt

        Returns:
            Task result
        """
        agent = await self.resolve_direct_agent(agent_type, agents, tenant_id)

        if not agent:
            raise RuntimeError(f"No agent found with type: {agent_type}")

        if resume_attempt:
            if getattr(agent, "agent_id", None) == resume_attempt.get("agent_id"):
                task[TaskSchema.TASK_ID] = resume_attempt["task_id"]
                task["resume_from_journal"] = True
                logger.info(
                    "[REHYDRATE] project_id=%s resuming task %s with direct agent %s",
                    project_id, resume_attempt.get("task_id"), agent.agent_id,
                )
            else:
                # At parity with the auction path (run_phase): the node's
                # agent_type changed between crash and resume, so the fresh run
                # may re-execute the attempt's side effects — hence ERROR.
                logger.error(
                    "[REHYDRATE] project_id=%s recorded resume agent %s no longer "
                    "resolves for this node (resolved to %s instead) — re-running "
                    "task %s from scratch as a fresh direct run; any external side "
                    "effects the interrupted attempt caused may re-execute.",
                    project_id, resume_attempt.get("agent_id"),
                    getattr(agent, "agent_id", None),
                    resume_attempt.get("task_id"),
                )

        # Direct selection is an explicit workflow override. Keep auction strict,
        # but only log here so intentionally assigned agents still run.
        effective_phase = phase or task.get("phase_label")
        if effective_phase and hasattr(agent, "can_bid_on_phase"):
            try:
                if not agent.can_bid_on_phase(effective_phase):
                    logger.info(
                        "[PHASE_RUNNER] Direct override: agent '%s' assigned to phase '%s' "
                        "(outside allowed_phases=%s)",
                        getattr(agent, "agent_id", agent_type),
                        effective_phase,
                        getattr(agent, "allowed_phases", []),
                    )
            except Exception as e:
                logger.warning(
                    "[PHASE_RUNNER] phase=%s direct_phase_check_error agent=%s err=%s",
                    effective_phase,
                    getattr(agent, "agent_id", "unknown"),
                    str(e),
                )

        try:
            agent.current_task = task
        except Exception:
            pass

        # AppFactory-154: attach the delegation reviewer seam + reset the per-run
        # delegation trajectory for this execute_task only, so reviewers/critics
        # never leak across phase runs of a pooled per-project agent clone.
        prior_reviewers = getattr(agent, "delegation_reviewers", None)
        agent.delegation_reviewers = delegation_reviewers
        agent._delegation_trajectory = []
        run_id = getattr(getattr(agent, "shared_context", None), "run_id", None)
        attempt_number = await self._claim_trace_attempt(run_id, task, agent)

        # Execute directly
        try:
            selection_mode = (
                "resume" if resume_attempt and task.get("resume_from_journal") else "direct"
            )
            await self._emit_agent_invocation(
                project_id, agent, task, selection_mode=selection_mode,
                attempt=attempt_number,
            )
            result = await agent.execute_task(task)
        except asyncio.CancelledError:
            await self._emit_task_terminal(
                project_id,
                getattr(getattr(agent, "shared_context", None), "run_id", None),
                agent,
                task,
                error=RuntimeError("cancelled"),
                attempt=attempt_number,
            )
            raise
        except Exception as exc:
            await self._emit_task_terminal(
                project_id,
                getattr(getattr(agent, "shared_context", None), "run_id", None),
                agent,
                task,
                error=exc,
                attempt=attempt_number,
            )
            raise
        else:
            await self._emit_task_terminal(
                project_id,
                getattr(getattr(agent, "shared_context", None), "run_id", None),
                agent,
                task,
                result=result,
                attempt=attempt_number,
            )
            return result
        finally:
            # Drop this run's resolved delegation approvals (race-free: the run is
            # over, no gate is being awaited) before detaching the reviewer seam.
            if delegation_reviewers is not None and hasattr(delegation_reviewers, "cleanup"):
                delegation_reviewers.cleanup()
            agent.delegation_reviewers = prior_reviewers
    
    async def emit_phase_event(
        self,
        phase: str,
        project_id: str,
        status: str,
        project_state: Optional[Dict] = None,
    ):
        """Emit phase transition event"""
        # Keep project state in sync for UI status cards
        if project_state:
            project_state["current_phase"] = phase

            # On phase.X.started, force status='running'. Without this, a
            # freshly created project keeps status='initialized' for the entire
            # requirements phase (update_project_status('running') only fires
            # post-plan-approval at orchestrator.py:856), so the UI's
            # projectIsRunning check fails and falls through to "Ready" even
            # while phases are actively executing. Verified against project
            # 0a711fa7-c0ab-4e56-b458-e1efb007fb2e: status remained
            # "initialized" with updated_at == created_at across 524 events.
            if status == EventSchema.PHASE_STARTED:
                project_state["status"] = "running"
            persisted_status = project_state.get("status", "running")

            # Persist phase change to MongoDB
            await self.storage.save_project(project_id, {
                "user_prompt": project_state.get("user_prompt", ""),
                "title": project_state.get("title", "Untitled Project"),
                "status": persisted_status,
                "current_phase": phase,
                "approval_mode": project_state.get("approval_mode", "human"),
                "created_at": project_state.get("created_at"),
                "metadata": project_state.get("metadata", {})
            })
        
        event_name = EventSchema.format_phase_event(phase, status)
        # Pull run_id from project_state — phase events are run-scoped, so the
        # UI can filter them when viewing previous runs.
        run_id = project_state.get("run_id") if project_state else None
        await self.event_emitter.emit(event_name, run_id, {
            "project_id": project_id,
            "phase": phase,
            "status": status,
            # Timezone-aware UTC so the FE's Date.parse treats this as UTC,
            # not local time. The previous datetime.utcnow().isoformat()
            # emitted a naive "2026-05-14T09:57:53.275716" — per ECMAScript
            # spec, ISO date-time strings without a TZ designator are parsed
            # as LOCAL time, so a UTC+1 browser computed stopwatches that
            # were 1 hour off (verified on project 7c2243a7 where the chat
            # showed "phase → planning → started [60m 5s]" for a phase that
            # had fired 5s earlier).
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
