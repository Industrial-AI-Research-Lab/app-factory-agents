from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable
from typing import Any, Dict, Optional

from schemas import EventSchema, ResultSchema, TaskSchema, TaskStatus
from config.agent_delegation_identity import (
    agent_identities_equal,
    agent_identity_from_runtime,
    bare_delegation_name,
)
from orchestration.agent_result_review import AgentResultDecision
from .delegation_policy import (
    SYSTEM_TENANT_ID,
    base_agent_id,
    delegation_activation_error,
    delegation_target_error,
    normalize_delegation_targets,
)
from .delegation_targets import find_project_agent, target_display_name
from .a2a_delegate import delegate_to_a2a_agent

logger = logging.getLogger(__name__)

DELEGATED_TASK_TYPE = "delegated"


def _target_can_ask_human(target: Any) -> bool:
    """True if the delegated agent may call ``ask_human``.

    Delegation runs the child synchronously (``execute_task``). A child that
    parks on ask_human has no restart-recoverable resume: find_interrupted_attempt
    keys on the PARENT's node-bearing task_id and never sees the child's
    node-less ask_human, so on restart the parent's ``delegate_to_agent`` is
    force-closed and the child's live question is orphaned (F5). Until delegation
    sub-state is rehydrated, delegating to such a child is unsupported — the
    caller refuses it loudly instead of shipping that latent bug.
    """
    effective = getattr(target, "_effective_allowed_tools", None)
    if not isinstance(effective, (list, tuple, set)):
        try:
            from tools.agent_allowed_tools import effective_allowed_tool_ids

            effective = effective_allowed_tool_ids(getattr(target, "config", {}) or {})
        except Exception:
            return False
    try:
        return "ask_human" in set(effective or [])
    except TypeError:
        return False


class DelegationManager:
    """Runs a full project agent as a delegated sub-agent."""

    def __init__(self, agent: Any):
        self.agent = agent

    async def delegate_to_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        agent_id = str(args.get("agent_id") or "").strip()
        instruction = str(args.get("instruction") or "").strip()
        if not agent_id or not instruction:
            return {"status": "error", "error": "Missing agent_id or instruction"}

        permission_error = self._validate_parent_permission(agent_id)
        if permission_error:
            return {"status": "error", "agent_id": agent_id, "error": permission_error}

        requested_base_id = base_agent_id(agent_id)
        permission_error = self._validate_target_permission(requested_base_id)
        if permission_error:
            return {"status": "error", "agent_id": agent_id, "error": permission_error}

        target = self._find_target_agent(agent_id)
        if not target:
            # Pool miss: the target may name a registered a2a server. Policy (activation,
            # target allow-list, tenant) already ran above and applies to a2a too, so this
            # is not a policy bypass. None here means "not a usable a2a target either".
            #
            # No separate delegation-cycle check runs on this path, and our chain-based one
            # would be a no-op here anyway: the chain only holds base ids of POOL agents
            # already traversed, and we reach here only because no pool agent resolves this
            # name — so the a2a name can never be in it. (A registered a2a server could itself
            # be another AppFactory instance that delegates back; the delegation chain is not
            # propagated over the protocol, so a cross-instance cycle isn't structurally
            # impossible — but each hop is bounded by the server's request_timeout_seconds, so
            # it cannot hang forever.) Tenant scope is enforced by the tenant-scoped lookup.
            a2a_result = await self._maybe_delegate_to_a2a(agent_id, instruction)
            if a2a_result is not None:
                return a2a_result
            return {
                "status": "error",
                "agent_id": agent_id,
                "error": f"Agent '{agent_id}' not found in project agent_pool",
            }
        if not callable(getattr(target, "execute_task", None)):
            return {
                "status": "error",
                "agent_id": agent_id,
                "error": f"Agent '{agent_id}' cannot execute delegated tasks",
            }

        target_id = str(getattr(target, "agent_id", agent_id))
        target_base_id = base_agent_id(target_id)

        if _target_can_ask_human(target):
            logger.warning(
                "[DELEGATION] ask_human_child_denied parent=%s child=%s",
                getattr(self.agent, "agent_id", "unknown"),
                target_id,
            )
            return {
                "status": "error",
                "agent_id": agent_id,
                "error": (
                    f"Delegation to '{target_base_id}' is not supported: it can ask "
                    "a human (ask_human), but a delegated child's question cannot be "
                    "answered or recovered after a backend restart (ADR-0010). Remove "
                    "ask_human from the delegated agent's tools."
                ),
            }

        tenant_error = self._validate_tenant_scope(target)
        if tenant_error:
            logger.warning(
                "[DELEGATION] tenant_scope_denied parent=%s parent_tenant=%s "
                "child=%s child_tenant=%s",
                getattr(self.agent, "agent_id", "unknown"),
                self._parent_tenant_id(),
                target_id,
                self._target_tenant_id(target),
            )
            return {"status": "error", "agent_id": agent_id, "error": tenant_error}

        cycle_error = self._validate_delegation_chain(target_base_id)
        if cycle_error:
            logger.warning(
                "[DELEGATION] cycle_denied parent=%s child=%s chain=%s",
                getattr(self.agent, "agent_id", "unknown"),
                target_id,
                self._current_delegation_chain(),
            )
            return {"status": "error", "agent_id": agent_id, "error": cycle_error}

        project_id = self._project_id()
        target_name = target_display_name(target)
        sub_task = self._create_sub_task(project_id, target_base_id, instruction)

        # AppFactory-154 reviewer seam (set per-run by PhaseRunner). None → unchanged behavior.
        reviewers = getattr(self.agent, "delegation_reviewers", None)
        pre_review = None
        if reviewers is not None:
            pre = await reviewers.review_pre(
                agent=self.agent,
                child_agent=target_id,
                instruction=instruction,
                task_id=TaskSchema.get_id(sub_task),
            )
            if pre.get("decision") == "reject":
                reason = pre.get("reason", "")
                msg = "Delegation rejected by reviewer"
                if reason:
                    msg = f"{msg}: {reason}"
                await self._emit_failed(
                    project_id, target_id, target_name, sub_task, msg,
                    child_status="rejected_by_reviewer",
                )
                # The outcome must NOT ride the "status" key — the runner's
                # _format_tool_output strips "status" before the orchestrator LLM
                # sees the tool result. Use dedicated fields instead.
                return {
                    "agent_id": target_id,
                    "review_outcome": "rejected",
                    "review_reason": reason,
                    "reviewer": pre.get("reviewer", "unknown"),
                }
            revised = pre.get("instruction")
            if revised and revised != instruction:
                instruction = revised
                sub_task = self._create_sub_task(project_id, target_base_id, instruction)
            pre_review = pre.get("review")  # PRE-critic verdict, for event observability

        started_payload = {
            "project_id": project_id,
            "parent_agent": getattr(self.agent, "agent_id", "unknown"),
            "child_agent": target_id,
            "child_agent_name": target_name,
            "requested_agent": agent_id,
            "task_id": TaskSchema.get_id(sub_task),
            "instruction": instruction[:200],
        }
        if pre_review:
            started_payload["review"] = pre_review
        await self._emit(EventSchema.AGENT_DELEGATION_STARTED, started_payload)

        suppress_raw_message = bool(
            reviewers is not None
            and reviewers.requires_human_post_review(target_id)
        )
        if suppress_raw_message:
            sub_task[TaskSchema.SUPPRESS_ASSISTANT_MESSAGE] = True

        logger.info(
            "[DELEGATION] started project_id=%s parent=%s child=%s task_id=%s",
            project_id,
            getattr(self.agent, "agent_id", "unknown"),
            target_id,
            TaskSchema.get_id(sub_task),
        )

        try:
            result = await target.execute_task(sub_task)
        except Exception as exc:
            logger.exception(
                "[DELEGATION] exception project_id=%s parent=%s child=%s err=%s",
                project_id,
                getattr(self.agent, "agent_id", "unknown"),
                target_id,
                exc,
            )
            await self._emit_failed(project_id, target_id, target_name, sub_task, str(exc))
            return {"status": "error", "agent_id": target_id, "error": str(exc)}

        if not isinstance(result, dict):
            error = f"Sub-agent returned invalid result type '{type(result).__name__}'"
            await self._emit_failed(project_id, target_id, target_name, sub_task, error)
            return {"status": "error", "agent_id": target_id, "error": error}

        child_status = self._status_value(ResultSchema.get_status(result))
        if not ResultSchema.is_completed(result):
            await self._emit_failed(
                project_id,
                target_id,
                target_name,
                sub_task,
                f"Sub-agent returned status '{child_status}'",
                child_status=child_status,
            )
            return {
                "status": "error",
                "agent_id": target_id,
                "child_status": child_status,
                "result": result,
            }

        critic_annotation = None
        post_review = None
        if reviewers is not None:
            async def _rerun(
                *,
                decision: str,
                feedback: str,
                previous_result: Optional[Dict[str, Any]],
            ):
                retry_feedback = feedback
                if decision == AgentResultDecision.REJECT.value:
                    retry_feedback = (
                        "The human reviewer rejected the previous result in full. "
                        "Restart from the original delegated instruction and produce "
                        "a new result."
                    )
                    if feedback:
                        retry_feedback += f"\nReviewer note: {feedback}"
                retry_sub_task = self._create_sub_task(
                    project_id,
                    target_base_id,
                    instruction,
                    retry_feedback=retry_feedback,
                    review_decision=decision,
                    previous_result=previous_result,
                )
                if suppress_raw_message:
                    retry_sub_task[TaskSchema.SUPPRESS_ASSISTANT_MESSAGE] = True
                return await target.execute_task(retry_sub_task)

            post = await reviewers.review_post(
                agent=self.agent,
                child_agent=target_id,
                result=result,
                task_id=TaskSchema.get_id(sub_task),
                instruction=instruction,
                rerun=_rerun,
            )
            if post.get("decision") == "reject":
                reason = post.get("reason", "")
                msg = "Delegation rejected by human"
                if reason:
                    msg = f"{msg}: {reason}"
                await self._emit_failed(
                    project_id, target_id, target_name, sub_task, msg,
                    child_status="rejected_by_human",
                )
                return {
                    "agent_id": target_id,
                    "review_outcome": "rejected_by_human",
                    "review_reason": reason,
                    "reviewer": "human",
                }
            result = post.get("result", result)
            critic_annotation = post.get("critic")  # post-critic REFINE/REPLAN directive, if any
            post_review = post.get("review")  # critic verdict (incl. happy path), for event observability
            # A human-approved re-run (or any reviewer-substituted result) bypasses the
            # first-run validation above, so re-validate and recompute child_status from
            # the FINAL result — otherwise a failed re-run is returned as success with a
            # stale "completed" status.
            if not isinstance(result, dict) or not ResultSchema.is_completed(result):
                bad_status = (
                    self._status_value(ResultSchema.get_status(result))
                    if isinstance(result, dict) else "invalid"
                )
                await self._emit_failed(
                    project_id, target_id, target_name, sub_task,
                    f"Approved delegation result is not completed (status={bad_status})",
                    child_status=bad_status,
                )
                return {
                    "status": "error",
                    "agent_id": target_id,
                    "child_status": bad_status,
                    "result": result if isinstance(result, dict) else None,
                }
            child_status = self._status_value(ResultSchema.get_status(result))

        if hasattr(self.agent, "_delegation_trajectory"):
            self.agent._delegation_trajectory.append({
                "child_agent": target_id,
                "instruction": instruction[:500],
                "output": self._result_preview(ResultSchema.get_output(result), 800),
            })

        completed_payload = {
            "project_id": project_id,
            "parent_agent": getattr(self.agent, "agent_id", "unknown"),
            "child_agent": target_id,
            "child_agent_name": target_name,
            "task_id": TaskSchema.get_id(sub_task),
            "status": child_status,
            "result": self._result_preview(ResultSchema.get_output(result)),
        }
        if post_review:
            completed_payload["review"] = post_review
        await self._emit(EventSchema.AGENT_DELEGATION_COMPLETED, completed_payload)
        logger.info(
            "[DELEGATION] completed project_id=%s parent=%s child=%s task_id=%s status=%s",
            project_id,
            getattr(self.agent, "agent_id", "unknown"),
            target_id,
            TaskSchema.get_id(sub_task),
            child_status,
        )

        success = {
            "status": "success",
            "agent_id": target_id,
            "child_status": child_status,
            "result": ResultSchema.get_output(result),
            "artifacts": ResultSchema.get_artifacts(result),
            "reasoning": ResultSchema.get_reasoning(result),
        }
        if critic_annotation:
            # Top-level so it survives _format_tool_output (which strips "status").
            # The orchestrator reacts to it per CRITIC_PROTOCOL (REFINE/REPLAN).
            success["_critic"] = critic_annotation
        return success

    def _validate_parent_permission(self, requested_agent_id: str) -> Optional[str]:
        task = getattr(self.agent, "current_task", None)
        activation_error = delegation_activation_error(self.agent, task)
        if activation_error:
            return activation_error

        tenant_id = self._parent_tenant_id()
        parent_tenant, parent_name = agent_identity_from_runtime(
            self.agent,
            parent_tenant=tenant_id,
        )
        requested_name = bare_delegation_name(requested_agent_id, tenant_id)
        if agent_identities_equal(
            parent_tenant,
            parent_name,
            parent_tenant,
            requested_name,
            scope_tenant=tenant_id,
        ):
            return "Self-delegation is not allowed"
        return None

    def _validate_target_permission(self, target_base_id: str) -> Optional[str]:
        return delegation_target_error(self.agent, target_base_id)

    def _validate_tenant_scope(self, target_agent: Any) -> Optional[str]:
        parent_tenant = self._parent_tenant_id()
        target_tenant = self._target_tenant_id(target_agent)
        if target_tenant == SYSTEM_TENANT_ID:
            return None
        if not target_tenant:
            return "Target agent tenant is unknown"
        if not parent_tenant:
            return "Parent agent tenant is unknown"
        if parent_tenant == target_tenant:
            return None
        return (
            f"Target agent tenant '{target_tenant}' is outside parent tenant "
            f"'{parent_tenant}'"
        )

    def _validate_delegation_chain(self, target_base_id: str) -> Optional[str]:
        chain = self._current_delegation_chain()
        if target_base_id in chain:
            return (
                "Delegation cycle is not allowed: agent "
                f"'{target_base_id}' is already in delegation chain"
            )
        return None

    def _current_delegation_chain(self) -> list[str]:
        task = getattr(self.agent, "current_task", None)
        if not isinstance(task, dict):
            return []

        context = TaskSchema.get_context(task)
        if not isinstance(context, dict):
            return []

        delegation = context.get("delegation")
        if not isinstance(delegation, dict):
            return []

        chain = delegation.get("chain")
        if isinstance(chain, list):
            return self._normalize_agent_chain(chain)

        legacy_chain = [
            delegation.get("parent_agent"),
            delegation.get("target_agent"),
        ]
        return self._normalize_agent_chain(legacy_chain)

    def _next_delegation_chain(self, target_base_id: str) -> list[str]:
        chain = self._current_delegation_chain()
        parent_base_id = base_agent_id(getattr(self.agent, "agent_id", ""))
        if parent_base_id:
            if parent_base_id in chain:
                chain = chain[: chain.index(parent_base_id) + 1]
            else:
                chain.append(parent_base_id)
        if target_base_id:
            chain.append(target_base_id)
        return chain

    @staticmethod
    def _normalize_agent_chain(agent_ids: Iterable[Any]) -> list[str]:
        normalized: list[str] = []
        for agent_id in agent_ids:
            base_id = base_agent_id(str(agent_id or ""))
            if base_id and base_id not in normalized:
                normalized.append(base_id)
        return normalized

    def _parent_tenant_id(self) -> Optional[str]:
        shared_context = getattr(self.agent, "shared_context", None)
        tenant_id = getattr(shared_context, "tenant_id", None)
        if tenant_id:
            return str(tenant_id)
        return self._configured_tenant_id(self.agent)

    def _target_tenant_id(self, target_agent: Any) -> Optional[str]:
        return self._configured_tenant_id(target_agent)

    @staticmethod
    def _configured_tenant_id(agent: Any) -> Optional[str]:
        config = getattr(agent, "config", None)
        if isinstance(config, dict) and config.get("tenant_id"):
            return str(config["tenant_id"])
        tenant_id = getattr(agent, "tenant_id", None)
        if tenant_id:
            return str(tenant_id)
        return None

    def _find_target_agent(self, requested_agent_id: str) -> Optional[Any]:
        return find_project_agent(self.agent, requested_agent_id)

    def _a2a_target_explicitly_allowed(
        self, requested_agent_id: str, tenant_id: Optional[str]
    ) -> bool:
        """True only if the a2a name is listed by name in allowed_delegation_targets.

        The '*' wildcard is deliberately NOT honored here: it expands to pool agents in
        the target catalog, and letting it also authorize an unlisted a2a server at call
        time would reopen the exact policy hole the named-only catalog closes.
        """
        normalized = normalize_delegation_targets(
            getattr(self.agent, "allowed_delegation_targets", None),
            tenant_id=tenant_id,
        )
        bare = bare_delegation_name(requested_agent_id, tenant_id)
        return bool(bare) and bare in normalized

    async def _maybe_delegate_to_a2a(
        self, requested_agent_id: str, instruction: str
    ) -> Optional[Dict[str, Any]]:
        """Delegate to a registered a2a server when the pool-miss target names one.

        Returns the a2a tool result, or None to fall through to the pool-not-found error
        (the name is neither a pool agent nor a usable, explicitly-allowed a2a target).
        Wraps the send in the same delegation events as the pool path so the UI treats
        both target kinds uniformly.
        """
        tenant_id = self._parent_tenant_id()
        if not tenant_id or not self._a2a_target_explicitly_allowed(
            requested_agent_id, tenant_id
        ):
            return None
        shared_context = getattr(self.agent, "shared_context", None)
        storage = getattr(shared_context, "storage", None)
        if storage is None or not callable(
            getattr(storage, "get_a2a_server_by_name", None)
        ):
            return None
        # Look up by the SAME bare name the allow-check used (both strip the @project
        # suffix). Otherwise a reference like "name@proj" would pass the allow-check but
        # miss the lookup and fall through to the generic pool-not-found error.
        lookup_name = bare_delegation_name(requested_agent_id, tenant_id)
        try:
            server = await storage.get_a2a_server_by_name(lookup_name, tenant_id)
        except Exception as exc:
            logger.warning(
                "[DELEGATION] a2a lookup failed parent=%s target=%s err=%s",
                getattr(self.agent, "agent_id", "unknown"),
                lookup_name,
                exc,
            )
            return None
        if not server:
            return None
        server_name = str(server.get("name") or requested_agent_id)
        if server.get("enabled") is False:
            # A disabled server is a KNOWN a2a target, not a missing one — return an explicit
            # typed error instead of falling through to the generic "not in agent_pool", so an
            # LLM retrying a name that worked yesterday gets an actionable reason.
            return {
                "status": "error",
                "target": server_name,
                "target_kind": "a2a",
                "error_type": "a2a_disabled",
                "error": f"a2a server '{server_name}' is disabled",
            }

        project_id = self._project_id()
        run_id = getattr(shared_context, "run_id", None)
        delegation_id = f"a2a_{uuid.uuid4().hex[:8]}"
        parent_agent = getattr(self.agent, "agent_id", "unknown")
        fallback_models_override = getattr(
            shared_context, "_ephemeral_fallback_models", None
        )
        if getattr(shared_context, "_force_model_override", False):
            fallback_models_override = []

        await self._emit(
            EventSchema.AGENT_DELEGATION_STARTED,
            {
                "project_id": project_id,
                "parent_agent": parent_agent,
                "child_agent": server_name,
                "requested_agent": requested_agent_id,
                "target_kind": "a2a",
                "task_id": delegation_id,
                "instruction": instruction[:200],
            },
        )
        logger.info(
            "[DELEGATION] a2a started project_id=%s parent=%s target=%s task_id=%s",
            project_id,
            parent_agent,
            server_name,
            delegation_id,
        )

        try:
            result = await delegate_to_a2a_agent(
                server,
                instruction,
                storage=storage,
                llm_client=getattr(self.agent, "llm_client", None),
                model=self.agent._resolve_project_model(),
                api_key_override=getattr(shared_context, "_ephemeral_api_key", None),
                fallback_models_override=fallback_models_override,
                tenant_id=tenant_id,
                project_id=project_id,
                run_id=run_id,
                delegation_id=delegation_id,
                message_store=getattr(shared_context, "message_store", None),
            )
        except Exception as exc:
            # The adapter returns typed errors for its own failure modes, but a truly
            # unexpected escape must still close the STARTED event with a FAILED — the pool
            # path wraps execute_task the same way. Without this the UI shows the delegation
            # forever "in progress".
            logger.exception(
                "[DELEGATION] a2a adapter raised parent=%s target=%s task_id=%s",
                parent_agent, server_name, delegation_id,
            )
            await self._emit(
                EventSchema.AGENT_DELEGATION_FAILED,
                {
                    "project_id": project_id,
                    "parent_agent": parent_agent,
                    "child_agent": server_name,
                    "target_kind": "a2a",
                    "task_id": delegation_id,
                    "status": "a2a_error",
                    "error": str(exc),
                },
            )
            return {
                "status": "error",
                "target": server_name,
                "target_kind": "a2a",
                "error_type": "a2a_error",
                "error": str(exc),
            }

        # Note: a2a delegations are intentionally NOT appended to _delegation_trajectory
        # (which the pool path fills and the AppFactory-154 reviewer seam reads) — a2a bypasses
        # the reviewer seam in this MVP. A pool delegation running after an a2a one is thus
        # reviewed without a2a context; giving a2a first-class trajectory/approval treatment
        # is deferred to AppFactory-181 (approval/HITL on the ledger).
        if result.get("status") == "success":
            await self._emit(
                EventSchema.AGENT_DELEGATION_COMPLETED,
                {
                    "project_id": project_id,
                    "parent_agent": parent_agent,
                    "child_agent": server_name,
                    "target_kind": "a2a",
                    "task_id": delegation_id,
                    "status": "completed",
                    "result": self._result_preview(result.get("result")),
                },
            )
        else:
            await self._emit(
                EventSchema.AGENT_DELEGATION_FAILED,
                {
                    "project_id": project_id,
                    "parent_agent": parent_agent,
                    "child_agent": server_name,
                    "target_kind": "a2a",
                    "task_id": delegation_id,
                    "status": result.get("error_type"),
                    "error": result.get("error", ""),
                },
            )
        logger.info(
            "[DELEGATION] a2a finished parent=%s target=%s task_id=%s status=%s",
            parent_agent,
            server_name,
            delegation_id,
            result.get("status"),
        )
        return result

    def _create_sub_task(
        self,
        project_id: str,
        target_base_id: str,
        instruction: str,
        retry_feedback: Optional[str] = None,
        review_decision: Optional[str] = None,
        previous_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        parent_task = getattr(self.agent, "current_task", {}) or {}
        extra: Dict[str, Any] = {}
        if retry_feedback:
            # Consumed on the sub-agent's prompt by BaseAgent._append_retry_feedback
            # (TaskSchema.get_retry_feedback) — drives the planner's re-plan loop.
            extra[TaskSchema.RETRY_FEEDBACK] = retry_feedback
        context = {
            "delegation": {
                "parent_agent": getattr(self.agent, "agent_id", "unknown"),
                "target_agent": target_base_id,
                "chain": self._next_delegation_chain(target_base_id),
                "parent_task_id": TaskSchema.get_id(parent_task)
                if isinstance(parent_task, dict)
                else "",
            }
        }
        if review_decision:
            review_context = {
                "decision": review_decision,
                "feedback": retry_feedback or "",
            }
            if previous_result is not None:
                review_context["previous_output"] = ResultSchema.get_output(
                    previous_result
                )
            context["agent_result_review"] = review_context

        return TaskSchema.create(
            task_id=f"delegation_{uuid.uuid4().hex[:8]}",
            project_id=project_id,
            task_type=DELEGATED_TASK_TYPE,
            description=instruction,
            can_delegate=parent_task.get("can_delegate") is True,
            writes=[],
            context=context,
            **extra,
        )

    def _project_id(self) -> str:
        shared_context = getattr(self.agent, "shared_context", None)
        project_id = getattr(shared_context, "project_id", None)
        if project_id:
            return str(project_id)
        task = getattr(self.agent, "current_task", None)
        if isinstance(task, dict):
            return TaskSchema.get_project_id(task) or "unknown"
        return "unknown"

    async def _emit_failed(
        self,
        project_id: str,
        target_id: str,
        target_name: str,
        sub_task: Dict[str, Any],
        error: str,
        child_status: Optional[str] = None,
    ) -> None:
        payload = {
            "project_id": project_id,
            "parent_agent": getattr(self.agent, "agent_id", "unknown"),
            "child_agent": target_id,
            "child_agent_name": target_name,
            "task_id": TaskSchema.get_id(sub_task),
            "error": error,
        }
        if child_status:
            payload["status"] = child_status
        await self._emit(EventSchema.AGENT_DELEGATION_FAILED, payload)

    async def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        emitter = getattr(self.agent, "event_emitter", None)
        if not emitter:
            return
        # EventEmitter.emit signature is (event_type, run_id, data); data MUST carry a
        # real project_id (strict contract). The previous 2-arg call silently raised
        # TypeError (swallowed below), so delegation events never reached SSE.
        shared_context = getattr(self.agent, "shared_context", None)
        run_id = getattr(shared_context, "run_id", None)
        pid = payload.get("project_id") or self._project_id()
        payload["project_id"] = pid
        if not pid or pid == "unknown":
            # Do not push a contract-violating event to a bogus SSE bucket; the
            # emitter's strict guard rejects None/"global" but lets "unknown" slip.
            logger.warning("[DELEGATION] skip %s emit: unresolved project_id", event_type)
            return
        try:
            await emitter.emit(event_type, run_id, payload)
        except Exception as exc:
            logger.warning(
                "[DELEGATION] event_emit_failed event=%s parent=%s error=%s",
                event_type,
                getattr(self.agent, "agent_id", "unknown"),
                exc,
            )

    @staticmethod
    def _status_value(status: Any) -> str:
        if isinstance(status, TaskStatus):
            return status.value
        return str(status or "")

    @staticmethod
    def _result_preview(output: Any, limit: int = 2000) -> Any:
        """Truncated, event-safe view of a sub-agent's output for the completed event.

        Small outputs (mocks, short reports) pass through unchanged; large ones are
        serialized and clipped so delegation events stay light on the SSE stream.
        """
        try:
            text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, default=str)
        except Exception:
            text = str(output)
        if len(text) > limit:
            return text[:limit] + f"… (+{len(text) - limit} chars)"
        return output
