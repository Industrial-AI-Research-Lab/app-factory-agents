"""
Workflow Engine — fully dynamic DAG executor for AppFactory workflows.

Reads a workflow definition (nodes + edges) from MongoDB and walks the
directed graph from *start* to *end*.  Node types:

- **start / end**      – entry / exit markers
- **phase**            – runs an agent via PhaseRunner (auction or direct).
                         Configured by ``task_type`` or ``description`` plus
                         optional prompt-driven metadata like ``reads`` /
                         ``writes`` / ``output_schema``.
- **approval_gate**    – generic pause: request human yes/no via
                         ApprovalManager.  No hardcoded approval type.
- **execution**        – parallel task execution via TaskExecutor.
- **deploy**           – deploy flow: check env vars → request deploy
                         approval → execute via GenericAgent (deploy config).

The engine knows NOTHING about specific phases.  New workflows
(with arbitrary phases/agents) can be added in MongoDB without
touching Python code.
"""

from __future__ import annotations

from copy import deepcopy
import asyncio
import json
import logging
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any, AsyncIterator, Collection, Dict, List, Optional

from context.shared_context import FULL_CONTEXT_READS_TOKEN, WORKFLOW_DEFAULT_READS_TOKEN
from schemas.configuration_schemas import normalize_workflow_execution_mode
from integrations import A2AClientFactory
from integrations.a2a_artifacts import (
    a2a_artifact_paths,
    a2a_artifact_scope,
    a2a_artifact_to_files,
    a2a_response_text,
    dedupe_a2a_path,
    merge_a2a_artifact_update,
)
from integrations.a2a_params import A2AParamExtractionError, extract_extension_dataparts
from integrations.a2a_contracts import A2AContractError, validate_agent_card
from integrations.a2a_output_modes import A2AOutputModeError
from integrations.a2a_progress import extract_coscientist_progress, status_update_text
from integrations.checkpoint_client import CheckpointError
from a2a.utils.errors import TaskNotFoundError
from schemas import TaskSchema, ResultSchema, ApprovalStatus, EventSchema, TaskStatus
from orchestration.agent_result_review import (
    AgentResultDecision,
    build_agent_result_interaction_schema,
    decision_from_approval,
)
from orchestration.cancellation import is_genuinely_cancelled
from orchestration.failure_messages import build_failure_chat_message
from orchestration.phase_runner import _resolve_direct_agent
from orchestration.report_publish_node import run_report_publish_node
from orchestration.map_node import MapNodeRunner, failure_for_log
from orchestration.map_node_contract import map_node_errors
from orchestration.map_node_progress import MapProgress
from orchestration.tool_node import ToolNodeRunner
from orchestration.tool_node_contract import binding_errors, tool_node_errors
from orchestration.validator_bindings import (
    bound_field_rejection,
    node_bound_outputs,
    validator_bound_outputs,
)
from orchestration.validators import ValidatorRunner
from orchestration.workflow_approval import publish_workflow_approval
from orchestration.workflow_next_step import (
    approval_next_steps,
    find_conditioned_edge,
    find_default_edge,
    next_workflow_step,
    node_label,
)
from storage.checkpoint_store import CheckpointConflict
from telemetry.run_context import current_run_traceparent
from telemetry.run_scope import run_trace_scope

logger = logging.getLogger(__name__)
_MISSING = object()

# AppFactory-280: a2a_agent node states in which the external MAS task is still
# in flight — anything else (COMPLETED/FAILED/CANCELED/INPUT_REQUIRED/REJECTED/
# AUTH_REQUIRED/an unrecognized value) is terminal for polling purposes, even
# though not all of those are a *success*; the existing post-loop logic already
# decides success/failure from the terminal state, unchanged by polling.
_A2A_NON_TERMINAL_STATES = {
    "TASK_STATE_UNSPECIFIED", "TASK_STATE_SUBMITTED", "TASK_STATE_WORKING", None,
}


def _a2a_task_is_terminal(event: Dict[str, Any]) -> bool:
    """Whether a submit/poll event ends the poll loop.

    A bare Message reply (no Task) is inherently terminal — return_immediately
    only ever produces one when the agent finished before the server needed to
    answer, so there is nothing left to poll.
    """
    if event.get("type") != "task":
        return True
    state = ((event.get("data") or {}).get("status") or {}).get("state")
    return state not in _A2A_NON_TERMINAL_STATES


class _A2APollExhaustedError(Exception):
    """tasks/get failed too many times in a row — the adapter may be alive but
    unreachable long enough that continuing to retry in-process stops being
    reasonable. Distinct from TaskNotFoundError: here the task might still be
    running on the MAS side, we just can't currently confirm it."""


class _A2ASubmissionOutcomeUnknownError(Exception):
    """submit_and_track raised before task_id was confirmed — the adapter may or
    may not have created the task, and core A2A JSON-RPC gives the platform no
    way to check. Carries the message_id so the caller can close the right
    pending_submit cursor without re-deriving it."""

    def __init__(self, message_id: str, original: Exception):
        self.message_id = message_id
        self.original = original
        super().__init__(
            f"submit_and_track failed before task_id was confirmed for "
            f"message_id={message_id}: {original}"
        )


class _A2AAnswerReconciliationUnknownError(Exception):
    """The human answer's delivery status could not be confirmed — covers TWO
    distinct moments (AppFactory-281 P1 review fixes, bug #3 2nd follow-up and
    8th finding):
      - the pre-send reconciliation check itself failed (reconcile_before_resend's
        tasks/get, repeated network errors, or the task not found) — nothing
        was (re)dispatched yet;
      - a FRESH continue_task call itself raised.

    These two used to be split (the second one raised a separate
    _A2AInputAnswerRejectedError that closed the cursor as a definite typed
    rejection) — that was wrong: continue_task shares the exact same
    _send_once_and_track transport as the ORIGINAL submission, where an
    identical exception is already treated as unknown-not-rejected
    (_A2ASubmissionOutcomeUnknownError above). "Request delivered, response
    lost" is exactly as possible here as it is there — an exception is not
    proof the adapter never received the answer. Treating it as a definite
    rejection closed the cursor on an unproven guess, permanently failing a
    run over what might be a transient network blip — the exact failure mode
    this whole mechanism exists to rule out (a network hiccup right after a
    human finally answers a multi-hour run must not be able to kill it). A
    genuine terminal refusal from the adapter still surfaces correctly:
    continue_task succeeds (no exception) and the task itself then settles at
    TASK_STATE_FAILED/CANCELED, handled by the ordinary unexpected_final_state
    branch below (AC4).

    Distinct from _A2APollExhaustedError: that one gives up on the whole task
    and closes the cursor (reasonable once a decision to poll was already made
    and dispatched). Here nothing has been (re)dispatched yet, or its outcome
    is unknown — so failing closed must NOT close the cursor: it is left
    exactly as it was (pending_human_answer and answer_dispatch_started_at
    both still set), so the next resume attempt rebuilds the same
    reconcile_before_resend marker and tries to reconcile again once the
    adapter/network recovers."""

    def __init__(self, task_id: str, reason: str):
        self.task_id = task_id
        self.reason = reason
        super().__init__(
            f"Could not reconcile whether the human answer already reached "
            f"the adapter for task_id={task_id}: {reason}"
        )


def _get_phase_label(node: dict) -> Optional[str]:
    """Return the explicit phase label configured on the workflow node."""
    phase_label = (node.get("phase_label") or "").strip()
    return phase_label or None


class WorkflowEngine:
    """Execute a workflow DAG using generic managers (PhaseRunner, ApprovalManager, TaskExecutor)."""

    _MAX_ITERATIONS = 50

    # AppFactory-280: consecutive tasks/get failures (network errors, not a definitive
    # "task not found") tolerated before giving up on a long_running a2a_agent node.
    # Backoff doubles each failure starting from the server's poll_interval_seconds,
    # capped at _A2A_POLL_BACKOFF_CAP_SECONDS — worst case before giving up is on the
    # order of tens of minutes, not hours: a real restart is recovered by
    # ensure_workflow_running picking the durable cursor back up (no cap there), this
    # cap is only for transient trouble the SAME in-process poll loop can ride out.
    _A2A_MAX_CONSECUTIVE_POLL_ERRORS = 8
    _A2A_POLL_BACKOFF_CAP_SECONDS = 300

    def __init__(self, orchestrator):
        self.orch = orchestrator
        self.phase_runner = orchestrator.phases
        self.approvals = orchestrator.approvals
        self.task_executor = orchestrator.tasks
        self.event_emitter = orchestrator.event_emitter
        self.snapshot_manager = orchestrator.snapshot_manager
        self.tracer = orchestrator.tracer

    # ── public API ───────────────────────────────────────────────

    async def execute(
        self,
        project_id: str,
        workflow_def: dict,
        start_node_id: Optional[str] = None,
        initial_result: Optional[dict] = None,
    ) -> dict:
        project = self.orch.active_projects.get(project_id) or {}
        async with run_trace_scope(
            self.orch.storage, self.tracer, project_id, project.get("run_id")
        ):
            with self.tracer.start_span(
                "workflow.execute", {"project.id": project_id}
            ):
                return await self._execute(
                    project_id, workflow_def, start_node_id, initial_result
                )

    async def _execute(
        self,
        project_id: str,
        workflow_def: dict,
        start_node_id: Optional[str] = None,
        initial_result: Optional[dict] = None,
    ) -> dict:
        """Walk the DAG from *start* (or custom node) to *end* and return the final result."""
        try:
            self.validate_dag(workflow_def)
        except ValueError as e:
            # A malformed DAG here (e.g. a bundle-imported a2a node the route validator
            # missed) must fail the run cleanly. An uncaught raise reaches the background
            # task, which only emits project_failed and leaves the run "running" forever.
            # Route it through the same failure path as a preflight failure below.
            failure = {"status": "failed", "error": str(e), "error_type": "invalid_workflow"}
            await self._handle_workflow_failure(project_id, failure)
            return failure

        # Fail fast only on a genuine fresh start (no explicit start node). On resume,
        # rehydration re-enters execute() at the parked gate, where the referenced A2A
        # nodes may already be in the project's past; re-validating them would let an
        # unrelated server outage or admin-disable kill an already-progressed project and
        # swallow its pending approval (preflight returns before the walk reaches the
        # gate). A server that's still needed is re-checked at the node's own send path.
        # `not start_node_id` mirrors the `start_node_id or _find_start_node_id` walk below.
        if not start_node_id:
            await self._get_project(project_id)["shared_context"].record_workflow_approval(None)
            preflight_failure = await self._preflight_a2a_servers(project_id, workflow_def)
            if preflight_failure:
                await self._handle_workflow_failure(project_id, preflight_failure)
                return preflight_failure

        nodes_map: Dict[str, dict] = {n["id"]: n for n in workflow_def["nodes"]}
        edges: List[dict] = workflow_def["edges"]

        current_id = start_node_id or self._find_start_node_id(nodes_map)
        if current_id not in nodes_map:
            raise ValueError(f"Unknown workflow node id: {current_id}")
        last_result: Optional[dict] = initial_result
        current_phase_label: Optional[str] = None
        reject_feedback: Optional[str] = None
        # Tracks whether reject_feedback came from a human gate (vs a validator):
        # the human-recovery block below must not fire for machine rejects — a
        # validator will just re-reject the unrevised result.
        reject_feedback_from_gate = False
        # Per-validator reject→retry counter (AppFactory-77 F3): keyed by validator node id,
        # persists for the whole run (no reset on a later approve) so the cap is a hard
        # per-validator LLM-spend ceiling, not a per-streak one.
        validator_reject_counts: dict[str, int] = {}
        iterations = 0
        # Resolved once per execute() call (not per iteration): a workflow-def
        # edit mid-run shouldn't retroactively change the budget for a run
        # already in flight.
        max_iterations = self._resolve_max_iterations(workflow_def)
        run_deadline = self._resolve_run_deadline(workflow_def)  # monotonic seconds, or None (guard off)

        logger.info("[ENGINE] Starting workflow '%s' for project %s",
                     workflow_def.get("_id", "?"), project_id)

        while iterations < max_iterations:
            iterations += 1
            node = nodes_map[current_id]
            node_type = node["type"]

            if run_deadline is not None and time.monotonic() >= run_deadline:
                # Loop-top check: node types with no wrapped await below (start,
                # end, unrecognized) cost nothing to process, so without this a
                # run could spin through them past the deadline until
                # max_iterations instead of stopping promptly. A node that DOES
                # await below (including a parked gate — ADR-0012) is caught by
                # the wait_for wrapping instead, since this check only runs
                # between nodes.
                return await self._fail_on_run_timeout(project_id, workflow_def, current_id)

            # ── end ──
            if node_type == "end":
                if current_phase_label:
                    await self._emit_phase_event(current_phase_label, project_id, "completed")
                    await self._create_snapshot(project_id, current_phase_label)
                await self._finalize_workflow(project_id, last_result)
                logger.info("[ENGINE] Reached end node for project %s", project_id)
                break

            # ── phase group transitions (auto emit started/completed) ──
            new_label = node.get("phase_label")
            if new_label and new_label != current_phase_label:
                if current_phase_label:
                    await self._emit_phase_event(current_phase_label, project_id, "completed")
                    await self._create_snapshot(project_id, current_phase_label)
                current_phase_label = new_label
                await self._emit_phase_event(current_phase_label, project_id, "started")

            # ── run the node (work, not routing) ──
            # Capture before consume: a phase reached via a gate's rejected edge
            # carries reject_feedback. Used below to keep a human reject loop
            # recoverable instead of fatal.
            entered_via_reject = (
                node_type == "phase" and reject_feedback is not None and reject_feedback_from_gate
            )
            if node_type == "map" and reject_feedback is not None:
                # Reject feedback revises every item, so none of the saved ones is reused.
                await self._forget_map_progress(project_id, node["id"])
            if node_type in (
                "phase", "approval_gate", "execution", "deploy", "a2a_agent", "validator", "tool",
                "map",
            ):
                if run_deadline is None:
                    node_result = await self._execute_node(
                        node, project_id, workflow_def, prev_result=last_result,
                        reject_feedback=reject_feedback,
                    )
                else:
                    # wait_for is the actual guarantee: the loop-top check alone
                    # can't interrupt a node await that never returns.
                    remaining = max(0.001, run_deadline - time.monotonic())
                    try:
                        node_result = await asyncio.wait_for(
                            self._execute_node(
                                node, project_id, workflow_def, prev_result=last_result,
                                reject_feedback=reject_feedback,
                            ),
                            timeout=remaining,
                        )
                    except asyncio.TimeoutError:
                        return await self._fail_on_run_timeout(project_id, workflow_def, current_id)
                reject_feedback = None  # consume-once: only the rejected edge's target sees it
                reject_feedback_from_gate = False
            else:
                node_result = None

            # ── post-execution: failure handling + edge resolution ──
            if node_type == "phase":
                last_result = node_result
                logger.info("[ENGINE] Phase '%s' result: %s",
                             node["id"],
                             last_result.get("status") if last_result else "None(ok)")

                if self._is_failure(last_result):
                    retry_id = self._find_edge(current_id, "rejected", edges)
                    if retry_id:
                        current_id = retry_id
                        continue
                    if entered_via_reject:
                        # A human rejection sent us back here and the agent could
                        # not satisfy the phase contract within max_retries. Re-show
                        # the gate (the phase's default edge) instead of killing the
                        # project: the human stays the progress signal and can reject
                        # again or accept the unrevised result. Drop the FAILED result
                        # so a later _finalize_workflow can still complete the run.
                        gate_id = self._find_default_edge(current_id, edges)
                        if gate_id:
                            logger.warning(
                                "[ENGINE] Phase '%s' failed on a post-reject re-run; "
                                "re-showing gate '%s' instead of failing project %s",
                                current_id, gate_id, project_id,
                            )
                            last_result = {
                                "status": "completed",
                                "node_id": current_id,
                                "revised": False,
                                "note": "agent could not revise after reject; "
                                        "showing prior result for re-review",
                            }
                            current_id = gate_id
                            continue
                    await self._handle_workflow_failure(project_id, last_result)
                    return last_result

                next_id = self._find_default_edge(current_id, edges)

            elif node_type == "approval_gate":
                gate_result = (node_result or {}).get("status")
                logger.info("[ENGINE] Gate '%s' result: %s", node["id"], gate_result)

                if gate_result == "timeout":
                    return {"status": "waiting_approval_timeout"}

                if self._is_failure(node_result):
                    await self._handle_workflow_failure(project_id, node_result)
                    return node_result

                if gate_result == "approved" and "output" in (node_result or {}):
                    last_result = {**node_result, "status": TaskStatus.COMPLETED}

                if gate_result == "rejected":
                    review_decision = (node_result or {}).get("decision")
                    reject_feedback = self._build_reject_feedback(
                        node,
                        (node_result or {}).get("reason"),
                        decision=review_decision,
                    )
                    reject_feedback_from_gate = True
                    if review_decision == AgentResultDecision.REJECT.value:
                        last_result = None

                next_id = self._find_edge(current_id, gate_result, edges)
                if next_id is None:
                    next_id = self._find_default_edge(current_id, edges)

            elif node_type == "execution":
                last_result = node_result
                logger.info("[ENGINE] Execution result: %s",
                             last_result.get("status") if last_result else "None")

                if last_result and last_result.get("status") == "cancelled":
                    return last_result

                if self._is_failure(last_result):
                    await self._handle_workflow_failure(project_id, last_result)
                    return last_result

                next_id = self._find_default_edge(current_id, edges)

            elif node_type == "deploy":
                last_result = node_result
                logger.info("[ENGINE] Deploy result: %s",
                             last_result.get("status") if last_result else "None")
                next_id = self._find_default_edge(current_id, edges)

            elif node_type == "tool":
                last_result = node_result
                if last_result.get("status") == "cancelled":
                    return last_result
                if self._is_failure(last_result):
                    # A write whose outcome is unknown may have happened; a retry
                    # path could repeat it, so the run stops here instead.
                    retry_id = (
                        None
                        if last_result.get("outcome_unknown")
                        else self._find_edge(current_id, "rejected", edges)
                    )
                    if retry_id:
                        current_id = retry_id
                        continue
                    await self._handle_workflow_failure(project_id, last_result)
                    return last_result
                next_id = self._find_default_edge(current_id, edges)

            elif node_type == "map":
                last_result = node_result
                if last_result.get("status") == "cancelled":
                    return last_result
                if self._is_failure(last_result):
                    retry_id = self._find_edge(current_id, "rejected", edges)
                    if retry_id:
                        current_id = retry_id
                        continue
                    await self._handle_workflow_failure(project_id, last_result)
                    return last_result
                # Item tasks suppress their assistant finals, so without this
                # marker a restart would take the last item for an interrupted one.
                await self._mark_node_attempt_completed(node, project_id, own_attempt_only=True)
                next_id = self._find_default_edge(current_id, edges)

            elif node_type == "a2a_agent":
                last_result = node_result
                logger.info(
                    "[ENGINE] A2A agent '%s' result: %s",
                    node["id"],
                    last_result.get("status") if last_result else "None",
                )

                if (last_result or {}).get("status") == "paused":
                    # AppFactory-281: input_required parked the node on a durable
                    # a2a_task_state cursor — not a failure (don't route through
                    # _handle_workflow_failure) and not a success either (must NOT
                    # fall through to _find_default_edge, which would silently walk
                    # the DAG to "end" as if the node had completed while a human
                    # question is still open). Stop here, same as an approval_gate's
                    # "timeout" returning early above — the run stays "running" with
                    # no live task; ensure_workflow_running resumes it once answered.
                    return last_result

                if (last_result or {}).get("status") == "external_waiting":
                    logger.info(
                        "[ENGINE] project_id=%s node_id=%s — parked on external run %s",
                        project_id,
                        current_id,
                        last_result.get("external_run_id"),
                    )
                    return last_result

                if (last_result or {}).get("status") == "reconciliation_pending":
                    # AppFactory-281 P1 review fix (bug #3, 4th follow-up): a crash-
                    # mid-dispatch reconciliation couldn't confirm delivery (tasks/get
                    # itself kept failing) — the a2a_task_state cursor was
                    # deliberately left open for a retry on the next resume. Must
                    # not route through _handle_workflow_failure (would flip
                    # project/run to terminal "failed" and permanently block
                    # _maybe_resume_interrupted_a2a's own terminal-status guard).
                    # Same non-terminal park as "paused"/"external_waiting" above.
                    logger.info(
                        "[ENGINE] project_id=%s node_id=%s — A2A answer "
                        "reconciliation unresolved, parked for retry on next resume",
                        project_id,
                        current_id,
                    )
                    return last_result

                if self._is_failure(last_result):
                    retry_id = self._find_edge(current_id, "rejected", edges)
                    if retry_id:
                        logger.info("[ENGINE] A2A agent failed, retrying via rejected edge to '%s'", retry_id)
                        current_id = retry_id
                        continue
                    await self._handle_workflow_failure(project_id, last_result)
                    return last_result

                next_id = self._find_default_edge(current_id, edges)

            elif node_type == "validator":
                validator_status = (node_result or {}).get("status")
                logger.info(
                    "[ENGINE] project_id=%s node_id=%s - validator result: %s",
                    project_id, node["id"], validator_status,
                )

                if validator_status == "failed":
                    # A broken check config (e.g. unresolvable $ref) is not something the
                    # agent can fix by retrying — hard-fail here instead of falling into
                    # the rejected-edge branch below, which would just loop to the
                    # iteration cap re-hitting the same unresolvable schema every time.
                    last_result = node_result
                    await self._handle_workflow_failure(project_id, last_result)
                    return last_result
                elif validator_status == "rejected":
                    bound_failure = self._bound_field_rejection(node, node_result or {})
                    if bound_failure is not None:
                        last_result = bound_failure
                        await self._handle_workflow_failure(project_id, last_result)
                        return last_result
                    next_id = self._find_edge(current_id, "rejected", edges)
                    if next_id is None:
                        # A failed format assert must never continue forward —
                        # no retry path means the run dies loudly (unlike gates,
                        # which fall through to the default edge on reject).
                        last_result = {
                            "status": "failed",
                            "reason": "validator_rejected_no_retry_path",
                            "error": (node_result or {}).get("feedback")
                            or f"Validator '{node['id']}' rejected",
                            "node_id": node["id"],
                            "validator_errors": (node_result or {}).get("error_details"),
                        }
                        await self._handle_workflow_failure(project_id, last_result)
                        return last_result
                    # Per-validator retry cap (AppFactory-77 F3, ADR-0011): a validator
                    # stuck rejecting must fail loudly at its own configured budget
                    # rather than burning LLM calls up to the global max_iterations.
                    # cap=N allows N rejections that retry; the (N+1)th fails the run.
                    cap = self._resolve_validator_retry_cap(node)
                    if cap is not None:
                        count = validator_reject_counts.get(current_id, 0) + 1
                        validator_reject_counts[current_id] = count
                        if count > cap:
                            last_result = {
                                "status": "failed",
                                "reason": "validator_retry_limit_exhausted",
                                "error": (
                                    f"Validator '{node['id']}' still rejecting after "
                                    f"{cap} retr{'y' if cap == 1 else 'ies'}: "
                                    + ((node_result or {}).get("feedback") or "")
                                ).strip(),
                                "node_id": node["id"],
                                "validator_errors": (node_result or {}).get("error_details"),
                            }
                            await self._handle_workflow_failure(project_id, last_result)
                            return last_result
                    # Same consume-once channel the gates use: the phase reached
                    # via the rejected edge reads this feedback and revises.
                    reject_feedback = (node_result or {}).get("feedback") or ""
                else:
                    next_id = self._find_edge(current_id, "approved", edges)
                    if next_id is None:
                        next_id = self._find_default_edge(current_id, edges)

            else:
                # start / unknown
                next_id = self._find_default_edge(current_id, edges)

            if next_id is None:
                logger.warning("[ENGINE] No outgoing edge from '%s' — stopping", current_id)
                break
            current_id = next_id
        else:
            logger.error("[ENGINE] Hit max iterations (%d)", max_iterations)
            # Route through _handle_workflow_failure (like the run_timeout guard):
            # emit PROJECT_FAILED + cascade project/run status to failed. Without
            # this the returned dict is swallowed upstream (run_workflow reacts
            # only to 'completed'; the control.py/projects.py wrappers catch only
            # exceptions), leaving the project stuck — neither completed nor
            # failed, read as "hung" in the UI (AppFactory-157).
            return await self._fail_on_max_iterations(
                project_id, current_id, max_iterations
            )

        return last_result or {"status": "completed", "project_id": project_id}

    # ── phase execution (generic) ────────────────────────────────

    async def _run_phase_node(
        self,
        node: dict,
        project_id: str,
        prev_result: Optional[dict],
        default_reads: Optional[List[str]] = None,
        reject_feedback: Optional[str] = None,
        execution_mode: str = "dynamic",
    ) -> Optional[dict]:
        """Run a phase and enforce declared writes before the workflow can advance.

        `reject_feedback` (a human gate-rejection reason) seeds the first
        attempt's retry feedback; writes-contract retries then build their
        own feedback for subsequent attempts.
        """
        writes = self._get_declared_writes(node)
        if not writes:
            result = await self._run_phase_attempt(
                node,
                project_id,
                prev_result,
                default_reads=default_reads,
                retry_feedback=reject_feedback,
                execution_mode=execution_mode,
            )
            # No writes-contract to enforce: the phase is done once the attempt
            # returns without failing. A gated phase suppresses its assistant
            # final, so record the durable completion marker here — the agent no
            # longer marks at turn-end (see _mark_gated_phase_completed / F2).
            if not self._is_failure(result):
                await self._mark_gated_phase_completed(node, project_id)
            return result

        baseline = await self._capture_writes_baseline(project_id, writes)
        # On a resumed attempt the baseline is re-read AFTER the pre-crash
        # context_writes already landed, so an untouched key reads as
        # "unchanged" and false-fails the contract. The journal is the durable
        # proof the resumed attempt wrote it: those keys count as satisfied.
        resume_written = await self._resumed_attempt_written_keys(
            project_id, node["id"], writes
        )
        max_attempts = self._get_max_phase_attempts(node)
        retry_feedback = reject_feedback
        last_result: Optional[dict] = None

        for attempt in range(1, max_attempts + 1):
            last_result = await self._run_phase_attempt(
                node,
                project_id,
                prev_result,
                default_reads=default_reads,
                retry_feedback=retry_feedback,
                contract_attempt=attempt,
                execution_mode=execution_mode,
            )

            if self._is_failure(last_result):
                return last_result

            contract = await self._evaluate_writes_contract(
                project_id, writes, baseline, satisfied_keys=resume_written
            )
            if contract["fulfilled"]:
                logger.info(
                    "[ENGINE] project_id=%s node_id=%s writes=%s attempt=%s - writes contract fulfilled",
                    project_id,
                    node["id"],
                    writes,
                    attempt,
                )
                # Mark the gated phase complete ONLY now — after its declared
                # writes actually landed. The old agent-side marker fired at the
                # attempt's turn-end, before this check, so a contract-FAILED or
                # interrupted attempt looked done to restart recovery: it gated a
                # phase whose outputs never materialized and dropped the retry (F2).
                await self._mark_gated_phase_completed(node, project_id)
                return last_result

            retry_feedback = self._build_writes_retry_feedback(
                node_id=node["id"],
                writes=writes,
                contract=contract,
                attempt=attempt,
                max_attempts=max_attempts,
            )
            if attempt < max_attempts:
                logger.warning(
                    "[ENGINE] project_id=%s node_id=%s attempt=%s/%s missing=%s unchanged=%s - retrying phase",
                    project_id,
                    node["id"],
                    attempt,
                    max_attempts,
                    contract["missing"],
                    contract["unchanged"],
                )
                continue

            logger.error(
                "[ENGINE] project_id=%s node_id=%s attempts=%s writes=%s - writes contract failed",
                project_id,
                node["id"],
                max_attempts,
                writes,
            )
            return self._create_writes_contract_failure(
                node_id=node["id"],
                writes=writes,
                contract=contract,
                attempts=max_attempts,
                feedback=retry_feedback,
            )

        return last_result

    async def _mark_gated_phase_completed(self, node: dict, project_id: str) -> None:
        """Durably record a finished gated phase in the run's completed_attempts.

        A gated phase suppresses the assistant-final that otherwise tells restart
        recovery an attempt finished, so this marker stands in for it (ADR-0009,
        find_interrupted_attempt). Two rules keep it honest: callers invoke it
        only AFTER the writes-contract is satisfied, and it records the exact
        cursor recovery keys on — the run's last workflow-node tool_call task_id —
        so "phase finished" and "recovery treats it as finished" cannot diverge.
        Best-effort: a missed marker degrades to a possible re-run, never a crash.
        """
        if not TaskSchema.should_suppress_assistant_message(node):
            return
        await self._mark_node_attempt_completed(node, project_id)

    async def _mark_node_attempt_completed(
        self, node: dict, project_id: str, *, own_attempt_only: bool = False
    ) -> None:
        """`own_attempt_only` marks nothing when the run's last node-bearing
        tool call belongs to another node: marking an earlier node's attempt
        would stop recovery from re-entering where the run really was."""
        # The try wraps the project/store lookups, not just the marker write:
        # _get_project raises on a vanished project and the journal read can
        # hiccup — neither may crash node completion: a missed marker only
        # risks a re-run.
        try:
            project = self._get_project(project_id)
            sc = project.get("shared_context") if isinstance(project, dict) else None
            store = getattr(sc, "message_store", None) if sc else None
            storage = getattr(sc, "storage", None) if sc else None
            run_id = getattr(sc, "run_id", None) if sc else None
            marker = getattr(storage, "add_run_completed_attempt", None)
            if not (store and run_id and callable(marker)):
                return

            from agents.run_rehydration import last_node_attempt

            attempt = await last_node_attempt(store, project_id, run_id)
            if attempt is None:
                return
            if own_attempt_only and attempt["workflow_node_id"] != node.get("id"):
                logger.info(
                    "[ENGINE] project_id=%s node_id=%s last_node=%s — last node attempt"
                    " is another node's; no completion marker",
                    project_id, node.get("id"), attempt["workflow_node_id"],
                )
                return
            await marker(run_id, attempt["task_id"])
        except Exception as e:
            logger.warning(
                "[ENGINE] project_id=%s node_id=%s — completed-attempt marker failed: %s",
                project_id, node.get("id"), e,
            )

    async def _run_phase_attempt(
        self,
        node: dict,
        project_id: str,
        prev_result: Optional[dict],
        default_reads: Optional[List[str]] = None,
        retry_feedback: Optional[str] = None,
        contract_attempt: Optional[int] = None,
        execution_mode: str = "dynamic",
    ) -> Optional[dict]:
        """Run a phase via PhaseRunner — auction or direct, based on node config."""
        project = self._get_project(project_id)
        agents = project.get("agents", self.orch.agent_pool)
        token = project.get("token")

        # One-shot pop of the restart-resume marker (stamped by
        # ensure_workflow_running's mid-phase branch, mirroring
        # _rehydration_approval_id): only the node whose attempt was
        # interrupted may adopt it — later phases and reject-loop revisits
        # run fresh.
        resume_attempt = None
        marker = project.pop("_resume_attempt", None) if isinstance(project, dict) else None
        if marker:
            if marker.get("node_id") == node["id"]:
                resume_attempt = marker
            else:
                logger.warning(
                    "[REHYDRATE] project_id=%s resume marker for node %s reached node %s — discarded",
                    project_id, marker.get("node_id"), node["id"],
                )

        task_type = node.get("task_type") or node["id"]
        description = node.get("description", "")
        agent_selection = node.get("agent_selection", "auction")
        phase_metadata = self._build_phase_task_metadata(
            node,
            default_reads=default_reads,
            execution_mode=execution_mode,
        )
        phase_metadata["context"] = {
            **(phase_metadata.get("context") or {}),
            "project_id": project_id,
            "run_id": project.get("run_id"),
        }
        if contract_attempt is not None:
            phase_metadata["contract_attempt"] = contract_attempt
        if retry_feedback:
            phase_metadata[TaskSchema.RETRY_FEEDBACK] = retry_feedback

        logger.info("[ENGINE] Running phase '%s' (task_type=%s, selection=%s) for %s",
                     node["id"], task_type, agent_selection, project_id)

        if agent_selection == "direct":
            agent_type = str(node.get("agent_type") or "").strip()
            if not agent_type:
                logger.warning(
                    "[ENGINE] Phase node '%s' uses direct selection but has no agent_type",
                    node["id"],
                )
                raise ValueError(
                    f"Phase node '{node['id']}' uses direct selection but has no agent_type"
                )
            phase_label = _get_phase_label(node)
            task = TaskSchema.create(
                task_id=f"{project_id}_{node['id']}_{uuid.uuid4().hex[:8]}",
                project_id=project_id,
                task_type=task_type,
                description=description,
                **phase_metadata,
            )
            # Pass previous phase output as context for the agent
            if prev_result is not None:
                prev_output = ResultSchema.get_output(prev_result) if hasattr(ResultSchema, 'get_output') else prev_result
                existing_context = task.get("context") if isinstance(task.get("context"), dict) else {}
                task["context"] = {**existing_context, "previous_output": prev_output or {}}
            delegation_reviewers = await self._build_delegation_reviewers(
                node, project_id, project, execution_mode=execution_mode,
            )
            return await self.phase_runner.run_phase_direct(
                project_id,
                task,
                agent_type,
                agents,
                phase=phase_label,
                tenant_id=project.get("tenant_id"),
                delegation_reviewers=delegation_reviewers,
                resume_attempt=resume_attempt,
            )
        else:
            # Auction — PhaseRunner creates its own task
            delegation_reviewers = await self._build_delegation_reviewers(
                node,
                project_id,
                project,
                execution_mode=execution_mode,
            )
            return await self.phase_runner.run_phase(
                project_id, task_type, description,
                agents, token, lambda: self.orch._is_cancelled(project_id),
                task_metadata=phase_metadata,
                auction_phase=node.get("phase_label"),
                delegation_reviewers=delegation_reviewers,
                resume_attempt=resume_attempt,
            )

    async def _build_delegation_reviewers(
        self,
        node: dict,
        project_id: str,
        project: dict,
        *,
        execution_mode: str = "dynamic",
    ):
        """Build the per-target delegation reviewer seam for a direct phase node.

        Returns a DelegationReviewers (attached to the agent for the run by
        PhaseRunner.run_phase_direct) when the node declares `reviewers`, else
        None. It carries the ApprovalManager (for human gates), run/project ids,
        and the project approval_mode so human reviewers only fire in human mode.
        """
        reviewers_cfg = deepcopy(node.get("reviewers") or {})
        if normalize_workflow_execution_mode(execution_mode) != "static" and node.get("can_delegate") is True:
            targets = reviewers_cfg.setdefault("targets", {})
            default_policy = dict(reviewers_cfg.get("default") or {})
            default_policy["human"] = "post"
            reviewers_cfg["default"] = default_policy

            tenant_id = getattr(project.get("shared_context"), "tenant_id", None)
            agent_type_ref = str(node.get("agent_type") or "").strip()
            if agent_type_ref and tenant_id and self.orch.storage is not None:
                try:
                    from config.configuration_reference_validation import (
                        known_tenant_ids_from_storage,
                        normalize_workflow_node_agent_type_ref,
                    )

                    tenant_ids = await known_tenant_ids_from_storage(self.orch.storage)
                    agent_type_ref = await normalize_workflow_node_agent_type_ref(
                        agent_type_ref,
                        tenant_id=tenant_id,
                        storage=self.orch.storage,
                        known_tenant_ids=tenant_ids,
                    )
                except Exception as exc:
                    logger.warning(
                        "[ENGINE] delegation parent agent_type=%s tenant_id=%s — resolution failed: %s",
                        node.get("agent_type"),
                        tenant_id,
                        exc,
                    )
            parent = _resolve_direct_agent(
                project.get("agents", self.orch.agent_pool),
                agent_type_ref,
                tenant_id=tenant_id,
            )
            for target_id in getattr(parent, "allowed_delegation_targets", []) or []:
                target_policy = dict(targets.get(target_id) or {})
                target_policy["human"] = "post"
                targets[target_id] = target_policy

            for target_id, policy in list(targets.items()):
                merged_policy = dict(policy or {})
                merged_policy["human"] = "post"
                targets[target_id] = merged_policy

        if not reviewers_cfg:
            return None
        from agents.delegation_reviewers import DelegationReviewers
        return DelegationReviewers(
            reviewers_cfg,
            approvals=self.approvals,
            project_id=project_id,
            run_id=project.get("run_id"),
            approval_mode=project.get("approval_mode", "human"),
        )

    @staticmethod
    def _build_reject_feedback(
        node: dict,
        reason: Optional[str],
        decision: Optional[str] = None,
    ) -> str:
        """Phase-facing retry feedback for a human rejection at an approval gate."""
        gate_label = node.get("label") or node.get("id") or "approval gate"
        reason_text = str(reason).strip() if reason else ""
        if decision == AgentResultDecision.REVISE.value:
            lines = [
                f"A human reviewer requested point changes at gate '{gate_label}'."
            ]
        else:
            lines = [
                f"A human reviewer rejected the previous result in full at gate '{gate_label}'."
            ]
        if reason_text:
            lines.append(f"Reviewer feedback: {reason_text}")
        if decision == AgentResultDecision.REVISE.value:
            lines.append(
                "Revise the previous result only where needed and preserve valid parts."
            )
        else:
            lines.append(
                "Restart the stage from the original task and produce a new result."
            )
        return "\n".join(lines)

    @staticmethod
    def _get_declared_writes(node: dict) -> List[str]:
        """Return normalized writes declared on a workflow phase node."""
        writes = node.get("writes")
        if writes is None or not isinstance(writes, list):
            return []
        normalized = []
        for key in writes or []:
            if not isinstance(key, str):
                return []
            key = key.strip()
            if not key:
                return []
            normalized.append(key)
        return normalized

    @staticmethod
    def _get_max_phase_attempts(node: dict) -> int:
        """Return total phase attempts for writes enforcement, never below one."""
        # MVP semantics: max_retries stores total attempts, not retries after the first run.
        raw_attempts = node.get("max_retries", 3)
        try:
            attempts = int(raw_attempts)
        except (TypeError, ValueError):
            attempts = 3
        return max(1, attempts)

    async def _capture_writes_baseline(self, project_id: str, writes: List[str]) -> Dict[str, Any]:
        """Capture declared write values before the first phase attempt."""
        baseline = await self._read_context_values(project_id, writes)
        logger.info(
            "[ENGINE] project_id=%s writes=%s - captured writes baseline",
            project_id,
            writes,
        )
        return baseline

    async def _resumed_attempt_written_keys(
        self, project_id: str, node_id: str, writes: List[str]
    ) -> set:
        """Declared writes the resumed attempt already persisted via context_write.

        Empty for a fresh run (no resume marker). On resume, reads the resumed
        attempt's journal (peeking the marker without popping it — _run_phase_attempt
        pops it) and returns the declared keys whose closed, non-error
        context_write pair proves this attempt wrote them before the crash.
        """
        project = self._get_project(project_id)
        marker = project.get("_resume_attempt") if isinstance(project, dict) else None
        if not marker or marker.get("node_id") != node_id:
            return set()
        sc = project.get("shared_context")
        store = getattr(sc, "message_store", None) if sc else None
        run_id = getattr(sc, "run_id", None) if sc else None
        task_id = marker.get("task_id")
        if not (store and run_id and task_id):
            return set()

        from agents.run_rehydration import load_attempt_pairs

        try:
            pairs = await load_attempt_pairs(store, project_id, run_id, task_id)
        except Exception as e:
            logger.warning(
                "[ENGINE] project_id=%s node_id=%s — resume journal read failed: %s",
                project_id, node_id, e,
            )
            return set()

        wanted = set(writes)
        written: set = set()
        for pair in pairs or []:
            call = pair.get("call")
            result = pair.get("result")
            if not call or not result:
                continue
            call_data = call.get("data") or {}
            if call_data.get("name") != "context_write":
                continue
            if (result.get("status") or "").lower() == "error":
                continue
            try:
                args = json.loads(call_data.get("arguments") or "{}")
            except (TypeError, ValueError):
                continue
            key = str(args.get("key") or "").strip()
            if key in wanted:
                written.add(key)
        return written

    async def _evaluate_writes_contract(
        self,
        project_id: str,
        writes: List[str],
        baseline: Dict[str, Any],
        satisfied_keys: Optional[set] = None,
    ) -> Dict[str, Any]:
        """Check that all declared writes exist and changed from first-entry baseline.

        ``satisfied_keys`` (a resumed attempt's journaled context_writes) are
        accepted as-is: they were durably written before the crash, so an
        equal-to-baseline read is not staleness.
        """
        current = await self._read_context_values(project_id, writes)
        satisfied_keys = satisfied_keys or set()
        missing = []
        unchanged = []
        changed = []

        for key in writes:
            current_value = current.get(key, _MISSING)
            baseline_value = baseline.get(key, _MISSING)
            if key in satisfied_keys and current_value is not _MISSING:
                changed.append(key)
            elif current_value is _MISSING:
                missing.append(key)
            elif baseline_value is not _MISSING and current_value == baseline_value:
                unchanged.append(key)
            else:
                changed.append(key)

        return {
            "fulfilled": not missing and not unchanged,
            "missing": missing,
            "unchanged": unchanged,
            "changed": changed,
        }

    async def _read_context_values(self, project_id: str, keys: List[str]) -> Dict[str, Any]:
        """Read logical SharedContext keys using the contract-aware resolver."""
        shared_context = self._get_project(project_id)["shared_context"]
        values = {}
        reader = getattr(shared_context, "read_context_key_async", None)
        fallback_reader = getattr(shared_context, "read_context_key", None)
        full_context_reader = getattr(shared_context, "get_full_context", None)

        for key in keys:
            if callable(reader):
                value = await reader(key, default=_MISSING)
            elif callable(fallback_reader):
                value = fallback_reader(key, default=_MISSING)
            elif callable(full_context_reader):
                value = full_context_reader().get(key, _MISSING)
            else:
                logger.warning(
                    "[ENGINE] project_id=%s key=%s - shared_context has no contract read path",
                    project_id,
                    key,
                )
                value = _MISSING
            values[key] = _MISSING if value is _MISSING else deepcopy(value)

        return values

    @staticmethod
    def _build_writes_retry_feedback(
        node_id: str,
        writes: List[str],
        contract: Dict[str, Any],
        attempt: int,
        max_attempts: int,
    ) -> str:
        """Build human-readable feedback injected into the next agent attempt."""
        details = []
        if contract["missing"]:
            details.append(f"missing keys: {', '.join(contract['missing'])}")
        if contract["unchanged"]:
            details.append(
                "unchanged keys: "
                f"{', '.join(contract['unchanged'])} "
                "(values match the state before this node)"
            )
        detail_text = "; ".join(details) if details else "writes contract was not fulfilled"
        return (
            f"Workflow node '{node_id}' requires writes [{', '.join(writes)}]. "
            f"Attempt {attempt}/{max_attempts} did not satisfy the contract: {detail_text}. "
            "Update the required context keys before finishing this phase. "
            "For multiple writes, return a structured object keyed by write name or use context_write."
        )

    @staticmethod
    def _create_writes_contract_failure(
        node_id: str,
        writes: List[str],
        contract: Dict[str, Any],
        attempts: int,
        feedback: str,
    ) -> dict:
        """Create an explicit failed phase result for an unfulfilled writes contract."""
        reason = (
            f"Phase node '{node_id}' did not satisfy writes contract after "
            f"{attempts} attempt(s). Missing: {contract['missing']}. "
            f"Unchanged: {contract['unchanged']}."
        )
        return ResultSchema.create(
            status=TaskStatus.FAILED,
            output={
                "error": "writes_contract_not_satisfied",
                "node_id": node_id,
                "writes": writes,
                "missing": contract["missing"],
                "unchanged": contract["unchanged"],
                "changed": contract["changed"],
                "attempts": attempts,
                "retry_feedback": feedback,
            },
            reasoning=reason,
            reason=reason,
        )

    @staticmethod
    def _build_phase_task_metadata(
        node: dict,
        default_reads: Optional[List[str]] = None,
        *,
        execution_mode: str = "dynamic",
    ) -> dict:
        """Extract prompt-driven phase metadata to preserve it in runtime tasks."""
        metadata = {TaskSchema.WORKFLOW_NODE_ID: node["id"]}
        label = node_label(node)
        if label:
            metadata[TaskSchema.WORKFLOW_NODE_LABEL] = label
        # Kept when None: a phase with no route onward records a null next step.
        if TaskSchema.NEXT_STEP in node:
            metadata[TaskSchema.NEXT_STEP] = node[TaskSchema.NEXT_STEP]

        if "reads" in node and node.get("reads") is not None:
            metadata["reads"] = WorkflowEngine._resolve_phase_reads(
                node.get("reads"),
                default_reads,
            )
        elif default_reads is not None:
            metadata["reads"] = default_reads

        for key in (
            "writes",
            "max_retries",
            "max_tool_failures",
            "max_output_repairs",
            "output_schema",
            "checks",
            TaskSchema.SUPPRESS_ASSISTANT_MESSAGE,
        ):
            if key in node and node.get(key) is not None:
                metadata[key] = node.get(key)

        if node.get("phase_label"):
            metadata["workflow_phase_label"] = node.get("phase_label")

        if normalize_workflow_execution_mode(execution_mode) != "static" and node.get("can_delegate") is True:
            metadata["can_delegate"] = True

        return metadata

    @staticmethod
    def _resolve_phase_reads(
        node_reads: Optional[List[str]],
        default_reads: Optional[List[str]],
    ) -> List[str]:
        """Resolve node-level read tokens into the concrete runtime reads list."""
        reads = [
            str(key or "").strip()
            for key in (node_reads or [])
            if str(key or "").strip()
        ]
        if FULL_CONTEXT_READS_TOKEN in reads:
            return [FULL_CONTEXT_READS_TOKEN]
        if WORKFLOW_DEFAULT_READS_TOKEN not in reads:
            return reads

        resolved = []
        seen = set()
        for key in [*(default_reads or []), *reads]:
            key = str(key or "").strip()
            if not key or key == WORKFLOW_DEFAULT_READS_TOKEN or key in seen:
                continue
            if key == FULL_CONTEXT_READS_TOKEN:
                return [FULL_CONTEXT_READS_TOKEN]
            resolved.append(key)
            seen.add(key)
        return resolved

    # ── rehydration helper (shared by approval_gate + deploy) ────

    def _consume_rehydration_marker(self, project: dict, node_id: str) -> Optional[str]:
        """Reuse a resolved instance only for its explicit restart marker.

        Fresh loop visits have no marker and request a new review. Consuming
        a mismatched marker prevents it from affecting a later gate.
        """
        aid = project.pop("_rehydration_approval_id", None)
        if not aid:
            return None
        existing = self.approvals.pending_approvals.get(aid)
        existing_gate = (existing.get("data") or {}).get("gate_node_id") if existing else None
        if existing and existing_gate == node_id:
            return aid
        return None

    # ── approval gate (generic) ──────────────────────────────────

    async def _handle_approval_gate(
        self,
        node: dict,
        workflow_def: dict,
        project_id: str,
        agent_result: Optional[dict] = None,
    ) -> dict:
        """Generic approval gate — pause, ask human yes/no.

        Returns ``{"status": "approved"|"rejected"|"timeout"}``; on rejection
        the dict also carries the reviewer's ``reason`` so ``execute()`` can
        thread it into the re-run of the rejected step as retry feedback.

        The gate knows nothing about what is being approved.  It creates an
        approval request with the current project state; the UI decides what
        to show.
        """
        project = self._get_project(project_id)
        shared_context = project["shared_context"]
        run_id = project.get("run_id")
        gate_label = node.get("label", node["id"])
        reader = getattr(shared_context, "read_context_key_async", None)
        upstream_approval = (
            await reader("workflow_approval") if callable(reader) else None
        )
        await shared_context.record_workflow_approval(None)

        approval_mode = project.get("approval_mode", "human")
        if approval_mode != "human":
            # Auto mode: resolve the gate without minting a PENDING approval,
            # message, snapshot, or APPROVAL_REQUESTED event — otherwise an auto
            # run leaves an unresolved approval card and audit entry the user
            # never acts on. Pop any rehydration marker so a stale one can't
            # mis-route a later gate (mirrors the human path's consume below).
            self._consume_rehydration_marker(project, node["id"])
            logger.info(
                "[ENGINE] Auto-approved gate '%s' for %s (approval_mode=%s)",
                gate_label, project_id, approval_mode,
            )
            return {"status": "approved"}

        approval_data = await self.build_approval_data(
            workflow_def=workflow_def,
            gate_node_id=node["id"],
            shared_context=shared_context,
            project_id=project_id,
            run_id=run_id,
            agent_result=agent_result,
        )
        upstream_reference = None
        if (
            isinstance(upstream_approval, dict)
            and upstream_approval.get("status") == "approved"
            and upstream_approval.get("project_id") == project_id
            and upstream_approval.get("run_id") == run_id
            and upstream_approval.get("approval_id")
            and upstream_approval.get("gate_node_id")
        ):
            upstream_reference = {
                "approval_id": upstream_approval["approval_id"],
                "gate_node_id": upstream_approval["gate_node_id"],
            }
            approval_data["refine_from_approval"] = upstream_reference

        approval_id = self._consume_rehydration_marker(project, node["id"])
        if approval_id is None:
            approval_id = await self.approvals.request_approval(
                project_id, gate_label, approval_data, run_id=run_id
            )

        approval = await self.approvals.wait_for_approval(approval_id)
        if not approval or approval.get("status") == "timeout":
            return {"status": "timeout"}
        status = approval.get("status")
        if status == ApprovalStatus.REJECTED or status == "rejected":
            stored_reference = (approval.get("data") or {}).get(
                "refine_from_approval"
            )
            restore_reference = upstream_reference or stored_reference
            if restore_reference is not None:
                if not isinstance(restore_reference, dict):
                    return {
                        "status": "failed",
                        "reason": "refine_upstream_context_malformed",
                        "node_id": node["id"],
                    }
                upstream_approval_id = restore_reference.get("approval_id")
                upstream_gate_node_id = restore_reference.get("gate_node_id")
                if not upstream_approval_id or not upstream_gate_node_id:
                    return {
                        "status": "failed",
                        "reason": "refine_upstream_context_incomplete",
                        "node_id": node["id"],
                    }
                restored = await publish_workflow_approval(
                    shared_context,
                    self.approvals.message_store,
                    upstream_approval or {},
                    approval_id=upstream_approval_id,
                    project_id=project_id,
                    run_id=run_id,
                    gate_node_id=upstream_gate_node_id,
                )
                if restored.get("status") != "approved":
                    return restored
            typed_decision = decision_from_approval(approval)
            return {
                "status": "rejected",
                "decision": (
                    typed_decision.value
                    if typed_decision in {
                        AgentResultDecision.REVISE,
                        AgentResultDecision.REJECT,
                    }
                    else AgentResultDecision.REJECT.value
                ),
                "reason": approval.get("reason") or approval.get("feedback"),
            }

        return await publish_workflow_approval(
            shared_context,
            self.approvals.message_store,
            approval,
            approval_id=approval_id,
            project_id=project_id,
            run_id=run_id,
            gate_node_id=node["id"],
        )

    async def build_approval_data(
        self,
        *,
        workflow_def: dict,
        gate_node_id: str,
        shared_context,
        project_id: str,
        run_id: Optional[str] = None,
        agent_result: Optional[dict] = None,
    ) -> dict:
        """Construct approval_data payload for a gate.

        Shared by `_handle_approval_gate` (initial gate creation) and by
        `_refine_approval_core` (post-refine re-emit) so refined payloads
        carry the same field shape as freshly minted ones — including
        `refine_target_node_id`, which the refine endpoint reads to know
        which upstream phase to re-run without dispatching by gate name.

        Artifacts source-of-truth is the DB artifact_store, not
        `shared_context["artifacts"]`: the in-memory copy is empty for
        code-file artifacts (which flow through container/repo + DB), so
        reading from it left the output approval card with an empty file
        list while the file was already snapshotted to the DB.
        """
        nodes_map = {n["id"]: n for n in workflow_def.get("nodes", [])}
        gate_node = nodes_map.get(gate_node_id, {})
        gate_label = gate_node.get("label", gate_node_id)

        full_ctx = (
            shared_context.get_full_context()
            if hasattr(shared_context, "get_full_context")
            else {}
        )

        artifacts: list = []
        store = getattr(self.orch, "artifact_store", None)
        if store is not None:
            try:
                artifacts = await store.get_all_files(project_id, run_id=run_id) or []
            except Exception:
                artifacts = list(full_ctx.get("artifacts", []) or [])
        else:
            artifacts = list(full_ctx.get("artifacts", []) or [])

        context_snapshot = {
            "user_prompt": full_ctx.get("user_prompt", ""),
            "requirements": full_ctx.get("requirements", {}),
            "plan": full_ctx.get("plan", {}),
            "artifacts": artifacts,
        }
        for research_key in (
            "literature",
            "search_results",
            "research_answer",
            "approved_literature",
        ):
            value = full_ctx.get(research_key)
            if value not in (None, "", [], {}):
                context_snapshot[research_key] = value

        approval_data = {
            "gate_node_id": gate_node_id,
            "gate_label": gate_label,
            "context_snapshot": context_snapshot,
        }
        refine_target = self._find_refine_target(workflow_def, gate_node_id)
        interaction_schema = gate_node.get("interaction_schema")
        if isinstance(interaction_schema, dict) and interaction_schema:
            approval_data["interaction_schema"] = deepcopy(interaction_schema)
        elif refine_target:
            source_node = nodes_map.get(refine_target, {})
            if source_node.get("type") == "phase":
                if agent_result is not None:
                    output = await self._reconstruct_phase_output(
                        source_node, shared_context, full_ctx
                    )
                    if output is None:
                        output = (
                            ResultSchema.get_output(agent_result)
                            if isinstance(agent_result, dict)
                            else agent_result
                        )
                    show_widget = True
                else:
                    # Gate RE-ENTRY (restart recovery): execute() restarts at the
                    # gate with prev_result=None, so the phase's result object is
                    # gone — but the values it wrote survive under its declared
                    # writes-keys. Rebuild what the reviewer is asked to approve
                    # (inverting _save_contract_outputs) so the re-entered card
                    # shows the output instead of a blank review widget (F4).
                    output = await self._reconstruct_phase_output(
                        source_node, shared_context, full_ctx
                    )
                    show_widget = output not in (None, "", {}, [])
                if show_widget:
                    approval_data["interaction_schema"] = (
                        build_agent_result_interaction_schema()
                    )
                    approval_data["agent_result"] = {
                        "agent_id": source_node.get("agent_type") or refine_target,
                        "agent_display_name": source_node.get("label")
                        or source_node.get("agent_type")
                        or refine_target,
                        "output": output,
                    }

        show_keys = [
            str(key or "").strip()
            for key in (gate_node.get("show_keys") or [])
            if str(key or "").strip()
        ]
        if show_keys:
            reader = getattr(shared_context, "read_context_key_async", None)
            custom: Dict[str, Any] = {}
            for key in show_keys:
                if callable(reader):
                    custom[key] = await reader(key)
                else:
                    custom[key] = full_ctx.get(key)
            approval_data["context_snapshot"]["custom"] = custom

        if refine_target:
            approval_data["refine_target_node_id"] = refine_target
        approval_data["next_steps"] = approval_next_steps(
            workflow_def, gate_node_id, refine_target
        )

        return approval_data

    async def _reconstruct_phase_output(
        self, source_node: dict, shared_context, full_ctx: dict
    ):
        """Read a completed phase's durable output from its declared writes.

        This inverts GenericAgent._save_contract_outputs: a single-key contract
        stores the whole output, while a multi-key contract splits a dict by key.
        The durable value is also available after gate re-entry, when the phase
        result object no longer exists.
        """
        writes = self._get_declared_writes(source_node)
        if not writes:
            return None
        reader = getattr(shared_context, "read_context_key_async", None)

        async def _read(key):
            if callable(reader):
                return await reader(key)
            return full_ctx.get(key)

        if len(writes) == 1:
            return await _read(writes[0])
        output: Dict[str, Any] = {}
        for key in writes:
            value = await _read(key)
            if value not in (None, "", {}, []):
                output[key] = value
        return output or None

    @staticmethod
    def _find_refine_target(workflow_def: dict, gate_node_id: str) -> Optional[str]:
        """BFS back from a gate over incoming edges to the upstream work node.

        Returns the id of the nearest phase / execution / deploy node that
        feeds this gate. Approval_gate and validator nodes are skipped
        through so a chain of gates (and any pass-through validators)
        resolves to the same underlying phase. Returns None when no work
        node sits upstream (gate immediately follows start, which
        shouldn't happen in valid DAGs but is handled defensively).

        Stored as `approval.data["refine_target_node_id"]` at gate creation
        so the refine endpoint never has to traverse the graph at runtime —
        it reads the field and asks the engine to re-execute that node.
        """
        nodes_map = {n["id"]: n for n in workflow_def.get("nodes", [])}
        edges = workflow_def.get("edges", [])
        visited: set = set()
        queue: List[str] = [gate_node_id]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for edge in edges:
                if edge.get("to") != current:
                    continue
                src_id = edge.get("from")
                if not src_id or src_id in visited:
                    continue
                src_node = nodes_map.get(src_id)
                if not src_node:
                    continue
                src_type = src_node.get("type")
                if src_type in ("phase", "execution", "deploy", "map"):
                    return src_id
                # A tool node has no model to take the reviewer's revision, so
                # the refine goes on to the phase that feeds it.
                if src_type in ("approval_gate", "validator", "tool"):
                    queue.append(src_id)
        return None

    async def _execute_node(
        self,
        node: dict,
        project_id: str,
        workflow_def: dict,
        prev_result: Optional[dict] = None,
        reject_feedback: Optional[str] = None,
    ) -> dict:
        """Execute a single workflow node and return its result.

        Shared work-runner: used by `execute()` while walking the DAG and
        by `_refine_approval_core` when feedback arrives at a gate and the
        upstream phase has to re-run in place. Edge traversal stays in
        `execute()` — this method only runs the node's work, not the
        routing decision afterwards.

        `reject_feedback` carries a human gate-rejection reason into the
        re-run phase as retry feedback (phase nodes only).

        `start` / `end` / unknown node types return a sentinel; the caller
        decides whether reaching them is meaningful.
        """
        # Refinement enters here from a separate request, outside execute().
        project = self.orch.active_projects.get(project_id) or {}
        async with run_trace_scope(
            self.orch.storage, self.tracer, project_id, project.get("run_id")
        ):
            return await self._execute_node_in_run(
                node, project_id, workflow_def, prev_result, reject_feedback
            )

    async def rerun_refine_path(
        self,
        *,
        target_node: dict,
        gate_node_id: str,
        project_id: str,
        workflow_def: dict,
        feedback: Optional[str] = None,
    ) -> dict:
        """`feedback` becomes the retry feedback of a map target's item tasks;
        a phase target is re-run without it."""
        writes = self._get_declared_writes(target_node)
        if not writes:
            return {
                "status": "failed",
                "reason": "refine_target_missing_write_contract",
                "node_id": target_node.get("id"),
            }
        try:
            previous_state = await self._read_context_values(project_id, writes)
        except Exception as exc:
            logger.exception(
                "[REFINE] project_id=%s node_id=%s - failed to capture output state",
                project_id,
                target_node.get("id"),
            )
            return {
                "status": "failed",
                "reason": "refine_snapshot_failed",
                "error": str(exc),
            }
        missing = [key for key, value in previous_state.items() if value is _MISSING]
        if missing:
            return {
                "status": "failed",
                "reason": "refine_source_output_missing",
                "missing": missing,
            }
        previous_output = (
            previous_state[writes[0]]
            if len(writes) == 1
            else deepcopy(previous_state)
        )

        try:
            reject_feedback = None
            if target_node.get("type") == "map":
                await self._forget_map_progress(project_id, target_node["id"])
                reject_feedback = feedback
            result = await self._execute_node(
                target_node,
                project_id,
                workflow_def,
                prev_result=None,
                reject_feedback=reject_feedback,
            )
            if self._is_failure(result):
                return await self._rollback_refine_result(
                    target_node, project_id, previous_state, result
                )

            nodes = {node["id"]: node for node in workflow_def.get("nodes", [])}
            edges = workflow_def.get("edges", [])
            current_id = target_node["id"]
            visited = {current_id}
            next_id = self._find_default_edge(current_id, edges)
            while True:
                if next_id == gate_node_id:
                    return {
                        "status": "approved",
                        "phase_result": result,
                        "previous_state": deepcopy(previous_state),
                    }
                if not next_id or next_id in visited:
                    failure = {
                        "status": "failed",
                        "reason": "refine_path_invalid",
                        "node_id": current_id,
                    }
                    return await self._rollback_refine_result(
                        target_node, project_id, previous_state, failure
                    )
                visited.add(next_id)
                node = nodes.get(next_id)
                if not node or node.get("type") != "validator":
                    failure = {
                        "status": "failed",
                        "reason": "refine_path_unsupported_node",
                        "node_id": next_id,
                    }
                    return await self._rollback_refine_result(
                        target_node, project_id, previous_state, failure
                    )
                validation = await self._run_validator_node(
                    node,
                    project_id,
                    additional_checks=node.get("refine_checks") or [],
                    extra_context={"refine_previous": previous_output},
                    bound_outputs=node_bound_outputs(target_node),
                )
                if validation.get("status") != "approved":
                    return await self._rollback_refine_result(
                        target_node, project_id, previous_state, validation
                    )
                current_id = next_id
                next_id = self._find_edge(current_id, "approved", edges)
                if next_id is None:
                    next_id = self._find_default_edge(current_id, edges)
        except asyncio.CancelledError:
            try:
                await self._restore_refine_state(
                    target_node, project_id, previous_state
                )
            except Exception:
                logger.exception(
                    "[REFINE] project_id=%s node_id=%s - rollback failed after cancellation",
                    project_id,
                    target_node.get("id"),
                )
            raise
        except Exception as exc:
            logger.exception(
                "[REFINE] project_id=%s node_id=%s - refine execution failed",
                project_id,
                target_node.get("id"),
            )
            return await self._rollback_refine_result(
                target_node,
                project_id,
                previous_state,
                {
                    "status": "failed",
                    "reason": "refine_execution_exception",
                    "error": str(exc),
                },
            )

    async def _forget_map_progress(self, project_id: str, node_id: str) -> None:
        run_id = self._get_project(project_id).get("run_id")
        await MapProgress(self.orch.storage, run_id, node_id).clear()

    async def _rollback_refine_result(
        self,
        target_node: dict,
        project_id: str,
        previous_state: Dict[str, Any],
        result: dict,
    ) -> dict:
        try:
            await self._restore_refine_state(
                target_node, project_id, previous_state
            )
        except Exception as exc:
            logger.exception(
                "[REFINE] project_id=%s node_id=%s - output rollback failed",
                project_id,
                target_node.get("id"),
            )
            return {
                "status": "failed",
                "reason": "refine_rollback_failed",
                "error": str(exc),
                "cause": result,
            }
        return result

    async def _restore_refine_state(
        self,
        target_node: dict,
        project_id: str,
        previous_state: Dict[str, Any],
    ) -> None:
        await self._restore_refine_values(target_node, project_id, previous_state)
        if target_node.get("type") == "map":
            # The discarded run's items must not be reused by the next entry.
            await self._forget_map_progress(project_id, target_node["id"])

    async def _restore_refine_values(
        self,
        target_node: dict,
        project_id: str,
        previous_state: Dict[str, Any],
    ) -> None:
        shared_context = self._get_project(project_id)["shared_context"]
        writes = self._get_declared_writes(target_node)
        restore_values = {}
        for key in writes:
            if key not in previous_state or previous_state[key] is _MISSING:
                raise ValueError(f"Missing rollback value for context key '{key}'")
            restore_values[key] = deepcopy(previous_state[key])
        exact_restorer = getattr(
            shared_context, "restore_context_keys_exact", None
        )
        if asyncio.iscoroutinefunction(exact_restorer):
            await exact_restorer(restore_values)
            return

        failed = []
        for key in writes:
            try:
                await shared_context.write_context_key(
                    key, restore_values[key]
                )
            except Exception:
                failed.append(key)
                logger.exception(
                    "[REFINE] project_id=%s node_id=%s key=%s - context key restore failed",
                    project_id,
                    target_node.get("id"),
                    key,
                )
        if failed:
            raise RuntimeError(
                f"Failed to restore context keys: {', '.join(failed)}"
            )

    @classmethod
    def _refine_state_from_approval(
        cls,
        target_node: dict,
        approval_data: dict,
        fallback_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        writes = cls._get_declared_writes(target_node)
        snapshot = approval_data.get("context_snapshot") or {}
        custom = snapshot.get("custom") if isinstance(snapshot, dict) else {}
        output = (approval_data.get("agent_result") or {}).get("output", _MISSING)
        restored = {}
        for key in writes:
            if isinstance(snapshot, dict) and key in snapshot:
                restored[key] = deepcopy(snapshot[key])
            elif isinstance(custom, dict) and key in custom:
                restored[key] = deepcopy(custom[key])
            elif len(writes) == 1 and output is not _MISSING:
                restored[key] = deepcopy(output)
            elif isinstance(output, dict) and key in output:
                restored[key] = deepcopy(output[key])
            elif key in fallback_state:
                restored[key] = deepcopy(fallback_state[key])
        return restored

    async def _execute_node_in_run(
        self,
        node: dict,
        project_id: str,
        workflow_def: dict,
        prev_result: Optional[dict],
        reject_feedback: Optional[str],
    ) -> dict:
        node_type = node.get("type")
        default_reads = workflow_def.get("default_reads")
        execution_mode = normalize_workflow_execution_mode(workflow_def.get("execution_mode"))
        if node_type == "phase":
            phase_node = {
                **node,
                TaskSchema.NEXT_STEP: next_workflow_step(workflow_def, node["id"]),
            }
            approval_mode = self._get_project(project_id).get(
                "approval_mode",
                "human",
            )
            if (
                approval_mode == "human"
                and self._phase_is_followed_by_approval_gate(
                    node["id"],
                    workflow_def,
                )
            ):
                phase_node[TaskSchema.SUPPRESS_ASSISTANT_MESSAGE] = True
            return await self._run_phase_node(
                phase_node, project_id, prev_result,
                default_reads=default_reads,
                reject_feedback=reject_feedback,
                execution_mode=execution_mode,
            )
        if node_type == "approval_gate":
            return await self._handle_approval_gate(
                node,
                workflow_def,
                project_id,
                agent_result=prev_result,
            )
        if node_type == "execution":
            return await self._run_execution_node(node, project_id)
        if node_type == "deploy":
            return await self._run_deploy_node(node, project_id)
        if node_type == "tool":
            return await ToolNodeRunner(self.orch).run(node, project_id)
        if node_type == "map":
            metadata = self._build_phase_task_metadata(
                node, default_reads=default_reads, execution_mode=execution_mode
            )
            runner = MapNodeRunner(self.orch, self.phase_runner)
            return await runner.run(node, project_id, metadata, reject_feedback)
        if node_type == "a2a_agent":
            return await self._run_a2a_agent_node(node, project_id, prev_result, workflow_def)
        if node_type == "validator":
            return await self._run_validator_node(
                node,
                project_id,
                bound_outputs=self._validator_bound_outputs(node["id"], workflow_def),
            )
        return {"status": "skipped", "node_type": node_type}

    @staticmethod
    def _phase_is_followed_by_approval_gate(
        phase_node_id: str,
        workflow_def: dict,
    ) -> bool:
        """Does this phase feed a gate — directly, or through validators?

        Validators are pass-through format checks: on approve they hand the
        phase's own result to the gate, which shows it for review. So a phase
        gated *through* a validator must still suppress its assistant final,
        or the same content appears twice in chat (once from the phase, once
        on the approval card).
        """
        next_step = next_workflow_step(workflow_def, phase_node_id)
        return next_step is not None and next_step["type"] == "approval_gate"

    # ── execution node ───────────────────────────────────────────

    async def _run_execution_node(self, node: dict, project_id: str) -> dict:
        """Run sequential task execution via TaskExecutor + post-processing."""
        project = self._get_project(project_id)
        shared_context = project["shared_context"]
        agents = project.get("agents", self.orch.agent_pool)
        token = project.get("token")

        if self.orch._is_cancelled(project_id):
            return {"status": "cancelled"}

        execution_result = await self.task_executor.execute_tasks(
            project_id, shared_context, agents, token,
            lambda: self.orch._is_cancelled(project_id)
        )

        if execution_result.get("status") == "cancelled":
            logger.info("[ENGINE] Execution cancelled project_id=%s", project_id)
            return {
                "status": "cancelled",
                "execution_result": execution_result,
            }

        failed_task = self._first_failed_plan_task(execution_result)
        if failed_task is not None:
            agent_result = failed_task.get("result") if isinstance(failed_task.get("result"), dict) else {}
            task_id = failed_task.get("task_id")
            reason = (
                failed_task.get("reason")
                or failed_task.get("error")
                or ResultSchema.get_reasoning(agent_result)
                or (
                    failed_task.get("status")
                    if failed_task.get("status") in ("failed", "cancelled") and not task_id
                    else None
                )
                or (f"Plan task {task_id} failed" if task_id else "Execution failed")
            )
            logger.warning(
                "[ENGINE] Execution node failed project_id=%s task_id=%s reason=%s",
                project_id,
                failed_task.get("task_id"),
                reason,
            )
            return {
                "status": "failed",
                "reason": reason,
                "task_id": failed_task.get("task_id"),
                "execution_result": execution_result,
            }

        # Post-execution infrastructure (container sync, artifacts)
        final_artifacts = []
        try:
            if hasattr(self.orch, '_sync_container_to_repo'):
                await self.orch._sync_container_to_repo(project_id)
            if hasattr(self.orch, '_prepare_final_artifacts'):
                final_artifacts = await self.orch._prepare_final_artifacts(
                    project_id, project, shared_context
                ) or []
        except Exception as e:
            logger.warning("[ENGINE] Post-execution infrastructure error: %s", e)

        # Snapshot artifacts to DB
        if final_artifacts and self.orch.artifact_store:
            try:
                run_id = project.get("run_id")
                await self.orch.artifact_store.snapshot_from_artifacts(
                    project_id, final_artifacts, run_id, force=True
                )
            except Exception as e:
                logger.warning("[ENGINE] Artifact snapshot failed: %s", e)

        return {
            "status": "completed",
            "project_id": project_id,
            "artifacts": final_artifacts,
            "execution_result": execution_result,
        }

    # ── deploy node ────────────────────────────────────────────

    async def _run_deploy_node(self, node: dict, project_id: str) -> dict:
        """Run deploy: check env vars → request approval → execute deploy.

        If deploy is disabled (no env vars), returns completed immediately.
        """
        if node.get("html_from"):
            return await run_report_publish_node(self, node, project_id)

        import os

        prod_enabled = os.getenv("DEPLOY_AGENT_PROD_ENABLED", "false").lower() == "true"
        local_enabled = os.getenv("DEPLOY_AGENT_LOCAL_ENABLED", "false").lower() == "true"

        if not prod_enabled and not local_enabled:
            logger.info("[ENGINE] Deploy disabled (no env vars), skipping deploy node for %s", project_id)
            return {"status": "completed", "project_id": project_id, "deploy": "skipped"}

        project = self._get_project(project_id)
        run_id = project.get("run_id")

        # Build deploy approval data
        base_slug = (project_id.split("-")[0] or project_id).lower()
        deploy_data = {
            "gate_node_id": node["id"],
            "gate_label": "Deploy application",
            "deploy_slug": base_slug,
            "target_namespace": "AppFactory-apps",
            "deploy_mode": "prod" if prod_enabled else "local",
        }

        approval_mode = project.get("approval_mode", "human")
        if approval_mode != "human":
            # Auto mode: skip the deploy approval gate (no PENDING card) and
            # execute directly. Synthesize an approved approval carrying
            # deploy_data so _execute_deploy's data lookup and per-approval_id
            # idempotency behave exactly as on the human path. Pop any stale
            # rehydration marker defensively.
            self._consume_rehydration_marker(project, node["id"])
            approval = {
                "approval_id": f"auto-deploy-{project_id}-{node['id']}",
                "project_id": project_id,
                "gate_type": "deploy",
                "status": "approved",
                "data": deploy_data,
                "run_id": run_id,
            }
            logger.info("[ENGINE] Auto-approved deploy for %s (slug=%s), executing", project_id, base_slug)
        else:
            # Request deploy approval (shows card to user). Re-use the rehydration
            # marker'd approval if ensure_workflow_running spawned us — otherwise
            # mint fresh via request_approval. Same race the approval_gate marker
            # closes: a reject during the FSM-spawn → request_approval window
            # would otherwise slip past PENDING/APPROVED-only dedup and mint a
            # duplicate "Deploy application" card after a successful reject.
            logger.info("[ENGINE] Requesting deploy approval for %s (slug=%s)", project_id, base_slug)
            approval_id = self._consume_rehydration_marker(project, node["id"])
            if approval_id is None:
                approval_id = await self.approvals.request_approval(
                    project_id, "deploy", deploy_data, run_id=run_id
                )

            # Wait for user to approve/reject
            approval = await self.approvals.wait_for_approval(approval_id)
            if not approval or approval.get("status") in ("timeout", "rejected"):
                status = (approval or {}).get("status", "timeout")
                logger.info("[ENGINE] Deploy %s for %s", status, project_id)
                return {"status": "completed", "project_id": project_id, "deploy": status}

        # Execute deploy via orchestrator
        logger.info("[ENGINE] Deploy approved for %s, executing", project_id)
        try:
            deploy_result = await self.orch._execute_deploy(project_id, project, approval)
            if isinstance(deploy_result, dict) and deploy_result.get("status") == "failed":
                reason = deploy_result.get("reason", "unknown")
                recoverable = deploy_result.get("recoverable", False)
                logger.warning("[ENGINE] Deploy failed for %s: reason=%s recoverable=%s", project_id, reason, recoverable)
                return {"status": "completed", "project_id": project_id, "deploy": "failed", "reason": reason, "recoverable": recoverable}
            return {"status": "completed", "project_id": project_id, "deploy": "success"}
        except Exception as e:
            logger.error("[ENGINE] Deploy failed for %s: %s", project_id, e)
            return {"status": "completed", "project_id": project_id, "deploy": "failed", "error": str(e)}

    async def _run_validator_node(
        self,
        node: dict,
        project_id: str,
        *,
        additional_checks: Optional[List[dict]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
        bound_outputs: Collection[str] = (),
    ) -> dict:
        """Run a validator node's checks against the project's SharedContext.

        Deterministic format checks only (ADR-0011) — approved/rejected plus
        feedback for the rejected-edge retry loop, or a hard "failed"
        (validator_invalid_check_config) when the check CONFIG itself is
        broken and retrying could never help.
        """
        project = self._get_project(project_id)
        shared_context = project["shared_context"]
        checks = [*(node.get("checks") or []), *(additional_checks or [])]
        result = await ValidatorRunner().run(
            checks,
            shared_context,
            extra_context=extra_context,
            bound_outputs=bound_outputs,
        )
        if result.get("config_errors"):
            logger.warning(
                "[ENGINE] project_id=%s node_id=%s - validator has invalid check config: %s",
                project_id, node["id"], result["config_errors"],
            )
            return {
                "status": "failed",
                "reason": "validator_invalid_check_config",
                "error": "; ".join(result["config_errors"]),
                "node_id": node["id"],
            }
        if not result["passed"]:
            logger.info(
                "[ENGINE] project_id=%s node_id=%s - validator failed checks: %s",
                project_id, node["id"], result["errors"],
            )
            await self._emit_validator_rejected(project_id, node, result["errors"])
        return {
            "status": "approved" if result["passed"] else "rejected",
            "feedback": result.get("feedback", ""),
            "errors": result.get("errors", []),
            "error_details": result.get("error_details", []),
            "bound_keys": result.get("bound_keys", []),
            "node_id": node["id"],
        }

    async def _emit_validator_rejected(
        self, project_id: str, node: dict, errors: List[str]
    ) -> None:
        """Surface a validator rejection in the Events feed.

        Best-effort on purpose: an observability signal must never be able to
        break the run it is reporting on.
        """
        try:
            project = self._get_project(project_id)
            await self.event_emitter.emit(
                EventSchema.VALIDATOR_REJECTED,
                project.get("run_id"),
                {
                    "project_id": project_id,
                    "node_id": node["id"],
                    "errors": errors,
                },
            )
        except Exception as e:
            logger.warning("[ENGINE] Validator rejected-event emit failed: %s", e)

    # ── lifecycle helpers ────────────────────────────────────────

    async def _finalize_workflow(self, project_id: str, last_result: Optional[dict]):
        """Mark workflow as completed at end node."""
        if last_result and last_result.get("status") == "completed":
            try:
                project = self._get_project(project_id)
                project["status"] = "completed"
                await self.orch.storage.save_project(project_id, {
                    "user_prompt": project.get("user_prompt", ""),
                    "title": project.get("title", ""),
                    "status": "completed",
                    "current_phase": "completed",
                    "approval_mode": project.get("approval_mode", "human"),
                    "created_at": project.get("created_at"),
                    "metadata": project.get("metadata", {}),
                })
                # Finalize the run record too. This save path updates the project but
                # never touched the runs collection, so run_status stayed "running" and
                # workflow_phase "requirements" (their run-creation defaults) — the Runs
                # panel showed a completed project as still running. update_project_status
                # cascades to the run; this manual save does not, so finalize explicitly.
                run_id = project.get("run_id")
                if run_id:
                    try:
                        await self.orch.storage.update_run_status(run_id, "completed")
                        await self.orch.storage.update_run_phase(run_id, "completed")
                    except Exception as run_err:
                        logger.warning("[ENGINE] Failed to finalize run %s: %s", run_id, run_err)
                await self.event_emitter.emit(EventSchema.PROJECT_COMPLETED, project.get("run_id"), {
                    "project_id": project_id,
                })
                mcpx = getattr(self.orch, "mcp_executor", None)
                if mcpx is not None and hasattr(mcpx, "finalize_project_mcp_runtimes"):
                    try:
                        await mcpx.finalize_project_mcp_runtimes(project_id)
                    except Exception as mcp_err:
                        logger.warning(
                            "[ENGINE] MCP finalize failed project_id=%s: %s", project_id, mcp_err
                        )
            except Exception as e:
                logger.warning("[ENGINE] Finalize failed: %s", e)

    async def _handle_workflow_failure(self, project_id: str, result: dict):
        if self.orch._is_cancelled(project_id):
            logger.info(
                "[ENGINE] project_id=%s already cancelled — skipping failure cascade",
                project_id,
            )
            return
        logger.error(
            "[ENGINE] Workflow failed for %s: %s", project_id, failure_for_log(result)
        )
        try:
            # Full raw cause for PROJECT_FAILED + the Events tab (the chat card gets a
            # summary instead, below). A2A/execution nodes report via `error`/`error_type`,
            # phases via `reason`; prefer either over a stringified result dict.
            error_text = result.get("error") or result.get("reason") or str(result)
            error_type = result.get("error_type")
            project = self._get_project(project_id) or {}
            run_id = project.get("run_id")
            # The failed status is stored before PROJECT_FAILED goes out: the page rereads
            # the project once on that event, so a later write would leave it showing the
            # project still running, with no run time, until a reload.
            try:
                await self.orch.update_project_status(project_id, "failed")
            except Exception as status_err:
                logger.warning("[ENGINE] update_project_status failed, falling back to run status: %s", status_err)
                if run_id:
                    try:
                        await self.orch.storage.update_run_status(run_id, "failed")
                    except Exception as run_err:
                        logger.warning(
                            "[ENGINE] project_id=%s run status fallback failed, "
                            "reporting the failure anyway: %s",
                            project_id,
                            run_err,
                        )
            await self.event_emitter.emit(EventSchema.PROJECT_FAILED, run_id, {
                "project_id": project_id,
                "error": error_text,
                "error_type": error_type,
            })
            # Surface the failure in the Chat tab. PROJECT_FAILED only reaches the Events
            # tab + an 8s toast, and update_run_status only the Runs panel — none write to
            # the messages collection, so without this a dead run reads as "Ready" in chat.
            # The card shows a plain-language summary; the raw trace (kept whole above for
            # PROJECT_FAILED/Events) goes to the capped detail pane.
            chat_content, chat_data = build_failure_chat_message(result)
            message_store = getattr(self.orch, "message_store", None)
            if message_store is not None:
                try:
                    await message_store.append_system_message(
                        project_id,
                        "error",
                        chat_content,
                        run_id=run_id,
                        data=chat_data,
                    )
                except Exception as msg_err:
                    logger.warning("[ENGINE] Failed to append failure chat message: %s", msg_err)
        except Exception:
            pass

    async def _emit_phase_event(self, phase_label: str, project_id: str, status: str):
        """Emit phase started/completed event."""
        try:
            project = self._get_project(project_id)
            await self.phase_runner.emit_phase_event(phase_label, project_id, status, project)
        except Exception as e:
            logger.warning("[ENGINE] Phase event emit failed: %s", e)

    async def _create_snapshot(self, project_id: str, phase_label: str):
        """Create a phase snapshot."""
        try:
            project = self._get_project(project_id)
            shared_context = project["shared_context"]
            await self.snapshot_manager.create_snapshot(
                project_id, shared_context,
                snap_type="phase_complete",
                label=f"{phase_label} completed",
                phase=phase_label,
                meta={"tags": ["system_checkpoint", f"phase_{phase_label}"]},
            )
        except Exception as e:
            logger.warning("[ENGINE] Snapshot creation failed: %s", e)

    # ── utilities ────────────────────────────────────────────────

    def _get_project(self, project_id: str) -> dict:
        project = self.orch.active_projects.get(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found in active projects")
        return project

    @staticmethod
    def _task_result_is_failed(task_result: dict) -> bool:
        if task_result.get("status") in ("failed", "cancelled"):
            return True
        agent_result = task_result.get("result")
        if not isinstance(agent_result, dict):
            return "result" in task_result
        return ResultSchema.is_failed(agent_result) or not ResultSchema.is_completed(agent_result)

    @staticmethod
    def _first_failed_plan_task(execution_result: Optional[dict]) -> Optional[dict]:
        """Return the first failed plan task from TaskExecutor.execute_tasks output."""
        if not execution_result:
            return None
        for task_result in execution_result.get("task_results") or []:
            if isinstance(task_result, dict) and WorkflowEngine._task_result_is_failed(task_result):
                return task_result
        if execution_result.get("status") == "failed":
            return execution_result
        return None

    @staticmethod
    def _is_failure(result: Optional[dict]) -> bool:
        """Check if a phase result indicates failure."""
        if result is None:
            return False
        status = result.get("status", "")
        return status in ("failed", "cancelled")

    # ── run guards (AppFactory-77 Part B, ADR-0012) ───────────────────

    @classmethod
    def _resolve_max_iterations(cls, workflow_def: dict) -> int:
        """DAG-walk iteration cap for this run: workflow override or engine default.

        Reads a raw Mongo doc (no Pydantic gate at this point — see
        mongo_backend.get_workflow_definition), so a malformed value must fall
        back quietly rather than crash a run.
        """
        raw = workflow_def.get("max_iterations")
        if raw is None:
            return cls._MAX_ITERATIONS
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 0
        if value >= 1:
            return value
        logger.warning(
            "[ENGINE] Ignoring invalid max_iterations=%r, using default %d",
            raw, cls._MAX_ITERATIONS,
        )
        return cls._MAX_ITERATIONS

    @staticmethod
    def _resolve_validator_retry_cap(node: dict) -> Optional[int]:
        """Per-validator reject→retry budget (AppFactory-77 F3, ADR-0011), or None if off.

        Reads a raw Mongo doc (no Pydantic gate at run time — mirror
        _resolve_max_iterations), so a malformed value falls back to None: guard
        off = pre-existing behavior, where the validator→rejected→phase loop is
        bounded only by the global max_iterations. Save-time validation
        (workflow_definitions._validate_validator_node_contract) already rejects
        malformed values, so this fallback only shields hand-edited/legacy docs.
        """
        raw = node.get("max_reject_retries")
        if raw is None or isinstance(raw, bool):
            return None
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            return None
        return cap if cap >= 0 else None

    @staticmethod
    def _resolve_run_deadline(workflow_def: dict) -> Optional[float]:
        """Wall-clock deadline (monotonic seconds) for this run, or None if the
        guard is off (no run_timeout_seconds, or an invalid value)."""
        raw = workflow_def.get("run_timeout_seconds")
        if raw is None:
            return None
        try:
            seconds = float(raw)
        except (TypeError, ValueError):
            seconds = 0.0
        if seconds <= 0:
            logger.warning("[ENGINE] Ignoring invalid run_timeout_seconds=%r", raw)
            return None
        return time.monotonic() + seconds

    async def _fail_on_run_timeout(self, project_id: str, workflow_def: dict, node_id: str) -> dict:
        """Fail the run on a blown run_timeout_seconds budget.

        Like the iteration-limit exit (_fail_on_max_iterations), this goes
        through _handle_workflow_failure (PROJECT_FAILED + chat message + status
        cascade) — a silently hung run is exactly the failure mode this guard
        exists to surface.
        """
        failure = {
            "status": "failed",
            "reason": "workflow_timeout",
            "error": (
                f"Workflow exceeded run_timeout_seconds="
                f"{workflow_def.get('run_timeout_seconds')} at node '{node_id}'"
            ),
            "node_id": node_id,
        }
        await self._handle_workflow_failure(project_id, failure)
        return failure

    async def _fail_on_max_iterations(
        self, project_id: str, node_id: str, max_iterations: int
    ) -> dict:
        """Fail the run when the DAG walk exhausts its iteration budget.

        Mirrors _fail_on_run_timeout: route the exit through
        _handle_workflow_failure so PROJECT_FAILED is emitted, the failure
        surfaces in chat, and project/run status cascade to failed. Before
        AppFactory-157 this exit returned a bare dict with no side effects, so the
        run was left "running" — swallowed by run_workflow (reacts only to
        'completed') and revivable by _maybe_resume_interrupted_run.

        The failure dict keeps ``reason == "workflow_max_iterations"`` (callers
        and tests key on it) and adds ``error``/``node_id`` for diagnostics —
        ``node_id`` is the node the walk was about to enter when the budget ran
        out.
        """
        failure = {
            "status": "failed",
            "reason": "workflow_max_iterations",
            "error": (
                f"Workflow exceeded max_iterations={max_iterations} "
                f"at node '{node_id}'"
            ),
            "node_id": node_id,
        }
        await self._handle_workflow_failure(project_id, failure)
        return failure

    _validator_bound_outputs = staticmethod(validator_bound_outputs)
    _bound_field_rejection = staticmethod(bound_field_rejection)

    @staticmethod
    def _find_edge(current_id: str, condition: str, edges: List[dict]) -> Optional[str]:
        """Find outgoing edge with a specific condition."""
        return find_conditioned_edge(current_id, condition, edges)

    @staticmethod
    def _find_default_edge(current_id: str, edges: List[dict]) -> Optional[str]:
        """Find outgoing edge without a condition (unconditional)."""
        return find_default_edge(current_id, edges)

    @staticmethod
    def _find_start_node_id(nodes_map: Dict[str, dict]) -> str:
        for nid, node in nodes_map.items():
            if node.get("type") == "start":
                return nid
        raise ValueError("Workflow has no start node")

    # ── DAG validation ───────────────────────────────────────────

    @staticmethod
    def validate_dag(workflow_def: dict) -> None:
        """Validate structural integrity of a workflow definition.

        Raises ``ValueError`` on problems.
        """
        nodes = workflow_def.get("nodes") or []
        edges = workflow_def.get("edges") or []

        if not nodes:
            raise ValueError("Workflow has no nodes")

        # Exactly 1 start and 1 end
        start_nodes = [n for n in nodes if n.get("type") == "start"]
        end_nodes = [n for n in nodes if n.get("type") == "end"]
        if len(start_nodes) != 1:
            raise ValueError(f"Workflow must have exactly 1 start node, found {len(start_nodes)}")
        if len(end_nodes) != 1:
            raise ValueError(f"Workflow must have exactly 1 end node, found {len(end_nodes)}")

        # All edges reference existing nodes
        node_ids = {n["id"] for n in nodes}
        for edge in edges:
            if edge["from"] not in node_ids:
                raise ValueError(f"Edge references unknown source node: {edge['from']}")
            if edge["to"] not in node_ids:
                raise ValueError(f"Edge references unknown target node: {edge['to']}")

        # Phase nodes need enough metadata to run safely.
        for n in nodes:
            for error in binding_errors(n):
                raise ValueError(error)
            if n.get("type") == "phase":
                has_task_type = bool(n.get("task_type"))
                has_description = bool(str(n.get("description") or "").strip())
                # Auction-only, mirroring the route validator: bidders judge fit by
                # these fields; a direct node names its agent and task_type defaults
                # to the node id at run time.
                if n.get("agent_selection") != "direct" and not has_task_type and not has_description:
                    raise ValueError(
                        f"Phase node '{n['id']}' runs an auction but has neither "
                        f"'task_type' nor 'description' for agents to bid on"
                    )

                if n.get("agent_selection") == "direct" and not str(n.get("agent_type") or "").strip():
                    raise ValueError(
                        f"Phase node '{n['id']}' uses direct selection but has no agent_type"
                    )
            elif n.get("type") == "a2a_agent":
                if not n.get("server_id"):
                    raise ValueError(f"A2A node '{n['id']}' must define 'server_id'")
                if "reads" not in n:
                    raise ValueError(f"A2A node '{n['id']}' must define 'reads' (can be empty list)")
                if "writes" not in n:
                    raise ValueError(f"A2A node '{n['id']}' must define 'writes' (can be empty list)")
                if not isinstance(n.get("reads"), list):
                    raise ValueError(f"A2A node '{n['id']}' 'reads' must be a list")
                if not isinstance(n.get("writes"), list):
                    raise ValueError(f"A2A node '{n['id']}' 'writes' must be a list")

                for i, write in enumerate(n.get("writes", [])):
                    if not isinstance(write, dict):
                        raise ValueError(f"A2A node '{n['id']}' writes[{i}] must be an object")
                    if "artifact_name" not in write:
                        raise ValueError(f"A2A node '{n['id']}' writes[{i}] missing 'artifact_name'")
            elif n.get("type") == "validator":
                if not n.get("checks"):
                    raise ValueError(f"Validator node '{n['id']}' has no checks")
            elif n.get("type") in ("tool", "map"):
                errors = tool_node_errors(n) if n["type"] == "tool" else map_node_errors(n)
                if errors:
                    raise ValueError(errors[0])

        # Cycle detection (ignoring gate rejected back-edges)
        WorkflowEngine._check_no_cycles(nodes, edges)

    @staticmethod
    def _check_no_cycles(nodes: list, edges: list) -> None:
        """Kahn's algorithm — raises ``ValueError`` if a cycle is detected.

        Rejected edges from ``approval_gate``, ``validator`` and ``map`` nodes
        are excluded because they represent controlled retry loops (e.g.
        reject → re-run phase), not true infinite cycles.
        """
        node_types = {n["id"]: n.get("type") for n in nodes}
        adjacency: Dict[str, List[str]] = {n["id"]: [] for n in nodes}
        in_degree: Dict[str, int] = {n["id"]: 0 for n in nodes}

        for edge in edges:
            if node_types.get(edge["from"]) in ("approval_gate", "validator", "map") and edge.get("condition") == "rejected":
                continue
            adjacency[edge["from"]].append(edge["to"])
            in_degree[edge["to"]] += 1

        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        visited = 0

        while queue:
            nid = queue.popleft()
            visited += 1
            for neighbour in adjacency[nid]:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        if visited != len(nodes):
            raise ValueError("Workflow DAG contains a cycle")

    async def _preflight_a2a_servers(self, project_id: str, workflow_def: dict) -> Optional[dict]:
        """Check + refresh every A2A server the workflow references, before the DAG runs.

        Hard-fail (return a failure result) if a referenced server is unreachable,
        disabled, missing, or serving an invalid card — so a broken dependency stops the
        project at the start instead of deep in the DAG. A reachable server whose card
        merely changed since last validation (version/skills drift) is a warning, not a
        failure: emit an event and proceed. Returns ``None`` when all servers are healthy.
        """
        server_ids = sorted({
            n["server_id"] for n in workflow_def.get("nodes", [])
            if n.get("type") == "a2a_agent" and n.get("server_id")
        })
        if not server_ids:
            return None

        project = self._get_project(project_id)
        tenant_id = project.get("tenant_id")
        run_id = project.get("run_id")
        a2a_client = A2AClientFactory.get_client(self.orch.storage)

        server_configs = {
            server_id: await self.orch.storage.get_a2a_server(server_id, tenant_id)
            for server_id in server_ids
        }
        from orchestration.checkpoints import enabled
        checkpoint_nodes = [n for n in workflow_def.get("nodes", [])
                            if n.get("type") == "a2a_agent" and enabled(server_configs.get(n.get("server_id")))]
        if len(checkpoint_nodes) > 1:
            failure = {"status": "failed", "error": "Checkpoint pilot requires a single enabled A2A node",
                       "error_type": "checkpoint_workflow_ambiguous"}
            logger.warning("[CHECKPOINT] project_id=%s run_id=%s enabled_nodes=%s — ambiguous workflow refused", project_id, run_id, len(checkpoint_nodes))
            await self._emit_a2a_event("a2a_agent_failed", project_id, run_id, failure)
            return failure

        for server_id in server_ids:
            prior = server_configs[server_id]
            prior_summary = (prior or {}).get("cached_agent_card_summary") or {}
            try:
                # Keep the preflight CAS revision stable: endpoint-card caching also writes
                # ``updated_at``, so it must not run before the guarded canonical update below.
                card = await a2a_client.get_agent_card(
                    server_id,
                    tenant_id,
                    force_refresh=True,
                    persist_endpoint_cache=False,
                )
                contract = validate_agent_card(card)
                # Refresh the canonical cached summary (what the UI and the node read) so a
                # reachable server's stored identity reflects this run, not registration time.
                # Kept inside the try so a malformed runtime card (e.g. build_a2a_card_summary
                # choking on a non-dict supportedInterfaces element) is classified as a
                # preflight failure instead of escaping uncaught with no PROJECT_FAILED event.
                cache_kwargs = {"contract": contract.as_dict()}
                if (prior or {}).get("updated_at") is not None:
                    cache_kwargs["expected_updated_at"] = prior["updated_at"]
                updated = await self.orch.storage.update_a2a_server_cache(
                    server_id,
                    tenant_id,
                    card,
                    datetime.utcnow(),
                    **cache_kwargs,
                )
                if not updated:
                    raise RuntimeError(
                        "A2A server configuration changed during preflight; retry the run"
                    )
                new_summary = (updated or {}).get("cached_agent_card_summary") \
                    or self.orch.storage.build_a2a_card_summary(card)
            except Exception as e:
                # Any failure here — unreachable, disabled, not-found, invalid/malformed
                # card — is a hard stop. The send path can't recover from a missing agent.
                error = f"A2A server '{server_id}' failed preflight: {e}"
                contract_error_type = (
                    e.error_type if isinstance(e, A2AContractError) else None
                )
                logger.error(
                    "[A2A_CONTRACT] server_id=%s error_type=%s — %s",
                    server_id,
                    contract_error_type or "a2a_preflight_failed",
                    error,
                )
                await self._emit_a2a_event(
                    event_type="a2a_agent_failed",
                    project_id=project_id,
                    run_id=run_id,
                    data={"server_id": server_id, "error": str(e), "phase": "preflight"},
                )
                failure = {
                    "status": "failed",
                    "error": error,
                    "error_type": "a2a_preflight_failed",
                    "server_id": server_id,
                }
                if contract_error_type:
                    failure["contract_error_type"] = contract_error_type
                return failure

            drift = self._card_drift(prior_summary, new_summary)
            if drift:
                logger.warning(
                    "[ENGINE] A2A server '%s' card drift since last validation: %s",
                    server_id, drift,
                )
                await self._emit_a2a_event(
                    event_type="a2a_agent_drift",
                    project_id=project_id,
                    run_id=run_id,
                    data={"server_id": server_id, "drift": drift},
                )
        return None

    @staticmethod
    def _card_drift(prior_summary: dict, new_summary: dict) -> Optional[dict]:
        """Compare two card summaries; return a dict of changes, or ``None`` if unchanged.

        No prior summary (server never validated before) is not drift — there is nothing
        to compare against — so it returns ``None``.
        """
        if not prior_summary:
            return None
        changes: Dict[str, Any] = {}
        if prior_summary.get("version") != new_summary.get("version"):
            changes["version"] = [prior_summary.get("version"), new_summary.get("version")]
        prior_skills = {s.get("id") for s in (prior_summary.get("skills") or []) if isinstance(s, dict)}
        new_skills = {s.get("id") for s in (new_summary.get("skills") or []) if isinstance(s, dict)}
        if prior_skills != new_skills:
            # key=str: skill ids come from an unvalidated runtime card (response.json(),
            # no schema coercion), so a non-conformant agent can mix types (e.g. "a" and
            # 1). A bare sorted() would raise TypeError here — and this runs outside the
            # preflight try/except, so it would crash execute with no PROJECT_FAILED.
            changes["skills_added"] = sorted((s for s in new_skills - prior_skills if s), key=str)
            changes["skills_removed"] = sorted((s for s in prior_skills - new_skills if s), key=str)
        return changes or None

    async def _run_a2a_agent_node(
            self,
            node: dict,
            project_id: str,
            prev_result: Optional[dict] = None,
            workflow_def: Optional[dict] = None,
    ) -> dict:
        """Execute A2A agent node."""
        project = self._get_project(project_id)
        shared_context = project["shared_context"]
        tenant_id = project.get("tenant_id")
        run_id = project.get("run_id")

        server_id = node.get("server_id")
        reads = node.get("reads", [])
        writes = node.get("writes", [])

        if not server_id:
            return {
                "status": "failed",
                "error": "A2A node missing 'server_id'",
                "error_type": "configuration_error"
            }

        a2a_client = A2AClientFactory.get_client(self.orch.storage)

        # Resolve configuration before the resume marker. Unknown configuration
        # cannot prove checkpoints are disabled; retain any error and fail inside
        # the node's protected execution region before an outbound submission.
        long_running = False
        poll_interval_seconds = 15
        _server_cfg = {}
        server_config_error = None
        try:
            _server_cfg = await a2a_client.get_server_config(server_id, tenant_id)
            long_running = bool(_server_cfg.get("long_running"))
            poll_interval_seconds = _server_cfg.get("poll_interval_seconds", 15)
        except Exception as exc:
            server_config_error = exc


        # A restart/reconnect resume lands here with a marker instead of a fresh call.
        # Two shapes, set by orchestrator._maybe_resume_interrupted_a2a from the two
        # a2a_task_state statuses (AppFactory-280 finding #2):
        #   {"node_id", "task_id"}    — task_id was confirmed before the interruption:
        #                                skip message building/param extraction/started
        #                                event entirely, just reconcile via tasks/get.
        #   {"node_id", "message_id"} — the process died before task_id was ever
        #                                confirmed: nothing to poll yet, so message
        #                                building runs as normal below, but the SAME
        #                                message_id is reused for the retry instead of
        #                                minting a new one.
        resume_cursor = None
        resume_marker = project.get("_resume_a2a")
        if resume_marker and resume_marker.get("node_id") == node["id"]:
            resume_cursor = resume_marker
            project.pop("_resume_a2a", None)
            # A resume marker always has a durable task cursor. Reuse the
            # submit-once-and-poll path even when the original server used the
            # standard message/send stream before it returned TASK_STATE_WORKING —
            # whether the node originally ran long_running or paused on
            # input_required via the plain send_message fast path (AppFactory-281
            # review finding: the bridge is no longer gated on long_running, see
            # the pause branch below).
            long_running = True
            resumed_task_id = resume_cursor.get("task_id")
            resumed_human_answer = resume_cursor.get("human_answer")
            if resumed_task_id and resumed_human_answer:
                # AppFactory-281: the human answered an input_required question —
                # distinct from a plain reconnect (below), since this resume DOES
                # make one new continuation call (the answer itself, continuing
                # the same task_id — never a second primary submission).
                if resume_cursor.get("reconcile_before_resend"):
                    # AppFactory-281 P1 review fix (bug #3): the PREVIOUS attempt
                    # crashed strictly between "about to dispatch" and "confirmed
                    # dispatched" — ambiguous whether the adapter ever received
                    # this answer. _a2a_submit_and_poll checks tasks/get FIRST and
                    # only (re)sends if the task is still at input_required,
                    # instead of blindly resending.
                    logger.info(
                        "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s — resuming after "
                        "a crash mid-dispatch of the human answer, reconciling via tasks/get "
                        "before deciding whether to (re)send",
                        project_id, node["id"], resumed_task_id,
                    )
                else:
                    logger.info(
                        "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s — resuming after "
                        "human answered input-required, sending answer then continuing poll",
                        project_id, node["id"], resumed_task_id,
                    )
                await self._emit_a2a_event(
                    event_type="a2a_agent_input_answered",
                    project_id=project_id,
                    run_id=run_id,
                    data={
                        "node_id": node["id"],
                        "server_id": server_id,
                        "task_id": resumed_task_id,
                    },
                )
                await self._append_a2a_chat_message(
                    shared_context,
                    f"➡️ Sending your answer to the running A2A task "
                    f"(task `{resumed_task_id}`) and resuming.",
                    {"node_id": node["id"], "server_id": server_id, "a2a": True, "resumed": True},
                )
            elif resumed_task_id:
                logger.info(
                    "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s — resuming poll after "
                    "restart/reconnect, message/send will NOT be repeated",
                    project_id, node["id"], resumed_task_id,
                )
                await self._emit_a2a_event(
                    event_type="a2a_agent_resumed",
                    project_id=project_id,
                    run_id=run_id,
                    data={
                        "node_id": node["id"],
                        "server_id": server_id,
                        "task_id": resumed_task_id,
                    },
                )
                await self._append_a2a_chat_message(
                    shared_context,
                    f"🔄 Reconnected to the running A2A task after a service restart "
                    f"(task `{resumed_task_id}`) — resuming without "
                    f"re-submitting the request.",
                    {"node_id": node["id"], "server_id": server_id, "a2a": True, "resumed": True},
                )
            else:
                # AppFactory-280 P1 (3rd re-review): NO scenario may repeat the primary
                # message/send — core A2A JSON-RPC has no lookup-by-correlation-id, so
                # there is no protocol-level way to check whether the adapter already
                # created a task for this message_id before the crash. A prior design
                # let an operator opt a specific adapter in via message_id_dedup_trusted;
                # the reviewer correctly rejected that — a manual flag is not a verifiable
                # technical guarantee of the adapter's actual behavior. Always fail closed.
                resumed_message_id = resume_cursor.get("message_id")
                return await self._fail_a2a_submission_outcome_unknown(
                    shared_context=shared_context,
                    project_id=project_id,
                    run_id=run_id,
                    node_id=node["id"],
                    server_id=server_id,
                    message_id=resumed_message_id,
                    reason="a service restart interrupted the submission before it was confirmed",
                )

        agent_summary: Dict[str, Any] = {}
        agent_name = None
        message = None
        checkpoint_binding = project.get("_checkpoint_binding")
        restored_node = (checkpoint_binding and checkpoint_binding.get("run_id") == run_id
                         and checkpoint_binding.get("node_id") == node["id"])
        outgoing_context_id = checkpoint_binding["context_id"] if restored_node else run_id
        if restored_node:
            message = deepcopy(checkpoint_binding["message"])

        if not restored_node and (not resume_cursor or not resume_cursor.get("task_id")):
            # The agent card was already fetched + refreshed by the project-start preflight
            # (_preflight_a2a_servers), so read the cached summary instead of re-fetching.
            # name/version feed only the log line and the started event; a stored read can't
            # fail an otherwise-working node on what used to be a cosmetic live fetch.
            server = await self.orch.storage.get_a2a_server(server_id, tenant_id)
            agent_summary = (server or {}).get("cached_agent_card_summary") or {}
            agent_name = agent_summary.get("name")
            logger.info(
                "[ENGINE] A2A node '%s' using agent '%s' (version %s)",
                node["id"],
                agent_name,
                agent_summary.get("version"),
            )

            message = self._build_a2a_message(reads, shared_context, prev_result)

            # Augment the message with any structured DataPart the agent's card REQUIRES
            # (e.g. IDU's scenario_id), extracted from the intent text by a schema-constrained
            # LLM call. Card-driven: agents that declare no required extension get nothing here
            # and pay no LLM call. A required param we can't extract fails the node NOW, before
            # contacting the agent, so the reason is "missing scenario_id" rather than a generic
            # downstream rejection. See docs/adr/0006-a2a-param-extraction.md.
            try:
                # message is normally a parts list, but _build_a2a_message / send_message also
                # accept a bare string (one text part); handle both so the intent is readable
                # either way. base_parts is the list form we append the DataPart(s) to.
                if isinstance(message, str):
                    intent_text = message
                    base_parts = [{"kind": "text", "text": message}]
                else:
                    intent_text = "\n".join(
                        part.get("text", "")
                        for part in message
                        if isinstance(part, dict) and part.get("kind") == "text"
                    )
                    base_parts = message
                # Keys the author already supplied as a data-read (DataParts in base_parts)
                # count as satisfied: extraction won't fail on them and won't emit a second
                # part for the same key. Without this, a required param passed as data (the
                # shape _build_a2a_message's docstring endorses) would still fail "missing".
                provided_data: Dict[str, Any] = {}
                for part in base_parts:
                    if (
                        isinstance(part, dict)
                        and part.get("kind") == "data"
                        and isinstance(part.get("data"), dict)
                    ):
                        provided_data.update(part["data"])
                fallback_models_override = getattr(
                    shared_context, "_ephemeral_fallback_models", None
                )
                if getattr(shared_context, "_force_model_override", False):
                    fallback_models_override = []
                extension_parts = await extract_extension_dataparts(
                    intent_text,
                    agent_summary.get("capabilities"),
                    self.orch.llm_client,
                    model=shared_context.get_model("agent_default"),
                    api_key_override=getattr(shared_context, "_ephemeral_api_key", None),
                    fallback_models_override=fallback_models_override,
                    already_provided=provided_data,
                )
                if extension_parts:
                    message = list(base_parts) + extension_parts
            except A2AParamExtractionError as exc:
                error_msg = f"A2A required parameter extraction failed: {exc}"
                logger.error("[ENGINE] A2A node '%s' %s", node["id"], error_msg)
                await self._emit_a2a_event(
                    event_type="a2a_agent_failed",
                    project_id=project_id,
                    run_id=run_id,
                    data={
                        "node_id": node["id"],
                        "error": error_msg,
                        "error_type": "a2a_param_extraction_failed",
                    },
                )
                await self._append_a2a_chat_message(
                    shared_context,
                    f"❌ {error_msg}",
                    {"node_id": node["id"], "server_id": server_id, "a2a": True, "error": True},
                )
                return {
                    "status": "failed",
                    "error": error_msg,
                    "error_type": "a2a_param_extraction_failed",
                }
            except Exception as exc:
                # A transient error from the extraction LLM call (e.g. RuntimeError out of
                # chat_completion) is NOT an A2AParamExtractionError, so without this it would
                # escape ahead of the node's own try/except and bypass the failure handling the
                # execute loop applies to a returned failed result — the run would hang
                # "running" with an empty chat. Fail the node cleanly instead.
                error_msg = f"A2A parameter extraction errored: {exc}"
                logger.exception("[ENGINE] A2A node '%s' %s", node["id"], error_msg)
                await self._emit_a2a_event(
                    event_type="a2a_agent_failed",
                    project_id=project_id,
                    run_id=run_id,
                    data={
                        "node_id": node["id"],
                        "error": error_msg,
                        "error_type": "a2a_param_extraction_error",
                    },
                )
                await self._append_a2a_chat_message(
                    shared_context,
                    f"❌ {error_msg}",
                    {"node_id": node["id"], "server_id": server_id, "a2a": True, "error": True},
                )
                return {
                    "status": "failed",
                    "error": error_msg,
                    "error_type": "a2a_param_extraction_error",
                }

            await self._emit_a2a_event(
                event_type="a2a_agent_started",
                project_id=project_id,
                run_id=run_id,
                data={
                    "node_id": node["id"],
                    "server_id": server_id,
                    "agent_name": agent_name,
                    "message": message,
                }
            )

        baseline = await self._capture_a2a_writes_baseline(project_id, writes)

        cancelled = False
        standard_poll_cursor_opened = False
        local_exit_requires_remote_cancel = False
        remote_cancellation_pending = False
        remote_cancellation_resolved = False
        try:
            final_task = None
            final_artifacts = []
            last_state = None
            active_task_id = None
            active_task_state = None
            active_context_id = None
            is_final = False
            message_reply_parts = None
            progress_event_ids: set[str] = set()
            # Initialized here (not just after the event loop, ~line 2397) so a
            # reference to it in `finally` below is always safe — an exception
            # raised mid-loop (e.g. A2AOutputModeError while processing the very
            # first message-only event) would otherwise leave it unbound and
            # crash `finally` with NameError (AppFactory-280 Issue 2).
            completed = False
            # Set only in the _A2ASubmissionOutcomeUnknownError handler below —
            # that path already closes the pending_submit cursor itself (via
            # _fail_a2a_submission_outcome_unknown), so `finally` must not try
            # to close it a second time.
            outcome_unknown = False
            # Set only on the new TASK_STATE_INPUT_REQUIRED branch below (AppFactory-281)
            # — that path deliberately leaves the cursor at "awaiting_human" itself,
            # so `finally` must not try to close it as if the node were done.
            paused = False
            # Set only in the _A2AAnswerReconciliationUnknownError handler below
            # — that path deliberately does NOT close the cursor: whether the
            # answer already reached the adapter is genuinely unknown, not a
            # definite outcome, so the next resume must still see
            # pending_human_answer/answer_dispatch_started_at and retry
            # reconciliation instead of `finally` treating this exit as done.
            reconciliation_unknown = False
            # Set later, right after the event loop (~"message_only_completion =
            # message_reply_parts is not None and last_state is None") — initialized
            # here too so `finally` can safely reference it even if an exception fires
            # before that line runs (same class of bug as the completed/outcome_unknown
            # init above — AppFactory-280).
            message_only_completion = False

            # Setup failures happen before scientific submission. Keep their codes
            # without catching errors from a task that may already be running.
            try:
                if server_config_error is not None:
                    raise CheckpointError("a2a_configuration_unavailable", 503, False)

                if not resume_cursor:
                    from orchestration.checkpoints import register_invocation
                    message, outgoing_context_id = await register_invocation(
                        self.orch, {**project, "project_id": project_id}, node,
                        workflow_def, message, _server_cfg,
                        traceparent=current_run_traceparent(self.tracer),
                    )
            except (CheckpointError, CheckpointConflict) as exc:
                error_type = (
                    exc.code if isinstance(exc, CheckpointError)
                    else "checkpoint_registration_conflict"
                )
                error_msg = (
                    f"A2A setup failed ({error_type}); submission refused. "
                    "Check server/checkpoint configuration and adapter availability."
                )
                logger.warning(
                    "[CHECKPOINT] project_id=%s run_id=%s node_id=%s "
                    "server_id=%s error_type=%s — submission refused during setup",
                    project_id, run_id, node["id"], server_id, error_type,
                )
                await self._emit_a2a_event(
                    "a2a_agent_failed", project_id, run_id,
                    {"node_id": node["id"], "error": error_msg, "error_type": error_type},
                )
                await self._append_a2a_chat_message(
                    shared_context, f"❌ {error_msg}",
                    {"node_id": node["id"], "server_id": server_id, "a2a": True, "error": True},
                )
                return {"status": "failed", "error": error_msg, "error_type": error_type}

            a2a_metadata = {
                "workflow_id": project.get("workflow_id"),
                "node_id": node["id"],
                "project_id": project_id,
            }
            if long_running:
                event_source = self._a2a_submit_and_poll(
                    a2a_client=a2a_client,
                    server_id=server_id,
                    tenant_id=tenant_id,
                    project_id=project_id,
                    run_id=run_id,
                    node_id=node["id"],
                    message=message,
                    context_id=outgoing_context_id,
                    skill_id=node.get("skill_id"),
                    metadata=a2a_metadata,
                    resume_cursor=resume_cursor,
                    poll_interval_seconds=poll_interval_seconds,
                    human_answer=resume_cursor.get("human_answer") if resume_cursor else None,
                )
            else:
                event_source = a2a_client.send_message(
                    server_id=server_id,
                    tenant_id=tenant_id,
                    message=message,
                    context_id=outgoing_context_id,
                    skill_id=node.get("skill_id"),
                    metadata=a2a_metadata,
                )

            event_iterator = event_source.__aiter__()
            while True:
                try:
                    event = await event_iterator.__anext__()
                except StopAsyncIteration:
                    break
                except Exception as exc:
                    # A stream can disconnect after it has already acknowledged a
                    # non-terminal task.  The task id is now durable, so adopt that
                    # task through tasks/get; repeating message/send is unsafe.
                    if (
                        not long_running
                        and active_task_id
                        and active_task_state in {
                            "TASK_STATE_SUBMITTED", "TASK_STATE_WORKING"
                        }
                    ):
                        logger.warning(
                            "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                            "action=stream_failed_fallback_to_poll error=%s",
                            project_id,
                            node["id"],
                            active_task_id,
                            exc,
                        )
                        break
                    raise

                event_type = event.get("type")
                event_data = event.get("data", {})

                if event_type == "task":
                    observed_task = event_data or {}
                    observed_task_id = observed_task.get("id")
                    observed_state = (observed_task.get("status") or {}).get("state")
                    if observed_task_id:
                        active_task_id = str(observed_task_id)
                        active_task_state = observed_state
                        active_context_id = observed_task.get("context_id")

                # A server configured for the standard message/send path may still
                # acknowledge with a non-terminal Task and require tasks/get polling.
                # Once that task id is known, persist it BEFORE processing the frame or
                # awaiting the first poll. Otherwise a backend restart leaves the remote
                # task running without a durable recovery cursor.
                if not long_running and event_type == "task":
                    task_status = event_data.get("status") or {}
                    task_state = task_status.get("state")
                    task_id = event_data.get("id")
                    if (
                        task_state in {"TASK_STATE_SUBMITTED", "TASK_STATE_WORKING"}
                        and task_id
                        and not standard_poll_cursor_opened
                    ):
                        try:
                            await self.orch.storage.create_a2a_task_state(
                                project_id=project_id,
                                run_id=run_id,
                                node_id=node["id"],
                                tenant_id=tenant_id,
                                server_id=server_id,
                                task_id=str(task_id),
                                context_id=event_data.get("context_id"),
                            )
                        except Exception:
                            logger.exception(
                                "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                                "action=persist_standard_poll_cursor status=failed",
                                project_id,
                                node["id"],
                                task_id,
                            )
                            # The outer exception handler marks this as a local exit;
                            # its finally block owns the one durable cancellation
                            # attempt below. Do not call tasks/cancel here: an empty
                            # response is not confirmation and a second immediate
                            # request would only duplicate the remote side effect.
                            raise
                        standard_poll_cursor_opened = True
                        logger.info(
                            "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s "
                            "action=persist_standard_poll_cursor status=opened",
                            project_id,
                            node["id"],
                            task_id,
                        )

                await self._process_a2a_event(
                    event=event,
                    project_id=project_id,
                    run_id=run_id,
                    node_id=node["id"],
                    shared_context=shared_context,
                    writes=writes,
                )

                if event_type == "task":
                    final_task = event_data
                    last_state = final_task.get("status", {}).get("state")
                    if final_task.get("artifacts"):
                        final_artifacts = final_task.get("artifacts", [])

                elif event_type == "status_update":
                    status = event_data.get("status") or event_data
                    state = status.get("state")
                    if state:
                        last_state = state
                        logger.debug(f"[ENGINE] A2A state updated: {state}")

                    if status.get("final", False):
                        is_final = True
                        logger.info(f"[ENGINE] A2A task reached final state: {state}")

                    if "artifacts" in event_data:
                        artifacts = event_data.get("artifacts", [])
                        if artifacts:
                            final_artifacts.extend(artifacts)

                    progress = extract_coscientist_progress(status)
                    if progress is not None:
                        if progress.event_id in progress_event_ids:
                            logger.warning(
                                "[A2A_PROGRESS] node_id=%s event_id=%s — duplicate progress event skipped",
                                node["id"], progress.event_id,
                            )
                        else:
                            progress_event_ids.add(progress.event_id)
                            progress_data = progress.model_dump(by_alias=True)
                            await self._emit_a2a_event(
                                event_type="a2a_agent_progress",
                                project_id=project_id,
                                run_id=run_id,
                                data={
                                    "node_id": node["id"],
                                    "server_id": server_id,
                                    "task_id": event_data.get("task_id"),
                                    "progress": progress_data,
                                },
                            )
                            summary = status_update_text(status)
                            if summary:
                                await self._append_a2a_chat_message(
                                    shared_context,
                                    summary,
                                    {
                                        "node_id": node["id"],
                                        "server_id": server_id,
                                        "a2a": True,
                                        "a2a_progress": True,
                                        "event_id": progress.event_id,
                                        "run_id": progress.run_id,
                                        "sequence": progress.sequence,
                                        "event_type": progress.event_type,
                                    },
                                )

                elif event_type == "artifact_update":
                    # _serialize_stream_response puts artifact_id/name/parts at the TOP
                    # level of event_data (there is no nested "artifact" key), so fall back
                    # to event_data itself. Without this, streamed artifacts never reach
                    # final_artifacts -> _save_a2a_artifacts.
                    artifact = event_data.get("artifact", event_data)
                    if artifact:
                        final_artifacts = merge_a2a_artifact_update(final_artifacts, artifact)
                        logger.debug(f"[ENGINE] Received artifact: {artifact.get('name', 'unnamed')}")

                elif event_type == "message":
                    # Non-streaming send_message can return a bare Message instead of a Task
                    # (the A2A spec allows either). The agent replied successfully; capture
                    # its parts so the terminal check treats this as completion rather than
                    # falling through to "unexpected state: None". last_state stays None
                    # because there is no Task/state here — message presence is the signal.
                    if event_data.get("parts"):
                        message_reply_parts = event_data["parts"]

            # A standard A2A agent may acknowledge message/send with a working Task
            # and expose its terminal result through tasks/get.  Treat that as the
            # normal protocol continuation, not as a CoScientist-specific lifecycle.
            if last_state in {"TASK_STATE_SUBMITTED", "TASK_STATE_WORKING"}:
                task_id = (final_task or {}).get("id")
                if not task_id:
                    raise A2AContractError(
                        "a2a_working_task_missing_id",
                        "A working A2A task must include an id for tasks/get polling",
                        location="message/send.result.id",
                    )
                try:
                    poll_interval_value = node.get("a2a_poll_interval_seconds")
                    task_timeout_value = node.get("a2a_task_timeout_seconds")
                    poll_interval = float(
                        1.0 if poll_interval_value is None else poll_interval_value
                    )
                    task_timeout = float(
                        3600.0 if task_timeout_value is None else task_timeout_value
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError("A2A polling values must be numeric") from exc
                if poll_interval <= 0 or task_timeout <= 0:
                    raise ValueError("A2A polling interval and timeout must be positive")
                deadline = time.monotonic() + task_timeout
                logger.info(
                    "[ENGINE] A2A node '%s' task_id=%s — polling standard A2A task",
                    node["id"],
                    task_id,
                )
                async for poll_event in self._a2a_poll_task(
                    a2a_client=a2a_client,
                    server_id=server_id,
                    tenant_id=tenant_id,
                    task_id=str(task_id),
                    project_id=project_id,
                    node_id=node["id"],
                    poll_interval_seconds=poll_interval,
                    deadline=deadline,
                    initial_delay=True,
                ):
                    final_task = poll_event["data"]
                    last_state = (final_task.get("status") or {}).get("state")
                    active_task_id = str(final_task.get("id") or task_id)
                    active_task_state = last_state
                    if final_task.get("artifacts"):
                        final_artifacts = final_task["artifacts"]
                    await self._process_a2a_event(
                        event={"type": "task", "data": final_task},
                        project_id=project_id,
                        run_id=run_id,
                        node_id=node["id"],
                        shared_context=shared_context,
                        writes=writes,
                    )

            # Do not expose any response-derived state until every frame has passed the
            # client's output-mode validation and the terminal outcome is successful.
            message_only_completion = message_reply_parts is not None and last_state is None
            completed = (
                last_state == "TASK_STATE_COMPLETED" or is_final or message_only_completion
            )
            if completed and not final_artifacts and message_reply_parts:
                final_artifacts = [{
                    "name": f"{node['id']}-response",
                    "parts": message_reply_parts,
                }]
            contract_result = (
                await self._validate_a2a_writes_contract(
                    project_id=project_id,
                    writes=writes,
                    baseline=baseline,
                    shared_context=shared_context,
                    staged_artifacts=final_artifacts,
                )
                if completed
                else {"fulfilled": True, "missing": [], "unchanged": []}
            )

            if not contract_result["fulfilled"]:
                logger.warning(
                    "[ENGINE] A2A node '%s' writes contract failed: missing=%s unchanged=%s",
                    node["id"],
                    contract_result["missing"],
                    contract_result["unchanged"],
                )
                contract_error = f"Writes contract not satisfied: missing {contract_result['missing']}"
                await self._emit_a2a_event(
                    event_type="a2a_agent_failed",
                    project_id=project_id,
                    run_id=run_id,
                    data={
                        "node_id": node["id"],
                        "error": contract_error,
                        "error_type": "writes_contract_violation",
                    },
                )
                await self._append_a2a_chat_message(
                    shared_context,
                    f"❌ {contract_error}",
                    {"node_id": node["id"], "server_id": server_id, "a2a": True, "error": True},
                )
                return {
                    "status": "failed",
                    "error": contract_error,
                    "error_type": "writes_contract_violation",
                    "missing_artifacts": contract_result["missing"],
                    "unchanged_artifacts": contract_result["unchanged"],
                }

            if completed and final_artifacts:
                await self._write_a2a_artifacts_to_context(
                    project_id=project_id,
                    artifacts=final_artifacts,
                    writes=writes,
                )

            if completed and final_artifacts:
                await self._save_a2a_artifacts(
                    project_id=project_id,
                    run_id=run_id,
                    artifacts=final_artifacts,
                    node_id=node["id"],
                )

            if completed:
                await self._emit_a2a_event(
                    event_type="a2a_agent_completed",
                    project_id=project_id,
                    run_id=run_id,
                    data={
                        "node_id": node["id"],
                        "artifacts_count": len(final_artifacts),
                        "final_state": last_state,
                    }
                )

                # Put the agent's reply in the chat. Without this the Chat tab shows
                # the user's prompt and nothing back: the prose reply and the layers
                # only existed as downloadable artifacts. Prose first, then links to
                # those artifacts so the conversation references what the tab holds.
                response_text = a2a_response_text(final_artifacts)
                if not response_text and message_reply_parts:
                    # Message-only reply (no artifacts): its parts carry the prose. Reuse the
                    # same text reconstruction by wrapping them as a single pseudo-artifact.
                    response_text = a2a_response_text(
                        [{"parts": message_reply_parts}]
                    )
                artifact_paths = a2a_artifact_paths(
                    final_artifacts, a2a_artifact_scope(run_id, node["id"])
                )
                message_sections = []
                if response_text:
                    message_sections.append(response_text)
                if artifact_paths:
                    message_sections.append(
                        "**Generated artifacts:**\n"
                        + "\n".join(f"- `{p}`" for p in artifact_paths)
                    )
                chat_content = "\n\n".join(message_sections) or (
                    f"A2A agent '{agent_name or server_id}' completed "
                    f"with {len(final_artifacts)} artifact(s)."
                )
                await self._append_a2a_chat_message(
                    shared_context,
                    chat_content,
                    {"node_id": node["id"], "server_id": server_id,
                     "agent_name": agent_name, "a2a": True},
                )

                return {
                    "status": "completed",
                    "node_id": node["id"],
                    "artifacts": final_artifacts,
                    "final_state": last_state,
                }
            else:
                if last_state == "TASK_STATE_INPUT_REQUIRED" and (final_task or {}).get("id"):
                    # AppFactory-281: bridge into a human question instead of failing —
                    # applies regardless of long_running (review finding: the ticket
                    # doesn't scope this to long_running; long_running only changes HOW
                    # the node talks to the adapter, not whether it can pause on a
                    # question). A node that took the plain send_message fast path
                    # (long_running=False) has no a2a_task_state cursor yet at this
                    # point — create one lazily below so the pause survives a restart
                    # exactly like a long_running node's already does. The `and
                    # (final_task or {}).get("id")` guard is defensive: a Task-typed
                    # event always carries an id in practice, but without one there is
                    # nothing durable to pause on — falls through to the unchanged
                    # typed-failure branch below.
                    task_id = final_task["id"]
                    context_id = (final_task or {}).get("context_id") or run_id
                    status_message = (final_task or {}).get("status", {}).get("message")
                    question_text = self._a2a_status_message_text(status_message)
                    question_message_id = self._a2a_status_message_id(status_message)
                    meta = self._a2a_status_message_metadata(status_message)
                    role = meta.get("role")
                    step = meta.get("step")

                    existing_cursor = await self.orch.storage.get_a2a_task_state(
                        project_id=project_id, node_id=node["id"], run_id=run_id,
                    )
                    if existing_cursor is None:
                        # Only the long_running/_a2a_submit_and_poll path creates a
                        # cursor up front (AppFactory-280). task_id is already truthy here,
                        # so this lands directly at "in_flight" (mongo_backend.py's
                        # create_a2a_task_state), which is what makes the scoped
                        # mark_a2a_task_awaiting_human below able to transition it.
                        await self.orch.storage.create_a2a_task_state(
                            project_id=project_id,
                            run_id=run_id,
                            node_id=node["id"],
                            tenant_id=tenant_id,
                            server_id=server_id,
                            task_id=task_id,
                            context_id=context_id,
                            message_id=str(uuid.uuid4()),
                        )

                    await self.orch.storage.mark_a2a_task_awaiting_human(
                        project_id=project_id,
                        run_id=run_id,
                        node_id=node["id"],
                        question_text=question_text,
                        question_message_id=question_message_id,
                        role=role,
                        step=step,
                    )
                    # Deliberately NOT task_id=/workflow_node_id= (BaseAgent-attempt
                    # fields) — find_interrupted_attempt (agents/run_rehydration.py)
                    # only treats a tool_call as a resumable "attempt" when it carries
                    # workflow_node_id. Omitting it keeps this card invisible to
                    # _maybe_resume_interrupted_run, which would otherwise treat a
                    # hanging a2a_human_input call as an ordinary crashed tool and
                    # close it with a synthetic error result.
                    await self.orch.message_store.append_tool_call(
                        project_id,
                        tool_call_id=task_id,
                        name="a2a_human_input",
                        arguments=json.dumps({
                            "question": question_text,
                            "role": role,
                            "step": step,
                            "agent_name": agent_name,
                            "server_id": server_id,
                        }),
                        run_id=run_id,
                        agent_id=agent_name,
                    )
                    await self._emit_a2a_event(
                        event_type="a2a_agent_input_required",
                        project_id=project_id,
                        run_id=run_id,
                        data={
                            "node_id": node["id"],
                            "server_id": server_id,
                            "task_id": task_id,
                            "question": question_text,
                            "role": role,
                            "step": step,
                        },
                    )
                    paused = True
                    return {
                        "status": "paused",
                        "reason": "a2a_input_required",
                        "node_id": node["id"],
                        "task_id": task_id,
                    }

                if last_state == "TASK_STATE_INPUT_REQUIRED":
                    # Reached only when the bridge above couldn't run — a Task-typed
                    # event reporting this state with no `id` (protocol-improbable, kept
                    # as a defensive fallback rather than assumed impossible). Without a
                    # task_id there is nothing durable to pause on, so this still fails
                    # with the same distinct error_type as before AppFactory-281.
                    error_msg = "A2A agent requires additional input (input_required)"
                    error_type = "a2a_input_required"
                elif last_state == "TASK_STATE_FAILED":
                    error_msg = "A2A task explicitly failed"
                    error_type = "unexpected_final_state"
                elif last_state == "TASK_STATE_CANCELED":
                    error_msg = "A2A task was cancelled"
                    error_type = "unexpected_final_state"
                else:
                    error_msg = f"A2A task ended with unexpected state: {last_state}"
                    error_type = "unexpected_final_state"

                # Append the agent's OWN reason when it sent one. Non-streaming returns a
                # Task carrying status.message; streaming status_update frames don't, so
                # this is best-effort and stays the generic label when absent.
                reason = self._a2a_status_message_text(
                    (final_task or {}).get("status", {}).get("message")
                )
                if reason:
                    error_msg = f"{error_msg}: {reason}"

                logger.error(f"[ENGINE] {error_msg}")

                await self._emit_a2a_event(
                    event_type="a2a_agent_failed",
                    project_id=project_id,
                    run_id=run_id,
                    data={
                        "node_id": node["id"],
                        "error": error_msg,
                        "error_type": error_type,
                        "final_state": last_state,
                        "is_final": is_final,
                    }
                )
                await self._append_a2a_chat_message(
                    shared_context,
                    f"❌ {error_msg}",
                    {"node_id": node["id"], "server_id": server_id, "a2a": True, "error": True},
                )
                return {
                    "status": "failed",
                    "error": error_msg,
                    "error_type": error_type,
                    "final_state": last_state,
                    "is_final": is_final,
                }

        except TaskNotFoundError as e:
            # AppFactory-280 acceptance criterion #4: the adapter's own task store lost the
            # task (its process restarted — InMemoryTaskStore-backed adapters like
            # CoScientist's don't survive that) — not recoverable by retrying tasks_get,
            # fail cleanly instead of burning the poll-error budget on a lost cause.
            last_state = "a2a_task_not_found"
            error_msg = f"A2A task not found on the adapter (it may have restarted): {e}"
            logger.error("[A2A_RECOVER] node_id=%s task not found — %s", node["id"], error_msg)
            await self._emit_a2a_event(
                event_type="a2a_agent_failed",
                project_id=project_id,
                run_id=run_id,
                data={"node_id": node["id"], "error": error_msg, "error_type": "a2a_task_not_found"},
            )
            await self._append_a2a_chat_message(
                shared_context,
                f"❌ {error_msg}",
                {"node_id": node["id"], "server_id": server_id, "a2a": True, "error": True},
            )
            return {
                "status": "failed",
                "error": error_msg,
                "error_type": "a2a_task_not_found",
            }
        except _A2APollExhaustedError as e:
            # tasks/get failed _A2A_MAX_CONSECUTIVE_POLL_ERRORS times in a row — the task
            # may still be alive on the MAS side, we just can't currently confirm it.
            last_state = "a2a_reconcile_exhausted"
            local_exit_requires_remote_cancel = True
            error_msg = f"A2A adapter unreachable after repeated reconciliation attempts: {e}"
            logger.error("[A2A_RECOVER] node_id=%s poll exhausted — %s", node["id"], error_msg)
            await self._emit_a2a_event(
                event_type="a2a_agent_failed",
                project_id=project_id,
                run_id=run_id,
                data={"node_id": node["id"], "error": error_msg, "error_type": "a2a_reconcile_exhausted"},
            )
            await self._append_a2a_chat_message(
                shared_context,
                f"❌ {error_msg}",
                {"node_id": node["id"], "server_id": server_id, "a2a": True, "error": True},
            )
            return {
                "status": "failed",
                "error": error_msg,
                "error_type": "a2a_reconcile_exhausted",
            }
        except _A2ASubmissionOutcomeUnknownError as e:
            # AppFactory-280 P1 (3rd re-review): a network exception during the
            # ORIGINAL (non-resume) submission is the same unconfirmed-outcome
            # ambiguity as a restart-interrupted resume — must not be downgraded
            # to a generic execution_failed that leaves the pending_submit cursor
            # dangling and blocks future A2A rehydration (orchestrator.py treats
            # "failed" projects as terminal). Fail closed the same way.
            outcome_unknown = True
            return await self._fail_a2a_submission_outcome_unknown(
                shared_context=shared_context,
                project_id=project_id,
                run_id=run_id,
                node_id=node["id"],
                server_id=server_id,
                message_id=e.message_id,
                reason="a network error interrupted the submission before it was confirmed",
            )
        except _A2AAnswerReconciliationUnknownError as e:
            # AppFactory-281 P1 review fix (bug #3, 4th follow-up + 8th finding):
            # covers both "tasks/get itself failed while reconciling a
            # crash-mid-dispatch answer" and "a FRESH continue_task call
            # itself raised" — neither proves the answer wasn't delivered
            # (see the class docstring above). Deliberately do NOT close the
            # cursor — it stays exactly as it was (pending_human_answer and
            # answer_dispatch_started_at both still set) so the next resume
            # attempt reconciles again instead of guessing.
            #
            # status is deliberately NOT "failed": that would route through
            # execute()'s _is_failure -> _handle_workflow_failure and flip
            # project/run to terminal "failed", which permanently blocks
            # _maybe_resume_interrupted_a2a's own terminal-status guard from
            # ever retrying the reconciliation this branch just asked for —
            # exactly the bug the reviewer found. "reconciliation_pending" is
            # a non-terminal park, same family as "paused"/"external_waiting".
            reconciliation_unknown = True
            error_msg = f"Could not confirm delivery of the human's answer: {e.reason}"
            logger.error("[A2A_RECOVER] node_id=%s task_id=%s — %s", node["id"], e.task_id, error_msg)
            await self._emit_a2a_event(
                event_type="a2a_agent_reconciliation_pending",
                project_id=project_id,
                run_id=run_id,
                data={
                    "node_id": node["id"],
                    "error": error_msg,
                    "error_type": "a2a_answer_reconciliation_unknown",
                    "task_id": e.task_id,
                },
            )
            await self._append_a2a_chat_message(
                shared_context,
                f"⚠️ {error_msg} — will retry reconciliation on the next resume "
                f"instead of resending the answer blindly.",
                {"node_id": node["id"], "server_id": server_id, "a2a": True, "reconciliation_pending": True},
            )
            return {
                "status": "reconciliation_pending",
                "error": error_msg,
                "error_type": "a2a_answer_reconciliation_unknown",
            }
        except (A2AOutputModeError, A2AContractError) as e:
            local_exit_requires_remote_cancel = True
            logger.error(
                "[A2A_CONTRACT] node_id=%s error_type=%s location=%s — %s",
                node["id"],
                e.error_type,
                e.location,
                e,
            )
            await self._emit_a2a_event(
                event_type="a2a_agent_failed",
                project_id=project_id,
                run_id=run_id,
                data={"node_id": node["id"], **e.as_dict()},
            )
            await self._append_a2a_chat_message(
                shared_context,
                f"❌ A2A output contract error: {e}",
                {"node_id": node["id"], "server_id": server_id, "a2a": True, "error": True},
            )
            return {
                "status": "failed",
                "error": str(e),
                "error_type": e.error_type,
            }
        except asyncio.CancelledError:
            # A user Stop or run_timeout_seconds cancels this Task for real and must
            # request remote cancellation. A CancelledError can also leak from a
            # torn-down scope without cancelling this Task; it must not cause a
            # destructive remote side effect. Both cases are re-raised unchanged.
            project = self.orch.active_projects.get(project_id) or {}
            genuine_cancel = is_genuinely_cancelled(project.get("token"))
            cancelled = True
            local_exit_requires_remote_cancel = genuine_cancel
            if not genuine_cancel:
                logger.warning(
                    "[A2A_RECOVER] project_id=%s node_id=%s "
                    "action=remote_cancel_skipped reason=spurious_cancelled_error",
                    project_id,
                    node["id"],
                )
            raise
        except Exception as e:
            local_exit_requires_remote_cancel = True
            logger.exception("[ENGINE] A2A node '%s' execution failed: %s", node["id"], e)

            await self._emit_a2a_event(
                event_type="a2a_agent_failed",
                project_id=project_id,
                run_id=run_id,
                data={
                    "node_id": node["id"],
                    "error": str(e),
                }
            )

            await self._append_a2a_chat_message(
                shared_context,
                f"❌ A2A agent error: {e}",
                {"node_id": node["id"], "server_id": server_id, "a2a": True, "error": True},
            )

            return {
                "status": "failed",
                "error": str(e),
                "error_type": "execution_failed",
            }
        finally:
            if (
                local_exit_requires_remote_cancel
                and active_task_id
                and active_task_state in {
                    "TASK_STATE_SUBMITTED", "TASK_STATE_WORKING"
                }
            ):
                remote_cancellation_resolved = await self.orch.request_a2a_cancellation_delivery(
                    a2a_client=a2a_client,
                    project_id=project_id,
                    run_id=run_id,
                    node_id=node["id"],
                    tenant_id=tenant_id,
                    server_id=server_id,
                    task_id=active_task_id,
                    context_id=active_context_id,
                    reason="local_workflow_exit",
                )
                remote_cancellation_pending = not remote_cancellation_resolved

            # AppFactory-280: close the durable cursor only on a clean in-process exit
            # (success, business failure, or a caught exception turned into a typed
            # failed dict) — a hard process kill never reaches this finally at all,
            # which is exactly when the cursor must stay "in_flight" for the next
            # ensure_workflow_running to reconcile. Re-fetches from storage rather than
            # tracking a local flag so it self-heals regardless of which of the many
            # exits above was taken. `not cancelled` covers the same "process didn't
            # really finish this attempt" case for an in-process CancelledError, which —
            # unlike a hard kill — DOES reach this finally (AppFactory-280 finding #3).
            if (
                (long_running or standard_poll_cursor_opened)
                and not cancelled
                and not outcome_unknown
                and not paused
                and not reconciliation_unknown
                and not remote_cancellation_pending
                and not remote_cancellation_resolved
            ):
                cursor = await self.orch.storage.get_a2a_task_state(
                    project_id=project_id, node_id=node["id"], run_id=run_id,
                )
                if cursor and cursor.get("status") == "in_flight":
                    # AppFactory-281: a message-only reply to an answered input_required
                    # question (no Task, just a bare Message — a valid A2A shape, same
                    # as the original submission's own message-only path below) leaves
                    # last_state=None even though the node genuinely completed. Recording
                    # that as final_status="error" would permanently mislabel a success
                    # in the audit trail — found by re-reading this exact interaction,
                    # not by any test, so worth being extra literal about it here.
                    await self.orch.storage.close_a2a_task_state(
                        project_id=project_id,
                        node_id=node["id"],
                        run_id=run_id,
                        final_status="message_only_completed" if message_only_completion else (last_state or "error"),
                    )
                elif cursor and cursor.get("status") == "pending_submit":
                    # AppFactory-280 Issue 2: task_id was never confirmed because
                    # submit_and_track returned a bare Message (a valid A2A
                    # outcome, not an unresolved one) — the ONLY way a pending_submit
                    # cursor is still open here is a resolved attempt (message-only
                    # completion, or a downstream failure after it e.g. an
                    # A2AContractError from _process_a2a_event). A genuinely unknown
                    # outcome never reaches this point: it already closed the cursor
                    # itself via _fail_a2a_submission_outcome_unknown (outcome_unknown
                    # guard above). Leaving this one open would make the NEXT restart's
                    # ensure_workflow_running() treat this already-resolved attempt as
                    # unconfirmed and fail it with a2a_submission_outcome_unknown.
                    await self.orch.storage.close_a2a_task_state(
                        project_id=project_id,
                        node_id=node["id"],
                        run_id=run_id,
                        final_status="message_only_completed" if completed else (last_state or "message_only_failed"),
                    )

    async def _a2a_submit_and_poll(
            self,
            *,
            a2a_client,
            server_id: str,
            tenant_id: str,
            project_id: str,
            run_id: Optional[str],
            node_id: str,
            message,
            context_id: Optional[str],
            skill_id: Optional[str],
            metadata: Dict[str, Any],
            resume_cursor: Optional[Dict[str, Any]],
            poll_interval_seconds: int,
            human_answer: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield task snapshots shaped exactly like a2a_client.send_message's events —
        {"type": "task", "data": <serialized task>} — so the caller's existing
        accumulation loop needs no changes.

        Submits ONCE, persisting a correlation id BEFORE the network call even
        starts (mirrors ADR-0008's "journal before execution" principle, and
        closes AppFactory-280 finding #2 — a crash between the adapter accepting
        the task and this process recording its task_id used to leave nothing
        to resume from). Two cases, discriminated by resume_cursor:
          - no resume_cursor: brand-new attempt — mint a message_id, persist a
            pending_submit cursor, then submit. If submit_and_track itself raises
            (network drop before task_id was confirmed), that's re-raised as
            _A2ASubmissionOutcomeUnknownError — the caller fails closed instead of
            guessing whether the adapter created the task (AppFactory-280 P1, 3rd
            re-review — no scenario may repeat the primary message/send).
          - resume_cursor has task_id: already confirmed by a prior attempt —
            skip submission entirely, poll directly. (A resume_cursor carrying
            only a message_id, no task_id, never reaches this function — that
            case fails closed earlier, in _run_a2a_agent_node, before any message
            is built.)
        Polls tasks_get on that durable cursor instead of holding one open
        connection for the task's whole lifetime (AppFactory-280 plan §3) — this is
        what makes the normal path and the reconnect path the SAME code below,
        not two divergent ones.

        human_answer (AppFactory-281, review-fixed): when resume_cursor carries one,
        it is delivered via continue_task (non-blocking, unlike the old blocking
        send_message call) before falling into the same poll loop. If
        resume_cursor also carries reconcile_before_resend (P1 bug #3 fix), the
        previous attempt crashed at an ambiguous point mid-dispatch — this
        reconciles via one tasks_get call first, and only actually (re)sends if
        the task is still sitting at input_required (i.e. nothing progressed
        without our help in the meantime).
        """
        task_id = resume_cursor.get("task_id") if resume_cursor else None
        if not task_id:
            message_id = str(uuid.uuid4())
            await self.orch.storage.create_a2a_task_state(
                project_id=project_id,
                run_id=run_id,
                node_id=node_id,
                tenant_id=tenant_id,
                server_id=server_id,
                task_id=None,
                message_id=message_id,
            )

            try:
                submitted = await a2a_client.submit_and_track(
                    server_id=server_id,
                    tenant_id=tenant_id,
                    message=message,
                    context_id=context_id,
                    metadata=metadata,
                    skill_id=skill_id,
                    message_id=message_id,
                )
            except Exception as exc:
                raise _A2ASubmissionOutcomeUnknownError(message_id, exc) from exc
            if submitted["task_id"] is None:
                # Finished before the server needed to answer (bare Message reply) —
                # nothing to poll, feed the single event straight through. The
                # pending_submit cursor is left as-is (not closed here) — same as
                # every other exit from this generator; _run_a2a_agent_node's
                # finally handles in_flight closure, pending_submit is deliberately
                # left for a human/future cleanup pass rather than guessed at.
                yield submitted["event"]
                return
            task_id = submitted["task_id"]
            await self.orch.storage.mark_a2a_task_submitted(
                project_id=project_id,
                run_id=run_id,
                node_id=node_id,
                task_id=task_id,
                context_id=submitted["context_id"],
            )
            yield submitted["event"]
            if _a2a_task_is_terminal(submitted["event"]):
                return

        if human_answer:
            # AppFactory-281 P1 review fix: continuing an existing task must be a
            # non-blocking round trip (continue_task, return_immediately=True),
            # not the old send_message(task_id=...) call — that one hardcoded
            # return_immediately=False and blocked on request_timeout_seconds
            # (default 60s), so a legitimate "still working" reply from the agent
            # after accepting the answer got misread as the adapter rejecting it.
            # This now mirrors the ORIGINAL submission's shape exactly (see
            # submit_and_track above): one non-blocking send, then fall through
            # into the SAME poll loop below for the real outcome.
            answer_message_id = resume_cursor.get("answer_message_id") if resume_cursor else None
            skip_dispatch = False

            if resume_cursor and resume_cursor.get("reconcile_before_resend"):
                # AppFactory-281 P1 bug #3: the PREVIOUS attempt crashed strictly
                # between "about to dispatch the answer" (mark_a2a_task_answer_
                # dispatching) and "confirmed dispatched" (mark_a2a_task_answer_
                # delivered) — genuinely ambiguous whether the adapter ever
                # received it. Blindly resending here is the exact bug (the
                # answer going out twice for one human answer). Reconcile first:
                # if the task has already moved past input_required, OR it is
                # still input_required but now asking a DIFFERENT question (P1
                # bug #3 follow-up — a task can ask several questions in a row;
                # state equality alone doesn't prove nothing progressed), the
                # previous dispatch demonstrably landed — mark it delivered and
                # fall straight into the normal poll loop below (which re-checks
                # this same snapshot on its first iteration) WITHOUT sending
                # again. Only a still-pending SAME question (same message_id)
                # means nothing progressed, so it's safe to (re)send below,
                # reusing the SAME answer_message_id.
                #
                # A failed reconciliation call must NOT fall through to an
                # unconditional resend (P1 review fix, bug #3 3rd follow-up) —
                # a transient tasks/get outage is not proof the previous
                # dispatch failed, and resending on that basis is exactly the
                # ungrounded guess this whole mechanism exists to avoid. Retry
                # with the same bounded budget/backoff the main poll loop uses
                # (self._A2A_MAX_CONSECUTIVE_POLL_ERRORS); a task genuinely not
                # found isn't worth retrying at all. Either way, exhaustion
                # fails closed via _A2AAnswerReconciliationUnknownError instead
                # of resending.
                snapshot = None
                reconcile_errors = 0
                while True:
                    try:
                        snapshot = await a2a_client.tasks_get(server_id, tenant_id, task_id)
                        break
                    except TaskNotFoundError as exc:
                        raise _A2AAnswerReconciliationUnknownError(task_id, str(exc)) from exc
                    except Exception as exc:
                        reconcile_errors += 1
                        if reconcile_errors > self._A2A_MAX_CONSECUTIVE_POLL_ERRORS:
                            raise _A2AAnswerReconciliationUnknownError(task_id, str(exc)) from exc
                        backoff = min(
                            poll_interval_seconds * (2 ** (reconcile_errors - 1)),
                            self._A2A_POLL_BACKOFF_CAP_SECONDS,
                        )
                        logger.warning(
                            "[A2A_RECOVER] server_id=%s task_id=%s reconcile tasks/get "
                            "failed (%d/%d), retrying in %ss before deciding whether to "
                            "resend the human answer — %s",
                            server_id, task_id, reconcile_errors,
                            self._A2A_MAX_CONSECUTIVE_POLL_ERRORS, backoff, exc,
                        )
                        await asyncio.sleep(backoff)
                snapshot_status = (snapshot or {}).get("status") or {}
                snapshot_state = snapshot_status.get("state")
                snapshot_message_id = self._a2a_status_message_id(snapshot_status.get("message"))
                answered_question_message_id = resume_cursor.get("answered_question_message_id")
                already_progressed = bool(snapshot_state) and (
                    snapshot_state != "TASK_STATE_INPUT_REQUIRED"
                    or (
                        answered_question_message_id is not None
                        and snapshot_message_id is not None
                        and snapshot_message_id != answered_question_message_id
                    )
                )
                if already_progressed:
                    await self.orch.storage.mark_a2a_task_answer_delivered(
                        project_id=project_id, run_id=run_id, node_id=node_id,
                    )
                    skip_dispatch = True

            if not skip_dispatch:
                await self.orch.storage.mark_a2a_task_answer_dispatching(
                    project_id=project_id, run_id=run_id, node_id=node_id,
                )
                try:
                    continued = await a2a_client.continue_task(
                        server_id=server_id,
                        tenant_id=tenant_id,
                        message=[{"kind": "text", "text": human_answer}],
                        task_id=task_id,
                        context_id=context_id,
                        metadata=metadata,
                        skill_id=skill_id,
                        message_id=answer_message_id,
                    )
                except Exception as exc:
                    # AppFactory-281 P1 review fix (8th finding): NOT a definite
                    # rejection — continue_task shares the identical
                    # _send_once_and_track transport as the original
                    # submission, where the same exception is already treated
                    # as unknown-not-rejected. mark_a2a_task_answer_dispatching
                    # above already left answer_dispatch_started_at set, so the
                    # next resume naturally reconciles via tasks/get instead of
                    # this run being failed over what might be a transient
                    # network blip.
                    raise _A2AAnswerReconciliationUnknownError(task_id, str(exc)) from exc
                await self.orch.storage.mark_a2a_task_answer_delivered(
                    project_id=project_id, run_id=run_id, node_id=node_id,
                )
                yield continued["event"]
                if _a2a_task_is_terminal(continued["event"]):
                    return

        async for event in self._a2a_poll_task(
            a2a_client=a2a_client,
            server_id=server_id,
            tenant_id=tenant_id,
            task_id=task_id,
            project_id=project_id,
            node_id=node_id,
            poll_interval_seconds=poll_interval_seconds,
        ):
            yield event

    async def _a2a_poll_task(
        self,
        *,
        a2a_client,
        server_id: str,
        tenant_id: str,
        task_id: str,
        project_id: str,
        node_id: str,
        poll_interval_seconds: float,
        deadline: float | None = None,
        initial_delay: bool = False,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Poll a known A2A task without ever submitting the user message again."""
        consecutive_errors = 0
        poll_count = 0
        wait_seconds = poll_interval_seconds if initial_delay else 0.0
        wait_before_poll = initial_delay

        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"A2A task '{task_id}' did not finish within its local timeout"
                    )
                if wait_before_poll:
                    wait_seconds = min(wait_seconds, remaining)

            if wait_before_poll:
                await asyncio.sleep(wait_seconds)
                wait_seconds = 0.0
                wait_before_poll = False

            try:
                snapshot = await a2a_client.tasks_get(server_id, tenant_id, task_id)
                if not snapshot:
                    raise RuntimeError("tasks_get returned an empty response")
            except TaskNotFoundError:
                raise
            except Exception as exc:
                consecutive_errors += 1
                if consecutive_errors > self._A2A_MAX_CONSECUTIVE_POLL_ERRORS:
                    raise _A2APollExhaustedError(
                        f"tasks/get failed {consecutive_errors} times in a row "
                        f"for task '{task_id}': {exc}"
                    ) from exc
                wait_seconds = min(
                    poll_interval_seconds * (2 ** (consecutive_errors - 1)),
                    self._A2A_POLL_BACKOFF_CAP_SECONDS,
                )
                logger.warning(
                    "[A2A_RECOVER] server_id=%s task_id=%s poll failed (%d/%d), "
                    "retrying in %ss — %s",
                    server_id,
                    task_id,
                    consecutive_errors,
                    self._A2A_MAX_CONSECUTIVE_POLL_ERRORS,
                    wait_seconds,
                    exc,
                )
                wait_before_poll = True
                continue

            consecutive_errors = 0
            poll_count += 1
            event = {"type": "task", "data": snapshot}
            state = (snapshot.get("status") or {}).get("state")
            logger.debug(
                "[A2A_RECOVER] project_id=%s node_id=%s task_id=%s poll_attempt=%d state=%s",
                project_id,
                node_id,
                task_id,
                poll_count,
                state,
            )
            yield event
            if _a2a_task_is_terminal(event):
                return
            wait_seconds = poll_interval_seconds
            wait_before_poll = True

    def _build_a2a_message(
            self,
            reads: List[Dict[str, str]],
            shared_context,
            prev_result: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        """Build A2A message parts from SharedContext values per the reads config.

        Each read is routed by its optional ``part`` field:
          - "text" (default): appended as a ``"key: value"`` line to a single TextPart.
            Text is the safe default because every A2A agent accepts it.
          - "data": merged, under ``key``, into a single DataPart's JSON object. Opt-in,
            for agents that declare structured input — flipping one read to ``"data"``
            is the whole change; no client or wire-format edit.

        A read's source is either a context key (``context_key``, default ``key``) or a
        literal ``value`` carried in the spec. A literal is for a constant the author fixes
        at authoring time when nothing upstream writes the key — e.g. a standalone A2A node
        that must still emit a structured ``scenario_id`` with no context source to read.

        Returns a list of part dicts (``{"kind":"text",...}`` / ``{"kind":"data",...}``)
        for the A2A client to serialize. Never agent-specific: the workflow author, not
        this code, decides text vs data per read.
        """
        text_lines: List[str] = []
        data_obj: Dict[str, Any] = {}

        for read_spec in reads:
            # The DAG validator checks ``reads`` is a list but not its element shape, so a
            # hand-authored/imported/bundled workflow can slip in a bare string. Skip it
            # rather than let ``.get`` raise AttributeError and kill the whole node.
            if not isinstance(read_spec, dict):
                logger.warning("[ENGINE] Skipping non-dict A2A read spec: %r", read_spec)
                continue
            key = read_spec.get("key")

            # A literal ``value`` short-circuits the context lookup: it's a constant the
            # author injected, not something stored in SharedContext. Checked by key
            # presence (not truthiness) so an intentional 0/""/False still sends.
            if "value" in read_spec:
                value = read_spec["value"]
            else:
                # ``or key`` (not a get-default): the UI always sends context_key, blank as
                # "", which would otherwise override the fallback and read the empty key.
                context_key = read_spec.get("context_key") or key
                value = None
                if hasattr(shared_context, "read_context_key"):
                    value = shared_context.read_context_key(context_key)
                elif hasattr(shared_context, "get"):
                    value = shared_context.get(context_key)

            if value is None:
                continue

            if read_spec.get("part") == "data":
                data_obj[key] = value
            else:
                text_lines.append(f"{key}: {value}")

        if prev_result and prev_result.get("status") == "completed":
            prev_output = prev_result.get("output", prev_result)
            if prev_output and prev_output != {"status": "completed"}:
                text_lines.append(f"previous_result: {prev_output}")

        parts: List[Dict[str, Any]] = []
        if text_lines:
            parts.append({"kind": "text", "text": "\n".join(text_lines)})
        if data_obj:
            parts.append({"kind": "data", "data": data_obj})

        if not parts:
            parts.append({"kind": "text", "text": "Execute workflow task"})

        return parts

    @staticmethod
    def _a2a_status_message_text(status_message: Optional[dict]) -> str:
        """Extract the agent's own explanation from an A2A TaskStatus.message.

        On a non-success final state the agent may attach a human-readable reason
        (e.g. IDU's "scenario not found"). Without surfacing it the monitor shows only
        our generic label and the real cause is invisible. Returns "" when absent so
        callers can cheaply guard before appending. Parts use the serialized shape
        {"type":"text","text":...} (see a2a_client._serialize_parts)."""
        if not isinstance(status_message, dict):
            return ""
        texts = [
            p.get("text", "")
            for p in (status_message.get("parts") or [])
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return " ".join(t.strip() for t in texts if t.strip()).strip()

    @staticmethod
    def _a2a_status_message_metadata(status_message: Optional[dict]) -> Dict[str, Any]:
        """Best-effort role/step off an A2A TaskStatus.message (AppFactory-281).

        Message.metadata is a supported, open A2A object (see
        a2a_client._serialize_message) — role/step on an input_required question
        are read from it if the adapter sends them, absent otherwise ("если
        переданы" — neither key is required). Own convention, not dictated by
        any existing contract: this platform also owns the acceptance mock's
        shape, since the scientists' side has 0 lines on input_required today."""
        if not isinstance(status_message, dict):
            return {}
        metadata = status_message.get("metadata")
        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _a2a_status_message_id(status_message: Optional[dict]) -> Optional[str]:
        """The adapter's own message_id for a TaskStatus.message, when present
        (AppFactory-281 P1 review fix, bug #3 follow-up).

        Unlike role/step above, message_id is a REQUIRED A2A protocol field on
        every Message (a2a_client._serialize_message always includes it) — not
        our own convention. One task can ask several questions in a row, each
        as its own Message; only this id (not question text, not state alone)
        proves whether a later TASK_STATE_INPUT_REQUIRED snapshot is still the
        SAME unanswered question or the adapter already moved on to a new one
        after accepting a prior answer."""
        if not isinstance(status_message, dict):
            return None
        message_id = status_message.get("message_id")
        return message_id if isinstance(message_id, str) and message_id else None

    async def _capture_a2a_writes_baseline(
            self,
            project_id: str,
            writes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Capture baseline values for writes contract."""
        baseline = {}
        project = self._get_project(project_id)
        shared_context = project["shared_context"]

        for write_spec in writes:
            context_key = write_spec.get("context_key", write_spec.get("artifact_name"))
            if context_key:
                if hasattr(shared_context, "read_context_key"):
                    value = shared_context.read_context_key(context_key)
                elif hasattr(shared_context, "get"):
                    value = shared_context.get(context_key)
                else:
                    value = None
                baseline[context_key] = value

        return baseline

    async def _validate_a2a_writes_contract(
            self,
            project_id: str,
            writes: List[Dict[str, Any]],
            baseline: Dict[str, Any],
            shared_context,
            staged_artifacts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Validate that all required writes are present and changed."""
        missing = []
        unchanged = []
        fulfilled = True

        staged_by_name = {
            artifact.get("name"): artifact
            for artifact in staged_artifacts or []
            if isinstance(artifact, dict) and artifact.get("name")
        }

        for write_spec in writes:
            artifact_name = write_spec.get("artifact_name")
            context_key = write_spec.get("context_key", artifact_name)
            required = write_spec.get("required", True)

            if staged_artifacts is not None:
                artifact = staged_by_name.get(artifact_name)
                current_value = (
                    {
                        "artifact_id": artifact.get("artifact_id"),
                        "name": artifact_name,
                        "parts": artifact.get("parts", []),
                    }
                    if artifact
                    else None
                )
            else:
                current_value = None
                if hasattr(shared_context, "read_context_key"):
                    current_value = shared_context.read_context_key(context_key)
                elif hasattr(shared_context, "get"):
                    current_value = shared_context.get(context_key)

            if required and current_value is None:
                missing.append(artifact_name)
                fulfilled = False
            elif baseline.get(context_key) is not None and current_value == baseline.get(context_key):
                unchanged.append(artifact_name)
                if required:
                    fulfilled = False

        return {
            "fulfilled": fulfilled,
            "missing": missing,
            "unchanged": unchanged,
        }

    async def _process_a2a_event(
            self,
            event: Dict[str, Any],
            project_id: str,
            run_id: Optional[str],
            node_id: str,
            shared_context,
            writes: List[Dict[str, Any]],
    ):
        """Process A2A streaming event and update state."""
        event_type = event.get("type")
        data = event.get("data", {})

        if event_type == "status_update":
            await self._emit_a2a_event(
                event_type="a2a_status_update",
                project_id=project_id,
                run_id=run_id,
                data={
                    "node_id": node_id,
                    "task_id": data.get("task_id"),
                    "state": data.get("state"),
                    "timestamp": data.get("timestamp"),
                    "message": data.get("message"),
                }
            )

        elif event_type == "artifact_update":
            artifact_name = data.get("name")
            artifact_id = data.get("artifact_id")
            parts = data.get("parts", [])

            await self._emit_a2a_event(
                event_type="a2a_artifact_update",
                project_id=project_id,
                run_id=run_id,
                data={
                    "node_id": node_id,
                    "task_id": data.get("task_id"),
                    "artifact_id": artifact_id,
                    "parts": parts,
                    "name": artifact_name,
                    "append": data.get("append"),
                    "last_chunk": data.get("last_chunk"),
                }
            )

        elif event_type == "message":
            await self._emit_a2a_event(
                event_type="a2a_message",
                project_id=project_id,
                run_id=run_id,
                data={
                    "node_id": node_id,
                    "role": data.get("role"),
                    "parts": data.get("parts"),
                }
            )

    async def _write_a2a_artifacts_to_context(
            self,
            project_id: str,
            artifacts: List[Dict[str, Any]],
            writes: List[Dict[str, Any]],
    ):
        """Write A2A artifacts into shared_context per the node's writes contract.

        Matches each artifact's `name` against writes[].artifact_name and stores
        {artifact_id, name, parts} under writes[].context_key. Called from BOTH the
        Called once after all response frames are validated and a successful terminal
        outcome is known, so a later invalid frame cannot leave partial output behind.
        """
        if not writes:
            return
        write_by_name = {w.get("artifact_name"): w for w in writes}
        for artifact in artifacts or []:
            name = artifact.get("name")
            if not name:
                continue
            spec = write_by_name.get(name)
            if not spec:
                continue
            context_key = spec.get("context_key", name)
            await self._update_a2a_shared_context(
                project_id=project_id,
                key=context_key,
                value={
                    "artifact_id": artifact.get("artifact_id"),
                    "name": name,
                    "parts": artifact.get("parts", []),
                },
            )
            logger.debug(
                "[ENGINE] Wrote A2A artifact '%s' -> shared_context['%s']",
                name, context_key,
            )

    async def _update_a2a_shared_context(
            self,
            project_id: str,
            key: str,
            value: Any,
    ):
        """Update shared_context with artifact value."""
        project = self._get_project(project_id)
        shared_context = project["shared_context"]

        if not hasattr(shared_context, "write_context_key"):
            raise RuntimeError(
                f"SharedContext for project {project_id} missing write_context_key. "
                f"Cannot safely write A2A artifacts."
            )
        try:
            await shared_context.write_context_key(key, value)
            logger.debug(
                "[ENGINE] Updated shared_context key '%s' from A2A artifact for project %s",
                key, project_id
            )
        except ValueError as e:
            logger.error(
                "[ENGINE] Failed to write protected key '%s' from A2A agent: %s",
                key, e
            )
            raise ValueError(f"Cannot write to protected context key: {key}") from e

    async def _save_a2a_artifacts(
            self,
            project_id: str,
            run_id: Optional[str],
            artifacts: List[Dict[str, Any]],
            node_id: Optional[str] = None,
    ):
        """Save A2A artifacts to file_artifacts with versioning.

        Parts become files in the artifact_store so the Artifacts tab — which reads
        ArtifactStore.get_all_files — surfaces them. Previously these were
        written to the snapshots collection under type "a2a_artifact", which
        no tab reads, so they never appeared anywhere.

        They go straight to save_file rather than through
        snapshot_from_artifacts: that path's should_include_file() allowlist is
        built for container scrapes and omits .geojson, so it would silently
        drop every GeoJSON layer an agent returns.
        """
        store = getattr(self.orch, "artifact_store", None)
        if store is None:
            logger.warning(
                "[ENGINE] No artifact_store; cannot persist %d A2A artifact(s) for %s",
                len(artifacts), project_id,
            )
            return

        scope = a2a_artifact_scope(run_id, node_id)
        used_paths: set = set()
        for artifact in artifacts:
            for path, content in a2a_artifact_to_files(artifact, scope):
                path = dedupe_a2a_path(path, used_paths)
                try:
                    result = await store.save_file(project_id, path, content, run_id)
                    if result.get("status") in {"created", "updated"}:
                        logger.debug("[ENGINE] Saved A2A artifact file '%s'", path)
                    else:
                        # save_file signals a refusal (e.g. an oversized file whose
                        # spill failed) by return value, not by raising.
                        logger.warning(
                            "[ENGINE] A2A artifact file '%s' not saved: %s",
                            path, result.get("reason") or result.get("status"),
                        )
                except Exception as e:
                    logger.warning("[ENGINE] Failed to save A2A artifact file '%s': %s", path, e)

    async def _append_a2a_chat_message(
            self,
            shared_context,
            content: str,
            metadata: Dict[str, Any],
    ):
        """Surface an A2A node's outcome in the chat conversation.

        The A2A node otherwise only emits events and saves artifacts, so the Chat
        tab shows the user's prompt and nothing back — the agent's reply lived only
        in the artifacts. Best-effort: a chat-write failure must not fail the node,
        which already succeeded or failed on its own terms.
        """
        try:
            await shared_context.add_conversation_message(
                role="assistant", content=content, metadata=metadata,
            )
        except Exception as e:
            logger.warning("[ENGINE] Failed to append A2A chat message: %s", e)

    async def _emit_a2a_event(
            self,
            event_type: str,
            project_id: str,
            run_id: Optional[str],
            data: Dict[str, Any],
    ):
        """Emit A2A event to EventEmitter for SSE delivery."""
        try:
            await self.event_emitter.emit(
                event_type=event_type,
                run_id=run_id,
                data={
                    "project_id": project_id,
                    **data
                }
            )
            logger.debug("[ENGINE] Emitted A2A event '%s' for project %s", event_type, project_id)
        except Exception as e:
            logger.warning("[ENGINE] Failed to emit A2A event %s: %s", event_type, e)


    async def _fail_a2a_submission_outcome_unknown(
            self,
            *,
            shared_context,
            project_id: str,
            run_id: Optional[str],
            node_id: str,
            server_id: str,
            message_id: Optional[str],
            reason: str,
    ) -> Dict[str, Any]:
        """AppFactory-280 P1 (3rd re-review): shared fail-closed path for BOTH ways a
        submission's outcome can end up unconfirmed — a restart before task_id was
        saved (the resume branch in _run_a2a_agent_node), or a network exception
        during the original submit_and_track call (_A2ASubmissionOutcomeUnknownError,
        caught in _run_a2a_agent_node). Neither case has a protocol-level way to ask
        the adapter which task, if any, it already created — so both must close the
        durable cursor and fail the run instead of guessing via retry. Kept as one
        method so the two scenarios can't drift apart.
        """
        logger.warning(
            "[A2A_RECOVER] project_id=%s node_id=%s message_id=%s server_id=%s "
            "— submission outcome unknown (%s); failing closed, not retrying",
            project_id, node_id, message_id, server_id, reason,
        )
        await self.orch.storage.close_a2a_task_state(
            project_id=project_id,
            node_id=node_id,
            run_id=run_id,
            final_status="a2a_submission_outcome_unknown",
        )
        await self._emit_a2a_event(
            event_type="a2a_agent_failed",
            project_id=project_id,
            run_id=run_id,
            data={
                "node_id": node_id,
                "server_id": server_id,
                "message_id": message_id,
                "error_type": "a2a_submission_outcome_unknown",
            },
        )
        await self._append_a2a_chat_message(
            shared_context,
            f"❌ The A2A task submission outcome could not be confirmed — {reason}. "
            f"Stopping instead of risking a duplicate task: the A2A protocol gives "
            f"the platform no way to ask the adapter which task, if any, it already "
            f"created for this request, so no automatic retry is possible.",
            {"node_id": node_id, "server_id": server_id, "a2a": True, "error": True},
        )
        return {
            "status": "failed",
            "error": f"A2A submission outcome unknown for message_id={message_id} ({reason})",
            "error_type": "a2a_submission_outcome_unknown",
        }
