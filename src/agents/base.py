"""
Base Agent class that all specialized agents inherit from.

This provides:
- Task claiming mechanism (bidding)
- Access to shared context
- Tool discovery via RAG
- Retry logic
- Event emission for UI updates
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from enum import Enum
import logging
import asyncio
import json
import time
import uuid

from schemas import TaskSchema, ResultSchema, TaskStatus, EventSchema
from telemetry.tracer import get_tracer
from context.shared_context import FULL_CONTEXT_READS_TOKEN, WORKFLOW_DEFAULT_READS_TOKEN
from llm.agent_model_params import DEFAULT_AGENT_MODEL
from llm.model_config import get_context_length_from_cache, get_model_registry
from llm.request_policy import UNSET
from agents.vision_tool_result import journal_safe_tool_result
from tools.external_call_outcome import without_call_observation

# Interim valve until the compaction engine (AppFactory-149) owns context
# pressure. Budget is estimated tokens (chars/4 of this share of the live
# model's window), never a message count — a count window evicts by burst,
# not cost, so one tool-heavy phase blinds the next agent to the last
# human decision.
HISTORY_CONTEXT_WINDOW_FRACTION = 0.6
ESTIMATED_CHARS_PER_TOKEN = 4

# Sentinel: "the conversation fold did not run this build, so _rolling_summary_item
# has no pinned version to seed and falls back to the latest stored summary." A
# resolved value (str or None) means the fold picked exactly what to seed.
_UNSET_SEED = object()


# ---------------------------------------------------------------------------
# Concurrency instrumentation (DEBUG-ONLY)
# ---------------------------------------------------------------------------
# We do not currently expect two LLM streams to be open simultaneously for the
# same project — orchestration is meant to be sequential. The UI has shown
# what looks like overlap, and we don't yet know whether the cause is:
#   (a) the backend genuinely opening concurrent streams,
#   (b) an orphaned stream from a previous step still emitting,
#   (c) UI rendering interleaving sequential events incorrectly.
#
# This tracker records every open stream by project_id, logs a WARN line
# whenever a new stream opens while another is already open for the same
# project, and emits a public event so the UI can highlight + bundle the
# evidence for inspection. It does NOT change behavior — pure instrumentation.
# Single asyncio loop -> no lock needed.
_active_streams_by_project: Dict[str, Dict[str, Dict[str, Any]]] = {}


def _stream_open(project_id: str, stream_id: str, agent_id: str, phase: Any) -> List[Dict[str, Any]]:
    """Register a stream as open. Returns the list of OTHER streams already
    open for this project (so callers can decide whether to flag overlap)."""
    if not project_id:
        return []
    streams = _active_streams_by_project.setdefault(project_id, {})
    others = [v for k, v in streams.items() if k != stream_id]
    streams[stream_id] = {
        "agent_id": agent_id,
        "phase": phase,
        "opened_at": time.time(),
    }
    return others


def _stream_close(project_id: str, stream_id: str) -> None:
    """Unregister a stream. Tolerant: missing entries are fine."""
    if not project_id:
        return
    streams = _active_streams_by_project.get(project_id)
    if not streams:
        return
    streams.pop(stream_id, None)
    if not streams:
        _active_streams_by_project.pop(project_id, None)


class AgentType(str, Enum):
    """Agent type classifications"""
    GENERIC = "generic"
    REQUIREMENTS_GATHERER = "requirements_gatherer"
    REQUIREMENTS_VALIDATOR = "requirements_validator"
    PLANNER = "planner"
    CODING = "coding"
    QA = "qa"
    INTEGRATION = "integration"
    CRITIC = "critic"
    REQUIREMENTS_FINALIZER = "requirements_finalizer"
    ORCHESTRATOR = "orchestrator"


class BaseAgent(ABC):
    """
    Base class for all agents in the AppFactory system.
    
    Each agent can:
    1. Evaluate if it can handle a task (bid)
    2. Execute tasks
    3. Access shared/task context
    4. Use tools via RAG discovery
    5. Emit events for UI updates
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        model: str = DEFAULT_AGENT_MODEL,
        temperature: Optional[float] = 0.2,
        max_retries: int = 3,
        evaluation_model: str | None = None,  # Override model for task evaluation
        allowed_phases: str | List[str] | None = "all",  # "all" | "none" | list of phase names
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.evaluation_model = evaluation_model or model  # Use main model if not specified
        # Phase bidding policy: which high-level project phases this agent may bid in.
        # Examples: "all", "none", ["requirements"], ["planning", "execution"].
        # Important: empty list from DB means "no allowed phases", not "all".
        self.allowed_phases = "all" if allowed_phases is None else allowed_phases
        
        # Dependencies (injected)
        self.shared_context = None
        self.tool_registry = None
        self.llm_client = None
        self.event_emitter = None
        self.mcp_executor = None
        self.agent_pool: List[Any] | None = None
        self.allowed_delegation_targets: List[str] | None = None
        # AppFactory-154: delegation reviewer seam, set per orchestrate-run by the
        # engine (PhaseRunner.run_phase_direct). None means "no reviewers";
        # _delegation_trajectory accumulates completed delegations for the pre-critic.
        self.delegation_reviewers = None
        self._delegation_trajectory: List[Dict] = []
        
        # State
        self.current_task = None
        self.task_history: List[Dict] = []
        self.cancellation_token = None
        # Tracing
        self.tracer = get_tracer()
        self.logger = logging.getLogger(__name__)

    def _capture_callback_enabled(self) -> bool:
        """True when capture can run in the BaseAgent.call_llm streaming path."""
        sc = self.shared_context
        if not sc or not getattr(sc, "storage", None):
            return False
        store = getattr(sc.storage, "agent_llm_calls_store", None)
        return bool(store and getattr(sc, "project_id", None) and self.agent_id)

    def _begin_callback_capture(
        self,
        messages,
        model,
        temperature,
        kwargs: Dict[str, Any],
    ):
        """Snapshot the kwargs heading into stream_completion_with_callback."""
        if not self._capture_callback_enabled():
            return None
        return {
            "call_id": str(uuid.uuid4()),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "tools": kwargs.get("tools"),
            "max_tokens": kwargs.get("max_tokens"),
            "reasoning_effort": kwargs.get("reasoning_effort"),
            "events": [],
        }

    def _capture_callback_event(self, capture, event) -> None:
        if not capture:
            return
        try:
            capture["events"].append(event)
        except Exception:
            pass

    @staticmethod
    def _extract_callback_response(events):
        text_parts: List[str] = []
        thinking_parts: List[str] = []
        text_done_full = None
        thinking_done_full = None
        usage = None
        for ev in events or []:
            if not isinstance(ev, dict):
                continue
            t = ev.get("type", "")
            if t == "text.delta":
                text_parts.append(ev.get("content", "") or "")
            elif t == "text.done":
                if ev.get("content"):
                    text_done_full = ev["content"]
            elif t == "thinking.delta":
                thinking_parts.append(ev.get("content", "") or "")
            elif t == "thinking.done":
                if ev.get("content"):
                    thinking_done_full = ev["content"]
                if ev.get("usage"):
                    usage = ev["usage"]
        return {
            "text": text_done_full if text_done_full is not None else "".join(text_parts),
            "thinking": thinking_done_full if thinking_done_full is not None else "".join(thinking_parts),
            "tool_uses": [],
            "stop_reason": None,
            "usage": usage,
            "error": None,
            "response_id": None,
        }

    async def _finalize_callback_capture(
        self,
        capture,
        result,
        error=None,
    ) -> None:
        """Write the capture doc + emit agent.invocation.captured. Never raises."""
        if not capture:
            return
        try:
            system_text = None
            msgs: List[Dict[str, Any]] = []
            for m in (capture.get("messages") or []):
                if isinstance(m, dict) and m.get("role") == "system" and system_text is None:
                    c = m.get("content")
                    if isinstance(c, str):
                        system_text = c
                    elif c is not None:
                        try:
                            system_text = json.dumps(c, default=str)
                        except Exception:
                            system_text = str(c)
                else:
                    msgs.append(m)

            response = self._extract_callback_response(capture.get("events"))

            if isinstance(result, dict):
                if not response.get("text") and result.get("content"):
                    response["text"] = result["content"]
                if not response.get("tool_uses") and result.get("tool_calls"):
                    tu = []
                    for tc in (result.get("tool_calls") or []):
                        fn = (tc.get("function") if isinstance(tc, dict) else None) or {}
                        tu.append({
                            "id": tc.get("id") if isinstance(tc, dict) else None,
                            "name": fn.get("name"),
                            "input": fn.get("arguments"),
                        })
                    response["tool_uses"] = tu
                if result.get("usage") and not response.get("usage"):
                    response["usage"] = result["usage"]
                result_stop = result.get("stop_reason") or result.get("finish_reason")
                if result_stop and not response.get("stop_reason"):
                    response["stop_reason"] = result_stop
                if result.get("response_id"):
                    response["response_id"] = result["response_id"]

            if error and not response.get("error"):
                response["error"] = error
            if response.get("error") and not response.get("stop_reason"):
                response["stop_reason"] = "error"

            sc = self.shared_context
            store = sc.storage.agent_llm_calls_store
            doc = {
                "_id": capture["call_id"],
                "project_id": sc.project_id,
                "run_id": getattr(sc, "run_id", None),
                "agent_id": self.agent_id,
                # current_task is a plain dict — getattr silently yielded None
                # here for every capture until restart recovery needed the id.
                "task_id": (
                    (TaskSchema.get_id(self.current_task) or None)
                    if isinstance(self.current_task, dict) else None
                ),
                "turn_index": 0,
                "started_at": capture["started_at"],
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "model": capture.get("model"),
                "request": {
                    "system": system_text,
                    "messages": msgs,
                    "tools": capture.get("tools"),
                    "temperature": capture.get("temperature"),
                    "max_tokens": capture.get("max_tokens"),
                    "params": {
                        "reasoning_effort": capture.get("reasoning_effort"),
                    },
                },
                "response": response,
            }
            await store.insert_call(doc)
            if self.event_emitter is not None:
                try:
                    await self.event_emitter.emit(
                        EventSchema.AGENT_INVOCATION_CAPTURED,
                        getattr(sc, "run_id", None),
                        {
                            "project_id": sc.project_id,
                            "call_id": capture["call_id"],
                            "agent_id": self.agent_id,
                            "task_id": doc["task_id"],
                            "turn_index": 0,
                            "token_counts": response.get("usage"),
                            "stop_reason": response.get("stop_reason"),
                            "truncated": doc.get("truncated", False),
                        },
                    )
                except Exception as exc:
                    self.logger.warning("agent.invocation.captured emit failed: %s", exc)
        except Exception as exc:
            self.logger.warning("base callback capture finalize failed: %s", exc)

    def _capture_kwargs_for_runner(self) -> Dict[str, Any]:
        """Build kwargs to wire Inspector capture into an agent runner.

        Returns a kwargs dict that StreamingAgentRunner / SimpleAgentRunner
        accept verbatim (project_id, run_id, agent_id, task_id,
        agent_llm_calls_store, event_emitter, message_store). If any field
        is unavailable the runner treats capture as a no-op — see runners'
        _capture_enabled(). message_store carries the tool ledger
        (ADR-0008) and everything riding it (ask_human, restart recovery);
        it lives here, not per-agent, so no agent type silently ships
        ledger-less again.
        """
        sc = self.shared_context
        store = None
        try:
            store = sc.storage.agent_llm_calls_store if sc and getattr(sc, "storage", None) else None
        except AttributeError:
            store = None
        return {
            "project_id": getattr(sc, "project_id", None) if sc else None,
            "run_id": getattr(sc, "run_id", None) if sc else None,
            "agent_id": self.agent_id,
            # current_task is a plain dict; getattr(dict, "task_id") is always
            # None, so journal records and captures shipped without task_id
            # until restart recovery (ADR-0009) made the attempt id load-bearing.
            "task_id": (
                (TaskSchema.get_id(self.current_task) or None)
                if isinstance(self.current_task, dict) else None
            ),
            "agent_llm_calls_store": store,
            "message_store": getattr(sc, "message_store", None) if sc else None,
            "event_emitter": self.event_emitter,
            "workflow_node_id": (
                self.current_task.get("workflow_node_id")
                if isinstance(self.current_task, dict) else None
            ),
        }

    def can_bid_on_phase(self, phase: str | None) -> bool:
        """Return True if this agent is allowed to bid in the given high-level phase.

        Phases are coarse project stages like "requirements", "planning", "execution".
        "all" means no restriction; "none" means this agent never bids (only used directly).
        """
        if not phase:
            return True
        phase_l = str(phase).lower()
        policy = self.allowed_phases
        if policy is None or policy == "all":
            return True
        if policy == "none":
            return False
        try:
            return phase_l in {str(p).lower() for p in policy}
        except Exception as e:
            # If misconfigured, fail open rather than silently disabling the agent.
            self.logger.warning(
                "[AUCTION] can_bid_on_phase misconfigured agent=%s phase=%s policy=%s err=%s — fail_open=true",
                self.agent_id,
                phase_l,
                policy,
                str(e),
            )
            return True

    def is_cancelled(self) -> bool:
        """Return True if either the shared context or token marked the agent cancelled."""
        try:
            if self.cancellation_token and getattr(self.cancellation_token, "is_cancelled", lambda: False)():
                return True
        except Exception:
            pass
        return bool(self.shared_context and getattr(self.shared_context, "_cancelled", False))

    def ensure_not_cancelled(self, location: str = "operation") -> None:
        """Fail fast before doing more work for a cancelled project."""
        if self.is_cancelled():
            try:
                self.logger.info(
                    "[AGENT_CANCEL] agent=%s location=%s - stopping cancelled work",
                    self.agent_id,
                    location,
                )
            except Exception:
                pass
            raise RuntimeError("Cancelled")

    async def await_with_cancellation(self, coro, location: str):
        """Await an operation while racing it against the project cancellation token.

        The try/finally is load-bearing for orphan-stream prevention: if our own
        await_with_cancellation is cancelled from outside (e.g. by an outer
        asyncio.wait_for hitting its timeout, as auction.py does on bid_on_task),
        the success-path cleanup never runs and `op_task`/`waiter` would
        otherwise leak as background tasks — which is what produced the
        bid-orphan stream observed during the auction phase.
        """
        self.ensure_not_cancelled(location)
        if not self.cancellation_token:
            return await coro

        from orchestration.workflow_task_lifecycle import cancel_and_await

        op_task = asyncio.create_task(coro)
        waiter = asyncio.create_task(self.cancellation_token.wait())
        try:
            done, pending = await asyncio.wait({op_task, waiter}, return_when=asyncio.FIRST_COMPLETED)
            for pending_task in pending:
                pending_task.cancel()

            if waiter in done:
                await cancel_and_await([op_task], label=location)
                self.ensure_not_cancelled(location)
                raise RuntimeError("Cancelled")

            return op_task.result()
        finally:
            leftover = [t for t in (op_task, waiter) if not t.done()]
            if leftover:
                await cancel_and_await(leftover, label=location)
        
    def inject_dependencies(
        self,
        shared_context,
        tool_registry,
        llm_client,
        event_emitter,
        mcp_executor=None
    ):
        """Inject dependencies (called by orchestrator)"""
        self.shared_context = shared_context
        self.tool_registry = tool_registry
        self.llm_client = llm_client
        self.event_emitter = event_emitter
        self.mcp_executor = mcp_executor
    
    def get_display_name(self) -> str:
        t = str(self.agent_type)
        v = getattr(self.agent_type, "value", t)
        m = {
            "requirements_gatherer": "Requirements Gatherer",
            "requirements_validator": "Requirements Validator",
            "planner": "Planner",
            "coding": "Coding Agent",
            "qa": "QA Agent",
            "integration": "Integration Agent",
            "critic": "Critic Expert",
            "orchestrator": "Orchestrator",
        }
        return m.get(v, v.replace("_", " ").title())

    async def _notify_param_downgrade(self, model: str, param_downgrade: Dict[str, Any]) -> None:
        """Surface a temperature/reasoning_effort strip-and-retry to the project chat.

        Always persists (so the message shows up on next chat load) even when
        there's no live event_emitter to push it through immediately.
        """
        if not self.shared_context or not self.shared_context.message_store:
            return
        dropped = param_downgrade.get("dropped") or {}
        task_id = TaskSchema.get_id(self.current_task) if self.current_task else None
        task_title = self._get_task_title(self.current_task) if self.current_task else None
        step_note = f" (step: {task_title})" if task_title else ""
        try:
            message = await self.shared_context.message_store.append_system_message(
                self.shared_context.project_id,
                "param_degraded",
                content=(
                    f"{self.agent_id}{step_note}: model {model} rejected "
                    f"{', '.join(dropped)} — continuing with default values"
                ),
                run_id=self.shared_context.run_id,
                data={
                    "agent_id": self.agent_id,
                    "model": model,
                    "dropped": dropped,
                    "task_id": task_id,
                    "task_title": task_title,
                },
            )
            if self.event_emitter:
                await self.event_emitter.emit(
                    "message_appended",
                    self.shared_context.run_id,
                    {"project_id": self.shared_context.project_id, "message": message},
                )
        except Exception as exc:
            self.logger.warning("[MODEL_PARAMS] failed to append param_degraded message: %s", exc)
    
    def _get_task_title(self, task: Dict[str, Any]) -> str:
        desc = TaskSchema.get_description(task)
        if desc:
            return desc
        t = TaskSchema.get_type(task) or "task"
        fallback = {
            "gather_requirements": "Gather project requirements from user",
            "answer_questions": "Answer requirements questions automatically",
            "create_plan": "Create hierarchical task breakdown",
            "coding": "Implement task",
            "qa": "Validate implementation",
            "testing": "Run tests",
        }
        return fallback.get(str(t), TaskSchema.get_id(task) or "Task")
    
    def _build_task_context(self, task: Dict[str, Any]) -> str:
        """Build legacy sync task context for older specialized agents."""
        if not self.shared_context:
            return self._format_task_only_context(task)

        full_context = self.shared_context.get_full_context()
        return self._format_legacy_task_context(task, full_context)

    async def _build_task_context_async(
        self,
        task: Dict[str, Any],
        exclude_journal_task_id: str = None,
        fold_conversation: bool = False,
        summary_model: Optional[str] = None,
        keep_recent_conversation_tokens: Optional[int] = None,
    ) -> str:
        """Build task context from async SharedContext sources when available.

        ``exclude_journal_task_id`` drops that attempt's tool turns from the
        history render — a restart-resumed attempt re-enters the prompt as
        structured transcript items, and rendering the same pairs as text too
        would feed the model two copies.

        ``fold_conversation`` folds conversation older than the history budget into
        the rolling summary instead of dropping it, leaving only the recent turns to
        render verbatim. The streaming runner seeds the summary separately as a
        strippable item; the tool-less simple path inlines it into the returned
        string via _rolling_summary_item, having no item channel. Off only for the
        sync eval fit-check, which keeps the valve's truncation.

        ``summary_model`` is the plugin's configured summariser model, threaded
        to the fold; None uses the compaction subsystem default.
        """
        # Reset the fold's seed pin for this build — _rolling_summary_item falls
        # back to get_latest while this stays the sentinel (no fold ran).
        self._pending_summary_seed = _UNSET_SEED
        if not self.shared_context:
            return self._format_task_only_context(task)

        if TaskSchema.has_reads(task):
            # Contract tasks render only declared keys. Default the summary seed to
            # None so a contract that omits conversation leaks none; the fold block
            # below sets it to the summary ONLY when the contract actually reads
            # conversation — delivered through the seed channel, never inlined.
            self._pending_summary_seed = None
            reads = TaskSchema.get_reads(task) or []
            full_context = await self._get_full_context_async(
                exclude_journal_task_id=exclude_journal_task_id
            )
            conversation_block = None
            if fold_conversation and self._reads_include_conversation(reads):
                # Same fold as the legacy path — summarise older turns, keep the
                # recent tail verbatim — closing the contract path's unbudgeted dump.
                # Trims full_context['conversation_history'] in place.
                budget = await self._resolve_history_char_budget(keep_recent_conversation_tokens)
                summary = await self._fold_conversation_into_summary(
                    full_context, budget, summary_model=summary_model,
                    exclude_journal_task_id=exclude_journal_task_id,
                )
                # Deliver the summary through the SAME seed channel as legacy, not
                # inlined into the block. Inlining put it inside the task message,
                # which the in-run compaction plugin peels into the verbatim head
                # (_split_head) — out of reach of its body-only _is_injected_summary
                # dedup — so an enabled plugin folding the trajectory would inject a
                # SECOND rolling-summary item. Seeded, the streaming path appends it
                # as a strippable item the plugin replaces and the simple path
                # prepends it: exactly one summary either way.
                self._pending_summary_seed = summary
                conversation_block = self._render_contract_conversation_block(
                    full_context.get("conversation_history") or [], budget
                )
            selected_context = await self._select_context_for_reads(
                reads, full_context,
            )
            exact_runtime = getattr(self, "_exact_tool_runtime", None)
            if exact_runtime is not None:
                selected_context = await exact_runtime.project_context(selected_context)
            return self._format_contract_task_context(
                task, selected_context, conversation_block=conversation_block
            )

        full_context = await self._get_full_context_async(
            exclude_journal_task_id=exclude_journal_task_id
        )
        budget = await self._resolve_history_char_budget(keep_recent_conversation_tokens)
        if fold_conversation:
            self._pending_summary_seed = await self._fold_conversation_into_summary(
                full_context, budget, summary_model=summary_model,
                exclude_journal_task_id=exclude_journal_task_id,
            )
        exact_runtime = getattr(self, "_exact_tool_runtime", None)
        if exact_runtime is not None:
            full_context = await exact_runtime.project_context(full_context)
        return self._format_legacy_task_context(
            task, full_context, history_char_budget=budget
        )

    def _history_char_budget_from_tokens(self, context_window_tokens: int) -> int:
        return int(
            context_window_tokens
            * ESTIMATED_CHARS_PER_TOKEN
            * HISTORY_CONTEXT_WINDOW_FRACTION
        )

    async def _resolve_history_char_budget(
        self, budget_tokens_override: Optional[int] = None
    ) -> int:
        """History budget derived from the model actually running.

        The gateway's models_cache knows each model's real window; the
        registry default covers a cache miss, so a miss degrades to a
        smaller prompt, never an overflow.

        ``budget_tokens_override`` (plugins.context_compaction.keep_recent_conversation_tokens)
        pins the verbatim conversation slice to a flat token count instead of the
        window fraction — a smaller value folds older turns sooner. Clamped to that
        same window fraction (not the full window) so history plus the mandatory
        system/task/plan prefix can't overflow the model; 0 / absent keeps the
        window-derived default.
        """
        window = None
        try:
            window = await get_context_length_from_cache(self.model)
        except Exception:
            window = None
        if not window:
            window = get_model_registry().get_config(self.model).max_context_tokens
        # Tenant plugin config is unvalidated and the UI Save isn't blocked by the
        # schema squiggle, so this override can arrive as junk ("abc", "10.5"). Degrade
        # to the window default instead of raising into the task build — a config typo
        # must not fail the run. Shared with the plugin's sibling budget knobs so both
        # coerce identically. 0 / absent → window default (override is not > 0).
        from plugins.context_compaction import _coerce_number
        override = _coerce_number(budget_tokens_override, 0, int)
        if override > 0:
            # Clamp to the same window fraction as the auto-default (and the plugin's
            # sibling tool-tail cap), NOT the full window: a verbatim slice sized at
            # 100% of the window leaves no room for the system/task/plan prefix and
            # overflows on send, which the conversation-fold path can't recover.
            return min(override, int(window * HISTORY_CONTEXT_WINDOW_FRACTION)) * ESTIMATED_CHARS_PER_TOKEN
        return self._history_char_budget_from_tokens(window)

    def _conversation_line(self, msg: Dict[str, Any]) -> str:
        """One rendered history line — the shared shape for both the verbatim
        render and the fold input, so a folded turn reads to the compactor exactly
        as it would have on screen."""
        role = str(msg.get("role", "unknown")).upper()
        return f"{role}: {msg.get('content', '') or ''}"

    def _render_history_within_budget(
        self, conversation: List[Dict[str, Any]], history_char_budget: int
    ) -> List[str]:
        """Render conversation newest-first until the char budget is spent, back in
        stream order, with an omitted-turns notice on top. Shared by the legacy
        formatter and the contract render so BOTH bound the same way when the fold
        left history untrimmed (a degrade branch) — without a shared valve the
        contract render dumped the whole un-summarised backlog and overflowed."""
        if not conversation:
            return []
        kept: List[str] = []
        used = 0
        omitted = 0
        for msg in reversed(conversation):
            line = self._conversation_line(msg)
            if used + len(line) > history_char_budget:
                if not kept:
                    # The newest turn alone exceeds the whole budget — the agent
                    # still needs it, so clip rather than render empty history.
                    overflow = len(line) - history_char_budget
                    kept.append(
                        line[:history_char_budget]
                        + f"\n…(+{overflow} chars over context budget)"
                    )
                omitted = len(conversation) - len(kept)
                break
            kept.append(line)
            used += len(line)
        out: List[str] = []
        if omitted:
            out.append(
                f"(… {omitted} older turns not shown — context budget; "
                "the full history stays in the project journal)"
            )
            self.logger.info(
                "History render dropped %d/%d oldest turns (budget %d chars)",
                omitted, len(conversation), history_char_budget,
            )
        out.extend(reversed(kept))
        return out

    @staticmethod
    def _reads_include_conversation(reads: List[str]) -> bool:
        """True when a reads contract exposes conversation_history — named directly
        or via the ``*`` full-context token — so it must get the folded render, not
        the raw dump. A contract naming neither still sees no conversation at all."""
        for key in reads:
            k = str(key or "").strip()
            if k == FULL_CONTEXT_READS_TOKEN or k == "conversation_history":
                return True
        return False

    def _render_contract_conversation_block(
        self,
        tail_turns: List[Dict[str, Any]],
        history_char_budget: int,
    ) -> str:
        """Render conversation_history for a reads contract as the budget-valved
        verbatim tail. The rolling summary is NOT composed here — it rides the seed
        channel (_pending_summary_seed) like the legacy path, so the in-run
        compaction plugin can recognise and dedup it; inlining it would bury it in
        the task message, past the plugin's body-only dedup, and let an enabled
        plugin stack a second copy. The tail goes through the same budget valve as
        legacy: on the fold's success path it already fits (no-op), but on a degrade
        branch it is the whole backlog and the valve is what keeps this bounded."""
        return "\n".join(
            self._render_history_within_budget(tail_turns, history_char_budget)
        )

    async def _resolve_compaction_config(self) -> Dict[str, Any]:
        """Resolve the context_compaction plugin config for this execution,
        independent of whether the plugin's hooks are enabled.

        The conversation-fold seeder reads its own knobs here — its on/off switch
        (``summarize_overflow_conversation``, default on) and ``summary_model``. The plugin
        ``enabled`` flag gates only the in-run hook path (build_plugin_host); the
        seeder is a separate streaming-path mechanism (ADR-0014) with its own
        switch, so it resolves config directly rather than through the host — which
        returns None precisely when the plugin is disabled, the common case.

        Mirrors build_plugin_host's chain inputs; any failure degrades a level to
        "not configured" rather than raising into the run path.
        """
        from plugins.config_chain import resolve_plugin_config
        from plugins.registry import PLUGIN_REGISTRY

        sc = self.shared_context
        spec = PLUGIN_REGISTRY.get("context_compaction")
        code_defaults = spec.code_defaults if spec else {}
        tenant_id = getattr(sc, "tenant_id", None) if sc else None
        storage = getattr(sc, "storage", None) if sc else None
        run_config = None
        cache = getattr(sc, "cache", None) if sc else None
        if isinstance(cache, dict):
            run_config = cache.get("run_config")
        tenant_settings = None
        if storage is not None and tenant_id and hasattr(storage, "get_tenant_settings"):
            try:
                tenant_settings = await storage.get_tenant_settings(tenant_id)
            except Exception:
                tenant_settings = None
        return resolve_plugin_config(
            "context_compaction",
            code_defaults=code_defaults,
            tenant_settings=tenant_settings,
            run_config=run_config,
            agent_config=getattr(self, "config", None),
        )

    async def _fold_conversation_into_summary(
        self,
        full_context: Dict[str, Any],
        history_char_budget: int,
        *,
        summary_model: Optional[str] = None,
        exclude_journal_task_id: Optional[str] = None,
    ) -> Optional[str]:
        """Fold conversation older than the history budget into the rolling summary
        (AppFactory-149 slice 5), replacing the valve's drop-the-oldest with a
        recoverable fold. Mutates ``full_context['conversation_history']`` in place
        to hold only the recent verbatim turns; the folded turns now live in the
        summary, which the streaming runner seeds as a separate item.

        The un-summarized tail is read forward from the frontier — NL via
        ``SharedContext.get_conversation_since`` and the tool journal via
        ``get_journal_since`` (both gap-free, uncapped), not from the bounded window
        ``full_context`` was built with — so the fold covers every record past the
        frontier and never advances it over one it did not read (ADR-0014 D1 & D3).

        Returns the summary text the caller should seed alongside that verbatim
        window: the just-written version on a successful fold, else the PRIOR
        version (which covers only ``<= frontier`` — strictly older than every turn
        kept verbatim here). Deliberately never a fresh get_latest: a sibling run
        racing a newer version in could cover turns we kept verbatim, and the model
        would then see them twice. None when there is nothing safe to seed.

        No-ops (leaving the valve to truncate, seeding nothing) when the summary
        chain or a project id is unreachable — tests and non-Mongo contexts.
        """
        sc = self.shared_context
        storage = getattr(sc, "storage", None) if sc else None
        store = getattr(storage, "rolling_summary_store", None) if storage else None
        project_id = getattr(sc, "project_id", None) if sc else None
        conversation = full_context.get("conversation_history", []) or []
        if store is None or not project_id or not conversation:
            return None
        # Sequence is the fold's coordinate. If any turn lacks a real one (a
        # legacy or duck-typed store that never stamped sequences), the filter
        # would read it as 0 and drop it as "already folded" — leave the valve to
        # render rather than risk wiping turns we can't place.
        if any(int(m.get("sequence", 0)) <= 0 for m in conversation):
            return None

        try:
            prior = await store.get_latest(project_id)
        except Exception as exc:
            self.logger.warning(
                "[COMPACT] fold aborted: rolling-summary read failed for project "
                "%s (%s) — valve truncates this run", project_id, exc,
            )
            return None
        conv_frontier = int(prior.get("conversation_covers_to_sequence", 0)) if prior else 0
        # Prior covers only <= conv_frontier, so it is always safe to seed beside a
        # verbatim window of turns > conv_frontier. Every degrade path below returns
        # this, not a fresh get_latest a racing sibling may have advanced.
        prior_text = prior.get("summary") if prior else None

        # A tool pair sits at its call's sequence but spans to its result's
        # (entry["end_sequence"]); NL turns span one sequence. "Past the frontier"
        # is decided by the span's END: a pair whose call is summarised but whose
        # result arrived later still has new information to render.
        def _span_end(m: Dict[str, Any]) -> int:
            return int(m.get("end_sequence") or m.get("sequence", 0) or 0)

        # Everything past the frontier is a candidate to render or fold. Read BOTH
        # streams forward from the frontier (gap-free, uncapped) instead of reusing
        # the bounded window get_full_context_async loaded: that window keeps only the
        # oldest 1000 NL turns and the newest 1000 tool records, so folding on it would
        # advance the frontier — which gates NL and tool alike — over records it never
        # saw. For NL that drops the newest turns past the 1000th; for the tool journal
        # it marks old pairs (below the cut but off the newest-1000 tail) as covered
        # without summarising them. Reading each forward keeps the stored frontier
        # honest for the whole unified stream (ADR-0014 Decisions 1 & 3).
        reader = getattr(sc, "get_conversation_since", None)
        journal_reader = getattr(sc, "get_journal_since", None)
        if callable(reader):
            try:
                nl_fresh = await reader(conv_frontier)
            except Exception as exc:
                # These are extra store round-trips after the window already loaded;
                # a transient failure here degrades to the valve like every other
                # fold failure — it must not fail the phase. Frontier untouched, the
                # loaded window past it stays verbatim, prior seeded (no overlap).
                self.logger.warning(
                    "[COMPACT] fold aborted: forward read failed for project %s (%s) "
                    "— valve truncates this run, seeded prior", project_id, exc,
                )
                full_context["conversation_history"] = [
                    m for m in conversation if _span_end(m) > conv_frontier
                ]
                return prior_text
            # Tool entries the loaded window already delivered — the safe journal
            # material for the duck-typed path and for a failed precise read below.
            window_tools = [
                m for m in conversation
                if m.get("role") == "tool" and _span_end(m) > conv_frontier
            ]
            if callable(journal_reader):
                try:
                    journal_fresh = await journal_reader(
                        conv_frontier, exclude_journal_task_id=exclude_journal_task_id
                    )
                except Exception as exc:
                    # The NL tail is already in hand — falling back to the 1000-cap
                    # window would drop every turn past it (PR145 r3814153349). Keep
                    # the tail + window tools verbatim; no fold: the frontier must
                    # not advance over journal pairs nobody read.
                    self.logger.warning(
                        "[COMPACT] fold aborted: journal read failed for project %s "
                        "(%s) — kept fresh conversation verbatim, seeded prior",
                        project_id, exc,
                    )
                    full_context["conversation_history"] = sorted(
                        nl_fresh + window_tools,
                        key=lambda m: int(m.get("sequence", 0)),
                    )
                    return prior_text
            else:
                # Duck-typed context with an NL reader but no journal reader: fall
                # back to the tool entries in the loaded window.
                journal_fresh = window_tools
            fresh = sorted(
                nl_fresh + journal_fresh,
                key=lambda m: int(m.get("sequence", 0)),
            )
        else:
            # Duck-typed / non-Mongo contexts (tests): no forward reader, so fall
            # back to the loaded window. Turns with no sequence read as 0 and are
            # treated as already-covered — the valve still bounds size.
            fresh = [m for m in conversation if _span_end(m) > conv_frontier]
        if not fresh:
            full_context["conversation_history"] = []
            return prior_text

        # Newest-first, keep whole turns until the budget is spent; the overflow
        # (older turns) folds in. A clean sequence cut — everything kept is newer
        # than everything folded — so the stored frontier is exact and next run's
        # filter can never skip a turn that was left verbatim.
        kept: List[Dict[str, Any]] = []
        used = 0
        to_fold: List[Dict[str, Any]] = []
        for i in range(len(fresh) - 1, -1, -1):
            line = self._conversation_line(fresh[i])
            if kept and used + len(line) > history_char_budget:
                to_fold = fresh[: i + 1]
                break
            used += len(line)
            kept.insert(0, fresh[i])
            to_fold = fresh[:i]

        if not to_fold:
            full_context["conversation_history"] = kept
            return prior_text

        # The frontier is a raw-sequence cut, so it must clear the END of every
        # folded pair, not just its anchor — else the result is re-read next run as
        # an orphan of a call already in the summary. But a turn interleaved while
        # that tool ran sits between the call and the result: kept, it would then
        # be below the frontier without ever being summarised (silent loss). Fold
        # such turns too, until nothing kept starts at or below the cut. Fresh is
        # sorted by anchor, so they are always the front of ``kept``.
        cut = max(_span_end(m) for m in to_fold)
        while kept and int(kept[0].get("sequence", 0)) <= cut:
            moved = kept.pop(0)
            to_fold.append(moved)
            cut = max(cut, _span_end(moved))
        new_text = "\n\n".join(self._conversation_line(m) for m in to_fold)
        summary = await self._fold_summary_text(
            prior_text, new_text, summary_model=summary_model
        )
        if not summary:
            # Compactor unavailable or empty: do NOT drop the turns silently. Keep
            # the full fresh window verbatim (the formatter's budget still caps the
            # absolute size) so this degrades to the valve, never to data loss. The
            # prior summary still covers <= frontier, so seeding it stays safe.
            self.logger.warning(
                "[COMPACT] fold degraded to valve for project %s: compactor "
                "produced no summary; kept %d fresh turn(s) verbatim, seeded prior",
                project_id, len(fresh),
            )
            full_context["conversation_history"] = fresh
            return prior_text

        prior_covers = int(prior.get("covers_to_sequence", 0)) if prior else 0
        try:
            await store.append_version(
                project_id,
                summary=summary,
                covers_from_sequence=0,
                # Revert frontier advances to at least the folded conversation cut;
                # it inherits the prior (possibly a trajectory high-water) so the
                # chain's coverage never goes backwards. Seeder filter reads the
                # separate conversation frontier below.
                covers_to_sequence=max(prior_covers, cut),
                conversation_covers_to_sequence=cut,
                run_id=getattr(sc, "run_id", None),
                meta={"source": "seeder", "folded_turns": len(to_fold)},
            )
        except Exception as exc:
            # A concurrent fold won the unique version slot. Keep the fresh window
            # verbatim this run and seed the PRIOR summary (covers <= frontier, no
            # overlap with the verbatim turns) — NOT the winner's newer version,
            # which may cover turns we just kept verbatim and duplicate them.
            self.logger.warning(
                "[COMPACT] fold version append lost the race for project %s (%s) "
                "— kept %d fresh turn(s) verbatim, seeded prior",
                project_id, exc, len(fresh),
            )
            full_context["conversation_history"] = fresh
            return prior_text
        full_context["conversation_history"] = kept
        return summary

    async def _fold_summary_text(
        self,
        prior_text: Optional[str],
        new_text: str,
        *,
        summary_model: Optional[str] = None,
    ) -> Optional[str]:
        """Run the shared compactor over (prior summary + new turns). Returns the
        folded text, or None if the summariser is unreachable or empty — the caller
        then keeps the turns verbatim rather than losing them.

        ``summary_model`` (plugins.context_compaction.summary_model) keeps the
        seeder's fold on its own cheap model instead of the agent's default; None
        falls through to the compaction subsystem chain in _plugin_llm_call."""
        try:
            from plugins.context_compaction import (
                build_compactor_messages,
                _DEFAULT_SUMMARY_TEMPERATURE,
            )
            text = await self._plugin_llm_call(
                build_compactor_messages(prior_text, new_text),
                model=summary_model,
                temperature=_DEFAULT_SUMMARY_TEMPERATURE,
            )
        except Exception as exc:
            self.logger.warning("[COMPACT] fold summariser call failed: %s", exc)
            return None
        return (text or "").strip() or None

    async def _rolling_summary_item(self) -> Optional[Dict[str, Any]]:
        """The current rolling summary as a strippable transcript item, or None.

        Seeded after the task message so the model sees folded history, and shaped
        with the shared prefix so an in-run compaction fold recognises and replaces
        it instead of duplicating it. None when no summary exists yet or the store
        is unreachable.

        Prefers the exact version the fold pinned this build (_pending_summary_seed)
        over a fresh get_latest, so a sibling run's newer version can't seed turns
        this run kept verbatim (AppFactory-149 slice 5). Falls back to get_latest only
        when no fold ran this build (the sentinel is untouched).
        """
        from plugins.context_compaction import SUMMARY_PREFIX
        pending = getattr(self, "_pending_summary_seed", _UNSET_SEED)
        if pending is not _UNSET_SEED:
            if not pending:
                return None
            return {"type": "message", "role": "user", "content": SUMMARY_PREFIX + pending}
        sc = self.shared_context
        storage = getattr(sc, "storage", None) if sc else None
        store = getattr(storage, "rolling_summary_store", None) if storage else None
        project_id = getattr(sc, "project_id", None) if sc else None
        if store is None or not project_id:
            return None
        try:
            latest = await store.get_latest(project_id)
        except Exception:
            return None
        if not latest or not latest.get("summary"):
            return None
        return {
            "type": "message",
            "role": "user",
            "content": SUMMARY_PREFIX + latest["summary"],
        }

    async def _get_full_context_async(
        self, exclude_journal_task_id: str = None
    ) -> Dict[str, Any]:
        """Read full context using the async source when SharedContext supports it."""
        if not self.shared_context:
            return {}
        if hasattr(self.shared_context, "get_full_context_async"):
            # Kwarg only when set: duck-typed stand-ins predate the parameter.
            if exclude_journal_task_id:
                return await self.shared_context.get_full_context_async(
                    exclude_journal_task_id=exclude_journal_task_id
                )
            return await self.shared_context.get_full_context_async()
        return self.shared_context.get_full_context()

    async def _select_context_for_reads(
        self,
        reads: List[str],
        full_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Select logical context keys for an explicit reads contract."""
        normalized_reads = [
            str(key or "").strip()
            for key in reads
            if str(key or "").strip()
        ]
        if FULL_CONTEXT_READS_TOKEN in normalized_reads:
            return dict(full_context)

        selected: Dict[str, Any] = {}
        for key in normalized_reads:
            if key == WORKFLOW_DEFAULT_READS_TOKEN:
                continue
            reader = getattr(self.shared_context, "read_context_key_async", None)
            if callable(reader):
                selected[key] = await reader(key)
            else:
                selected[key] = full_context.get(key)
        return selected

    def _format_legacy_task_context(
        self,
        task: Dict[str, Any],
        full_context: Dict[str, Any],
        history_char_budget: int = None,
    ) -> str:
        """Format legacy full-context prompts using async-loaded context data."""
        context_parts = []
        self._append_retry_feedback(context_parts, task)

        # User's original request (always available from SharedContext cache)
        user_prompt = full_context.get('user_prompt', '')
        if user_prompt:
            context_parts.append("=== USER REQUEST ===")
            context_parts.append(user_prompt)
            context_parts.append("")

        # Conversation history under the model-window budget: newest turns
        # first until the budget is spent, rendered back in stream order.
        # Sync callers can't await the models_cache lookup — registry default.
        if history_char_budget is None:
            history_char_budget = self._history_char_budget_from_tokens(
                get_model_registry().get_config(self.model).max_context_tokens
            )
        conversation = full_context.get('conversation_history', [])
        if conversation:
            context_parts.append("=== CONVERSATION HISTORY ===")
            context_parts.extend(
                self._render_history_within_budget(conversation, history_char_budget)
            )
            context_parts.append("")
        
        # Flat task list (if available)
        plan = full_context.get('plan', {})
        if plan:
            tasks = plan.get('tasks', [])
            context_parts.append(f"=== PROJECT PLAN ({len(tasks)} tasks) ===")
            current_task_id = task.get('task_id')
            
            for i, t in enumerate(tasks, 1):
                tid = t.get('task_id')
                desc = t.get('description', 'No description')
                is_current = (tid == current_task_id)
                marker = "→" if is_current else str(i)
                context_parts.append(f"{marker}. {desc}")
            context_parts.append("")
        
        # Hint about existing files (agent should use tools to read them)
        artifacts = full_context.get('artifacts', [])
        if artifacts:
            file_paths = [art.get('path', 'unknown') for art in artifacts]
            context_parts.append(f"=== EXISTING PROJECT FILES ({len(artifacts)}) ===")
            context_parts.append("Use list_files and read_file tools to explore. Use edit_file to modify.")
            for p in file_paths[:20]:
                context_parts.append(f"  - {p}")
            if len(file_paths) > 20:
                context_parts.append(f"  ... and {len(file_paths) - 20} more")
            context_parts.append("")

        project_attachments = full_context.get("project_attachments") or []
        if project_attachments:
            context_parts.append(f"=== USER ATTACHMENTS ({len(project_attachments)}) ===")
            for att in project_attachments:
                name = att.get("filename") or "file"
                aid = att.get("id") or ""
                size = att.get("size_bytes")
                size_note = f", {size} bytes" if size is not None else ""
                context_parts.append(f"- {name} (id={aid}{size_note})")
                text = att.get("text_content")
                if isinstance(text, str) and text.strip():
                    context_parts.append("  Content:")
                    context_parts.append(text.rstrip())
            context_parts.append(
                "For large or binary attachments, use attachment_view, or attachment_fetch + read."
            )
            context_parts.append("")

        context_parts.append("=== CURRENT TASK ===")
        context_parts.append(f"{task.get('description', 'No description')}")
        context_parts.append("")
        self._append_task_runtime_context(context_parts, task)
        return "\n".join(context_parts)

    def _format_contract_task_context(
        self,
        task: Dict[str, Any],
        selected_context: Dict[str, Any],
        conversation_block: Optional[str] = None,
    ) -> str:
        """Format explicit reads context without leaking undeclared keys.

        ``conversation_block`` is the pre-rendered [summary][tail] text for the
        conversation_history key; when present it replaces the raw JSON dump of the
        message array so a long history can't overflow the window.
        """
        context_parts = []
        self._append_retry_feedback(context_parts, task)

        if selected_context:
            for key, value in selected_context.items():
                context_parts.append(f"=== CONTEXT: {key} ===")
                if key == "conversation_history" and conversation_block is not None:
                    context_parts.append(conversation_block)
                else:
                    context_parts.append(self._json_dumps(value))
                context_parts.append("")
        else:
            context_parts.append("=== CONTEXT ===")
            context_parts.append("No context keys were requested for this task.")
            context_parts.append("")

        context_parts.append("=== CURRENT TASK ===")
        context_parts.append(f"{task.get('description', 'No description')}")
        context_parts.append("")
        self._append_task_runtime_context(context_parts, task)
        return "\n".join(context_parts)

    def _format_task_only_context(self, task: Dict[str, Any]) -> str:
        context_parts = [
            "=== CURRENT TASK ===",
            task.get("description", "No description"),
            "",
        ]
        self._append_task_runtime_context(context_parts, task)
        return "\n".join(context_parts)

    def _append_retry_feedback(
        self,
        context_parts: List[str],
        task: Dict[str, Any],
    ) -> None:
        retry_feedback = TaskSchema.get_retry_feedback(task)
        if retry_feedback:
            context_parts.append("=== RETRY FEEDBACK ===")
            context_parts.append(str(retry_feedback))
            context_parts.append("")

        previous_attempts = task.get("previous_attempts")
        if previous_attempts:
            context_parts.append("=== PREVIOUS ATTEMPTS ===")
            context_parts.append(self._json_dumps(previous_attempts))
            context_parts.append("")

    def _append_task_runtime_context(
        self,
        context_parts: List[str],
        task: Dict[str, Any],
    ) -> None:
        task_context = TaskSchema.get_context(task)
        if task_context:
            context_parts.append("=== TASK CONTEXT ===")
            context_parts.append(self._json_dumps(task_context))
            context_parts.append("")

    @staticmethod
    def _json_dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str, indent=2)
    
    @abstractmethod
    async def evaluate_task(self, task: Dict[str, Any]) -> float:
        """
        Evaluate if agent can handle this task.
        
        Args:
            task: Task dictionary with:
                - task_id: str
                - description: str
                - context: Dict
                - requirements: Dict
        
        Returns:
            fit_score (0.0 to 1.0) - how confident agent is it CAN HANDLE this task
            - 0.0 = I cannot do this task
            - 1.0 = I am perfectly suited to handle this task
        """
        pass
    
    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the assigned task.
        
        Args:
            task: Task to execute
        
        Returns:
            Result dictionary with:
                - status: TaskStatus
                - output: Any (task result)
                - artifacts: List[Dict] (files created, etc.)
                - reasoning: str (explanation)
        """
        pass
    
    async def bid_on_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit bid for a task (used in auction).
        
        Returns:
            Bid dictionary:
                - agent_id: str
                - agent_type: AgentType
                - fit_score: float (0-1) how confident I am that I CAN HANDLE this task
                - reasoning: str
        """
        print(f"⏳ {self.agent_id} starting bid evaluation...")
        
        # Set bidding phase flag so call_llm knows we're in auction
        self._bidding_phase = True
        try:
            fit_score = await self.evaluate_task(task)
            print(f"✓ {self.agent_id} evaluated: {fit_score}")
            
            reasoning = await self._generate_bid_reasoning(task, fit_score)
            print(f"✓ {self.agent_id} bid complete: {fit_score}")
            
            return {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "fit_score": fit_score,  # How confident I am that I CAN HANDLE this task
                "reasoning": reasoning,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            print(f"❌ {self.agent_id} bid failed: {e}")
            raise
        finally:
            # Always clear bidding phase flag
            self._bidding_phase = False
    
    async def execute_with_retry(
        self, 
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute task with automatic retry on failure.
        
        Retries up to max_retries times, each time providing
        error context to help agent refine approach.
        
        Args:
            task: Task dictionary to execute
            
        Returns:
            Result dictionary with execution outcome
        """
        self.current_task = task
        attempts: List[Dict[str, Any]] = []
        trace_attempt = task.get(TaskSchema.TRACE_ATTEMPT)
        if not isinstance(trace_attempt, int) or trace_attempt < 1:
            trace_attempt = None

        # Root span for this agent's execution 
        with self.tracer.start_span(
            "agent.execute_task",
            attributes={
                "agent.id": self.agent_id,
                "agent.type": self.agent_type.value,
                "agent.model": self.model,
                "agent.temperature": self.temperature,
                "task.id": TaskSchema.get_id(task),
                "task.type": TaskSchema.get_type(task),
                "task.title": self._get_task_title(task),
                "task.max_retries": self.max_retries,
                "AppFactory.agent.execution_type": "task_execution",
                "AppFactory.agent.retry_enabled": self.max_retries > 1,
                "AppFactory.agent.tools_available": len(self.tool_registry.tools) if hasattr(self.tool_registry, 'tools') else 0
            },
        ) as agent_span:
            # AppFactory-316: same backoff ladder as streaming round retries.
            legacy_backoff_s = (1.0, 5.0, 20.0)
            for attempt in range(1, self.max_retries + 1):
                try:
                    self.logger.info(
                        "[AGENT] attempt.start agent=%s task_id=%s attempt=%d/%d",
                        self.agent_id,
                        TaskSchema.get_id(task),
                        attempt,
                        self.max_retries,
                    )
                except Exception:
                    pass
                await self._emit_event("task_attempt", {
                    "agent_id": self.agent_id,
                    "agent_display_name": self.get_display_name(),
                    "task_id": TaskSchema.get_id(task),
                    "task_description": self._get_task_title(task),
                    "task_type": TaskSchema.get_type(task),
                    "attempt": trace_attempt or attempt,
                    "agent_retry": attempt,
                    "max_retries": self.max_retries
                })

                try:
                    # Add error context from previous attempts
                    if attempts:
                        task["previous_attempts"] = attempts

                    result = await self.execute_task(task)

                    # Use ResultSchema to check status
                    if ResultSchema.is_completed(result):
                        try:
                            self.logger.info(
                                "[AGENT] attempt.end agent=%s task_id=%s status=completed attempt=%d",
                                self.agent_id,
                                TaskSchema.get_id(task),
                                attempt,
                            )
                        except Exception:
                            pass
                        if agent_span:
                            agent_span.set_attribute("task.status", "completed")
                            self.tracer.set_success(agent_span)
                        await self._emit_event("task_completed", {
                            "agent_id": self.agent_id,
                            "agent_display_name": self.get_display_name(),
                            "task_id": TaskSchema.get_id(task),
                            "task_description": self._get_task_title(task),
                            "task_type": TaskSchema.get_type(task),
                            "attempt": trace_attempt or attempt,
                            "agent_retry": attempt,
                        })

                        # Record success
                        self.task_history.append({
                            "task_id": TaskSchema.get_id(task),
                            "status": "completed",
                            "attempts": attempt,
                            "timestamp": datetime.utcnow().isoformat()
                        })

                        return result

                    # Task returned failure status — not a transient provider blip.
                    attempts.append({
                        "attempt": attempt,
                        "result": result,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    break

                except Exception as e:
                    from llm.errors import is_transient_provider_error

                    try:
                        self.logger.error(
                            "[AGENT] attempt.error agent=%s task_id=%s attempt=%d error=%s",
                            self.agent_id,
                            TaskSchema.get_id(task),
                            attempt,
                            str(e),
                        )
                    except Exception:
                        pass
                    attempts.append({
                        "attempt": attempt,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    })

                    await self._emit_event("task_error", {
                        "agent_id": self.agent_id,
                        "agent_display_name": self.get_display_name(),
                        "task_id": TaskSchema.get_id(task),
                        "task_description": self._get_task_title(task),
                        "task_type": TaskSchema.get_type(task),
                        "attempt": trace_attempt or attempt,
                        "agent_retry": attempt,
                        "error": str(e)
                    })
                    self.tracer.add_event(agent_span, "retry_error", {"attempt": attempt, "error": str(e)})

                    if (
                        attempt < self.max_retries
                        and is_transient_provider_error(e)
                    ):
                        backoff = legacy_backoff_s[
                            min(attempt - 1, len(legacy_backoff_s) - 1)
                        ]
                        await self.await_with_cancellation(
                            asyncio.sleep(backoff),
                            "legacy_retry_backoff",
                        )
                        continue
                    break
        
        # All retries exhausted
        if 'agent_span' in locals() and agent_span:
            agent_span.set_attribute("task.status", "failed")
            self.tracer.set_error(agent_span, RuntimeError("max_retries_exhausted"))
        failure_payload = {
            "agent_id": self.agent_id,
            "agent_display_name": self.get_display_name(),
            "task_id": TaskSchema.get_id(task),
            "task_description": self._get_task_title(task),
            "task_type": TaskSchema.get_type(task),
            "attempts": attempts
        }
        if trace_attempt is not None:
            failure_payload["attempt"] = trace_attempt
        await self._emit_event("task_failed", failure_payload)
        try:
            self.logger.error(
                "[AGENT] task.failed agent=%s task_id=%s attempts=%d",
                self.agent_id,
                TaskSchema.get_id(task),
                len(attempts),
            )
        except Exception:
            pass
        
        last_err = None
        if attempts:
            last = attempts[-1]
            last_err = last.get("error")
            if not last_err and isinstance(last.get("result"), dict):
                last_err = last["result"].get("error") or last["result"].get("reasoning")
        # Use ResultSchema to create failure result
        return ResultSchema.create(
            status=TaskStatus.FAILED,
            output=None,
            reasoning=last_err or f"Failed after {len(attempts)} attempt(s)",
            error=last_err,
            attempts=attempts,
            escalate_to_human=True
        )
    
    async def discover_tools(self, description: str) -> List[Dict]:
        """
        Use RAG to find relevant tools for a task.
        
        Args:
            description: What the agent needs to do
        
        Returns:
            List of tool definitions
        """
        if not self.tool_registry:
            return []

        tenant_id = None
        if self.shared_context is not None:
            tenant_id = self.shared_context.tenant_id
            if tenant_id == "__default__":
                tenant_id = "__root__"

        effective = getattr(self, "_effective_allowed_tools", None)
        if effective is None:
            from tools.agent_allowed_tools import effective_allowed_tool_ids

            effective = effective_allowed_tool_ids(getattr(self, "config", {}) or {})
        return await self.tool_registry.search_tools(
            query=description,
            agent_type=self.agent_type.value,
            top_k=5,
            allowed_tool_ids=list(effective or []),
            tenant_id=tenant_id,
        )
    
    async def call_llm(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ):
        """
        Call LLM with given messages. Retries up to 3 times on failure.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            **kwargs: Additional parameters (temperature, max_tokens, tools, etc.)
        
        Returns:
            Dict with 'content' and optional 'tool_calls' if tools are provided,
            otherwise just the string content for backward compatibility
        """
        if not self.llm_client:
            raise RuntimeError("LLM client not injected")
        subsystem = kwargs.pop("subsystem", "agent_default")
        requested_model = kwargs.pop("model", None)
        model = self._resolve_project_model(requested_model, subsystem=subsystem)
        temperature = kwargs.pop("temperature", UNSET)
        return_details = bool(kwargs.pop("return_details", False))
        if temperature is UNSET:
            temperature = self._resolve_temperature(self.temperature)

        max_retries = 3
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                # Abort if project was cancelled
                if self.shared_context and getattr(self.shared_context, "_cancelled", False):
                    raise RuntimeError("Cancelled")
                print(f"🔄 {self.agent_id} LLM call attempt {attempt}/{max_retries} (model: {model})")
                
                # Use per-project API key override if provided (ephemeral on SharedContext)
                api_key_override = getattr(self.shared_context, "_ephemeral_api_key", None) if self.shared_context else None
                fallback_models_override = getattr(self.shared_context, "_ephemeral_fallback_models", None) if self.shared_context else None
                if self.shared_context and getattr(self.shared_context, "_force_model_override", False):
                    # A project-forced model must actually be the model that answers —
                    # a Bifrost fallback substituting a different one silently defeats
                    # the point of forcing it.
                    fallback_models_override = []
                
                # Use streaming if we have an event emitter (for real-time reasoning display)
                if self.event_emitter:
                    import time
                    _seq = [0]
                    # Get current phase for bidding distinction
                    # Check _bidding_phase flag first (set during bid_on_task)
                    if getattr(self, '_bidding_phase', False):
                        current_phase = 'bidding'
                    else:
                        current_phase = getattr(self.shared_context, 'current_phase', None) if self.shared_context else None
                    
                    async def emit_thinking_event(event: Dict):
                        """Emit thinking events to UI"""
                        event_type = event.get("type", "unknown")
                        _seq[0] += 1
                        await self.event_emitter.emit(
                            f"agent.streaming.{event_type}",
                            self.shared_context.run_id if self.shared_context else None,
                            {
                                "project_id": self.shared_context.project_id if self.shared_context else None,
                                "agent_id": self.agent_id,
                                "phase": current_phase,
                                "timestamp": f"{time.time()}-{_seq[0]}",
                                **event
                            },
                        )
                    
                    _stream_id = uuid.uuid4().hex[:8]
                    _project_id = self.shared_context.project_id if self.shared_context else None

                    async def _do_streaming_call():
                        # Bracket the call from inside the closure so the [STREAM.end]
                        # log fires even if the awaiter is cancelled and this task is
                        # orphaned (continues running in the background).
                        _t0 = time.monotonic()
                        # DEBUG: register stream and warn if overlap with another
                        # open stream for this project. See concurrency tracker
                        # comments at module top.
                        _existing = _stream_open(_project_id, _stream_id, self.agent_id, current_phase)
                        if _existing:
                            self.logger.warning(
                                "[CONCURRENCY] new stream opened while %d already active "
                                "project=%s new_stream=%s new_agent=%s new_phase=%s existing=%s",
                                len(_existing), _project_id, _stream_id, self.agent_id,
                                current_phase,
                                [(o.get("agent_id"), o.get("phase")) for o in _existing],
                            )
                            if self.event_emitter:
                                try:
                                    asyncio.create_task(self.event_emitter.emit(
                                        EventSchema.AGENT_STREAM_CONCURRENCY,
                                        self.shared_context.run_id if self.shared_context else None,
                                        {
                                            "project_id": _project_id,
                                            "new_stream": {
                                                "stream_id": _stream_id,
                                                "agent_id": self.agent_id,
                                                "phase": current_phase,
                                                "path": "callback",
                                            },
                                            "existing_streams": _existing,
                                            "active_count": len(_existing) + 1,
                                            "timestamp": f"{time.time()}-0",
                                        }
                                    ))
                                except Exception:
                                    pass
                        self.logger.info(
                            "[STREAM.start] stream_id=%s agent=%s phase=%s path=callback",
                            _stream_id, self.agent_id, current_phase,
                        )
                        _exc_name = None
                        _capture = self._begin_callback_capture(messages, model, temperature, kwargs)
                        if _capture:
                            async def _cb_teed(_event):
                                self._capture_callback_event(_capture, _event)
                                await emit_thinking_event(_event)
                            _callback_to_pass = _cb_teed
                        else:
                            _callback_to_pass = emit_thinking_event
                        _result = None
                        _capture_error = None
                        try:
                            _result = await self.llm_client.stream_completion_with_callback(
                                messages=messages,
                                model=model,
                                temperature=temperature,
                                api_key_override=api_key_override,
                                fallback_models_override=fallback_models_override,
                                event_callback=_callback_to_pass,
                                **kwargs
                            )
                            return _result
                        except BaseException as _e:
                            _exc_name = type(_e).__name__
                            _capture_error = str(_e)
                            raise
                        finally:
                            if _capture:
                                try:
                                    await self._finalize_callback_capture(_capture, _result, error=_capture_error)
                                except Exception:
                                    pass
                            _elapsed = time.monotonic() - _t0
                            _stream_close(_project_id, _stream_id)
                            self.logger.info(
                                "[STREAM.end] stream_id=%s agent=%s phase=%s path=callback elapsed=%.2fs exc=%s",
                                _stream_id, self.agent_id, current_phase,
                                _elapsed, _exc_name,
                            )
                            # Emit a public termination event when the stream did NOT
                            # complete naturally — UI uses this to badge the matching
                            # thought (e.g. "cancelled"). Fire-and-forget so the original
                            # exception keeps propagating without being masked.
                            if self.event_emitter:
                                if _exc_name:
                                    _reason = "cancelled" if _exc_name == "CancelledError" else "error"
                                    try:
                                        asyncio.create_task(self.event_emitter.emit(
                                            EventSchema.AGENT_STREAM_TERMINATED,
                                            self.shared_context.run_id if self.shared_context else None,
                                            {
                                                "project_id": _project_id,
                                                "agent_id": self.agent_id,
                                                "phase": current_phase,
                                                "stream_id": _stream_id,
                                                "reason": _reason,
                                                "exc_name": _exc_name,
                                                "elapsed": _elapsed,
                                                "timestamp": f"{time.time()}-0",
                                            }
                                        ))
                                    except Exception:
                                        pass
                                else:
                                    # Natural completion. The UI needs `elapsed`
                                    # to detect provider-buffered responses
                                    # (deltaCount==1 && elapsed > threshold) and
                                    # to display a meaningful "wire time" badge
                                    # in place of the misleading thinking_time≈0
                                    # that buffered streams produce.
                                    try:
                                        asyncio.create_task(self.event_emitter.emit(
                                            EventSchema.AGENT_STREAM_CLOSED,
                                            self.shared_context.run_id if self.shared_context else None,
                                            {
                                                "project_id": _project_id,
                                                "agent_id": self.agent_id,
                                                "phase": current_phase,
                                                "stream_id": _stream_id,
                                                "elapsed": _elapsed,
                                                "timestamp": f"{time.time()}-0",
                                            }
                                        ))
                                    except Exception:
                                        pass

                    result = await self.await_with_cancellation(
                        _do_streaming_call(),
                        'llm.streaming.callback',
                    )
                else:
                    # Fallback to non-streaming
                    async def _do_call():
                        return await self.llm_client.chat_completion(
                            messages=messages,
                            model=model,
                            temperature=temperature,
                            api_key_override=api_key_override,
                            fallback_models_override=fallback_models_override,
                            **kwargs
                        )
                    result = await self.await_with_cancellation(
                        _do_call(),
                        'llm.chat_completion',
                    )
                
                print(f"✓ {self.agent_id} LLM call succeeded")

                if isinstance(result, dict) and result.get("param_downgrade"):
                    await self._notify_param_downgrade(model, result["param_downgrade"])

                # Backward compatibility: if no tools, return just content string
                if not kwargs.get("tools"):
                    if return_details and isinstance(result, dict):
                        return {
                            "content": result.get("content", "") or "",
                            "finish_reason": result.get("finish_reason")
                            or result.get("stop_reason"),
                            "error": result.get("error"),
                        }
                    return result.get("content", "") if isinstance(result, dict) else result
                
                # With tools, return full dict
                return result
                
            except Exception as e:
                last_error = e
                print(f"❌ {self.agent_id} LLM call failed (attempt {attempt}/{max_retries}): {e}")
                # Also stop retrying if cancelled during wait
                if self.shared_context and getattr(self.shared_context, "_cancelled", False):
                    raise RuntimeError("Cancelled")
                # AppFactory-316: same transient gate as streaming/legacy retries —
                # 403/401/400/overflow/etc. must not burn the attempt ladder.
                from llm.errors import is_transient_provider_error

                if not is_transient_provider_error(e):
                    raise

                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                    print(f"⏳ Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
        
        # All retries exhausted
        raise RuntimeError(
            f"{self.agent_id} LLM call failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    async def _plugin_llm_call(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Single-shot, non-streaming, invisible LLM call for plugins (AppFactory-149).

        The compactor door wired into PluginServices.call_model. It deliberately
        bypasses call_llm's streaming path so the summariser never surfaces in the
        execution feed (ADR-0004 amendment: the compactor call is private) and
        honours cancellation.

        Model precedence (ADR-0014 Decision 8): an explicit compactor model —
        plugins.context_compaction.summary_model, passed here — is honoured
        ABOVE the agent's run-config default chain, so compaction stays on its own
        (cheap) model instead of the agent's big models.default. A project-wide
        force override still wins (pinning everything to one model means
        everything). A None model falls through to the compaction subsystem chain
        (models.compaction → models.default → project → the agent's own model).
        """
        if not self.llm_client:
            raise RuntimeError("LLM client not injected")
        # Force override only defers the compactor model when it actually names a
        # model — force flag set with no project model is a no-op, so the plugin's
        # summary_model should still win there rather than fall to models.default.
        sc = self.shared_context
        force_active = bool(
            getattr(sc, "_force_model_override", False)
            and getattr(sc, "_model_override", None)
        ) if sc else False
        if model and not force_active:
            resolved = model
        else:
            resolved = self._resolve_project_model(model, subsystem="compaction")
        api_key_override = (
            getattr(self.shared_context, "_ephemeral_api_key", None)
            if self.shared_context
            else None
        )
        fallback_models_override = (
            getattr(self.shared_context, "_ephemeral_fallback_models", None)
            if self.shared_context
            else None
        )
        if force_active:
            fallback_models_override = []
        result = await self.await_with_cancellation(
            self.llm_client.chat_completion(
                messages=messages,
                model=resolved,
                temperature=temperature,
                api_key_override=api_key_override,
                fallback_models_override=fallback_models_override,
            ),
            "plugin.compactor",
        )
        if isinstance(result, dict):
            return result.get("content", "") or ""
        return result or ""

    async def call_llm_json(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict:
        if not self.llm_client:
            raise RuntimeError("LLM client not injected")
        subsystem = kwargs.pop("subsystem", "agent_default")
        requested_model = kwargs.pop("model", None)
        model = self._resolve_project_model(requested_model, subsystem=subsystem)
        temperature = kwargs.pop("temperature", UNSET)
        if temperature is UNSET:
            temperature = self._resolve_temperature(self.temperature)

        # Abort if cancelled
        if self.shared_context and getattr(self.shared_context, "_cancelled", False):
            raise RuntimeError("Cancelled")
        # No _ephemeral_fallback_models resolution here: both stream_completion_with_json
        # and chat_completion_with_json below never send Bifrost fallbacks regardless of
        # override (extra_body conflicts with response_format on some providers — see
        # their docstrings/comments), so there is nothing for an override to control.
        api_key_override = getattr(self.shared_context, "_ephemeral_api_key", None) if self.shared_context else None

        # Use streaming if we have an event emitter (for real-time reasoning display)
        if self.event_emitter:
            import time
            _seq = [0]
            # Get current phase for bidding distinction
            # Check _bidding_phase flag first (set during bid_on_task)
            if getattr(self, '_bidding_phase', False):
                current_phase = 'bidding'
            else:
                current_phase = getattr(self.shared_context, 'current_phase', None) if self.shared_context else None
            
            async def emit_thinking_event(event: Dict):
                """Emit thinking events to UI"""
                event_type = event.get("type", "unknown")
                _seq[0] += 1
                await self.event_emitter.emit(
                    f"agent.streaming.{event_type}",
                    self.shared_context.run_id if self.shared_context else None,
                    {
                        "project_id": self.shared_context.project_id if self.shared_context else None,
                        "agent_id": self.agent_id,
                        "phase": current_phase,
                        "timestamp": f"{time.time()}-{_seq[0]}",
                        **event
                    },
                )
            
            _stream_id = uuid.uuid4().hex[:8]
            _project_id = self.shared_context.project_id if self.shared_context else None

            async def _do_streaming_call():
                # Bracket from inside the closure so [STREAM.end] fires even if the
                # awaiter is cancelled and this task continues as an orphan.
                _t0 = time.monotonic()
                _existing = _stream_open(_project_id, _stream_id, self.agent_id, current_phase)
                if _existing:
                    self.logger.warning(
                        "[CONCURRENCY] new stream opened while %d already active "
                        "project=%s new_stream=%s new_agent=%s new_phase=%s existing=%s",
                        len(_existing), _project_id, _stream_id, self.agent_id,
                        current_phase,
                        [(o.get("agent_id"), o.get("phase")) for o in _existing],
                    )
                    if self.event_emitter:
                        try:
                            asyncio.create_task(self.event_emitter.emit(
                                EventSchema.AGENT_STREAM_CONCURRENCY,
                                self.shared_context.run_id if self.shared_context else None,
                                {
                                    "project_id": _project_id,
                                    "new_stream": {
                                        "stream_id": _stream_id,
                                        "agent_id": self.agent_id,
                                        "phase": current_phase,
                                        "path": "json",
                                    },
                                    "existing_streams": _existing,
                                    "active_count": len(_existing) + 1,
                                    "timestamp": f"{time.time()}-0",
                                }
                            ))
                        except Exception:
                            pass
                self.logger.info(
                    "[STREAM.start] stream_id=%s agent=%s phase=%s path=json",
                    _stream_id, self.agent_id, current_phase,
                )
                _exc_name = None
                try:
                    return await self.llm_client.stream_completion_with_json(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        api_key_override=api_key_override,
                        event_callback=emit_thinking_event,
                        **kwargs
                    )
                except BaseException as _e:
                    _exc_name = type(_e).__name__
                    raise
                finally:
                    _elapsed = time.monotonic() - _t0
                    _stream_close(_project_id, _stream_id)
                    self.logger.info(
                        "[STREAM.end] stream_id=%s agent=%s phase=%s path=json elapsed=%.2fs exc=%s",
                        _stream_id, self.agent_id, current_phase,
                        _elapsed, _exc_name,
                    )
                    # Same termination/close signals as the callback path — see
                    # comments there for rationale.
                    if self.event_emitter:
                        if _exc_name:
                            _reason = "cancelled" if _exc_name == "CancelledError" else "error"
                            try:
                                asyncio.create_task(self.event_emitter.emit(
                                    EventSchema.AGENT_STREAM_TERMINATED,
                                    self.shared_context.run_id if self.shared_context else None,
                                    {
                                        "project_id": _project_id,
                                        "agent_id": self.agent_id,
                                        "phase": current_phase,
                                        "stream_id": _stream_id,
                                        "reason": _reason,
                                        "exc_name": _exc_name,
                                        "elapsed": _elapsed,
                                        "timestamp": f"{time.time()}-0",
                                    }
                                ))
                            except Exception:
                                pass
                        else:
                            try:
                                asyncio.create_task(self.event_emitter.emit(
                                    EventSchema.AGENT_STREAM_CLOSED,
                                    self.shared_context.run_id if self.shared_context else None,
                                    {
                                        "project_id": _project_id,
                                        "agent_id": self.agent_id,
                                        "phase": current_phase,
                                        "stream_id": _stream_id,
                                        "elapsed": _elapsed,
                                        "timestamp": f"{time.time()}-0",
                                    }
                                ))
                            except Exception:
                                pass

            return await self.await_with_cancellation(
                _do_streaming_call(),
                'llm.streaming.json',
            )

        # Fallback to non-streaming if no event emitter
        async def _do_call():
            return await self.llm_client.chat_completion_with_json(
                messages=messages,
                model=model,
                temperature=temperature,
                api_key_override=api_key_override,
                return_reasoning=False,
                **kwargs
            )

        return await self.await_with_cancellation(
            _do_call(),
            'llm.chat_completion_with_json',
        )
    
    async def _generate_bid_reasoning(
        self, 
        task: Dict, 
        fit_score: float
    ) -> str:
        """Generate explanation for bid fit_score (how well agent can handle task)"""
        if fit_score == 0:
            return f"{self.agent_type} cannot handle this task type"
        elif fit_score < 0.3:
            return f"{self.agent_type} has minimal capability for this task"
        elif fit_score < 0.7:
            return f"{self.agent_type} can handle this but not optimal"
        else:
            return f"{self.agent_type} is well-suited for this task"
    
    async def execute_tool(
        self, tool_id: str, params: Dict[str, Any], *,
        public_params=None, result_transform=None,
    ) -> Dict[str, Any]:
        """
        Execute tool in container via MCP.
        
        Args:
            tool_id: Tool identifier (e.g., 'create_file', 'run_command')
            params: Tool parameters
        
        Returns:
            Tool execution result
        """
        if not self.mcp_executor:
            self.logger.error(f"[AGENT] {self.agent_id} has no mcp_executor! Cannot execute {tool_id}")
            raise RuntimeError("MCP executor not injected")
        
        if not self.shared_context:
            raise RuntimeError("Shared context not available")
        
        # Get current project ID from context
        project_id = self.shared_context.project_id

        try:
            visible = params if public_params is None else public_params
            safe_params = {k: v for k, v in visible.items() if not k.startswith("_")}

            await self._emit_event("tool_started", {
                "agent_id": self.agent_id,
                "agent_display_name": self.get_display_name(),
                "tool_id": tool_id,
                "params": safe_params,
                "task_id": TaskSchema.get_id(self.current_task) if self.current_task else None,
                "task_description": self._get_task_title(self.current_task) if self.current_task else None,
                "task_type": TaskSchema.get_type(self.current_task) if self.current_task else None,
            })
        except Exception:
            pass

        # Execute via MCP (cancellation-aware)
        async def _do_exec():
            return await self.mcp_executor.execute_tool(
                tool_id,
                project_id,
                params
            )
        if self.cancellation_token:
            exec_task = asyncio.create_task(_do_exec())
            waiter = asyncio.create_task(self.cancellation_token.wait())
            done, pending = await asyncio.wait({exec_task, waiter}, return_when=asyncio.FIRST_COMPLETED)
            for p in pending:
                p.cancel()
            if waiter in done:
                exec_task.cancel()
                raise RuntimeError("Cancelled")
            result = exec_task.result()
        else:
            result = await _do_exec()
        result = without_call_observation(result)

        # Emit event for UI
        if result_transform is not None:
            result = await result_transform(result)
        visible = params if public_params is None else public_params
        safe_params = {k: v for k, v in visible.items() if not k.startswith("_")}

        # Live result keeps image_data_url for the VL follow-up; events must not.
        await self._emit_event("tool_executed", {
            "agent_id": self.agent_id,
            "agent_display_name": self.get_display_name(),
            "tool_id": tool_id,
            "params": safe_params,
            "result": journal_safe_tool_result(result),
            "task_id": TaskSchema.get_id(self.current_task) if self.current_task else None,
            "task_description": self._get_task_title(self.current_task) if self.current_task else None,
            "task_type": TaskSchema.get_type(self.current_task) if self.current_task else None,
        })
        
        return result
    
    async def _emit_event(self, event_type: str, data: Dict):
        """Emit event for UI updates"""
        if self.event_emitter:
            try:
                if "project_id" not in data:
                    pid = None
                    if self.shared_context and getattr(self.shared_context, "project_id", None):
                        pid = self.shared_context.project_id
                    elif self.current_task:
                        pid = TaskSchema.get_project_id(self.current_task)
                    if pid:
                        data = {**data, "project_id": pid}
            except Exception:
                pass
            run_id = self.shared_context.run_id if self.shared_context else None
            await self.event_emitter.emit(event_type, run_id, data)

    def _resolve_project_model(
        self,
        requested_model: str | None = None,
        subsystem: str = "agent_default",
    ) -> str:
        """Resolve model with run configuration priority and UI-selected fallback.

        `subsystem` picks the run-config models entry consulted first (e.g.
        "evaluation" keeps auction bids on a cheap model regardless of the
        agent's own main model).
        """
        fallback_model = requested_model or self.model
        if self.shared_context and hasattr(self.shared_context, "get_model"):
            return self.shared_context.get_model(subsystem, fallback_model, agent_id=self.agent_id)
        return fallback_model

    def _resolve_reasoning_effort(
        self,
        default: Optional[str] = "medium",
    ) -> Optional[str]:
        """Resolve reasoning effort with project-level override priority.

        Order: SharedContext._reasoning_effort_override → `default` (typically
        the agent's static `reasoning_effort` from its config). Used by
        subclasses when constructing StreamingAgentRunner / SimpleAgentRunner
        so a project-create-time choice ("Enable Reasoning, Effort=Low") wins
        over per-agent defaults without mutating the agent record.
        """
        if self.shared_context and hasattr(self.shared_context, "get_reasoning_effort"):
            return self.shared_context.get_reasoning_effort(default=default, agent_id=self.agent_id)
        return default
    
    def _resolve_temperature(
        self,
        default: Optional[float] = 1.0,
    ) -> Optional[float]:
        if self.shared_context and hasattr(self.shared_context, "get_temperature"):
            return self.shared_context.get_temperature(default=default, agent_id=self.agent_id)
        return default

    def _resolve_step_limit(self, default: int = 30) -> int:
        if self.shared_context and hasattr(self.shared_context, "get_step_limit"):
            return self.shared_context.get_step_limit(default=default, agent_id=self.agent_id)
        return default

    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.agent_id}, type={self.agent_type})>"
