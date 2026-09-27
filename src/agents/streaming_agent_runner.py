"""
Streaming Agent Runner

Handles streaming responses with:
- Real-time thinking/reasoning display
- Inline tool execution during generation
- Structured event emission for UI
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional

from agents.tool_failure_budget import ToolFailureBudget
from llm.errors import ContextOverflowError, is_transient_provider_error
from agents.vision_tool_result import (
    format_tool_output as _format_tool_output_impl,
    journal_safe_tool_result as _journal_safe_tool_result_impl,
    scrub_for_capture,
    vision_followups_from_tool_result as _vision_followups_impl,
)

logger = logging.getLogger(__name__)

ToolHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]

# AppFactory-316: provider/stream blip retries before first tool execution of the round.
_ROUND_PROVIDER_RETRY_BACKOFF_S = (1.0, 5.0, 20.0)
_MAX_ROUND_PROVIDER_RETRIES = len(_ROUND_PROVIDER_RETRY_BACKOFF_S)


@dataclass
class ThinkingBlock:
    """Represents a thinking/reasoning block"""
    content: str = ""
    thinking_time: float = 0.0
    collapsed: bool = True


@dataclass
class ToolCallBlock:
    """Represents a tool call and its result"""
    call_id: str = ""
    item_id: str = ""   # original item id from the Responses API response
    name: str = ""
    arguments: str = ""
    result: Optional[Dict[str, Any]] = None
    status: str = "pending"  # pending, executing, completed, error


@dataclass
class StreamingRunnerResult:
    """Result from a streaming agent run"""
    content: str = ""
    thinking_blocks: List[ThinkingBlock] = field(default_factory=list)
    tool_calls: List[ToolCallBlock] = field(default_factory=list)
    input_items: List[Dict[str, Any]] = field(default_factory=list)
    # Terminal stop_reason of the last round ("length" = output truncated by the
    # token cap). Exposed so the caller can refuse to mark a cut-off run COMPLETED.
    stop_reason: Optional[str] = None
    # Last provider/stream error text when the run ends without a good answer
    # (AppFactory-316 — prefer this over a bare "no final content").
    error: Optional[str] = None


class _RunRoundMutable:
    """Per-round mutable state while draining ``stream_responses`` (run mode)."""

    def __init__(self) -> None:
        self.current_thinking = ThinkingBlock()
        self.pending_tool_calls: Dict[str, ToolCallBlock] = {}
        self.text_content = ""
        self.has_tool_calls = False
        self.usage: Optional[Dict[str, Any]] = None
        self.stop_reason: Optional[str] = None
        self.stream_error: Optional[str] = None
        self.stream_error_type: Optional[str] = None


class StreamingAgentRunner:
    """
    Agent runner that streams responses with thinking + inline tool execution.
    
    Flow:
    1. Send input to LLM via Responses API (streaming)
    2. Yield thinking tokens as they arrive
    3. When tool call completes, execute it inline
    4. Submit tool result and continue generation
    5. Yield text tokens as they arrive
    6. Return final result with all blocks
    """
    
    def __init__(
        self,
        llm_client,
        tools: List[Dict[str, Any]],
        tool_handler: ToolHandler,
        local_tools: Optional[Dict[str, ToolHandler]] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = "medium",
        max_tool_rounds: int = 10,
        event_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        cancellation_token=None,
        stream_round_timeout_seconds: Optional[float] = None,
        temperature: Optional[float] = None,
        # Inspector capture context (all optional; if any of project_id /
        # agent_id / agent_llm_calls_store is None the capture is a no-op).
        project_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        agent_llm_calls_store=None,
        event_emitter=None,
        # Tool ledger: persists a tool_call record before each tool
        # executes and a tool_result record after. No-op when None.
        message_store=None,
        workflow_node_id: Optional[str] = None,
        # Plugin hook dispatch (ADR-0004). None => every hook site is a
        # no-op branch and behavior is byte-identical to the pre-plugin
        # runner, including the ledger records.
        plugin_host=None,
        max_tool_failures: Optional[int] = None,
    ):
        self.llm_client = llm_client
        self.tools = list(tools)
        self.tool_handler = tool_handler
        self.local_tools = dict(local_tools or {})
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tool_rounds = max_tool_rounds
        self.event_callback = event_callback
        self.cancellation_token = cancellation_token
        self.stream_round_timeout_seconds = stream_round_timeout_seconds
        self.temperature = temperature
        self.project_id = project_id
        self.run_id = run_id
        self.agent_id = agent_id
        self.task_id = task_id
        self.agent_llm_calls_store = agent_llm_calls_store
        self.event_emitter = event_emitter
        self.message_store = message_store
        self.workflow_node_id = workflow_node_id
        self.plugin_host = plugin_host
        self.tool_failure_budget = ToolFailureBudget(max_tool_failures)
        self._ledger_gap_warned = False

    @staticmethod
    def stream_round_timeout_from_env() -> Optional[float]:
        raw = (os.environ.get("AppFactory_STREAM_ROUND_TIMEOUT_SECONDS") or "").strip()
        if not raw:
            return None
        try:
            v = float(raw)
            return v if v > 0 else None
        except ValueError:
            return None

    def _is_cancelled(self) -> bool:
        # Genuine cancel = token set OR a real Task-level cancel in flight
        # (WorkflowEngine's run_timeout_seconds cancels via asyncio.wait_for
        # without setting the token); a spurious scope-leak cancel stays False
        # so the tool-handler guards below still contain it. See host.py:_dispatch.
        # Imported lazily: orchestration/__init__ is heavy and cycles back here.
        from orchestration.cancellation import is_genuinely_cancelled
        return is_genuinely_cancelled(self.cancellation_token)

    def _ensure_not_cancelled(self, location: str) -> None:
        if self._is_cancelled():
            raise RuntimeError("Cancelled")

    async def _cancelable_sleep(self, seconds: float, location: str = "round_retry_backoff") -> None:
        """Sleep in short slices so Stop/cancel does not wait out the full backoff."""
        if seconds <= 0:
            self._ensure_not_cancelled(location)
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(seconds)
        while True:
            self._ensure_not_cancelled(location)
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.2, remaining))

    @staticmethod
    def _round_has_finalized_tools(st: _RunRoundMutable) -> bool:
        """True once any tool_call.done fired — side effects must not be re-asked."""
        return any(tc.status == "executing" for tc in st.pending_tool_calls.values())

    async def _emit_round_provider_retry(
        self,
        *,
        round_num: int,
        attempt: int,
        error: str,
        backoff: float,
    ) -> None:
        logger.warning(
            "[STREAMING] Round %d provider retry %d/%d after error=%s backoff=%.0fs",
            round_num + 1,
            attempt,
            _MAX_ROUND_PROVIDER_RETRIES,
            error,
            backoff,
        )
        await self._emit_event({
            "type": "round.retry",
            "attempt": attempt,
            "max_attempts": _MAX_ROUND_PROVIDER_RETRIES,
            "error": error,
            "backoff_seconds": backoff,
            "round": round_num + 1,
        })

    async def _await_with_cancellation(self, coro, location: str):
        from orchestration.workflow_task_lifecycle import cancel_and_await

        self._ensure_not_cancelled(location)
        if not self.cancellation_token:
            return await coro

        op_task = asyncio.create_task(coro)
        waiter = asyncio.create_task(self.cancellation_token.wait())
        try:
            done, pending = await asyncio.wait({op_task, waiter}, return_when=asyncio.FIRST_COMPLETED)
            for pending_task in pending:
                pending_task.cancel()

            if waiter in done:
                await cancel_and_await([op_task], label=location)
                raise RuntimeError("Cancelled")

            return op_task.result()
        finally:
            leftover = [t for t in (op_task, waiter) if not t.done()]
            if leftover:
                await cancel_and_await(leftover, label=location)

    async def _emit_event(self, event: Dict[str, Any]):
        """Emit event to callback if registered"""
        if self.event_callback:
            try:
                await self.event_callback(event)
            except Exception as e:
                logger.warning("Event callback error: %s", e)

    # ------------------------------------------------------------------
    # Inspector capture (Agent Invocation Inspector — see storage/agent_llm_calls_store.py)
    # ------------------------------------------------------------------

    def _capture_enabled(self) -> bool:
        return bool(
            self.agent_llm_calls_store
            and self.project_id
            and self.agent_id
        )

    def _begin_capture(
        self,
        round_num: int,
        stream_kwargs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self._capture_enabled():
            return None
        return {
            "call_id": str(uuid.uuid4()),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "round_num": round_num,
            "stream_kwargs": stream_kwargs,
            "events": [],
        }

    @staticmethod
    async def _tee_stream(
        stream: AsyncGenerator[Dict[str, Any], None],
        capture: Dict[str, Any],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        async for ev in stream:
            try:
                capture["events"].append(ev)
            except Exception:
                pass  # capture is best-effort; never break the consumer
            yield ev

    @staticmethod
    def _split_system_messages(
        input_items: Optional[List[Dict[str, Any]]],
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        if not input_items:
            return None, []
        system_parts: List[str] = []
        messages: List[Dict[str, Any]] = []
        for item in input_items:
            if isinstance(item, dict) and item.get("role") == "system":
                content = item.get("content")
                if isinstance(content, str):
                    system_parts.append(content)
                else:
                    try:
                        system_parts.append(json.dumps(content, default=str))
                    except Exception:
                        system_parts.append(str(content))
            else:
                messages.append(item)
        return ("\n\n".join(system_parts) if system_parts else None), messages

    @staticmethod
    def _extract_response(events: List[Dict[str, Any]]) -> Dict[str, Any]:
        text_parts: List[str] = []
        thinking_parts: List[str] = []
        tool_uses: Dict[str, Dict[str, Any]] = {}
        usage: Optional[Dict[str, Any]] = None
        stop_reason: Optional[str] = None
        error: Optional[str] = None
        thinking_done_full: Optional[str] = None
        text_done_full: Optional[str] = None
        response_id: Optional[str] = None

        for ev in events:
            t = ev.get("type", "")
            if t == "text.delta":
                text_parts.append(ev.get("content", ""))
            elif t == "text.done":
                if ev.get("content"):
                    text_done_full = ev["content"]
            elif t == "thinking.delta":
                thinking_parts.append(ev.get("content", ""))
            elif t == "thinking.done":
                if ev.get("content"):
                    thinking_done_full = ev["content"]
                if ev.get("usage"):
                    usage = ev["usage"]
            elif t == "tool_call.start":
                cid = ev.get("call_id", "")
                tool_uses.setdefault(cid, {
                    "id": cid,
                    "name": ev.get("name"),
                    "input": None,
                })
            elif t == "tool_call.done":
                cid = ev.get("call_id", "")
                entry = tool_uses.setdefault(cid, {"id": cid, "name": None, "input": None})
                if ev.get("name"):
                    entry["name"] = ev["name"]
                entry["input"] = ev.get("arguments")
            elif t == "response.started":
                response_id = ev.get("response_id") or response_id
            elif t == "response.done":
                if ev.get("usage"):
                    usage = ev["usage"]
                if ev.get("stop_reason"):
                    stop_reason = ev["stop_reason"]
                response_id = ev.get("response_id") or response_id
            elif t == "error":
                error = error or ev.get("error", "Unknown error")

        return {
            "text": text_done_full if text_done_full is not None else "".join(text_parts),
            "thinking": thinking_done_full if thinking_done_full is not None else "".join(thinking_parts),
            "tool_uses": list(tool_uses.values()),
            "stop_reason": stop_reason,
            "usage": usage,
            "error": error,
            "response_id": response_id,
        }

    async def _finalize_capture(
        self,
        capture: Optional[Dict[str, Any]],
        error: Optional[str] = None,
    ) -> None:
        """Build and persist the capture doc. Defensive: must never raise."""
        if capture is None:
            return
        try:
            kwargs = capture.get("stream_kwargs", {})
            input_items = kwargs.get("input_items")
            system, messages = self._split_system_messages(input_items)
            messages = scrub_for_capture(messages)
            response = self._extract_response(capture.get("events") or [])
            if error and not response.get("error"):
                response["error"] = error
            if response.get("error") and not response.get("stop_reason"):
                response["stop_reason"] = "error"
            doc = {
                "_id": capture["call_id"],
                "project_id": self.project_id,
                "run_id": self.run_id,
                "agent_id": self.agent_id,
                "task_id": self.task_id,
                "turn_index": capture.get("round_num", 0),
                "started_at": capture.get("started_at"),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "model": kwargs.get("model"),
                "request": {
                    "system": system,
                    "messages": messages,
                    "tools": kwargs.get("tools"),
                    "temperature": kwargs.get("temperature"),
                    "max_tokens": None,
                    "params": {
                        "reasoning_effort": kwargs.get("reasoning_effort"),
                        "api_key_override": bool(kwargs.get("api_key_override")),
                    },
                },
                "response": response,
            }
            await self.agent_llm_calls_store.insert_call(doc)
            if self.event_emitter is not None:
                try:
                    from schemas.event_schema import EventSchema
                    await self.event_emitter.emit(
                        EventSchema.AGENT_INVOCATION_CAPTURED,
                        self.run_id,
                        {
                            "project_id": self.project_id,
                            "call_id": capture["call_id"],
                            "agent_id": self.agent_id,
                            "task_id": self.task_id,
                            "turn_index": capture.get("round_num", 0),
                            "token_counts": response.get("usage"),
                            "stop_reason": response.get("stop_reason"),
                            "truncated": doc.get("truncated", False),
                        },
                    )
                except Exception as exc:
                    logger.warning("agent.invocation.captured emit failed: %s", exc)
        except Exception as exc:
            logger.warning("agent_llm_calls capture finalize failed: %s", exc)
    
    @staticmethod
    def _format_tool_output(result: Dict[str, Any]) -> str:
        """Convert internal tool result dict to a clean string for the LLM."""
        return _format_tool_output_impl(result)

    @staticmethod
    def _vision_followups_from_tool_result(result: Any) -> List[Dict[str, Any]]:
        """Turn attachment_view image_data_url into Responses API multimodal input."""
        return _vision_followups_impl(result)

    @staticmethod
    def _journal_safe_tool_result(result: Any) -> Any:
        """Drop bulky vision payloads before persisting tool results."""
        return _journal_safe_tool_result_impl(result)

    def _ledger_enabled(self) -> bool:
        if not self.message_store:
            return False
        if not (self.project_id and self.run_id):
            # Records without run_id surface in EVERY run's view (the
            # project-lineage $or in MessageStore.get_messages) — better no
            # ledger for this run than a polluted one for all of them.
            if not self._ledger_gap_warned:
                self._ledger_gap_warned = True
                logger.warning(
                    "[LEDGER] message_store set but project_id/run_id missing "
                    "(project_id=%s run_id=%s) — tool ledger disabled for this runner",
                    self.project_id,
                    self.run_id,
                )
            return False
        return True

    async def _execute_tool(
        self, name: str, arguments: str, call_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a single tool call, journaling it as a tool_call/tool_result
        pair.

        Persist is deliberately LOUD, unlike the best-effort Inspector capture:
        a silent journal gap would make features that read the journal —
        resuming a run after a restart, pausing to ask a human — lie about
        what ran. The call record lands BEFORE execution: a persist failure
        blocks the tool, and an interruption mid-execution leaves the call
        hanging (no paired result), which is exactly the "agent stopped here"
        marker those features build on.
        """
        self.tool_failure_budget.check()
        ledger = self._ledger_enabled()
        if ledger and not call_id:
            # Breaks the "no execution without a record" invariant — the one
            # silent exception would be a provider omitting call_id, so make
            # it loud enough to notice in logs.
            logger.warning(
                "[LEDGER] executing tool %s WITHOUT a journal record — provider sent no call_id",
                name,
            )
            ledger = False
        if ledger:
            await self.message_store.append_tool_call(
                self.project_id,
                tool_call_id=call_id,
                name=name,
                arguments=arguments,
                run_id=self.run_id,
                agent_id=self.agent_id,
                task_id=self.task_id,
                workflow_node_id=self.workflow_node_id,
            )
        result = await self._execute_tool_inner(name, arguments, call_id)
        if self.plugin_host is not None:
            # BEFORE the journal append on purpose: the journal must record
            # what the dialog will contain, or restart rehydration would
            # reinject the untransformed payload (ADR-0004).
            result = await self.plugin_host.dispatch_tool_result(
                name=name, arguments=arguments, call_id=call_id, result=result
            )
        if ledger:
            status = (
                "error"
                if isinstance(result, dict) and result.get("status") == "error"
                else "ok"
            )
            await self.message_store.append_tool_result(
                self.project_id,
                tool_call_id=call_id,
                name=name,
                result=self._journal_safe_tool_result(result),
                status=status,
                run_id=self.run_id,
                agent_id=self.agent_id,
                task_id=self.task_id,
            )
            if name == "ask_human":
                # The answer is journaled now — the pair is closed, so any further
                # answer 409s on the journal itself. Release the in-process live
                # claim (F3) so a later question reusing this call id (providers
                # recycle ids, ADR-0008) isn't falsely blocked on the restart path.
                from tools.ask_human import clear_live_answer_claim

                clear_live_answer_claim(self.project_id, call_id)
        if isinstance(result, dict):
            from orchestration.workflow_task_lifecycle import (
                raise_if_session_unavailable,
                raise_if_tool_outcome_unknown,
            )

            raise_if_session_unavailable(result, name, call_id or "")
            if result.get("outcome_unknown"):
                raise_if_tool_outcome_unknown(result, name, call_id or "")
        self.tool_failure_budget.record(name, result)
        return result

    async def _execute_tool_inner(
        self, name: str, arguments: str, call_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a single tool call"""
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as e:
            logger.warning("[STREAMING] tool=%s invalid JSON arguments: %s", name, e)
            return {
                "status": "error",
                "error": f"Invalid tool arguments (not valid JSON): {e}",
            }

        self._ensure_not_cancelled(f"tool:{name}")

        # Check local tools first
        handler = self.local_tools.get(name)
        if handler is not None:
            try:
                return await self._await_with_cancellation(handler(args), f"local_tool:{name}")
            except asyncio.CancelledError:
                # A library can raise this without our run being cancelled — a
                # failed MCP connection unwinds its cancel scope on the wrong
                # task and the scope's own CancelledError comes out here. It is
                # not an Exception, so every handler below misses it, and the
                # run then dies with no result recorded and no error logged.
                # After a live MCP session the remote side-effect may already
                # have applied — treat as outcome_unknown, not a retryable error.
                if self._is_cancelled():
                    raise
                return {
                    "status": "error",
                    "error": (
                        f"{name} did not complete: an underlying connection was torn down. "
                        "Outcome is unknown — do not retry the same side-effecting call."
                    ),
                    "outcome_unknown": True,
                }
            except Exception as e:
                if self._is_cancelled():
                    raise RuntimeError("Cancelled") from e
                return {"status": "error", "error": str(e)}
        
        # Fall back to general tool handler. tool_call_id rides along so a
        # tool that parks awaiting external input (ask_human) can be woken by
        # the journal identity its answer will pair with.
        try:
            return await self._await_with_cancellation(
                self.tool_handler({"tool": name, "args": args, "tool_call_id": call_id}),
                f"tool_handler:{name}",
            )
        except asyncio.CancelledError:
            if self._is_cancelled():
                raise
            return {
                "status": "error",
                "error": (
                    f"{name} did not complete: an underlying connection was torn down. "
                    "Outcome is unknown — do not retry the same side-effecting call."
                ),
                "outcome_unknown": True,
            }
        except Exception as e:
            if self._is_cancelled():
                raise RuntimeError("Cancelled") from e
            return {"status": "error", "error": str(e)}

    async def _dispatch_run_stream_event(
        self,
        event: Dict[str, Any],
        round_num: int,
        st: _RunRoundMutable,
        result: StreamingRunnerResult,
    ) -> None:
        self._ensure_not_cancelled("stream_response")
        event_type = event.get("type", "")

        if event_type == "thinking.delta":
            st.current_thinking.content += event.get("content", "")
            await self._emit_event({
                "type": "thinking.delta",
                "content": event.get("content", ""),
                "round": round_num + 1,
            })

        elif event_type == "thinking.done":
            st.current_thinking.thinking_time = event.get("thinking_time", 0)
            usage = event.get("usage")
            if st.current_thinking.content:
                # Diagnostic: pair this with [THINKING.DONE.EMIT] log
                # lines in llm/client.py to trace duplicate emissions
                # (observed as side-by-side "Human Expert thought"
                # blocks). If two of these fire in one round, the LLM
                # client yielded thinking.done twice — see paired logs.
                logger.info(
                    "[THINKING.DONE.FORWARD] round=%d chars=%d "
                    "elapsed=%.3fs has_usage=%s",
                    round_num + 1,
                    len(st.current_thinking.content),
                    st.current_thinking.thinking_time,
                    bool(usage),
                )
                result.thinking_blocks.append(st.current_thinking)
                emit_data = {
                    "type": "thinking.done",
                    "content": st.current_thinking.content,
                    "thinking_time": st.current_thinking.thinking_time,
                    "round": round_num + 1,
                }
                if usage:
                    emit_data["usage"] = usage
                await self._emit_event(emit_data)
            else:
                logger.info(
                    "[THINKING.DONE.SKIP] round=%d reason=empty_content",
                    round_num + 1,
                )
            st.current_thinking = ThinkingBlock()

        elif event_type == "tool_call.start":
            st.has_tool_calls = True
            call_id = event.get("call_id", "")
            name = event.get("name", "")
            item_id = event.get("item_id") or call_id
            st.pending_tool_calls[call_id] = ToolCallBlock(
                call_id=call_id,
                item_id=item_id,
                name=name,
                status="pending",
            )
            await self._emit_event({
                "type": "tool_call.start",
                "call_id": call_id,
                "name": name,
                "round": round_num + 1,
            })

        elif event_type == "tool_call.delta":
            call_id = event.get("call_id", "")
            if call_id in st.pending_tool_calls:
                st.pending_tool_calls[call_id].arguments += event.get("arguments_delta", "")

        elif event_type == "tool_call.done":
            call_id = event.get("call_id", "")
            if call_id in st.pending_tool_calls:
                tc = st.pending_tool_calls[call_id]
                tc.arguments = event.get("arguments", tc.arguments)
                tc.status = "executing"

                await self._emit_event({
                    "type": "tool_call.executing",
                    "call_id": call_id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "round": round_num + 1,
                })

        elif event_type == "text.delta":
            delta = event.get("content", "")
            st.text_content += delta
            await self._emit_event({
                "type": "text.delta",
                "content": delta,
                "round": round_num + 1,
            })

        elif event_type == "text.done":
            result.content = st.text_content
            await self._emit_event({
                "type": "text.done",
                "content": st.text_content,
                "round": round_num + 1,
            })

        elif event_type == "response.done":
            usage = event.get("usage")
            st.usage = usage or st.usage
            st.stop_reason = event.get("stop_reason") or st.stop_reason
            if usage:
                await self._emit_event({
                    "type": "usage",
                    "usage": usage,
                    "round": round_num + 1,
                })

        elif event_type == "error":
            err = str(event.get("error") or "Unknown error")
            logger.error("[STREAMING] Error: %s", err)
            st.stream_error = err
            err_type = event.get("error_type")
            st.stream_error_type = str(err_type) if err_type else None
            st.stop_reason = "error"
            await self._emit_event({
                "type": "error",
                "error": err,
            })

    async def run(
        self,
        input_items: List[Dict[str, Any]],
        api_key_override: Optional[str] = None,
        fallback_models_override: Optional[List[str]] = None,
    ) -> StreamingRunnerResult:
        """
        Run the agent with streaming, handling tool calls inline.
        
        Args:
            input_items: Initial input items (user messages, etc.)
            api_key_override: Optional API key override
            
        Returns:
            StreamingRunnerResult with content, thinking blocks, and tool calls
        """
        result = StreamingRunnerResult(input_items=list(input_items))
        current_input = list(input_items)
        completed_cleanly = False

        for round_num in range(self.max_tool_rounds):
            self._ensure_not_cancelled("streaming_round")
            logger.info("[STREAMING] Round %d/%d", round_num + 1, self.max_tool_rounds)

            if self.plugin_host is not None:
                current_input = await self.plugin_host.dispatch_pre_send(
                    current_input, turn=round_num
                )

            overflow_retried = False
            provider_retries = 0
            while True:
                content_snap = result.content
                thinking_snap = len(result.thinking_blocks)
                st = _RunRoundMutable()
                stream_timed_out = False
                stream_kwargs = {
                    "input_items": current_input,
                    "model": self.model,
                    "tools": self.tools if self.tools else None,
                    "reasoning_effort": self.reasoning_effort,
                    "api_key_override": api_key_override,
                    "fallback_models_override": fallback_models_override,
                    **({"temperature": self.temperature} if self.temperature is not None else {}),
                }
                capture = self._begin_capture(round_num, stream_kwargs)
                raw_stream = self.llm_client.stream_responses(**stream_kwargs)
                stream = self._tee_stream(raw_stream, capture) if capture is not None else raw_stream
                tr = self.stream_round_timeout_seconds
                capture_error: Optional[str] = None
                overflow_recovery: Optional[List[Dict[str, Any]]] = None
                try:
                    if tr is not None and tr > 0:
                        try:
                            async with asyncio.timeout(tr):
                                async for event in stream:
                                    await self._dispatch_run_stream_event(event, round_num, st, result)
                        except TimeoutError:
                            stream_timed_out = True
                            capture_error = "LLM stream round timed out"
                            logger.warning(
                                "[STREAMING] Round %d stream timed out after %ss",
                                round_num + 1,
                                tr,
                            )
                            await self._emit_event({
                                "type": "error",
                                "error": "LLM stream round timed out",
                                "round": round_num + 1,
                            })
                            # Retry only when no tool was finalized for execution.
                            if not self._round_has_finalized_tools(st):
                                st.stream_error = st.stream_error or capture_error
                                st.stop_reason = st.stop_reason or "error"
                    else:
                        async for event in stream:
                            await self._dispatch_run_stream_event(event, round_num, st, result)
                except ContextOverflowError as exc:
                    capture_error = str(exc)
                    if self.plugin_host is None or overflow_retried:
                        raise
                    overflow_recovery = await self.plugin_host.dispatch_overflow(
                        error=exc, input_items=current_input, turn=round_num
                    )
                    if overflow_recovery is None:
                        raise
                except Exception as exc:
                    capture_error = str(exc)
                    can_retry = (
                        is_transient_provider_error(exc)
                        and provider_retries < _MAX_ROUND_PROVIDER_RETRIES
                        and not self._round_has_finalized_tools(st)
                    )
                    if can_retry:
                        provider_retries += 1
                        backoff = _ROUND_PROVIDER_RETRY_BACKOFF_S[provider_retries - 1]
                        result.content = content_snap
                        del result.thinking_blocks[thinking_snap:]
                        await self._emit_round_provider_retry(
                            round_num=round_num,
                            attempt=provider_retries,
                            error=str(exc),
                            backoff=backoff,
                        )
                        await self._cancelable_sleep(backoff)
                        continue
                    if (
                        is_transient_provider_error(exc)
                        and not self._round_has_finalized_tools(st)
                    ):
                        result.content = content_snap
                        del result.thinking_blocks[thinking_snap:]
                        result.error = str(exc)
                        result.stop_reason = "error"
                        return result
                    raise
                finally:
                    # CancelledError / BaseException must finalize too (parity with
                    # run_streaming); do not leave agent_llm_calls open on Stop.
                    await self._finalize_capture(capture, error=capture_error)

                if overflow_recovery is not None:
                    # One recovery resend per Turn: a second overflow on the
                    # recovered input means the transformation didn't help —
                    # propagate instead of looping.
                    overflow_retried = True
                    current_input = overflow_recovery
                    continue

                # AppFactory-316: stream error/timeout before any tool of this round ran.
                if st.stream_error and not self._round_has_finalized_tools(st):
                    if (
                        is_transient_provider_error(
                            st.stream_error, error_type=st.stream_error_type
                        )
                        and provider_retries < _MAX_ROUND_PROVIDER_RETRIES
                    ):
                        provider_retries += 1
                        backoff = _ROUND_PROVIDER_RETRY_BACKOFF_S[provider_retries - 1]
                        result.content = content_snap
                        del result.thinking_blocks[thinking_snap:]
                        await self._emit_round_provider_retry(
                            round_num=round_num,
                            attempt=provider_retries,
                            error=st.stream_error,
                            backoff=backoff,
                        )
                        await self._cancelable_sleep(backoff)
                        continue
                    result.content = content_snap
                    del result.thinking_blocks[thinking_snap:]
                    result.error = st.stream_error
                    result.stop_reason = "error"
                    return result

                break

            if self.plugin_host is not None:
                await self.plugin_host.dispatch_post_turn(
                    turn=round_num, usage=st.usage, stop_reason=st.stop_reason
                )

            # Last round drained wins: on the terminal (no-tool) round this is the
            # generation's real stop_reason; the caller reads it to reject truncation.
            result.stop_reason = st.stop_reason
            # Even when we must not retry after tool_call.done, keep provider text
            # for FAILED (AppFactory-316 contract 4). Clear on a later non-error round.
            if st.stream_error:
                result.error = st.stream_error
            elif st.stop_reason != "error":
                result.error = None

            pending_tool_calls = st.pending_tool_calls
            has_tool_calls = st.has_tool_calls

            # If no tool calls, we're done
            if not has_tool_calls:
                logger.info("[STREAMING] No tool calls, generation complete")
                completed_cleanly = True
                break

            # Execute tool calls and prepare for next round
            tool_results = []
            for call_id, tc in pending_tool_calls.items():
                self._ensure_not_cancelled(f"tool:{tc.name}")
                if tc.status != "executing":
                    logger.warning(
                        "[STREAMING] Skipping tool call_id=%s name=%s — stream ended before tool_call.done",
                        call_id,
                        tc.name,
                    )
                    await self._emit_event({
                        "type": "tool_call.error",
                        "call_id": call_id,
                        "name": tc.name,
                        "error": "LLM stream ended before tool arguments were finalized",
                        "round": round_num + 1,
                    })
                    continue
                logger.info("[STREAMING] Executing tool: %s", tc.name)

                tc.result = await self._await_with_cancellation(
                    self._execute_tool(tc.name, tc.arguments, call_id),
                    f"tool:{tc.name}",
                )
                tc.status = (
                    "error"
                    if isinstance(tc.result, dict) and tc.result.get("status") == "error"
                    else "completed"
                )
                result.tool_calls.append(tc)
                
                await self._emit_event({
                    "type": "tool_call.result",
                    "call_id": call_id,
                    "name": tc.name,
                    "result": self._journal_safe_tool_result(tc.result),
                    "status": tc.status,
                    "round": round_num + 1,
                })
                
                # Add function_call (what model asked for) to conversation.
                # Use the original item_id from the response as the "id" field —
                # some providers validate that it matches the issued response item.
                current_input.append({
                    "type": "function_call",
                    "id": tc.item_id,
                    "call_id": call_id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                })
                
                # Add function_call_output (our result) to conversation
                tool_results.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": self._format_tool_output(tc.result),
                })
                tool_results.extend(self._vision_followups_from_tool_result(tc.result))
            
            # Append all tool results to input for next round
            # (full conversation accumulation - OpenRouter doesn't support previous_response_id)
            current_input.extend(tool_results)
            result.input_items = current_input

            if stream_timed_out and not tool_results and has_tool_calls:
                logger.warning(
                    "[STREAMING] Round %d timed out with only incomplete tool calls; stopping tool rounds",
                    round_num + 1,
                )
                break

        # Parity with SimpleAgentRunner max_steps: exhausted tool rounds without a
        # clean no-tool terminal generation is truncation, not success.
        if not completed_cleanly and result.stop_reason != "error":
            result.stop_reason = "length"
            if not (result.error or "").strip():
                result.error = (
                    "StreamingAgentRunner exceeded max_tool_rounds without final content"
                )

        return result
    
    async def continue_run(
        self,
        previous: StreamingRunnerResult,
        message: str,
        **run_kwargs: Any,
    ) -> StreamingRunnerResult:
        return await self.run(
            [
                *previous.input_items,
                {"type": "message", "role": "assistant", "content": previous.content},
                {"type": "message", "role": "user", "content": message},
            ],
            **run_kwargs,
        )

    async def run_streaming(
        self,
        input_items: List[Dict[str, Any]],
        api_key_override: Optional[str] = None,
        fallback_models_override: Optional[List[str]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run with streaming, yielding events as they occur.
        
        This is useful when you want to process events directly without
        waiting for the full result.
        """
        current_input = list(input_items)
        
        for round_num in range(self.max_tool_rounds):
            self._ensure_not_cancelled("streaming_round")

            if self.plugin_host is not None:
                current_input = await self.plugin_host.dispatch_pre_send(
                    current_input, turn=round_num
                )

            overflow_retried = False
            while True:
                pending_tool_calls: Dict[str, ToolCallBlock] = {}
                has_tool_calls = False
                stream_timed_out = False
                round_usage: Optional[Dict[str, Any]] = None
                round_stop_reason: Optional[str] = None

                stream_kwargs = {
                    "input_items": current_input,
                    "model": self.model,
                    "tools": self.tools if self.tools else None,
                    "reasoning_effort": self.reasoning_effort,
                    "api_key_override": api_key_override,
                    "fallback_models_override": fallback_models_override,
                    **({"temperature": self.temperature} if self.temperature is not None else {}),
                }
                capture = self._begin_capture(round_num, stream_kwargs)
                raw_stream = self.llm_client.stream_responses(**stream_kwargs)
                stream = self._tee_stream(raw_stream, capture) if capture is not None else raw_stream
                tr = self.stream_round_timeout_seconds

                async def _pump_stream():
                    nonlocal has_tool_calls, round_usage, round_stop_reason
                    async for event in stream:
                        self._ensure_not_cancelled("stream_response")
                        event["round"] = round_num + 1
                        yield event
                        event_type = event.get("type", "")
                        if event_type == "tool_call.start":
                            has_tool_calls = True
                            call_id = event.get("call_id", "")
                            item_id = event.get("item_id") or call_id
                            pending_tool_calls[call_id] = ToolCallBlock(
                                call_id=call_id,
                                item_id=item_id,
                                name=event.get("name", ""),
                                status="pending",
                            )
                        elif event_type == "tool_call.delta":
                            call_id = event.get("call_id", "")
                            if call_id in pending_tool_calls:
                                pending_tool_calls[call_id].arguments += event.get("arguments_delta", "")
                        elif event_type == "tool_call.done":
                            call_id = event.get("call_id", "")
                            if call_id in pending_tool_calls:
                                tc_done = pending_tool_calls[call_id]
                                tc_done.arguments = event.get("arguments", "")
                                tc_done.status = "executing"
                        elif event_type == "response.done":
                            round_usage = event.get("usage") or round_usage
                            round_stop_reason = event.get("stop_reason") or round_stop_reason

                capture_error: Optional[str] = None
                overflow_recovery: Optional[List[Dict[str, Any]]] = None
                try:
                    if tr is not None and tr > 0:
                        try:
                            async with asyncio.timeout(tr):
                                async for _out in _pump_stream():
                                    yield _out
                        except TimeoutError:
                            stream_timed_out = True
                            capture_error = "LLM stream round timed out"
                            yield {
                                "type": "error",
                                "error": "LLM stream round timed out",
                                "round": round_num + 1,
                            }
                    else:
                        async for _out in _pump_stream():
                            yield _out
                except ContextOverflowError as exc:
                    capture_error = str(exc)
                    if self.plugin_host is None or overflow_retried:
                        raise
                    overflow_recovery = await self.plugin_host.dispatch_overflow(
                        error=exc, input_items=current_input, turn=round_num
                    )
                    if overflow_recovery is None:
                        raise
                except Exception as exc:
                    capture_error = str(exc)
                    raise
                finally:
                    await self._finalize_capture(capture, error=capture_error)
                if overflow_recovery is not None:
                    # One recovery resend per Turn — mirrors run().
                    overflow_retried = True
                    current_input = overflow_recovery
                    continue
                break

            if self.plugin_host is not None:
                await self.plugin_host.dispatch_post_turn(
                    turn=round_num, usage=round_usage, stop_reason=round_stop_reason
                )

            if not has_tool_calls:
                break

            # Execute tools and continue with full conversation accumulation
            tool_results = []
            for call_id, tc in pending_tool_calls.items():
                self._ensure_not_cancelled(f"tool:{tc.name}")
                if tc.status != "executing":
                    logger.warning(
                        "[STREAMING] run_streaming skip call_id=%s name=%s — no tool_call.done",
                        call_id,
                        tc.name,
                    )
                    yield {
                        "type": "tool_call.error",
                        "call_id": call_id,
                        "name": tc.name,
                        "error": "LLM stream ended before tool arguments were finalized",
                        "round": round_num + 1,
                    }
                    continue
                yield {"type": "tool_call.executing", "call_id": call_id, "name": tc.name, "round": round_num + 1}

                result = await self._await_with_cancellation(
                    self._execute_tool(tc.name, tc.arguments, call_id),
                    f"tool:{tc.name}",
                )
                
                yield {
                    "type": "tool_call.result",
                    "call_id": call_id,
                    "name": tc.name,
                    "result": self._journal_safe_tool_result(result),
                    "round": round_num + 1,
                }
                
                # Add function_call (what model asked for) to conversation
                current_input.append({
                    "type": "function_call",
                    "id": tc.item_id,
                    "call_id": call_id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                })
                
                # Add function_call_output (our result) to conversation
                tool_results.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": self._format_tool_output(result),
                })
                tool_results.extend(self._vision_followups_from_tool_result(result))
            
            # Append all tool results (OpenRouter doesn't support previous_response_id)
            current_input.extend(tool_results)

            if stream_timed_out and not tool_results and has_tool_calls:
                logger.warning(
                    "[STREAMING] run_streaming round %d timed out with only incomplete tool calls; stopping",
                    round_num + 1,
                )
                break
