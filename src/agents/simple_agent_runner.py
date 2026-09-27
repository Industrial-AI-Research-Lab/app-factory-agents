from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from agents.vision_tool_result import (
    format_tool_output,
    scrub_for_capture,
    vision_chat_followups_from_tool_result,
)
from llm.async_buffer import AsyncBuffer
from agents.tool_failure_budget import ToolFailureBudget

logger = logging.getLogger(__name__)

ToolHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]

_STRAY_CANCEL = (
    "{name} did not complete: an underlying connection was torn down. "
    "Outcome is unknown — do not retry the same side-effecting call."
)


@dataclass
class RunnerResult:
    content: str
    messages: List[Dict[str, Any]]
    # same facts streaming exposes (for terminal gate parity).
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: Optional[str] = None
    error: Optional[str] = None


class SimpleAgentRunner:
    def __init__(
        self,
        llm_client,
        tools: Sequence[Dict[str, Any]],
        tool_handler: ToolHandler,
        local_tools: Optional[Dict[str, ToolHandler]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.2,
        max_steps: int = 30,
        reasoning_effort: Optional[str] = None,
        buffer: Optional[AsyncBuffer] = None,
        cancellation_token=None,
        # Inspector capture context — optional; no-op if any of
        # project_id / agent_id / agent_llm_calls_store is None.
        project_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        agent_llm_calls_store=None,
        event_emitter=None,
        # Accepted so _capture_kwargs_for_runner stays runner-agnostic; the
        # simple runner keeps no tool ledger (streaming-only), so unused.
        workflow_node_id: Optional[str] = None,
        message_store=None,
        max_tool_failures: Optional[int] = None,
    ):
        self.llm_client = llm_client
        self.tools = list(tools)
        self.tool_handler = tool_handler
        self.local_tools = dict(local_tools or {})
        self.model = model
        self.temperature = None if temperature is None else float(temperature)
        self.max_steps = int(max_steps)
        self.reasoning_effort = reasoning_effort
        self.buffer = buffer or AsyncBuffer()
        self.cancellation_token = cancellation_token
        self.project_id = project_id
        self.run_id = run_id
        self.agent_id = agent_id
        self.task_id = task_id
        self.agent_llm_calls_store = agent_llm_calls_store
        self.event_emitter = event_emitter
        self.tool_failure_budget = ToolFailureBudget(max_tool_failures)

    def _is_cancelled(self) -> bool:
        # Token set OR a real Task-level cancel in flight (run_timeout_seconds
        # cancels via asyncio.wait_for without setting the token); a spurious
        # scope-leak cancel reads False so the guards below still contain it.
        # Imported lazily: orchestration/__init__ is heavy and cycles back here.
        from orchestration.cancellation import is_genuinely_cancelled
        return is_genuinely_cancelled(self.cancellation_token)

    def _ensure_not_cancelled(self, location: str) -> None:
        if self._is_cancelled():
            raise RuntimeError("Cancelled")

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
        kwargs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self._capture_enabled():
            return None
        return {
            "call_id": str(uuid.uuid4()),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "round_num": round_num,
            "kwargs": kwargs,
        }

    @staticmethod
    def _split_system_messages(
        messages: Optional[List[Dict[str, Any]]],
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        if not messages:
            return None, []
        system_parts: List[str] = []
        rest: List[Dict[str, Any]] = []
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
                content = m.get("content", "")
                if isinstance(content, str):
                    system_parts.append(content)
                else:
                    try:
                        system_parts.append(json.dumps(content, default=str))
                    except Exception:
                        system_parts.append(str(content))
            else:
                rest.append(m)
        return ("\n\n".join(system_parts) if system_parts else None), rest

    @staticmethod
    def _tool_calls_to_dicts(tool_calls: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for tc in tool_calls or []:
            if isinstance(tc, dict):
                out.append(tc)
                continue
            try:
                fn = getattr(tc, "function", None)
                out.append({
                    "id": getattr(tc, "id", None),
                    "type": getattr(tc, "type", "function"),
                    "name": getattr(fn, "name", None) if fn else None,
                    "input": getattr(fn, "arguments", None) if fn else None,
                })
            except Exception:
                out.append({"_raw_repr": repr(tc)})
        return out

    async def _finalize_capture(
        self,
        capture: Optional[Dict[str, Any]],
        response: Any = None,
        error: Optional[str] = None,
    ) -> None:
        if capture is None:
            return
        try:
            kwargs = capture.get("kwargs", {})
            system, messages = self._split_system_messages(kwargs.get("messages"))
            resp_dict = response if isinstance(response, dict) else {}
            tool_uses = self._tool_calls_to_dicts(resp_dict.get("tool_calls"))
            # chat_completion returns the field as "finish_reason"; keep "stop_reason"
            # as a fallback for clients that use the Anthropic-style key.
            stop_reason = resp_dict.get("stop_reason") or resp_dict.get("finish_reason")
            if error:
                stop_reason = "error"
            elif stop_reason is None:
                stop_reason = "tool_use" if tool_uses else "end_turn"
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
                    "messages": scrub_for_capture(messages),
                    "tools": kwargs.get("tools"),
                    "temperature": kwargs.get("temperature"),
                    "max_tokens": kwargs.get("max_tokens"),
                    "params": {
                        "tool_choice": kwargs.get("tool_choice"),
                    },
                },
                "response": {
                    "text": resp_dict.get("content") or "",
                    "thinking": resp_dict.get("reasoning") or "",
                    "tool_uses": tool_uses,
                    "stop_reason": stop_reason,
                    "usage": resp_dict.get("usage"),
                    "error": error,
                    "response_id": resp_dict.get("response_id"),
                },
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
                            "token_counts": resp_dict.get("usage"),
                            "stop_reason": stop_reason,
                            "truncated": doc.get("truncated", False),
                        },
                    )
                except Exception as exc:
                    logger.warning("agent.invocation.captured emit failed: %s", exc)
        except Exception as exc:
            logger.warning("agent_llm_calls capture finalize failed: %s", exc)

    async def run(self, messages: List[Dict[str, Any]]) -> RunnerResult:
        msgs: List[Dict[str, Any]] = list(messages)
        tool_results: List[Dict[str, Any]] = []

        for round_num in range(self.max_steps):
            msgs = await self._await_with_cancellation(
                self.buffer.maybe_compress(self.llm_client, msgs, model=self.model),
                "buffer_compress",
            )

            chat_kwargs = {
                "messages": msgs,
                "model": self.model,
                "temperature": self.temperature,
                "tools": self.tools,
                "tool_choice": "auto",
            }

            if self.reasoning_effort is not None:
                chat_kwargs["reasoning_effort"] = self.reasoning_effort

            capture = self._begin_capture(round_num, chat_kwargs)
            capture_error: Optional[str] = None
            resp = None
            try:
                resp = await self._await_with_cancellation(
                    self.llm_client.chat_completion(**chat_kwargs),
                    "chat_completion",
                )
            except Exception as exc:
                capture_error = str(exc)
                raise
            finally:
                await self._finalize_capture(capture, response=resp, error=capture_error)

            tool_calls = (resp or {}).get("tool_calls") or []
            content = (resp or {}).get("content") or ""
            # Do not invent "stop" when the provider omitted finish_reason.
            stop_reason = (resp or {}).get("finish_reason") or (resp or {}).get(
                "stop_reason"
            )

            if not tool_calls:
                return RunnerResult(
                    content=content,
                    messages=msgs + [{"role": "assistant", "content": content}],
                    tool_results=tool_results,
                    stop_reason=stop_reason,
                )

            msgs.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

            batches = _split_parallelizable(tool_calls)
            for batch in batches:
                self._ensure_not_cancelled("tool_batch")
                results = await self._execute_batch(batch)
                for tc, tool_result in results:
                    try:
                        tool_args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        tool_args = {}
                    tool_results.append(
                        {
                            "name": tc.function.name,
                            "call_id": tc.id,
                            "args": tool_args if isinstance(tool_args, dict) else {},
                            "result": tool_result,
                        }
                    )
                    msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": format_tool_output(tool_result),
                        }
                    )
                    msgs.extend(vision_chat_followups_from_tool_result(tool_result))

        return RunnerResult(
            content="",
            messages=msgs,
            tool_results=tool_results,
            stop_reason="length",
            error="SimpleAgentRunner exceeded max_steps without final content",
        )

    async def continue_run(self, previous: RunnerResult, message: str) -> RunnerResult:
        return await self.run(
            [*previous.messages, {"role": "user", "content": message}]
        )

    async def _execute_batch(self, batch: List[Any]) -> List[Tuple[Any, Dict[str, Any]]]:
        async def _run_one(tc: Any) -> Tuple[Any, Dict[str, Any]]:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}

            self._ensure_not_cancelled(f"tool:{name}")
            handler = self.local_tools.get(name)
            if handler is not None:
                try:
                    return tc, await self._await_with_cancellation(handler(args), f"local_tool:{name}")
                except asyncio.CancelledError:
                    # Not an Exception, so the handler below misses it. A failed
                    # MCP connection raises one without our run being cancelled;
                    # uncaught it kills the call silently — no result, no log.
                    # After a live session the side-effect may already apply.
                    if self._is_cancelled():
                        raise
                    from orchestration.workflow_task_lifecycle import raise_if_tool_outcome_unknown

                    raise_if_tool_outcome_unknown(
                        {
                            "status": "error",
                            "error": _STRAY_CANCEL.format(name=name),
                            "outcome_unknown": True,
                        },
                        name,
                        tc.id,
                    )
                except Exception as e:
                    if self._is_cancelled():
                        raise RuntimeError("Cancelled") from e
                    return tc, {"status": "error", "error": str(e)}

            try:
                # tool_call_id rides along for the dispatcher's consumers
                # (archive provenance, ask_human wake key) — same contract as
                # the streaming runner; omitting it stored null provenance.
                result = await self._await_with_cancellation(
                    self.tool_handler({"tool": name, "args": args, "tool_call_id": tc.id}),
                    f"tool_handler:{name}",
                )
                from orchestration.workflow_task_lifecycle import (
                    raise_if_session_unavailable,
                    raise_if_tool_outcome_unknown,
                )

                raise_if_session_unavailable(result, name, tc.id)
                raise_if_tool_outcome_unknown(result, name, tc.id)
                return tc, result
            except asyncio.CancelledError:
                if self._is_cancelled():
                    raise
                from orchestration.workflow_task_lifecycle import raise_if_tool_outcome_unknown

                raise_if_tool_outcome_unknown(
                    {
                        "status": "error",
                        "error": _STRAY_CANCEL.format(name=name),
                        "outcome_unknown": True,
                    },
                    name,
                    tc.id,
                )
            except Exception as e:
                from orchestration.workflow_task_lifecycle import (
                    ExternalToolOutcomeUnknownError,
                    SessionUnavailableStopError,
                )

                if isinstance(
                    e, (ExternalToolOutcomeUnknownError, SessionUnavailableStopError)
                ):
                    raise
                if self._is_cancelled():
                    raise RuntimeError("Cancelled") from e
                return tc, {"status": "error", "error": str(e)}

        if self.tool_failure_budget.limit is not None:
            results = []
            for tc in batch:
                self.tool_failure_budget.check()
                pair = await _run_one(tc)
                self.tool_failure_budget.record(tc.function.name, pair[1])
                results.append(pair)
            return results
        return await asyncio.gather(*[_run_one(tc) for tc in batch])


def _split_parallelizable(tool_calls: Sequence[Any]) -> List[List[Any]]:
    read_only = {
        "read",
        "grep",
        "glob",
        "todo_read",
        "web_search",
        "task",
    }

    batches: List[List[Any]] = []
    current: List[Any] = []
    current_parallel = True

    for tc in tool_calls:
        name = getattr(getattr(tc, "function", None), "name", "")
        is_ro = name in read_only
        if not current:
            current = [tc]
            current_parallel = is_ro
            continue
        if is_ro and current_parallel:
            current.append(tc)
            continue
        batches.append(current)
        current = [tc]
        current_parallel = is_ro

    if current:
        batches.append(current)
    return batches
