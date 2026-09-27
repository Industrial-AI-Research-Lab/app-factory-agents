"""Workflow asyncio.Task outcome classification and child-task cancellation."""

from __future__ import annotations

import asyncio
import logging
import weakref
from enum import Enum
from typing import Iterable

logger = logging.getLogger(__name__)

# Tasks cancelled via cancel_project / shutdown — not unexpected cancel.
_explicit_cancel_tasks: weakref.WeakSet[asyncio.Task] = weakref.WeakSet()


def mark_workflow_task_explicit_cancel(task: asyncio.Task) -> None:
    """Record intentional cancellation before task.cancel()."""
    _explicit_cancel_tasks.add(task)


def is_workflow_task_explicit_cancel(task: asyncio.Task) -> bool:
    return task in _explicit_cancel_tasks

UNEXPECTED_CANCEL_REASON = "workflow_task_cancelled_unexpectedly"
ABNORMAL_TERMINAL_RESUME_BLOCKED = "abnormal_workflow_terminal"
EXTERNAL_TOOL_OUTCOME_UNKNOWN = "external_tool_outcome_unknown"
SESSION_UNAVAILABLE_STOP = "session_unavailable"
TOOL_INTERRUPTED_BEFORE_RESULT = "tool interrupted before result was recorded"


class ExternalToolOutcomeUnknownError(Exception):
    """Side-effecting tool may have run; live execution must not retry."""

    def __init__(self, tool_name: str = "", call_id: str = ""):
        self.tool_name = tool_name
        self.call_id = call_id
        super().__init__(EXTERNAL_TOOL_OUTCOME_UNKNOWN)


class SessionUnavailableStopError(Exception):
    """Sandbox session locked out; attempt stops and may resume after /recover."""

    def __init__(self, tool_name: str = "", call_id: str = ""):
        self.tool_name = tool_name
        self.call_id = call_id
        super().__init__(SESSION_UNAVAILABLE_STOP)


class ResumeBlockedLookupError(Exception):
    """Active run could not be read to verify resume safety."""


def raise_if_tool_outcome_unknown(
    result: dict,
    tool_name: str,
    call_id: str = "",
) -> None:
    """Fail closed when a tool result marks its side-effect outcome unknown."""
    if isinstance(result, dict) and result.get("outcome_unknown"):
        raise ExternalToolOutcomeUnknownError(tool_name, call_id)


def raise_if_session_unavailable(
    result: dict,
    tool_name: str,
    call_id: str = "",
) -> None:
    """Stop the agent loop when the sandbox session is locked out (resumable)."""
    if not isinstance(result, dict):
        return
    from schemas.infra_error import InfraErrorType

    code = result.get("code") or result.get("error_type")
    if code == InfraErrorType.SESSION_UNAVAILABLE.value:
        raise SessionUnavailableStopError(tool_name, call_id)
    if result.get("session_state") == "unavailable":
        raise SessionUnavailableStopError(tool_name, call_id)


class WorkflowTaskOutcome(str, Enum):
    SUCCESS = "success"
    EXPLICIT_CANCEL = "explicit_cancel"
    UNEXPECTED_CANCEL = "unexpected_cancel"
    EXCEPTION = "exception"


def classify_workflow_task_outcome(
    task: asyncio.Task,
    *,
    project_cancelled: bool,
) -> WorkflowTaskOutcome:
    """Classify a finished workflow runtime task for terminal persistence."""
    if task.cancelled():
        if project_cancelled or is_workflow_task_explicit_cancel(task):
            return WorkflowTaskOutcome.EXPLICIT_CANCEL
        return WorkflowTaskOutcome.UNEXPECTED_CANCEL
    exc = task.exception()
    if exc is not None:
        return WorkflowTaskOutcome.EXCEPTION
    return WorkflowTaskOutcome.SUCCESS


async def cancel_and_await(
    tasks: Iterable[asyncio.Task],
    *,
    label: str = "child",
) -> None:
    """Cancel child tasks and await completion before the parent returns."""
    pending = [t for t in tasks if t is not None and not t.done()]
    for task in pending:
        task.cancel()
    for i, task in enumerate(pending):
        try:
            await task
        except asyncio.CancelledError:
            if asyncio.current_task() is not None and asyncio.current_task().cancelling():
                # Drain remaining tasks before propagating so they don't leak.
                remaining = [t for t in pending[i + 1:] if not t.done()]
                if remaining:
                    await asyncio.gather(*remaining, return_exceptions=True)
                raise
        except Exception as exc:
            logger.warning(
                "[WORKFLOW_TASK] %s task=%s raised during cancel-and-await: %s",
                label,
                getattr(task, "get_name", lambda: "child")(),
                exc,
            )
