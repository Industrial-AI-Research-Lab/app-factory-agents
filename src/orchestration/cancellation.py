from __future__ import annotations

import asyncio
from typing import Any


def is_genuinely_cancelled(cancellation_token: Any) -> bool:
    """True when the run is really being cancelled — the discriminator a
    ``except CancelledError`` guard must consult before swallowing.

    Two independent signals count as genuine:

    1. The project cancellation token is set (a user/system cancel).
    2. A real asyncio cancellation is in flight on the current Task
       (``current_task().cancelling() > 0``).

    Signal 2 is the one the token-only check missed. ``WorkflowEngine``
    enforces ``run_timeout_seconds`` with ``asyncio.wait_for`` (see
    ``workflow_engine.py``), which cancels the node Task **without** setting
    the token. Without this check a hard timeout is indistinguishable from a
    *spurious* cancel — a torn-down connection unwinding its cancel scope on
    the wrong Task — so the guard swallows it, the node runs to completion,
    and the timeout silently never fires. ``cancelling()`` is 0 for the
    spurious case (nobody called *our* Task's ``cancel()``) and >0 for a real
    cancel, which is exactly the distinction we need.

    ``Task.cancelling()`` is Python 3.11+; on older runtimes this degrades to
    the token-only behaviour. ``current_task()`` needs a running loop, so the
    RuntimeError from a rare sync-context call degrades the same way.
    """
    if cancellation_token and getattr(cancellation_token, "is_cancelled", lambda: False)():
        return True
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return False
    if task is None:
        return False
    return getattr(task, "cancelling", lambda: 0)() > 0


class CancellationToken:
    """Per-project cancellation token.
    - cancel(): triggers cancellation
    - is_cancelled: returns boolean
    - wait(): await until token is cancelled
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()
