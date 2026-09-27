"""Ask a human: durable pause on the tool journal (ADR-0010).

The question IS the hanging ``tool_call`` record the runner persists before
dispatch (ADR-0008) — this module adds only the in-process wait and the
wake-up, no state machine and no approval row. Live path: the handler parks
on a Future until the answer route resolves it, then returns the answer as
an ordinary tool result, which the runner persists (closing the pair).
After a backend restart the Future is gone: the hanging record parks the
project (ADR-0009), and the answer route closes the pair in the journal
itself, so nothing here needs to survive the process.

The record becomes answerable at persist time, but the waiter exists only
once dispatch reaches this module — an answer landing in that gap finds no
waiter and is written to the journal as if the backend had restarted. Two
moves close that race: parking re-checks the journal AFTER registering the
waiter, and the route re-attempts a wake AFTER its write; whichever side
acted second sees the other.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from storage.message_store import pair_tool_records

logger = logging.getLogger(__name__)

# One waiter per parked question. Keyed by (project_id, tool_call_id) — the
# same identity the journal pairs records by, and what the answer route
# addresses.
_waiters: Dict[Tuple[str, str], asyncio.Future] = {}

# Questions answered LIVE in this process whose woken runner has not yet
# persisted the result. The answer route consults this on its restart path: a
# live answer already resolved the question (and the agent is acting on it), so
# a second, DIFFERENT answer arriving in the gap before the runner's journal
# write must 409 — not take the restart path and append a rival result that
# positional pairing would canonicalize over the one the agent used
# (split-brain). Keyed like _waiters and cleared by the runner post-persist,
# after which the closed journal pair 409s duplicates on its own.
_answered_live: Dict[Tuple[str, str], str] = {}


def live_answer_claimed(project_id: str, tool_call_id: str) -> bool:
    """True if this question was answered live here and not yet persisted."""
    return (project_id, tool_call_id) in _answered_live


def clear_live_answer_claim(project_id: str, tool_call_id: str) -> None:
    """Release a live claim once its result is journaled (runner, post-persist).

    Kept until then so a rival answer 409s; released so a later question that
    reuses this call id (providers recycle ids, ADR-0008) isn't falsely blocked.
    """
    _answered_live.pop((project_id, tool_call_id), None)


def records_for_run(
    records: List[Dict[str, Any]], run_id: Optional[str]
) -> List[Dict[str, Any]]:
    """Just this run's records.

    Providers reuse call ids across runs (ADR-0008), so an unscoped read mixes
    a past run's question with the current one. Scoping before the pairing rule
    is what stops a stranger's closed pair from standing in for this question.
    A caller with no run matches records stamped with no run, which is what a
    run-less context writes — never everything, since "I don't know which run"
    must not widen the search to other people's answers.
    """
    return [r for r in records if r.get("run_id") == run_id]


def answered_result(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The ``tool_result`` record that closed the LATEST call for this id,
    or None while that call is still open.

    Pair state is ``pair_tool_records``' positional rule — the one resume
    and the feed already use — not a result count: a duplicate result left
    by the double-delivery tail stays an orphan, so it can neither
    pre-answer a reused call id nor 409 its answer route. Providers reuse
    ids across runs too (ADR-0008); the (run_id, id) pair key keeps an
    older run's closed pair from satisfying or blocking a new question.
    Shared by the route's open-check and the parking recheck so all three
    readers agree.
    """
    call_pairs = [p for p in pair_tool_records(records) if p.get("call")]
    if not call_pairs:
        return None
    return call_pairs[-1].get("result")


async def handle_ask_human(
    question: str,
    project_id: str,
    tool_call_id: Optional[str] = None,
    message_store: Optional[Any] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Park until a human answers; return the answer as the tool result.

    Requires a journaled call record for this run: the card renders from that
    record and the answer is filed against it, so a question without one can
    never be answered. Holding an id is not evidence that a record exists —
    some runners keep no journal at all and still pass the provider's id
    through — so this verifies the record rather than trusting the id. The
    returned shape must stay identical to what the answer route writes on
    the restart path: the agent sees the same result either way.
    """
    if not question or not question.strip():
        return {"status": "error", "error": "ask_human: question cannot be empty"}

    if not tool_call_id:
        return {
            "status": "error",
            "error": (
                "ask_human: no tool_call_id — the question needs a tool journal "
                "record to receive its answer (streaming runner only)"
            ),
        }

    key = (project_id, tool_call_id)
    if key in _waiters:
        return {
            "status": "error",
            "error": f"ask_human: a question with call id {tool_call_id} is already waiting",
        }

    future: asyncio.Future = asyncio.get_running_loop().create_future()
    _waiters[key] = future
    # ONE finally owns deregistration for every exit — recovered, refused,
    # answered, or cancelled mid-recheck. A waiter that outlives this
    # coroutine would "successfully" swallow a later answer nobody awaits.
    try:
        # An answer may have landed between the runner persisting the
        # tool_call and this park — the route then closed the pair itself,
        # believing the backend restarted. Register-then-recheck pairs with
        # the route's write-then-wake order: whichever side ran second sees
        # the other, so the answer is never lost to the gap.
        records = None
        read_failed = False
        if message_store is not None:
            try:
                records = await message_store.get_tool_records_for_call(
                    project_id, tool_call_id
                )
            except Exception as e:
                read_failed = True
                logger.warning(
                    "[ASK_HUMAN] project=%s call=%s journal recheck failed: %s",
                    project_id, tool_call_id, e,
                )

        if message_store is None:
            # No journal in this deployment at all, so nothing recorded the
            # question and no card can render — it is unanswerable by
            # construction, and an error the agent can retry beats a park that
            # would hang with nothing able to cancel it.
            return {
                "status": "error",
                "error": (
                    "ask_human: no tool journal in this deployment — the "
                    "question could never be answered"
                ),
            }

        # A failed read is not evidence the record is missing — and it normally
        # is not, since the runner writes it before dispatch and fails loudly.
        # Erroring here would file that error as this question's own ANSWER,
        # closing it: the human's reply is then rejected as already-answered and
        # a real decision is lost to a database hiccup. Parking costs nothing —
        # this check is a shortcut, and the answer route does the delivering.
        if not read_failed:
            # Everything below needs THIS run's records: an id can repeat across
            # runs, so an unscoped list may describe a different question entirely.
            mine = records_for_run(records or [], run_id)

            if not any(r.get("type") == "tool_call" for r in mine):
                # The read worked and holds no call record for this run: nothing
                # journaled the question (disabled ledger, or a runner that keeps
                # no ledger), so no card renders and no answer can arrive.
                return {
                    "status": "error",
                    "error": (
                        "ask_human: the question was not journaled (tool "
                        "ledger disabled) — it could never be answered"
                    ),
                }

            closed = answered_result(mine)
            if closed is not None:
                logger.info(
                    "[ASK_HUMAN] project=%s call=%s answer was already in the journal",
                    project_id, tool_call_id,
                )
                result = (closed.get("data") or {}).get("result")
                if isinstance(result, dict):
                    return result
                return {"status": "success", "answer": result}

        logger.info(
            "[ASK_HUMAN] project=%s call=%s parked: %r",
            project_id, tool_call_id, question[:200],
        )
        answer = await future
    finally:
        _waiters.pop(key, None)

    logger.info(
        "[ASK_HUMAN] project=%s call=%s answered: %r",
        project_id, tool_call_id, str(answer)[:200],
    )
    return {"status": "success", "answer": answer}


def resolve_ask_human(project_id: str, tool_call_id: str, answer: str) -> bool:
    """Wake the parked question, if this process holds it.

    False means no live waiter — the caller (answer route) then owns closing
    the pair in the journal directly (post-restart case).
    """
    key = (project_id, tool_call_id)
    future = _waiters.get(key)
    if future is None or future.done():
        return False
    # Claim BEFORE popping the waiter. This function is synchronous (no await),
    # so a concurrent answer that later finds the waiter already gone is
    # guaranteed to observe this claim and 409, instead of racing a rival write
    # into the gap before the woken runner persists (F3).
    _answered_live[key] = answer
    _waiters.pop(key, None)
    future.set_result(answer)
    return True
