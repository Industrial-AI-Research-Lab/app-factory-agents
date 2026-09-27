"""Restore a run from the tool journal after a backend restart (ADR-0009).

The journal (ADR-0008) is the only durable mid-phase cursor: each tool call
lands as a ``tool_call`` record before execution and a ``tool_result`` after,
stamped with the attempt's ``task_id`` and the workflow node executing it.
This module answers two questions from those records alone:

- which attempt was interrupted (no completion marker), and
- what LLM input items replay it without re-executing anything.

Restore never runs a tool: closed pairs become ``function_call`` /
``function_call_output`` items verbatim, a hanging call is a parking point
(the project waits for its result — e.g. a human's eventual answer), and a
run without usable records degrades to today's fresh re-run.
"""

import logging
from typing import Any, Dict, List, Optional

from storage.message_store import TOOL_LEDGER_TYPES, pair_tool_records
from context.shared_context import TOOL_JOURNAL_CONTEXT_RECORDS

from .streaming_agent_runner import StreamingAgentRunner

logger = logging.getLogger(__name__)

# How many recent NL messages to scan for the attempt's completion marker.
# An assistant final lands immediately after its attempt's records, so it is
# always near the tail; 50 leaves room for system/user chatter around it.
_COMPLETION_SCAN_MESSAGES = 50


def _record_task_id(record: Dict[str, Any]) -> Optional[str]:
    return (record.get("data") or {}).get("task_id")


def _record_node_id(record: Dict[str, Any]) -> Optional[str]:
    return (record.get("data") or {}).get("workflow_node_id")


async def find_interrupted_attempt(
    message_store,
    project_id: str,
    run_id: str,
    completed_task_ids: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    """Return the run's interrupted attempt, or None when nothing to resume.

    The candidate is the last tool_call that carries a ``workflow_node_id`` —
    i.e. the last WORKFLOW-NODE attempt. A tool that spawns a nested agent
    (``delegate_to_agent``) makes the delegated child write its own
    tool_call/tool_result records under a different task_id with NO node id
    (delegated subtasks aren't workflow nodes); the newest raw tool_call is
    then the child's, and resuming it would fail (no node to re-enter). Keying
    on the node-bearing call treats those child records as internal to the
    parent tool execution and returns the parent's resumable cursor instead.

    It counts as completed — hence not resumable — when any marker exists:
    an assistant message whose metadata.task_id matches (normal phases); any
    approval after the attempt's last record (a gated phase suppresses the
    assistant final, so the gate row is its marker); or the run document
    listing this task_id in ``completed_task_ids`` (the durable marker a gated
    phase writes when it finishes, closing the window between phase completion
    and the approval row where neither of the first two exists).

    Records with task_id=None (written before the attempt id was wired
    through) can't be grouped into an attempt — treat as nothing to resume
    rather than fusing unrelated phases into one fake attempt.
    """
    journal = await message_store.get_messages(
        project_id,
        run_id=run_id,
        only_types=list(TOOL_LEDGER_TYPES),
        tail=True,
        limit=TOOL_JOURNAL_CONTEXT_RECORDS,
    )
    if not journal:
        return None

    last_task_id = None
    for record in reversed(journal):
        if record.get("type") == "tool_call" and _record_node_id(record):
            last_task_id = _record_task_id(record)
            break
    if not last_task_id:
        return None

    if completed_task_ids and last_task_id in completed_task_ids:
        return None

    attempt_records = [r for r in journal if _record_task_id(r) == last_task_id]
    last_sequence = max(r.get("sequence", 0) for r in attempt_records)

    finals = await message_store.get_messages(
        project_id,
        run_id=run_id,
        only_types=["assistant"],
        tail=True,
        limit=_COMPLETION_SCAN_MESSAGES,
    )
    for msg in finals:
        if (msg.get("metadata") or {}).get("task_id") == last_task_id:
            return None

    approvals_after = await message_store.get_messages(
        project_id,
        run_id=run_id,
        only_types=["approval"],
        after_sequence=last_sequence,
        limit=1,
    )
    if approvals_after:
        return None

    pairs = pair_tool_records(attempt_records)
    hanging = [p for p in pairs if p.get("call") and not p.get("result")]

    agent_id = None
    workflow_node_id = None
    for pair in pairs:
        data = (pair.get("call") or {}).get("data") or {}
        agent_id = data.get("agent_id") or agent_id
        workflow_node_id = data.get("workflow_node_id") or workflow_node_id

    return {
        "task_id": last_task_id,
        "agent_id": agent_id,
        "workflow_node_id": workflow_node_id,
        "pairs": [p for p in pairs if p.get("call") and p.get("result")],
        "hanging": hanging,
        "last_sequence": last_sequence,
    }


def attempt_has_outcome_unknown(attempt: Dict[str, Any]) -> bool:
    """True when closed tool pairs carry an outcome_unknown recovery marker."""
    for pair in attempt.get("pairs") or []:
        result_data = (pair.get("result") or {}).get("data") or {}
        inner = result_data.get("result") or {}
        if inner.get("outcome_unknown"):
            return True
    return False


async def last_completed_node_attempt(
    message_store,
    project_id: str,
    run_id: str,
    completed_task_ids: set,
) -> Optional[Dict[str, Any]]:
    """The last workflow-node attempt IF the run already marked it completed.

    A gated phase writes its ``completed_attempts`` marker when it finishes, but
    the approval row lands one node later; a restart in that window leaves the
    phase done yet ungated. ``find_interrupted_attempt`` returns None for it (by
    design — it must not be re-run), so this reports it separately as
    ``{task_id, workflow_node_id}`` for the caller to advance to its gate.
    Returns None when the newest node-bearing call isn't a completed one (then
    it's either interrupted — ``find_interrupted_attempt``'s job — or absent).
    """
    if not completed_task_ids:
        return None
    journal = await message_store.get_messages(
        project_id,
        run_id=run_id,
        only_types=list(TOOL_LEDGER_TYPES),
        tail=True,
        limit=TOOL_JOURNAL_CONTEXT_RECORDS,
    )
    for record in reversed(journal or []):
        if record.get("type") == "tool_call" and _record_node_id(record):
            task_id = _record_task_id(record)
            if task_id in completed_task_ids:
                return {
                    "task_id": task_id,
                    "workflow_node_id": _record_node_id(record),
                }
            return None
    return None


async def last_node_attempt(
    message_store,
    project_id: str,
    run_id: str,
) -> Optional[Dict[str, Any]]:
    """``{task_id, workflow_node_id}`` of the run's most recent WORKFLOW-NODE
    tool_call, or None.

    This is the exact cursor ``find_interrupted_attempt`` keys on (its
    ``last_task_id``), so recording THIS task id in ``completed_attempts`` is
    what makes recovery treat the just-finished node as done rather than
    re-running it. Deliberately kept in lockstep with that scan: a node that
    ran no node-bearing tool leaves nothing for either path to key on, so
    returning None (mark nothing) stays self-consistent instead of guessing.
    """
    journal = await message_store.get_messages(
        project_id,
        run_id=run_id,
        only_types=list(TOOL_LEDGER_TYPES),
        tail=True,
        limit=TOOL_JOURNAL_CONTEXT_RECORDS,
    )
    for record in reversed(journal or []):
        if record.get("type") == "tool_call" and _record_node_id(record):
            return {
                "task_id": _record_task_id(record),
                "workflow_node_id": _record_node_id(record),
            }
    return None


async def load_attempt_pairs(
    message_store,
    project_id: str,
    run_id: str,
    task_id: str,
    *,
    complete: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """Fetch and pair one attempt's journal records; None when it has none."""
    journal = []
    after_sequence = 0
    while True:
        page = await message_store.get_messages(
            project_id,
            run_id=run_id,
            only_types=list(TOOL_LEDGER_TYPES),
            tail=not complete,
            after_sequence=after_sequence,
            limit=TOOL_JOURNAL_CONTEXT_RECORDS,
        )
        journal.extend(page)
        if not complete or len(page) < TOOL_JOURNAL_CONTEXT_RECORDS:
            break
        next_sequence = page[-1]["sequence"]
        if next_sequence <= after_sequence:
            raise ValueError("Tool journal pagination did not advance")
        after_sequence = next_sequence
    records = [r for r in journal if _record_task_id(r) == task_id]
    if not records:
        return None
    return pair_tool_records(records)


def build_resume_input_items(
    pairs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rebuild the LLM transcript items for closed pairs — restore executes
    nothing.

    Mirrors the live loop's wire shape (streaming_agent_runner.run): raw
    ``arguments`` verbatim, results through the same output formatter. The
    provider item id was never persisted, so the ``id`` key is omitted
    entirely — an explicit None would be sent on the wire. A hanging call is
    skipped: a ``function_call`` without its output is a protocol error, and
    the trigger parks the run instead of resuming while one exists.
    """
    items: List[Dict[str, Any]] = []
    for pair in pairs:
        call = pair.get("call")
        result = pair.get("result")
        if not call or not result:
            continue
        call_data = call.get("data") or {}
        result_data = result.get("data") or {}
        call_id = call_data.get("tool_call_id")
        items.append({
            "type": "function_call",
            "call_id": call_id,
            "name": call_data.get("name"),
            "arguments": call_data.get("arguments") or "",
        })
        items.append({
            "type": "function_call_output",
            "call_id": call_id,
            "output": StreamingAgentRunner._format_tool_output(
                result_data.get("result")
            ),
        })
    return items
