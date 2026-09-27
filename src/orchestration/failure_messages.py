"""Render a workflow failure result as a chat card: a plain-language summary
plus a capped technical detail.

A failure result carries raw internals — a validator's rejected value, a
stringified exception. The card's headline must explain the failure in a
sentence or two with no raw strings and no serialized values; the raw trace
belongs in the collapsible detail, length-capped. `PROJECT_FAILED` and the
Events tab keep the full raw error separately, so nothing is lost for debugging.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

DETAIL_MAX_CHARS = 2000

_SHAPE_WORDS = {
    "array": "a list",
    "list": "a list",
    "object": "an object",
    "dict": "an object",
    "string": "text",
    "str": "text",
    "number": "a number",
    "int": "a number",
    "float": "a number",
    "integer": "a whole number",
    "boolean": "a true/false value",
    "bool": "a true/false value",
    "null": "empty",
    "NoneType": "empty",
    "missing": "missing",
}

_HUMAN_BY_ERROR_TYPE = {
    "invalid_workflow": "The workflow definition is invalid, so the run could not start.",
    "configuration_error": "The run hit a configuration problem and stopped.",
    "checkpoint_workflow_ambiguous": "The run could not resume because its checkpoint was ambiguous.",
}


def build_failure_chat_message(result: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    result = result or {}
    reason = result.get("reason")
    error_type = result.get("error_type")
    node_id = result.get("node_id")
    raw = result.get("error") or result.get("reason") or str(result)

    content = _summarize(reason, error_type, node_id, result)

    data: Dict[str, Any] = {"detail": _cap(raw)}
    if error_type:
        data["error_type"] = error_type
    if node_id:
        data["node_id"] = node_id
    return content, data


def _summarize(
    reason: Optional[str],
    error_type: Optional[str],
    node_id: Optional[str],
    result: Dict[str, Any],
) -> str:
    step = f"'{node_id}'" if node_id else "a step"

    if reason in ("validator_retry_limit_exhausted", "validator_rejected_no_retry_path"):
        base = _validator_shape_sentence(result, step)
        if reason == "validator_retry_limit_exhausted":
            return base + " The run stopped after the retry limit."
        return base + " No retry path was configured, so the run stopped."

    if reason == "validator_bound_field_rejected":
        return (
            f"A format check ({step}) rejected fields AppFactory fills itself. "
            "This is a system error, not the model's, so the run stopped."
        )

    if reason == "validator_invalid_check_config":
        return f"A validator ({step}) is misconfigured, so the run could not continue."

    if reason == "workflow_timeout":
        return f"The run took too long and was stopped at {step}."
    if reason == "workflow_max_iterations":
        return f"The run hit its step limit before finishing (stopped at {step})."

    if error_type in _HUMAN_BY_ERROR_TYPE:
        return _HUMAN_BY_ERROR_TYPE[error_type]
    if error_type == "execution_failed":
        return f"A task ({step}) failed while running, so the run stopped."
    if error_type == "writes_contract_violation":
        return f"A step ({step}) did not produce its required output, so the run stopped."
    if error_type and error_type.startswith("a2a_"):
        return f"An external agent step ({step}) failed, so the run stopped."

    return "The run failed unexpectedly. Open Details for the technical error."


def _validator_shape_sentence(result: Dict[str, Any], step: str) -> str:
    detail = _first_validator_detail(result)
    if detail and detail.get("key"):
        key = detail["key"]
        expected_raw = detail.get("expected")
        got = _shape(detail.get("got"))
        # Only the shape phrasing when `expected` is a concrete type; other
        # jsonschema keywords (required/enum/minItems) land in `expected` as the
        # keyword name and would read as "but required was required".
        if expected_raw in _SHAPE_WORDS and got:
            return f"A step returned '{key}' as {got}, but {_SHAPE_WORDS[expected_raw]} was required."
        return f"A step produced '{key}' in the wrong format."
    return f"A format check ({step}) rejected the step's output."


def _first_validator_detail(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    details = result.get("validator_errors")
    if isinstance(details, list) and details and isinstance(details[0], dict):
        return details[0]
    return None


def _shape(word: Optional[str]) -> Optional[str]:
    if not word:
        return None
    return _SHAPE_WORDS.get(word, word)


def _cap(text: str) -> str:
    if len(text) <= DETAIL_MAX_CHARS:
        return text
    return text[:DETAIL_MAX_CHARS] + "… (truncated)"
