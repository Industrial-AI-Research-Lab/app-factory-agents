"""Structural, schema, and cross-context relation checks only —
no LLM calls or semantic judgment. Feedback strings are written for the
retrying agent: they name which key failed which format expectation.
"""

from __future__ import annotations

import logging
from typing import Any, Collection, Dict, List, Optional

from .validator_feedback import (
    bound_key,
    build_feedback,
    feedback_for_relation,
    relation_diff,
)
from .validator_id_sets import InvalidIdSetConfig, check_id_set_equals
from .validator_operators import has_inline_values, is_unary, other_key_label
from .validator_relations import (
    InvalidCheckConfig,
    check_relation_config,
    evaluate_relation,
)

logger = logging.getLogger(__name__)

# bool is deliberately its own entry: isinstance(True, int) is True in
# Python, so "number" must exclude bools explicitly (see _check_structural).
_TYPE_MAP = {
    "dict": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "bool": bool,
}


class ValidatorRunner:
    """Run a validator node's checks against a SharedContext (or plain dict)."""

    feedback_for_relation = staticmethod(feedback_for_relation)

    async def run(
        self,
        checks: List[Any],
        shared_context,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
        bound_outputs: Collection[str] = (),
    ) -> Dict[str, Any]:
        """A failure inside `bound_outputs` (see `bound_key`) is not the
        model's to fix."""
        errors: List[str] = []
        error_details: List[Dict[str, Any]] = []
        config_errors: List[str] = []
        for check in checks:
            if not isinstance(check, dict):
                config_errors.append(f"Invalid check (expected object): {check!r}")
                continue
            kind = check.get("kind", "structural")
            try:
                if kind == "structural":
                    err = await self._check_structural(check, shared_context)
                elif kind == "json_schema":
                    err = await self._check_json_schema(check, shared_context)
                elif kind == "value_relation":
                    err = await self._check_value_relation(
                        check,
                        shared_context,
                        extra_context or {},
                    )
                elif kind == "id_set_equals":
                    value = await self._resolve(shared_context, check.get("key", ""))
                    other = await self._resolve(
                        shared_context, check.get("other_key", "")
                    )
                    err = check_id_set_equals(check, value, other)
                else:
                    raise InvalidCheckConfig(f"Unknown check kind: {kind}")
            except (InvalidCheckConfig, InvalidIdSetConfig) as e:
                config_errors.append(str(e))
                continue
            if err:
                errors.append(err["message"])
                error_details.append(
                    {k: err.get(k) for k in ("key", "expected", "got")}
                )
        return {
            "passed": not errors and not config_errors,
            "errors": errors,
            # Same failures as `errors`, keyed for a caller that must name the
            # field and shape (the chat failure card) without parsing the strings.
            "error_details": error_details,
            "config_errors": config_errors,
            "bound_keys": [
                detail["key"]
                for detail in error_details
                if bound_key(detail["key"], bound_outputs)
            ],
            # Retry-facing text only: config errors are for the operator, not
            # something the agent can act on by revising its output.
            "feedback": build_feedback(errors, error_details, bound_outputs),
        }

    @staticmethod
    def _err(
        key: str,
        message: str,
        expected: Optional[str] = None,
        got: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {"message": message, "key": key, "expected": expected, "got": got}

    @staticmethod
    async def _resolve(shared_context, key_path: str) -> Any:
        """Resolve a dot-notation path like 'plan.tasks' or 'plan.tasks.0'."""
        parts = [p for p in (key_path or "").split(".") if p]
        if not parts:
            return None
        # Mirror the engine's own read chain (_read_context_values): only
        # read_context_key(_async) sees custom_context, where normal phase
        # writes land — SharedContext.get() would miss every custom key.
        async_reader = getattr(shared_context, "read_context_key_async", None)
        sync_reader = getattr(shared_context, "read_context_key", None)
        if callable(async_reader):
            value = await async_reader(parts[0])
        elif callable(sync_reader):
            value = sync_reader(parts[0])
        else:
            value = shared_context.get(parts[0])
        for part in parts[1:]:
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list):
                try:
                    value = value[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
        return value

    async def _check_structural(
        self, check: Dict[str, Any], shared_context
    ) -> Optional[Dict[str, Any]]:
        key = check.get("key", "")
        value = await self._resolve(shared_context, key)
        if value is None:
            return self._err(key, f"Key '{key}' is missing", got="missing")

        expected_type = check.get("type")
        if expected_type:
            py_type = _TYPE_MAP.get(expected_type)
            if py_type is None:
                raise InvalidCheckConfig(
                    f"Check on '{key}': unknown expected type '{expected_type}'"
                )
            if not isinstance(value, py_type) or (
                expected_type == "number" and isinstance(value, bool)
            ):
                return self._err(
                    key,
                    f"Key '{key}' expected type {expected_type}, got {type(value).__name__}",
                    expected=expected_type,
                    got=type(value).__name__,
                )

        if "min_length" in check:
            if not hasattr(value, "__len__") or len(value) < check["min_length"]:
                got = len(value) if hasattr(value, "__len__") else "?"
                return self._err(
                    key, f"Key '{key}' length < {check['min_length']} (got {got})"
                )
        if "max_length" in check:
            if hasattr(value, "__len__") and len(value) > check["max_length"]:
                return self._err(key, f"Key '{key}' length > {check['max_length']}")

        required_keys = check.get("required_keys") or []
        if required_keys:
            if not isinstance(value, dict):
                return self._err(
                    key,
                    f"Key '{key}' expected an object with keys {required_keys}, "
                    f"got {type(value).__name__}",
                    expected="object",
                    got=type(value).__name__,
                )
            missing = [rk for rk in required_keys if rk not in value]
            if missing:
                return self._err(
                    key, f"Key '{key}' missing required subkey(s): {', '.join(missing)}"
                )

        return None

    async def _check_json_schema(
        self, check: Dict[str, Any], shared_context
    ) -> Optional[Dict[str, Any]]:
        try:
            import jsonschema
        except ImportError:
            # Direct dep since AppFactory-77, but a stripped environment is a
            # broken CONFIG (not a value the agent can fix) — hard-fail.
            raise InvalidCheckConfig("jsonschema library not installed")
        key = check.get("key", "")
        schema = check.get("json_schema")
        if not schema:
            raise InvalidCheckConfig(f"Check on '{key}' missing 'json_schema'")
        value = await self._resolve(shared_context, key)
        if value is None:
            return self._err(key, f"Key '{key}' is missing", got="missing")
        try:
            jsonschema.validate(value, schema)
            return None
        except jsonschema.ValidationError as e:
            # e.message includes rejected values; the path identifies the field
            # without copying private content into feedback or the chat card.
            key = ".".join([key, *(str(part) for part in e.absolute_path)])
            keyword = e.validator
            got = type(e.instance).__name__
            if keyword == "type":
                expected = (
                    e.validator_value
                    if isinstance(e.validator_value, str)
                    else str(e.validator_value)
                )
                message = (
                    f"Key '{key}' schema violation: expected type {expected}, got {got}"
                )
            else:
                expected = str(keyword)
                message = f"Key '{key}' schema violation: failed '{keyword}' constraint, got {got}"
            return self._err(key, message, expected=expected, got=got)
        except jsonschema.SchemaError as e:
            raise InvalidCheckConfig(
                f"Check on '{key}' has an invalid json_schema: {e.message}"
            )
        except Exception as e:
            # jsonschema.validate() can also raise referencing errors (e.g.
            # jsonschema.exceptions._WrappedReferencingError for a dangling
            # or unresolvable $ref) that are neither ValidationError nor
            # SchemaError. Note: `except Exception` does not catch
            # asyncio.CancelledError (a BaseException) — that must keep
            # propagating.
            raise InvalidCheckConfig(
                f"Check on '{key}' has an unresolvable json_schema (e.g. broken $ref): {e}"
            )

    async def _check_value_relation(
        self,
        check: Dict[str, Any],
        shared_context,
        extra_context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        key = check.get("key", "")
        other_key = check.get("other_key", "")
        operator = check.get("operator")
        unary = is_unary(operator)
        check_relation_config(check)
        value = await self._resolve(shared_context, key)
        other_value = await self._resolve_other(check, shared_context, extra_context)
        if operator != "date_not_after":
            if value is None:
                return self._err(key, f"Key '{key}' is missing", got="missing")
            if other_value is None and not unary:
                return self._err(
                    key,
                    f"Comparison key '{other_key}' is missing",
                    expected=f"value from {other_key}",
                    got="missing",
                )
        if evaluate_relation(operator, check, value, other_value):
            return None
        diff = relation_diff(check, value, other_value) or {}
        return self._err(
            key,
            self.feedback_for_relation(check, **diff),
            expected=operator if unary else f"{operator} {other_key_label(check)}",
            got="relation_failed",
        )

    async def _resolve_other(
        self, check: Dict[str, Any], shared_context, extra_context: Dict[str, Any]
    ) -> Any:
        if is_unary(check.get("operator")):
            return None
        if has_inline_values(check):
            return check["values"]
        other_key = check["other_key"]
        other_value = await self._resolve(extra_context, other_key)
        other_root = next(
            (part for part in other_key.split(".") if part),
            "",
        )
        if other_value is None and other_root not in extra_context:
            other_value = await self._resolve(shared_context, other_key)
        return other_value
