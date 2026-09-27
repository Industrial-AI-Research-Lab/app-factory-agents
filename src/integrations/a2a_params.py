"""Card-extension-driven A2A request augmentation.

An A2A agent can declare a *required Profile Extension* in its agent card
(``capabilities.extensions[]``). The extension's ``params`` is a JSON-Schema that the
incoming message must satisfy — e.g. IDU's ``scenario-context/v1`` requires a DataPart
``{"scenario_id": <int>}``. This module turns the run's free-text intent into that
schema-valid DataPart via one schema-constrained (forced-tool) LLM call, and exposes the
URIs to activate via the ``A2A-Extensions`` request header.

Why here, and why this shape:
  - The agent card is the single source of truth for which params exist and their types.
    Adding a param is the agent updating its card — zero change on our side. We never
    re-declare the param set in workflow config or hand-written regex.
  - Pure + agent-agnostic: the only inputs are the card capabilities, the intent text,
    and an LLM client. Reused by the workflow a2a node today and the future delegate
    path; designed to drop into a before-invocation plugin hook unchanged.
  - The full request is ``text parts`` (built by the caller) ++ ``extract_extension_dataparts``.
    We deliberately do NOT rebuild the text/reads here — that stays the caller's job.

See docs/adr/0006-a2a-param-extraction.md.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Per the A2A spec (topics/extensions): a client activates a profile extension by listing
# its URI(s) in this request header (comma-separated). Note: NO ``X-`` prefix — IDU's card
# names it the same way. Without it the agent may ignore the DataPart and fall back to
# parsing the intent text.
EXTENSIONS_ACTIVATION_HEADER = "A2A-Extensions"

_INTEGER_RE = re.compile(r"-?\d+")


class A2AParamExtractionError(Exception):
    """A required extension param could not be extracted from the intent text.

    Raised *before* the agent is contacted so the caller fails fast with the real
    reason (e.g. "missing required ['scenario_id']") instead of letting the agent
    reject a malformed/absent param after a full round-trip.
    """


def required_extensions_with_schema(
    capabilities: Optional[Dict[str, Any]],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Return ``[(uri, params_schema), ...]`` for required extensions that publish an
    object params schema.

    Only ``required: true`` extensions whose ``params`` is an object-typed JSON-Schema
    are returned — those are the ones we must satisfy on every message. Anything else
    (data-only extensions, missing/blank schema) yields nothing, so callers do no
    extension work at all (no header, no LLM call).
    """
    extensions = (capabilities or {}).get("extensions")
    if not isinstance(extensions, list):
        return []

    out: List[Tuple[str, Dict[str, Any]]] = []
    for ext in extensions:
        if not isinstance(ext, dict) or not ext.get("required"):
            continue
        uri = ext.get("uri")
        params = ext.get("params")
        # A profile extension we must satisfy declares an object schema with properties.
        # Non-object or property-less schemas carry nothing to extract → skip.
        if not uri or not isinstance(params, dict):
            continue
        if params.get("type") != "object" or not isinstance(params.get("properties"), dict):
            continue
        out.append((str(uri), params))
    return out


def extension_activation_header_value(capabilities: Optional[Dict[str, Any]]) -> str:
    """Comma-joined extension URIs for the ``A2A-Extensions`` header, or ``""`` when the
    agent declares no required extension (caller then sends no activation header)."""
    return ",".join(uri for uri, _ in required_extensions_with_schema(capabilities))


def _schema_for_tool(params_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Strip keys that belong to a standalone JSON-Schema document but aren't valid in a
    tool/function ``parameters`` block (some gateways reject ``$schema``/``$id``)."""
    return {k: v for k, v in params_schema.items() if not k.startswith("$")}


def _first_tool_arguments(tool_calls: Any) -> Optional[Dict[str, Any]]:
    """Pull the first tool call's arguments as a dict, tolerating both the pydantic
    object shape (``tc.function.arguments`` as a JSON string) and the raw-dict shape
    some providers return. Returns None when there is no usable tool call.

    JSON decode errors propagate as A2AParamExtractionError to the caller.
    """
    if not tool_calls:
        return None
    tc = tool_calls[0]
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        raw = fn.get("arguments")
    else:
        fn = getattr(tc, "function", None)
        raw = getattr(fn, "arguments", None) if fn else None

    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise A2AParamExtractionError(f"tool arguments were not valid JSON: {exc}")
        return parsed if isinstance(parsed, dict) else None
    return None


def _coerce_scalar(value: Any, json_type: Any, name: str) -> Any:
    """Coerce one extracted value to its schema-declared scalar type.

    ``bool`` is a subclass of ``int`` in Python, so a JSON ``true`` would silently
    satisfy an ``integer`` schema as ``1``; we reject it explicitly for numeric types.
    Numeric strings ("772") are coerced because a model emitting structured output may
    still quote a number. Non-scalar types (array/object) and unknown types pass through
    unchanged — the forced-tool schema already shaped them.
    """
    if json_type == "integer":
        if isinstance(value, bool):
            raise A2AParamExtractionError(f"'{name}' must be an integer, got boolean")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and _INTEGER_RE.fullmatch(value.strip()):
            return int(value.strip())
        raise A2AParamExtractionError(f"'{name}' must be an integer, got {value!r}")

    if json_type == "number":
        if isinstance(value, bool):
            raise A2AParamExtractionError(f"'{name}' must be a number, got boolean")
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                pass
        raise A2AParamExtractionError(f"'{name}' must be a number, got {value!r}")

    if json_type == "boolean":
        if isinstance(value, bool):
            return value
        raise A2AParamExtractionError(f"'{name}' must be a boolean, got {value!r}")

    if json_type == "string":
        return value if isinstance(value, str) else str(value)

    return value


def _coerce_and_validate(
    args: Dict[str, Any],
    schema: Dict[str, Any],
    already_provided: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Project extracted args onto the schema: keep only declared properties, coerce each
    to its declared type, and ensure every ``required`` property is present.

    Absent optional properties are dropped, never fabricated — the structured-output
    guarantee is *format*, not *correctness*, so we don't invent values the user never
    gave. Raises A2AParamExtractionError when a required property is missing.

    ``already_provided`` are keys the caller already sent as a DataPart (e.g. a workflow
    data-read). They count toward ``required`` (so a key supplied that way isn't reported
    missing) and are NOT re-emitted here, so the wire never carries two parts for one key.
    """
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    already_provided = already_provided or {}

    out: Dict[str, Any] = {}
    for name, spec in props.items():
        # The caller already sent this key as its own DataPart; don't duplicate it.
        if name in already_provided:
            continue
        if name not in args or args[name] is None:
            continue
        json_type = spec.get("type") if isinstance(spec, dict) else None
        out[name] = _coerce_scalar(args[name], json_type, name)

    missing = [
        name for name in required
        if name not in out and name not in already_provided
    ]
    if missing:
        raise A2AParamExtractionError(f"missing required {missing}")
    return out


async def extract_extension_dataparts(
    intent_text: str,
    capabilities: Optional[Dict[str, Any]],
    llm_client: Any,
    *,
    model: Optional[str] = None,
    api_key_override: Optional[str] = None,
    fallback_models_override: Optional[List[str]] = None,
    already_provided: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build one schema-valid DataPart per required extension, extracting values from
    ``intent_text`` via a forced-tool LLM call.

    Returns ``[]`` (and makes NO LLM call) when the agent declares no required extension
    or every declared property is already supplied. Each returned part is
    ``{"kind": "data", "data": {...}}`` whose ``data`` conforms to that extension's params
    schema. Raises A2AParamExtractionError if a required property cannot be extracted.

    ``already_provided`` are keys the caller already supplied as a DataPart (e.g. a workflow
    data-read). A required key satisfied that way is not re-extracted and not duplicated,
    so supplying ``scenario_id`` via a data-read is an alternative to stating it in text —
    not a guaranteed "missing required" failure.
    """
    targets = required_extensions_with_schema(capabilities)
    if not targets:
        return []

    already_provided = already_provided or {}

    parts: List[Dict[str, Any]] = []
    for uri, params_schema in targets:
        if all(
            name in already_provided
            for name in (params_schema.get("properties") or {})
        ):
            continue
        tool_schema = _schema_for_tool(params_schema)
        tool = {
            "type": "function",
            "function": {
                "name": "report_parameters",
                "description": (
                    "Report the parameter values explicitly stated in the user's request. "
                    "Include a value only if the user actually gave it; omit anything not stated."
                ),
                "parameters": tool_schema,
            },
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You extract structured parameters from a user's request for a downstream "
                    "agent. Call the tool with ONLY values the user explicitly stated. Never "
                    "guess, infer, or invent a value that is not present in the text."
                ),
            },
            {"role": "user", "content": intent_text or ""},
        ]

        result = await llm_client.chat_completion(
            messages=messages,
            model=model,
            temperature=0,
            api_key_override=api_key_override,
            fallback_models_override=fallback_models_override,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "report_parameters"}},
        )

        args = _first_tool_arguments((result or {}).get("tool_calls"))
        if args is None:
            # Forced tool-use returned no usable call. Fail only for required props the
            # caller did NOT already supply as a DataPart; otherwise nothing to send here.
            unmet = [
                r for r in (params_schema.get("required") or [])
                if r not in already_provided
            ]
            if unmet:
                raise A2AParamExtractionError(
                    f"extension {uri}: model returned no parameters for required {unmet}"
                )
            continue

        data = _coerce_and_validate(args, params_schema, already_provided)
        if data:
            logger.info("[A2A_PARAMS] extracted %s for extension %s", sorted(data), uri)
            parts.append({"kind": "data", "data": data})

    return parts
