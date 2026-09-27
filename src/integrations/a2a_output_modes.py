"""Card-driven validation for A2A output MIME modes.

The A2A SDK normalizes both supported protocol versions to the same serialized
part shape used by AppFactory.  This module validates that shape without I/O so
registration, preflight, workflow nodes, delegation, and admin calls can share
one contract instead of growing caller-specific MIME checks.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


class A2AOutputModeError(ValueError):
    """A structured output-mode contract violation."""

    def __init__(
        self,
        error_type: str,
        message: str,
        *,
        declared_modes: Sequence[str] = (),
        actual_mode: Optional[str] = None,
        location: Optional[str] = None,
        part_type: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.declared_modes = tuple(declared_modes)
        self.actual_mode = actual_mode
        self.location = location
        self.part_type = part_type

    def as_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": str(self),
            "declared_modes": list(self.declared_modes),
            "actual_mode": self.actual_mode,
            "location": self.location,
            "part_type": self.part_type,
        }


def normalize_mime_type(value: Any) -> str:
    """Return the comparison form of a MIME type, or an empty string."""
    if not isinstance(value, str):
        return ""
    return value.split(";", 1)[0].strip().lower()


def own_media_type(part: Dict[str, Any]) -> str:
    return normalize_mime_type(
        part.get("media_type") or part.get("mediaType") or part.get("mimeType")
    )


def _normalize_modes(values: Iterable[Any]) -> Tuple[str, ...]:
    modes = []
    seen = set()
    for value in values or ():
        mode = normalize_mime_type(value)
        if mode and mode not in seen:
            seen.add(mode)
            modes.append(mode)
    return tuple(modes)


def is_supported_output_mode(mode: str) -> bool:
    """Whether AppFactory can interpret the MIME semantically, not just preserve it."""
    normalized = normalize_mime_type(mode)
    return normalized.startswith("text/") or (
        normalized == "application/json" or normalized.endswith("+json")
    )


def validate_declared_output_modes(values: Iterable[Any]) -> Tuple[str, ...]:
    """Normalize card modes and require at least one mode AppFactory understands."""
    modes = _normalize_modes(values)
    if not modes:
        raise A2AOutputModeError(
            "a2a_output_modes_missing",
            "Agent card must declare at least one output mode",
        )
    if not any(is_supported_output_mode(mode) for mode in modes):
        raise A2AOutputModeError(
            "a2a_output_modes_unsupported",
            "Agent card declares no output mode supported by AppFactory",
            declared_modes=modes,
        )
    return modes


def resolve_declared_output_modes(
    card: Dict[str, Any], *, skill_id: Optional[str] = None
) -> Tuple[str, ...]:
    """Resolve per-skill overrides when a caller explicitly selected a skill."""
    if skill_id:
        for skill in card.get("skills") or []:
            if isinstance(skill, dict) and skill.get("id") == skill_id:
                override = skill.get("outputModes")
                if override is not None:
                    return _normalize_modes(override)
                break
    return _normalize_modes(card.get("defaultOutputModes") or ())


def supported_declared_output_modes(values: Iterable[Any]) -> Tuple[str, ...]:
    """Return normalized declared modes that AppFactory may advertise as accepted."""
    modes = validate_declared_output_modes(values)
    return tuple(mode for mode in modes if is_supported_output_mode(mode))


def compatible_declared_modes(
    part_type: str, declared_modes: Iterable[Any]
) -> Tuple[str, ...]:
    """Declared modes a ``part_type`` part may be read as when it has no media type."""
    modes = _normalize_modes(declared_modes)
    if part_type == "text":
        return tuple(mode for mode in modes if mode.startswith("text/"))
    if part_type == "data":
        return tuple(
            mode
            for mode in modes
            if mode == "application/json" or mode.endswith("+json")
        )
    return ()


def _part_actual_mode(
    part: Dict[str, Any],
    declared_modes: Tuple[str, ...],
    protocol_version: Optional[str],
) -> Tuple[str, str]:
    part_type = str(part.get("type") or "")
    explicit = own_media_type(part)
    if explicit:
        return part_type, explicit
    candidates = compatible_declared_modes(part_type, declared_modes)
    if len(candidates) == 1:
        return part_type, candidates[0]
    if len(candidates) > 1:
        if protocol_version == "0.3":
            # Legacy TextPart/DataPart has no mediaType field. All candidates use the
            # same parser in AppFactory, so acceptedOutputModes is the only negotiation
            # signal available and any compatible declared candidate is valid.
            return part_type, candidates[0]
        raise A2AOutputModeError(
            "a2a_output_mode_ambiguous",
            f"A2A {part_type} part needs mediaType because multiple declared modes match",
            part_type=part_type,
        )
    if part_type == "text":
        return part_type, "text/plain"
    if part_type == "data":
        return part_type, "application/json"
    raise A2AOutputModeError(
        "a2a_output_part_invalid",
        f"A2A {part_type or 'unknown'} part has no media type",
        part_type=part_type or None,
    )


def validate_part_output_mode(
    part: Dict[str, Any],
    declared_modes: Iterable[Any],
    *,
    location: str,
    protocol_version: Optional[str] = None,
) -> str:
    """Validate one serialized SDK Part and return its normalized actual MIME."""
    modes = validate_declared_output_modes(declared_modes)
    if not isinstance(part, dict):
        raise A2AOutputModeError(
            "a2a_output_part_invalid",
            "A2A output part must be an object",
            declared_modes=modes,
            location=location,
        )
    try:
        part_type, actual_mode = _part_actual_mode(part, modes, protocol_version)
    except A2AOutputModeError as exc:
        exc.declared_modes = modes
        exc.location = location
        raise
    if not is_supported_output_mode(actual_mode):
        raise A2AOutputModeError(
            "a2a_output_modes_unsupported",
            f"AppFactory cannot interpret A2A output mode '{actual_mode}'",
            declared_modes=modes,
            actual_mode=actual_mode,
            location=location,
            part_type=part_type,
        )
    if actual_mode not in modes:
        raise A2AOutputModeError(
            "a2a_output_mode_violation",
            f"A2A output mode '{actual_mode}' was not declared by the agent",
            declared_modes=modes,
            actual_mode=actual_mode,
            location=location,
            part_type=part_type,
        )
    return actual_mode


def _validate_parts(
    parts: Iterable[Dict[str, Any]],
    declared_modes: Iterable[Any],
    *,
    prefix: str,
    protocol_version: Optional[str],
) -> None:
    for index, part in enumerate(parts or ()):
        validate_part_output_mode(
            part,
            declared_modes,
            location=f"{prefix}.parts[{index}]",
            protocol_version=protocol_version,
        )
        if own_media_type(part):
            continue
        card_types = compatible_declared_modes(
            str(part.get("type") or ""), declared_modes
        )
        if card_types:
            part["card_media_types"] = list(card_types)


def _validate_artifacts(
    artifacts: Iterable[Dict[str, Any]],
    declared_modes: Iterable[Any],
    *,
    prefix: str,
    protocol_version: Optional[str],
) -> None:
    for index, artifact in enumerate(artifacts or ()):
        if not isinstance(artifact, dict):
            raise A2AOutputModeError(
                "a2a_output_part_invalid",
                "A2A artifact must be an object",
                declared_modes=_normalize_modes(declared_modes),
                location=f"{prefix}.artifacts[{index}]",
            )
        _validate_parts(
            artifact.get("parts") or (),
            declared_modes,
            prefix=f"{prefix}.artifacts[{index}]",
            protocol_version=protocol_version,
        )


def validate_event_output_modes(
    event: Dict[str, Any],
    declared_modes: Iterable[Any],
    *,
    protocol_version: Optional[str] = None,
) -> None:
    """Validate result Parts, excluding status and history narrative.

    Agent-card output modes negotiate payloads AppFactory consumes as results:
    direct messages and artifact parts.  Task/status messages and task history
    are service narrative, so their human-readable Parts are not constrained by
    the result MIME contract.  A result Part without its own media type gets the
    card's compatible types as ``card_media_types``, so file naming follows the
    same declaration the validator accepted.
    """
    modes = validate_declared_output_modes(declared_modes)
    event_type = event.get("type")
    data = event.get("data") or {}
    if event_type == "message":
        _validate_parts(
            data.get("parts") or (),
            modes,
            prefix="message",
            protocol_version=protocol_version,
        )
        return
    if event_type == "task":
        _validate_artifacts(
            data.get("artifacts") or (),
            modes,
            prefix="task",
            protocol_version=protocol_version,
        )
        return
    if event_type == "artifact_update":
        artifact = data.get("artifact") if isinstance(data.get("artifact"), dict) else data
        _validate_parts(
            artifact.get("parts") or (),
            modes,
            prefix="artifact_update",
            protocol_version=protocol_version,
        )
        return
    if event_type == "status_update":
        return
    raise A2AOutputModeError(
        "a2a_output_event_invalid",
        f"A2A response event type '{event_type}' is not recognized",
        declared_modes=modes,
        location="event.type",
    )
