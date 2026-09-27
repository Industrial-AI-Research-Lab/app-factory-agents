"""Strict Agent Card validation backed by the versioned contracts in docs/."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from jsonschema import Draft202012Validator, FormatChecker

from integrations.a2a_output_modes import (
    A2AOutputModeError,
    supported_declared_output_modes,
    validate_declared_output_modes,
)


_SCHEMA_FILES = {
    "0.3": "AppFactory-a2a-0.3.schema.json",
    "1.0": "AppFactory-a2a-1.0.schema.json",
}
_LEGACY_CARD_FIELDS = {
    "protocolVersion",
    "url",
    "preferredTransport",
    "additionalInterfaces",
    "supportsAuthenticatedExtendedCard",
}


class A2AContractError(ValueError):
    """A structured Agent Card or runtime contract failure."""

    def __init__(
        self,
        error_type: str,
        message: str,
        *,
        location: str = "agentCard",
        protocol_version: Optional[str] = None,
        declared_modes: Tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.location = location
        self.protocol_version = protocol_version
        self.declared_modes = declared_modes

    def as_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": str(self),
            "location": self.location,
            "protocol_version": self.protocol_version,
            "declared_modes": list(self.declared_modes),
        }


@dataclass(frozen=True)
class A2AContractSnapshot:
    protocol_version: str
    agent_version: str
    declared_output_modes: Tuple[str, ...]
    accepted_output_modes: Tuple[str, ...]
    card_hash: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "agent_version": self.agent_version,
            "declared_output_modes": list(self.declared_output_modes),
            "accepted_output_modes": list(self.accepted_output_modes),
            "card_hash": self.card_hash,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "A2AContractSnapshot":
        """Rebuild and validate an immutable contract loaded from persistence."""
        if not isinstance(value, dict):
            raise A2AContractError(
                "a2a_task_contract_invalid",
                "Persisted A2A task contract must be an object",
                location="a2aTaskContract.contract",
            )
        protocol_version = value.get("protocol_version")
        agent_version = value.get("agent_version")
        card_hash = value.get("card_hash")
        if protocol_version not in _SCHEMA_FILES:
            raise A2AContractError(
                "a2a_task_contract_invalid",
                "Persisted A2A task contract has an unsupported protocol version",
                location="a2aTaskContract.contract.protocol_version",
                protocol_version=str(protocol_version) if protocol_version else None,
            )
        for field, field_value in (
            ("agent_version", agent_version),
            ("card_hash", card_hash),
        ):
            if not isinstance(field_value, str) or not field_value:
                raise A2AContractError(
                    "a2a_task_contract_invalid",
                    f"Persisted A2A task contract has an invalid {field}",
                    location=f"a2aTaskContract.contract.{field}",
                    protocol_version=protocol_version,
                )
        try:
            declared_modes = validate_declared_output_modes(
                value.get("declared_output_modes") or ()
            )
            accepted_modes = validate_declared_output_modes(
                value.get("accepted_output_modes") or ()
            )
        except A2AOutputModeError as exc:
            raise A2AContractError(
                "a2a_task_contract_invalid",
                str(exc),
                location="a2aTaskContract.contract",
                protocol_version=protocol_version,
                declared_modes=exc.declared_modes,
            ) from exc
        expected_accepted = supported_declared_output_modes(declared_modes)
        if accepted_modes != expected_accepted:
            raise A2AContractError(
                "a2a_task_contract_invalid",
                "Persisted A2A accepted output modes do not match declared modes",
                location="a2aTaskContract.contract.accepted_output_modes",
                protocol_version=protocol_version,
                declared_modes=declared_modes,
            )
        return cls(
            protocol_version=protocol_version,
            agent_version=agent_version,
            declared_output_modes=declared_modes,
            accepted_output_modes=accepted_modes,
            card_hash=card_hash,
        )


def _contracts_dir() -> Path:
    configured = os.getenv("A2A_CONTRACTS_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "docs" / "contracts" / "a2a"


@lru_cache(maxsize=2)
def _agent_card_validator(protocol_version: str) -> Draft202012Validator:
    filename = _SCHEMA_FILES.get(protocol_version)
    if not filename:
        raise A2AContractError(
            "a2a_protocol_version_unsupported",
            f"Unsupported A2A protocol version '{protocol_version}'",
            protocol_version=protocol_version,
        )
    schema_path = _contracts_dir() / filename
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load A2A contract schema '{schema_path}': {exc}") from exc
    agent_card_schema = {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        "$ref": "#/$defs/agentCard",
    }
    Draft202012Validator.check_schema(agent_card_schema)
    return Draft202012Validator(agent_card_schema, format_checker=FormatChecker())


def detect_agent_card_protocol(card: Dict[str, Any]) -> str:
    """Select one strict contract and reject cards that mix version families."""
    if not isinstance(card, dict):
        raise A2AContractError(
            "a2a_contract_invalid", "Agent card must be a JSON object"
        )
    has_interfaces = "supportedInterfaces" in card
    has_legacy = any(field in card for field in _LEGACY_CARD_FIELDS)
    if has_interfaces and has_legacy:
        raise A2AContractError(
            "a2a_contract_invalid",
            "Agent card mixes A2A 0.3 and 1.0 fields",
        )
    if card.get("protocolVersion") == "0.3.0":
        return "0.3"
    if has_interfaces:
        versions = {
            interface.get("protocolVersion")
            for interface in card.get("supportedInterfaces") or []
            if isinstance(interface, dict)
        }
        if versions == {"1.0"}:
            return "1.0"
        version = ", ".join(sorted(str(value) for value in versions)) or "missing"
        raise A2AContractError(
            "a2a_protocol_version_unsupported",
            f"Unsupported A2A interface protocol version '{version}'",
            protocol_version=version,
        )
    version = card.get("protocolVersion")
    raise A2AContractError(
        "a2a_protocol_version_unsupported",
        f"Unsupported or missing A2A protocol version '{version}'",
        protocol_version=str(version) if version is not None else None,
    )


def _validation_location(error) -> str:
    path = [str(part) for part in error.absolute_path]
    if error.validator == "required":
        missing = next(
            (
                name
                for name in error.validator_value
                if f"'{name}' is a required property" == error.message
            ),
            None,
        )
        if missing:
            path.append(missing)
    return ".".join(["agentCard", *path])


def _validate_skill_output_modes(card: Dict[str, Any], protocol_version: str) -> None:
    for index, skill in enumerate(card.get("skills") or []):
        if not isinstance(skill, dict) or "outputModes" not in skill:
            continue
        try:
            validate_declared_output_modes(skill.get("outputModes") or ())
        except A2AOutputModeError as exc:
            raise A2AContractError(
                exc.error_type,
                str(exc),
                location=f"agentCard.skills[{index}].outputModes",
                protocol_version=protocol_version,
                declared_modes=exc.declared_modes,
            ) from exc


def validate_agent_card(card: Dict[str, Any]) -> A2AContractSnapshot:
    """Validate a raw Agent Card and return its immutable runtime contract."""
    protocol_version = detect_agent_card_protocol(card)
    errors = sorted(
        _agent_card_validator(protocol_version).iter_errors(card),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        error = errors[0]
        raise A2AContractError(
            "a2a_contract_invalid",
            error.message,
            location=_validation_location(error),
            protocol_version=protocol_version,
        )
    try:
        declared_modes = validate_declared_output_modes(card["defaultOutputModes"])
        accepted_modes = supported_declared_output_modes(declared_modes)
    except A2AOutputModeError as exc:
        raise A2AContractError(
            exc.error_type,
            str(exc),
            location="agentCard.defaultOutputModes",
            protocol_version=protocol_version,
            declared_modes=exc.declared_modes,
        ) from exc
    _validate_skill_output_modes(card, protocol_version)
    canonical = json.dumps(
        card, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return A2AContractSnapshot(
        protocol_version=protocol_version,
        agent_version=str(card.get("version") or ""),
        declared_output_modes=declared_modes,
        accepted_output_modes=accepted_modes,
        card_hash=hashlib.sha256(canonical).hexdigest(),
    )


def url_origin(parsed) -> str:
    """scheme://netloc with the default port stripped; '' when there's no host.

    Lets "host" and "host:80" (or https "host:443") compare equal so an explicit
    default port doesn't read as a different origin.
    """
    if not parsed.netloc:
        return ""
    netloc = parsed.netloc
    if parsed.scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif parsed.scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]
    return f"{parsed.scheme}://{netloc}"
