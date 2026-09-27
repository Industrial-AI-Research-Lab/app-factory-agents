"""Typed extraction of CoScientist progress payloads from A2A status updates."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


COSCIENTIST_PROGRESS_SCHEMA_VERSION = "coscientist-a2a-progress-v1"


class CoScientistProgressEvent(BaseModel):
    """A redacted, ordered progress fact published in an A2A status message."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: str = Field(pattern=f"^{COSCIENTIST_PROGRESS_SCHEMA_VERSION}$")
    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    sequence: int = Field(gt=0)
    event_type: str = Field(alias="type", min_length=1)
    agent: str | None = None
    tool_name: str | None = None


def extract_coscientist_progress(status_update: dict[str, Any]) -> CoScientistProgressEvent | None:
    """Return the typed trace event carried by an A2A status update, if any."""

    message = status_update.get("message")
    if not isinstance(message, dict):
        return None
    parts = message.get("parts")
    if not isinstance(parts, list):
        return None
    for part in parts:
        if not isinstance(part, dict) or part.get("type") != "data":
            continue
        data = part.get("data")
        if not isinstance(data, dict) or data.get("schema_version") != COSCIENTIST_PROGRESS_SCHEMA_VERSION:
            continue
        try:
            return CoScientistProgressEvent.model_validate(data)
        except ValidationError:
            return None
    return None


def status_update_text(status_update: dict[str, Any]) -> str | None:
    """Return the human-readable progress summary without interpreting its content."""

    message = status_update.get("message")
    if not isinstance(message, dict):
        return None
    parts = message.get("parts")
    if not isinstance(parts, list):
        return None
    text = "\n".join(
        part["text"].strip()
        for part in parts
        if isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
        and part["text"].strip()
    )
    return text or None
