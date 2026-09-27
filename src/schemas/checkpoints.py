"""Checkpoint integration configuration schemas."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class A2ACheckpointConfig(BaseModel):
    """Non-secret checkpoint settings stored with an A2A server."""

    enabled: bool = False
    callback_nonce_env: str | None = None

    @model_validator(mode="after")
    def validate_enabled_config(self) -> "A2ACheckpointConfig":
        if self.enabled and (
            not self.callback_nonce_env
            or not _ENV_NAME_PATTERN.fullmatch(self.callback_nonce_env)
        ):
            raise ValueError(
                "enabled checkpoints require callback_nonce_env to be an "
                "environment-variable name"
            )
        return self


class CheckpointPoint(BaseModel):
    """Opaque durable checkpoint notification from an A2A adapter."""

    model_config = ConfigDict(extra="forbid")

    point_id: str = Field(min_length=1, max_length=255)
    run_id: str = Field(min_length=1, max_length=255)
    time: datetime
    snapshot_ref: str = Field(min_length=1, max_length=4096)
    label: str | None = Field(default=None, max_length=1024)

    @field_validator("point_id", "run_id", "snapshot_ref")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("time")
    @classmethod
    def normalize_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
