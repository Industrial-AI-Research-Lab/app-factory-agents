"""Small HTTP client for the CoScientist checkpoint adapter."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx


class CheckpointError(Exception):
    """Safe, structured failure returned by the checkpoint adapter client."""

    def __init__(
        self,
        code: str,
        status_code: int,
        outcome_unknown: bool,
        message: str = "Checkpoint operation failed",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.outcome_unknown = outcome_unknown


class CheckpointClient:
    """Issue single-attempt checkpoint requests using operational A2A config."""

    def __init__(self, config: dict[str, Any], *, transport=None) -> None:
        self._base_url = str(config["url"]).rstrip("/")
        self._headers = dict(config.get("headers") or {})
        self._auth = config.get("auth")
        self._timeout = config.get("request_timeout_seconds", 60)
        self._transport = transport

    async def register_run(
        self,
        context_id: str,
        run_id: str,
        traceparent: str | None = None,
    ) -> None:
        self._validate_id(context_id, "context_id")
        self._validate_id(run_id, "run_id")
        response = await self._post(
            "/api/checkpoints/runs",
            {"context_id": context_id, "run_id": run_id, "traceparent": traceparent},
        )
        try:
            accepted = response.json().get("ok") is True
        except (ValueError, AttributeError):
            accepted = False
        if not accepted:
            raise CheckpointError("checkpoint_rejected", 502, False)

    async def restore(self, point_id: str) -> str:
        self._validate_id(point_id, "point_id")
        encoded_id = quote(point_id, safe="")
        response = await self._post(
            f"/api/checkpoints/{encoded_id}/restore",
            {"compat": "relaxed", "import_stores": True},
            server_error_outcome_unknown=True,
        )
        try:
            context_id = response.json().get("context_id")
        except (ValueError, AttributeError):
            context_id = None
        if not isinstance(context_id, str) or not context_id.strip():
            raise CheckpointError(
                "checkpoint_malformed_response",
                502,
                True,
                "Checkpoint restore outcome is unknown",
            )
        return context_id

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        server_error_outcome_unknown: bool = False,
    ) -> httpx.Response:
        try:
            async with httpx.AsyncClient(
                headers=self._headers,
                auth=self._auth,
                timeout=self._timeout,
                transport=self._transport,
                follow_redirects=False,
            ) as client:
                response = await client.post(f"{self._base_url}{path}", json=payload)
        except httpx.RequestError as exc:
            raise CheckpointError(
                "checkpoint_transport_error",
                502,
                True,
                "Checkpoint request outcome is unknown",
            ) from exc

        if response.status_code == 404:
            raise CheckpointError("checkpoint_not_found", 404, False)
        if response.status_code == 409:
            raise CheckpointError("checkpoint_conflict", 409, False)
        if not 200 <= response.status_code < 300:
            raise CheckpointError(
                "checkpoint_rejected",
                502,
                server_error_outcome_unknown and response.status_code >= 500,
            )
        return response

    @staticmethod
    def _validate_id(value: str, field_name: str) -> None:
        if not isinstance(value, str) or not value.strip() or value in {".", ".."}:
            raise ValueError(f"{field_name} must be a nonempty non-dot identifier")
