"""Keep MCP delete reservations alive while host cleanup is in progress."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any

from tools.mcp_package_storage import (
    MCP_LEASE_DURATION,
    heartbeat_mcp_lease_worker,
    renew_mcp_built_image_deletion,
    renew_mcp_package_deletion,
)

logger = logging.getLogger(__name__)


class McpDeleteLeaseLost(RuntimeError):
    """A long-running deletion no longer owns one of its reservations."""


class McpDeleteLeaseHeartbeat:
    """Renew package and image deletion leases until the operation finishes."""

    def __init__(
        self,
        *,
        storage: Any,
        tenant_id: str,
        package_server_ids: list[str] | None = None,
        package_reservation_id: str | None = None,
        image_doc_ids: list[str] | None = None,
        image_reservation_id: str | None = None,
        interval_seconds: float | None = None,
    ) -> None:
        self._storage = storage
        self._tenant_id = tenant_id
        self._package_server_ids = list(package_server_ids or [])
        self._package_reservation_id = str(package_reservation_id or "").strip()
        self._image_doc_ids = list(image_doc_ids or [])
        self._image_reservation_id = str(image_reservation_id or "").strip()
        self._interval_seconds = interval_seconds or max(MCP_LEASE_DURATION.total_seconds() / 3, 1)
        self._task: asyncio.Task[None] | None = None
        self._failure: McpDeleteLeaseLost | None = None

    async def start(self) -> None:
        if self._task is None and (self._package_server_ids or self._image_doc_ids):
            try:
                await self._renew_all()
            except McpDeleteLeaseLost as exc:
                self._failure = exc
                return
            except Exception as exc:
                logger.warning("[MCP_DELETE_LEASE] heartbeat_failed error=%s", exc)
                self._failure = McpDeleteLeaseLost("MCP deletion lease heartbeat failed")
                return
            self._task = asyncio.create_task(self._run(), name="mcp-delete-lease-heartbeat")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def assert_healthy(self) -> None:
        if self._failure is not None:
            raise self._failure

    async def _run(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._interval_seconds)
                await self._renew_all()
        except asyncio.CancelledError:
            raise
        except McpDeleteLeaseLost as exc:
            self._failure = exc
        except Exception as exc:
            logger.warning("[MCP_DELETE_LEASE] heartbeat_failed error=%s", exc)
            self._failure = McpDeleteLeaseLost("MCP deletion lease heartbeat failed")

    async def _renew_all(self) -> None:
        await heartbeat_mcp_lease_worker(self._storage)
        if self._package_reservation_id:
            for server_id in self._package_server_ids:
                if not await renew_mcp_package_deletion(
                    self._storage,
                    self._tenant_id,
                    server_id,
                    reservation_id=self._package_reservation_id,
                ):
                    raise McpDeleteLeaseLost("MCP package deletion lease was lost")
        if self._image_reservation_id:
            for doc_id in self._image_doc_ids:
                if not await renew_mcp_built_image_deletion(
                    self._storage,
                    doc_id,
                    reservation_id=self._image_reservation_id,
                ):
                    raise McpDeleteLeaseLost("MCP image deletion lease was lost")
