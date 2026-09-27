"""Durable local outbox for A2A cancellation intents.

MongoDB is the system of record for normal workflow recovery, but it may be
unavailable precisely when a running remote task needs cancelling. This WAL is
written before the first ``tasks/cancel`` request and survives an API restart.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


class A2ACancellationWal:
    """SQLite WAL backed cancellation intent store with process-safe writes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    async def record_intent(self, **intent: Any) -> Dict[str, Any]:
        """Synchronously commit one cancellation intent before remote I/O."""
        return await asyncio.to_thread(self._record_intent_sync, intent)

    async def list_open_intents(self) -> List[Dict[str, Any]]:
        """Return every intent that still needs Mongo or remote reconciliation."""
        return await asyncio.to_thread(self._list_open_intents_sync)

    async def mark_mongo_pending(self, intent: Dict[str, Any]) -> None:
        """Record that MongoDB now owns a cancellation-pending cursor."""
        await asyncio.to_thread(self._mark_state_sync, intent, "mongo_pending", None)

    async def mark_remote_terminal(
        self, intent: Dict[str, Any], *, final_status: str
    ) -> None:
        """Keep the intent until the corresponding Mongo cursor closes."""
        await asyncio.to_thread(
            self._mark_state_sync, intent, "remote_terminal", final_status
        )

    async def mark_closed(self, intent: Dict[str, Any]) -> None:
        """Delete an intent only after MongoDB confirms the cursor is closed."""
        await asyncio.to_thread(self._mark_closed_sync, intent)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS a2a_cancellation_intents (
                project_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                server_id TEXT NOT NULL,
                context_id TEXT,
                reason TEXT NOT NULL,
                state TEXT NOT NULL,
                final_status TEXT,
                PRIMARY KEY (project_id, run_id, node_id, task_id)
            )
            """
        )
        return connection

    def _record_intent_sync(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO a2a_cancellation_intents (
                    project_id, run_id, node_id, task_id, tenant_id, server_id,
                    context_id, reason, state, final_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'wal_pending', NULL)
                ON CONFLICT(project_id, run_id, node_id, task_id) DO NOTHING
                """,
                self._identity_values(intent) + (
                    intent["tenant_id"],
                    intent["server_id"],
                    intent.get("context_id"),
                    intent["reason"],
                ),
            )
            row = connection.execute(
                """
                SELECT project_id, run_id, node_id, tenant_id, server_id, task_id,
                       context_id, reason, state, final_status
                FROM a2a_cancellation_intents
                WHERE project_id = ? AND run_id = ? AND node_id = ? AND task_id = ?
                """,
                self._identity_values(intent),
            ).fetchone()
            return self._row_to_intent(row)
        finally:
            connection.close()

    def _list_open_intents_sync(self) -> List[Dict[str, Any]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT project_id, run_id, node_id, tenant_id, server_id, task_id,
                       context_id, reason, state, final_status
                FROM a2a_cancellation_intents
                WHERE state != 'closed'
                ORDER BY project_id, run_id, node_id, task_id
                """
            ).fetchall()
            return [self._row_to_intent(row) for row in rows]
        finally:
            connection.close()

    def _mark_state_sync(
        self,
        intent: Dict[str, Any],
        state: str,
        final_status: str | None,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                UPDATE a2a_cancellation_intents
                SET state = ?, final_status = ?
                WHERE project_id = ? AND run_id = ? AND node_id = ? AND task_id = ?
                """,
                (state, final_status, *self._identity_values(intent)),
            )
        finally:
            connection.close()

    def _mark_closed_sync(self, intent: Dict[str, Any]) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                DELETE FROM a2a_cancellation_intents
                WHERE project_id = ? AND run_id = ? AND node_id = ? AND task_id = ?
                """,
                self._identity_values(intent),
            )
        finally:
            connection.close()

    @staticmethod
    def _identity_values(intent: Dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            intent["project_id"],
            intent.get("run_id") or "",
            intent["node_id"],
            intent["task_id"],
        )

    @staticmethod
    def _row_to_intent(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "project_id": row["project_id"],
            "run_id": row["run_id"] or None,
            "node_id": row["node_id"],
            "tenant_id": row["tenant_id"],
            "server_id": row["server_id"],
            "task_id": row["task_id"],
            "context_id": row["context_id"],
            "reason": row["reason"],
            "state": row["state"],
            **({"final_status": row["final_status"]}
               if row["final_status"] is not None else {}),
        }
