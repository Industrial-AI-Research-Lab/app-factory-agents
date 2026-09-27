"""
Snapshot Manager

DEPRECATED (State Simplification 2.4.5):
This module is being phased out in favor of:
- Messages (unified append-only log with atomic sequences)
- file_artifacts with checkpoint_sequence markers

New code should use:
- MessageStore for conversation/state tracking
- ArtifactStore.save_file() (stamps checkpoint_sequence on every write)
- ArtifactStore.restore_files_to_checkpoint() for revert

This module is kept for backward compatibility with existing snapshots.

Original purpose:
Creates and restores project snapshots (SharedContext + repo commit),
and coordinates environment reset/recreate during revert.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from context.shared_context import SharedContext
from schemas import EventSchema


class SnapshotManager:
    def __init__(self, storage, container_manager, event_emitter, message_store=None):
        self.storage = storage
        self.container_manager = container_manager
        self.event_emitter = event_emitter
        self.message_store = message_store

    async def create_snapshot(
        self,
        project_id: str,
        shared_context: SharedContext,
        *,
        snap_type: str,
        label: Optional[str] = None,
        phase: Optional[str] = None,
        event_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Capture the checkpoint moment BEFORE the git commit below, which can
        # take tens of seconds. created_at (stamped at save time, after the
        # commit) would otherwise sit past any user message typed during the
        # commit, inflating the revert anchor past this checkpoint. Read via
        # snapshot_anchor.snapshot_boundary.
        checkpoint_at = datetime.now(timezone.utc).isoformat()
        # Export state
        state = shared_context.export_state()
        # Record current repo commit if available
        git_commit = None
        try:
            # Determine current repo name from container status (fallback to branch name)
            repo_name = f"AppFactory-{project_id}"
            try:
                status = await self.container_manager.get_container_status(project_id)
                repo_path = (status or {}).get("repo_path")
                if repo_path:
                    from pathlib import Path as _Path
                    repo_name = _Path(repo_path).name
            except Exception:
                pass
            repo = self.container_manager.repo_manager
            git_commit = await asyncio.to_thread(repo.commit_snapshot, repo_name, label or snap_type)
        except Exception:
            git_commit = None
        run_id = getattr(shared_context, "run_id", None)
        meta = dict(meta or {})
        meta.setdefault("checkpoint_at", checkpoint_at)
        # Persist snapshot
        snapshot_id = await self.storage.save_snapshot(project_id, {
            "type": snap_type,
            "label": label or snap_type,
            "phase": phase,
            "event_id": event_id,
            "git_commit": git_commit,
            "context_state": state,
            "meta": meta,
            "run_id": run_id,
        })
        # Emit lifecycle event (fire-and-forget; UI updates list).
        # run_id from shared_context — snapshots are run-scoped (a snapshot
        # belongs to the run that created it).
        try:
            await self.event_emitter.emit(EventSchema.SNAPSHOT_CREATED, run_id, {
                "project_id": project_id,
                "snapshot_id": snapshot_id,
                "type": snap_type,
                "label": label or snap_type,
                "phase": phase,
            })
        except Exception:
            pass
        return snapshot_id

    async def list_snapshots(self, project_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        return await self.storage.list_snapshots(project_id, limit=limit)

    async def resolve_latest_auction(self, project_id: str) -> Optional[Dict[str, Any]]:
        return await self.storage.get_latest_snapshot_by_types(project_id, [
            "auction_started", "auction_completed"
        ])

    async def restore_snapshot(self, project_id: str, snapshot_id: str) -> Dict[str, Any]:
        snap = await self.storage.get_snapshot(snapshot_id)
        if not snap:
            raise RuntimeError("Snapshot not found")
        # Run-less deliberately: this context only loads, imports and syncs
        # state, then is discarded — it never reaches a runner. The live
        # context is rebuilt by the caller (revert_manager).
        sc = SharedContext.runless(project_id, self.storage, message_store=self.message_store)
        await sc.load_from_db()
        await sc.import_state(snap.get("context_state") or {})
        # Reset environment
        try:
            await self.container_manager.reset_environment(project_id)
        except Exception:
            pass
        # Snapshot restore is a new attempt: drop the timeout lockout and the
        # preserved workspace so get_or_create below neither refuses nor rebuilds
        # from the post-timeout environment branch.
        try:
            self.container_manager.clear_session_unavailable(project_id)
        except Exception:
            pass
        try:
            clear_preserved = getattr(
                self.container_manager, "clear_preserved_workspace", None
            )
            if clear_preserved is not None:
                clear_preserved(project_id)
        except Exception:
            pass
        # Checkout repo commit if provided
        commit = snap.get("git_commit")
        if commit:
            try:
                # Determine current repo name
                repo_name = f"AppFactory-{project_id}"
                try:
                    status = await self.container_manager.get_container_status(project_id)
                    repo_path = (status or {}).get("repo_path")
                    if repo_path:
                        from pathlib import Path as _Path
                        repo_name = _Path(repo_path).name
                except Exception:
                    pass
                repo = self.container_manager.repo_manager
                await asyncio.to_thread(repo.checkout_commit, repo_name, commit)
            except Exception:
                # If checkout fails, continue with context-only rollback
                pass
        env_info = await self.container_manager.get_or_create_container(project_id)
        return {"snapshot": snap, "environment": env_info}
