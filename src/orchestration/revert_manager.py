"""
Revert Manager

Handles project reversion:
- Revert to message by sequence number (PRIMARY - uses unified message system)
- Revert to snapshot by ID or tags (LEGACY - being deprecated)
- Revert to user message by conversation index (LEGACY - uses snapshots)
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import logging

from context.shared_context import SharedContext
from schemas import EventSchema
from storage.message_store import TOOL_LEDGER_TYPES
from storage.snapshot_anchor import compute_target_sequence, snapshot_boundary

logger = logging.getLogger(__name__)


class RevertManager:
    """
    Manages project reversion using snapshots.
    
    Reverts restore SharedContext, container state, and re-seed approval gates.
    """
    
    def __init__(
        self,
        storage_backend,
        event_emitter,
        snapshot_manager,
        container_manager=None,
        message_store=None,
        artifact_store=None,
    ):
        self.storage = storage_backend
        self.event_emitter = event_emitter
        self.snapshot_manager = snapshot_manager
        self.container_manager = container_manager
        self.message_store = message_store
        self.artifact_store = artifact_store

    async def _cancel_and_wait_for_workflow(
        self,
        project_id: str,
        cancel_project_fn: callable,
        reason: str,
        log_tag: str,
        wait_for_task_completion_fn: Optional[callable] = None,
        timeout: float = 10.0,
    ) -> None:
        """Cancel the active workflow and wait briefly before mutating restored state."""
        await cancel_project_fn(project_id, reason=reason)
        if not wait_for_task_completion_fn:
            return

        try:
            stopped = await wait_for_task_completion_fn(project_id, timeout=timeout)
            if not stopped:
                logger.warning("[%s] Workflow task did not stop within %.1fs, proceeding with revert", log_tag, timeout)
        except asyncio.CancelledError:
            logger.info("[%s] Workflow task reported cancelled during revert wait", log_tag)
        except Exception as e:
            logger.warning("[%s] Error waiting for workflow task stop: %s", log_tag, e)

    async def _clear_stale_resume_blocked_after_revert(
        self,
        project_id: str,
        project_state: Dict,
    ) -> None:
        """Revert restores a safe journal; drop stale resume_blocked on the run doc."""
        run_id = project_state.get("run_id")
        if not run_id:
            try:
                run = await self.storage.get_active_run(project_id)
                run_id = (run or {}).get("run_id")
            except Exception as exc:
                logger.warning(
                    "[REVERT] project_id=%s resume_blocked clear skipped — run lookup: %s",
                    project_id,
                    exc,
                )
                return
        if not run_id:
            return
        try:
            await self.storage.clear_run_resume_blocked(run_id)
            logger.info(
                "[REVERT] project_id=%s run_id=%s cleared stale resume_blocked after revert",
                project_id,
                run_id,
            )
        except Exception as exc:
            logger.warning(
                "[REVERT] project_id=%s run_id=%s clear resume_blocked failed: %s",
                project_id,
                run_id,
                exc,
            )

    async def _restore_run_running_after_revert(
        self,
        project_id: str,
        project_state: Dict,
    ) -> None:
        """Revert uses cancel_project for ephemeral stop; run must return to running."""
        run_id = project_state.get("run_id")
        if not run_id:
            try:
                run = await self.storage.get_active_run(project_id)
                run_id = (run or {}).get("run_id")
            except Exception as exc:
                logger.warning(
                    "[REVERT] project_id=%s run restore skipped — run lookup: %s",
                    project_id,
                    exc,
                )
                return
        if not run_id:
            return
        try:
            await self.storage.update_run_status(run_id, "running")
            logger.info(
                "[REVERT] project_id=%s run_id=%s restored run_status=running after revert",
                project_id,
                run_id,
            )
        except Exception as exc:
            logger.warning(
                "[REVERT] project_id=%s run_id=%s restore run_status failed: %s",
                project_id,
                run_id,
                exc,
            )

    async def _finalize_revert_durable_state(
        self,
        project_id: str,
        project_state: Dict,
    ) -> None:
        await self._clear_stale_resume_blocked_after_revert(project_id, project_state)
        await self._restore_run_running_after_revert(project_id, project_state)
    
    async def resolve_latest_by_tags(self, project_id: str, required_tags: List[str]) -> Optional[Dict[str, Any]]:
        """Find latest snapshot matching all required tags."""
        snaps = await self.snapshot_manager.list_snapshots(project_id, limit=200)
        req = set(required_tags or [])
        for s in snaps:
            tags = set((s.get("meta", {}) or {}).get("tags", []) or [])
            if req.issubset(tags):
                return s
        return None

    async def _unsafe_truncation_reason(
        self, project_id: str, snap: Dict[str, Any], conversation_index: int
    ) -> Optional[str]:
        """Why reverting on this snapshot would truncate past its user checkpoint
        (so it must be refused before any mutation), or None when the coordinate
        is safe to truncate at.

        conversation_index is what truncate_conversation keeps up to and deletes
        past. The checkpoint is the highest type="user" message at or before the
        snapshot's checkpoint moment (snapshot_boundary: meta.checkpoint_at, else
        created_at) — the same coordinate the backfill migration computes. A revert may drop everything after the checkpoint, including
        free-form chat typed later (that is what revert-to-checkpoint is for), so
        a coordinate at or above the checkpoint is safe. Two are not:
          - below a real checkpoint (anchor > 0): truncating deletes the
            checkpoint and the initial prompt — the stale-index bug the backfill
            repairs;
          - below sequence 1: sequences are 1-based, so index 0 keeps nothing and
            wipes the whole run. When anchor is also 0 (no user message precedes
            the snapshot — a run whose anchor was already wiped, or broken legacy)
            the first check's anchor>0 guard can't catch it, so index 0 is caught
            explicitly here.
        compute_target_sequence returns None ONLY for an unparseable snapshot
        timestamp — a genuine can't-compute that fails open, kept distinct from an
        anchor of 0, so a data quirk never blocks a legitimate revert. A failed
        message lookup, by contrast, fails closed (returns a reason): we could not
        check safety at all, which is not the same as checking and finding none.
        """
        if not self.message_store:
            return None
        try:
            user_messages = await self.message_store.get_messages(
                project_id, only_types=["user"], limit=10000
            )
        except Exception as e:
            logger.warning("[REVERT] stale-index guard lookup failed for %s: %s", project_id, e)
            # A lookup error means we could not VERIFY safety — refuse, don't
            # proceed. Failing open here would let a transient Mongo/timeout error
            # wave through a stale index-0 snapshot and wipe the run (the exact
            # incident this guard prevents). Distinct from the anchor-None case
            # below, which is a genuine can't-compute on already-fetched data.
            return (
                "could not verify the snapshot's user checkpoint (message lookup "
                f"failed: {e}); refusing to revert rather than risk truncating the "
                "run — retry"
            )
        anchor = compute_target_sequence(snapshot_boundary(snap), user_messages)
        if anchor is None:
            return None
        if anchor > 0 and conversation_index < anchor:
            return (
                f"conversation_index {conversation_index} is behind the snapshot's user "
                f"checkpoint at sequence {anchor}; reverting would delete the checkpoint "
                "and earlier messages — refusing (stale snapshot; run the "
                "conversation_index backfill)"
            )
        if conversation_index < 1:
            return (
                f"conversation_index {conversation_index} is not a valid checkpoint "
                "(message sequences are 1-based); reverting would truncate the entire "
                "run — refusing (unrepairable legacy snapshot; no user checkpoint "
                "precedes it)"
            )
        return None

    async def _refuse_revert(
        self, project_id: str, run_id: Optional[str], reason: str, log_tag: str,
    ) -> Dict[str, Any]:
        await self.event_emitter.emit(
            EventSchema.PROJECT_REVERT_FAILED,
            run_id,
            {"project_id": project_id, "error": reason, "timestamp": datetime.utcnow().isoformat()},
        )
        logger.error("[%s] %s (project=%s)", log_tag, reason, project_id)
        return {"status": "failed", "error": reason}

    async def revert_project(
        self,
        project_id: str,
        target: Dict[str, Any],
        project_state: Dict,
        reinstate_runtime_fn: callable,
        cancel_project_fn: callable,
        approval_manager,
        wait_for_task_completion_fn: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Revert project to a snapshot target."""
        # Resolve snapshot by ID or by tags (user_action default), fallback to baseline
        await self.event_emitter.emit(EventSchema.PROJECT_REVERTING, project_state.get("run_id"), {"project_id": project_id, "requested_target": target})
        
        snap = None
        ttype = (target or {}).get("type")
        if ttype == "snapshot":
            sid = (target or {}).get("id")
            if sid:
                snap = await self.storage.get_snapshot(sid)
        elif ttype == "tags":
            tags = (target or {}).get("all") or ["user_action"]
            snap = await self.resolve_latest_by_tags(project_id, tags)
        elif ttype == "latest_auction":  # deprecated
            snap = await self.resolve_latest_by_tags(project_id, ["user_action"])
        
        # Fallback to baseline
        if not snap:
            snap = await self.resolve_latest_by_tags(project_id, ["system_checkpoint"])
        
        # As an absolute last resort, create a fresh baseline now
        if not snap:
            sc_tmp = project_state.get("shared_context")
            try:
                await self.snapshot_manager.create_snapshot(
                    project_id,
                    sc_tmp,
                    snap_type="project_started",
                    label="project started",
                    phase="initialization",
                    meta={"tags": ["system_checkpoint"]},
                )
                snap = await self.resolve_latest_by_tags(project_id, ["system_checkpoint"])
            except Exception:
                pass
        
        if not snap:
            await self.event_emitter.emit(EventSchema.PROJECT_REVERT_FAILED, project_state.get("run_id"), {"project_id": project_id, "error": "No suitable snapshot found"})
            return {"status": "failed", "error": "No suitable snapshot found"}
        
        # No active run means the project is corrupted state (revert keeps the
        # same run, and the rebuilt context below must carry its run_id — a
        # run-less context turns the tool ledger off for everything after the
        # revert). Fail BEFORE restore_snapshot: it resets the container and
        # checks out the repo, mutations we must not make for a doomed revert.
        # The check deliberately sits OUTSIDE the best-effort try below.
        active_run = await self.storage.get_active_run(project_id)
        active_run_id = (active_run or {}).get("run_id")
        if not active_run_id:
            await self.event_emitter.emit(EventSchema.PROJECT_REVERT_FAILED, project_state.get("run_id"), {"project_id": project_id, "error": "No active run — refusing to revert"})
            return {"status": "failed", "error": "No active run — refusing to revert"}

        # Same stale-coordinate guard as stop-and-revert-latest, before restore.
        # The Snapshots-tab revert (target.type="snapshot") truncates on this
        # snapshot's stored conversation_index further below; if that value is
        # behind the snapshot's user checkpoint (legacy 0 / stale list index) the
        # truncation wipes the run, so refuse here rather than mutate.
        _meta = snap.get("meta") or {}
        if "conversation_index" in _meta:
            try:
                _idx = int(_meta["conversation_index"])
            except (TypeError, ValueError):
                _idx = None
            if _idx is not None:
                reason = await self._unsafe_truncation_reason(project_id, snap, _idx)
                if reason is not None:
                    return await self._refuse_revert(
                        project_id, project_state.get("run_id"), reason, "REVERT",
                    )

        # Every read-only refusal gate has passed — only NOW stop the workflow,
        # right before the restore/truncate mutations. Cancelling earlier killed
        # the in-flight run for reverts we still went on to refuse.
        await self._cancel_and_wait_for_workflow(
            project_id,
            cancel_project_fn,
            reason="Reverting",
            log_tag="REVERT",
            wait_for_task_completion_fn=wait_for_task_completion_fn,
        )

        # Restore
        try:
            logger.info("[REVERT] restore.begin project_id=%s snapshot_id=%s type=%s phase=%s", project_id, snap.get("id"), snap.get("type"), snap.get("phase"))
        except Exception:
            pass

        await self.snapshot_manager.restore_snapshot(project_id, snap.get("id"))

        # Refresh in-memory SharedContext on the run checked above.
        try:
            sc = SharedContext(
                project_id,
                self.storage,
                run_id=active_run_id,
                message_store=self.message_store,
            )
            await sc.load_from_db()
            project_state["shared_context"] = sc
        except Exception:
            pass
        
        # Reinstate runtime
        try:
            reinstate_runtime_fn(project_id, project_state["shared_context"])
        except Exception:
            pass
        
        # Phase comes straight from the snapshot — no string normalization.
        # The reverted FSM, once spawned by ensure_workflow_running, will
        # decide what to do based on shared_context + pending approvals,
        # not on the value stored here. We keep it for UI display.
        try:
            new_phase = snap.get("phase") or "general"
            project_state["current_phase"] = new_phase
            project_state["status"] = "initialized"

            await self.storage.save_project(project_id, {
                "user_prompt": project_state.get("user_prompt", ""),
                "title": project_state.get("title", "Untitled Project"),
                "status": project_state.get("status", "initialized"),
                "current_phase": new_phase,
                "approval_mode": project_state.get("approval_mode", "human"),
                "created_at": project_state.get("created_at"),
                "metadata": project_state.get("metadata", {})
            })
            logger.info("[REVERT] restore.saved project_id=%s new_phase=%s", project_id, new_phase)
        except Exception:
            pass
        
        # Handle prefill from snapshot
        prefill = None
        try:
            meta = (snap.get("meta") or {})
            if "conversation_index" in meta:
                idx = int(meta["conversation_index"])
                await project_state["shared_context"].truncate_conversation(before_index=idx)
            prefill = meta.get("input_prefill")
        except Exception:
            pass
        
        # Save prefill to metadata
        try:
            if isinstance(prefill, str) and prefill.strip():
                md = project_state.get("metadata")
                if not isinstance(md, dict):
                    md = {}
                md = {**md, "last_revert_prefill": prefill}
                project_state["metadata"] = md
                await self.storage.save_project(project_id, {
                    "user_prompt": project_state.get("user_prompt", ""),
                    "title": project_state.get("title", "Untitled Project"),
                    "status": project_state.get("status", "initialized"),
                    "current_phase": project_state.get("current_phase"),
                    "approval_mode": project_state.get("approval_mode", "human"),
                    "created_at": project_state.get("created_at"),
                    "metadata": md,
                })
        except Exception:
            pass
        
        # Cancel pending approvals BEFORE the flip-back re-seed below
        # (ADR 0007). The scan in `_reseed_last_pre_snapshot_approval`
        # picks the latest pre-snapshot approval row regardless of
        # status — so the fence isn't what helps the scan find the right
        # row (the time-bound does that). Its job is to ensure that
        # AFTER the flip-back, the row we just flipped is the ONLY
        # pending row for this project — without the fence, a stale
        # pending row from a prior failed workflow would survive
        # alongside it and the FE would render two amber cards.
        # Belt-and-braces: also clears in-memory pending_approvals so
        # request_approval's dedup loop doesn't reference a ghost.
        approval_manager.clear_project_approvals(project_id)
        if self.message_store:
            try:
                cancelled_count = await self.message_store.cancel_pending_approvals_for_project(project_id)
                logger.info("[REVERT] Cancelled %d pending approvals in messages", cancelled_count)
            except Exception as e:
                logger.warning("[REVERT] Failed to cancel message approvals: %s", e)

        await self.event_emitter.emit(EventSchema.PROJECT_REVERTED, project_state.get("run_id"), {
            "project_id": project_id,
            "snapshot_id": snap.get("id"),
            "label": snap.get("label"),
            "tags": (snap.get("meta") or {}).get("tags"),
            "prefill_input": prefill,
        })

        # Re-seed the approval gate the user is rewinding past (ADR 0007).
        # truncate_conversation above (line 186) ran only if
        # snap.meta.conversation_index was present — otherwise the
        # message store is whole. Either way, surviving rows can still
        # include approvals created after `snap.created_at` (append-time
        # isn't strictly ordered with truncation depth), so the scan in
        # `_reseed_last_pre_snapshot_approval` filters those out via the
        # created_at bound. No-op if no pre-snapshot approval row exists
        # (e.g., reverting to a project_started snapshot before any gate).
        await self._reseed_last_pre_snapshot_approval(
            project_id, project_state, approval_manager, snap
        )

        await self._finalize_revert_durable_state(project_id, project_state)
        return {"status": "ok", "snapshot": snap}

    async def revert_to_user_message(
        self,
        project_id: str,
        conversation_index: int,
        project_state: Dict,
        reinstate_runtime_fn: callable,
        cancel_project_fn: callable,
        approval_manager,
        wait_for_task_completion_fn: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Revert project to the state at a specific user message checkpoint."""
        revert_start_ts = datetime.utcnow().isoformat()
        
        # Resolve snapshot by conversation_index
        snap = await self.storage.get_user_snapshot_by_conversation_index(project_id, conversation_index)
        if not snap and conversation_index == 0:
            try:
                snap = await self.resolve_latest_by_tags(project_id, ["system_checkpoint"])
            except Exception:
                snap = None
        
        if not snap:
            await self.event_emitter.emit(
                EventSchema.PROJECT_REVERT_FAILED,
                project_state.get("run_id"),
                {"project_id": project_id, "error": "No user_message snapshot found", "timestamp": revert_start_ts},
            )
            return {"status": "failed", "error": "No user_message snapshot found"}

        await self.event_emitter.emit(
            EventSchema.PROJECT_REVERTING,
            project_state.get("run_id"),
            {
                "project_id": project_id,
                "requested_target": {"type": "user_message", "conversation_index": conversation_index},
                "snapshot_id": snap.get("id"),
                "timestamp": revert_start_ts,
            },
        )

        # No active run = corrupted state (same-run revert: the rebuilt
        # context below must carry the run's id or the tool ledger dies for
        # the rest of the run). Fail BEFORE restore_snapshot: it resets the
        # container and checks out the repo — mutations we must not make for
        # a doomed revert.
        active_run = await self.storage.get_active_run(project_id)
        active_run_id = (active_run or {}).get("run_id")
        if not active_run_id:
            await self.event_emitter.emit(
                EventSchema.PROJECT_REVERT_FAILED,
                project_state.get("run_id"),
                {"project_id": project_id, "error": "No active run — refusing to revert", "timestamp": revert_start_ts},
            )
            return {"status": "failed", "error": "No active run — refusing to revert"}

        # Guard the destructive truncation before any mutation: refuse when the
        # stored coordinate is behind the snapshot's own user checkpoint (the
        # index-0 / stale list-index bug), which would truncate past the
        # checkpoint into the initial prompt and wipe the run. A valid coordinate
        # — even one with later free-form user chat above it — passes and is
        # truncated normally. The backfill migration repairs stale coordinates.
        reason = await self._unsafe_truncation_reason(project_id, snap, conversation_index)
        if reason is not None:
            return await self._refuse_revert(
                project_id, project_state.get("run_id"), reason, "REVERT_USER",
            )

        # Every read-only refusal gate (missing snapshot, no active run, unsafe
        # truncation) has passed — only NOW stop the workflow, right before the
        # restore/truncate mutations. Cancelling any earlier killed the in-flight
        # run for reverts we still went on to refuse (dominant on stale/legacy
        # snapshots), leaving the project stopped-but-not-reverted.
        await self._cancel_and_wait_for_workflow(
            project_id,
            cancel_project_fn,
            reason="Reverting to user_message",
            log_tag="REVERT_USER",
            wait_for_task_completion_fn=wait_for_task_completion_fn,
        )

        # Restore snapshot
        logger.info("[REVERT_USER] restore.begin project_id=%s snapshot_id=%s", project_id, snap.get("id"))
        await self.snapshot_manager.restore_snapshot(project_id, snap.get("id"))

        # Refresh in-memory SharedContext on the run checked above.
        sc = SharedContext(
            project_id,
            self.storage,
            run_id=active_run_id,
            message_store=self.message_store,
        )
        await sc.load_from_db()
        project_state["shared_context"] = sc

        # Truncate conversation
        try:
            await sc.truncate_conversation(before_index=conversation_index)
        except Exception:
            pass

        # Reinstate runtime
        reinstate_runtime_fn(project_id, project_state["shared_context"])

        # Trim events and clear approvals
        try:
            created_at = snap.get("created_at")
            since_dt = None
            if isinstance(created_at, str):
                try:
                    since_dt = datetime.fromisoformat(created_at)
                except Exception:
                    since_dt = None
            elif isinstance(created_at, datetime):
                since_dt = created_at
            if since_dt is not None:
                await self.storage.delete_events_since(project_id, since_dt)
        except Exception:
            pass
        
        # Note: delete_approvals removed - approvals are now in messages collection
        
        # Restore files to checkpoint (State Simplification 2.4.4)
        try:
            if self.artifact_store and self.message_store:
                # Find the message sequence for this conversation_index
                # Use the snapshot's conversation_index from meta
                meta = snap.get("meta") or {}
                checkpoint_seq = meta.get("conversation_index", conversation_index)
                run_id = project_state.get("run_id")
                await self.artifact_store.restore_files_to_checkpoint(
                    project_id, checkpoint_seq, run_id
                )
                logger.info(f"[REVERT] Files restored to checkpoint sequence {checkpoint_seq}")
        except Exception as e:
            logger.warning(f"[REVERT] Failed to restore files to checkpoint: {e}")
        
        # Clear in-memory pending approvals and cancel in messages collection
        approval_manager.clear_project_approvals(project_id)
        if self.message_store:
            try:
                cancelled_count = await self.message_store.cancel_pending_approvals_for_project(project_id)
                logger.info(f"[REVERT] Cancelled {cancelled_count} pending approvals in messages")
            except Exception as e:
                logger.warning(f"[REVERT] Failed to cancel message approvals: {e}")

        # Phase from snapshot, no string normalization. State cleanup
        # (zeroing artifacts/requirements/plan) intentionally removed: the
        # reverted FSM will replay declared writes from the source of truth
        # (agent outputs); state-strings here were a workaround for the old
        # linear sequencer and now lie to the engine about what's present.
        new_phase = snap.get("phase") or "general"
        project_state["current_phase"] = new_phase
        project_state["status"] = "initialized"

        # Persist project summary
        await self.storage.save_project(
            project_id,
            {
                "user_prompt": project_state.get("user_prompt", ""),
                "title": project_state.get("title", "Untitled Project"),
                "status": project_state.get("status", "initialized"),
                "current_phase": new_phase,
                "approval_mode": project_state.get("approval_mode", "human"),
                "created_at": project_state.get("created_at"),
                "metadata": project_state.get("metadata", {}),
            },
        )

        # Emit reverted event with prefill metadata
        prefill = None
        try:
            meta = snap.get("meta") or {}
            prefill = meta.get("input_prefill")
            if not prefill and conversation_index == 0:
                prefill = project_state.get("user_prompt")
        except Exception:
            pass
        
        # Save prefill
        try:
            if isinstance(prefill, str) and prefill.strip():
                md = project_state.get("metadata")
                if not isinstance(md, dict):
                    md = {}
                md = {**md, "last_revert_prefill": prefill}
                project_state["metadata"] = md
                await self.storage.save_project(
                    project_id,
                    {
                        "user_prompt": project_state.get("user_prompt", ""),
                        "title": project_state.get("title", "Untitled Project"),
                        "status": project_state.get("status", "initialized"),
                        "current_phase": project_state.get("current_phase"),
                        "approval_mode": project_state.get("approval_mode", "human"),
                        "created_at": project_state.get("created_at"),
                        "metadata": md,
                    },
                )
        except Exception:
            pass
        
        await self.event_emitter.emit(
            EventSchema.PROJECT_REVERTED,
            project_state.get("run_id"),
            {
                "project_id": project_id,
                "snapshot_id": snap.get("id"),
                "label": snap.get("label"),
                "tags": (snap.get("meta") or {}).get("tags"),
                "prefill_input": prefill,
            },
        )

        # Re-seed the approval gate the user is rewinding past (ADR 0007).
        # truncate_conversation above (line 307) already removed messages
        # past `conversation_index`, but surviving rows can still include
        # approvals created after `snap.created_at` — anything whose
        # sequence falls inside the surviving range but whose append-time
        # post-dates the snapshot. The scan in
        # `_reseed_last_pre_snapshot_approval` filters those out via the
        # created_at bound, picks the latest pre-snapshot approval
        # regardless of status, and the cancel-fence above ensures no
        # other pending row competes with the one we flip back.
        await self._reseed_last_pre_snapshot_approval(
            project_id, project_state, approval_manager, snap
        )

        await self._finalize_revert_durable_state(project_id, project_state)
        return {"status": "ok", "snapshot": snap}

    async def revert_to_latest_user_message(
        self,
        project_id: str,
        project_state: Dict,
        reinstate_runtime_fn: callable,
        cancel_project_fn: callable,
        approval_manager,
        wait_for_task_completion_fn: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Revert project to the most recent user_message checkpoint."""
        snap = await self.storage.get_latest_user_snapshot(project_id)
        if not snap:
            await self.event_emitter.emit(
                EventSchema.PROJECT_REVERT_FAILED,
                project_state.get("run_id"),
                {"project_id": project_id, "error": "No user_message snapshots found"},
            )
            return {"status": "failed", "error": "No user_message snapshots found"}

        try:
            meta = snap.get("meta") or {}
            conv_index = int(meta.get("conversation_index"))
        except Exception:
            return {"status": "failed", "error": "Latest user snapshot missing conversation_index"}

        return await self.revert_to_user_message(
            project_id,
            conv_index,
            project_state,
            reinstate_runtime_fn,
            cancel_project_fn,
            approval_manager,
            wait_for_task_completion_fn=wait_for_task_completion_fn,
        )

    # =========================================================================
    # MESSAGE-BASED REVERT (Primary - State Simplification 2.4.4)
    # =========================================================================
    
    async def revert_to_message_sequence(
        self,
        project_id: str,
        target_sequence: int,
        project_state: Dict,
        cancel_project_fn: callable,
        approval_manager,
        reinstate_runtime_fn: Optional[callable] = None,
        wait_for_task_completion_fn: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Revert to a specific message sequence (PRIMARY revert method).
        
        This uses the unified message system directly:
        1. Delete messages after target_sequence
        2. Restore files to checkpoint at target_sequence
        3. Determine phase from remaining messages
        4. Seed appropriate approval gate
        
        Args:
            project_id: Project ID
            target_sequence: Message sequence to revert to
            project_state: Active project state dict
            cancel_project_fn: Function to cancel running workflow
            approval_manager: ApprovalManager instance
        """
        revert_start_ts = datetime.utcnow().isoformat()
        
        # Stop current workflow
        await self._cancel_and_wait_for_workflow(
            project_id,
            cancel_project_fn,
            reason=f"Reverting to message sequence {target_sequence}",
            log_tag="REVERT_MSG",
            wait_for_task_completion_fn=wait_for_task_completion_fn,
        )
        
        # Get the target message to determine type and prefill content
        target_message = None
        prefill = ""
        is_user_message = False
        if self.message_store:
            try:
                messages = await self.message_store.get_messages(
                    project_id, limit=1000, exclude_types=list(TOOL_LEDGER_TYPES)
                )
                for msg in messages:
                    if msg.get("sequence") == target_sequence:
                        target_message = msg
                        is_user_message = msg.get("type") == "user"
                        # Only prefill for user messages that aren't approval actions
                        msg_content = msg.get("content", "").strip().lower()
                        is_approval_action = (
                            msg_content.startswith("[approve]") or
                            msg_content == "approved"
                        )
                        if is_user_message and not is_approval_action:
                            prefill = msg.get("content", "")
                        # For approval actions, no prefill (Approve button handles it)
                        break
            except Exception as e:
                logger.warning(f"[REVERT_MSG] Failed to get target message: {e}")
        
        if not target_message:
            await self.event_emitter.emit(
                EventSchema.PROJECT_REVERT_FAILED,
                project_state.get("run_id"),
                {"project_id": project_id, "error": f"Message sequence {target_sequence} not found", "timestamp": revert_start_ts},
            )
            return {"status": "failed", "error": f"Message sequence {target_sequence} not found"}

        await self.event_emitter.emit(
            EventSchema.PROJECT_REVERTING,
            project_state.get("run_id"),
            {
                "project_id": project_id,
                "requested_target": {"type": "message_sequence", "sequence": target_sequence},
                "timestamp": revert_start_ts,
            },
        )
        
        logger.info(f"[REVERT_MSG] Starting revert project={project_id} target_sequence={target_sequence}")

        # A revert starts a new attempt: drop the timeout lockout and the
        # preserved workspace so the next tool call seeds a sandbox and restores
        # the checkpoint from the DB.
        if self.container_manager is not None:
            try:
                self.container_manager.clear_session_unavailable(project_id)
            except Exception as e:
                logger.warning(
                    "[REVERT_MSG] clear_session_unavailable failed project=%s: %s",
                    project_id,
                    e,
                )
            try:
                clear_preserved = getattr(
                    self.container_manager, "clear_preserved_workspace", None
                )
                if clear_preserved is not None:
                    clear_preserved(project_id)
            except Exception as e:
                logger.warning(
                    "[REVERT_MSG] clear_preserved_workspace failed project=%s: %s",
                    project_id,
                    e,
                )
        
        # 1. Delete messages from target sequence (including target - it goes to prefill)
        # NOTE: Do NOT filter by run_id - revert should delete ALL messages from target
        # regardless of which run they belong to
        deleted_count = 0
        if self.message_store:
            try:
                deleted_count = await self.message_store.delete_messages_after_sequence(
                    project_id, target_sequence, include_target=True
                )
                logger.info(f"[REVERT_MSG] Deleted {deleted_count} messages from sequence {target_sequence}")
            except Exception as e:
                logger.error(f"[REVERT_MSG] Failed to delete messages: {e}")
        
        # 2. Restore files to checkpoint
        if self.artifact_store:
            try:
                run_id = project_state.get("run_id")
                restore_result = await self.artifact_store.restore_files_to_checkpoint(
                    project_id, target_sequence, run_id
                )
                logger.info(f"[REVERT_MSG] Files restored: {restore_result}")
            except Exception as e:
                logger.warning(f"[REVERT_MSG] Failed to restore files: {e}")
        
        # 3. Clear in-memory approvals and cancel in messages collection
        approval_manager.clear_project_approvals(project_id)
        if self.message_store:
            try:
                cancelled_count = await self.message_store.cancel_pending_approvals_for_project(project_id)
                logger.info(f"[REVERT_MSG] Cancelled {cancelled_count} pending approvals in messages")
            except Exception as e:
                logger.warning(f"[REVERT_MSG] Failed to cancel message approvals: {e}")
        
        # 4. Find the last remaining approval message (after deletion). We
        # read raw `subtype` / `metadata.phase` directly — no string-to-canon
        # mapping. The value flows through to the FE for display and to
        # `current_phase` for project summary; the FSM doesn't dispatch on
        # it after task 1.
        target_phase = "general"
        last_approval_msg: Optional[Dict[str, Any]] = None
        remaining_messages: List[Dict[str, Any]] = []
        if self.message_store:
            try:
                remaining_messages = await self.message_store.get_messages(
                    project_id, limit=1000, exclude_types=list(TOOL_LEDGER_TYPES)
                )
                for msg in reversed(remaining_messages):
                    if msg.get("type") == "approval":
                        last_approval_msg = msg
                        target_phase = (
                            msg.get("metadata", {}).get("phase")
                            or msg.get("subtype")
                            or "general"
                        )
                        logger.info(
                            "[REVERT_MSG] last approval msg approval_id=%s subtype=%r phase=%r",
                            (msg.get("data") or {}).get("approval_id"),
                            msg.get("subtype"), target_phase,
                        )
                        break
                    msg_phase = msg.get("metadata", {}).get("phase")
                    if msg_phase:
                        target_phase = msg_phase
                        break
            except Exception as e:
                logger.warning(f"[REVERT_MSG] Failed to scan remaining messages: {e}")

        # 5. Update project state
        project_state["current_phase"] = target_phase
        project_state["status"] = "initialized"

        await self.storage.save_project(
            project_id,
            {
                "user_prompt": project_state.get("user_prompt", ""),
                "title": project_state.get("title", "Untitled Project"),
                "status": "initialized",
                "current_phase": target_phase,
                "approval_mode": project_state.get("approval_mode", "human"),
            }
        )

        logger.info(f"[REVERT_MSG] Completed revert project={project_id} phase={target_phase}")

        await self.event_emitter.emit(EventSchema.PROJECT_REVERTED, project_state.get("run_id"), {
            "project_id": project_id,
            "target_sequence": target_sequence,
            "deleted_messages": deleted_count,
            "new_phase": target_phase,
            "prefill_input": prefill,
            "timestamp": datetime.utcnow().isoformat(),  # For UI event filtering
        })

        if reinstate_runtime_fn:
            sc = project_state.get("shared_context")
            if sc:
                reinstate_runtime_fn(project_id, sc)
                logger.info(f"[REVERT_MSG] Runtime reinstated for project={project_id}")

        # Re-seed the approval gate by flipping the preserved approval
        # row's status back to pending IN PLACE (ADR 0007). The original
        # approval_id is preserved across the rewind; the events log
        # (APPROVAL_GIVEN, APPROVAL_REQUESTED, PROJECT_REVERTED) carries
        # the audit trail of the approve/cancel transitions, not a chain
        # of duplicate approval rows in the message store.
        if last_approval_msg:
            await self._reseed_approval_from_message(
                project_id, project_state, approval_manager, last_approval_msg
            )

        await self._finalize_revert_durable_state(project_id, project_state)
        return {
            "status": "ok",
            "target_sequence": target_sequence,
            "deleted_messages": deleted_count,
            "new_phase": target_phase,
            "prefill_input": prefill,
        }

    async def _reseed_approval_from_message(
        self,
        project_id: str,
        project_state: Dict,
        approval_manager,
        approval_msg: Dict[str, Any],
    ) -> None:
        """Re-seed an approval gate by flipping its row's status back to
        pending IN PLACE (ADR 0007).

        Delegates to `approval_manager.reseed_from_message`, which mutates
        the existing DB row (no new row appended), re-hydrates the
        in-memory pending_approvals entry, and re-emits SSE +
        APPROVAL_REQUESTED so the FE re-renders the card and the FSM
        rehydrator (ensure_workflow_running) binds to the same
        approval_id on its next pass.

        The audit trail of the approve/cancel transitions lives in the
        events log, not in chat-visible row history.

        Skips silently if the row lacks `data.approval_id` (legacy
        un-backfilled rows) — project sits in initialized state until the
        user re-engages via chat.
        """
        # project_id: setdefault — every persisted approval msg already
        # carries project_id from append-time, this is just belt-and-braces
        # for stub callers in tests.
        #
        # run_id: ALWAYS overwrite with project_state's current run_id, do
        # NOT use setdefault. The persisted msg.run_id reflects the run
        # that originally created the gate; after a session restart the
        # project's current run_id differs. SSE channels and downstream
        # event consumers route on the run_id we emit, so a stale value
        # routes to a run the FE is no longer listening on — the card
        # flip arrives, but on a channel nobody's subscribed to.
        # (This preserves the behavior the pre-ADR-0007 code had via its
        # explicit `run_id=project_state.get("run_id")` argument to
        # request_approval.)
        msg = dict(approval_msg)
        msg.setdefault("project_id", project_id)
        msg["run_id"] = project_state.get("run_id")

        try:
            flipped_id = await approval_manager.reseed_from_message(msg)
            if flipped_id:
                logger.info(
                    "[REVERT] approval.reseeded project_id=%s approval_id=%s gate_node_id=%r",
                    project_id, flipped_id,
                    (msg.get("data") or {}).get("gate_node_id"),
                )
            else:
                # reseed_from_message returns None on legitimate skip
                # paths (missing data.approval_id, DB no-match) without
                # raising. Log here at the [REVERT] namespace so the
                # failure mode is visible from the revert layer —
                # otherwise the only trace is an [APPROVAL] warning
                # several call frames away, easy to miss when debugging
                # "I reverted and the gate disappeared."
                logger.warning(
                    "[REVERT] approval.reseed_skipped project_id=%s approval_id=%s gate_node_id=%r "
                    "(reseed_from_message returned None — likely missing data.approval_id or DB no-match)",
                    project_id,
                    (msg.get("data") or {}).get("approval_id"),
                    (msg.get("data") or {}).get("gate_node_id"),
                )
        except Exception as e:
            logger.error(
                "[REVERT] approval.reseed_failed project_id=%s gate_node_id=%r err=%s",
                project_id, (msg.get("data") or {}).get("gate_node_id"), e,
            )

    async def _reseed_last_pre_snapshot_approval(
        self,
        project_id: str,
        project_state: Dict,
        approval_manager,
        snapshot: Dict[str, Any],
    ) -> None:
        """Scan the message store for the latest approval row created at
        or before the snapshot time, and flip-back via
        `_reseed_approval_from_message` (ADR 0007).

        Snapshot reverts DO call `truncate_conversation` upstream
        (`revert_to_user_message` unconditionally; `revert_project` only
        when snap.meta.conversation_index is set), but surviving rows can
        still include approvals created after `snap.created_at` —
        append-time isn't strictly ordered with truncation depth. The
        created_at bound is the load-bearing filter that prevents
        resurrecting a gate the user was never at.

        No-op when message_store is absent, snapshot.created_at is
        missing (refuses rather than picking blind — see body for
        rationale), or no qualifying row exists (e.g., reverting to a
        `project_started` snapshot before any gate was created — project
        sits in initialized state until user re-engagement).
        """
        if not self.message_store:
            return
        try:
            messages = await self.message_store.get_messages(
                project_id, limit=1000, exclude_types=list(TOOL_LEDGER_TYPES)
            )
        except Exception as e:
            logger.warning("[REVERT] Failed to read messages for re-seed: %s", e)
            return

        snap_created = snapshot.get("created_at") if isinstance(snapshot, dict) else None
        if not snap_created:
            # The time bound is the only thing keeping the scan from
            # picking a gate created AFTER the snapshot. Without a
            # snapshot timestamp the bound is meaningless; degrading to
            # "pick the latest approval period" is the exact failure
            # mode the bound exists to prevent (would resurrect a
            # downstream gate the user was never at). Refusing is
            # safer — project sits in initialized, user re-engages via
            # chat to recover.
            logger.warning(
                "[REVERT] approval.reseed_skipped project_id=%s "
                "reason=snapshot_missing_created_at "
                "(scan disabled to avoid resurrecting downstream gate)",
                project_id,
            )
            return

        last_approval_msg: Optional[Dict[str, Any]] = None
        for msg in reversed(messages or []):
            if msg.get("type") != "approval":
                continue
            msg_created = msg.get("created_at")
            # Per-row fall-through: if THIS msg lacks created_at,
            # include it rather than skip — including one extra
            # legacy row is less harmful than skipping a legitimate
            # candidate. (The snap_created=None case is handled by the
            # early return above; the bound itself is never silently
            # disabled.) Lexicographic comparison on ISO timestamps;
            # both production paths emit tz-aware ISO strings so this
            # is well-defined. TypeError catch handles format drift
            # (datetime vs string) without crashing the revert.
            if msg_created:
                try:
                    if msg_created > snap_created:
                        continue
                except TypeError:
                    pass
            last_approval_msg = msg
            break

        if last_approval_msg:
            await self._reseed_approval_from_message(
                project_id, project_state, approval_manager, last_approval_msg
            )
