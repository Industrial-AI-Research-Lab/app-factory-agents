"""
Approval Manager

Handles approval gates for the orchestration workflow:
- Request approvals (requirements, plan, output, deploy)
- Wait for approval resolution
- Approve/reject pending gates
- Activity tracking for timeout prevention
"""

from typing import Any, Dict, Optional
from datetime import datetime
import asyncio
import uuid
import logging

from schemas import ApprovalSchema, ApprovalStatus, EventSchema

logger = logging.getLogger(__name__)


class ApprovalManager:
    """
    Manages approval gates for the orchestration workflow.
    
    Approval gates pause workflow execution until human approval/rejection.
    """
    
    def __init__(
        self,
        storage_backend,
        event_emitter,
        snapshot_manager,
        message_store=None,
        container_manager=None,
        artifact_store=None,
    ):
        self.storage = storage_backend
        self.event_emitter = event_emitter
        self.snapshot_manager = snapshot_manager
        self.message_store = message_store
        self.container_manager = container_manager
        self.artifact_store = artifact_store
        
        # Pending approvals (in-memory cache, backed by DB)
        self.pending_approvals: Dict[str, Dict] = {}
    
    async def request_approval(
        self,
        project_id: str,
        approval_type: str,
        data: Dict,
        run_id: Optional[str] = None,
        persist: bool = True,
        snapshot: bool = True,
    ) -> str:
        """Request human approval (per-instance UUID-based with run_id scoping)
        
        Returns the approval_id for tracking in the workflow.
        """
        # Completed instances belong to a prior visit. Restart recovery reuses
        # them through the engine's explicit rehydration marker.
        incoming_gate_node_id = (data or {}).get("gate_node_id")
        active_statuses = {
            ApprovalStatus.PENDING.value,
        }
        for existing_id, existing in self.pending_approvals.items():
            if existing.get("project_id") != project_id:
                continue
            if existing.get("run_id") != run_id:
                continue
            existing_status = existing.get("status")
            if hasattr(existing_status, "value"):
                existing_status = existing_status.value
            if existing_status not in active_statuses:
                continue
            existing_gate_node_id = (existing.get("data") or {}).get("gate_node_id")
            existing_type = (
                existing.get("gate_type")
                or existing.get("subtype")
                or existing.get("approval_type")
            )
            matches_gate = (
                (incoming_gate_node_id and existing_gate_node_id == incoming_gate_node_id)
                or (existing_type == approval_type)
            )
            if matches_gate:
                logger.info(
                    "[APPROVAL] Duplicate %s approval prevented (memory) for %s, returning existing %s (status=%s)",
                    approval_type, project_id, existing_id, existing_status,
                )
                return existing_id

        if self.message_store:
            try:
                pending_in_db = await self.message_store.get_pending_approvals_for_project(project_id, limit=10)
                for pa in pending_in_db or []:
                    if pa.get("run_id") != run_id:
                        continue
                    pa_type = pa.get("subtype") or pa.get("type")
                    pa_gate_node_id = (pa.get("data") or {}).get("gate_node_id")
                    matches = (
                        (incoming_gate_node_id and pa_gate_node_id == incoming_gate_node_id)
                        or (pa_type == approval_type)
                    )
                    if matches:
                        existing_id = (pa.get("data") or {}).get("approval_id")
                        if existing_id:
                            logger.info(
                                "[APPROVAL] Duplicate %s approval prevented (DB pending) for %s, returning existing %s",
                                approval_type, project_id, existing_id,
                            )
                            # Hydrate in-memory so wait_for_approval finds it.
                            self.pending_approvals.setdefault(existing_id, pa)
                            return existing_id
            except Exception as e:
                logger.warning(f"[APPROVAL] Failed to check DB for existing approvals: {e}")
        
        approval_id = str(uuid.uuid4())
        
        # Use ApprovalSchema for consistent format
        approval = ApprovalSchema.create(
            approval_id=approval_id,
            project_id=project_id,
            approval_type=approval_type,
            data=data,
            status=ApprovalStatus.PENDING,
            run_id=run_id,
            created_at=datetime.utcnow().isoformat()
        )
        
        self.pending_approvals[approval_id] = approval
        
        # Append approval message to message store (SINGLE SOURCE OF TRUTH).
        # persist=False remains available for explicitly ephemeral internal gates.
        if persist and self.message_store:
            try:
                # Merge approval_id into data (single source of truth - no separate metadata)
                approval_data = {**data, "approval_id": approval_id}
                message = await self.message_store.append_approval(
                    project_id=project_id,
                    subtype=approval_type,
                    data=approval_data,
                    run_id=run_id,
                    content=f"Please review the {approval_type} and approve or reject.",
                )
                # Emit message_appended event for real-time updates
                await self.event_emitter.emit("message_appended", run_id, {
                    "project_id": project_id,
                    "message": message
                })
            except Exception as e:
                logger.warning(f"Failed to append approval message: {e}")

        # Snapshot files from container/repo to DB when user input is awaited
        # This ensures files are versioned and recoverable. Skipped for ephemeral
        # Agent-result gates pass snapshot=False: they review an LLM result rather
        # than a file checkpoint, avoiding a repository snapshot on every call.
        if snapshot:
            await self._snapshot_files_to_db(project_id, run_id)

        # Emit event with approval_id and run_id (legacy)
        await self.event_emitter.emit(EventSchema.APPROVAL_REQUESTED, run_id, {
            "approval_id": approval_id,
            "project_id": project_id,
            "run_id": run_id,
            "gate_type": approval_type,
            "data": data
        })
        
        return approval_id
    
    async def _snapshot_files_to_db(self, project_id: str, run_id: Optional[str] = None) -> int:
        """Snapshot current files from container/repo to DB.
        
        Called when user input is awaited to persist file state.
        """
        logger.info(f"[SNAPSHOT] Starting snapshot for {project_id}, run_id={run_id}")
        logger.info(f"[SNAPSHOT] container_manager={self.container_manager is not None}, artifact_store={self.artifact_store is not None}")
        
        if not self.container_manager or not self.artifact_store:
            logger.warning("[SNAPSHOT] Missing dependencies - skipping snapshot")
            return 0
        
        try:
            # Get files from container/repo
            artifacts = await self._get_files_from_container_or_repo(project_id)
            logger.info(f"[SNAPSHOT] Got {len(artifacts) if artifacts else 0} artifacts from container/repo")
            if not artifacts:
                return 0
            
            # Save to DB (force=True to skip confirmation during approval flow)
            result = await self.artifact_store.snapshot_from_artifacts(
                project_id, artifacts, run_id, force=True
            )
            saved = result.get("saved", 0)
            logger.info(f"[SNAPSHOT] Saved {saved} files to DB for {project_id}")
            return saved
        except Exception as e:
            logger.warning(f"[SNAPSHOT] Failed to snapshot files for {project_id}: {e}")
            import traceback
            logger.warning(f"[SNAPSHOT] Traceback: {traceback.format_exc()}")
            return 0
    
    async def _get_files_from_container_or_repo(self, project_id: str) -> list:
        """Get files from container (if active) or git repo."""
        import subprocess
        from pathlib import Path
        from config.artifacts import should_include_file, has_conflict_markers
        
        # Try container first
        try:
            status = await self.container_manager.get_container_status(project_id)
            if status.get("active"):
                files = await self.container_manager.list_files_in_container(project_id, ".")
                if files:
                    artifacts = []
                    for f in files:
                        path = f if isinstance(f, str) else f.get("path", f.get("name", ""))
                        if not path or not should_include_file(path):
                            continue
                        try:
                            content = await self.container_manager.read_file_from_container(project_id, path)
                            if content and not has_conflict_markers(content):
                                artifacts.append({"path": path, "content": content})
                        except Exception:
                            continue
                    if artifacts:
                        return artifacts
        except Exception as e:
            logger.debug(f"Container read failed, trying repo: {e}")
        
        # Fallback to git repo (container-use branch)
        try:
            status = await self.container_manager.get_container_status(project_id)
            repo_path = status.get("repo_path")
            if not repo_path:
                return []
            
            base = Path(repo_path)
            if not base.exists() or not (base / ".git").exists():
                return []
            
            # Checkout container-use branch
            result = subprocess.run(
                ["git", "branch", "-a", "--list", "*container-use/*"],
                cwd=str(base), capture_output=True, text=True
            )
            cu_branches = [b.strip().lstrip("* ") for b in result.stdout.strip().splitlines() if b.strip()]
            if cu_branches:
                subprocess.run(["git", "checkout", cu_branches[-1]], cwd=str(base), capture_output=True)
            
            # Read files using centralized filter
            artifacts = []
            for p in base.rglob("*"):
                try:
                    if p.is_file():
                        rel = str(p.relative_to(base)).replace("\\", "/")
                        if not should_include_file(rel):
                            continue
                        txt = p.read_text(encoding="utf-8", errors="replace")
                        if has_conflict_markers(txt):
                            continue
                        artifacts.append({"path": rel, "content": txt})
                except Exception:
                    continue
            return artifacts
        except Exception as e:
            logger.warning(f"Repo read failed: {e}")
            return []
    
    async def wait_for_approval(
        self,
        approval_id: str,
        timeout: int = 3600
    ) -> Dict:
        """Wait for human to approve/reject (UUID-based approval_id)"""
        start_time = datetime.utcnow()
        
        while True:
            approval = self.pending_approvals.get(approval_id)
            
            if approval and not ApprovalSchema.is_pending(approval):
                return approval
            
            # Check if user is actively interacting (last_activity timestamp)
            if approval and "last_activity" in approval:
                # Reset timer based on last activity
                start_time = datetime.fromisoformat(approval["last_activity"])
            
            # Check timeout
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if timeout != -1 and elapsed > timeout:
                # Soft timeout: return a sentinel status so callers can pause gracefully
                return {"status": "timeout", "approval_id": approval_id}
            
            await asyncio.sleep(1)  # Poll every second

    def update_approval_activity(self, approval_id: str):
        """Update last activity timestamp for an approval to prevent timeout"""
        if approval_id in self.pending_approvals:
            self.pending_approvals[approval_id]["last_activity"] = datetime.utcnow().isoformat()
            # Note: activity tracking removed from DB - messages don't track last_activity
    
    async def approve(
        self,
        approval_id: str,
        feedback: Optional[str] = None,
        project_state: Optional[Dict] = None,
        interaction_response: Optional[Dict[str, Any]] = None,
        expected_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Approve a pending gate.

        Flips the approval status, emits the approval-result message, creates
        the user_message snapshot for revertability, and emits the legacy
        APPROVAL_GIVEN event. The workflow runtime parked at the corresponding
        gate (rehydrated by Orchestrator.ensure_workflow_running) wakes from
        wait_for_approval and follows the workflow definition's edge — no
        post-approval Python dispatch table is consulted.

        Args:
            approval_id: The approval to approve
            feedback: Optional feedback from the user
            project_state: Optional project state for creating snapshots
        """
        if approval_id not in self.pending_approvals:
            return False

        approval = self.pending_approvals[approval_id]
        if not ApprovalSchema.is_pending(approval):
            return False
        project_id = ApprovalSchema.get_project_id(approval)
        run_id = ApprovalSchema.get_run_id(approval)
        resolution = self._build_resolution(interaction_response, feedback)

        # Update approval message status in messages collection (SINGLE SOURCE OF TRUTH)
        updated_msg = None
        if self.message_store:
            resolver = getattr(
                self.message_store,
                "resolve_approval_by_approval_id",
                None,
            )
            if callable(resolver):
                resolve_kwargs = (
                    {"expected_data": expected_data}
                    if expected_data is not None
                    else {}
                )
                updated_msg = await resolver(
                    approval_id,
                    "approved",
                    resolution,
                    **resolve_kwargs,
                )
                if not updated_msg:
                    logger.warning(
                        "[APPROVAL] approval_id=%s decision=approve - CAS lost",
                        approval_id,
                    )
                    return False
            else:
                updated_msg = await self.message_store.update_approval_status_by_approval_id(
                    approval_id,
                    "approved",
                )

        approval["status"] = ApprovalStatus.APPROVED
        approval["approved_at"] = datetime.utcnow().isoformat()
        approval["feedback"] = feedback
        if resolution:
            approval["interaction_response"] = resolution["interaction_response"]
            approval["data"] = {
                **(approval.get("data") or {}),
                "resolution": resolution,
            }

        if self.message_store:
            # Emit SSE event so UI updates in real-time
            if updated_msg:
                await self.event_emitter.emit("message_appended", run_id, {
                    "project_id": project_id,
                    "message": updated_msg
                })
        
        # Create a synthetic user_message checkpoint so this approval is revertable.
        # Delegation gates are mid-agent (not graph gates), so they get no
        # user_message snapshot — they must not appear in conversation history.
        _is_delegation = bool((self.pending_approvals[approval_id].get("data") or {}).get("delegation_gate"))
        if project_state and not _is_delegation:
            try:
                sc = project_state.get("shared_context")
                if sc:
                    approval_type = self.pending_approvals[approval_id].get("type", "general")
                    content = (feedback or "").strip() or f"[APPROVE] {approval_type}"
                    await sc.add_conversation_message(
                        role="user",
                        content=content,
                        phase=approval_type,
                        metadata={"action": "approval"}
                    )
                    # Revert consumes conversation_index as a message-store
                    # sequence (truncate before_index), so record the real
                    # latest sequence — not a count of the deprecated,
                    # always-empty in-memory conversation cache.
                    idx = (
                        await self.message_store.get_latest_sequence(project_id)
                        if self.message_store
                        else 0
                    )

                    if self.snapshot_manager:
                        await self.snapshot_manager.create_snapshot(
                            project_id,
                            sc,
                            snap_type="user_message",
                            label="User approval",
                            phase=project_state.get("current_phase") or None,
                            meta={
                                "tags": ["user_action", "user_message", "approval"],
                                "conversation_index": idx,
                                "input_prefill": content,
                            },
                        )
                        logger.info(f"[APPROVE] Created user_message snapshot for {project_id} at conv_index={idx}")
                    else:
                        logger.warning(f"[APPROVE] snapshot_manager is None, cannot create snapshot for {project_id}")
                else:
                    logger.warning(f"[APPROVE] shared_context is None for {project_id}, cannot create snapshot")
            except Exception as e:
                logger.error(f"[APPROVE] Failed to create snapshot for {project_id}: {e}")

        # Append system message for approval result (unified message system).
        # Skipped for delegation gates: they are surfaced only via agent.delegation.*
        # events + the ephemeral card/Inspector, and must NOT leak a receipt bubble
        # into conversation history (mirrors the user_message snapshot guard above).
        if self.message_store and not _is_delegation:
            try:
                approval_type = self.pending_approvals[approval_id].get("type", "general")
                message = await self.message_store.append_system_message(
                    project_id=project_id,
                    subtype="approval_result",
                    content=f"{approval_type.capitalize()} approved. Continuing workflow...",
                    run_id=run_id,
                    data={"approval_id": approval_id, "result": "approved", "feedback": feedback}
                )
                await self.event_emitter.emit("message_appended", run_id, {
                    "project_id": project_id,
                    "message": message
                })
            except Exception as e:
                logger.warning(f"Failed to append approval result message: {e}")

        await self.event_emitter.emit(EventSchema.APPROVAL_GIVEN, run_id, {
            "approval_id": approval_id,
            "status": "approved",
            "project_id": project_id,
        })
        return True

    async def reject(
        self,
        approval_id: str,
        reason: str,
        interaction_response: Optional[Dict[str, Any]] = None,
        expected_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Reject a pending gate"""
        if approval_id not in self.pending_approvals:
            return False

        approval = self.pending_approvals[approval_id]
        if not ApprovalSchema.is_pending(approval):
            return False
        project_id = ApprovalSchema.get_project_id(approval)
        run_id = ApprovalSchema.get_run_id(approval)
        resolution = self._build_resolution(interaction_response, reason)
        _is_delegation = bool((approval.get("data") or {}).get("delegation_gate"))

        # Update approval message status in messages collection (SINGLE SOURCE OF TRUTH)
        updated_msg = None
        if self.message_store:
            resolver = getattr(
                self.message_store,
                "resolve_approval_by_approval_id",
                None,
            )
            if callable(resolver):
                resolve_kwargs = (
                    {"expected_data": expected_data}
                    if expected_data is not None
                    else {}
                )
                updated_msg = await resolver(
                    approval_id,
                    "rejected",
                    resolution,
                    **resolve_kwargs,
                )
                if not updated_msg:
                    logger.warning(
                        "[APPROVAL] approval_id=%s decision=reject - CAS lost",
                        approval_id,
                    )
                    return False
            else:
                updated_msg = await self.message_store.update_approval_status_by_approval_id(
                    approval_id,
                    "rejected",
                )

        approval["status"] = ApprovalStatus.REJECTED
        approval["rejected_at"] = datetime.utcnow().isoformat()
        approval["reason"] = reason
        if resolution:
            approval["interaction_response"] = resolution["interaction_response"]
            approval["data"] = {
                **(approval.get("data") or {}),
                "resolution": resolution,
            }

        if self.message_store:
            # Emit SSE event so UI updates in real-time
            if updated_msg:
                await self.event_emitter.emit("message_appended", run_id, {
                    "project_id": project_id,
                    "message": updated_msg
                })

        # Append system message for rejection — skipped for delegation gates (see approve()).
        if self.message_store and not _is_delegation:
            try:
                approval_type = self.pending_approvals[approval_id].get("type", "general")
                message = await self.message_store.append_system_message(
                    project_id=project_id,
                    subtype="approval_result",
                    content=f"{approval_type.capitalize()} rejected: {reason}",
                    run_id=run_id,
                    data={"approval_id": approval_id, "result": "rejected", "reason": reason}
                )
                await self.event_emitter.emit("message_appended", run_id, {
                    "project_id": project_id,
                    "message": message
                })
            except Exception as e:
                logger.warning(f"Failed to append rejection message: {e}")

        await self.event_emitter.emit(EventSchema.APPROVAL_GIVEN, run_id, {
            "approval_id": approval_id,
            "status": "rejected",
            "reason": reason,
            "project_id": project_id,
        })
        return True

    @staticmethod
    def _build_resolution(
        interaction_response: Optional[Dict[str, Any]],
        feedback: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(interaction_response, dict) or not interaction_response:
            return None
        return {
            "interaction_response": dict(interaction_response),
            "feedback": feedback,
            "recorded_at": datetime.utcnow().isoformat(),
        }

    def get_pending(self, approval_id: str) -> Optional[Dict]:
        """Get a pending approval by ID"""
        return self.pending_approvals.get(approval_id)
    
    def set_pending(self, approval_id: str, approval: Dict):
        """Set a pending approval"""
        self.pending_approvals[approval_id] = approval
    
    def remove_pending(self, approval_id: str):
        """Remove a pending approval"""
        self.pending_approvals.pop(approval_id, None)

    async def cancel_pending(
        self,
        approval_id: str,
        reason: str = "cancelled",
    ) -> bool:
        """Drop a still-pending gate resolved WITHOUT a user decision (timeout /
        programmatic cancel) AND emit APPROVAL_GIVEN(status="cancelled").

        approve()/reject() emit APPROVAL_GIVEN so the frontend clears the ephemeral
        delegation review card (it only listens for approval_given). The silent-drop
        paths (remove_pending) did not, leaving an orphaned card that also
        resurrected on SSE replay. Emitting here makes the lifecycle symmetric.
        Returns True only when cancellation won the pending-state transition.
        """
        approval = self.pending_approvals.get(approval_id)
        if not approval:
            return False

        updated = None
        try:
            if self.message_store:
                resolver = getattr(
                    self.message_store,
                    "resolve_approval_by_approval_id",
                    None,
                )
                if callable(resolver):
                    updated = await resolver(approval_id, "cancelled")
                    if not updated:
                        logger.info(
                            "[APPROVAL] approval_id=%s decision=cancel - CAS lost",
                            approval_id,
                        )
                        return False
                else:
                    updated = await self.message_store.update_approval_status_by_approval_id(
                        approval_id,
                        "cancelled",
                    )
                    if not updated:
                        return False
        except Exception as e:
            logger.warning(
                "[APPROVAL] cancel_pending persistence failed for %s: %s",
                approval_id,
                e,
            )
            return False

        self.pending_approvals.pop(approval_id, None)
        try:
            if updated:
                await self.event_emitter.emit(
                    "message_appended",
                    ApprovalSchema.get_run_id(approval),
                    {
                        "project_id": ApprovalSchema.get_project_id(approval),
                        "message": updated,
                    },
                )
            if self.event_emitter:
                await self.event_emitter.emit(EventSchema.APPROVAL_GIVEN, ApprovalSchema.get_run_id(approval), {
                    "approval_id": approval_id,
                    "status": "cancelled",
                    "reason": reason,
                    "project_id": ApprovalSchema.get_project_id(approval),
                })
        except Exception as e:
            logger.warning("[APPROVAL] cancel_pending emit failed for %s: %s", approval_id, e)
        return True

    def clear_project_approvals(self, project_id: str):
        """Clear all pending approvals for a project"""
        to_remove = [
            aid for aid, a in self.pending_approvals.items()
            if a.get("project_id") == project_id
        ]
        for aid in to_remove:
            self.pending_approvals.pop(aid, None)
        return len(to_remove)

    def clear_project_delegation_gates(self, project_id: str) -> int:
        """Remove a project's resolved delegation-gate approvals from memory.

        Called once at the end of an orchestrate run (DelegationReviewers.cleanup),
        when no gate is being awaited — so it cannot race the wait_for_approval poll
        loop the way per-gate removal in approve()/reject() would.
        """
        to_remove = [
            aid for aid, a in self.pending_approvals.items()
            if a.get("project_id") == project_id and (a.get("data") or {}).get("delegation_gate")
        ]
        for aid in to_remove:
            self.pending_approvals.pop(aid, None)
        return len(to_remove)

    async def reseed_from_message(
        self,
        approval_msg: Dict,
    ) -> Optional[str]:
        """Flip an existing approval row back to pending in place (ADR 0007).

        Used by reverters to restore a gate to actionable state without
        appending a new approval row. The original approval_id is
        preserved, so the FSM rehydrator binds to the same identifier and
        any in-flight caller holding the old id still works.

        Steps:
          1. DB row status → 'pending' via update_approval_status_by_approval_id.
          2. In-memory pending_approvals re-hydrated with the same approval_id.
          3. message_appended SSE re-announces the row (FE re-renders the
             card from green/red → amber by status).
          4. APPROVAL_REQUESTED event re-fired for parity with first-time
             gate creation in request_approval.

        Returns the approval_id on success, or None if the row lacks
        data.approval_id (legacy un-backfilled row) or the DB update found
        no matching row.
        """
        data = dict(approval_msg.get("data") or {})
        if data.get("delegation_gate"):
            # A delegation gate's decision is consumed only by an in-agent
            # awaiter that dies with the run, and its synthetic gate_node_id
            # matches no workflow node — so ensure_workflow_running refuses to
            # rehydrate it. Re-seeding would only leave an orphan card that
            # swallows the next message as fake feedback; skip so the project
            # stays open to agent continuation instead.
            logger.info(
                "[APPROVAL] reseed_from_message: skipping delegation gate %r — no resumable executor after rewind",
                data.get("gate_node_id"),
            )
            return None
        approval_id = data.get("approval_id")
        if not approval_id:
            logger.warning(
                "[APPROVAL] reseed_from_message: approval row missing data.approval_id, cannot flip in place"
            )
            return None

        project_id = approval_msg.get("project_id")
        run_id = approval_msg.get("run_id")
        approval_type = (
            data.get("gate_label")
            or approval_msg.get("subtype")
            or "Review"
        )

        updated_msg = None
        if self.message_store:
            try:
                updated_msg = await self.message_store.update_approval_status_by_approval_id(
                    approval_id, ApprovalStatus.PENDING.value
                )
            except Exception as e:
                logger.error(
                    "[APPROVAL] reseed_from_message: failed to flip approval_id=%s status: %s",
                    approval_id, e,
                )
                return None

        if self.message_store and not updated_msg:
            logger.error(
                "[APPROVAL] reseed_from_message: no DB row matched approval_id=%s; aborting flip",
                approval_id,
            )
            return None

        approval = ApprovalSchema.create(
            approval_id=approval_id,
            project_id=project_id,
            approval_type=approval_type,
            data=data,
            status=ApprovalStatus.PENDING,
            run_id=run_id,
            created_at=datetime.utcnow().isoformat(),
        )
        self.pending_approvals[approval_id] = approval

        if updated_msg:
            await self.event_emitter.emit("message_appended", run_id, {
                "project_id": project_id,
                "message": updated_msg,
            })

        await self.event_emitter.emit(EventSchema.APPROVAL_REQUESTED, run_id, {
            "approval_id": approval_id,
            "project_id": project_id,
            "run_id": run_id,
            "gate_type": approval_type,
            "data": data,
        })

        logger.info(
            "[APPROVAL] reseed_from_message: flipped approval_id=%s gate=%r → pending",
            approval_id, data.get("gate_node_id"),
        )
        return approval_id
