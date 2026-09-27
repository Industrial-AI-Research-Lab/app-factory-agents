"""
Project Loader - Lazy Loading & Recovery

Projects are NOT loaded at startup. They are loaded on-demand when user accesses them.
Handles container recovery and deployment health checks.

Section 2.4 of TASKS.MD
"""

from typing import Dict, Any, Optional, Set, List
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import logging
import httpx

from storage.message_store import TOOL_LEDGER_TYPES
from sandbox.cu_mcp_inventory import CU_RECOVERY_RECREATE_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


@dataclass
class ProjectState:
    """Represents the loaded state of a project."""
    project: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    approvals: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    deployment: Optional[Dict[str, Any]] = None
    container_ready: bool = False
    deployment_ready: Optional[bool] = None
    needs_container_recovery: bool = False
    needs_deployment_recovery: bool = False
    recovery_in_progress: bool = False


class ProjectLoader:
    """
    Lazy project loader. Nothing loaded until accessed.
    
    Key principle: Projects are NOT loaded at startup.
    When user accesses `/monitor/{project_id}`:
    1. Load project state from DB
    2. Check container health, recover if needed
    3. Check deployment health, prompt user if down
    4. Enable UI for interaction
    """
    
    def __init__(
        self,
        storage_backend,
        container_manager,
        artifact_store,
        message_store=None,
        event_emitter=None,
    ):
        self.storage = storage_backend
        self.container_manager = container_manager
        self.artifact_store = artifact_store
        self.message_store = message_store
        self.event_emitter = event_emitter
        
        # Track which projects are loaded (in-memory cache)
        self.loaded_projects: Set[str] = set()
        self._project_cache: Dict[str, ProjectState] = {}
        self._loading_locks: Dict[str, asyncio.Lock] = {}
    
    async def startup_init(self):
        """
        Minimal startup - NO full project loading.
        
        This is intentionally empty. Projects are loaded on-demand.
        """
        logger.info("ProjectLoader initialized (lazy mode - no projects loaded)")
    
    def _get_lock(self, project_id: str) -> asyncio.Lock:
        """Get or create a lock for a project to prevent concurrent loading."""
        if project_id not in self._loading_locks:
            self._loading_locks[project_id] = asyncio.Lock()
        return self._loading_locks[project_id]
    
    async def load_project(self, project_id: str, force_reload: bool = False) -> ProjectState:
        """
        Load project on-demand when user accesses it.
        
        This is the main entry point for lazy loading.
        
        Args:
            project_id: Project identifier
            force_reload: Force reload even if cached
            
        Returns:
            ProjectState with all project data and health status
        """
        async with self._get_lock(project_id):
            # Check cache first
            if not force_reload and project_id in self._project_cache:
                return self._project_cache[project_id]
            
            logger.info(f"Loading project on-demand: {project_id}")
            
            # 1. Load basic project info from DB
            project = await self.storage.load_project(project_id)
            if not project:
                raise ProjectNotFoundError(project_id)
            
            # 2. Load context (for requirements/plan data)
            context = await self.storage.load_context(project_id) or {}
            
            # 3. Load messages (conversation shape only — without the
            # exclusion every cold load hauls tool_result bodies up to 512KB
            # each into a field nothing currently reads)
            messages = []
            if self.message_store:
                try:
                    messages = await self.message_store.get_messages(
                        project_id, exclude_types=list(TOOL_LEDGER_TYPES)
                    )
                except Exception as e:
                    logger.warning(f"Failed to load messages: {e}")
            
            # 4. Load pending approvals from messages (SINGLE SOURCE OF TRUTH)
            approvals = []
            try:
                if self.message_store:
                    approvals = await self.message_store.get_pending_approvals_for_project(project_id)
            except Exception as e:
                logger.warning(f"Failed to load approvals: {e}")
            
            # 5. Load file artifacts
            artifacts = []
            if self.artifact_store:
                try:
                    artifacts = await self.artifact_store.get_all_files(project_id)
                except Exception as e:
                    logger.warning(f"Failed to load artifacts: {e}")
            
            # 6. Load deployment info
            deployment = None
            try:
                deployment = await self.storage.get_active_deployment(project_id)
            except Exception as e:
                logger.debug(f"No active deployment: {e}")
            
            # 7. Check container health
            container_ready = False
            needs_container_recovery = False
            if self.container_manager:
                container_ready = await self._check_container_health(project_id)
                needs_container_recovery = not container_ready and len(artifacts) > 0
            
            # 8. Check deployment health (if exists)
            deployment_ready = None
            needs_deployment_recovery = False
            if deployment:
                deployment_ready = await self._check_deployment_health(deployment)
                needs_deployment_recovery = not deployment_ready
            
            # Build state object
            state = ProjectState(
                project=project,
                context=context,
                messages=messages,
                approvals=approvals,
                artifacts=artifacts,
                deployment=deployment,
                container_ready=container_ready,
                deployment_ready=deployment_ready,
                needs_container_recovery=needs_container_recovery,
                needs_deployment_recovery=needs_deployment_recovery,
            )
            
            # Cache the state
            self._project_cache[project_id] = state
            self.loaded_projects.add(project_id)
            
            logger.info(
                f"Project loaded: {project_id} "
                f"(container_ready={container_ready}, "
                f"artifacts={len(artifacts)}, "
                f"needs_recovery={needs_container_recovery})"
            )
            
            return state
    
    async def _check_container_health(self, project_id: str) -> bool:
        """Check if container exists and is healthy."""
        if not self.container_manager:
            return False
        
        try:
            status = await self.container_manager.get_container_status(project_id)
            return status.get("active", False) and status.get("environment_id") is not None
        except Exception as e:
            logger.debug(f"Container health check failed: {e}")
            return False
    
    async def _check_deployment_health(self, deployment: Dict[str, Any]) -> bool:
        """Check if deployment is healthy and accessible."""
        if not deployment:
            return False
        
        url = deployment.get("url")
        if not url:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url, follow_redirects=True)
                is_healthy = response.status_code < 500
                
                # Update health check timestamp in DB
                if is_healthy and self.storage:
                    deployment_id = deployment.get("deployment_id")
                    if deployment_id:
                        await self.storage.update_deployment_health(deployment_id)
                
                return is_healthy
        except Exception as e:
            logger.debug(f"Deployment health check failed for {url}: {e}")
            return False
    
    async def recover_container(self, project_id: str) -> Dict[str, Any]:
        """
        Recover container by restoring all artifacts from DB.
        
        1. Create fresh container
        2. Load all artifacts from DB
        3. Write all files to container
        
        Returns status dict with recovery result.
        """
        if not self.container_manager or not self.artifact_store:
            return {"status": "failed", "reason": "missing_dependencies"}
        
        logger.info(f"Starting container recovery for project: {project_id}")
        
        # Mark recovery in progress
        if project_id in self._project_cache:
            self._project_cache[project_id].recovery_in_progress = True
        
        try:
            # Emit recovery started event
            if self.event_emitter:
                # Container is one-per-project — recovery events are project-scoped,
                # not run-scoped. run_id=None so they show across all runs.
                await self.event_emitter.emit("container_recovery_started", None, {
                    "project_id": project_id,
                })
            
            # On failure the project stays unavailable and the workspace is kept
            # for another /recover, a revert or a snapshot. A missing folder
            # means the branch is gone: seed and hydrate from the DB.
            preserved = None
            if hasattr(self.container_manager, "preserved_workspace"):
                preserved = self.container_manager.preserved_workspace(project_id)
            prior_lockout = self.container_manager.session_unavailable_reason(project_id)
            recovery_mode = None

            def _remake_lockout(reason: str) -> None:
                mark = getattr(self.container_manager, "mark_session_unavailable", None)
                if mark is not None:
                    mark(project_id, reason=reason or "unavailable")

            async def _bounded_recreate(coro):
                try:
                    return await asyncio.wait_for(
                        coro, timeout=float(CU_RECOVERY_RECREATE_TIMEOUT_SECONDS)
                    )
                except asyncio.TimeoutError as exc:
                    raise RuntimeError(
                        f"recreate exceeded {CU_RECOVERY_RECREATE_TIMEOUT_SECONDS:g}s"
                    ) from exc

            def _abandon_preserved() -> None:
                clear_p = getattr(
                    self.container_manager, "clear_preserved_workspace", None
                )
                if clear_p is not None:
                    clear_p(project_id)

            try:
                used_preserved = bool(preserved and preserved.get("repo_path"))
                if used_preserved:
                    preserved_path = Path(str(preserved["repo_path"]))
                    if not preserved_path.is_dir():
                        logger.warning(
                            "[RECOVERY] preserved path missing project=%s repo=%s — "
                            "abandon and seed",
                            project_id,
                            preserved_path,
                        )
                        _abandon_preserved()
                        used_preserved = False
                if used_preserved:
                    try:
                        from sandbox.run_command_recovery import (
                            reopen_or_replace_on_workspace,
                        )

                        container, recovery_mode = await reopen_or_replace_on_workspace(
                            self.container_manager,
                            project_id,
                            repo_path=str(preserved["repo_path"]),
                            previous_environment_id=preserved.get(
                                "environment_id"
                            ),
                        )
                    except Exception as replace_err:
                        reason = str(replace_err) or type(replace_err).__name__
                        logger.warning(
                            "[RECOVERY] preserved reopen/replace failed project=%s "
                            "reason=%s — fail-closed (no HEAD seed)",
                            project_id,
                            reason,
                        )
                        _remake_lockout(reason)
                        cleanup = getattr(
                            self.container_manager, "cleanup_container", None
                        )
                        if cleanup is not None:
                            try:
                                await cleanup(project_id, keep_for_review=True)
                            except Exception as cleanup_err:
                                logger.warning(
                                    "[RECOVERY] cleanup after replace fail project=%s err=%s",
                                    project_id,
                                    cleanup_err,
                                )
                        return {
                            "status": "unavailable",
                            "reason": reason,
                            "container_id": None,
                        }
                else:
                    if prior_lockout:
                        self.container_manager.clear_session_unavailable(project_id)
                        logger.info(
                            "[RECOVERY] cleared session_unavailable for explicit recover project=%s",
                            project_id,
                        )
                    container = await _bounded_recreate(
                        self.container_manager.get_or_create_container(project_id)
                    )
            except Exception as recreate_err:
                reason = str(recreate_err) or type(recreate_err).__name__
                logger.warning(
                    "[RECOVERY] recreate/replace failed project=%s reason=%s",
                    project_id,
                    reason,
                )
                _abandon_preserved()
                _remake_lockout(reason)
                return {
                    "status": "unavailable",
                    "reason": reason,
                    "container_id": None,
                }

            if (
                not container.get("client")
                or container.get("status") in ("simulated", "unavailable")
            ):
                reason = container.get("error") or "container_use_unavailable"
                status = (
                    "unavailable"
                    if container.get("status") == "unavailable"
                    else "simulated"
                    if container.get("status") == "simulated"
                    else "failed"
                )
                if not used_preserved:
                    _abandon_preserved()
                _remake_lockout(reason)
                cleanup = getattr(self.container_manager, "cleanup_container", None)
                if cleanup is not None:
                    try:
                        await cleanup(project_id, keep_for_review=True)
                    except Exception as cleanup_err:
                        logger.warning(
                            "[RECOVERY] cleanup after non-ready create project=%s err=%s",
                            project_id,
                            cleanup_err,
                        )
                return {
                    "status": status,
                    "reason": reason,
                    "container_id": None,
                }

            try:
                from sandbox.run_command_recovery import smoke_session_alive

                await smoke_session_alive(container["client"])
            except Exception as smoke_err:
                reason = str(smoke_err) or type(smoke_err).__name__
                logger.warning(
                    "[RECOVERY] smoke failed after replace/create project=%s reason=%s",
                    project_id,
                    reason,
                )
                cleanup = getattr(self.container_manager, "cleanup_container", None)
                if cleanup is not None:
                    try:
                        await cleanup(project_id, keep_for_review=True)
                    except Exception as cleanup_err:
                        logger.warning(
                            "[RECOVERY] cleanup after smoke fail project=%s err=%s",
                            project_id,
                            cleanup_err,
                        )
                _remake_lockout(reason)
                return {
                    "status": "unavailable",
                    "reason": reason,
                    "container_id": None,
                }

            self.container_manager.clear_session_unavailable(project_id)
            if hasattr(self.container_manager, "clear_preserved_workspace"):
                self.container_manager.clear_preserved_workspace(project_id)

            replaced_preserved = used_preserved
            if replaced_preserved:
                if project_id in self._project_cache:
                    self._project_cache[project_id].container_ready = True
                    self._project_cache[project_id].needs_container_recovery = False
                    self._project_cache[project_id].recovery_in_progress = False
                mode = recovery_mode or "replace"
                reason = (
                    "preserved_workspace_reopened"
                    if mode == "reopen"
                    else "preserved_workspace_replaced"
                )
                if self.event_emitter:
                    await self.event_emitter.emit("container_recovery_completed", None, {
                        "project_id": project_id,
                        "files_restored": 0,
                        "errors": 0,
                        "preserved_workspace": True,
                        "recovery_mode": mode,
                    })
                logger.info(
                    "[RECOVERY] restored preserved workspace project=%s env=%s mode=%s",
                    project_id,
                    container.get("environment_id"),
                    mode,
                )
                return {
                    "status": "ready",
                    "reason": reason,
                    "recovery_mode": mode,
                    "container_id": container.get("environment_id"),
                    "files_restored": 0,
                }

            # 2. Load all artifacts from DB (try multiple sources)
            artifacts = await self.artifact_store.get_all_files(project_id)
            
            # Fallback 1: contexts.artifacts if file_artifacts is empty
            if not artifacts:
                try:
                    context = await self.storage.load_context(project_id)
                    if context and context.get("artifacts"):
                        artifacts = [a for a in context["artifacts"] if a.get("path") and a.get("content")]
                        logger.info(f"Using {len(artifacts)} artifacts from contexts.artifacts")
                except Exception as e:
                    logger.warning(f"Failed to load context artifacts: {e}")
            
            # Fallback 2: Read from git repo (container-use branch)
            if not artifacts:
                try:
                    artifacts = await self._load_artifacts_from_repo(project_id)
                    if artifacts:
                        logger.info(f"Using {len(artifacts)} artifacts from git repo")
                except Exception as e:
                    logger.warning(f"Failed to load repo artifacts: {e}")
            
            if not artifacts:
                return {
                    "status": "ready",
                    "reason": "no_artifacts_to_restore",
                    "container_id": container.get("environment_id"),
                    "files_restored": 0,
                }
            
            # 3. Write all files to container
            restored_count = 0
            errors = []
            
            for artifact in artifacts:
                path = artifact.get("path")
                if not path:
                    continue

                # A file spilled to object storage has no inline content; stream
                # it straight into the container instead of skipping it, which
                # would drop the file from the restore.
                if artifact.get("spilled") and artifact.get("archive_ref_id"):
                    try:
                        if await self.container_manager.hydrate_spilled_file(
                            project_id, path, artifact["archive_ref_id"], self.storage
                        ):
                            restored_count += 1
                        else:
                            errors.append({"path": path, "error": "hydrate_failed"})
                    except Exception as e:
                        errors.append({"path": path, "error": str(e)})
                        logger.warning(f"Failed to hydrate {path}: {e}")
                    continue

                content = artifact.get("content")
                if content is None:
                    continue

                try:
                    await self.container_manager.write_file_in_container(
                        project_id, path, content
                    )
                    restored_count += 1
                except Exception as e:
                    errors.append({"path": path, "error": str(e)})
                    logger.warning(f"Failed to restore {path}: {e}")
            
            # Update cache
            if project_id in self._project_cache:
                self._project_cache[project_id].container_ready = True
                self._project_cache[project_id].needs_container_recovery = False
                self._project_cache[project_id].recovery_in_progress = False
            
            # Emit recovery completed event
            if self.event_emitter:
                await self.event_emitter.emit("container_recovery_completed", None, {
                    "project_id": project_id,
                    "files_restored": restored_count,
                    "errors": len(errors),
                })
            
            logger.info(f"Container recovery completed: {restored_count} files restored")
            
            return {
                "status": "ready",
                "container_id": container.get("environment_id"),
                "files_restored": restored_count,
                "errors": errors if errors else None,
            }
            
        except Exception as e:
            logger.error(f"Container recovery failed: {e}")
            
            if project_id in self._project_cache:
                self._project_cache[project_id].recovery_in_progress = False
            
            if self.event_emitter:
                await self.event_emitter.emit("container_recovery_failed", None, {
                    "project_id": project_id,
                    "error": str(e),
                })
            
            return {"status": "failed", "reason": str(e)}
    
    async def _load_artifacts_from_repo(self, project_id: str) -> List[Dict[str, Any]]:
        """Load artifacts from git repo (container-use branch) as fallback."""
        import subprocess
        from pathlib import Path
        
        try:
            status = await self.container_manager.get_container_status(project_id)
            repo_path = status.get("repo_path")
            if not repo_path:
                return []
            
            base = Path(repo_path)
            if not base.exists() or not (base / ".git").exists():
                return []
            
            # Checkout container-use branch (has latest files)
            result = subprocess.run(
                ["git", "branch", "-a", "--list", "*container-use/*"],
                cwd=str(base), capture_output=True, text=True
            )
            cu_branches = [b.strip().lstrip("* ") for b in result.stdout.strip().splitlines() if b.strip()]
            
            if cu_branches:
                latest_branch = cu_branches[-1]
                subprocess.run(["git", "checkout", latest_branch], cwd=str(base), capture_output=True)
            
            # Read files using centralized config
            from config.artifacts import should_include_file, has_conflict_markers, MAX_FILES_HARD_LIMIT
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
                        if len(artifacts) >= MAX_FILES_HARD_LIMIT:
                            break
                except Exception:
                    continue
            
            return artifacts
        except Exception as e:
            logger.warning(f"_load_artifacts_from_repo failed: {e}")
            return []
    
    def build_recovery_prompt(self, state: ProjectState) -> Optional[Dict[str, Any]]:
        """Build a user-friendly recovery prompt based on project state."""
        prompts = []
        
        if state.needs_container_recovery:
            prompts.append({
                "type": "container_recovery",
                "message": "Your project environment needs to be restored. "
                          f"We have {len(state.artifacts)} files saved that can be recovered.",
                "action": "recover_container",
                "auto_recover": True,  # Can auto-recover containers
            })
        
        if state.needs_deployment_recovery:
            deployment = state.deployment or {}
            prompts.append({
                "type": "deployment_recovery",
                "message": "Your deployment is down. Would you like to redeploy?",
                "action": "redeploy",
                "deployment_id": deployment.get("deployment_id"),
                "slug": deployment.get("slug"),
                "auto_recover": False,  # Requires user confirmation
            })
        
        if not prompts:
            return None
        
        return {
            "needs_recovery": True,
            "prompts": prompts,
        }
    
    def get_project_status(self, project_id: str) -> Dict[str, Any]:
        """Get current status of a loaded project."""
        if project_id not in self._project_cache:
            return {"status": "not_loaded", "project_id": project_id}
        
        state = self._project_cache[project_id]
        
        # Determine overall status
        if state.recovery_in_progress:
            status = "recovering"
        elif state.needs_container_recovery or state.needs_deployment_recovery:
            status = "needs_recovery"
        elif state.container_ready:
            status = "ready"
        else:
            status = "loaded"
        
        return {
            "status": status,
            "project_id": project_id,
            "container_ready": state.container_ready,
            "deployment_ready": state.deployment_ready,
            "needs_container_recovery": state.needs_container_recovery,
            "needs_deployment_recovery": state.needs_deployment_recovery,
            "recovery_in_progress": state.recovery_in_progress,
            "artifact_count": len(state.artifacts),
            "recovery_prompt": self.build_recovery_prompt(state),
        }
    
    def unload_project(self, project_id: str) -> None:
        """Remove project from cache (e.g., after long inactivity)."""
        self._project_cache.pop(project_id, None)
        self.loaded_projects.discard(project_id)
    
    def is_loaded(self, project_id: str) -> bool:
        """Check if a project is currently loaded."""
        return project_id in self.loaded_projects


class ProjectNotFoundError(Exception):
    """Raised when a project is not found in the database."""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        super().__init__(f"Project not found: {project_id}")
