"""
Project Manager

Handles project lifecycle:
- Start new projects
- Cancel projects
- Generate project titles
- State recovery from database
- Agent cloning per project
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import uuid
import logging
import copy

from context.shared_context import SharedContext
from orchestration.agent_clone import clone_agent_for_task
from orchestration.cancellation import CancellationToken
from orchestration.backend_boot import get_backend_boot_id
from orchestration.workflow_task_lifecycle import (
    ABNORMAL_TERMINAL_RESUME_BLOCKED,
    UNEXPECTED_CANCEL_REASON,
    WorkflowTaskOutcome,
    classify_workflow_task_outcome,
    is_workflow_task_explicit_cancel,
    mark_workflow_task_explicit_cancel,
)
from schemas import EventSchema
from storage.mongo_backend import MongoStorageBackend
from storage.project_finish import TERMINAL_PROJECT_STATUSES
from tools.agent_allowed_tools import (
    apply_agent_tool_allowlist_normalization,
    effective_allowed_tool_ids,
)
from config.agent_loader import load_tenant_agent_prototypes
from llm.agent_model_params import (
    EffectiveAgentModelParams,
)
from llm.effective_agent_model_validation import (
    AgentModelParamsTarget,
    validate_effective_agent_model_params,
)
from utils.run_config import normalize_run_config

logger = logging.getLogger(__name__)


class ProjectManager:
    """
    Manages project lifecycle and state.
    
    Handles project creation, cancellation, and recovery.
    """
    
    def __init__(
            self,
            storage_backend,
            llm_client,
            event_emitter,
            container_manager=None,
            message_store=None,
            run_config_store=None,
    ):
        self.storage = storage_backend
        self.llm_client = llm_client
        self.event_emitter = event_emitter
        self.container_manager = container_manager
        self.message_store = message_store
        self.run_config_store = run_config_store or getattr(storage_backend, "run_config_store", None)

        # Active projects (in-memory state)
        self.active_projects: Dict[str, Dict] = {}
        
        # Track running workflow tasks and per-project locks
        self.project_tasks: Dict[str, asyncio.Task] = {}
        self.project_locks: Dict[str, asyncio.Lock] = {}
        # Set synchronously in done-callback before async finalize; blocks rehydrate
        # spawn while terminal state is still being persisted (AppFactory-274).
        self._pending_workflow_finalizes: set[str] = set()

    def _get_run_config_store(self):
        """Return injected run config store or storage-backed fallback."""
        return self.run_config_store or getattr(self.storage, "run_config_store", None)

    @staticmethod
    def _get_project_model_id(project_data: Dict[str, Any]) -> Optional[str]:
        """Return canonical model_id, falling back to legacy model_override only if absent."""
        if "model_id" in project_data:
            return project_data.get("model_id")
        return project_data.get("model_override")

    async def _load_run_config(self, project_id: str, run_config_id: Optional[str]) -> Dict[str, Any]:
        """Load and normalize run configuration for a project."""
        if not run_config_id:
            return {}

        run_config_store = self._get_run_config_store()
        if not run_config_store:
            logger.warning(
                "[RUN_CONFIG] project_id=%s run_config_id=%s - run_config_store unavailable, skipping config load",
                project_id,
                run_config_id,
            )
            return {}

        run_config = await run_config_store.get_config(run_config_id)
        if run_config is None:
            logger.warning(
                "[RUN_CONFIG] project_id=%s run_config_id=%s - config not found during load",
                project_id,
                run_config_id,
            )
            return {}

        return normalize_run_config(run_config)

    @staticmethod
    def _resolve_approval_mode(
        requested: Optional[str],
        run_config: Optional[Dict[str, Any]],
    ) -> str:
        """Effective approval mode: explicit request > run config > "human".

        Only "human"/"auto" count at each tier; anything else (including None,
        meaning "the caller didn't choose") falls through to the next tier, so a
        stray or unknown value can never silently disable the human gates.
        """
        valid = {"human", "auto"}
        if requested in valid:
            return requested  # type: ignore[return-value]
        rc_mode = (run_config or {}).get("approval_mode")
        if rc_mode in valid:
            return rc_mode
        return "human"

    def is_cancelled(self, project_id: str) -> bool:
        """Check if a project has been cancelled."""
        proj = self.active_projects.get(project_id)
        return bool(proj and proj.get("cancelled"))

    @staticmethod
    def _runtime_deps_from_agent_pool(agent_pool: List[Any]) -> Dict[str, Any]:
        """Copy injected runtime deps from the orchestrator's registered agent pool."""
        for agent in agent_pool or []:
            if getattr(agent, "llm_client", None) is not None:
                return {
                    "llm_client": agent.llm_client,
                    "tool_registry": getattr(agent, "tool_registry", None),
                    "event_emitter": getattr(agent, "event_emitter", None),
                    "mcp_executor": getattr(agent, "mcp_executor", None),
                }
        return {}

    async def _resolve_agent_pool_for_project(
        self,
        agent_pool: List[Any],
        tenant_id: Optional[str],
    ) -> List[Any]:
        """Return tenant-resolved prototypes when ``tenant_id`` is set."""
        tid = str(tenant_id or "").strip()
        if not tid or not self.storage:
            return agent_pool
        prototypes = await load_tenant_agent_prototypes(self.storage, tid)
        deps = self._runtime_deps_from_agent_pool(agent_pool)
        llm_client = deps.get("llm_client") or self.llm_client
        event_emitter = deps.get("event_emitter") or self.event_emitter
        tool_registry = deps.get("tool_registry")
        mcp_executor = deps.get("mcp_executor")
        for agent in prototypes:
            agent.inject_dependencies(
                shared_context=None,
                tool_registry=tool_registry,
                llm_client=llm_client,
                event_emitter=event_emitter,
                mcp_executor=mcp_executor,
            )
        logger.info(
            "[PROJECT] tenant_id=%s resolved_prototypes=%d runtime_deps_injected=%s",
            tid,
            len(prototypes),
            llm_client is not None,
        )
        return prototypes

    async def validate_effective_agent_model_params(
        self,
        agent_pool: List[Any],
        *,
        run_config: Optional[Dict[str, Any]],
        project_model: Optional[str],
        force_project_model: bool,
        project_reasoning_effort: Optional[str],
        project_temperature: Optional[float] = None,
    ) -> Dict[str, EffectiveAgentModelParams]:
        """Validate the fully resolved model triple for every project agent."""
        targets: list[AgentModelParamsTarget] = []
        for agent in agent_pool:
            runtime_id = str(getattr(agent, "agent_id", "") or "")
            agent_id = runtime_id.split("@", 1)[0]
            config = getattr(agent, "config", {}) or {}
            targets.append(
                AgentModelParamsTarget(
                    agent_id=agent_id,
                    model=getattr(agent, "model", None) or config.get("model"),
                    temperature=getattr(
                        agent,
                        "temperature",
                        config.get("temperature"),
                    ),
                    reasoning_effort=getattr(
                        agent,
                        "reasoning_effort",
                        config.get("reasoning_effort"),
                    ),
                )
            )

        return await validate_effective_agent_model_params(
            targets,
            run_config=run_config,
            storage=self.storage,
            project_model=project_model,
            force_project_model=force_project_model,
            project_reasoning_effort=project_reasoning_effort,
            project_temperature=project_temperature,
            allow_legacy_reasoning_fallback=True,
        )
    
    def create_project_agents(
            self,
            project_id: str,
            shared_context: SharedContext,
            token: CancellationToken,
            agent_pool: List[Any],
    ) -> List[Any]:
        """Create per-project agent clones with injected context and token."""
        clones: List[Any] = []
        for proto in agent_pool:
            a = clone_agent_for_task(proto)
            a.allowed_tools = list(getattr(proto, "allowed_tools", []) or [])
            a.allowed_mcp_tools = list(getattr(proto, "allowed_mcp_tools", []) or [])
            targets = getattr(proto, "allowed_delegation_targets", None)
            a.allowed_delegation_targets = list(targets) if isinstance(targets, list) else targets
            a.config = apply_agent_tool_allowlist_normalization(
                copy.deepcopy(getattr(proto, "config", {}) or {})
            )
            a.allowed_tools = list(a.config.get("allowed_tools") or [])
            a.allowed_mcp_tools = list(a.config.get("allowed_mcp_tools") or [])
            a._effective_allowed_tools = effective_allowed_tool_ids(a.config)
            a.shared_context = shared_context
            try:
                setattr(a, "cancellation_token", token)
            except Exception:
                pass
            try:
                a.agent_id = f"{getattr(a, 'agent_id', 'agent')}@{project_id}"
            except Exception:
                pass
            # Preserve orchestrator-injected runtime deps from prototype
            if getattr(proto, "llm_client", None):
                a.llm_client = proto.llm_client
            if getattr(proto, "tool_registry", None):
                a.tool_registry = proto.tool_registry
            if getattr(proto, "event_emitter", None):
                a.event_emitter = proto.event_emitter
            if hasattr(proto, "mcp_executor") and proto.mcp_executor:
                a.mcp_executor = proto.mcp_executor
            if hasattr(proto, "deploy_service") and proto.deploy_service:
                a.deploy_service = proto.deploy_service
            clones.append(a)
        for clone in clones:
            try:
                clone.agent_pool = clones
            except Exception:
                pass
        return clones
    
    def reinstate_runtime(self, project_id: str, sc: SharedContext, agent_pool: List[Any]) -> None:
        """Reinstate runtime state after revert (clear cancel flags, fresh token and clones)."""
        proj = self.active_projects.get(project_id)
        if not proj:
            return
        proj["cancelled"] = False
        # Bump epoch so in-flight abnormal finalizers from the pre-revert workflow
        # cannot overwrite the restored project with failed.
        proj["workflow_epoch"] = int(proj.get("workflow_epoch") or 0) + 1
        setattr(sc, "_cancelled", False)
        # Fresh token and clones
        token = CancellationToken()
        pool = proj.get("resolved_agent_pool") or agent_pool
        clones = self.create_project_agents(project_id, sc, token, pool)
        proj["token"] = token
        proj["agents"] = clones

    async def generate_project_title(
            self,
            user_prompt: str,
            api_key_override: Optional[str] = None,
            fallback_models_override: Optional[List[str]] = None,
            run_config: Optional[Dict[str, Any]] = None,
            base_model: Optional[str] = None,
            force_model: bool = False,
    ) -> str:
        """
        Generate a concise project title from user prompt using gpt-5-nano.

        Args:
            user_prompt: User's project description
            force_model: When True, base_model wins over run_config entries.

        Returns:
            Generated project title (max 60 chars)
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You generate concise, descriptive project titles. "
                               "Output ONLY the title, nothing else. "
                               "Keep it under 60 characters. "
                               "Use title case. No quotes or punctuation at the end."
                },
                {
                    "role": "user",
                    "content": f"Create a project title for this request:\n\n{user_prompt}"
                }
            ]
            if force_model and base_model:
                model = base_model
                fallback_models_override = []
            else:
                normalized_run_config = normalize_run_config(run_config)
                models = normalized_run_config.get("models", {})
                model = models.get("project_name") or models.get("default") or base_model or "gpt-5-nano"

            response = await self.llm_client.chat_completion(
                messages=messages,
                model=model,
                temperature=1.0,
                api_key_override=api_key_override,
                fallback_models_override=fallback_models_override,
            )
            
            title = response.get("content", "") if isinstance(response, dict) else response
            title = title.strip().strip('"').strip("'")
            if len(title) > 60:
                title = title[:57] + "..."
            
            return title
            
        except Exception as e:
            fallback_title = user_prompt[:50] + "..." if len(user_prompt) > 50 else user_prompt
            print(f"⚠️  Title generation failed: {e}. Using fallback: {fallback_title}")
            return fallback_title

    async def start_project(
        self,
        user_prompt: str,
        agent_pool: List[Any],
        project_name: Optional[str] = None,
        approval_mode: Optional[str] = None,
        api_key_override: Optional[str] = None,
        model_override: Optional[str] = None,
        fallback_models_override: Optional[List[str]] = None,
        force_model: bool = False,
        workflow_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        run_config_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        created_by: Optional[str] = None,
    ) -> str:
        """
        Start a new project.
        
        Args:
            user_prompt: User's project description
            agent_pool: List of prototype agents to clone
            project_name: Optional project name
            approval_mode: "human" (blocks for approval), "auto" (skips approval
                gates), or None to inherit from the run config (falling back to
                "human" when neither is set)
            api_key_override: Optional API key override

        Returns:
            project_id: Unique identifier for this project
        """
        project_id = str(uuid.uuid4())

        # Get config
        normalized_run_config = await self._load_run_config(project_id, run_config_id)

        # Precedence: explicit request value > run config's approval_mode > "human".
        approval_mode = self._resolve_approval_mode(approval_mode, normalized_run_config)

        print(f"🚀 Starting project with approval_mode={approval_mode}")

        # Resolve and validate before title generation or run persistence so a
        # bad project/run override cannot leave partially-created state.
        resolved_pool = await self._resolve_agent_pool_for_project(agent_pool, tenant_id)
        effective_agent_model_params = await self.validate_effective_agent_model_params(
            resolved_pool,
            run_config=normalized_run_config,
            project_model=model_override,
            force_project_model=force_model,
            project_reasoning_effort=reasoning_effort,
            project_temperature=temperature,
        )
        for agent_id, effective in effective_agent_model_params.items():
            logger.info(
                "[MODEL_PARAMS] project_id=%s agent=%s model=%s temperature=%s "
                "reasoning_effort=%s sources=%s",
                project_id,
                agent_id,
                effective.model,
                effective.temperature,
                effective.reasoning_effort,
                effective.sources,
            )

        # Generate project title
        project_title = await self.generate_project_title(
            user_prompt,
            api_key_override=api_key_override,
            fallback_models_override=fallback_models_override,
            run_config=normalized_run_config,
            base_model=model_override,
            force_model=force_model,
        )

        # Create the run BEFORE the shared context: the runner reads run_id
        # from the context (not the project dict), and without it the tool
        # ledger is off for the whole run — ask_human can't park.
        run_id = str(uuid.uuid4())
        try:
            await self.storage.create_run(
                project_id=project_id,
                run_id=run_id,
                parent_run_id=None,
                forked_from_conversation_index=None
            )
            # Set initial run status to "running" (State Simplification 2.4.1)
            await self.storage.update_run_status(run_id, "running")
            if isinstance(self.storage, MongoStorageBackend):
                await self.storage.stamp_run_backend_boot_id(
                    run_id, get_backend_boot_id()
                )
        except Exception as e:
            # A project without a run doc can't journal tool calls or park
            # human questions — fail the start honestly instead of running
            # in a silently degraded mode.
            raise RuntimeError(
                f"Cannot start project {project_id}: run creation failed: {e}"
            ) from e

        # Create shared context
        shared_context = SharedContext(
            project_id,
            self.storage,
            run_id=run_id,
            message_store=self.message_store,
            run_config=normalized_run_config,
        )
        await shared_context.initialize(user_prompt)
        # Validation can omit an inherited effort that is incompatible with a
        # project-forced model. Keep that resolved value on the project context
        # so agent runners never reconstruct and send the stale value.
        setattr(
            shared_context,
            "_effective_agent_model_params",
            effective_agent_model_params,
        )
        
        # Store ephemeral API key on shared context
        if api_key_override:
            setattr(shared_context, "_ephemeral_api_key", api_key_override)
        if fallback_models_override is not None:
            setattr(shared_context, "_ephemeral_fallback_models", fallback_models_override)

        # Store the UI-selected model as a project-level fallback.
        # Run configuration models normally take priority during resolution,
        # unless `force_model=True` — then the user-selected model wins.
        if model_override:
            setattr(shared_context, "_model_override", model_override)
        if force_model and model_override:
            setattr(shared_context, "_force_model_override", True)
        # Project-level reasoning effort beats per-agent static config.
        # Stored on SharedContext so any agent reading via
        # `BaseAgent._resolve_reasoning_effort` picks it up automatically.
        if reasoning_effort:
            setattr(shared_context, "_reasoning_effort_override", reasoning_effort)
        if temperature is not None:
            setattr(shared_context, "_temperature_override", temperature)
        if tenant_id:
            setattr(shared_context, "_tenant_id", tenant_id)

        # Create per-project cancellation token and agent clones
        token = CancellationToken()
        project_agents = self.create_project_agents(
            project_id, shared_context, token, resolved_pool
        )

        # Store project
        self.active_projects[project_id] = {
            "project_id": project_id,
            "run_id": run_id,
            "user_prompt": user_prompt,
            "title": project_title,
            "shared_context": shared_context,
            "status": "initialized",
            "created_at": datetime.utcnow().isoformat(),
            "current_phase": "requirements",
            "approval_mode": approval_mode,
            "api_key_override": api_key_override,
            "model_id": model_override,
            "force_model": bool(force_model and model_override),
            "reasoning_effort": reasoning_effort,
            "temperature": temperature,
            "workflow_id": workflow_id,
            "tenant_id": tenant_id,
            "run_config_id": run_config_id,
            "agents": project_agents,
            "resolved_agent_pool": resolved_pool,
            "token": token
        }
        
        # Create container for project
        if self.container_manager and self.container_manager.enabled:
            try:
                container_info = await self.container_manager.get_or_create_container(project_id)
                env_id = container_info.get("environment_id") if isinstance(container_info, dict) else None
                repo_path = container_info.get("repo_path") if isinstance(container_info, dict) else None
                
                try:
                    await self.storage.save_container_log(project_id, "created", container_info or {})
                except Exception:
                    pass
                
                await self.event_emitter.emit(EventSchema.CONTAINER_CREATED, run_id, {
                    "project_id": project_id,
                    "branch": f"AppFactory-{project_id}",
                    "environment_id": env_id,
                    "repo_path": repo_path,
                })

                # STRICT: If container is not ready, halt immediately
                if not env_id:
                    self.active_projects[project_id]["status"] = "failed"
                    await self.storage.save_project(
                        project_id,
                        self.active_projects[project_id],
                        actor_id=created_by,
                    )
                    await self.event_emitter.emit(EventSchema.PROJECT_FAILED, run_id, {
                        "project_id": project_id,
                        "error": "Container created event returned null environment_id"
                    })
                    return project_id
            except Exception as e:
                print(f"⚠️  Container creation failed: {e}. Falling back to simulation.")

        # Sync initial project state to DB immediately (bulletproof persistence)
        # The first save of the project fields is the only one that records its author.
        await self._sync_project_to_db(project_id, actor_id=created_by)

        await self.event_emitter.emit(EventSchema.PROJECT_STARTED, run_id, {
            "project_id": project_id,
            "user_prompt": user_prompt
        })
        
        return project_id
    
    async def cancel_project(
        self,
        project_id: str,
        reason: str = "Cancelled by user",
        *,
        already_locked: bool = False,
    ):
        """Cancel a running project.

        ``already_locked=True`` when the caller already holds ``get_lock(project_id)``
        (e.g. revert paths) — asyncio.Lock is not reentrant.
        """
        if already_locked:
            await self._cancel_project_unlocked(project_id, reason)
            return
        async with self.get_lock(project_id):
            await self._cancel_project_unlocked(project_id, reason)

    async def _cancel_project_unlocked(self, project_id: str, reason: str):
        """Cancel body; caller must own the project lock (or accept races)."""
        proj = self.active_projects.get(project_id)
        if not proj:
            return
        
        await self.event_emitter.emit(EventSchema.PROJECT_STOPPING, proj.get("run_id"), {"project_id": project_id, "reason": reason})

        proj["status"] = "cancelled"
        proj["cancelled"] = True
        
        sc = proj.get("shared_context")
        try:
            if sc is not None:
                setattr(sc, "_cancelled", True)
        except Exception:
            pass
        
        # Fire per-project cancellation token
        try:
            token = proj.get("token")
            if token:
                token.cancel()
        except Exception:
            pass
        
        task = self.project_tasks.get(project_id)
        if task and not task.done():
            try:
                mark_workflow_task_explicit_cancel(task)
                task.cancel()
            except Exception:
                pass
        
        await self.storage.save_project(project_id, {
            "user_prompt": proj.get("user_prompt", ""),
            "title": proj.get("title", "Untitled Project"),
            "status": "cancelled",
            "current_phase": proj.get("current_phase"),
            "approval_mode": proj.get("approval_mode", "human"),
            "workflow_id": proj.get("workflow_id"),
            "run_config_id": proj.get("run_config_id"),
            "tenant_id": proj.get("tenant_id"),
            "created_at": proj.get("created_at"),
            "metadata": {"phase": proj.get("current_phase")},
        })
        
        # Also update run status (State Simplification 2.4.1)
        run_id = proj.get("run_id")
        if run_id:
            try:
                await self.storage.update_run_status(run_id, "cancelled")
            except Exception as e:
                logger.warning(f"Failed to update run status on cancel: {e}")
        
        await self.event_emitter.emit(EventSchema.PROJECT_STOPPED, proj.get("run_id"), {
            "project_id": project_id,
            "reason": reason,
        })

    async def wait_for_task_completion(self, project_id: str, timeout: float = 10.0) -> bool:
        """Wait for the active workflow task to stop after cancellation."""
        task = self.project_tasks.get(project_id)
        if not task:
            logger.info("[EXECUTE] project_id=%s - no workflow task to await", project_id)
            return True
        if task.done():
            return True

        logger.info(
            "[EXECUTE] project_id=%s - waiting for workflow task to stop (timeout=%ss)",
            project_id,
            timeout,
        )
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            return True
        except asyncio.CancelledError:
            logger.info("[EXECUTE] project_id=%s - workflow task stopped via cancellation", project_id)
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "[EXECUTE] project_id=%s - workflow task did not stop within %ss",
                project_id,
                timeout,
            )
            return False
        except Exception as e:
            logger.warning(
                "[EXECUTE] project_id=%s - workflow task ended with error while stopping: %s",
                project_id,
                e,
            )
            return True

    async def recover_state_from_db(self, agent_pool: List[Any]):
        """
        Recover running projects and pending approvals from MongoDB.
        
        Call this on server startup to restore state after restart.
        """
        logger.info("🔄 Recovering state from MongoDB...")
        
        # Recover running projects
        running_projects = await self.storage.find_running_projects()
        logger.info(f"Found {len(running_projects)} running projects")
        
        for proj_data in running_projects:
            project_id = proj_data["project_id"]
            normalized_run_config = await self._load_run_config(project_id, proj_data.get("run_config_id"))

            # Reconstruct SharedContext on the project's active run. A project
            # without one cannot journal tool calls or park human questions —
            # refuse to load it at all rather than run it degraded; any later
            # touch fails loudly in load_project_to_active.
            active_run = await self.storage.get_active_run(project_id)
            active_run_id = (active_run or {}).get("run_id")
            if not active_run_id:
                logger.error(
                    "❌ recover_state: project %s is 'running' but has no active run — "
                    "skipping recovery instead of loading it with a dead tool ledger",
                    project_id,
                )
                continue
            shared_context = SharedContext(
                project_id,
                self.storage,
                run_id=active_run_id,
                message_store=self.message_store,
                run_config=normalized_run_config,
            )
            await shared_context.load_from_db()

            selected_model = self._get_project_model_id(proj_data)
            force_model_flag = bool(proj_data.get("force_model")) and bool(selected_model)
            reasoning_effort = proj_data.get("reasoning_effort")
            temperature = proj_data.get("temperature")
            if selected_model:
                setattr(shared_context, "_model_override", selected_model)
            if force_model_flag:
                setattr(shared_context, "_force_model_override", True)
            if reasoning_effort:
                setattr(shared_context, "_reasoning_effort_override", reasoning_effort)
            if temperature is not None:
                setattr(shared_context, "_temperature_override", temperature)
            if proj_data.get("tenant_id"):
                setattr(shared_context, "_tenant_id", proj_data.get("tenant_id"))

            # Restore to active_projects
            token = CancellationToken()
            self.active_projects[project_id] = {
                "project_id": project_id,
                "user_prompt": proj_data.get("user_prompt", ""),
                "title": proj_data.get("title", "Untitled Project"),
                "shared_context": shared_context,
                "run_id": shared_context.run_id,
                "status": proj_data.get("status", "running"),
                "created_at": proj_data.get("created_at"),
                "current_phase": proj_data.get("current_phase", "requirements"),
                "approval_mode": proj_data.get("approval_mode", "human"),
                "model_id": selected_model,
                "force_model": force_model_flag,
                "reasoning_effort": reasoning_effort,
                "temperature": temperature,
                "workflow_id": proj_data.get("workflow_id"),
                "run_config_id": proj_data.get("run_config_id"),
                "tenant_id": proj_data.get("tenant_id"),
                "token": token
            }

            # Create per-project agents
            resolved_pool = await self._resolve_agent_pool_for_project(
                agent_pool,
                proj_data.get("tenant_id"),
            )
            self.active_projects[project_id]["resolved_agent_pool"] = resolved_pool
            self.active_projects[project_id]["agents"] = self.create_project_agents(
                project_id, shared_context, token, resolved_pool
            )
            
            logger.info(f"✅ Recovered project: {project_id} (phase: {proj_data.get('current_phase')})")
        
        logger.info(f"🎉 State recovery complete: {len(self.active_projects)} projects")
        return len(self.active_projects)
    
    def get_project(self, project_id: str) -> Optional[Dict]:
        """Get an active project by ID."""
        return self.active_projects.get(project_id)
    
    async def load_project_to_active(self, project_id: str, agent_pool: List[Any]) -> Optional[Dict]:
        """
        Load a single project from DB into active_projects.
        
        Used for lazy loading when a project isn't in memory.
        """
        proj_data = await self.storage.load_project(project_id)
        if not proj_data:
            return None
        normalized_run_config = await self._load_run_config(project_id, proj_data.get("run_config_id"))

        # Reconstruct SharedContext on the project's active run. No active run
        # means the project cannot journal tool calls or park human questions:
        # fail the load loudly — callers surface it as a failed request — never
        # hand back a context with the ledger silently off.
        active_run = await self.storage.get_active_run(project_id)
        active_run_id = (active_run or {}).get("run_id")
        if not active_run_id:
            logger.error(
                "load_project_to_active: project %s has no active run — refusing to load",
                project_id,
            )
            raise RuntimeError(f"project {project_id} has no active run — cannot load it")
        shared_context = SharedContext(
            project_id,
            self.storage,
            run_id=active_run_id,
            message_store=self.message_store,
            run_config=normalized_run_config,
        )
        await shared_context.load_from_db()

        selected_model = self._get_project_model_id(proj_data)
        force_model_flag = bool(proj_data.get("force_model")) and bool(selected_model)
        reasoning_effort = proj_data.get("reasoning_effort")
        temperature = proj_data.get("temperature")
        if selected_model:
            setattr(shared_context, "_model_override", selected_model)
        if force_model_flag:
            setattr(shared_context, "_force_model_override", True)
        if reasoning_effort:
            setattr(shared_context, "_reasoning_effort_override", reasoning_effort)
        if temperature is not None:
            setattr(shared_context, "_temperature_override", temperature)
        if proj_data.get("tenant_id"):
            setattr(shared_context, "_tenant_id", proj_data.get("tenant_id"))

        # Restore to active_projects
        token = CancellationToken()
        self.active_projects[project_id] = {
            "project_id": project_id,
            "user_prompt": proj_data.get("user_prompt", ""),
            "title": proj_data.get("title", "Untitled Project"),
            "shared_context": shared_context,
            "run_id": shared_context.run_id,
            "status": proj_data.get("status", "initialized"),
            "created_at": proj_data.get("created_at"),
            "current_phase": proj_data.get("current_phase", "requirements"),
            "approval_mode": proj_data.get("approval_mode", "human"),
            "model_id": selected_model,
            "force_model": force_model_flag,
            "reasoning_effort": reasoning_effort,
            "temperature": temperature,
            "workflow_id": proj_data.get("workflow_id"),
            "run_config_id": proj_data.get("run_config_id"),
            "tenant_id": proj_data.get("tenant_id"),
            "token": token,
            "_reconstructed": True
        }
        
        self.active_projects[project_id]["resolved_agent_pool"] = await self._resolve_agent_pool_for_project(
            agent_pool,
            proj_data.get("tenant_id"),
        )
        self.active_projects[project_id]["agents"] = self.create_project_agents(
            project_id,
            shared_context,
            token,
            self.active_projects[project_id]["resolved_agent_pool"],
        )
        
        logger.info(f"✅ Loaded project to active: {project_id}")
        return self.active_projects[project_id]
    
    def get_lock(self, project_id: str) -> asyncio.Lock:
        """Get or create a lock for a project."""
        return self.project_locks.setdefault(project_id, asyncio.Lock())

    def workflow_finalize_pending(self, project_id: str) -> bool:
        """True while a workflow task done-callback is persisting abnormal terminal state."""
        return project_id in self._pending_workflow_finalizes
    
    # === State Sync Methods (Bulletproof Persistence) ===
    
    async def _sync_project_to_db(
        self, project_id: str, actor_id: Optional[str] = None
    ) -> None:
        """
        Sync current in-memory project state to MongoDB.
        
        Called after every state mutation to ensure DB reflects memory.
        This is the core of bulletproof persistence - every change persists.
        """
        proj = self.active_projects.get(project_id)
        if not proj:
            return
        
        try:
            await self.storage.save_project(project_id, {
                "user_prompt": proj.get("user_prompt", ""),
                "title": proj.get("title", "Untitled Project"),
                "status": proj.get("status", "initialized"),
                "current_phase": proj.get("current_phase", "requirements"),
                "approval_mode": proj.get("approval_mode", "human"),
                "model_id": proj.get("model_id"),
                "force_model": bool(proj.get("force_model")),
                "reasoning_effort": proj.get("reasoning_effort"),
                "temperature": proj.get("temperature"),
                "workflow_id": proj.get("workflow_id"),
                "run_config_id": proj.get("run_config_id"),
                "tenant_id": proj.get("tenant_id"),
                "created_at": proj.get("created_at"),
                "metadata": {"phase": proj.get("current_phase")},
            }, actor_id=actor_id)
            logger.debug(
                f"Project {project_id} synced to DB: status={proj.get('status')}, phase={proj.get('current_phase')}")
        except Exception as e:
            logger.error(f"Failed to sync project {project_id} to DB: {e}")
    
    async def update_project_phase(self, project_id: str, phase: str) -> None:
        """
        Update project phase atomically (memory + DB + run).
        
        Use this instead of directly modifying active_projects[project_id]["current_phase"].
        """
        proj = self.active_projects.get(project_id)
        if not proj:
            logger.warning(f"Cannot update phase for unknown project: {project_id}")
            return
        
        old_phase = proj.get("current_phase")
        proj["current_phase"] = phase
        await self._sync_project_to_db(project_id)
        
        # Also update run phase (State Simplification 2.4.1)
        run_id = proj.get("run_id")
        if run_id:
            try:
                await self.storage.update_run_phase(run_id, phase)
                logger.debug(f"Run {run_id} phase updated: {phase}")
            except Exception as e:
                logger.warning(f"Failed to update run phase: {e}")
        
        logger.info(f"Project {project_id} phase: {old_phase} → {phase}")
    
    async def _persist_terminal_status(
        self,
        project_id: str,
        status: str,
        *,
        run_id: str | None = None,
    ) -> bool:
        """Atomic terminal project+run persist when storage supports it."""
        proj = self.active_projects.get(project_id)
        run_id = run_id or (proj.get("run_id") if proj else None)
        if not run_id:
            return False
        if isinstance(self.storage, MongoStorageBackend):
            ok = await self.storage.transition_project_run_terminal(
                project_id,
                run_id,
                status,
            )
            if ok and proj:
                proj["status"] = status
                await self._sync_project_to_db(project_id)
            return ok
        try:
            await self.storage.update_run_status(run_id, status)
        except Exception as exc:
            logger.warning(
                "[WORKFLOW_TASK] project_id=%s terminal fallback run status err=%s",
                project_id,
                exc,
            )
            return False
        try:
            if proj:
                await self.storage.save_project(project_id, {
                    "user_prompt": proj.get("user_prompt", ""),
                    "title": proj.get("title", "Untitled Project"),
                    "status": status,
                    "current_phase": proj.get("current_phase", "requirements"),
                    "approval_mode": proj.get("approval_mode", "human"),
                    "model_id": proj.get("model_id"),
                    "force_model": bool(proj.get("force_model")),
                    "reasoning_effort": proj.get("reasoning_effort"),
                    "temperature": proj.get("temperature"),
                    "workflow_id": proj.get("workflow_id"),
                    "run_config_id": proj.get("run_config_id"),
                    "tenant_id": proj.get("tenant_id"),
                    "created_at": proj.get("created_at"),
                    "metadata": {"phase": proj.get("current_phase")},
                })
            else:
                doc = await self.storage.load_project(project_id)
                if not doc:
                    return False
                payload = dict(doc)
                payload["status"] = status
                await self.storage.save_project(project_id, payload)
        except Exception as exc:
            logger.warning(
                "[WORKFLOW_TASK] project_id=%s terminal fallback project save err=%s",
                project_id,
                exc,
            )
        if await self._durable_terminal_pair(project_id, run_id, status):
            if proj:
                proj["status"] = status
            return True
        return False

    async def update_project_status(self, project_id: str, status: str) -> None:
        """
        Update project status atomically (memory + DB + run).
        
        Use this instead of directly modifying active_projects[project_id]["status"].
        """
        proj = self.active_projects.get(project_id)
        if not proj:
            logger.warning(f"Cannot update status for unknown project: {project_id}")
            return
        
        old_status = proj.get("status")
        run_id = proj.get("run_id")
        if status in TERMINAL_PROJECT_STATUSES and run_id:
            proj["status"] = status
            persisted = False
            if isinstance(self.storage, MongoStorageBackend):
                try:
                    persisted = await self.storage.transition_project_run_terminal(
                        project_id, run_id, status
                    )
                    if persisted:
                        await self._sync_project_to_db(project_id)
                except Exception as exc:
                    logger.warning(
                        "[WORKFLOW_TASK] project_id=%s terminal transition err=%s",
                        project_id,
                        exc,
                    )
            if not persisted:
                await self._sync_project_to_db(project_id)
                try:
                    await self.storage.update_run_status(run_id, status)
                except Exception as e:
                    logger.warning(f"Failed to update run status: {e}")
                # Both docs must match — project-only success is a split-brain.
                persisted = await self._durable_terminal_pair(
                    project_id, run_id, status
                )
            if not persisted:
                logger.warning(
                    "[WORKFLOW_TASK] project_id=%s terminal transition incomplete status=%s",
                    project_id,
                    status,
                )
            logger.info(f"Project {project_id} status: {old_status} → {status}")
            return

        proj["status"] = status
        await self._sync_project_to_db(project_id)
        
        # Also update run status (State Simplification 2.4.1)
        if run_id:
            try:
                await self.storage.update_run_status(run_id, status)
                logger.debug(f"Run {run_id} status updated: {status}")
            except Exception as e:
                logger.warning(f"Failed to update run status: {e}")
        
        logger.info(f"Project {project_id} status: {old_status} → {status}")
    
    async def _durable_project_status(self, project_id: str, expected: str) -> bool:
        """True when MongoDB project document carries expected status."""
        try:
            doc = await self.storage.load_project(project_id)
            return bool(doc and doc.get("status") == expected)
        except Exception as exc:
            logger.warning(
                "[WORKFLOW_TASK] project_id=%s durable status check failed: %s",
                project_id,
                exc,
            )
            return False

    async def _durable_terminal_pair(
        self, project_id: str, run_id: str | None, expected: str
    ) -> bool:
        """True when both project doc and the specific run doc are terminal."""
        if not run_id:
            return False
        if not await self._durable_project_status(project_id, expected):
            return False
        try:
            get_run = getattr(self.storage, "get_run", None)
            if callable(get_run):
                run = await get_run(run_id)
            else:
                run = await self.storage.get_active_run(project_id)
                if run and run.get("run_id") != run_id:
                    return False
            return bool(run and run.get("run_status") == expected)
        except Exception as exc:
            logger.warning(
                "[WORKFLOW_TASK] project_id=%s run_id=%s durable run status check failed: %s",
                project_id,
                run_id,
                exc,
            )
            return False
    
    def register_workflow_task(self, project_id: str, task: asyncio.Task):
        """Register a workflow task for a project."""
        self.project_tasks[project_id] = task

        def _on_done(completed: asyncio.Task) -> None:
            try:
                if self.project_tasks.get(project_id) is completed:
                    self.project_tasks.pop(project_id, None)
            except Exception:
                pass
            proj = self.active_projects.get(project_id) or {}
            try:
                outcome = classify_workflow_task_outcome(
                    completed,
                    project_cancelled=bool(proj.get("cancelled")),
                )
            except Exception as exc:
                logger.warning(
                    "[WORKFLOW_TASK] project_id=%s classify failed: %s",
                    project_id,
                    exc,
                )
                outcome = WorkflowTaskOutcome.UNEXPECTED_CANCEL
            task_name = ""
            try:
                task_name = completed.get_name()
            except Exception:
                pass
            logger.info(
                "[WORKFLOW_TASK] project_id=%s run_id=%s phase=%s outcome=%s task=%s",
                project_id,
                proj.get("run_id"),
                proj.get("current_phase"),
                outcome.value,
                task_name or "workflow",
            )
            if is_workflow_task_explicit_cancel(completed):
                logger.info(
                    "[WORKFLOW_TASK] project_id=%s explicit task cancel — skip abnormal finalize",
                    project_id,
                )
                return
            if outcome in (WorkflowTaskOutcome.SUCCESS, WorkflowTaskOutcome.EXPLICIT_CANCEL):
                return
            registered = self.project_tasks.get(project_id)
            if registered is not None and registered is not completed:
                logger.info(
                    "[WORKFLOW_TASK] project_id=%s stale task done — skip abnormal finalize",
                    project_id,
                )
                return
            # Capture epoch at schedule time; revert bumps it under reinstate_runtime.
            epoch_at_schedule = int(proj.get("workflow_epoch") or 0)
            self._pending_workflow_finalizes.add(project_id)
            from telemetry.tracer import create_task_with_context

            try:
                create_task_with_context(
                    self._finalize_abnormal_workflow_outcome(
                        project_id, outcome, completed, epoch_at_schedule
                    )
                )
            except Exception as _spawn_exc:
                logger.error(
                    "[WORKFLOW_TASK] project_id=%s failed to spawn finalize task: %s — releasing pending latch",
                    project_id,
                    _spawn_exc,
                )
                self._pending_workflow_finalizes.discard(project_id)

        task.add_done_callback(_on_done)

    async def _finalize_abnormal_workflow_outcome(
        self,
        project_id: str,
        outcome: WorkflowTaskOutcome,
        task: asyncio.Task,
        epoch_at_schedule: int = 0,
    ) -> None:
        """Persist failed terminal state for unexpected cancel or task exception."""
        guard_set = False
        try:
            async with self.get_lock(project_id):
                registered = self.project_tasks.get(project_id)
                if registered is not None and registered is not task:
                    logger.info(
                        "[WORKFLOW_TASK] project_id=%s stale task finalize skipped — "
                        "newer task registered",
                        project_id,
                    )
                    # Latch belongs to this done-callback; newer task is not finalizing.
                    guard_set = True
                    return

                proj = self.active_projects.get(project_id) or {}
                current_epoch = int(proj.get("workflow_epoch") or 0)
                if current_epoch != epoch_at_schedule:
                    logger.info(
                        "[WORKFLOW_TASK] project_id=%s stale epoch finalize skipped — "
                        "scheduled=%s current=%s (revert/resume advanced generation)",
                        project_id,
                        epoch_at_schedule,
                        current_epoch,
                    )
                    guard_set = True
                    return

                run_id = proj.get("run_id")
                if not run_id:
                    try:
                        active_run = await self.storage.get_active_run(project_id)
                        run_id = (active_run or {}).get("run_id")
                    except Exception as exc:
                        logger.warning(
                            "[WORKFLOW_TASK] project_id=%s active-run lookup before abnormal finalize: %s",
                            project_id,
                            exc,
                        )

                try:
                    _doc = await self.storage.load_project(project_id)
                    _durable_status = (_doc or {}).get("status")
                except Exception as _exc:
                    logger.warning(
                        "[WORKFLOW_TASK] project_id=%s durable status check failed: %s",
                        project_id,
                        _exc,
                    )
                    _durable_status = None
                if _durable_status in ("failed", "cancelled", "completed"):
                    in_mem = self.active_projects.get(project_id)
                    if in_mem and in_mem.get("status") != _durable_status:
                        in_mem["status"] = _durable_status
                    logger.info(
                        "[WORKFLOW_TASK] project_id=%s outcome=%s project doc already %s — skip",
                        project_id,
                        outcome.value,
                        _durable_status,
                    )
                    # Durable terminal is the guard — do not keep the in-memory latch.
                    guard_set = True
                    return

                # Stop may have won under the same lock (or left cancelled flags).
                # Never overwrite cancelled → failed.
                in_mem = self.active_projects.get(project_id) or {}
                if (
                    in_mem.get("cancelled")
                    or in_mem.get("status") == "cancelled"
                    or _durable_status == "cancelled"
                ):
                    logger.info(
                        "[WORKFLOW_TASK] project_id=%s outcome=%s already cancelled — skip failed",
                        project_id,
                        outcome.value,
                    )
                    guard_set = True
                    return

                if outcome == WorkflowTaskOutcome.UNEXPECTED_CANCEL:
                    error = UNEXPECTED_CANCEL_REASON
                    traceback_text = ""
                else:
                    exc = task.exception()
                    error = str(exc) if exc else "workflow_task_failed"
                    traceback_text = ""
                    if exc is not None:
                        import traceback

                        traceback_text = "".join(
                            traceback.format_exception(type(exc), exc, exc.__traceback__)
                        )

                persisted = False
                if project_id in self.active_projects:
                    try:
                        await self.update_project_status(project_id, "failed")
                        persisted = await self._durable_terminal_pair(
                            project_id, run_id, "failed"
                        )
                    except Exception as exc:
                        logger.error(
                            "[WORKFLOW_TASK] project_id=%s persist failed status err=%s",
                            project_id,
                            exc,
                        )
                else:
                    logger.warning(
                        "[WORKFLOW_TASK] project_id=%s not in active_projects — "
                        "abnormal finalize via storage fallback only",
                        project_id,
                    )

                if not persisted and run_id:
                    try:
                        persisted = await self._persist_terminal_status(
                            project_id,
                            "failed",
                            run_id=run_id,
                        )
                        if persisted:
                            logger.warning(
                                "[WORKFLOW_TASK] project_id=%s run_id=%s abnormal outcome "
                                "persisted via storage fallback",
                                project_id,
                                run_id,
                            )
                        else:
                            logger.error(
                                "[WORKFLOW_TASK] project_id=%s run_id=%s abnormal outcome "
                                "partial persist: terminal transition failed",
                                project_id,
                                run_id,
                            )
                    except Exception as exc:
                        logger.error(
                            "[WORKFLOW_TASK] project_id=%s storage fallback for abnormal "
                            "outcome failed: %s",
                            project_id,
                            exc,
                        )

                if persisted:
                    guard_set = True
                    if run_id:
                        try:
                            await self.storage.clear_run_resume_blocked(run_id)
                        except Exception as exc:
                            logger.warning(
                                "[WORKFLOW_TASK] project_id=%s clear resume_blocked err=%s",
                                project_id,
                                exc,
                            )
                    try:
                        await self.event_emitter.emit(
                            EventSchema.PROJECT_FAILED,
                            run_id,
                            {
                                "project_id": project_id,
                                "error": error,
                                "traceback": traceback_text,
                                "outcome": outcome.value,
                            },
                        )
                    except Exception as exc:
                        logger.error(
                            "[WORKFLOW_TASK] project_id=%s emit project_failed err=%s",
                            project_id,
                            exc,
                        )
                    if self.message_store is not None:
                        try:
                            await self.message_store.append_system_message(
                                project_id,
                                "error",
                                error,
                                run_id=run_id,
                                data={"outcome": outcome.value},
                            )
                        except Exception as exc:
                            logger.warning(
                                "[WORKFLOW_TASK] project_id=%s append failure chat message err=%s",
                                project_id,
                                exc,
                            )
                else:
                    if run_id:
                        try:
                            await self.storage.set_run_resume_blocked(
                                run_id, ABNORMAL_TERMINAL_RESUME_BLOCKED
                            )
                            guard_set = True
                            logger.warning(
                                "[WORKFLOW_TASK] project_id=%s run_id=%s abnormal outcome "
                                "persist failed — resume blocked on run doc",
                                project_id,
                                run_id,
                            )
                        except Exception as exc:
                            logger.error(
                                "[WORKFLOW_TASK] project_id=%s set resume_blocked err=%s",
                                project_id,
                                exc,
                            )
                    logger.error(
                        "[WORKFLOW_TASK] project_id=%s outcome=%s terminal state not persisted",
                        project_id,
                        outcome.value,
                    )
                    # ponytail: latch always released in finally; durable resume_blocked
                    # (if written) is the guard across process restart.
        finally:
            # Always drop the in-memory latch. An uncaught exception / CancelledError
            # before guard_set must not freeze resume/rehydrate until process restart.
            # Durable terminal / resume_blocked (when written) remain the real guards.
            self._pending_workflow_finalizes.discard(project_id)
            if not guard_set:
                logger.warning(
                    "[WORKFLOW_TASK] project_id=%s finalize ended without durable "
                    "terminal guard — latch released anyway",
                    project_id,
                )
