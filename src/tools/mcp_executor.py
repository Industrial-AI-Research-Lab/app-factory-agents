"""
MCP Tool Executor

Bridges ToolRegistry and Container-Use MCP server for actual tool execution.
"""

import asyncio
import logging
import re
import time
from typing import Any, Dict, Optional, Set, Tuple
from urllib.parse import urlparse

from telemetry.tracer import get_tracer
from tools.agent_allowed_tools import known_builtin_tool_ids
from tools.external_call_outcome import log_raised_call, observed_call
from tools.external_mcp import (
    ExternalMCPClient,
    ExternalMCPConfig,
    MCPCallTimeoutError,
    stdio_docker_container_name,
    stdio_docker_remove_force,
)
from tools.mcp_runtime_scope import MCP_TENANT_SCOPE_PROJECT_ID
from tools.mcp_runtime_headers import runtime_headers_dict as _runtime_headers_dict
from tools.mcp_auth import build_mcp_auth

logger = logging.getLogger(__name__)
DEFAULT_EXTERNAL_MCP_IMAGE = "node:22-bookworm-slim"
REMOTE_CONNECT_RETRY_DELAYS = (1.0, 3.0)


def _is_tool_timeout_exception(exc: BaseException) -> bool:
    return isinstance(exc, (TimeoutError, asyncio.TimeoutError, MCPCallTimeoutError))


def _workspace_io_infra_fields(exc: BaseException) -> Dict[str, Any]:
    """Labels so listing/read failures are not mistaken for empty success."""
    from schemas.infra_error import InfraErrorType

    if _is_tool_timeout_exception(exc):
        return {"error_type": InfraErrorType.TIMEOUT.value}
    if isinstance(exc, ConnectionError):
        return {"error_type": InfraErrorType.CONNECTION_ERROR.value}
    if isinstance(exc, RuntimeError) and (
        "Container not available" in str(exc) or "Not connected" in str(exc)
    ):
        # Soft no-client / dead session: gate veto without hard-abort.
        return {"error_type": InfraErrorType.UNAVAILABLE.value}
    return {"outcome_unknown": True}


def _container_mutation_error(path: str, result: Any) -> Optional[Dict[str, Any]]:
    """Typed tool error when a container write/delete did not succeed; None if OK.

    Soft-fail shapes (no client → exit_code 1) must not become status=success or the
    terminal gate treats them as write recovery and clears unrecovered infra.

    Known soft refusals stay domain errors (no outcome_unknown): the sandbox said
    no, so the agent may continue. outcome_unknown is reserved for raised
    exceptions where the side-effect is truly unknown (_mutation_exception_result).
    """
    if isinstance(result, dict) and (
        result.get("status") == "success"
        or result.get("exit_code") in (0, "0")
    ):
        return None
    err = ""
    if isinstance(result, dict):
        err = str(
            result.get("stderr") or result.get("error") or result.get("message") or ""
        ).strip()
    out: Dict[str, Any] = {
        "status": "error",
        "path": path,
        "error": err or "Container write failed",
    }
    if isinstance(result, dict) and result.get("error_type"):
        out["error_type"] = result["error_type"]
    return out


def _mutation_exception_result(path: str, exc: BaseException) -> Dict[str, Any]:
    """Same typed infra labels as read/list when a write call raises."""
    out: Dict[str, Any] = {
        "status": "error",
        "path": path,
        "error": str(exc) or f"{type(exc).__name__} (no message)",
    }
    out.update(_workspace_io_infra_fields(exc))
    return out


def _external_tool_call_error_result(
    tool_id: str,
    exc: BaseException,
    *,
    was_connected: bool,
    client: Optional[ExternalMCPClient] = None,
) -> Dict[str, Any]:
    """Connect failures stay retriable; call failures after connect are outcome_unknown.

    ``was_connected`` must be captured before ``disconnect()`` — cleanup clears ``client._mode``.
    Post-connect ``ConnectionError`` subclasses (BrokenPipe, ConnectionReset) are
    still outcome_unknown: the remote side may already have applied the write.
    """
    del client  # reserved for future diagnostics
    payload: Dict[str, Any] = {
        "status": "error",
        "error": str(exc) or f"{type(exc).__name__} (no message)",
        "tool_id": tool_id,
    }
    if isinstance(exc, MCPCallTimeoutError):
        payload["reason"] = "timeout"
        payload["error_type"] = "timeout"
    if was_connected:
        # Connection torn down *after* a live session is not a safe retry signal.
        payload["outcome_unknown"] = True
    # Pre-connect (and any other non-connected) failure stays a plain retriable error.
    return payload

# Tools that run without a project container: ask_human parks on a human,
# archive_inspect/query are backend+S3 reads, and archive_fetch / attachment_fetch
# provision the sandbox themselves — they need one, but on their own terms
# (their own error wording, and inside their own long deadline rather than this
# block's unbounded one). These skip both the "container execution disabled"
# gate and the readiness/recovery block.
_CONTAINERLESS_TOOLS = frozenset({
    "ask_human",
    "archive_inspect",
    "archive_query",
    "archive_fetch",
    "attachment_list",
    "attachment_view",
    "attachment_fetch",
    "tenant_artifact_list",
    "tenant_artifact_fetch",
})

# Tools with no deadline at all: only ask_human, which legitimately waits on a
# person. Everything else stays under the global cap; do not widen this set
# without an internal timeout to replace it. The archive tools are deliberately
# NOT here — each takes a larger deadline rather than none, because an internal
# budget only covers the part of the call it wraps, and a stall outside that
# part (a wedged database, a sandbox that stopped answering) has nothing left to
# fire.
_NO_OUTER_TIMEOUT_TOOLS = frozenset({"ask_human"})

# Own their transport budget inside the handler (SDK wait_for). Outer 30s
# wait_for would race and turn TimeoutError into an empty catch-all error.
_INTERNAL_TRANSPORT_TIMEOUT_TOOLS = frozenset({"run_command"})

RUN_COMMAND_TRANSPORT_TIMEOUT_SECONDS = 120.0


def _legacy_root_project_tenant_id(project_tenant: Optional[str]) -> str:
    """Sentinel tenant for projects without ``tenant_id``.

    ``__default__`` is a legacy alias and is normalized to ``__root__`` (same as ``main``).
    """
    if not project_tenant or project_tenant == "__default__":
        return "__root__"
    return project_tenant


def _tool_tenant_normalized_for_external_scope(tool_tid: Optional[str]) -> Optional[str]:
    """Normalize tool ``tenant_id`` for access checks (legacy ``__default__`` equals ``__root__``)."""
    if tool_tid is None:
        return None
    if tool_tid == "__default__":
        return "__root__"
    return tool_tid


def _mcp_composite_tool_name(server_id: str, tool_id: str) -> str:
    """Strip storage ``_id`` to bare MCP tool name (supports ``tenant__server.tool`` ids)."""
    from tools.mcp_tool_ids import composite_tool_name_from_storage_id

    return composite_tool_name_from_storage_id(server_id, tool_id)


def _resolve_remote_mcp_endpoint(runtime: Dict[str, Any]) -> str:
    """Build full MCP URL from metadata (Discover often stores .../mcp; base URL needs path)."""
    ep = (str(runtime.get("endpoint") or "")).strip()
    if not ep:
        return ""
    path = (str(runtime.get("path") or "/mcp")).strip()
    if not path.startswith("/"):
        path = "/" + path
    parsed = urlparse(ep)
    if parsed.path and len(parsed.path) > 1:
        return ep
    return ep.rstrip("/") + path


class MCPToolExecutor:
    """
    Executes tools via MCP (Container-Use).
    
    Routes tool calls from agents to appropriate container operations.
    """
    
    def __init__(self, container_manager, storage=None, external_mcp_manager=None,
                 message_store=None):
        self.container_manager = container_manager
        self.storage = storage
        self.external_mcp_manager = external_mcp_manager
        # For ask_human's park-time journal recheck (ADR-0010); None (cli,
        # tests) degrades to parking without the recheck.
        self.message_store = message_store
        self.tracer = get_tracer()
        # Lightweight in-memory flood guard for tool calls per project/tool.
        # Keyed by f"{project_id}:{tool_id}" and stores recent call timestamps.
        self._tool_call_history: Dict[str, list[float]] = {}
        self._project_tenant_cache: Dict[str, Optional[str]] = {}
        self._local_stdio_clients: Dict[str, ExternalMCPClient] = {}
        # Project-scoped runtimes to tear down when the project workflow fully completes.
        self._project_manager_runtimes: Dict[str, Set[Tuple[str, str, str]]] = {}
        self._project_stdio_keys: Dict[str, Set[str]] = {}
        self._tool_call_window_seconds: float = 5.0
        self._tool_call_limit: int = 30
        self.handlers = {
            "create_file": self._handle_create_file,
            "list_files": self._handle_list_files,
            "read_file": self._handle_read_file,
            "edit_file": self._handle_edit_file,
            "run_command": self._handle_run_command,
            "delete_file": self._handle_delete_file,
            "find_replace_in_file": self._handle_find_replace_in_file,
            "ask_human": self._handle_ask_human,
            "archive_inspect": self._handle_archive_inspect,
            "archive_query": self._handle_archive_query,
            "archive_fetch": self._handle_archive_fetch,
            "attachment_list": self._handle_attachment_list,
            "attachment_view": self._handle_attachment_view,
            "attachment_fetch": self._handle_attachment_fetch,
            "attachment_presign_get": self._handle_attachment_presign_get,
            "attachment_presign_put": self._handle_attachment_presign_put,
            "tenant_artifact_list": self._handle_tenant_artifact_list,
            "tenant_artifact_fetch": self._handle_tenant_artifact_fetch,
        }
        # Read-side ArchiveStore; built on first archive_* call so deployments
        # that never spill pay nothing.
        self._archive_store = None
        self._file_blob_store = None

    @staticmethod
    def _runtime_scope_from_metadata(runtime: Dict[str, Any]) -> str:
        s = (str(runtime.get("runtime_scope") or "project")).strip().lower()
        return "tenant" if s == "tenant" else "project"

    @staticmethod
    def _manager_project_id_for_scope(scope: str, project_id: str) -> str:
        return MCP_TENANT_SCOPE_PROJECT_ID if scope == "tenant" else project_id

    def _track_project_manager(self, project_id: str, mpid: str, tenant_id: str, server_id: str) -> None:
        if mpid == MCP_TENANT_SCOPE_PROJECT_ID:
            return
        self._project_manager_runtimes.setdefault(project_id, set()).add((mpid, tenant_id, server_id))

    def _track_project_stdio_key(self, project_id: str, cache_key: str) -> None:
        # stdio cache key is always per-project (including runtime_scope=tenant),
        # so always track for project finalization cleanup.
        self._project_stdio_keys.setdefault(project_id, set()).add(cache_key)

    @staticmethod
    def _stdio_cache_key(project_id: str, tenant_id: str, server_id: str) -> str:
        # Include tenant in key to preserve tenant isolation during config cleanup.
        return f"{project_id}:{tenant_id}:{server_id}"

    async def finalize_project_mcp_runtimes(self, project_id: str) -> None:
        """Remove project-scoped local MCP (Docker HTTP + stdio) after the project is fully done."""
        stdio_keys = set(self._project_stdio_keys.pop(project_id, set()) or set())
        mgr_tuples = set(
            self._project_manager_runtimes.pop(project_id, set()) or set()
        )
        logger.info(
            "[EXTERNAL_MCP] [FINALIZE] project_id=%s stdio_keys=%d http_mcp_runtimes=%d",
            project_id,
            len(stdio_keys),
            len(mgr_tuples),
        )
        for cache_key in stdio_keys:
            client = self._local_stdio_clients.pop(cache_key, None)
            if client:
                cname = getattr(client, "stdio_docker_container_name", None) or "n/a"
                try:
                    await client.disconnect()
                except Exception as e:
                    logger.warning(
                        "[EXTERNAL_MCP] [FINALIZE] project_id=%s key=%s stdio_docker=%s — disconnect failed: %s",
                        project_id,
                        cache_key,
                        cname,
                        e,
                    )
            # If disconnect failed, client was popped on error path, or aclose was flaky: force rm by
            # deterministic name (cache_key = f"{project_id}:{tenant_id}:{server_id}"; legacy: project:server).
            parts = cache_key.split(":", 2)
            if len(parts) == 3:
                stdio_project_id, _, stdio_server_id = parts
            elif len(parts) == 2:
                stdio_project_id, stdio_server_id = parts
            else:
                continue
            try:
                await stdio_docker_remove_force(
                    stdio_docker_container_name(stdio_project_id, stdio_server_id),
                    phase="finalize_backup",
                    log_tenant=project_id,
                    log_server=cache_key,
                )
            except Exception as rm_e:
                logger.warning(
                    "[EXTERNAL_MCP] [FINALIZE] project_id=%s stdio backup rm: %s",
                    project_id,
                    rm_e,
                )
        if not self.external_mcp_manager:
            if mgr_tuples:
                logger.warning(
                    "[EXTERNAL_MCP] [FINALIZE] project_id=%s — no external_mcp_manager, "
                    "cannot stop %d HTTP runtimes",
                    project_id,
                    len(mgr_tuples),
                )
            logger.info(
                "[EXTERNAL_MCP] [FINALIZE] project_id=%s — end (stdio closed, HTTP manager absent)",
                project_id,
            )
            return
        for mpid, tenant_id, server_id in mgr_tuples:
            try:
                await self.external_mcp_manager.stop_server(
                    project_id=mpid,
                    tenant_id=tenant_id,
                    server_id=server_id,
                    stop_reason="project_finalize",
                )
            except Exception as e:
                logger.warning(
                    "[EXTERNAL_MCP] [FINALIZE] project_id=%s stop_server failed server=%s: %s",
                    project_id,
                    server_id,
                    e,
                )
        logger.info(
            "[EXTERNAL_MCP] [FINALIZE] project_id=%s — done (stdio + HTTP)",
            project_id,
        )

    def _remove_stdio_cache_key_everywhere(self, cache_key: str) -> None:
        for s in self._project_stdio_keys.values():
            s.discard(cache_key)

    async def on_mcp_server_config_removed(self, *, tenant_id: str, mcp_server_id: str) -> None:
        """Stop all local MCP runtimes for this server/tenant (config deleted from DB)."""
        t = tenant_id or "__root__"
        to_disconnect: list[str] = []
        for cache_key in list(self._local_stdio_clients.keys()):
            parts = cache_key.split(":", 2)
            if len(parts) == 2:
                # Legacy key format: project_id:server_id (no tenant segment).
                # Only allow matching for root tenant to avoid cross-tenant disconnects.
                if t != "__root__":
                    continue
                if parts[1] != mcp_server_id:
                    continue
            elif len(parts) == 3:
                if (
                    parts[1] != t
                    or parts[2] != mcp_server_id
                ):
                    continue
            else:
                continue
            to_disconnect.append(cache_key)
        logger.info(
            "[EXTERNAL_MCP] [CONFIG_REMOVED] tenant_id=%s mcp_server_id=%s — "
            "stdio keys to close=%d",
            t,
            mcp_server_id,
            len(to_disconnect),
        )
        for cache_key in to_disconnect:
            client = self._local_stdio_clients.pop(cache_key, None)
            self._remove_stdio_cache_key_everywhere(cache_key)
            if client:
                try:
                    await client.disconnect()
                except Exception as e:
                    logger.warning(
                        "[EXTERNAL_MCP] [CONFIG_REMOVED] stdio disconnect key=%s: %s",
                        cache_key,
                        e,
                    )
        if self.external_mcp_manager and hasattr(
            self.external_mcp_manager, "stop_all_for_tenant_mcp_server"
        ):
            try:
                await self.external_mcp_manager.stop_all_for_tenant_mcp_server(
                    tenant_id=t, mcp_server_id=mcp_server_id
                )
            except Exception as e:
                logger.warning(
                    "[EXTERNAL_MCP] [CONFIG_REMOVED] manager stop_all failed: %s", e
                )
        else:
            logger.info(
                "[EXTERNAL_MCP] [CONFIG_REMOVED] tenant_id=%s mcp_server_id=%s — "
                "no external_mcp_manager (HTTP skip)",
                t,
                mcp_server_id,
            )
        logger.info(
            "[EXTERNAL_MCP] [CONFIG_REMOVED] tenant_id=%s mcp_server_id=%s — completed",
            t,
            mcp_server_id,
        )

    @staticmethod
    def _truncate_str(v: Any, limit: int = -1) -> str:
        """Stringify *v* and truncate to *limit* chars (-1 = unlimited)."""
        try:
            s = v if isinstance(v, str) else ("\n".join(v) if isinstance(v, list) else str(v))
        except Exception:
            s = str(v)
        if limit >= 0 and len(s) > limit:
            return s[:limit] + "... (truncated)"
        return s

    def _get_tool_category(self, tool_id: str) -> str:
        """Get tool category for telemetry."""
        if "file" in tool_id.lower():
            return "file_operations"
        elif "run" in tool_id.lower() or "command" in tool_id.lower():
            return "execution"
        elif "install" in tool_id.lower():
            return "package_management"
        elif "web" in tool_id.lower():
            return "web_operations"
        elif "api" in tool_id.lower():
            return "api_operations"
        else:
            return tool_id.lower()

    async def execute_tool(
        self,
        tool_id: str,
        project_id: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a tool: routes to external MCP or local container handler."""
        # Builtin ids never route to external MCP: the create-time guard
        # (AppFactory-268) can be bypassed by direct DB writes — the incident's own
        # vector — and a colliding doc would otherwise capture builtin calls for
        # every tenant now that __system__ docs resolve shared.
        is_builtin_id = tool_id in known_builtin_tool_ids()
        # External MCP tools (source=mcp_server) may use remote HTTP and do not
        # need a local container — check them before the container_manager guard.
        external_cfg = (
            None if is_builtin_id
            else await self._get_external_tool_config(tool_id, project_id)
        )
        if external_cfg is not None:
            # Resolution ignores `enabled` (it must find disabled docs for admin
            # flows), so dispatch is where a disabled tool gets refused loudly.
            # Falsy, not `is False`: the schema layer hides every falsy value, and
            # the two ADR-0013 predicates must not disagree (absent still allows).
            if not external_cfg.get("enabled", True):
                return {
                    "status": "error",
                    "error": f"MCP tool '{tool_id}' is disabled",
                    "tool_id": tool_id,
                }
            started = time.monotonic()
            try:
                result = await self._execute_external_tool(
                    tool_id=tool_id,
                    project_id=project_id,
                    params=params,
                    tool_config=external_cfg,
                )
            except Exception as error:
                log_raised_call(project_id, tool_id, started, error)
                raise
            return observed_call(result, project_id, tool_id, started)

        # ask_human does not use container
        unscoped_mcp = (
            None if is_builtin_id else await self._lookup_mcp_tool_document(tool_id)
        )
        if unscoped_mcp is not None:
            project_tenant = (
                await self._get_project_tenant_id(project_id) if project_id else None
            )
            from tools.mcp_tool_ids import _tenant_matches

            scope_tenant = _legacy_root_project_tenant_id(project_tenant)
            if not _tenant_matches(unscoped_mcp, scope_tenant):
                logger.warning(
                    "[EXTERNAL_MCP] tenant deny tool=%s project=%s scope=%s tool_tenant=%s",
                    tool_id,
                    project_id,
                    scope_tenant,
                    unscoped_mcp.get("tenant_id"),
                )
                return {
                    "status": "error",
                    "error": "External MCP tool is not accessible for this tenant",
                    "tool_id": tool_id,
                }

        if not self.container_manager.enabled:
            if tool_id not in _CONTAINERLESS_TOOLS:
                return {
                    "status": "error",
                    "error": "Container execution disabled",
                    "error_type": "unavailable",
                    "tool_id": tool_id,
                }

        handler = self.handlers.get(tool_id)
        if not handler:
            logger.error("Unknown tool: %s", tool_id)
            return {
                "status": "error",
                "error": f"Unknown tool: {tool_id}",
                "tool_id": tool_id,
            }

        # Flood guard: short-circuit if this (project, tool) has exceeded the
        # allowed rate in the recent time window. This prevents infinite loops
        # (e.g., repeated list_files/run_command on an error path) from
        # hammering container-use.
        try:
            key = f"{project_id}:{tool_id}"
            now = time.time()
            window = self._tool_call_window_seconds
            limit = self._tool_call_limit
            hist = self._tool_call_history.get(key) or []
            # Drop entries older than the window
            hist = [t for t in hist if now - t <= window]
            hist.append(now)
            self._tool_call_history[key] = hist
            if len(hist) > limit:
                logger.warning(
                    "MCP tool flood guard triggered: project=%s tool=%s count=%s window=%ss",
                    project_id,
                    tool_id,
                    len(hist),
                    window,
                )
                base_err = {
                    "status": "error",
                    "error": "Tool flood guard: too many calls in a short window",
                    "error_type": "unavailable",
                    "tool_id": tool_id,
                }
                # For list_files, also return files: [] so BFS harvesters stop cleanly
                if tool_id == "list_files":
                    base_err["files"] = []
                return base_err
        except Exception:
            # Flood guard must never break normal execution; if it fails, ignore.
            pass
        
        # Ensure container is ready before invoking handler - attempt recovery if not.
        # Containerless tools skip this entirely: for ask_human (ADR-0010) a
        # question needs no container, and provisioning would widen the gap
        # between the answerable journal record and the parked waiter; the
        # archive / attachment tools read from backend/S3, and fetch tools
        # provision the sandbox themselves.
        if tool_id not in _CONTAINERLESS_TOOLS:
            unavailable = self.container_manager.session_unavailable_reason(project_id)
            if unavailable:
                return self._structured_session_unavailable(
                    project_id,
                    params,
                    tool_id,
                    reason=(
                        f"{tool_id}: MCP session unavailable after timeout recovery "
                        f"({unavailable}); refusing further container calls"
                    ),
                )
            try:
                status = await self.container_manager.get_container_status(project_id)
                # get_container_status can return the preserved workspace's
                # environment_id with no live session — branch on active, not on
                # environment_id.
                if not status.get("active"):
                    # Attempt to recover/bootstrap container before giving up
                    logger.info("[RECOVERY] Container not available for %s, attempting recovery...", project_id)
                    try:
                        # Create new container
                        container = await self.container_manager.get_or_create_container(project_id)
                        remade = self.container_manager.session_unavailable_reason(project_id)
                        if remade or container.get("status") == "unavailable":
                            return self._structured_session_unavailable(
                                project_id,
                                params,
                                tool_id,
                                reason=(
                                    f"{tool_id}: MCP session unavailable after timeout recovery "
                                    f"({remade or 'unavailable'}); refusing further container calls"
                                ),
                            )
                        if container.get("status") == "simulated" or not container.get("client"):
                            return {
                                "status": "error",
                                "error": "Container environment not available (recovery failed)",
                                "error_type": "unavailable",
                                "tool_id": tool_id,
                            }
                        # Restore files from projects.context.artifacts to the new container
                        if self.storage:
                            restored = await self.container_manager.restore_container_from_context(
                                project_id, self.storage
                            )
                            logger.info("[RECOVERY] Container recreated, restored %s files", restored)
                    except Exception as recovery_err:
                        logger.warning("[RECOVERY] Recovery attempt failed: %s", recovery_err)
                        return {
                            "status": "error",
                            "error": f"Container environment not available (recovery failed: {recovery_err})",
                            "error_type": "unavailable",
                            "tool_id": tool_id,
                        }
            except Exception:
                # If status check fails, attempt anyway but guard with timeout
                pass
        
        try:
            attributes = {
                "tool.id": tool_id,
                "project.id": project_id,
                "params": params,
                "AppFactory.tool.execution_type": "mcp_container",
                "AppFactory.tool.container_use": True,
                "AppFactory.tool.category": self._get_tool_category(tool_id),
                "AppFactory.tool.requires_container": True,
            }
            # Trace execution - include tool name in span for easy filtering
            with self.tracer.start_span(
                f"tool.execute.{tool_id}",
                attributes=attributes,
            ) as span:
                # Log request preview to Jaeger to aid debugging
                try:
                    import os as _os
                    req_limit = int(_os.getenv("OTEL_TOOL_REQUEST_PREVIEW_LIMIT", "-1"))
                    req_preview: Dict[str, Any] = {
                        "tool": tool_id,
                        "project_id": project_id,
                        "params": self._truncate_str(params, req_limit),
                    }
                    self.tracer.add_event(span, "tool.request", req_preview)
                except Exception:
                    pass
                
                # ask_human waits on a person, so it escapes the global cap
                # entirely (see _NO_OUTER_TIMEOUT_TOOLS)
                if tool_id in _NO_OUTER_TIMEOUT_TOOLS:
                    result = await handler(project_id, params)
                elif tool_id in _INTERNAL_TRANSPORT_TIMEOUT_TOOLS:
                    from sandbox.command_lifecycle import CommandTransportTimeout

                    try:
                        result = await handler(project_id, params)
                    except CommandTransportTimeout as exc:
                        result = await self._on_run_command_transport_timeout(
                            project_id, params, exc
                        )
                elif tool_id in (
                    "archive_fetch",
                    "archive_query",
                    "attachment_fetch",
                    "tenant_artifact_fetch",
                ):
                    # These need far longer than 30s, and each owns an internal
                    # budget that covers only part of the call — curl cannot fire
                    # if the sandbox stopped answering, and the scan budget never
                    # sees the ownership lookup that precedes it. These deadlines
                    # sit above the internal ones and cover the whole call.
                    from tools import archive_tools
                    from tools import attachment_tools
                    from tools import tenant_artifact_tools

                    if tool_id == "attachment_fetch":
                        ceiling = attachment_tools.fetch_ceiling_seconds()
                    elif tool_id == "archive_fetch":
                        ceiling = archive_tools.fetch_ceiling_seconds()
                    elif tool_id == "tenant_artifact_fetch":
                        ceiling = tenant_artifact_tools.fetch_ceiling_seconds()
                    else:
                        ceiling = archive_tools.query_ceiling_seconds()
                    try:
                        result = await asyncio.wait_for(
                            handler(project_id, params), timeout=ceiling
                        )
                    except asyncio.TimeoutError:
                        # The catch-all below would str() this into "" — an empty
                        # error the agent cannot act on. Only this deadline can
                        # fire on a stalled backend/sandbox, so it has to explain
                        # itself and name a tool that is still reachable.
                        logger.error(
                            "Tool %s exceeded its %.0fs ceiling for project %s",
                            tool_id, ceiling, project_id,
                        )
                        if tool_id.startswith("attachment_"):
                            hint = "attachment_list"
                        elif tool_id.startswith("tenant_artifact_"):
                            hint = "tenant_artifact_list"
                        else:
                            hint = "archive_inspect"
                        return {
                            "status": "error",
                            "error_type": "timeout",
                            "error": (
                                f"{tool_id}: timed out after {ceiling:.0f}s waiting on "
                                "storage or the sandbox; try "
                                f"{hint} for metadata"
                            ),
                            "tool_id": tool_id,
                        }
                else:
                    result = await asyncio.wait_for(handler(project_id, params), timeout=30.0)

                logger.debug("Tool %s executed successfully", tool_id)
                if span:
                    span.set_attribute("tool.status", result.get("status"))
                    exit_code = result.get("exit_code")
                    if exit_code is not None:
                        span.set_attribute("tool.exit_code", exit_code)
                    # Add a richer preview of the tool reply as an event for Jaeger
                    try:
                        # Compute raw content length if present before truncation for observability
                        raw_content = result.get("content")
                        try:
                            if isinstance(raw_content, str):
                                span.set_attribute("result.content_len", len(raw_content))
                            elif isinstance(raw_content, list):
                                span.set_attribute("result.content_len", len("\n".join([str(x) for x in raw_content])))
                        except Exception:
                            pass
                        preview_keys = [
                            "status",
                            "path",
                            "size",
                            "files",
                            "exit_code",
                            "stdout",
                            "stderr",
                            "error",
                            "content",
                        ]
                        preview: Dict[str, Any] = {
                            k: result.get(k) for k in preview_keys if k in result
                        }
                        # stdout/stderr and content may be large; limit configurable via env
                        import os as _os
                        res_limit = int(_os.getenv("OTEL_TOOL_RESULT_PREVIEW_LIMIT", "1024"))
                        for key in ["stdout", "stderr", "content"]:
                            if key in preview and isinstance(preview[key], (str, list)):
                                preview[key] = self._truncate_str(preview[key], res_limit)

                        # Summarize potentially large file lists
                        if "files" in preview and isinstance(preview["files"], list):
                            try:
                                names = []
                                for f in preview["files"]:
                                    if isinstance(f, dict):
                                        p = f.get("path") or f.get("name") or str(f)
                                    else:
                                        p = str(f)
                                    names.append(p)
                                preview["files"] = names[:50]
                                if len(names) > 50:
                                    preview["files"].append(f"... (+{len(names) - 50} more)")
                            except Exception:
                                preview["files"] = self._truncate_str(preview["files"], res_limit)

                        # Flatten into key/value pairs so Jaeger renders them in the UI
                        flat_attrs: Dict[str, Any] = {}
                        for k, v in preview.items():
                            key = f"result.{k}"
                            flat_attrs[key] = v
                        self.tracer.add_event(span, "tool.result", flat_attrs)
                    except Exception:
                        pass
                self.tracer.set_success(span)
                return result
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_id, e)
            # str(TimeoutError()) is often empty; without error_type a consumer
            # cannot tell timeout apart from a domain tool failure.
            payload: Dict[str, Any] = {
                "status": "error",
                "error": str(e) or f"{type(e).__name__} (no message)",
                "tool_id": tool_id,
            }
            if _is_tool_timeout_exception(e):
                payload["error_type"] = "timeout"
            elif isinstance(e, ConnectionError):
                # Write handlers type this locally; keep the same label if a
                # mutation raises past them so the terminal gate still vetoes.
                payload["error_type"] = "connection_error"
            return payload

    async def _handle_create_file(
        self, 
        project_id: str, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create file in container."""
        path = params.get("path")
        content = params.get("content", "")
        
        if not path:
            raise ValueError("Missing required param: path")

        # create_file is create-only: refuse to overwrite existing files.
        # This prevents agents from clobbering work from previous tasks without reading.
        try:
            norm = str(path).strip().lstrip("./").rstrip("/")
            parent = "."
            name = norm
            if "/" in norm:
                parent, name = norm.rsplit("/", 1)
            lf = await self.container_manager.list_files_in_container(project_id, parent or ".")
            for entry in lf or []:
                if not isinstance(entry, str):
                    continue
                base = entry.rstrip("/").rsplit("/", 1)[-1]
                if base == name:
                    return {
                        "status": "error",
                        "error": "File already exists (create_file does not overwrite)",
                        "path": path,
                    }
        except TimeoutError:
            raise
        except ConnectionError:
            raise
        except Exception:
            # Domain listing miss: do not block creation.
            pass
        
        try:
            write = await self.container_manager.write_file_in_container(
                project_id,
                path,
                content,
            )
        except TimeoutError:
            raise
        except Exception as e:
            return _mutation_exception_result(path, e)
        refused = self._refuse_if_manager_session_unavailable(
            project_id, params, "create_file", write
        )
        if refused:
            return refused
        fail = _container_mutation_error(path, write)
        if fail:
            return fail

        return {
            "status": "success",
            "path": path,
            "size": len(content),
        }
    
    async def _handle_list_files(
        self,
        project_id: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """List files in container workspace."""
        from sandbox.command_lifecycle import SessionUnavailableError

        path = params.get("path", ".")  # Default to current directory
        
        try:
            files = await self.container_manager.list_files_in_container(
                project_id,
                path,
                storage=self.storage
            )
            
            return {
                "status": "success",
                "path": path,
                "files": files,
            }
        except SessionUnavailableError as e:
            return self._structured_session_unavailable(
                project_id,
                params,
                "list_files",
                reason=(
                    f"list_files: MCP session unavailable after timeout recovery "
                    f"({e.reason}); refusing further container calls"
                ),
            )
        except TimeoutError:
            raise
        except Exception as e:
            from sandbox.mcp_client_sdk import SandboxListingError

            # Sandbox said the path is not a directory — domain, not infra.
            if isinstance(e, SandboxListingError):
                return {
                    "status": "error",
                    "error": str(e) or f"Could not list {path}",
                    "path": path,
                    "files": [],
                }
            # `files` stays so directory walkers still terminate, but claiming
            # success made "this path is unreachable" arrive as "this directory
            # is empty" — which a search then reports as "no matches found".
            out: Dict[str, Any] = {
                "status": "error",
                "error": f"Could not list {path}: {e}",
                "path": path,
                "files": [],
            }
            out.update(_workspace_io_infra_fields(e))
            return out
    
    async def _handle_read_file(
        self,
        project_id: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Read file from container."""
        from sandbox.command_lifecycle import SessionUnavailableError

        path = params.get("path")
        
        if not path:
            raise ValueError("Missing required param: path")
        
        try:
            content = await self.container_manager.read_file_from_container(
                project_id,
                path,
                storage=self.storage
            )
        except SessionUnavailableError as e:
            return self._structured_session_unavailable(
                project_id,
                params,
                "read_file",
                reason=(
                    f"read_file: MCP session unavailable after timeout recovery "
                    f"({e.reason}); refusing further container calls"
                ),
            )
        except TimeoutError:
            raise
        except Exception as e:
            from sandbox.mcp_client_sdk import SandboxReadError

            # Sandbox refused the path — domain error, not an empty successful read.
            if isinstance(e, SandboxReadError):
                return {
                    "status": "error",
                    "error": str(e) or f"Could not read {path}",
                    "path": path,
                }
            out: Dict[str, Any] = {
                "status": "error",
                "error": str(e) or f"{type(e).__name__} (no message)",
                "path": path,
            }
            out.update(_workspace_io_infra_fields(e))
            return out

        return {
            "status": "success",
            "path": path,
            "content": content,
            "size": len(content),
        }
    
    async def _handle_edit_file(
        self,
        project_id: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Edit existing file in container."""
        path = params.get("path")
        content = params.get("content", "")
        
        if not path:
            raise ValueError("Missing required param: path")
        
        try:
            write = await self.container_manager.write_file_in_container(
                project_id,
                path,
                content,
            )
        except TimeoutError:
            raise
        except Exception as e:
            return _mutation_exception_result(path, e)
        refused = self._refuse_if_manager_session_unavailable(
            project_id, params, "edit_file", write
        )
        if refused:
            return refused
        fail = _container_mutation_error(path, write)
        if fail:
            return fail

        return {
            "status": "success",
            "path": path,
            "action": "updated",
        }

    async def _handle_delete_file(
        self,
        project_id: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Delete a file in the container."""
        path = params.get("path")
        if not path:
            raise ValueError("Missing required param: path")
        try:
            deleted = await self.container_manager.delete_file_in_container(
                project_id, path
            )
        except TimeoutError:
            raise
        except Exception as e:
            return _mutation_exception_result(path, e)
        refused = self._refuse_if_manager_session_unavailable(
            project_id, params, "delete_file", deleted
        )
        if refused:
            return refused
        fail = _container_mutation_error(path, deleted)
        if fail:
            return fail
        return {"status": "success", "path": path}

    async def _handle_find_replace_in_file(
        self,
        project_id: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Find and replace text in a file in the container."""
        path = params.get("path")
        search_text = params.get("search_text")
        replace_text = params.get("replace_text")
        which_match = params.get("which_match")
        if not path or search_text is None or replace_text is None:
            raise ValueError("Missing required params: path, search_text, replace_text")
        try:
            replaced = await self.container_manager.find_replace_in_file(
                project_id,
                path,
                search_text,
                replace_text,
                which_match,
            )
        except TimeoutError:
            raise
        except Exception as e:
            return _mutation_exception_result(path, e)
        refused = self._refuse_if_manager_session_unavailable(
            project_id, params, "find_replace_in_file", replaced
        )
        if refused:
            return refused
        fail = _container_mutation_error(path, replaced)
        if fail:
            return fail
        return {"status": "success", "path": path}
    
    def _structured_session_unavailable(
        self,
        project_id: str,
        params: Dict[str, Any],
        tool_id: str,
        *,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        from sandbox.command_lifecycle import structured_session_unavailable_error

        return structured_session_unavailable_error(
            timeout_seconds=0.0,
            project_id=project_id,
            run_id=params.get("_run_id"),
            tool_call_id=params.get("_tool_call_id"),
            environment_id=None,
            tool_id=tool_id,
            detail=reason or "unavailable",
        )

    def _refuse_if_manager_session_unavailable(
        self,
        project_id: str,
        params: Dict[str, Any],
        tool_id: str,
        result: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not result:
            return None
        from schemas.infra_error import InfraErrorType

        code = result.get("code") or result.get("error_type")
        if (
            result.get("session_state") != "unavailable"
            and code != InfraErrorType.SESSION_UNAVAILABLE.value
        ):
            return None
        return self._structured_session_unavailable(
            project_id,
            params,
            tool_id,
            reason=result.get("stderr") or result.get("error") or None,
        )

    async def _on_run_command_transport_timeout(
        self,
        project_id: str,
        params: Dict[str, Any],
        exc: "CommandTransportTimeout",
    ) -> Dict[str, Any]:
        """Structured timeout + session recovery; never auto-retry the command."""
        from sandbox.command_lifecycle import (
            READY,
            structured_session_unavailable_error,
            structured_transport_timeout_error,
        )
        from sandbox.run_command_recovery import recover_session_after_timeout

        env_id = getattr(exc, "environment_id", None)
        recovery = await recover_session_after_timeout(
            self.container_manager,
            project_id,
            environment_id=env_id,
        )
        timeout_seconds = float(getattr(exc, "timeout_seconds", 0) or 0)
        if not recovery.get("ok"):
            detail = recovery.get("error") or "recovery failed"
            return structured_session_unavailable_error(
                timeout_seconds=timeout_seconds,
                project_id=project_id,
                run_id=params.get("_run_id"),
                tool_call_id=params.get("_tool_call_id"),
                environment_id=recovery.get("environment_id") or env_id,
                tool_id="run_command",
                detail=(
                    f"transport wait timed out after {timeout_seconds:g}s; "
                    f"{detail}; workspace kept for review"
                ),
            )
        return structured_transport_timeout_error(
            timeout_seconds=timeout_seconds,
            project_id=project_id,
            run_id=params.get("_run_id"),
            tool_call_id=params.get("_tool_call_id"),
            environment_id=recovery.get("environment_id") or env_id,
            tool_id="run_command",
            session_state=recovery.get("session_state") or READY,
            recovery_ok=True,
            recovery_mode=str(recovery.get("recovery_mode") or "replace"),
        )

    async def _handle_run_command(
        self,
        project_id: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute shell command in container (default cwd=/workdir)."""
        command = params.get("command")
        cwd = params.get("cwd", "/workdir")
        
        if not command:
            raise ValueError("Missing required param: command")
        
        from sandbox.command_lifecycle import CommandTransportTimeout

        try:
            result = await self.container_manager.execute_in_container(
                project_id,
                command,
                cwd,
                timeout_seconds=RUN_COMMAND_TRANSPORT_TIMEOUT_SECONDS,
            )
        except CommandTransportTimeout:
            raise
        except TimeoutError:
            raise
        except Exception as e:
            out = {
                "status": "error",
                "command": command,
                "stdout": "",
                "stderr": str(e) or f"{type(e).__name__} (no message)",
                "exit_code": 1,
                "error": str(e) or f"{type(e).__name__} (no message)",
            }
            out.update(_workspace_io_infra_fields(e))
            return out
        refused = self._refuse_if_manager_session_unavailable(
            project_id, params, "run_command", result
        )
        if refused:
            return refused

        # Strip container-use commit/push footer from stdout/stderr if present to reduce noise in agent observations
        try:
            stdout_raw = result.get("stdout", "") or ""
            stderr_raw = result.get("stderr", "") or ""
            # Remove either single-line or split two-line footer variants by deleting lines that contain these phrases
            footer_line_re = r"(?mi)^\s*(?:Any changes to the container workdir[^\n]*|and pushed to container-use/ remote[^\n]*)\s*$"
            cleaned_stdout = re.sub(footer_line_re, "", stdout_raw).rstrip()
            cleaned_stderr = re.sub(footer_line_re, "", stderr_raw).rstrip()
        except Exception:
            cleaned_stdout = result.get("stdout", "") or ""
            cleaned_stderr = result.get("stderr", "") or ""

        exit_code = result.get("exit_code", -1)
        out: Dict[str, Any] = {
            "status": "success" if exit_code == 0 else "failed",
            "command": command,
            "stdout": cleaned_stdout,
            "stderr": cleaned_stderr,
            "exit_code": exit_code,
        }
        # host_cli / GNU timeout: 124 means wall-clock kill, not domain fail.
        if exit_code == 124:
            out["error_type"] = "timeout"
            out["error"] = cleaned_stderr or "command timed out"
        elif exit_code not in (0, "0"):
            # Prefer labels from the sandbox layer; do not scan stderr text.
            if result.get("error_type"):
                out["error_type"] = result["error_type"]
                out["error"] = cleaned_stderr or str(result.get("error") or "") or "command failed"
            elif result.get("outcome_unknown"):
                out["outcome_unknown"] = True
        elif result.get("outcome_unknown"):
            out["outcome_unknown"] = result["outcome_unknown"]
        return out

    async def _lookup_mcp_tool_document(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Load MCP config by wire/public/storage id without tenant filter."""
        if not self.storage:
            return None
        from tools.mcp_tool_ids import resolve_mcp_tool_doc

        mcp_get = getattr(self.storage, "get_mcp_tool_configuration", None)
        if callable(mcp_get):
            doc = await mcp_get(tool_id)
            if isinstance(doc, dict) and doc.get("source") == "mcp_server":
                return doc

        list_mcp = getattr(self.storage, "get_mcp_tool_configurations", None)
        if callable(list_mcp):
            docs = await list_mcp(enabled_only=False, tenant_id=None)
            tenants = {
                str(d.get("tenant_id") or "__root__")
                for d in (docs or [])
                if isinstance(d, dict) and d.get("source") == "mcp_server"
            }
            for tid in sorted(tenants):
                doc = await resolve_mcp_tool_doc(self.storage, tid, tool_id)
                if isinstance(doc, dict) and doc.get("source") == "mcp_server":
                    return doc

        doc = await self.storage.get_tool_configuration(tool_id)
        if isinstance(doc, dict) and doc.get("source") == "mcp_server":
            return doc
        return None

    async def _get_external_tool_config(
        self, tool_id: str, project_id: str | None = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.storage:
            return None
        try:
            project_tenant = None
            if project_id:
                project_tenant = await self._get_project_tenant_id(project_id)
            from tools.mcp_tool_ids import _tenant_matches, resolve_mcp_tool_doc

            scope_tenant = _legacy_root_project_tenant_id(project_tenant)
            cfg = await resolve_mcp_tool_doc(self.storage, scope_tenant, tool_id)
            if isinstance(cfg, dict) and cfg.get("source") == "mcp_server":
                return cfg

            mcp_get = getattr(self.storage, "get_mcp_tool_configuration", None)
            if callable(mcp_get):
                cfg = await mcp_get(tool_id)
                if (
                    isinstance(cfg, dict)
                    and cfg.get("source") == "mcp_server"
                    and _tenant_matches(cfg, scope_tenant)
                ):
                    return cfg
            return None
        except Exception as e:
            logger.warning("[EXTERNAL_MCP] tool=%s config lookup failed: %s", tool_id, e)
            return None

    async def _get_project_tenant_id(self, project_id: str) -> Optional[str]:
        if project_id in self._project_tenant_cache:
            return self._project_tenant_cache[project_id]
        if not self.storage:
            return None
        try:
            if hasattr(self.storage, "load_project"):
                doc = await self.storage.load_project(project_id)
            elif hasattr(self.storage, "get_project"):
                doc = await self.storage.get_project(project_id)
            else:
                doc = None
            tenant_id = doc.get("tenant_id") if isinstance(doc, dict) else None
            self._project_tenant_cache[project_id] = tenant_id
            return tenant_id
        except Exception as e:
            logger.warning("[EXTERNAL_MCP] project=%s tenant lookup failed: %s", project_id, e)
            return None

    async def _execute_external_tool(
        self,
        tool_id: str,
        project_id: str,
        params: Dict[str, Any],
        tool_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        project_tenant = await self._get_project_tenant_id(project_id)
        tool_tenant = tool_config.get("tenant_id")
        effective_project_tenant = _legacy_root_project_tenant_id(project_tenant)
        tool_tn = _tool_tenant_normalized_for_external_scope(tool_tenant)
        # __system__ tools are accessible to all tenants (same as agent_configurations).
        tool_is_global = tool_tn in ("__system__", "__root__", None)
        if not tool_is_global and effective_project_tenant != tool_tn:
            logger.warning(
                f"[EXTERNAL_MCP] tenant mismatch project={project_id} "
                f"project_tenant={effective_project_tenant} tool={tool_id} tool_tenant={tool_tenant}"
            )
            return {
                "status": "error",
                "error": "External MCP tool is not accessible for this tenant",
                "tool_id": tool_id,
            }

        server_id = tool_config.get("mcp_server")
        if not server_id:
            return {
                "status": "error",
                "error": "Missing mcp_server on tool configuration",
                "tool_id": tool_id,
            }

        metadata = tool_config.get("metadata") if isinstance(tool_config.get("metadata"), dict) else {}
        runtime = metadata.get("external_mcp") if isinstance(metadata.get("external_mcp"), dict) else {}
        image = (str(runtime.get("image") or "")).strip()
        remote_url = _resolve_remote_mcp_endpoint(runtime)
        from tools.mcp_tool_ids import mcp_rpc_tool_name

        mcp_tool_name = mcp_rpc_tool_name(tool_config)
        mode = (str(runtime.get("mode") or "auto")).strip().lower()
        command = (str(runtime.get("command") or "")).strip()
        tenant_for_runtime = _legacy_root_project_tenant_id(project_tenant)
        is_remote_http = bool(remote_url and urlparse(remote_url).scheme in ("http", "https") and not image)

        # Remote HTTP/streamable endpoint: no local Docker runtime.
        if is_remote_http:
            return await self._execute_external_tool_http_remote(
                tool_id=tool_id,
                project_id=project_id,
                params=params,
                runtime=runtime,
                remote_url=remote_url,
                mcp_tool_name=mcp_tool_name,
                project_tenant=project_tenant,
                server_id=server_id,
            )

        if not image and not command:
            image = await self._get_default_external_mcp_image()
            logger.info(
                "[EXTERNAL_MCP] project=%s tool=%s local runtime default image=%s",
                project_id,
                tool_id,
                image,
            )

        rscope = self._runtime_scope_from_metadata(runtime)
        manager_project_id = self._manager_project_id_for_scope(rscope, project_id)
        idle_timeout_param = runtime.get("idle_timeout_seconds", None)
        if idle_timeout_param is not None:
            try:
                idle_timeout_param = float(idle_timeout_param)
            except (TypeError, ValueError):
                idle_timeout_param = None

        # Local stdio runtime: always isolated per project.
        # Even if runtime_scope is tenant, we clone stdio client for each project
        # to guarantee "1 stdio = 1 client = 1 project".
        if command and mode in ("stdio", "auto"):
            timeout = float(runtime.get("timeout_seconds") or 60.0)
            cache_key = self._stdio_cache_key(project_id, tenant_for_runtime, server_id)
            client = self._local_stdio_clients.get(cache_key)
            if client is None:
                client = ExternalMCPClient(
                    ExternalMCPConfig(
                        server_id=server_id,
                        endpoint="",
                        tenant_id=tenant_for_runtime,
                        mode="stdio",
                        command=command,
                        args=runtime.get("command_args") if isinstance(runtime.get("command_args"), list) else None,
                        env=runtime.get("command_env") if isinstance(runtime.get("command_env"), dict) else None,
                        timeout_seconds=timeout,
                    )
                )
                self._local_stdio_clients[cache_key] = client
            self._track_project_stdio_key(project_id, cache_key)
            try:
                result = await client.call_tool(mcp_tool_name, params)
                return result if isinstance(result, dict) else {"status": "success", "data": result}
            except Exception as e:
                # Important: `disconnect()` clears `_mode` in real `ExternalMCPClient`,
                # so capture "was connected" before cleanup.
                # CancelledError is BaseException — left for runners (they know project cancel).
                was_connected = client._mode is not None
                try:
                    await client.disconnect()
                except Exception:
                    pass
                self._local_stdio_clients.pop(cache_key, None)
                return _external_tool_call_error_result(
                    tool_id, e, was_connected=was_connected
                )

        if image and mode in ("stdio", "auto"):
            docker_env_vars = runtime.get("docker_env_vars") if isinstance(runtime.get("docker_env_vars"), dict) else None
            docker_cmd_args = runtime.get("docker_cmd_args") if isinstance(runtime.get("docker_cmd_args"), list) else None
            timeout = float(runtime.get("timeout_seconds") or 60.0)
            cache_key = self._stdio_cache_key(project_id, tenant_for_runtime, server_id)
            stdio_name = stdio_docker_container_name(project_id, server_id)
            if rscope == "tenant":
                logger.info(
                    "[EXTERNAL_MCP] [STDIO_CLONE] project_id=%s tenant_id=%s server_id=%s "
                    "runtime_scope=tenant -> isolated project clone key=%s",
                    project_id,
                    tenant_for_runtime,
                    server_id,
                    cache_key,
                )
            client = self._local_stdio_clients.get(cache_key)
            if client is None:
                client = ExternalMCPClient(
                    ExternalMCPConfig(
                        server_id=server_id,
                        endpoint="",
                        tenant_id=tenant_for_runtime,
                        mode="stdio",
                        image=image,
                        docker_env_vars=docker_env_vars,
                        docker_cmd_args=docker_cmd_args,
                        timeout_seconds=timeout,
                        stdio_docker_name=stdio_name,
                    )
                )
                self._local_stdio_clients[cache_key] = client
                logger.info(
                    "[EXTERNAL_MCP] [STDIO] project=%s tenant=%s server=%s cache=created key=%s image=%s "
                    "scope=%s stdio_docker_name=%s",
                    project_id,
                    tenant_for_runtime,
                    server_id,
                    cache_key,
                    image,
                    rscope,
                    stdio_name,
                )
            else:
                logger.info(
                    "[EXTERNAL_MCP] [STDIO] project=%s tenant=%s server=%s cache=reused key=%s scope=%s",
                    project_id,
                    tenant_for_runtime,
                    server_id,
                    cache_key,
                    rscope,
                )
            self._track_project_stdio_key(project_id, cache_key)
            try:
                result = await client.call_tool(mcp_tool_name, params)
                return result if isinstance(result, dict) else {"status": "success", "data": result}
            except Exception as e:
                err_type = type(e).__name__
                err_msg = str(e) or f"{err_type} (no message)"
                logger.warning(
                    "[EXTERNAL_MCP] project=%s tool=%s stdio_image=%s — call failed: %s: %s",
                    project_id, tool_id, image, err_type, err_msg,
                )
                # `disconnect()` clears `_mode` in real client; preserve connectivity signal.
                was_connected = client._mode is not None
                try:
                    await client.disconnect()
                except Exception:
                    pass
                try:
                    await stdio_docker_remove_force(
                        stdio_name,
                        phase="tool_call_error",
                        log_tenant=project_id,
                        log_server=server_id,
                    )
                except Exception:
                    pass
                self._local_stdio_clients.pop(cache_key, None)
                return _external_tool_call_error_result(
                    tool_id, e, was_connected=was_connected
                )

        if not self.external_mcp_manager:
            return {
                "status": "error",
                "error": "External MCP manager is not initialized",
                "tool_id": tool_id,
            }

        if not image:
            return {
                "status": "error",
                "error": (
                    "Configure metadata.external_mcp.image or metadata.external_mcp.command for local stdio runtimes "
                    "or metadata.external_mcp.endpoint for remote HTTP runtimes"
                ),
                "tool_id": tool_id,
            }

        on_pc_raw = (str(runtime.get("on_project_complete") or "remove")).strip().lower()
        on_project_complete = (
            "stop_only" if on_pc_raw in ("stop_only", "stop", "soft") else "remove"
        )
        container_port = int(runtime.get("container_port", 8080))
        endpoint_path = runtime.get("path", "/mcp")
        runtime_mode = "streamable-http" if mode == "streamable-http" else "http"
        logger.info(
            "[EXTERNAL_MCP] [EXEC] tool_id=%s project_id=%s manager_project_id=%s "
            "tenant_id=%s server_id=%s scope=%s on_project_complete=%s idle_timeout=%s",
            tool_id,
            project_id,
            manager_project_id,
            tenant_for_runtime,
            server_id,
            rscope,
            on_project_complete,
            idle_timeout_param,
        )
        try:
            await self.external_mcp_manager.ensure_server(
                project_id=manager_project_id,
                server_id=server_id,
                tenant_id=tenant_for_runtime,
                image=image,
                container_port=container_port,
                endpoint_path=endpoint_path,
                mode=runtime_mode,
                docker_env_vars=runtime.get("docker_env_vars") if isinstance(runtime.get("docker_env_vars"), dict) else None,
                docker_cmd_args=runtime.get("docker_cmd_args") if isinstance(runtime.get("docker_cmd_args"), list) else None,
                idle_timeout_seconds=idle_timeout_param,
                runtime_scope=rscope,
                on_project_complete=on_project_complete,
            )
        except ConnectionError as e:
            return {
                "status": "error",
                "error": str(e),
                "tool_id": tool_id,
            }
        except Exception as e:
            # ensure_server failed before a live tool call — retriable, not outcome_unknown.
            return {
                "status": "error",
                "error": str(e) or f"{type(e).__name__} (no message)",
                "tool_id": tool_id,
            }
        self._track_project_manager(project_id, manager_project_id, tenant_for_runtime, server_id)
        # After ensure_server the session is connected; call failures may already
        # have applied a remote side-effect (same contract as stdio/remote HTTP).
        try:
            return await self.external_mcp_manager.call_tool(
                project_id=manager_project_id,
                tenant_id=tenant_for_runtime,
                server_id=server_id,
                tool_name=mcp_tool_name,
                arguments=params,
            )
        except Exception as e:
            # ensure_server already connected the session; ConnectionError subclasses
            # after that are still outcome_unknown (BrokenPipe mid-response, etc.).
            return _external_tool_call_error_result(
                tool_id, e, was_connected=True
            )

    async def _get_default_external_mcp_image(self) -> str:
        if self.storage and hasattr(self.storage, "get_default_external_mcp_image"):
            try:
                image = await self.storage.get_default_external_mcp_image()
                if isinstance(image, str) and image.strip():
                    return image.strip()
            except Exception as e:
                logger.warning("[EXTERNAL_MCP] failed reading default image from storage: %s", e)
        return DEFAULT_EXTERNAL_MCP_IMAGE

    async def _execute_external_tool_http_remote(
        self,
        *,
        tool_id: str,
        project_id: str,
        params: Dict[str, Any],
        runtime: Dict[str, Any],
        remote_url: str,
        mcp_tool_name: str,
        project_tenant: str,
        server_id: str,
    ) -> Dict[str, Any]:
        timeout = float(runtime.get("timeout_seconds") or 60.0)
        mode = (str(runtime.get("mode") or "http")).strip()
        tenant_remote = _legacy_root_project_tenant_id(project_tenant)
        headers = _runtime_headers_dict(runtime)
        # Same cached provider as discovery/health for this server → one token per TTL, and
        # OAuth2 refreshes + retries on 401 mid-run (contracts 2-4). None for no-auth configs.
        # Guarded: a malformed stored auth block or missing secret env becomes a tool error,
        # not an uncaught raise (execute_tool has no outer handler for this path).
        try:
            auth = build_mcp_auth(runtime.get("auth"), tenant_id=tenant_remote, server_id=server_id)
        except Exception as e:
            logger.warning(
                "[EXTERNAL_MCP] project=%s tool=%s remote_http=%s — auth build failed: %s",
                project_id, tool_id, remote_url, e,
            )
            return {"status": "error", "error": str(e), "tool_id": tool_id}
        for retry_delay in (*REMOTE_CONNECT_RETRY_DELAYS, None):
            client = ExternalMCPClient(
                ExternalMCPConfig(
                    server_id=server_id,
                    endpoint=remote_url,
                    tenant_id=tenant_remote,
                    timeout_seconds=timeout,
                    mode=mode,
                    headers=headers,
                    auth=auth,
                )
            )
            try:
                raw = await client.call_tool(mcp_tool_name, params)
                if isinstance(raw, dict):
                    result = raw
                else:
                    result = {"status": "success", "data": raw}
                break
            except Exception as e:
                logger.warning(
                    "[EXTERNAL_MCP] project=%s tool=%s remote_http=%s — call failed: %s",
                    project_id,
                    tool_id,
                    remote_url,
                    e,
                )
                was_connected = client._mode is not None
                # Connect-time ConnectionError has was_connected=False (retriable).
                # Post-connect BrokenPipe/Reset still have was_connected=True.
                try:
                    await client.disconnect()
                except Exception:
                    pass
                if was_connected or retry_delay is None:
                    return _external_tool_call_error_result(
                        tool_id, e, was_connected=was_connected
                    )
                logger.warning(
                    "[EXTERNAL_MCP] project=%s tool=%s remote_http=%s — session never opened, retrying in %.0fs",
                    project_id,
                    tool_id,
                    remote_url,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)
        try:
            await client.disconnect()
        except Exception as disc_exc:
            logger.error(
                "[EXTERNAL_MCP] project=%s tool=%s remote_http=%s — disconnect failed after success: %s",
                project_id,
                tool_id,
                remote_url,
                disc_exc,
            )
            if isinstance(result, dict):
                result = dict(result)
                result["cleanup_error"] = str(disc_exc)
            else:
                result = {
                    "status": "success",
                    "data": result,
                    "cleanup_error": str(disc_exc),
                }
            return result
        return result

    async def _handle_ask_human(
        self,
        project_id: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Park the agent on its journal record until a human answers."""
        import tools.ask_human as ask_human_module

        question = params.get("question", "").strip()
        if not question:
            return {"status": "error", "error": "ask_human: missing required param 'question'"}

        # Both injected by AgentToolDispatcher from the runner's journal
        # record — together they identify the one record the answer route
        # resolves against, since a call id alone repeats across runs.
        return await ask_human_module.handle_ask_human(
            question=question,
            project_id=project_id,
            tool_call_id=params.get("_tool_call_id"),
            message_store=self.message_store,
            run_id=params.get("_run_id"),
        )

    # ------------------------------------------------------------------
    # Archive retrieval tools (AppFactory-148). project_id is the caller's own
    # project from the runtime — the identity every ownership check keys on.
    # ------------------------------------------------------------------

    @property
    def archive_store(self):
        """Lazy read-side ArchiveStore (S3 settings from env, refs via storage)."""
        if self._archive_store is None:
            from storage.archive_store import ArchiveStore

            self._archive_store = ArchiveStore.from_env(self.storage)
        return self._archive_store

    async def _handle_archive_inspect(self, project_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        from tools import archive_tools

        return await archive_tools.inspect_ref(self.archive_store, project_id, params or {})

    async def _handle_archive_query(self, project_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        from tools import archive_tools

        return await archive_tools.query_ref(self.archive_store, project_id, params or {})

    async def _handle_archive_fetch(self, project_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        from tools import archive_tools

        return await archive_tools.fetch_ref(
            self.archive_store, self.container_manager, project_id, params or {},
            storage=self.storage,
        )

    @property
    def file_blob_store(self):
        """Lazy FileBlobStore (S3 settings from env)."""
        if self._file_blob_store is None:
            from storage.file_blob_store import FileBlobStore

            self._file_blob_store = FileBlobStore.from_env()
        return self._file_blob_store

    def _attachment_scope(self, params: Dict[str, Any] | None) -> Dict[str, Any]:
        """Drop agent-supplied project_id/tenant_id — scope comes from the project doc."""
        return {
            k: v for k, v in (params or {}).items() if k not in ("project_id", "tenant_id")
        }

    async def _handle_attachment_list(self, project_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        from tools import attachment_tools

        tenant_id = await self._get_project_tenant_id(project_id)
        return await attachment_tools.list_attachments(self.storage, project_id, tenant_id)

    async def _handle_attachment_view(self, project_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        from tools import attachment_tools

        tenant_id = await self._get_project_tenant_id(project_id)
        return await attachment_tools.view_attachment(
            self.storage,
            self.file_blob_store,
            project_id,
            tenant_id,
            self._attachment_scope(params),
        )

    async def _handle_attachment_fetch(self, project_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        from tools import attachment_tools

        tenant_id = await self._get_project_tenant_id(project_id)
        return await attachment_tools.fetch_attachment(
            self.storage,
            self.file_blob_store,
            self.container_manager,
            project_id,
            tenant_id,
            self._attachment_scope(params),
        )

    async def _handle_attachment_presign_get(self, project_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        from tools import attachment_tools

        tenant_id = await self._get_project_tenant_id(project_id)
        return await attachment_tools.attachment_presign_get(
            self.storage,
            self.file_blob_store,
            project_id,
            tenant_id,
            self._attachment_scope(params),
        )

    async def _handle_attachment_presign_put(self, project_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        from tools import attachment_tools

        tenant_id = await self._get_project_tenant_id(project_id)
        return await attachment_tools.attachment_presign_put(
            self.storage,
            self.file_blob_store,
            project_id,
            tenant_id,
            self._attachment_scope(params),
        )

    async def _handle_tenant_artifact_list(self, project_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        from tools import tenant_artifact_tools

        tenant_id = await self._get_project_tenant_id(project_id)
        return await tenant_artifact_tools.list_tenant_artifacts(self.storage, tenant_id)

    async def _handle_tenant_artifact_fetch(self, project_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        from tools import tenant_artifact_tools

        tenant_id = await self._get_project_tenant_id(project_id)
        return await tenant_artifact_tools.fetch_tenant_artifact(
            self.storage,
            self.file_blob_store,
            self.container_manager,
            project_id,
            tenant_id,
            self._attachment_scope(params),
        )
