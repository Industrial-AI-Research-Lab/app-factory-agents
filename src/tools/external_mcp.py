"""External MCP client (discover/call/disconnect).

Supports three transport modes:
- stdio      — subprocess via official MCP SDK (local command or docker image)
- http       — legacy JSON-RPC over HTTP POST
- streamable-http — MCP Streamable HTTP via official SDK (POST + optional SSE, not legacy GET-only SSE)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.stdio import stdio_client

from tools.mcp_ascii_transport import ascii_mcp_http_client

logger = logging.getLogger(__name__)

_AppFactory_MCP_IMAGE_PREFIX = "AppFactory-mcp/"


class MCPDisconnectCleanupError(RuntimeError):
    """MCP session teardown failed; tool outcome may be unknown."""


class MCPCallTimeoutError(TimeoutError):
    """A ``tools/call`` did not answer within the server's ``timeout_seconds``.

    The caller releases the connection and surfaces a tool error with reason
    ``timeout`` (AppFactory-267). Carries the tool name, server id and timeout so the
    executor can build a machine-readable result without parsing the message.
    """

    def __init__(self, *, tool_name: str, server_id: str, timeout_seconds: float) -> None:
        self.tool_name = tool_name
        self.server_id = server_id
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"external MCP tool {tool_name!r} on server {server_id!r} did not "
            f"respond within {timeout_seconds:g}s"
        )


async def docker_image_exists_locally(
    image: str,
    *,
    env: Optional[Dict[str, str]] = None,
) -> bool:
    """True when ``docker image inspect`` succeeds (image present on this daemon)."""
    ref = str(image or "").strip()
    if not ref:
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "image",
            "inspect",
            ref,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )
        await asyncio.wait_for(proc.communicate(), timeout=15)
        return proc.returncode == 0
    except asyncio.TimeoutError:
        logger.warning("[EXTERNAL_MCP] docker image inspect timed out image=%s", ref)
        return False


def stdio_docker_container_name(project_id: str, server_id: str) -> str:
    """Deterministic short Docker name for stdio+image MCP (per project + server)."""
    h = hashlib.sha256(f"{project_id}\0{server_id}".encode()).hexdigest()[:24]
    return f"synmcp{h}"


async def stdio_docker_remove_force(
    container_name: str,
    *,
    phase: str,
    log_tenant: str = "",
    log_server: str = "",
) -> None:
    """Idempotent `docker rm -f` for stdio MCP containers (may be called without ExternalMCPClient)."""
    if not container_name:
        return
    ex = 0
    err = ""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "-f",
            container_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        ex = int(proc.returncode or 0)
        err = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
    except asyncio.TimeoutError:
        logger.warning(
            "[EXTERNAL_MCP] [STDIO_DOCKER] rm timeout container=%s phase=%s ctx=%s/%s",
            container_name,
            phase,
            log_tenant,
            log_server,
        )
        try:
            if proc.returncode is None:
                proc.kill()
        except ProcessLookupError:
            pass
        return
    except FileNotFoundError:
        logger.warning(
            "[EXTERNAL_MCP] [STDIO_DOCKER] rm skipped — docker not in PATH (container=%s phase=%s)",
            container_name,
            phase,
        )
        return
    if ex == 0 or "No such container" in err:
        logger.info(
            "[EXTERNAL_MCP] [STDIO_DOCKER] rm container=%s phase=%s exit=%s ctx=%s/%s",
            container_name,
            phase,
            ex,
            log_tenant,
            log_server,
        )
        return
    logger.warning(
        "[EXTERNAL_MCP] [STDIO_DOCKER] rm failed container=%s phase=%s exit=%s ctx=%s/%s stderr=%s",
        container_name,
        phase,
        ex,
        log_tenant,
        log_server,
        err[:800],
    )


def format_mcp_user_message(exc: BaseException) -> str:
    """Flatten ExceptionGroup / TaskGroup sub-exceptions into one line for API error detail."""
    if isinstance(exc, BaseExceptionGroup):
        parts = [format_mcp_user_message(e) for e in exc.exceptions]
        return "; ".join(p for p in parts if p) or str(exc)
    return str(exc)


@dataclass
class ExternalMCPConfig:
    """Connection settings for an external MCP server."""

    server_id: str
    endpoint: str
    tenant_id: str
    timeout_seconds: float = 30.0
    mode: str = "auto"           # auto | stdio | http | streamable-http
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    # Outbound auth for http / streamable-http, attached via httpx `auth=` (NOT frozen into
    # static headers) so an OAuth2 bearer refreshes and retries on 401. None for no auth and
    # for stdio (which has no HTTP layer). See tools.mcp_auth / integrations.a2a_auth.
    auth: Optional[httpx.Auth] = None
    image: Optional[str] = None          # Docker image for stdio mode
    docker_env_vars: Optional[Dict[str, str]] = None
    docker_cmd_args: Optional[List[str]] = None
    # When set, `docker run` for stdio+image uses `--name` and best-effort `docker rm -f` on disconnect.
    stdio_docker_name: Optional[str] = None


class ExternalMCPClient:
    """External MCP client with discover/call/disconnect lifecycle."""

    def __init__(self, config: ExternalMCPConfig):
        self.config = config
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._request_id: int = 0
        self._mode: Optional[str] = None
        self._docker_env: Optional[Dict[str, str]] = None
        self._tmp_docker_dir: Optional[str] = None
        # Serialize connect/disconnect so concurrent call_tool / discover_tools cannot
        # race in _connect_* and overwrite _exit_stack (orphan docker stdio, leaked stacks).
        self._connect_lock = asyncio.Lock()
        # Set in _connect_stdio for docker+stdio; used for idempotent `docker rm -f` in disconnect.
        self._docker_stdio_container_name: Optional[str] = None
        # Task that opened the live MCP session — disconnect should run here (AppFactory-274 AC 2).
        self._lifecycle_task: Optional[asyncio.Task] = None

    @property
    def stdio_docker_container_name(self) -> Optional[str]:
        return self._docker_stdio_container_name

    def _cleanup_tmp_docker_dir(self) -> None:
        """Remove Docker auth tmpdir created for private-registry login (best-effort)."""
        path = self._tmp_docker_dir
        if not path:
            return
        self._tmp_docker_dir = None
        self._docker_env = None
        try:
            shutil.rmtree(path, ignore_errors=True)
            logger.info(
                "[EXTERNAL_MCP] cleaned tmp docker auth dir tenant=%s server=%s path=%s",
                self.config.tenant_id,
                self.config.server_id,
                path,
            )
        except Exception as e:
            logger.warning(
                "[EXTERNAL_MCP] tmp docker dir cleanup failed tenant=%s server=%s path=%s: %s",
                self.config.tenant_id,
                self.config.server_id,
                path,
                e,
            )

    async def _abort_stdio_connect_attempt(self) -> None:
        """Release partial stdio stack and Docker auth tmpdir after a failed connect (no full disconnect)."""
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass
            self._exit_stack = None
            self._session = None
        self._cleanup_tmp_docker_dir()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def discover_tools(self) -> List[Dict[str, Any]]:
        """Discover tools from external MCP server."""
        await self._ensure_connected()

        if self._mode in ("stdio", "streamable-http"):
            assert self._session is not None
            result = await asyncio.wait_for(
                self._session.list_tools(),
                timeout=self.config.timeout_seconds,
            )
            return [self._tool_to_dict(t) for t in result.tools]

        if self._mode == "http":
            payload = await self._http_rpc("tools/list", {})
            tools = payload.get("tools", [])
            if not isinstance(tools, list):
                return []
            return [self._http_tool_to_dict(t) for t in tools if isinstance(t, dict)]

        raise RuntimeError("External MCP client is not connected")

    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call a tool on the external MCP server."""
        await self._ensure_connected()
        # Strip None values so optional parameters use server-side defaults
        # instead of triggering AttributeError inside MCP server implementations.
        args = {k: v for k, v in (arguments or {}).items() if v is not None}

        if self._mode in ("stdio", "streamable-http"):
            assert self._session is not None
            # Bound the call the way discover_tools bounds list_tools. Without this the
            # await blocks until the transport's SSE idle timeout (>=300s) fires inside a
            # background task and is lost as "Task exception was never retrieved", so the
            # run hangs until a manual stop. The caller releases the connection on error.
            try:
                result = await asyncio.wait_for(
                    self._session.call_tool(tool_name, arguments=args),
                    timeout=self.config.timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise MCPCallTimeoutError(
                    tool_name=tool_name,
                    server_id=self.config.server_id,
                    timeout_seconds=self.config.timeout_seconds,
                ) from exc
            return self._normalize_stdio_result(result)

        if self._mode == "http":
            payload = await self._http_rpc("tools/call", {"name": tool_name, "arguments": args})
            return self._normalize_http_result(payload)

        raise RuntimeError("External MCP client is not connected")

    def peek_stdio_docker_name(self) -> Optional[str]:
        """Container name for deferred rm (set after stdio+image connect, cleared on disconnect)."""
        name = getattr(self, "_docker_stdio_container_name", None)
        if name:
            return name
        config = getattr(self, "config", None)
        if config is None:
            return None
        return getattr(config, "stdio_docker_name", None) or None

    async def disconnect(self, *, remove_stdio_docker: bool = True) -> Optional[str]:
        """Disconnect and release session resources.

        By default also best-effort ``docker rm -f`` for named stdio runtimes.
        Pass ``remove_stdio_docker=False`` to close anyio/http only and return the
        container name so the caller can schedule rm off the probe task (health
        ``background_cleanup``).
        """
        docker_name = self._docker_stdio_container_name
        mode_before = self._mode
        owner = self._lifecycle_task
        current = asyncio.current_task()
        if owner is not None and current is not owner:
            owner_name = owner.get_name() if hasattr(owner, "get_name") else str(owner)
            current_name = current.get_name() if current and hasattr(current, "get_name") else str(current)
            logger.warning(
                "[EXTERNAL_MCP] [LIFECYCLE] disconnect task mismatch tenant=%s server=%s "
                "owner=%s current=%s",
                self.config.tenant_id,
                self.config.server_id,
                owner_name,
                current_name,
            )
        logger.info(
            "[EXTERNAL_MCP] [DISCONNECT] start tenant=%s server=%s mode=%s stdio_docker_name=%s "
            "remove_stdio_docker=%s",
            self.config.tenant_id,
            self.config.server_id,
            mode_before,
            docker_name,
            remove_stdio_docker,
        )
        aclose_exc: Optional[BaseException] = None
        deferred_docker: Optional[str] = None
        async with self._connect_lock:
            if self._http is not None:
                try:
                    await self._http.aclose()
                    logger.info(
                        "[EXTERNAL_MCP] [DISCONNECT] http client closed tenant=%s server=%s",
                        self.config.tenant_id,
                        self.config.server_id,
                    )
                except Exception as e:
                    aclose_exc = e
                    logger.warning(
                        "[EXTERNAL_MCP] [DISCONNECT] http aclose failed tenant=%s server=%s: %s",
                        self.config.tenant_id,
                        self.config.server_id,
                        e,
                    )
                finally:
                    self._http = None

            if self._exit_stack is not None:
                try:
                    await self._exit_stack.aclose()
                    logger.info(
                        "[EXTERNAL_MCP] [DISCONNECT] stdio/stream exit_stack closed tenant=%s server=%s",
                        self.config.tenant_id,
                        self.config.server_id,
                    )
                except Exception as e:
                    aclose_exc = e
                    logger.warning(
                        "[EXTERNAL_MCP] [DISCONNECT] exit_stack aclose failed tenant=%s server=%s: %s",
                        self.config.tenant_id,
                        self.config.server_id,
                        e,
                    )
                finally:
                    self._exit_stack = None
                    self._session = None

            if docker_name:
                if remove_stdio_docker:
                    await stdio_docker_remove_force(
                        docker_name,
                        phase="after_aclose" if aclose_exc is None else "after_aclose_error",
                        log_tenant=self.config.tenant_id,
                        log_server=self.config.server_id,
                    )
                else:
                    deferred_docker = docker_name

            self._docker_stdio_container_name = None
            self._mode = None
            self._lifecycle_task = None
            self._cleanup_tmp_docker_dir()

        if aclose_exc is not None:
            logger.warning(
                "[EXTERNAL_MCP] [DISCONNECT] done tenant=%s server=%s — completed with aclose error (see above)",
                self.config.tenant_id,
                self.config.server_id,
            )
            raise MCPDisconnectCleanupError(
                f"MCP session cleanup failed for server {self.config.server_id!r}"
            ) from aclose_exc
        else:
            logger.info(
                "[EXTERNAL_MCP] [DISCONNECT] done tenant=%s server=%s mode=disconnected deferred_docker=%s",
                self.config.tenant_id,
                self.config.server_id,
                deferred_docker,
            )
        return deferred_docker

    # ------------------------------------------------------------------
    # Connection internals
    # ------------------------------------------------------------------

    async def _ensure_connected(self) -> None:
        if self._mode is not None:
            return
        async with self._connect_lock:
            if self._mode is not None:
                return
            mode = self._resolve_mode(self.config.mode, self.config.endpoint)
            try:
                if mode == "stdio":
                    await self._connect_stdio()
                elif mode == "streamable-http":
                    await self._connect_streamable_http()
                elif mode == "http":
                    await self._connect_http()
                else:
                    raise ValueError(f"Unsupported external MCP mode: {mode}")
            except asyncio.CancelledError as e:
                # anyio's cancel scope can surface CancelledError while unwinding
                # even when the remote side already accepted the request.  Only
                # treat it as a connection failure when the session was never
                # established (_mode is still None).  If _mode was already set
                # (call-level cancel), let CancelledError propagate so callers
                # preserve was_connected=True and set outcome_unknown correctly.
                if self._mode is None:
                    raise ConnectionError(
                        f"external MCP server {self.config.server_id!r} is unreachable "
                        f"at {self.config.endpoint}"
                    ) from e
                raise
            self._lifecycle_task = asyncio.current_task()

    async def _connect_stdio(self) -> None:
        if self.config.image:
            command = "docker"
            raw_name = (self.config.stdio_docker_name or "").strip()
            self._docker_stdio_container_name = raw_name or None
            if self._docker_stdio_container_name:
                await stdio_docker_remove_force(
                    self._docker_stdio_container_name,
                    phase="stale_before_run",
                    log_tenant=self.config.tenant_id,
                    log_server=self.config.server_id,
                )
            base_args: List[str] = ["run", "--rm", "-i"]
            if self._docker_stdio_container_name:
                base_args += ["--name", self._docker_stdio_container_name, "--label", "AppFactory.mcp.stdio=1"]
            for k, v in (self.config.docker_env_vars or {}).items():
                base_args += ["-e", f"{k}={v}"]
            base_args.append(self.config.image)
            if self.config.docker_cmd_args:
                base_args += self.config.docker_cmd_args
            args = base_args
        else:
            self._docker_stdio_container_name = None
            command = self.config.command or self.config.endpoint
            args = self.config.args or []

        if not command:
            raise ValueError("stdio mode requires either an image or a command/endpoint")

        if self.config.image:
            logger.info(
                "[EXTERNAL_MCP] [STDIO_DOCKER] run tenant=%s server=%s image=%s container_name=%s",
                self.config.tenant_id,
                self.config.server_id,
                self.config.image,
                self._docker_stdio_container_name or "(unnamed)",
            )
        logger.info(
            "[EXTERNAL_MCP] tenant=%s server=%s stdio cmd=%s args=%s",
            self.config.tenant_id,
            self.config.server_id,
            command,
            args,
        )

        if self.config.image:
            await self._docker_preflight(self.config.image)

        self._exit_stack = AsyncExitStack()
        run_env = self._docker_env or self.config.env
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=run_env,
            cwd=self.config.cwd,
        )
        try:
            read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))
            self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(self._session.initialize(), timeout=self.config.timeout_seconds)
        except FileNotFoundError as exc:
            cmd = (command or "").strip().lower()
            fname = (getattr(exc, "filename", None) or "").lower()
            hint = ""
            if cmd in ("npx", "node", "npm"):
                hint = (
                    " На образе API нет Node/npx — поставьте Node в контейнер backend или используйте "
                    "MCP по HTTPS (url) / Docker image вместо command=npx."
                )
            elif cmd == "docker" or "docker" in fname:
                hint = " Docker CLI недоступен процессу API (нет в PATH или нет сокета)."
            logger.error(
                "[EXTERNAL_MCP] tenant=%s server=%s stdio executable missing: %s",
                self.config.tenant_id,
                self.config.server_id,
                exc,
            )
            await self._abort_stdio_connect_attempt()
            raise RuntimeError(f"stdio MCP: программа не найдена ({exc.filename or exc}).{hint}") from exc
        except Exception as exc:
            logger.error(
                "[EXTERNAL_MCP] [CONNECT_FAIL] transport=stdio tenant=%s server=%s "
                "phase=session_init exc_type=%s errno=%s msg=%s",
                self.config.tenant_id,
                self.config.server_id,
                type(exc).__name__,
                getattr(exc, "errno", None),
                str(exc)[:800],
                exc_info=True,
            )
            await self._abort_stdio_connect_attempt()
            raise

        self._mode = "stdio"
        logger.info(
            "[EXTERNAL_MCP] tenant=%s server=%s mode=stdio connected",
            self.config.tenant_id,
            self.config.server_id,
        )

    async def _connect_streamable_http(self) -> None:
        """Connect via MCP Streamable HTTP (SDK streamablehttp_client: POST + optional SSE).

        Public hosts such as mcp.context7.com expect this transport, not legacy GET-only ``sse_client``.
        """
        if not self.config.endpoint:
            raise ValueError("streamable-http mode requires an endpoint URL")

        timeout = float(self.config.timeout_seconds)
        sse_read_timeout = float(max(timeout, 300.0))

        logger.info(
            "[EXTERNAL_MCP] tenant=%s server=%s streamable-http url=%s timeout=%s sse_read=%s",
            self.config.tenant_id,
            self.config.server_id,
            self.config.endpoint,
            timeout,
            sse_read_timeout,
        )

        hdr = self.config.headers
        headers_arg = hdr if hdr else None
        self._exit_stack = AsyncExitStack()
        try:
            read, write, _get_session_id = await self._exit_stack.enter_async_context(
                streamablehttp_client(
                    self.config.endpoint,
                    headers=headers_arg,
                    timeout=timeout,
                    sse_read_timeout=sse_read_timeout,
                    auth=self.config.auth,
                    httpx_client_factory=ascii_mcp_http_client,
                )
            )
            self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(self._session.initialize(), timeout=timeout)
        except Exception as exc:
            logger.error(
                "[EXTERNAL_MCP] [CONNECT_FAIL] transport=streamable-http tenant=%s server=%s "
                "url=%s timeout=%s exc_type=%s errno=%s msg=%s",
                self.config.tenant_id,
                self.config.server_id,
                self.config.endpoint,
                timeout,
                type(exc).__name__,
                getattr(exc, "errno", None),
                str(exc)[:800],
                exc_info=True,
            )
            raise

        self._mode = "streamable-http"
        logger.info(
            "[EXTERNAL_MCP] tenant=%s server=%s mode=streamable-http connected",
            self.config.tenant_id,
            self.config.server_id,
        )

    async def _connect_http(self) -> None:
        """Connect via legacy JSON-RPC over HTTP POST."""
        timeout = httpx.Timeout(self.config.timeout_seconds)
        self._http = httpx.AsyncClient(
            timeout=timeout,
            headers=self.config.headers or {},
            auth=self.config.auth,
        )
        try:
            await self._http_rpc(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "AppFactory-external-mcp", "version": "0.1.0"},
                },
            )
        except Exception as exc:
            logger.error(
                "[EXTERNAL_MCP] [CONNECT_FAIL] transport=http tenant=%s server=%s "
                "endpoint=%s exc_type=%s msg=%s",
                self.config.tenant_id,
                self.config.server_id,
                self.config.endpoint,
                type(exc).__name__,
                str(exc)[:800],
                exc_info=True,
            )
            try:
                await self._http.aclose()
            except Exception:
                pass
            self._http = None
            raise

        self._mode = "http"
        logger.info(
            "[EXTERNAL_MCP] tenant=%s server=%s mode=http connected",
            self.config.tenant_id,
            self.config.server_id,
        )

    # ------------------------------------------------------------------
    # Docker helpers
    # ------------------------------------------------------------------

    async def _docker_preflight(self, image: str) -> None:
        """Verify Docker daemon reachability and pull the image."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode != 0:
                msg = stderr.decode(errors="replace").strip()
                logger.error("[EXTERNAL_MCP] docker info failed (rc=%s): %s", proc.returncode, msg)
                raise RuntimeError(f"Docker daemon not accessible: {msg}")
            logger.info("[EXTERNAL_MCP] docker info ok")
        except asyncio.TimeoutError:
            raise RuntimeError("Docker daemon timed out (docker info)")

        first_segment = image.split("/")[0] if "/" in image else ""
        is_private_registry = "." in first_segment or ":" in first_segment
        image_host = first_segment if is_private_registry else ""

        registry_user = os.environ.get("DEPLOY_AGENT_REGISTRY_USERNAME", "")
        registry_pass = os.environ.get("DEPLOY_AGENT_REGISTRY_PASSWORD", "")

        self._tmp_docker_dir = tempfile.mkdtemp(prefix="AppFactory-docker-")
        docker_env = {**os.environ, "DOCKER_CONFIG": self._tmp_docker_dir}

        try:
            await self._docker_preflight_inner(image, image_host, docker_env, registry_user, registry_pass)
            self._docker_env = docker_env
        except Exception:
            self._cleanup_tmp_docker_dir()
            raise

    async def _docker_preflight_inner(
        self,
        image: str,
        image_host: str,
        docker_env: Dict[str, str],
        registry_user: str,
        registry_pass: str,
    ) -> None:
        if image_host and registry_user and registry_pass:
            logger.info("[EXTERNAL_MCP] docker login host=%s user=%s", image_host, registry_user)
            try:
                proc = await asyncio.create_subprocess_exec(
                    "docker", "login", image_host,
                    "--username", registry_user,
                    "--password-stdin",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=docker_env,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=registry_pass.encode()), timeout=30
                )
                out = (stdout + stderr).decode(errors="replace").strip()
                if proc.returncode != 0:
                    logger.error("[EXTERNAL_MCP] docker login failed (rc=%s): %s", proc.returncode, out)
                    raise RuntimeError(f"docker login {image_host} failed: {out}")
                logger.info("[EXTERNAL_MCP] docker login ok: %s", out[-200:])
            except asyncio.TimeoutError:
                raise RuntimeError(f"docker login {image_host} timed out")
        else:
            if not image_host:
                logger.info("[EXTERNAL_MCP] Docker Hub image detected — skipping login")
            else:
                logger.warning(
                    "[EXTERNAL_MCP] no registry credentials; image_host=%s user_set=%s",
                    image_host, bool(registry_user),
                )

        if await docker_image_exists_locally(image, env=docker_env):
            logger.info(
                "[EXTERNAL_MCP] image %s present locally — skipping pull",
                image,
            )
            return

        if image.startswith(_AppFactory_MCP_IMAGE_PREFIX):
            logger.warning(
                "[EXTERNAL_MCP] AppFactory-mcp image %s not found locally before pull",
                image,
            )

        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "pull", image,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                env=docker_env,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            stderr_text = stderr.decode(errors="replace").strip()
            if proc.returncode != 0:
                logger.error(
                    "[EXTERNAL_MCP] docker pull %s failed (rc=%s): %s",
                    image, proc.returncode, stderr_text,
                )
                raise RuntimeError(f"docker pull {image} failed: {stderr_text}")
            logger.info("[EXTERNAL_MCP] docker pull %s ok", image)
        except asyncio.TimeoutError:
            raise RuntimeError(f"docker pull {image} timed out")

    # ------------------------------------------------------------------
    # HTTP JSON-RPC helpers
    # ------------------------------------------------------------------

    async def _http_rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._http is None:
            raise RuntimeError("HTTP client is not initialized")

        self._request_id += 1
        req = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        try:
            response = await self._http.post(self.config.endpoint, json=req)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body_preview = ""
            try:
                body_preview = (exc.response.text or "")[:600]
            except Exception:
                body_preview = ""
            logger.error(
                "[EXTERNAL_MCP] [HTTP_RPC] tenant=%s server=%s method=%s endpoint=%s "
                "status=%s body_preview=%s",
                self.config.tenant_id,
                self.config.server_id,
                method,
                self.config.endpoint,
                exc.response.status_code if exc.response else None,
                body_preview.replace("\n", " ")[:600],
            )
            raise RuntimeError(
                f"MCP HTTP {method}: status {exc.response.status_code if exc.response else '?'}"
            ) from exc
        except httpx.RequestError as exc:
            logger.error(
                "[EXTERNAL_MCP] [HTTP_RPC] tenant=%s server=%s method=%s endpoint=%s "
                "exc_type=%s msg=%s",
                self.config.tenant_id,
                self.config.server_id,
                method,
                self.config.endpoint,
                type(exc).__name__,
                str(exc)[:600],
            )
            raise RuntimeError(f"MCP HTTP {method}: request failed ({type(exc).__name__})") from exc

        data = response.json()
        if not isinstance(data, dict):
            logger.error(
                "[EXTERNAL_MCP] [HTTP_RPC] tenant=%s server=%s method=%s — non-object JSON body",
                self.config.tenant_id,
                self.config.server_id,
                method,
            )
            raise RuntimeError("Invalid JSON-RPC response format")
        if "error" in data:
            err = data.get("error")
            logger.error(
                "[EXTERNAL_MCP] [HTTP_RPC] tenant=%s server=%s method=%s endpoint=%s mcp_rpc_error=%s",
                self.config.tenant_id,
                self.config.server_id,
                method,
                self.config.endpoint,
                repr(err)[:800],
            )
            raise RuntimeError(f"MCP RPC error: {err}")
        result = data.get("result", {})
        if not isinstance(result, dict):
            return {}
        return result

    # ------------------------------------------------------------------
    # Mode resolution
    # ------------------------------------------------------------------

    def _resolve_mode(self, mode: str, endpoint: str) -> str:
        if mode in ("stdio", "http", "streamable-http"):
            return mode
        # auto-detect from endpoint scheme
        parsed = urlparse(endpoint)
        if parsed.scheme in ("http", "https"):
            return "http"
        return "stdio"

    # ------------------------------------------------------------------
    # Result normalisation helpers
    # ------------------------------------------------------------------

    def _tool_to_dict(self, tool: Any) -> Dict[str, Any]:
        schema = getattr(tool, "inputSchema", None)
        return {
            "name": getattr(tool, "name", ""),
            "description": getattr(tool, "description", "") or "",
            "schema": schema if isinstance(schema, dict) else None,
            "mcp_server": self.config.server_id,
        }

    def _http_tool_to_dict(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        schema = tool.get("inputSchema")
        return {
            "name": tool.get("name", ""),
            "description": tool.get("description", "") or "",
            "schema": schema if isinstance(schema, dict) else None,
            "mcp_server": self.config.server_id,
        }

    def _normalized_tool_status(self, raw: Any, text: str = "") -> Dict[str, Any]:
        if isinstance(raw, dict):
            is_error = raw.get("isError") is True or raw.get("is_error") is True
        else:
            is_error = (
                getattr(raw, "isError", False) is True
                or getattr(raw, "is_error", False) is True
            )
        if not is_error:
            return {"status": "success"}
        message = text.strip() or "MCP tool returned isError=true"
        return {"status": "error", "isError": True, "error": message}

    def _normalize_stdio_result(self, result: Any) -> Dict[str, Any]:
        text_parts: List[str] = []
        raw_content = getattr(result, "content", None)
        content_blocks = raw_content if isinstance(raw_content, list) else []

        for block in content_blocks:
            if isinstance(block, types.TextContent):
                text_parts.append(block.text or "")
            else:
                text_parts.append(str(block))

        text = "\n".join(text_parts).strip()
        out = self._normalized_tool_status(result, text)
        if text_parts:
            out["text"] = text

        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict):
            out["data"] = structured
        elif isinstance(structured, list):
            out["data"] = structured
        elif "text" in out:
            try:
                out["data"] = json.loads(out["text"])
            except Exception:
                pass

        return out

    def _normalize_http_result(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload.get("text") if isinstance(payload.get("text"), str) else ""
        content = payload.get("content")
        if not text and isinstance(content, list):
            text = "\n".join(
                block["text"]
                for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            ).strip()
        out = self._normalized_tool_status(payload, text)
        if "structuredContent" in payload:
            out["data"] = payload.get("structuredContent")
        if "content" in payload:
            out["content"] = payload.get("content")
        if isinstance(payload.get("text"), str):
            out["text"] = text
        return out
