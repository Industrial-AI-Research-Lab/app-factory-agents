"""Server-level lifecycle operations for ZIP-hosted MCP configurations."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from config.configuration_cow import prepare_configuration_cow_update
from config.configuration_resolution import SYSTEM_TENANT_ID, agent_wire_name_from_doc
from config.tool_configuration_schema import tool_doc_matches_allow_ref
from sandbox.host_cli import run_host_cli
from storage.tool_doc_storage import delete_tool_document, get_mcp_tool_configurations
from tools.mcp_package_storage import (
    claim_mcp_built_image_deletion,
    claim_mcp_package_deletion,
    delete_built_image_doc,
    delete_package_doc,
    image_tag_in_use,
    list_built_image_docs,
    list_package_docs,
    release_mcp_built_image_deletion,
    release_mcp_package_deletion,
)
from tools.mcp_delete_lease_heartbeat import McpDeleteLeaseHeartbeat, McpDeleteLeaseLost
from tools.mcp_runtime_scope import MCP_TENANT_SCOPE_PROJECT_ID
from tools.mcp_zip_service import cleanup_prior_upload_extract


_LIFECYCLE_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}
_LIFECYCLE_LOCKS_GUARD = asyncio.Lock()
logger = logging.getLogger(__name__)


class McpLifecycleConflict(RuntimeError):
    """The requested lifecycle action is not supported by the server."""


class McpServerNotFound(RuntimeError):
    """No tenant-owned MCP configuration exists for the requested server."""


def _external_mcp_runtime(tool_doc: dict[str, Any]) -> dict[str, Any]:
    metadata = tool_doc.get("metadata") if isinstance(tool_doc.get("metadata"), dict) else {}
    runtime = metadata.get("external_mcp") if isinstance(metadata.get("external_mcp"), dict) else {}
    return runtime


def _restart_runtime_from_tool(tool_doc: dict[str, Any]) -> dict[str, Any] | None:
    """Return one normalized restart contract for comparing all server tools."""
    runtime = _external_mcp_runtime(tool_doc)
    try:
        container_port = int(runtime.get("container_port") or 8080)
    except (TypeError, ValueError):
        return None
    return {
        "source": str(runtime.get("source") or "").strip().lower(),
        "image": str(runtime.get("image") or "").strip(),
        "mode": str(runtime.get("mode") or "http").strip().lower(),
        "runtime_scope": str(runtime.get("runtime_scope") or "project").strip().lower(),
        "container_port": container_port,
        "path": str(runtime.get("path") or runtime.get("endpoint_path") or "/mcp"),
        "docker_env_vars": runtime.get("docker_env_vars")
        if isinstance(runtime.get("docker_env_vars"), dict)
        else None,
        "docker_cmd_args": runtime.get("docker_cmd_args")
        if isinstance(runtime.get("docker_cmd_args"), list)
        else None,
        "idle_timeout_seconds": runtime.get("idle_timeout_seconds"),
        "on_project_complete": str(runtime.get("on_project_complete") or "remove"),
    }


@dataclass(frozen=True)
class McpServerResource:
    tenant_id: str
    server_id: str
    runtime: dict[str, Any]
    restart_runtime: dict[str, Any] | None
    is_zip_hosted: bool
    tool_docs: list[dict[str, Any]]
    built_image_doc_ids: list[str]
    image_tags: list[str]
    package_docs: list[dict[str, Any]]

    def runtime_key(self) -> dict[str, str]:
        return {
            "project_id": MCP_TENANT_SCOPE_PROJECT_ID,
            "tenant_id": self.tenant_id,
            "server_id": self.server_id,
        }

    def ensure_kwargs(self) -> dict[str, Any]:
        if self.restart_runtime is None:
            raise McpLifecycleConflict(
                "Restart requires consistent ZIP runtime metadata across all server tools"
            )
        runtime = self.restart_runtime
        if runtime.get("source") != "zip":
            raise McpLifecycleConflict("Restart is available only for ZIP/self-hosted MCP servers")
        runtime_scope = str(runtime.get("runtime_scope") or "project").strip().lower()
        if runtime_scope != "tenant":
            raise McpLifecycleConflict("Restart requires runtime_scope=tenant")
        image = str(runtime.get("image") or "").strip()
        if not image:
            raise McpLifecycleConflict("Restart requires a self-hosted Docker image")
        mode = str(runtime.get("mode") or "http").strip().lower()
        if mode != "streamable-http":
            raise McpLifecycleConflict("Restart is supported only for streamable-http ZIP MCP servers")
        return {
            **self.runtime_key(),
            "image": image,
            "container_port": int(runtime.get("container_port") or 8080),
            "endpoint_path": str(runtime.get("path") or runtime.get("endpoint_path") or "/mcp"),
            "mode": "streamable-http",
            "docker_env_vars": runtime.get("docker_env_vars")
            if isinstance(runtime.get("docker_env_vars"), dict)
            else None,
            "docker_cmd_args": runtime.get("docker_cmd_args")
            if isinstance(runtime.get("docker_cmd_args"), list)
            else None,
            "idle_timeout_seconds": runtime.get("idle_timeout_seconds"),
            "runtime_scope": "tenant",
            "on_project_complete": str(runtime.get("on_project_complete") or "remove"),
        }

class McpServerLifecycle:
    """Coordinate lifecycle actions for one tenant-owned MCP server."""

    def __init__(
        self,
        *,
        storage: Any,
        manager: Any,
        mcp_executor: Any | None = None,
        remove_image=None,
        remove_containers=None,
    ):
        self._storage = storage
        self._manager = manager
        self._mcp_executor = mcp_executor
        self._remove_image = remove_image or self._remove_docker_image
        self._remove_containers = remove_containers or self._remove_docker_containers

    async def restart(self, *, tenant_id: str, server_id: str) -> dict[str, Any]:
        async with await self._lock_for(tenant_id, server_id):
            resource = await self._load_resource(tenant_id=tenant_id, server_id=server_id)
            if not resource.is_zip_hosted:
                raise McpLifecycleConflict("Restart is available only for ZIP/self-hosted MCP servers")
            ensure_kwargs = resource.ensure_kwargs()
            await self._manager.stop_all_for_tenant_mcp_server(
                tenant_id=tenant_id, mcp_server_id=server_id
            )
            await self._manager.ensure_server(**ensure_kwargs)
            tools = await self._manager.discover_tools(**resource.runtime_key())
        return {
            "status": "restarted",
            "tenant_id": tenant_id,
            "server_id": server_id,
            "tools_count": len(tools) if isinstance(tools, list) else 0,
        }

    async def delete(
        self, *, tenant_id: str, server_id: str, actor_id: str | None = None
    ) -> dict[str, Any]:
        async with await self._lock_for(tenant_id, server_id):
            resource = await self._load_resource(tenant_id=tenant_id, server_id=server_id)
            reservation_id = uuid.uuid4().hex
            reserved_package_ids: list[str] = []
            reserved_image_ids: list[str] = []
            heartbeat: McpDeleteLeaseHeartbeat | None = None
            try:
                reserved_package_ids = await self._reserve_package_deletions(
                    resource, reservation_id
                )
                reserved_image_ids = await self._reserve_image_deletions(resource, reservation_id)
                heartbeat = McpDeleteLeaseHeartbeat(
                    storage=self._storage,
                    tenant_id=tenant_id,
                    package_server_ids=reserved_package_ids,
                    package_reservation_id=reservation_id,
                    image_doc_ids=reserved_image_ids,
                    image_reservation_id=reservation_id,
                )
                await heartbeat.start()
                if resource.is_zip_hosted:
                    await self._assert_images_not_referenced_elsewhere(resource)
                containers_removed = 0
                await self._assert_lease_healthy(heartbeat)
                await self._remove_runtime(
                    tenant_id=tenant_id,
                    server_id=server_id,
                )
                if resource.is_zip_hosted:
                    await self._assert_lease_healthy(heartbeat)
                    containers_removed = await self._remove_containers(
                        tenant_id=tenant_id, server_id=server_id
                    )
                    for image_tag in resource.image_tags:
                        await self._assert_lease_healthy(heartbeat)
                        await self._remove_image(image_tag)
                    for package_doc in resource.package_docs:
                        await self._assert_lease_healthy(heartbeat)
                        package_meta = self._package_meta(package_doc)
                        await asyncio.to_thread(
                            cleanup_prior_upload_extract, package_meta.get("extract_path")
                        )
                await self._assert_lease_healthy(heartbeat)
                agent_tool_links_removed = await self._remove_agent_tool_references(
                    resource.tool_docs, tenant_id=tenant_id, actor_id=actor_id
                )
                for tool in resource.tool_docs:
                    await self._assert_lease_healthy(heartbeat)
                    await delete_tool_document(self._storage, tool)
                for doc_id in resource.built_image_doc_ids:
                    await self._assert_lease_healthy(heartbeat)
                    await delete_built_image_doc(self._storage, doc_id)
                if resource.is_zip_hosted:
                    for package_doc in resource.package_docs:
                        await self._assert_lease_healthy(heartbeat)
                        await delete_package_doc(
                            self._storage,
                            tenant_id,
                            str(package_doc.get("server_id") or "").strip(),
                        )
            finally:
                if heartbeat is not None:
                    await heartbeat.stop()
                await self._release_image_deletions(
                    reserved_image_ids, reservation_id=reservation_id
                )
                await self._release_package_deletions(
                    tenant_id=tenant_id,
                    server_ids=reserved_package_ids,
                    reservation_id=reservation_id,
                )
        return {
            "status": "deleted",
            "tenant_id": tenant_id,
            "server_id": server_id,
            "tools_count": len(resource.tool_docs),
            "containers_removed": containers_removed,
            "agent_tool_links_removed": agent_tool_links_removed,
        }

    @staticmethod
    async def _assert_lease_healthy(heartbeat: McpDeleteLeaseHeartbeat) -> None:
        try:
            await heartbeat.assert_healthy()
        except McpDeleteLeaseLost as exc:
            raise McpLifecycleConflict("MCP deletion lost its lease; retry the operation") from exc

    async def _assert_images_not_referenced_elsewhere(
        self, resource: McpServerResource
    ) -> None:
        for image_tag in resource.image_tags:
            if await image_tag_in_use(
                self._storage,
                None,
                image_tag,
                exclude_mcp_server_id=resource.server_id,
                exclude_tenant_id=resource.tenant_id,
                global_scope=True,
            ):
                raise McpLifecycleConflict(
                    "Cannot delete MCP server: "
                    f"image {image_tag} is still referenced by another MCP server"
                )

    async def _reserve_package_deletions(
        self, resource: McpServerResource, reservation_id: str
    ) -> list[str]:
        if not resource.is_zip_hosted:
            return []
        reserved: list[str] = []
        try:
            for package_doc in resource.package_docs:
                package_server_id = str(package_doc.get("server_id") or "").strip()
                if not package_server_id:
                    continue
                if await claim_mcp_package_deletion(
                    self._storage,
                    resource.tenant_id,
                    package_server_id,
                    reservation_id=reservation_id,
                ):
                    reserved.append(package_server_id)
                    continue
                raise McpLifecycleConflict(
                    "Cannot delete MCP server while a ZIP package build is in progress"
                )
        except BaseException:
            await self._release_package_deletions(
                tenant_id=resource.tenant_id,
                server_ids=reserved,
                reservation_id=reservation_id,
            )
            raise
        return reserved

    async def _reserve_image_deletions(
        self, resource: McpServerResource, reservation_id: str
    ) -> list[str]:
        if not resource.is_zip_hosted:
            return []
        reserved: list[str] = []
        try:
            for doc_id in resource.built_image_doc_ids:
                if await claim_mcp_built_image_deletion(
                    self._storage, doc_id, reservation_id=reservation_id
                ):
                    reserved.append(doc_id)
                    continue
                raise McpLifecycleConflict(
                    "Cannot delete MCP server while an image configuration write is in progress"
                )
        except BaseException:
            await self._release_image_deletions(reserved, reservation_id=reservation_id)
            raise
        return reserved

    async def _release_image_deletions(
        self, doc_ids: list[str], *, reservation_id: str
    ) -> None:
        for doc_id in reversed(doc_ids):
            await release_mcp_built_image_deletion(
                self._storage, doc_id, reservation_id=reservation_id
            )

    async def _release_package_deletions(
        self, *, tenant_id: str, server_ids: list[str], reservation_id: str
    ) -> None:
        for package_server_id in reversed(server_ids):
            await release_mcp_package_deletion(
                self._storage,
                tenant_id,
                package_server_id,
                reservation_id=reservation_id,
            )

    async def _remove_runtime(self, *, tenant_id: str, server_id: str) -> None:
        """Close stdio clients before removing Docker resources for a deleted server."""
        if self._mcp_executor is not None and hasattr(
            self._mcp_executor, "on_mcp_server_config_removed"
        ):
            await self._mcp_executor.on_mcp_server_config_removed(
                tenant_id=tenant_id,
                mcp_server_id=server_id,
            )
            return
        logger.warning(
            "[MCP_ZIP_DELETE] tenant=%s server=%s — no mcp_executor; manager fallback",
            tenant_id,
            server_id,
        )
        await self._manager.stop_all_for_tenant_mcp_server(
            tenant_id=tenant_id,
            mcp_server_id=server_id,
        )

    async def _load_resource(self, *, tenant_id: str, server_id: str) -> McpServerResource:
        docs = await get_mcp_tool_configurations(
            self._storage, enabled_only=False, tenant_id=tenant_id
        )
        tools = [
            doc
            for doc in docs
            if isinstance(doc, dict)
            and doc.get("source") == "mcp_server"
            and str(doc.get("tenant_id") or "__root__") == tenant_id
            and str(doc.get("mcp_server") or "").strip() == server_id
        ]
        if not tools:
            raise McpServerNotFound(f"MCP server '{server_id}' was not found")
        runtime = _external_mcp_runtime(tools[0])
        restart_runtimes = [_restart_runtime_from_tool(tool) for tool in tools]
        restart_runtime = (
            restart_runtimes[0]
            if restart_runtimes[0] is not None
            and all(candidate == restart_runtimes[0] for candidate in restart_runtimes)
            else None
        )
        runtime_images = {
            str(candidate.get("image") or "").strip()
            for candidate in (_external_mcp_runtime(tool) for tool in tools)
            if str(candidate.get("image") or "").strip()
        }
        built_images = await list_built_image_docs(self._storage, tenant_id)
        zip_images = [
            doc
            for doc in built_images
            if doc.get("record_type") == "mcp_built_image"
            and (
                str(doc.get("server_id") or "").strip() == server_id
                or str(doc.get("image_tag") or "").strip() in runtime_images
            )
        ]
        image_tags = list(
            dict.fromkeys(
                str(doc.get("image_tag") or "").strip()
                for doc in zip_images
                if str(doc.get("image_tag") or "").strip()
            )
        )
        is_zip_hosted = bool(zip_images) or any(
            str(candidate.get("image") or "").strip() in image_tags
            or str(candidate.get("source") or "").strip().lower() == "zip"
            for candidate in (_external_mcp_runtime(tool) for tool in tools)
        )
        package_server_ids = {
            server_id,
            *(
                str(doc.get("server_id") or "").strip()
                for doc in zip_images
                if str(doc.get("server_id") or "").strip()
            ),
        }
        package_docs = [
            doc
            for doc in await list_package_docs(self._storage, tenant_id)
            if str(doc.get("server_id") or "").strip() in package_server_ids
        ]
        return McpServerResource(
            tenant_id=tenant_id,
            server_id=server_id,
            runtime=runtime,
            restart_runtime=restart_runtime,
            is_zip_hosted=is_zip_hosted,
            tool_docs=tools,
            built_image_doc_ids=[str(doc["_id"]) for doc in zip_images],
            image_tags=image_tags,
            package_docs=package_docs,
        )

    @staticmethod
    def _package_meta(package_doc: dict[str, Any] | None) -> dict[str, Any]:
        return package_doc if isinstance(package_doc, dict) else {}

    @staticmethod
    async def _remove_docker_image(image: str) -> None:
        if not image:
            return
        result = await run_host_cli(["docker", "rmi", image], timeout=120)
        if int(result.get("exit_code", 1)) == 0:
            return
        message = str(result.get("stderr") or result.get("stdout") or "").strip()
        if "No such image" in message:
            return
        raise RuntimeError(f"docker rmi failed for {image}: {message[:500]}")

    async def _remove_docker_containers(self, *, tenant_id: str, server_id: str) -> int:
        """Remove all Docker containers carrying this server's immutable labels.

        This covers only the detached HTTP runtimes created by the external-MCP
        manager. It also catches their durable containers left after an API restart,
        which otherwise prevent ``docker rmi`` from removing a ZIP image. Docker
        stdio runtimes are project-scoped and do not carry these labels; they are
        closed first through ``McpExecutor.on_mcp_server_config_removed``.
        """
        listed = await run_host_cli(
            [
                "docker",
                "ps",
                "-aq",
                "--filter",
                "label=AppFactory.external_mcp=true",
                "--filter",
                f"label=tenant_id={tenant_id}",
                "--filter",
                f"label=server_id={server_id}",
            ],
            timeout=30,
        )
        if int(listed.get("exit_code", 1)) != 0:
            message = str(listed.get("stderr") or listed.get("stdout") or "").strip()
            raise RuntimeError(f"docker ps failed for ZIP MCP containers: {message[:500]}")
        container_ids = [line.strip() for line in str(listed.get("stdout") or "").splitlines() if line.strip()]
        for container_id in container_ids:
            removed = await run_host_cli(["docker", "rm", "-f", container_id], timeout=30)
            if int(removed.get("exit_code", 1)) == 0:
                continue
            message = str(removed.get("stderr") or removed.get("stdout") or "").strip()
            if "No such container" not in message:
                raise RuntimeError(
                    f"docker rm failed for ZIP MCP container {container_id}: {message[:500]}"
                )
        logger.info(
            "[MCP_ZIP_DELETE] tenant=%s server=%s containers_removed=%s",
            tenant_id,
            server_id,
            len(container_ids),
        )
        return len(container_ids)

    async def _remove_agent_tool_references(
        self,
        tool_docs: list[dict[str, Any]],
        *,
        tenant_id: str,
        actor_id: str | None,
    ) -> int:
        """Remove every runtime-valid reference to deleted MCP tools for one tenant."""
        deleted_tools = [doc for doc in tool_docs if isinstance(doc, dict)]
        if not deleted_tools or not hasattr(self._storage, "get_agent_configurations"):
            return 0
        agents = await self._storage.get_agent_configurations(
            enabled_only=False, tenant_id=tenant_id
        )
        tenant_agent_wires = {
            agent_wire_name_from_doc(agent, runtime_tenant_id=tenant_id)
            for agent in agents or []
            if isinstance(agent, dict)
            and str(agent.get("tenant_id") or "").strip() == tenant_id
        }
        removed = 0
        for agent in agents or []:
            if not isinstance(agent, dict):
                continue
            owner_tenant_id = str(agent.get("tenant_id") or "").strip()
            if owner_tenant_id == SYSTEM_TENANT_ID and tenant_id != SYSTEM_TENANT_ID:
                wire_name = agent_wire_name_from_doc(agent, runtime_tenant_id=tenant_id)
                if wire_name in tenant_agent_wires:
                    continue
            elif owner_tenant_id != tenant_id:
                continue
            updated = dict(agent)
            changed = False
            for field in ("allowed_tools", "allowed_mcp_tools"):
                current = list(agent.get(field) or [])
                retained = [
                    value
                    for value in current
                    if not any(
                        tool_doc_matches_allow_ref(
                            tool, str(value or "").strip()
                        )
                        for tool in deleted_tools
                    )
                ]
                removed += len(current) - len(retained)
                if retained != current:
                    updated[field] = retained
                    changed = True
            if changed:
                if owner_tenant_id == SYSTEM_TENANT_ID:
                    updated, _ = await prepare_configuration_cow_update(
                        agent,
                        tenant_id=tenant_id,
                        updates={
                            field: updated[field]
                            for field in ("allowed_tools", "allowed_mcp_tools")
                            if field in updated
                        },
                        resolve_configuration=self._storage.resolve_agent_configuration,
                        get_configuration=self._storage.get_agent_configuration,
                        find_by_name=self._storage.find_agent_configuration_by_name,
                    )
                await self._storage.save_agent_configuration(updated, actor_id=actor_id)
        if removed:
            logger.info(
                "[MCP_ZIP_DELETE] tenant=%s agent_tool_links_removed=%s", tenant_id, removed
            )
        return removed

    async def _lock_for(self, tenant_id: str, server_id: str) -> asyncio.Lock:
        key = (tenant_id, server_id)
        async with _LIFECYCLE_LOCKS_GUARD:
            return _LIFECYCLE_LOCKS.setdefault(key, asyncio.Lock())
