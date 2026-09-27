"""Cascade cleanup for MCP ZIP wizard state."""

from __future__ import annotations

import inspect
import logging
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tools.mcp_delete_lease_heartbeat import McpDeleteLeaseHeartbeat, McpDeleteLeaseLost
from tools.mcp_package_storage import (
    claim_mcp_built_image_deletion,
    claim_mcp_package_deletion,
    image_tag_in_use,
    release_mcp_built_image_deletion,
    release_mcp_package_deletion,
)
from storage.mcp_wizard_storage import (
    delete_built_image,
    delete_package,
    get_package,
    list_built_images,
)

logger = logging.getLogger(__name__)

MCP_PACKAGE_TERMINAL_STATUSES = frozenset(
    {"ready", "build_failed", "smoke_failed"}
)


@dataclass(frozen=True)
class CleanupResult:
    package_deleted: bool = False
    images_deleted: int = 0
    extract_cleaned: bool = False
    blocked: bool = False


@dataclass(frozen=True)
class TenantExtractCleanupResult:
    attempted: int = 0
    removed: int = 0
    skipped: int = 0
    failures: int = 0


def _tenant_build_root(tenant_id: str) -> Path:
    from tools.mcp_zip_service import _mcp_build_root

    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(tenant_id or "__root__"))
    return (_mcp_build_root() / normalized).resolve()


def cleanup_tenant_extracts(
    tenant_id: str,
    extract_paths: list[str] | tuple[str, ...] = (),
) -> TenantExtractCleanupResult:
    """Remove only upload trees belonging to one tenant after DB purge."""
    tenant_root = _tenant_build_root(tenant_id)
    candidates: set[Path] = set()
    skipped = 0
    for raw_path in extract_paths:
        path = Path(str(raw_path or "")).resolve()
        try:
            relative = path.relative_to(tenant_root)
        except ValueError:
            skipped += 1
            logger.warning(
                "[MCP_CLEANUP] tenant=%s trigger=tenant_purge skipped_unsafe_path=%s",
                tenant_id,
                path,
            )
            continue
        if len(relative.parts) != 2 or relative.parts[-1] != "src":
            skipped += 1
            logger.warning(
                "[MCP_CLEANUP] tenant=%s trigger=tenant_purge skipped_invalid_path=%s",
                tenant_id,
                path,
            )
            continue
        candidates.add(path.parent)

    if not tenant_root.exists():
        return TenantExtractCleanupResult(skipped=skipped)

    try:
        children = list(tenant_root.iterdir())
    except OSError as exc:
        logger.warning(
            "[MCP_CLEANUP] tenant=%s trigger=tenant_purge root_scan_failed=%s",
            tenant_id,
            exc,
        )
        return TenantExtractCleanupResult(skipped=skipped, failures=1)
    for child in children:
        if child.is_symlink() or not child.is_dir():
            skipped += 1
            continue
        candidates.add(child.resolve())

    attempted = removed = failures = 0
    for upload_dir in sorted(candidates):
        try:
            upload_dir.relative_to(tenant_root)
        except ValueError:
            skipped += 1
            continue
        attempted += 1
        try:
            shutil.rmtree(upload_dir)
        except FileNotFoundError:
            removed += 1
        except OSError as exc:
            failures += 1
            logger.warning(
                "[MCP_CLEANUP] tenant=%s trigger=tenant_purge extract_remove_failed path=%s error=%s",
                tenant_id,
                upload_dir,
                exc,
            )
        else:
            removed += 1
    try:
        if tenant_root.exists() and not any(tenant_root.iterdir()):
            tenant_root.rmdir()
    except OSError as exc:
        logger.warning(
            "[MCP_CLEANUP] tenant=%s trigger=tenant_purge tenant_root_remove_failed path=%s error=%s",
            tenant_id,
            tenant_root,
            exc,
        )
        failures += 1
    logger.info(
        "[MCP_CLEANUP] tenant=%s trigger=tenant_purge attempted=%d removed=%d skipped=%d failures=%d",
        tenant_id,
        attempted,
        removed,
        skipped,
        failures,
    )
    return TenantExtractCleanupResult(attempted, removed, skipped, failures)


async def list_tenant_package_extract_paths(storage: Any, tenant_id: str) -> list[str]:
    """Read package extract paths before tenant Mongo cascade removes them."""
    from storage.mcp_wizard_storage import list_packages

    packages = await list_packages(storage, tenant_id)
    return [
        str(package.get("extract_path"))
        for package in packages
        if str(package.get("extract_path") or "").strip()
    ]


async def _call_cleanup(callback: Callable[[str], Any], path: str) -> None:
    result = callback(path)
    if inspect.isawaitable(result):
        await result


async def cleanup_after_image_delete(
    storage: Any,
    *,
    tenant_id: str,
    server_id: str,
    deleted_upload_id: str,
    extract_cleanup: Callable[[str], Any] | None = None,
    package_deletion_reservation_id: str | None = None,
) -> CleanupResult:
    remaining = await list_built_images(storage, tenant_id, server_id)
    if remaining:
        logger.info(
            "[MCP_CLEANUP] tenant=%s server=%s trigger=image_deleted remaining_images=%d",
            tenant_id,
            server_id,
            len(remaining),
        )
        return CleanupResult()
    package = await get_package(storage, tenant_id, server_id)
    reservation_id = package_deletion_reservation_id or (uuid.uuid4().hex if package else None)
    owns_reservation = bool(package and not package_deletion_reservation_id)
    if package and owns_reservation and not await claim_mcp_package_deletion(
        storage,
        tenant_id,
        server_id,
        reservation_id=reservation_id,
    ):
        logger.warning(
            "[MCP_CLEANUP] tenant=%s server=%s trigger=last_image_deleted blocked=in_flight_or_reserved",
            tenant_id,
            server_id,
        )
        return CleanupResult(blocked=True)
    deleted = False
    cleaned = False
    try:
        deleted = await delete_package(storage, tenant_id, server_id)
        path = str((package or {}).get("extract_path") or "").strip()
        if path and extract_cleanup:
            await _call_cleanup(extract_cleanup, path)
            cleaned = True
    finally:
        if owns_reservation and reservation_id and not deleted:
            await release_mcp_package_deletion(
                storage,
                tenant_id,
                server_id,
                reservation_id=reservation_id,
            )
    logger.info(
        "[MCP_CLEANUP] tenant=%s server=%s trigger=last_image_deleted package=%s extract=%s",
        tenant_id,
        server_id,
        "yes" if deleted else "no",
        "yes" if cleaned else "no",
    )
    return CleanupResult(package_deleted=deleted, extract_cleaned=cleaned)


async def cleanup_server_wizard_state(
    storage: Any,
    *,
    tenant_id: str,
    server_id: str,
    rmi_images: bool = True,
    extract_cleanup: Callable[[str], Any] | None = None,
    docker_remove: Callable[[str], Any] | None = None,
) -> CleanupResult:
    """Remove all wizard state for a server after its last user tool is gone."""
    if extract_cleanup is None:
        from tools.mcp_zip_service import cleanup_prior_upload_extract

        extract_cleanup = cleanup_prior_upload_extract
    if docker_remove is None:

        async def _docker_remove(tag: str) -> None:
            from sandbox.host_cli import run_host_cli

            result = await run_host_cli(["docker", "rmi", tag], timeout=120)
            code = int(result.get("exit_code", 1))
            error = (result.get("stderr") or result.get("stdout") or "").strip()
            if code and "No such image" not in error:
                raise RuntimeError(error[:500] or f"docker rmi exited {code}")

        docker_remove = _docker_remove
    package = await get_package(storage, tenant_id, server_id)
    reservation_id = uuid.uuid4().hex if package else None
    if package and not await claim_mcp_package_deletion(
        storage,
        tenant_id,
        server_id,
        reservation_id=reservation_id,
    ):
        logger.warning(
            "[MCP_CLEANUP] tenant=%s server=%s trigger=last_tool_deleted blocked=in_flight_or_reserved",
            tenant_id,
            server_id,
        )
        return CleanupResult(blocked=True)
    deleted_package = False
    image_reservation_id = uuid.uuid4().hex
    reserved_image_ids: list[str] = []
    deleted_image_ids: set[str] = set()
    heartbeat: McpDeleteLeaseHeartbeat | None = None
    try:
        images = await list_built_images(storage, tenant_id, server_id)
        tags: list[str] = []
        for image in images:
            image_id = str(image.get("_id") or "").strip()
            if not image_id or not await claim_mcp_built_image_deletion(
                storage, image_id, reservation_id=image_reservation_id
            ):
                for reserved_id in reversed(reserved_image_ids):
                    await release_mcp_built_image_deletion(
                        storage, reserved_id, reservation_id=image_reservation_id
                    )
                reserved_image_ids.clear()
                logger.warning(
                    "[MCP_CLEANUP] tenant=%s server=%s trigger=last_tool_deleted "
                    "blocked=image_configuration_write",
                    tenant_id,
                    server_id,
                )
                return CleanupResult(blocked=True)
            reserved_image_ids.append(image_id)
        heartbeat = McpDeleteLeaseHeartbeat(
            storage=storage,
            tenant_id=tenant_id,
            package_server_ids=[server_id] if reservation_id else [],
            package_reservation_id=reservation_id,
            image_doc_ids=reserved_image_ids,
            image_reservation_id=image_reservation_id,
        )
        await heartbeat.start()
        await heartbeat.assert_healthy()
        for image in images:
            tag = str(image.get("image_tag") or "").strip()
            if tag and await image_tag_in_use(storage, tenant_id, tag, global_scope=True):
                logger.warning(
                    "[MCP_CLEANUP] tenant=%s server=%s trigger=last_tool_deleted "
                    "blocked=image_still_referenced",
                    tenant_id,
                    server_id,
                )
                return CleanupResult(blocked=True)
        for image in images:
            await heartbeat.assert_healthy()
            image_id = str(image.get("_id") or "").strip()
            tag = str(image.get("image_tag") or "").strip()
            if tag:
                tags.append(tag)
            if await delete_built_image(storage, image_id):
                deleted_image_ids.add(image_id)
        await heartbeat.assert_healthy()
        deleted_package = await delete_package(storage, tenant_id, server_id)
        path = str((package or {}).get("extract_path") or "").strip()
        cleaned = False
        if path and extract_cleanup:
            await heartbeat.assert_healthy()
            await _call_cleanup(extract_cleanup, path)
            cleaned = True
        if rmi_images and docker_remove:
            for tag in tags:
                await heartbeat.assert_healthy()
                if await image_tag_in_use(storage, tenant_id, tag, global_scope=True):
                    continue
                try:
                    await _call_cleanup(docker_remove, tag)
                except Exception as exc:
                    logger.warning(
                        "[MCP_CLEANUP] tenant=%s server=%s docker_rmi_failed tag=%s error=%s",
                        tenant_id,
                        server_id,
                        tag,
                        exc,
                    )
    except McpDeleteLeaseLost:
        logger.warning(
            "[MCP_CLEANUP] tenant=%s server=%s trigger=last_tool_deleted blocked=lease_lost",
            tenant_id,
            server_id,
        )
        return CleanupResult(blocked=True)
    finally:
        if heartbeat is not None:
            await heartbeat.stop()
        for image_id in reversed(reserved_image_ids):
            if image_id not in deleted_image_ids:
                await release_mcp_built_image_deletion(
                    storage, image_id, reservation_id=image_reservation_id
                )
        if reservation_id and not deleted_package:
            await release_mcp_package_deletion(
                storage,
                tenant_id,
                server_id,
                reservation_id=reservation_id,
            )
    logger.info(
        "[MCP_CLEANUP] tenant=%s server=%s trigger=last_tool_deleted images=%d package=%s extract=%s",
        tenant_id,
        server_id,
        len(images),
        "yes" if deleted_package else "no",
        "yes" if cleaned else "no",
    )
    return CleanupResult(
        package_deleted=deleted_package,
        images_deleted=len(images),
        extract_cleaned=cleaned,
    )


async def mark_server_wizard_cleanup_pending(
    storage: Any,
    *,
    tenant_id: str,
    server_id: str,
) -> bool:
    """Remember that server wizard cleanup must run after an in-flight build ends."""
    from storage.mcp_wizard_storage import patch_package

    if not await patch_package(
        storage,
        tenant_id,
        server_id,
        {"cleanup_pending": True},
    ):
        return False
    logger.info(
        "[MCP_CLEANUP] tenant=%s server=%s cleanup_pending=true",
        tenant_id,
        server_id,
    )
    return True


async def cleanup_pending_server_wizard_state(
    storage: Any,
    *,
    tenant_id: str,
    server_id: str,
    extract_cleanup: Callable[[str], Any] | None = None,
    docker_remove: Callable[[str], Any] | None = None,
) -> CleanupResult:
    """Complete deferred cleanup once a package build reaches a terminal state."""
    package = await get_package(storage, tenant_id, server_id)
    if not package or not package.get("cleanup_pending"):
        return CleanupResult()
    from storage.tool_doc_storage import count_mcp_tools_on_server

    if await count_mcp_tools_on_server(storage, tenant_id, server_id) > 0:
        from storage.mcp_wizard_storage import patch_package

        await patch_package(
            storage,
            tenant_id,
            server_id,
            {"cleanup_pending": False},
        )
        logger.info(
            "[MCP_CLEANUP] tenant=%s server=%s deferred cleanup cancelled=tools_present",
            tenant_id,
            server_id,
        )
        return CleanupResult()
    result = await cleanup_server_wizard_state(
        storage,
        tenant_id=tenant_id,
        server_id=server_id,
        extract_cleanup=extract_cleanup,
        docker_remove=docker_remove,
    )
    if result.blocked:
        logger.info(
            "[MCP_CLEANUP] tenant=%s server=%s deferred cleanup still_blocked=true",
            tenant_id,
            server_id,
        )
    return result


async def flush_pending_server_wizard_cleanup(
    storage: Any,
    *,
    tenant_id: str,
    server_id: str,
    terminal_status: str,
    trigger: str,
) -> CleanupResult:
    """Run deferred cleanup only after a successful terminal package transition."""
    status = str(terminal_status or "").strip()
    if status not in MCP_PACKAGE_TERMINAL_STATUSES:
        return CleanupResult()
    result = await cleanup_pending_server_wizard_state(
        storage,
        tenant_id=tenant_id,
        server_id=server_id,
    )
    logger.info(
        "[MCP_CLEANUP] tenant=%s server=%s trigger=%s pending_flush package=%s images=%d "
        "extract=%s blocked=%s",
        tenant_id,
        server_id,
        trigger,
        "yes" if result.package_deleted else "no",
        result.images_deleted,
        "yes" if result.extract_cleaned else "no",
        result.blocked,
    )
    return result
