"""Startup recovery for durable ZIP-hosted tenant MCP servers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from storage.tool_doc_storage import get_mcp_tool_configurations
from tools.mcp_package_storage import list_built_image_docs
from tools.mcp_server_lifecycle import McpServerLifecycle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class McpZipRecoveryTarget:
    tenant_id: str
    server_id: str


@dataclass(frozen=True)
class McpZipRecoverySkip:
    tenant_id: str
    server_id: str
    reason: str


@dataclass(frozen=True)
class McpZipRecoveryCollection:
    targets: tuple[McpZipRecoveryTarget, ...]
    skipped: tuple[McpZipRecoverySkip, ...]


def _zip_recovery_skip_reason(runtime: dict[str, Any], image_tags: set[str]) -> str | None:
    if runtime.get("recover_on_startup") is not True:
        return "recovery_disabled"
    if str(runtime.get("runtime_scope") or "").strip().lower() != "tenant":
        return "scope_not_tenant"
    if str(runtime.get("mode") or "").strip().lower() != "streamable-http":
        return "unsupported_mode"
    tenant_id = str(runtime.get("tenant_id") or "").strip()
    server_id = str(runtime.get("server_id") or "").strip()
    image = str(runtime.get("image") or "").strip()
    if not tenant_id or not server_id or not image:
        return "invalid_runtime"
    if image not in image_tags:
        return "image_not_ready"
    return None


def _runtime_is_recoverable(runtime: dict[str, Any], image_tags: set[str]) -> bool:
    return _zip_recovery_skip_reason(runtime, image_tags) is None


async def _collect_zip_mcp_recovery(storage: Any) -> McpZipRecoveryCollection:
    """Collect recoverable ZIP MCPs and server-level skip reasons."""
    tools = await get_mcp_tool_configurations(storage, enabled_only=False, tenant_id=None)
    image_docs = await list_built_image_docs(storage, None)
    ready_images: dict[str, set[str]] = {}
    for doc in image_docs:
        if not isinstance(doc, dict) or doc.get("record_type") != "mcp_built_image":
            continue
        if str(doc.get("status") or "").strip().lower() != "ready":
            continue
        tenant_id = str(doc.get("tenant_id") or "__root__").strip() or "__root__"
        image = str(doc.get("image_tag") or "").strip()
        if image:
            ready_images.setdefault(tenant_id, set()).add(image)

    candidates: dict[tuple[str, str], list[tuple[dict[str, Any], bool]]] = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("source") != "mcp_server":
            continue
        tenant_id = str(tool.get("tenant_id") or "__root__").strip() or "__root__"
        server_id = str(tool.get("mcp_server") or "").strip()
        metadata = tool.get("metadata") if isinstance(tool.get("metadata"), dict) else {}
        runtime = metadata.get("external_mcp") if isinstance(metadata.get("external_mcp"), dict) else {}
        candidate_runtime = {**runtime, "tenant_id": tenant_id, "server_id": server_id}
        image = str(candidate_runtime.get("image") or "").strip()
        if (
            candidate_runtime.get("source") != "zip"
            and image not in ready_images.get(tenant_id, set())
        ):
            continue
        candidates.setdefault((tenant_id, server_id), []).append(
            (candidate_runtime, bool(tool.get("enabled", True)))
        )

    targets: list[McpZipRecoveryTarget] = []
    skipped: list[McpZipRecoverySkip] = []
    for (tenant_id, server_id), runtimes in sorted(candidates.items()):
        enabled_runtimes = [runtime for runtime, enabled in runtimes if enabled]
        if not enabled_runtimes:
            skipped.append(McpZipRecoverySkip(tenant_id, server_id, "disabled"))
            continue
        if any(
            _runtime_is_recoverable(runtime, ready_images.get(tenant_id, set()))
            for runtime in enabled_runtimes
        ):
            targets.append(McpZipRecoveryTarget(tenant_id, server_id))
            continue
        reasons = {
            reason
            for runtime in enabled_runtimes
            if (reason := _zip_recovery_skip_reason(
                runtime, ready_images.get(tenant_id, set())
            ))
        }
        skipped.append(
            McpZipRecoverySkip(
                tenant_id,
                server_id,
                sorted(reasons)[0] if reasons else "invalid_runtime",
            )
        )
    return McpZipRecoveryCollection(tuple(targets), tuple(skipped))


async def collect_recoverable_zip_mcp_servers(storage: Any) -> list[McpZipRecoveryTarget]:
    """Return eligible ZIP MCPs across all tenants, ordered for deterministic startup."""
    return list((await _collect_zip_mcp_recovery(storage)).targets)


async def recover_zip_mcp_servers(
    storage: Any,
    *,
    manager: Any,
    lifecycle_factory: Callable[..., McpServerLifecycle] = McpServerLifecycle,
) -> dict[str, int]:
    """Warm every eligible ZIP MCP and isolate failures to a single server."""
    summary = {"recovered": 0, "skipped": 0, "failed": 0}
    collection = await _collect_zip_mcp_recovery(storage)
    summary["skipped"] = len(collection.skipped)
    for skipped in collection.skipped:
        logger.info(
            "[MCP_ZIP_RECOVERY] tenant=%s server=%s action=warmup status=skipped reason=%s",
            skipped.tenant_id,
            skipped.server_id,
            skipped.reason,
        )
    for target in collection.targets:
        try:
            lifecycle = lifecycle_factory(storage=storage, manager=manager)
            result = await lifecycle.restart(tenant_id=target.tenant_id, server_id=target.server_id)
        except Exception as exc:
            summary["failed"] += 1
            logger.exception(
                "[MCP_ZIP_RECOVERY] tenant=%s server=%s action=warmup status=failed error=%s",
                target.tenant_id,
                target.server_id,
                str(exc)[:500],
            )
            continue
        summary["recovered"] += 1
        logger.info(
            "[MCP_ZIP_RECOVERY] tenant=%s server=%s action=warmup status=ok tools=%s",
            target.tenant_id,
            target.server_id,
            result.get("tools_count", 0) if isinstance(result, dict) else 0,
        )
    logger.info(
        "[MCP_ZIP_RECOVERY] action=summary status=ok recovered=%d skipped=%d failed=%d",
        summary["recovered"],
        summary["skipped"],
        summary["failed"],
    )
    return summary
