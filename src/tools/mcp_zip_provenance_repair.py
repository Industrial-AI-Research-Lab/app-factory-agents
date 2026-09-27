"""Explicit repair of legacy ZIP MCP provenance."""

from __future__ import annotations

import logging
import os
from typing import Any

from tools.mcp_package_storage import ZIP_PROVENANCE

logger = logging.getLogger(__name__)


async def _list_collection_documents(collection: Any) -> list[dict]:
    if isinstance(collection, list):
        return list(collection)
    return await collection.find({}).to_list(length=None)


async def _patch_document(
    collection: Any,
    document_id: str,
    patch: dict[str, Any],
    *,
    expected: dict[str, Any],
    tenant_id: str | None = None,
) -> bool:
    if isinstance(collection, list):
        for document in collection:
            if str(document.get("_id") or "") != document_id or any(
                document.get(key) != value for key, value in expected.items()
            ):
                continue
            if tenant_id is not None and (
                str(document.get("tenant_id") or "__root__").strip() or "__root__"
            ) != tenant_id:
                continue
            for path, value in patch.items():
                target = document
                parts = path.split(".")
                for part in parts[:-1]:
                    child = target.get(part)
                    if not isinstance(child, dict):
                        child = {}
                        target[part] = child
                    target = child
                target[parts[-1]] = value
            return True
        return False
    query = {"_id": document_id, **expected}
    if tenant_id is not None:
        query["tenant_id"] = (
            tenant_id
            if tenant_id != "__root__"
            else {"$in": ["__root__", None]}
        )
    result = await collection.update_one(query, {"$set": patch})
    return bool(getattr(result, "matched_count", 0))


async def repair_mcp_zip_provenance(storage: Any) -> tuple[int, int]:
    """Restore provenance when canonical wizard state proves a tenant ZIP image.

    The caller owns the execution gate. This function is idempotent and honours
    ``MCP_WIZARD_MIGRATE_DRY_RUN=1`` without mutating storage.
    """
    source = getattr(storage, "tool_mcp_configurations", None)
    packages = getattr(storage, "mcp_packages", None)
    images = getattr(storage, "mcp_built_images", None)
    if source is None or packages is None or images is None:
        raise RuntimeError("MCP wizard collections are not initialized")

    dry_run = os.getenv("MCP_WIZARD_MIGRATE_DRY_RUN", "").strip() == "1"
    package_docs = await _list_collection_documents(packages)
    image_docs = await _list_collection_documents(images)
    ready_images: dict[str, set[str]] = {}
    for document in image_docs:
        if (
            not isinstance(document, dict)
            or document.get("record_type") != "mcp_built_image"
            or str(document.get("status") or "").strip().lower() != "ready"
        ):
            continue
        tenant_id = str(document.get("tenant_id") or "__root__").strip() or "__root__"
        image = str(document.get("image_tag") or "").strip()
        if image:
            ready_images.setdefault(tenant_id, set()).add(image)

    repaired_documents = 0
    for collection, documents, record_type in (
        (packages, package_docs, "mcp_package"),
        (images, image_docs, "mcp_built_image"),
    ):
        for document in documents:
            if not isinstance(document, dict) or document.get("record_type") != record_type:
                continue
            document_id = str(document.get("_id") or "").strip()
            if not document_id:
                continue
            patch = {
                key: value
                for key, value in ZIP_PROVENANCE.items()
                if document.get(key) != value
            }
            if not patch:
                continue
            if dry_run or await _patch_document(
                collection,
                document_id,
                patch,
                expected={"record_type": record_type},
            ):
                repaired_documents += 1
            else:
                logger.warning(
                    "[MCP_MIGRATE] mode=repair skipped_changed_document=%s",
                    document_id,
                )

    repaired_tools = 0
    for tool in await _list_collection_documents(source):
        if not isinstance(tool, dict) or tool.get("source") != "mcp_server":
            continue
        tenant_id = str(tool.get("tenant_id") or "__root__").strip() or "__root__"
        metadata = tool.get("metadata") if isinstance(tool.get("metadata"), dict) else {}
        runtime = metadata.get("external_mcp") if isinstance(metadata.get("external_mcp"), dict) else {}
        image = str(runtime.get("image") or "").strip()
        if not image or image not in ready_images.get(tenant_id, set()):
            continue
        document_id = str(tool.get("_id") or "").strip()
        if not document_id:
            continue
        patch = {
            f"metadata.external_mcp.{key}": value
            for key, value in ZIP_PROVENANCE.items()
            if runtime.get(key) != value
        }
        if not patch:
            continue
        if dry_run or await _patch_document(
            source,
            document_id,
            patch,
            expected={"source": "mcp_server"},
            tenant_id=tenant_id,
        ):
            repaired_tools += 1
        else:
            logger.warning(
                "[MCP_MIGRATE] mode=repair skipped_changed_tool=%s",
                document_id,
            )

    logger.warning(
        "[MCP_MIGRATE] mode=repair provenance_documents=%d tool_runtimes=%d dry_run=%s",
        repaired_documents,
        repaired_tools,
        dry_run,
    )
    return repaired_documents, repaired_tools
