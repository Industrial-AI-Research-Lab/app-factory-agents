"""One-shot migration of legacy MCP wizard documents."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from tools.mcp_package_storage import ZIP_PROVENANCE, built_image_doc_id, package_doc_id
from tools.mcp_zip_provenance_repair import repair_mcp_zip_provenance as _repair_provenance

logger = logging.getLogger(__name__)

MIGRATION_MODES = frozenset({"copy", "move", "repair"})


@dataclass(frozen=True)
class MigrationResult:
    packages: int = 0
    images: int = 0
    already_exists: int = 0
    legacy_deleted: int = 0
    skipped: int = 0
    provenance_documents: int = 0
    tool_runtimes_repaired: int = 0


@dataclass(frozen=True)
class McpWizardStorageReadiness:
    """Read-only startup audit of legacy and canonical wizard storage."""

    legacy_packages: int = 0
    legacy_images: int = 0
    canonical_packages: int = 0
    canonical_images: int = 0

    @property
    def legacy_total(self) -> int:
        return self.legacy_packages + self.legacy_images

    @property
    def canonical_total(self) -> int:
        return self.canonical_packages + self.canonical_images


async def _count_documents(collection: Any) -> int:
    if isinstance(collection, list):
        return len(collection)
    return int(await collection.count_documents({}))


async def _count_legacy_wizard_documents(source: Any) -> tuple[int, int]:
    if isinstance(source, list):
        names = [
            {
                str(doc.get("name") or "").strip(),
                str(doc.get("rpc_name") or "").strip(),
            }
            for doc in source
        ]
        return (
            sum("__package__" in document_names for document_names in names),
            sum(
                any(name.startswith("__image__") for name in document_names)
                for document_names in names
            ),
        )
    return (
        int(
            await source.count_documents(
                {"$or": [{"name": "__package__"}, {"rpc_name": "__package__"}]}
            )
        ),
        int(
            await source.count_documents(
                {
                    "$or": [
                        {"name": {"$regex": "^__image__"}},
                        {"rpc_name": {"$regex": "^__image__"}},
                    ]
                }
            )
        ),
    )


async def validate_mcp_wizard_storage_ready(
    storage: Any,
) -> McpWizardStorageReadiness:
    """Fail startup when only ignored legacy wizard state exists.

    The check intentionally compares aggregate presence only. Legacy copy rows may
    be stale after cutover, while canonical storage remains authoritative.
    """
    source = getattr(storage, "tool_mcp_configurations", None)
    packages = getattr(storage, "mcp_packages", None)
    images = getattr(storage, "mcp_built_images", None)
    if source is None or packages is None or images is None:
        raise RuntimeError("MCP wizard collections are not initialized")

    legacy_packages, legacy_images = await _count_legacy_wizard_documents(source)
    result = McpWizardStorageReadiness(
        legacy_packages=legacy_packages,
        legacy_images=legacy_images,
        canonical_packages=await _count_documents(packages),
        canonical_images=await _count_documents(images),
    )
    if result.legacy_total and not result.canonical_total:
        logger.error(
            "[MCP_MIGRATION_CHECK] legacy_packages=%d legacy_images=%d "
            "canonical_packages=%d canonical_images=%d status=blocked - "
            "legacy wizard documents exist but canonical storage is empty",
            result.legacy_packages,
            result.legacy_images,
            result.canonical_packages,
            result.canonical_images,
        )
        raise RuntimeError(
            "MCP wizard storage is not ready: legacy wizard documents exist "
            "but canonical storage is empty"
        )
    logger.info(
        "[MCP_MIGRATION_CHECK] legacy_packages=%d legacy_images=%d "
        "canonical_packages=%d canonical_images=%d status=ready",
        result.legacy_packages,
        result.legacy_images,
        result.canonical_packages,
        result.canonical_images,
    )
    return result


def migration_mode_from_env() -> str | None:
    """Return the explicitly enabled one-shot migration mode, if any."""
    mode = os.getenv("MCP_WIZARD_MIGRATION_MODE", "").strip().lower()
    if not mode:
        return None
    if mode not in MIGRATION_MODES:
        raise ValueError(
            "MCP_WIZARD_MIGRATION_MODE must be one of: "
            + ", ".join(sorted(MIGRATION_MODES))
        )
    return mode


async def _insert_if_absent(destination: Any, document: dict) -> bool:
    """Atomically create canonical wizard state without modifying an existing row."""
    if isinstance(destination, list):
        if any(doc.get("_id") == document["_id"] for doc in destination):
            return False
        destination.append(document)
        return True
    result = await destination.update_one(
        {"_id": document["_id"]},
        {"$setOnInsert": document},
        upsert=True,
    )
    return getattr(result, "upserted_id", None) is not None


async def _delete_legacy_source(source: Any, legacy_id: str) -> bool:
    if isinstance(source, list):
        before = len(source)
        source[:] = [doc for doc in source if doc.get("_id") != legacy_id]
        return len(source) != before
    result = await source.delete_one({"_id": legacy_id})
    return bool(getattr(result, "deleted_count", 0))


async def repair_mcp_zip_provenance(storage: Any) -> MigrationResult:
    """Run the explicit provenance repair and return the regular migration result."""
    repaired_documents, repaired_tools = await _repair_provenance(storage)
    return MigrationResult(
        provenance_documents=repaired_documents,
        tool_runtimes_repaired=repaired_tools,
    )


async def migrate_mcp_wizard_docs(storage: Any, *, mode: str = "copy") -> MigrationResult:
    """Copy legacy wizard rows without allowing legacy state to overwrite canonical rows."""
    selected_mode = str(mode or "").strip().lower()
    if selected_mode not in MIGRATION_MODES:
        raise ValueError(f"Unsupported MCP wizard migration mode: {mode!r}")
    if selected_mode == "repair":
        return await repair_mcp_zip_provenance(storage)
    source = getattr(storage, "tool_mcp_configurations", None)
    packages = getattr(storage, "mcp_packages", None)
    images = getattr(storage, "mcp_built_images", None)
    if source is None or packages is None or images is None:
        raise RuntimeError("MCP wizard collections are not initialized")
    dry_run = os.getenv("MCP_WIZARD_MIGRATE_DRY_RUN", "").strip() == "1"
    package_count = image_count = existing_count = deleted_count = skipped = 0
    docs = (
        await source.find({}).to_list(length=None)
        if not isinstance(source, list)
        else list(source)
    )
    for legacy in docs:
        name = str(legacy.get("name") or "")
        if name != "__package__" and not name.startswith("__image__"):
            continue
        tenant_id = str(legacy.get("tenant_id") or "__root__")
        server_id = str(legacy.get("mcp_server") or "").strip()
        metadata = (
            legacy.get("metadata") if isinstance(legacy.get("metadata"), dict) else {}
        )
        key = "mcp_package" if name == "__package__" else "mcp_built_image"
        payload = metadata.get(key) if isinstance(metadata.get(key), dict) else {}
        if not server_id or not payload:
            skipped += 1
            continue
        if name == "__package__":
            document = {
                **payload,
                "_id": package_doc_id(tenant_id, server_id),
                "tenant_id": tenant_id,
                "server_id": server_id,
                "record_type": "mcp_package",
                **ZIP_PROVENANCE,
            }
            destination = packages
        else:
            upload_id = str(payload.get("upload_id") or "").strip()
            if not upload_id:
                skipped += 1
                continue
            document = {
                **payload,
                "_id": built_image_doc_id(tenant_id, server_id, upload_id),
                "tenant_id": tenant_id,
                "server_id": server_id,
                "record_type": "mcp_built_image",
                **ZIP_PROVENANCE,
            }
            destination = images
        if not dry_run:
            inserted = await _insert_if_absent(destination, document)
            if not inserted:
                existing_count += 1
                logger.warning(
                    "[MCP_MIGRATE] mode=%s existing_destination=%s legacy_id=%s",
                    selected_mode,
                    document["_id"],
                    legacy.get("_id"),
                )
                continue
            if name == "__package__":
                package_count += 1
            else:
                image_count += 1
            if selected_mode == "move" and await _delete_legacy_source(
                source, str(legacy.get("_id") or "")
            ):
                deleted_count += 1
        else:
            if name == "__package__":
                package_count += 1
            else:
                image_count += 1
    logger.info(
        "[MCP_MIGRATE] mode=%s packages=%d images=%d existing=%d legacy_deleted=%d "
        "skipped=%d dry_run=%s",
        selected_mode,
        package_count,
        image_count,
        existing_count,
        deleted_count,
        skipped,
        dry_run,
    )
    return MigrationResult(
        packages=package_count,
        images=image_count,
        already_exists=existing_count,
        legacy_deleted=deleted_count,
        skipped=skipped,
    )


async def prepare_mcp_wizard_storage(storage: Any) -> MigrationResult | None:
    """Apply an explicitly requested migration before validating canonical state."""
    selected_mode = migration_mode_from_env()
    result = (
        await migrate_mcp_wizard_docs(storage, mode=selected_mode)
        if selected_mode
        else None
    )
    await validate_mcp_wizard_storage_ready(storage)
    return result
