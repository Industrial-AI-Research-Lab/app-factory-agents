"""Domain facade for MCP ZIP wizard package and image records."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, List

from tools.mcp_internal_docs import MCP_IMAGE_NAME_PREFIX, MCP_PACKAGE_NAME
from tools.mcp_zip_build_policy import PACKAGE_BUILD_IN_FLIGHT_STATUSES
from tools.mcp_tool_ids import (
    McpSegmentIdError,
    mcp_docker_tenant_slug,
    mcp_tool_document_id,
    validate_mcp_image_reference,
    validate_mcp_segment_id,
)

logger = logging.getLogger(__name__)

ZIP_PROVENANCE = {"source": "zip", "recover_on_startup": True}
MCP_PACKAGE_DELETE_RESERVATION_FIELD = "server_delete_reservation_id"
MCP_BUILT_IMAGE_DELETE_RESERVATION_FIELD = "image_delete_reservation_id"
MCP_BUILT_IMAGE_REFERENCE_WRITE_FIELD = "image_reference_write_ids"
MCP_PACKAGE_DELETE_RESERVATION_EXPIRES_AT_FIELD = "server_delete_reservation_expires_at"
MCP_BUILT_IMAGE_DELETE_RESERVATION_EXPIRES_AT_FIELD = "image_delete_reservation_expires_at"
MCP_BUILT_IMAGE_REFERENCE_WRITE_EXPIRES_AT_FIELD = "image_reference_write_expires_at"
MCP_LEASE_OWNER_EPOCH_FIELD = "mcp_lease_worker_epoch"
MCP_LEASE_DURATION = timedelta(minutes=5)
MCP_LEASE_WORKER_EPOCH = uuid.uuid4().hex
MCP_BUILT_IMAGE_TAG_PREFIX = "AppFactory-mcp/"

WORKER_RESTART_BUILD_ERROR = (
    "Build job was lost after worker restart; re-run analyze and build."
)
WORKER_RESTART_SMOKE_ERROR = (
    "Smoke test was lost after worker restart; the image may still exist on the host — "
    "re-run analyze and build."
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lease_expires_at_iso() -> str:
    return (datetime.now(timezone.utc) + MCP_LEASE_DURATION).isoformat()


def _lease_is_expired(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        expires_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at <= datetime.now(timezone.utc)


def mcp_lease_worker_epoch() -> str:
    """Return this API process epoch for MCP lease ownership."""
    return MCP_LEASE_WORKER_EPOCH


async def heartbeat_mcp_lease_worker(
    storage: Any, *, worker_epoch: str | None = None
) -> None:
    """Record that an API worker is still able to own MCP deletion leases."""
    epoch = str(worker_epoch or mcp_lease_worker_epoch()).strip()
    if not epoch:
        return
    heartbeat = getattr(storage, "heartbeat_mcp_lease_worker", None)
    if callable(heartbeat):
        await heartbeat(epoch, _lease_expires_at_iso())
        return
    workers = getattr(storage, "_mcp_lease_workers", None)
    if workers is None:
        workers = {}
        setattr(storage, "_mcp_lease_workers", workers)
    workers[epoch] = _lease_expires_at_iso()


async def active_mcp_lease_worker_epochs(storage: Any) -> set[str]:
    """Return worker epochs with a current heartbeat."""
    lister = getattr(storage, "active_mcp_lease_worker_epochs", None)
    if callable(lister):
        return {str(epoch) for epoch in await lister() if str(epoch).strip()}
    workers = getattr(storage, "_mcp_lease_workers", {})
    return {
        str(epoch)
        for epoch, expires_at in workers.items()
        if str(epoch).strip() and not _lease_is_expired(expires_at)
    }


def _package_meta_from_doc(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in doc.items()
        if key not in {"_id", "tenant_id", "server_id", "record_type"}
    }


def package_doc_id(tenant_id: str, server_id: str) -> str:
    return mcp_tool_document_id(tenant_id, server_id, MCP_PACKAGE_NAME)


def built_image_doc_id(tenant_id: str, server_id: str, upload_id: str) -> str:
    validate_mcp_segment_id(server_id, "mcp_server")
    return mcp_tool_document_id(tenant_id, server_id, built_image_name(upload_id))


def built_image_name(upload_id: str) -> str:
    return f"{MCP_IMAGE_NAME_PREFIX}{upload_id.replace('-', '')[:12]}"


async def get_package_doc(storage: Any, tenant_id: str, server_id: str) -> dict | None:
    from storage.mcp_wizard_storage import get_package

    return await get_package(storage, tenant_id, server_id)


async def list_package_docs(storage: Any, tenant_id: str | None) -> List[dict]:
    from storage.mcp_wizard_storage import list_packages

    return await list_packages(storage, tenant_id)


async def delete_package_doc(storage: Any, tenant_id: str, server_id: str) -> bool:
    from storage.mcp_wizard_storage import delete_package

    return await delete_package(storage, tenant_id, server_id)


async def save_package_doc(
    storage: Any,
    tenant_id: str,
    server_id: str,
    pkg_meta: dict,
    *,
    actor_id: str | None = None,
) -> dict:
    from storage.mcp_wizard_storage import save_package

    doc = {
        "_id": package_doc_id(tenant_id, server_id),
        "tenant_id": tenant_id,
        "server_id": server_id,
        "record_type": "mcp_package",
        **dict(pkg_meta),
        **ZIP_PROVENANCE,
    }
    now = _utc_now_iso()
    doc.setdefault("created_at", now)
    doc["updated_at"] = now
    if actor_id:
        doc["updated_by"] = actor_id
        doc.setdefault("created_by", actor_id)
    await save_package(storage, doc)
    return doc


async def get_package_meta(storage: Any, tenant_id: str, server_id: str) -> dict | None:
    doc = await get_package_doc(storage, tenant_id, server_id)
    if not doc:
        return None
    return {
        key: value
        for key, value in doc.items()
        if key not in {"_id", "tenant_id", "server_id", "record_type"}
    }


async def update_package_meta(
    storage: Any,
    tenant_id: str,
    server_id: str,
    patch: dict,
    *,
    actor_id: str | None = None,
) -> dict:
    existing = await get_package_doc(storage, tenant_id, server_id)
    current = await get_package_meta(storage, tenant_id, server_id) or {}
    merged = {**current, **patch}
    if existing and existing.get("created_at"):
        merged["created_at"] = existing["created_at"]
    await save_package_doc(storage, tenant_id, server_id, merged, actor_id=actor_id)
    return merged


async def claim_mcp_package_build_job(
    storage: Any,
    tenant_id: str,
    server_id: str,
    upload_id: str,
    claim: dict,
    *,
    force_rebuild: bool = False,
    actor_id: str | None = None,
) -> bool:
    """Atomically move package meta to ``building`` if no other build/smoke job is in flight.

    Returns True only when this caller won the claim (safe to ``create_task``).
    """
    uid = str(upload_id or "").strip()
    if not uid or not isinstance(claim, dict):
        return False
    doc_id = package_doc_id(tenant_id, server_id)
    if hasattr(storage, "claim_mcp_package_build_job"):
        return bool(
            await storage.claim_mcp_package_build_job(
                doc_id,
                uid,
                claim,
                force_rebuild=force_rebuild,
                actor_id=actor_id,
            )
        )
    current = await get_package_meta(storage, tenant_id, server_id) or {}
    if str(current.get("upload_id") or "") != uid:
        return False
    if current.get(MCP_PACKAGE_DELETE_RESERVATION_FIELD):
        return False
    st = str(current.get("status") or "").strip()
    if st in PACKAGE_BUILD_IN_FLIGHT_STATUSES:
        return False
    if force_rebuild:
        if st != "ready":
            return False
    elif st == "ready":
        return False
    merged = {**current, **claim}
    await save_package_doc(storage, tenant_id, server_id, merged, actor_id=actor_id)
    return True


async def claim_mcp_package_deletion(
    storage: Any,
    tenant_id: str,
    server_id: str,
    *,
    reservation_id: str,
) -> bool:
    """Reserve a non-building package so build and delete have one CAS winner."""
    token = str(reservation_id or "").strip()
    if not token:
        return False
    doc_id = package_doc_id(tenant_id, server_id)
    if hasattr(storage, "claim_mcp_package_deletion"):
        return bool(await storage.claim_mcp_package_deletion(doc_id, token))
    current = await get_package_meta(storage, tenant_id, server_id) or {}
    if not current:
        return False
    if current.get(MCP_PACKAGE_DELETE_RESERVATION_FIELD):
        return False
    if str(current.get("status") or "").strip() in PACKAGE_BUILD_IN_FLIGHT_STATUSES:
        return False
    current[MCP_PACKAGE_DELETE_RESERVATION_FIELD] = token
    current[MCP_PACKAGE_DELETE_RESERVATION_EXPIRES_AT_FIELD] = _lease_expires_at_iso()
    current[MCP_LEASE_OWNER_EPOCH_FIELD] = mcp_lease_worker_epoch()
    await save_package_doc(storage, tenant_id, server_id, current)
    return True


async def release_mcp_package_deletion(
    storage: Any,
    tenant_id: str,
    server_id: str,
    *,
    reservation_id: str,
) -> bool:
    """Release a reservation after lifecycle delete fails before package removal."""
    token = str(reservation_id or "").strip()
    if not token:
        return False
    doc_id = package_doc_id(tenant_id, server_id)
    if hasattr(storage, "release_mcp_package_deletion"):
        return bool(await storage.release_mcp_package_deletion(doc_id, token))
    current = await get_package_meta(storage, tenant_id, server_id) or {}
    if current.get(MCP_PACKAGE_DELETE_RESERVATION_FIELD) != token:
        return False
    current.pop(MCP_PACKAGE_DELETE_RESERVATION_FIELD, None)
    current.pop(MCP_PACKAGE_DELETE_RESERVATION_EXPIRES_AT_FIELD, None)
    current.pop(MCP_LEASE_OWNER_EPOCH_FIELD, None)
    await save_package_doc(storage, tenant_id, server_id, current)
    return True


async def renew_mcp_package_deletion(
    storage: Any,
    tenant_id: str,
    server_id: str,
    *,
    reservation_id: str,
) -> bool:
    """Extend a package-deletion lease held by the current operation only."""
    token = str(reservation_id or "").strip()
    if not token:
        return False
    doc_id = package_doc_id(tenant_id, server_id)
    if hasattr(storage, "renew_mcp_package_deletion"):
        return bool(await storage.renew_mcp_package_deletion(doc_id, token))
    current = await get_package_meta(storage, tenant_id, server_id) or {}
    if current.get(MCP_PACKAGE_DELETE_RESERVATION_FIELD) != token:
        return False
    current[MCP_PACKAGE_DELETE_RESERVATION_EXPIRES_AT_FIELD] = _lease_expires_at_iso()
    await save_package_doc(storage, tenant_id, server_id, current)
    return True


async def claim_mcp_built_image_deletion(
    storage: Any, doc_id: str, *, reservation_id: str
) -> bool:
    """Reserve a built image only while no configuration write references it."""
    token = str(reservation_id or "").strip()
    if not doc_id or not token:
        return False
    if hasattr(storage, "claim_mcp_built_image_deletion"):
        return bool(await storage.claim_mcp_built_image_deletion(doc_id, token))
    current = await get_built_image_doc(storage, doc_id)
    if (
        not current
        or current.get(MCP_BUILT_IMAGE_DELETE_RESERVATION_FIELD)
        or current.get(MCP_BUILT_IMAGE_REFERENCE_WRITE_FIELD)
    ):
        return False
    from storage.mcp_wizard_storage import save_built_image

    current[MCP_BUILT_IMAGE_DELETE_RESERVATION_FIELD] = token
    current[MCP_BUILT_IMAGE_DELETE_RESERVATION_EXPIRES_AT_FIELD] = _lease_expires_at_iso()
    current[MCP_LEASE_OWNER_EPOCH_FIELD] = mcp_lease_worker_epoch()
    await save_built_image(storage, current)
    return True


async def release_mcp_built_image_deletion(
    storage: Any, doc_id: str, *, reservation_id: str
) -> bool:
    """Release only the image deletion reservation owned by this operation."""
    token = str(reservation_id or "").strip()
    if not doc_id or not token:
        return False
    if hasattr(storage, "release_mcp_built_image_deletion"):
        return bool(await storage.release_mcp_built_image_deletion(doc_id, token))
    current = await get_built_image_doc(storage, doc_id)
    if current is None or current.get(MCP_BUILT_IMAGE_DELETE_RESERVATION_FIELD) != token:
        return False
    from storage.mcp_wizard_storage import save_built_image

    current.pop(MCP_BUILT_IMAGE_DELETE_RESERVATION_FIELD, None)
    current.pop(MCP_BUILT_IMAGE_DELETE_RESERVATION_EXPIRES_AT_FIELD, None)
    current.pop(MCP_LEASE_OWNER_EPOCH_FIELD, None)
    await save_built_image(storage, current)
    return True


async def renew_mcp_built_image_deletion(
    storage: Any, doc_id: str, *, reservation_id: str
) -> bool:
    """Extend an image-deletion lease held by the current operation only."""
    token = str(reservation_id or "").strip()
    if not doc_id or not token:
        return False
    if hasattr(storage, "renew_mcp_built_image_deletion"):
        return bool(await storage.renew_mcp_built_image_deletion(doc_id, token))
    current = await get_built_image_doc(storage, doc_id)
    if current is None or current.get(MCP_BUILT_IMAGE_DELETE_RESERVATION_FIELD) != token:
        return False
    from storage.mcp_wizard_storage import save_built_image

    current[MCP_BUILT_IMAGE_DELETE_RESERVATION_EXPIRES_AT_FIELD] = _lease_expires_at_iso()
    await save_built_image(storage, current)
    return True


async def claim_mcp_built_image_reference_write(
    storage: Any, doc_id: str, *, reservation_id: str
) -> bool:
    """Hold a built image against deletion while a configuration write commits."""
    token = str(reservation_id or "").strip()
    if not doc_id or not token:
        return False
    if hasattr(storage, "claim_mcp_built_image_reference_write"):
        return bool(await storage.claim_mcp_built_image_reference_write(doc_id, token))
    current = await get_built_image_doc(storage, doc_id)
    if (
        not current
        or str(current.get("status") or "").strip() != "ready"
        or current.get(MCP_BUILT_IMAGE_DELETE_RESERVATION_FIELD)
    ):
        return False
    from storage.mcp_wizard_storage import save_built_image

    tokens = list(current.get(MCP_BUILT_IMAGE_REFERENCE_WRITE_FIELD) or [])
    if token not in tokens:
        tokens.append(token)
    current[MCP_BUILT_IMAGE_REFERENCE_WRITE_FIELD] = tokens
    current[MCP_BUILT_IMAGE_REFERENCE_WRITE_EXPIRES_AT_FIELD] = _lease_expires_at_iso()
    await save_built_image(storage, current)
    return True


class McpBuiltImageLeaseConflict(RuntimeError):
    """A managed MCP image cannot accept a new configuration reference."""


async def claim_mcp_built_image_reference_write_for_tag(
    storage: Any,
    tenant_id: str,
    image_tag: str,
    *,
    reservation_id: str,
) -> str | None:
    """Lease a managed image while one MCP configuration is saved.

    ``None`` means the image is an unmanaged remote image. A managed namespace
    tag must have a ready built-image document and cannot be deleting.
    """
    tag = str(image_tag or "").strip()
    if not tag.startswith(MCP_BUILT_IMAGE_TAG_PREFIX):
        return None
    matches = [
        doc
        for doc in await list_built_image_docs(storage, None)
        if isinstance(doc, dict)
        and doc.get("record_type") == "mcp_built_image"
        and str(doc.get("image_tag") or "").strip() == tag
    ]
    if len(matches) != 1:
        raise McpBuiltImageLeaseConflict("Managed MCP image is unavailable or being deleted")
    doc_id = str(matches[0].get("_id") or "").strip()
    if not await claim_mcp_built_image_reference_write(
        storage, doc_id, reservation_id=reservation_id
    ):
        raise McpBuiltImageLeaseConflict("Managed MCP image is being deleted or is not ready")
    return doc_id


async def release_mcp_built_image_reference_write(
    storage: Any, doc_id: str, *, reservation_id: str
) -> bool:
    """Release a configuration-write image lease without affecting other writers."""
    token = str(reservation_id or "").strip()
    if not doc_id or not token:
        return False
    if hasattr(storage, "release_mcp_built_image_reference_write"):
        return bool(await storage.release_mcp_built_image_reference_write(doc_id, token))
    current = await get_built_image_doc(storage, doc_id)
    if current is None:
        return False
    tokens = [
        value
        for value in current.get(MCP_BUILT_IMAGE_REFERENCE_WRITE_FIELD) or []
        if value != token
    ]
    if len(tokens) == len(current.get(MCP_BUILT_IMAGE_REFERENCE_WRITE_FIELD) or []):
        return False
    from storage.mcp_wizard_storage import save_built_image

    if tokens:
        current[MCP_BUILT_IMAGE_REFERENCE_WRITE_FIELD] = tokens
    else:
        current.pop(MCP_BUILT_IMAGE_REFERENCE_WRITE_FIELD, None)
        current.pop(MCP_BUILT_IMAGE_REFERENCE_WRITE_EXPIRES_AT_FIELD, None)
    await save_built_image(storage, current)
    return True


async def update_package_meta_if_job(
    storage: Any,
    tenant_id: str,
    server_id: str,
    job_id: str,
    patch: dict,
    *,
    actor_id: str | None = None,
) -> bool:
    """Merge ``patch`` into package meta only when ``job_id`` still matches (atomic when supported)."""
    jid = str(job_id or "").strip()
    if not jid:
        return False
    doc_id = package_doc_id(tenant_id, server_id)
    if hasattr(storage, "update_mcp_package_meta_if_job"):
        return bool(
            await storage.update_mcp_package_meta_if_job(
                doc_id,
                jid,
                patch,
                actor_id=actor_id,
            )
        )
    current = await get_package_meta(storage, tenant_id, server_id) or {}
    if str(current.get("job_id") or "") != jid:
        return False
    await update_package_meta(storage, tenant_id, server_id, patch, actor_id=actor_id)
    return True


async def reconcile_orphaned_mcp_package_jobs(storage: Any) -> int:
    """Mark in-flight package rows failed on startup (background tasks do not survive restart)."""
    from storage.mcp_wizard_storage import list_packages

    docs = await list_packages(storage)
    reconciled = 0
    now = _utc_now_iso()
    for doc in docs or []:
        pkg = doc
        st = str(doc.get("status") or "").strip()
        if st not in PACKAGE_BUILD_IN_FLIGHT_STATUSES:
            continue
        tenant_id = str(doc.get("tenant_id") or "__root__")
        server_id = str(doc.get("server_id") or "").strip()
        if not server_id:
            continue
        prev_job = str(pkg.get("job_id") or "")
        if st == "building":
            terminal_status = "build_failed"
            phase = "build"
            error = WORKER_RESTART_BUILD_ERROR
        else:
            terminal_status = "smoke_failed"
            phase = "smoke"
            error = WORKER_RESTART_SMOKE_ERROR
        patch = {
            "status": terminal_status,
            "phase": phase,
            "error": error,
            "reconciled_at": now,
            "job_id": None,
        }
        if prev_job:
            updated = await update_package_meta_if_job(
                storage,
                tenant_id,
                server_id,
                prev_job,
                patch,
            )
        else:
            await update_package_meta(storage, tenant_id, server_id, patch)
            updated = True
        if not updated:
            logger.warning(
                "[MCP_PACKAGE] skip stale job reconcile tenant=%s server=%s job=%s",
                tenant_id,
                server_id,
                prev_job,
            )
            continue
        from tools.mcp_wizard_cleanup import flush_pending_server_wizard_cleanup

        await flush_pending_server_wizard_cleanup(
            storage,
            tenant_id=tenant_id,
            server_id=server_id,
            terminal_status=terminal_status,
            trigger="restart_reconcile",
        )
        reconciled += 1
        logger.warning(
            "[MCP_PACKAGE] reconciled stale in-flight package tenant=%s server=%s "
            "was=%s job=%s",
            tenant_id,
            server_id,
            st,
            prev_job,
        )
    return reconciled


async def reconcile_expired_mcp_leases(storage: Any) -> dict[str, int]:
    """Release expired leases only after their owning worker stopped heartbeating."""
    active_epochs = await active_mcp_lease_worker_epochs(storage)
    reconciler = getattr(storage, "reconcile_expired_mcp_leases", None)
    if callable(reconciler):
        result = await reconciler(active_epochs)
        return dict(result) if isinstance(result, dict) else {}

    released = {"package_deletions": 0, "image_deletions": 0, "image_writes": 0}
    for package in await list_package_docs(storage, None):
        if not package.get(MCP_PACKAGE_DELETE_RESERVATION_FIELD):
            continue
        expires_at = package.get(MCP_PACKAGE_DELETE_RESERVATION_EXPIRES_AT_FIELD)
        owner_epoch = str(package.get(MCP_LEASE_OWNER_EPOCH_FIELD) or "").strip()
        if (expires_at and not _lease_is_expired(expires_at)) or owner_epoch in active_epochs:
            continue
        package.pop(MCP_PACKAGE_DELETE_RESERVATION_FIELD, None)
        package.pop(MCP_PACKAGE_DELETE_RESERVATION_EXPIRES_AT_FIELD, None)
        package.pop(MCP_LEASE_OWNER_EPOCH_FIELD, None)
        await save_package_doc(
            storage,
            str(package.get("tenant_id") or "__root__"),
            str(package.get("server_id") or ""),
            _package_meta_from_doc(package),
        )
        released["package_deletions"] += 1
    for image in await list_built_image_docs(storage, None):
        changed = False
        if image.get(MCP_BUILT_IMAGE_DELETE_RESERVATION_FIELD):
            expires_at = image.get(MCP_BUILT_IMAGE_DELETE_RESERVATION_EXPIRES_AT_FIELD)
            owner_epoch = str(image.get(MCP_LEASE_OWNER_EPOCH_FIELD) or "").strip()
            if (not expires_at or _lease_is_expired(expires_at)) and owner_epoch not in active_epochs:
                image.pop(MCP_BUILT_IMAGE_DELETE_RESERVATION_FIELD, None)
                image.pop(MCP_BUILT_IMAGE_DELETE_RESERVATION_EXPIRES_AT_FIELD, None)
                image.pop(MCP_LEASE_OWNER_EPOCH_FIELD, None)
                released["image_deletions"] += 1
                changed = True
        if image.get(MCP_BUILT_IMAGE_REFERENCE_WRITE_FIELD):
            expires_at = image.get(MCP_BUILT_IMAGE_REFERENCE_WRITE_EXPIRES_AT_FIELD)
            owner_epoch = str(image.get(MCP_LEASE_OWNER_EPOCH_FIELD) or "").strip()
            if (not expires_at or _lease_is_expired(expires_at)) and owner_epoch not in active_epochs:
                image.pop(MCP_BUILT_IMAGE_REFERENCE_WRITE_FIELD, None)
                image.pop(MCP_BUILT_IMAGE_REFERENCE_WRITE_EXPIRES_AT_FIELD, None)
                image.pop(MCP_LEASE_OWNER_EPOCH_FIELD, None)
                released["image_writes"] += 1
                changed = True
        if changed:
            from storage.mcp_wizard_storage import save_built_image

            await save_built_image(storage, image)
    return released


async def save_built_image_doc(
    storage: Any,
    tenant_id: str,
    server_id: str,
    upload_id: str,
    image_meta: dict,
    *,
    actor_id: str | None = None,
) -> dict:
    from storage.mcp_wizard_storage import save_built_image

    doc = {
        "_id": built_image_doc_id(tenant_id, server_id, upload_id),
        "tenant_id": tenant_id,
        "server_id": server_id,
        "record_type": "mcp_built_image",
        **dict(image_meta),
        **ZIP_PROVENANCE,
    }
    doc.setdefault("created_at", _utc_now_iso())
    await save_built_image(storage, doc)
    return doc


async def list_built_image_docs(
    storage: Any,
    tenant_id: str | None,
) -> List[dict]:
    from storage.mcp_wizard_storage import list_built_images

    return await list_built_images(storage, tenant_id)


async def derive_zip_runtime_provenance(
    storage: Any,
    *,
    tenant_id: str,
    server_id: str,
    runtime: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return runtime with ZIP provenance derived only from a ready built image."""
    derived = dict(runtime) if isinstance(runtime, dict) else {}
    derived.pop("source", None)
    derived.pop("recover_on_startup", None)
    image = str(derived.get("image") or "").strip()
    if not image:
        return derived
    for doc in await list_built_image_docs(storage, tenant_id):
        if (
            isinstance(doc, dict)
            and doc.get("record_type") == "mcp_built_image"
            and str(doc.get("status") or "").strip().lower() == "ready"
            and str(doc.get("image_tag") or "").strip() == image
        ):
            return {**derived, **ZIP_PROVENANCE}
    return derived


async def get_built_image_doc(storage: Any, doc_id: str) -> dict | None:
    from storage.mcp_wizard_storage import get_built_image

    return await get_built_image(storage, doc_id)


def assert_wizard_built_image_doc(doc: dict) -> dict:
    """Ensure built-image row was created by the ZIP wizard (not user CRUD)."""
    from fastapi import HTTPException

    if not doc:
        raise HTTPException(status_code=404, detail="Built image not found")
    tenant_id = str(doc.get("tenant_id") or "__root__")
    server_id = str(doc.get("server_id") or doc.get("mcp_server") or "").strip()
    if doc.get("record_type") == "mcp_built_image":
        img = doc
    else:
        meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        img = (
            meta.get("mcp_built_image")
            if isinstance(meta.get("mcp_built_image"), dict)
            else {}
        )
        if str(img.get("record_type") or "") != "mcp_built_image":
            raise HTTPException(status_code=404, detail="Built image not found")
    upload_id = str(img.get("upload_id") or "").strip()
    if not upload_id or not server_id:
        raise HTTPException(status_code=403, detail="Built image record is incomplete")
    if str(doc.get("_id") or "").strip() != built_image_doc_id(
        tenant_id, server_id, upload_id
    ):
        raise HTTPException(
            status_code=403, detail="Built image id does not match wizard record"
        )
    if doc.get("record_type") != "mcp_built_image" and str(
        doc.get("name") or ""
    ).strip() != built_image_name(upload_id):
        raise HTTPException(
            status_code=403, detail="Built image name does not match wizard record"
        )
    return img


async def delete_built_image_doc(storage: Any, doc_id: str) -> bool:
    from storage.mcp_wizard_storage import delete_built_image

    return await delete_built_image(storage, doc_id)


async def image_tag_in_use(
    storage: Any,
    tenant_id: str | None,
    image_tag: str,
    *,
    exclude_mcp_server_id: str | None = None,
    exclude_tenant_id: str | None = None,
    global_scope: bool = False,
) -> bool:
    from storage.tool_doc_storage import get_mcp_tool_configurations

    tag = str(image_tag or "").strip()
    if not tag:
        return False
    excluded_server = str(exclude_mcp_server_id or "").strip()
    excluded_tenant = str(
        exclude_tenant_id if exclude_tenant_id is not None else tenant_id or ""
    ).strip()
    docs = await get_mcp_tool_configurations(
        storage,
        enabled_only=False,
        tenant_id=None if global_scope else tenant_id,
    )
    for doc in docs:
        if (
            excluded_server
            and excluded_tenant
            and str(doc.get("tenant_id") or "__root__") == excluded_tenant
            and str(doc.get("mcp_server") or "").strip() == excluded_server
        ):
            continue
        meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        ext = (
            meta.get("external_mcp")
            if isinstance(meta.get("external_mcp"), dict)
            else {}
        )
        if str(ext.get("image") or "").strip() == tag:
            return True
    return False


def new_upload_id() -> str:
    return uuid.uuid4().hex


def built_image_tag_tenant_prefix(tenant_id: str) -> str:
    """Expected ``AppFactory-mcp/{slug}/`` prefix for images owned by *tenant_id*."""
    return f"AppFactory-mcp/{mcp_docker_tenant_slug(tenant_id)}/"


def assert_built_image_tag_for_tenant(tag: str, tenant_id: str) -> str:
    """Validate tag format and ensure it belongs to the caller's tenant namespace."""
    from fastapi import HTTPException

    raw = str(tag or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Built image has no Docker tag")
    try:
        validated = validate_mcp_image_reference(raw)
    except McpSegmentIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    prefix = built_image_tag_tenant_prefix(tenant_id)
    if not validated.startswith(prefix):
        raise HTTPException(
            status_code=403,
            detail=f"Image tag is not in this tenant's namespace: {validated}",
        )
    return validated
