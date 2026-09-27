"""Agent tools for tenant-shared artifacts.

``tenant_artifact_list`` is Mongo-only. ``tenant_artifact_fetch`` mints a
presigned GET only after tenant ownership, then curls into
``/tmp/AppFactory-files/tenant-artifacts`` (outside /workdir). The URL never
enters the tool result.
"""

from __future__ import annotations

import logging
import os
import shlex
from copy import deepcopy
from typing import Any, Dict, Optional

from storage.file_attachment_store import FileAttachmentStore
from storage.file_blob_store import FileBlobStore
from tools.archive_tools import _trailing_byte_count, _fetch_soft_error

logger = logging.getLogger(__name__)

TENANT_ARTIFACT_TOOL_IDS = ("tenant_artifact_list", "tenant_artifact_fetch")
TENANT_ARTIFACT_FETCH_DIR = "/tmp/AppFactory-files/tenant-artifacts"
_DENIED_ERROR = "unknown or inaccessible artifact_id"
_STDERR_SNIPPET_MAX = 300
DEFAULT_FETCH_TIMEOUT_SECONDS = 120.0
_FETCH_TTL_SLACK_SECONDS = 60
_FETCH_CEILING_FACTOR = 2
_FETCH_CEILING_MARGIN_SECONDS = 60

_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "tenant_artifact_list",
            "description": (
                "List tenant-shared artifacts available to this project (id, filename, "
                "content type, size, title). Metadata only — use tenant_artifact_fetch "
                "to download one into the sandbox."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tenant_artifact_fetch",
            "description": (
                "Download a tenant-shared artifact into your sandbox and return its path "
                f"under {TENANT_ARTIFACT_FETCH_DIR}. Finished when the call returns. "
                "Do not read the whole file into the conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "artifact_id": {
                        "type": "string",
                        "description": "Artifact id from tenant_artifact_list or tenant_artifacts.",
                    },
                },
                "required": ["artifact_id"],
            },
        },
    },
]


def tenant_artifact_tool_schemas() -> list[Dict[str, Any]]:
    return deepcopy(_SCHEMAS)


async def tenant_has_artifacts(
    storage,
    tenant_id: Optional[str],
) -> bool:
    """True when this tenant already has shared tenant artifacts."""
    coll = getattr(storage, "tenant_artifacts", None)
    if coll is None or not tenant_id:
        return False
    try:
        doc = await coll.find_one({"tenant_id": tenant_id}, {"_id": 1})
        return doc is not None
    except Exception as exc:
        logger.warning("[TENANT_ARTIFACT] presence check failed tenant=%s: %s", tenant_id, exc)
        return False


def public_tenant_artifact(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(doc.get("_id") or ""),
        "filename": doc.get("filename"),
        "content_type": doc.get("content_type"),
        "size_bytes": doc.get("size_bytes"),
        "title": doc.get("title"),
        "description": doc.get("description"),
    }


def _denied() -> Dict[str, Any]:
    return {"status": "error", "error": _DENIED_ERROR}


def _fetch_budget_seconds() -> float:
    try:
        raw = os.getenv("FILE_TENANT_ARTIFACT_FETCH_TIMEOUT_SECONDS")
        if not raw:
            raw = os.getenv("FILE_ATTACHMENT_FETCH_TIMEOUT_SECONDS")
        value = float(raw) if raw else DEFAULT_FETCH_TIMEOUT_SECONDS
    except ValueError:
        value = DEFAULT_FETCH_TIMEOUT_SECONDS
    if value <= 0:
        value = DEFAULT_FETCH_TIMEOUT_SECONDS
    return max(value, 1.0)


def fetch_ceiling_seconds() -> float:
    return _fetch_budget_seconds() * _FETCH_CEILING_FACTOR + _FETCH_CEILING_MARGIN_SECONDS


def _dest_name(doc: dict[str, Any]) -> str:
    aid = str(doc.get("_id") or "file")
    name = os.path.basename(str(doc.get("filename") or "file"))
    if not name or name in (".", ".."):
        name = "file"
    return f"{aid}_{name}"


def _store(storage, blob: FileBlobStore | None = None) -> FileAttachmentStore:
    return FileAttachmentStore(storage, blob or FileBlobStore())


async def list_tenant_artifacts(storage, tenant_id: Optional[str]) -> Dict[str, Any]:
    if not storage or not tenant_id:
        return {"status": "success", "artifacts": [], "truncated": False}
    rows, truncated = await _store(storage).list_tenant_artifacts(tenant_id=str(tenant_id))
    out: Dict[str, Any] = {
        "status": "success",
        "artifacts": [public_tenant_artifact(row) for row in rows],
        "truncated": truncated,
    }
    if truncated:
        out["note"] = "Tenant artifact list is truncated to the newest entries."
    return out


async def fetch_tenant_artifact(
    storage,
    blob_store: FileBlobStore,
    container_manager,
    project_id: Optional[str],
    tenant_id: Optional[str],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    params = params or {}
    artifact_id = params.get("artifact_id")
    if not storage or not project_id or not tenant_id or not isinstance(artifact_id, str) or not artifact_id.strip():
        return _denied()

    doc = await _store(storage, blob_store).get_tenant_artifact(
        tenant_id=str(tenant_id),
        artifact_id=artifact_id.strip(),
    )
    if not doc:
        return _denied()

    object_key = doc.get("object_key")
    if not object_key:
        return _denied()

    if not blob_store.is_configured():
        return {
            "status": "error",
            "error": "tenant_artifact_fetch: object storage is not configured",
            "error_type": "unavailable",
        }

    if container_manager is None:
        return {
            "status": "error",
            "error": "tenant_artifact_fetch: no sandbox in this deployment",
            "error_type": "unavailable",
        }
    status = await container_manager.get_container_status(project_id)
    if not (status.get("active") and status.get("environment_id")):
        try:
            created = await container_manager.get_or_create_container(project_id)
        except Exception as exc:
            logger.error("[TENANT_ARTIFACT] fetch could not start a sandbox for %s: %s", project_id, exc)
            created = None
        if not (created or {}).get("environment_id"):
            return {
                "status": "error",
                "error": "tenant_artifact_fetch: could not start a sandbox for this project",
                "error_type": "unavailable",
            }
        try:
            await container_manager.restore_container_from_context(project_id, storage)
        except Exception as exc:
            logger.warning("[TENANT_ARTIFACT] fetch could not restore files for %s: %s", project_id, exc)

    try:
        head = await blob_store.head_blob(object_key)
    except Exception as exc:
        logger.error("[TENANT_ARTIFACT] fetch head failed artifact=%s: %s", artifact_id, exc)
        return {"status": "error", "error": "tenant_artifact_fetch: object storage unavailable", "error_type": "unavailable"}
    if head is None:
        return _denied()

    budget = _fetch_budget_seconds()
    ttl = int(budget) * _FETCH_CEILING_FACTOR + _FETCH_TTL_SLACK_SECONDS
    try:
        url = await blob_store.presign_get(object_key, ttl)
    except Exception as exc:
        logger.error("[TENANT_ARTIFACT] fetch presign failed artifact=%s: %s", artifact_id, exc)
        return {
            "status": "error",
            "error": "tenant_artifact_fetch: failed to prepare the download",
            "error_type": "unavailable",
        }

    dest = f"{TENANT_ARTIFACT_FETCH_DIR}/{_dest_name(doc)}"
    q = shlex.quote
    command = (
        f"mkdir -p {q(TENANT_ARTIFACT_FETCH_DIR)} && rm -f {q(dest)} && "
        f"curl -fsSL --retry 3 --retry-max-time {int(budget)} "
        f"--max-time {int(budget)} -o {q(dest)} {q(url)}; "
        f"wc -c < {q(dest)} 2>/dev/null || echo 0"
    )
    exec_res = await container_manager.execute_in_container(project_id, command)
    stderr = str((exec_res or {}).get("stderr") or "").replace(url, "<presigned-url>")[:_STDERR_SNIPPET_MAX]
    downloaded = _trailing_byte_count(str((exec_res or {}).get("stdout") or ""))

    if downloaded is None:
        return _fetch_soft_error(
            f"tenant_artifact_fetch: sandbox download produced no byte count: {stderr}",
            exec_res,
        )
    if downloaded == 0:
        return _fetch_soft_error(
            f"tenant_artifact_fetch: sandbox download wrote no bytes: {stderr}",
            exec_res,
        )
    expected = head.get("ContentLength") if isinstance(head, dict) else None
    if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
        expected = doc.get("size_bytes") if isinstance(doc.get("size_bytes"), int) else None
    if expected is not None and downloaded != expected:
        return _fetch_soft_error(
            (
                f"tenant_artifact_fetch: incomplete download — {downloaded} of {expected} bytes. {stderr}"
            ),
            exec_res,
        )

    logger.info(
        "[TENANT_ARTIFACT] fetch artifact=%s tenant=%s path=%s bytes=%s",
        artifact_id,
        tenant_id,
        dest,
        downloaded,
    )
    return {
        "status": "success",
        "id": str(doc.get("_id") or ""),
        "path": dest,
        "bytes": downloaded,
        "filename": doc.get("filename"),
        "note": (
            f"Ready at {dest} ({downloaded} bytes). Process it in the sandbox. "
            "Do not read the whole file into the conversation."
        ),
    }
