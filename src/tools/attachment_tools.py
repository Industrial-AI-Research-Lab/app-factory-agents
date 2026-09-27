"""Agent tools for project user attachments.

``attachment_list`` is Mongo-only. ``attachment_view`` previews on the host
(no sandbox): metadata always, text body when safe. ``attachment_fetch`` mints
a presigned GET only after tenant+project ownership, then curls into
``/tmp/AppFactory-files/attachments`` (outside /workdir). The URL never enters
the tool result. Sandbox agents then use ``read`` on the returned path.
"""

from __future__ import annotations

import re
import base64
import logging
import os
import shlex
import uuid
from copy import deepcopy
from typing import Any, Dict, Optional

from storage.file_attachment_store import FileAttachmentStore
from storage.file_blob_store import FileBlobStore
from storage.file_upload_validation import check_upload_type
from tools.archive_tools import _trailing_byte_count, _fetch_soft_error

logger = logging.getLogger(__name__)

ATTACHMENT_TOOL_IDS = (
    "attachment_list",
    "attachment_view",
    "attachment_fetch",
    "attachment_presign_get",
    "attachment_presign_put",

)
ATTACHMENT_FETCH_DIR = "/tmp/AppFactory-files/attachments"
ATTACHMENT_PRESIGN_TTL_SECONDS = 6000
_DENIED_ERROR = "unknown or inaccessible attachment_id"
_STDERR_SNIPPET_MAX = 300
DEFAULT_FETCH_TIMEOUT_SECONDS = 120.0
_FETCH_TTL_SLACK_SECONDS = 60
_FETCH_CEILING_FACTOR = 2
_FETCH_CEILING_MARGIN_SECONDS = 60
DEFAULT_READ_MAX_BYTES = 256 * 1024
DEFAULT_READ_MAX_LINES = 800
# ponytail: inline host read for requirements agents without sandbox; upgrade via env
DEFAULT_INLINE_TEXT_MAX_BYTES = 32 * 1024
# ponytail: sum of inline text_content in one manifest; raise via env if requirements need more
DEFAULT_INLINE_TEXT_TOTAL_MAX_BYTES = 128 * 1024
# ponytail: VL via data URL in attachment_view; raise via env for larger photos
DEFAULT_IMAGE_VIEW_MAX_BYTES = 4 * 1024 * 1024
DIRECTORY_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
FILENAME_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")
FILENAME_RULE = (
    "filename may contain only the characters a-zA-Z0-9_.- (no spaces or Cyrillic)"
)
_TEXT_EXTENSIONS = frozenset(
    {
        ".txt",
        ".md",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
        ".xml",
        ".html",
        ".htm",
        ".log",
        ".ini",
        ".toml",
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".go",
        ".rs",
        ".c",
        ".cpp",
        ".h",
        ".sql",
        ".sh",
    }
)

_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "attachment_list",
            "description": (
                "List user-uploaded files attached to this project (id, filename, "
                "content type, size). Metadata only — use attachment_view to preview "
                "text, or attachment_fetch to download into the sandbox."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "attachment_view",
            "description": (
                "Preview a project attachment without the sandbox: metadata always, "
                "text content for small text uploads, and image bytes for vision "
                "models (image_data_url). Prefer this before attachment_fetch. "
                "PDF/Office return metadata only (xlsx/docx MCP is separate)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "attachment_id": {
                        "type": "string",
                        "description": "Attachment id from attachment_list or project_attachments.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "1-indexed start line (optional).",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-indexed end line (optional).",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Max lines when no line range is given (default 800).",
                    },
                },
                "required": ["attachment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "attachment_fetch",
            "description": (
                "Download a project attachment into your sandbox and return its path "
                f"under {ATTACHMENT_FETCH_DIR}. For text preview prefer attachment_view. "
                "Finished when the call returns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "attachment_id": {
                        "type": "string",
                        "description": "Attachment id from attachment_list or project_attachments.",
                    },
                },
                "required": ["attachment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "attachment_presign_get",
            "description": (
                "Generate a temporary presigned URL for downloading a user attachment "
                "directly from object storage. Returns the URL, expiration time and "
                "attachment metadata. The URL can be used for HTTP GET without "
                "authentication."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "attachment_id": {
                        "type": "string",
                        "description": "ID of the user attachment to download.",
                    },
                },
                "required": ["attachment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "attachment_presign_put",
            "description": (
                "Generate a temporary presigned URL for uploading a new file "
                "directly to object storage. Files uploaded through this tool "
                "are stored under the presigned_tool directory. When used by "
                "an MCP server, the directory should be the MCP server name. "
                "The URL can be used for HTTP PUT without authentication."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": (
                            "Logical directory name. When called by an MCP "
                            "server, use the MCP server name."
                        ),
                    },
                    "filename": {
                        "type": "string",
                        "description": f"Name of the file to upload; {FILENAME_RULE}.",
                    },
                    "content_type": {
                        "type": "string",
                        "description": (
                            "MIME type of the file to upload, for example "
                            "'text/markdown'."
                        ),
                    },
                },
                "required": ["directory", "filename", "content_type"],
            },
        },
    }
]


def attachment_tool_schemas() -> list[Dict[str, Any]]:
    return deepcopy(_SCHEMAS)


def public_attachment(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(doc.get("_id") or ""),
        "filename": doc.get("filename"),
        "content_type": doc.get("content_type"),
        "size_bytes": doc.get("size_bytes"),
    }


async def project_has_attachments(
    storage,
    project_id: Optional[str],
    tenant_id: Optional[str],
) -> bool:
    """True when this tenant/project already has user attachments."""
    coll = getattr(storage, "user_attachments", None)
    if coll is None or not project_id or not tenant_id:
        return False
    try:
        doc = await coll.find_one(
            {"tenant_id": tenant_id, "project_id": project_id}, {"_id": 1}
        )
        return doc is not None
    except Exception as exc:
        logger.warning("[ATTACH] presence check failed project=%s: %s", project_id, exc)
        return False


def _denied() -> Dict[str, Any]:
    return {"status": "error", "error": _DENIED_ERROR}


def _fetch_budget_seconds() -> float:
    try:
        raw = os.getenv("FILE_ATTACHMENT_FETCH_TIMEOUT_SECONDS")
        value = float(raw) if raw else DEFAULT_FETCH_TIMEOUT_SECONDS
    except ValueError:
        value = DEFAULT_FETCH_TIMEOUT_SECONDS
    if value <= 0:
        value = DEFAULT_FETCH_TIMEOUT_SECONDS
    return max(value, 1.0)


def fetch_ceiling_seconds() -> float:
    return _fetch_budget_seconds() * _FETCH_CEILING_FACTOR + _FETCH_CEILING_MARGIN_SECONDS


def _read_max_bytes() -> int:
    try:
        raw = os.getenv("ATTACHMENT_READ_MAX_BYTES")
        value = int(raw) if raw else DEFAULT_READ_MAX_BYTES
    except ValueError:
        value = DEFAULT_READ_MAX_BYTES
    return max(value, 1024)


def inline_text_max_bytes() -> int:
    try:
        raw = os.getenv("ATTACHMENT_INLINE_TEXT_MAX_BYTES")
        value = int(raw) if raw else DEFAULT_INLINE_TEXT_MAX_BYTES
    except ValueError:
        value = DEFAULT_INLINE_TEXT_MAX_BYTES
    return max(value, 1024)


def inline_text_total_max_bytes() -> int:
    try:
        raw = os.getenv("ATTACHMENT_INLINE_TEXT_TOTAL_MAX_BYTES")
        value = int(raw) if raw else DEFAULT_INLINE_TEXT_TOTAL_MAX_BYTES
    except ValueError:
        value = DEFAULT_INLINE_TEXT_TOTAL_MAX_BYTES
    return value if value > 0 else DEFAULT_INLINE_TEXT_TOTAL_MAX_BYTES


def _dest_name(doc: dict[str, Any]) -> str:
    aid = str(doc.get("_id") or "file")
    name = os.path.basename(str(doc.get("filename") or "file"))
    if not name or name in (".", ".."):
        name = "file"
    return f"{aid}_{name}"


def _looks_text(content_type: str | None, filename: str | None) -> bool:
    ct = (content_type or "").lower().split(";", 1)[0].strip()
    if ct.startswith("text/"):
        return True
    if ct in ("application/json", "application/xml", "application/yaml", "application/x-yaml"):
        return True
    ext = os.path.splitext(str(filename or ""))[1].lower()
    return ext in _TEXT_EXTENSIONS


def _is_binary_text(content: str) -> bool:
    sample = content[:8192]
    return "\x00" in sample


def _slice_text_content(
    content: str,
    *,
    start_line: Any,
    end_line: Any,
    max_lines: Any,
    ends_with_newline: bool,
) -> tuple[str, dict[str, Any]]:
    extra: dict[str, Any] = {}
    if isinstance(start_line, int) and isinstance(end_line, int) and start_line >= 1 and end_line >= start_line:
        lines = content.splitlines(keepends=False)
        sliced = lines[start_line - 1 : end_line]
        content = "\n".join(sliced) + ("\n" if ends_with_newline and sliced else "")
        return content, extra

    limit = int(max_lines) if isinstance(max_lines, int) and max_lines > 0 else DEFAULT_READ_MAX_LINES
    lines = content.splitlines(keepends=False)
    total_lines = len(lines)
    if limit > 0 and total_lines > limit:
        content = "\n".join(lines[:limit])
        if content and ends_with_newline:
            content += "\n"
        extra.update(
            {
                "truncated": True,
                "total_lines": total_lines,
                "start_line": 1,
                "end_line": limit,
            }
        )
    return content, extra


async def inline_attachment_text(blob_store: FileBlobStore, doc: dict[str, Any]) -> str | None:
    """Host-side text for small attachments (requirements agents, context manifest)."""
    size = doc.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        return None
    cap = inline_text_max_bytes()
    if size > cap:
        return None
    if not _looks_text(doc.get("content_type"), doc.get("filename")):
        return None
    object_key = doc.get("object_key")
    if not object_key or blob_store is None or not blob_store.is_configured():
        return None
    data = await blob_store.get_blob(str(object_key), max_bytes=cap)
    if data is None or len(data) > cap:
        return None
    if b"\x00" in data[:8192]:
        return None
    text = data.decode("utf-8", errors="replace")
    if _is_binary_text(text):
        return None
    return text


def _store(storage, blob: FileBlobStore | None = None) -> FileAttachmentStore:
    return FileAttachmentStore(storage, blob or FileBlobStore())


async def list_attachments(storage, project_id: Optional[str], tenant_id: Optional[str]) -> Dict[str, Any]:
    if not storage or not project_id or not tenant_id:
        return {"status": "success", "attachments": [], "truncated": False}
    rows, truncated = await _store(storage).list_user_attachments(
        tenant_id=str(tenant_id), project_id=str(project_id)
    )
    out: Dict[str, Any] = {
        "status": "success",
        "attachments": [public_attachment(row) for row in rows],
        "truncated": truncated,
    }
    if truncated:
        out["note"] = (
            "Attachment list is truncated to the newest entries; "
            "older files may still appear on chat messages."
        )
    return out


def _view_kind(content_type: str | None, filename: str | None) -> str:
    ct = (content_type or "").lower().split(";", 1)[0].strip()
    ext = os.path.splitext(str(filename or ""))[1].lower()
    if ct.startswith("image/") or ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return "image"
    if (
        ext in {".pdf", ".docx", ".xlsx"}
        or ct == "application/pdf"
        or "openxmlformats-officedocument" in ct
    ):
        return "document"
    if _looks_text(content_type, filename):
        return "text"
    return "binary"


def _image_view_max_bytes() -> int:
    raw = os.environ.get("ATTACHMENT_IMAGE_VIEW_MAX_BYTES", "").strip()
    if raw.isdigit():
        return max(1024, int(raw))
    return DEFAULT_IMAGE_VIEW_MAX_BYTES


def _image_data_url(content_type: str | None, filename: str | None, data: bytes) -> str:
    ct = (content_type or "").lower().split(";", 1)[0].strip()
    if not ct.startswith("image/"):
        ext = os.path.splitext(str(filename or ""))[1].lower()
        ct = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "application/octet-stream")
    return f"data:{ct};base64,{base64.b64encode(data).decode('ascii')}"


async def view_attachment(
    storage,
    blob_store: FileBlobStore,
    project_id: Optional[str],
    tenant_id: Optional[str],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Host-side preview: metadata always, text body when safe, images for VL. No sandbox."""

    params = params or {}
    attachment_id = params.get("attachment_id")
    if not storage or not project_id or not tenant_id or not isinstance(attachment_id, str) or not attachment_id.strip():
        return _denied()

    doc = await _store(storage, blob_store).get_user_attachment(
        tenant_id=str(tenant_id),
        project_id=str(project_id),
        attachment_id=attachment_id.strip(),
    )
    if not doc:
        return _denied()

    doc = await _store(storage, blob_store).refresh_user_attachment_size(
        tenant_id=str(tenant_id),
        project_id=str(project_id),
        attachment_id=attachment_id.strip(),
        doc=doc,
    )

    kind = _view_kind(doc.get("content_type"), doc.get("filename"))
    payload: Dict[str, Any] = {
        "status": "success",
        **public_attachment(doc),
        "kind": kind,
    }

    if kind == "image":
        object_key = doc.get("object_key")
        if not object_key or blob_store is None or not blob_store.is_configured():
            payload["note"] = (
                "Image attachment, but object storage is not configured for vision preview."
            )
            return payload
        max_bytes = _image_view_max_bytes()
        data = await blob_store.get_blob(str(object_key), max_bytes=max_bytes + 1)
        if data is None:
            return _denied()
        if len(data) > max_bytes:
            payload["note"] = (
                f"Image exceeds vision preview cap ({max_bytes} bytes); "
                "use attachment_fetch for sandbox access."
            )
            payload["truncated_bytes"] = True
            payload["max_bytes"] = max_bytes
            return payload
        payload["image_data_url"] = _image_data_url(
            doc.get("content_type"), doc.get("filename"), data
        )
        payload["note"] = (
            "Image bytes attached for vision models (see following multimodal message). "
            "Transcribe visible text; use attachment_fetch only if you need sandbox bytes."
        )
        logger.info(
            "[ATTACH_VIEW] kind=image project=%s attachment=%s bytes=%d — vision data URL ready",
            project_id,
            attachment_id.strip(),
            len(data),
        )
        return payload

    if kind != "text":
        notes = {
            "document": (
                "PDF/Office document — content is not inlined. Use attachment_fetch for "
                "sandbox access; xlsx/docx MCP is a separate integration."
            ),
            "binary": "Binary attachment — use attachment_fetch for sandbox access.",
        }
        payload["note"] = notes.get(kind, notes["binary"])
        return payload

    max_bytes = _read_max_bytes()
    object_key = doc.get("object_key")
    if not object_key or blob_store is None or not blob_store.is_configured():
        payload["note"] = "Text attachment, but object storage is not configured for preview."
        return payload

    data = await blob_store.get_blob(str(object_key), max_bytes=max_bytes + 1)
    if data is None:
        return _denied()
    truncated_bytes = len(data) > max_bytes
    if truncated_bytes:
        data = data[:max_bytes]
    if b"\x00" in data[:8192]:
        payload["kind"] = "binary"
        payload["note"] = "File looked text by name but contains binary bytes; use attachment_fetch."
        return payload

    content = data.decode("utf-8", errors="replace")
    if _is_binary_text(content):
        payload["kind"] = "binary"
        payload["note"] = "File looked text by name but contains binary bytes; use attachment_fetch."
        return payload

    ends_with_newline = content.endswith("\n")
    content, extra = _slice_text_content(
        content,
        start_line=params.get("start_line"),
        end_line=params.get("end_line"),
        max_lines=params.get("max_lines"),
        ends_with_newline=ends_with_newline,
    )
    payload["content"] = content
    if truncated_bytes:
        payload["truncated_bytes"] = True
        payload["max_bytes"] = max_bytes
    payload.update(extra)
    return payload


async def fetch_attachment(
    storage,
    blob_store: FileBlobStore,
    container_manager,
    project_id: Optional[str],
    tenant_id: Optional[str],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    params = params or {}
    attachment_id = params.get("attachment_id")
    if not storage or not project_id or not tenant_id or not isinstance(attachment_id, str) or not attachment_id.strip():
        return _denied()

    doc = await _store(storage, blob_store).get_user_attachment(
        tenant_id=str(tenant_id),
        project_id=str(project_id),
        attachment_id=attachment_id.strip(),
    )
    if not doc:
        return _denied()

    doc = await _store(storage, blob_store).refresh_user_attachment_size(
        tenant_id=str(tenant_id),
        project_id=str(project_id),
        attachment_id=attachment_id.strip(),
        doc=doc,
    )

    object_key = doc.get("object_key")
    if not object_key:
        return _denied()

    if not blob_store.is_configured():
        return {
            "status": "error",
            "error": "attachment_fetch: object storage is not configured",
            "error_type": "unavailable",
        }

    if container_manager is None:
        return {
            "status": "error",
            "error": "attachment_fetch: no sandbox in this deployment",
            "error_type": "unavailable",
        }
    status = await container_manager.get_container_status(project_id)
    if not (status.get("active") and status.get("environment_id")):
        try:
            created = await container_manager.get_or_create_container(project_id)
        except Exception as exc:
            logger.error("[ATTACH] fetch could not start a sandbox for %s: %s", project_id, exc)
            created = None
        if not (created or {}).get("environment_id"):
            return {
                "status": "error",
                "error": "attachment_fetch: could not start a sandbox for this project",
                "error_type": "unavailable",
            }
        try:
            await container_manager.restore_container_from_context(project_id, storage)
        except Exception as exc:
            logger.warning("[ATTACH] fetch could not restore files for %s: %s", project_id, exc)

    try:
        head = await blob_store.head_blob(object_key)
    except Exception as exc:
        logger.error("[ATTACH] fetch head failed attachment=%s: %s", attachment_id, exc)
        return {"status": "error", "error": "attachment_fetch: object storage unavailable", "error_type": "unavailable"}
    if head is None:
        return _denied()

    budget = _fetch_budget_seconds()
    ttl = int(budget) * _FETCH_CEILING_FACTOR + _FETCH_TTL_SLACK_SECONDS
    try:
        url = await blob_store.presign_get(object_key, ttl)
    except Exception as exc:
        logger.error("[ATTACH] fetch presign failed attachment=%s: %s", attachment_id, exc)
        return {"status": "error", "error": "attachment_fetch: failed to prepare the download", "error_type": "unavailable"}

    dest = f"{ATTACHMENT_FETCH_DIR}/{_dest_name(doc)}"
    q = shlex.quote
    command = (
        f"mkdir -p {q(ATTACHMENT_FETCH_DIR)} && rm -f {q(dest)} && "
        f"curl -fsSL --retry 3 --retry-max-time {int(budget)} "
        f"--max-time {int(budget)} -o {q(dest)} {q(url)}; "
        f"wc -c < {q(dest)} 2>/dev/null || echo 0"
    )
    exec_res = await container_manager.execute_in_container(project_id, command)
    stderr = str((exec_res or {}).get("stderr") or "").replace(url, "<presigned-url>")[:_STDERR_SNIPPET_MAX]
    downloaded = _trailing_byte_count(str((exec_res or {}).get("stdout") or ""))

    if downloaded is None:
        return _fetch_soft_error(
            f"attachment_fetch: sandbox download produced no byte count: {stderr}",
            exec_res,
        )
    if downloaded == 0:
        return _fetch_soft_error(
            f"attachment_fetch: sandbox download wrote no bytes: {stderr}",
            exec_res,
        )
    expected = head.get("ContentLength") if isinstance(head, dict) else None
    if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
        expected = doc.get("size_bytes") if isinstance(doc.get("size_bytes"), int) else None
    if expected is not None and downloaded != expected:
        return _fetch_soft_error(
            (
                f"attachment_fetch: incomplete download — {downloaded} of {expected} bytes. {stderr}"
            ),
            exec_res,
        )

    logger.info("[ATTACH] fetch attachment=%s project=%s path=%s bytes=%s", attachment_id, project_id, dest, downloaded)
    return {
        "status": "success",
        "id": str(doc.get("_id") or ""),
        "path": dest,
        "bytes": downloaded,
        "filename": doc.get("filename"),
        "note": f"Ready at {dest} ({downloaded} bytes). Use read with this path for text.",
    }


async def attachment_presign_get(
    storage,
    blob_store: FileBlobStore,
    project_id: Optional[str],
    tenant_id: Optional[str],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    params = params or {}

    attachment_id = params.get("attachment_id")

    if (
        not storage
        or not project_id
        or not tenant_id
        or not isinstance(attachment_id, str)
        or not attachment_id.strip()
    ):
        return _denied()

    attachment_id = attachment_id.strip()

    doc = await _store(storage, blob_store).get_user_attachment(
        tenant_id=str(tenant_id),
        project_id=str(project_id),
        attachment_id=attachment_id,
    )

    if not doc:
        return _denied()


    object_key = doc.get("object_key")
    if not isinstance(object_key, str) or not object_key:
        return _denied()

    if not blob_store.is_configured():
        return {
            "status": "error",
            "error": "attachment_presign_get: object storage is not configured",
        }


    ttl = ATTACHMENT_PRESIGN_TTL_SECONDS

    try:
        head = await blob_store.head_blob(object_key)
    except Exception as exc:
        logger.error(
            "[ATTACH] presign GET head failed attachment=%s: %s",
            attachment_id,
            exc,
        )
        return {
            "status": "error",
            "error": "attachment_presign_get: object storage unavailable",
        }

    if head is None:
        return _denied()

    try:
        url = await blob_store.presign_get(
            object_key,
            ttl,
        )
    except Exception as exc:
        logger.error(
            "[ATTACH] presign GET failed attachment=%s: %s",
            attachment_id,
            exc,
        )
        return {
            "status": "error",
            "error": "attachment_presign_get: failed to prepare the download",
        }

    return {
        "status": "success",
        "id": str(doc.get("_id") or ""),
        "url": url,
        "expires_in": ttl,
        "filename": doc.get("filename"),
        "content_type": doc.get("content_type"),
        "size_bytes": head.get("ContentLength"),
    }

async def attachment_presign_put(
    storage,
    blob_store: FileBlobStore,
    project_id: Optional[str],
    tenant_id: Optional[str],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    params = params or {}

    directory = params.get("directory")
    filename = params.get("filename")
    content_type = params.get("content_type")

    if (
        not storage
        or not project_id
        or not tenant_id
        or not isinstance(directory, str)
        or not directory.strip()
        or not DIRECTORY_RE.fullmatch(directory)
        or not isinstance(filename, str)
        or not filename.strip()
        or not isinstance(content_type, str)
        or not content_type.strip()
    ):
        return _denied()
    if not FILENAME_RE.fullmatch(filename):
        return {
            "status": "error",
            "error": f"attachment_presign_put: {FILENAME_RULE}",
        }

    directory = directory.strip()
    filename = filename.strip()
    content_type = content_type.strip()

    if not blob_store.is_configured():
        return {
            "status": "error",
            "error": "attachment_presign_put: object storage is not configured",
        }
    filename = uuid.uuid4().hex + "_" + filename
    content_type = check_upload_type(filename, content_type)
    object_key = f"presigned_tool/{tenant_id}/{project_id}/{directory}/{filename}"

    ttl = ATTACHMENT_PRESIGN_TTL_SECONDS

    try:
        url = await blob_store.presign_put(
            object_key,
            ttl,
            content_type=content_type,
        )
    except Exception as exc:
        logger.error(
            "[ATTACH] presign PUT failed object_key=%s: %s",
            object_key,
            exc,
        )
        return {
            "status": "error",
            "error": "attachment_presign_put: failed to prepare the upload",
        }
    attachment = await _store(storage, blob_store).create_user_attachment(
        tenant_id=tenant_id,
        project_id=project_id,
        filename=filename,
        content_type=content_type,
        object_key=object_key,
    )

    return {
        "status": "success",
        "url": url,
        "expires_in": ttl,
        "attachment_id": attachment["_id"],
        "filename": filename,
        "content_type": content_type,
        "object_key": object_key,
    }
