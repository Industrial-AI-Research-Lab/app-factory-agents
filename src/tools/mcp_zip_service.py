"""Safe ZIP extract for MCP package upload."""

from __future__ import annotations

import logging
import os
import re
import shutil
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from tools.mcp_tool_ids import McpSegmentIdError, normalize_docker_repo_segment, validate_mcp_segment_id

logger = logging.getLogger(__name__)


class McpZipError(ValueError):
    """Invalid or unsafe MCP ZIP archive."""


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


_DEFAULT_MCP_BUILD_ROOT = "/var/AppFactory/mcp-builds"


def _mcp_build_root() -> Path:
    """Read at call time so tests and runtime can override ``MCP_BUILD_ROOT``."""
    return Path(os.getenv("MCP_BUILD_ROOT", _DEFAULT_MCP_BUILD_ROOT))


def _zip_max_bytes() -> int:
    return _env_int("MCP_ZIP_MAX_BYTES", 104857600)


MCP_ZIP_MAX_BYTES = _zip_max_bytes()
MCP_ZIP_MAX_FILES = _env_int("MCP_ZIP_MAX_FILES", 5000)
MCP_ZIP_MAX_DEPTH = _env_int("MCP_ZIP_MAX_DEPTH", 12)
def _zip_max_uncompressed() -> int:
    return _env_int("MCP_ZIP_MAX_UNCOMPRESSED", 524288000)


# Back-compat alias (read at import; prefer _zip_max_uncompressed() in extract).
MCP_ZIP_MAX_UNCOMPRESSED = _zip_max_uncompressed()

_STRIP_SUFFIXES = ("-main", "-master")
_ZIP_SERVER_ID_SUFFIX = "-zip"
_MAX_SERVER_ID_LEN = 64


def _docker_safe_zip_server_id(candidate: str) -> str:
    """Ensure ``server_id`` is valid for MCP segment ids and Docker image repo segments."""
    if candidate.endswith(_ZIP_SERVER_ID_SUFFIX):
        base = candidate[: -len(_ZIP_SERVER_ID_SUFFIX)].rstrip("-")
        base = normalize_docker_repo_segment(base or "mcp", fallback="mcp")
        max_base = max(1, _MAX_SERVER_ID_LEN - len(_ZIP_SERVER_ID_SUFFIX))
        base = base[:max_base].rstrip("-._")
        if not base:
            base = "mcp"
        return f"{base}{_ZIP_SERVER_ID_SUFFIX}"
    return normalize_docker_repo_segment(candidate, fallback="mcp")


def normalize_zip_import_server_id(server_id: str) -> str:
    """Ensure ZIP-import MCP servers end with ``-zip`` (distinct from manual MCP configs)."""
    stem = re.sub(r"[^a-z0-9_-]+", "-", str(server_id or "").strip().lower()).strip("-")
    if not stem:
        stem = "mcp"
    if stem.endswith(_ZIP_SERVER_ID_SUFFIX):
        candidate = stem
    else:
        max_base = _MAX_SERVER_ID_LEN - len(_ZIP_SERVER_ID_SUFFIX)
        base = stem[:max_base].rstrip("-")
        candidate = f"{base}{_ZIP_SERVER_ID_SUFFIX}"
    try:
        candidate = validate_mcp_segment_id(candidate, "server_id")
    except McpSegmentIdError:
        candidate = f"mcp-{uuid.uuid4().hex[:8]}{_ZIP_SERVER_ID_SUFFIX}"
    candidate = _docker_safe_zip_server_id(candidate)
    try:
        return validate_mcp_segment_id(candidate, "server_id")
    except McpSegmentIdError:
        return f"mcp-{uuid.uuid4().hex[:8]}{_ZIP_SERVER_ID_SUFFIX}"


def suggest_server_id_from_zip_filename(filename: str) -> str:
    """Derive ``server_id`` from archive name (e.g. ``context7-master.zip`` → ``context7-zip``)."""
    stem = Path(str(filename or "mcp")).stem.lower()
    for suf in _STRIP_SUFFIXES:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    stem = re.sub(r"[^a-z0-9_-]+", "-", stem).strip("-")
    if not stem:
        stem = "mcp"
    return normalize_zip_import_server_id(stem)


async def read_upload_bounded(file: Any, *, max_bytes: int | None = None) -> bytes:
    """Read upload body in chunks; reject before exceeding *max_bytes* (default MCP_ZIP_MAX_BYTES)."""
    limit = max_bytes if max_bytes is not None else _zip_max_bytes()
    header_size = getattr(file, "size", None)
    if header_size is not None and int(header_size) > limit:
        raise McpZipError(f"zip exceeds max size ({limit} bytes)")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise McpZipError(f"zip exceeds max size ({limit} bytes)")
        chunks.append(chunk)
    return b"".join(chunks)


def cleanup_prior_upload_extract(extract_path: str | None) -> None:
    """Remove a previous ``{upload_id}/`` tree when the same server_id is re-uploaded."""
    if not extract_path:
        return
    p = Path(str(extract_path)).resolve()
    root = _mcp_build_root().resolve()
    try:
        p.relative_to(root)
    except ValueError:
        logger.warning("[MCP_ZIP] skip cleanup outside MCP_BUILD_ROOT path=%s", p)
        return
    if p.name != "src" or not p.parent.is_dir():
        return
    upload_dir = p.parent
    shutil.rmtree(upload_dir, ignore_errors=True)
    logger.info("[MCP_ZIP] removed prior upload dir=%s", upload_dir)


def upload_extract_dir(tenant_id: str, upload_id: str) -> Path:
    tid = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(tenant_id or "__root__"))
    return _mcp_build_root() / tid / upload_id / "src"


def _cleanup_partial_extract(dest: Path) -> None:
    """Remove ``{upload_id}/`` tree after a failed extract."""
    upload_dir = dest.parent
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)
        logger.info("[MCP_ZIP] cleaned partial extract dir=%s", upload_dir)


def _posix_depth(rel: str) -> int:
    p = PurePosixPath(rel)
    return len(p.parts)


def _safe_member_path(dest_root: Path, member_name: str) -> Path:
    rel = member_name.replace("\\", "/").lstrip("/")
    if not rel or rel.endswith("/"):
        raise McpZipError(f"invalid zip entry: {member_name!r}")
    if ".." in PurePosixPath(rel).parts:
        raise McpZipError(f"path traversal in zip entry: {member_name!r}")
    if _posix_depth(rel) > MCP_ZIP_MAX_DEPTH:
        raise McpZipError(f"zip entry too deep: {member_name!r}")
    target = (dest_root / rel).resolve()
    root_resolved = dest_root.resolve()
    if not str(target).startswith(str(root_resolved) + os.sep) and target != root_resolved:
        raise McpZipError(f"path escapes extract root: {member_name!r}")
    return target


def extract_mcp_zip(
    stream: BinaryIO,
    *,
    tenant_id: str,
    upload_id: str,
    zip_filename: str,
    declared_size: int | None = None,
) -> Path:
    """Extract ZIP under ``{MCP_BUILD_ROOT}/{tenant}/{upload_id}/src``."""
    max_bytes = _zip_max_bytes()
    if declared_size is not None and declared_size > max_bytes:
        raise McpZipError(f"zip exceeds max size ({max_bytes} bytes)")
    dest = upload_extract_dir(tenant_id, upload_id)
    try:
        dest.mkdir(parents=True, exist_ok=True)

        max_uncompressed = _zip_max_uncompressed()
        total_declared = 0
        bytes_written = 0
        file_count = 0
        with zipfile.ZipFile(stream) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                file_count += 1
                if file_count > MCP_ZIP_MAX_FILES:
                    raise McpZipError(f"zip exceeds max file count ({MCP_ZIP_MAX_FILES})")
                total_declared += int(info.file_size or 0)
                if total_declared > max_uncompressed:
                    raise McpZipError(
                        f"zip exceeds max uncompressed size ({max_uncompressed} bytes)"
                    )
                target = _safe_member_path(dest, info.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, open(target, "wb") as out:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        bytes_written += len(chunk)
                        if bytes_written > max_uncompressed:
                            raise McpZipError(
                                f"zip exceeds max uncompressed size ({max_uncompressed} bytes)"
                            )
                        out.write(chunk)
        if file_count == 0:
            raise McpZipError("zip contains no files to extract")
    except zipfile.BadZipFile as exc:
        _cleanup_partial_extract(dest)
        raise McpZipError("invalid zip archive") from exc
    except McpZipError:
        _cleanup_partial_extract(dest)
        raise
    except OSError as exc:
        _cleanup_partial_extract(dest)
        logger.error(
            "[MCP_ZIP] tenant=%s upload=%s — extract I/O failed: %s",
            tenant_id,
            upload_id,
            exc,
        )
        raise McpZipError(f"cannot create extract directory: {exc}") from exc

    logger.info(
        "[MCP_ZIP] tenant=%s upload=%s files=%d bytes_declared=%s path=%s — extracted",
        tenant_id,
        upload_id,
        file_count,
        declared_size,
        dest,
    )
    return dest
