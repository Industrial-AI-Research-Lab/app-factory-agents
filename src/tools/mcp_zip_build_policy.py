"""Validation policy for MCP ZIP upload, build, and smoke (explicit user inputs)."""

from __future__ import annotations

import re
from pathlib import Path

_ZIP_SMOKE_SEGMENT = "-zip-smoke-"
_ZIP_SMOKE_SERVER_ID_RE = re.compile(rf"{re.escape(_ZIP_SMOKE_SEGMENT)}[0-9a-z]{{8}}$", re.IGNORECASE)
from typing import List, Optional, Tuple

from fastapi import HTTPException

from tools.mcp_dockerfile_finder import find_dockerfile_candidates
from tools.mcp_dockerfile_preflight import run_dockerfile_preflight, scan_dockerfile_security

ALLOWED_ZIP_BUILD_MODES = frozenset({"stdio", "streamable-http"})
PACKAGE_BUILD_IN_FLIGHT_STATUSES = frozenset({"building", "image_ready", "smoke_running"})


def package_build_in_flight(status: Optional[str]) -> bool:
    """True while a ZIP build job (docker build and/or smoke) is still running."""
    return (str(status or "").strip()) in PACKAGE_BUILD_IN_FLIGHT_STATUSES


def package_blocks_new_upload(status: Optional[str]) -> bool:
    """Block ZIP re-upload while any build/smoke job is in flight (includes ``image_ready`` gap)."""
    return package_build_in_flight(status)


def is_zip_smoke_ephemeral_server_id(server_id: str) -> bool:
    """True for ``{server_id}-zip-smoke-{upload_suffix}`` ids used only during ZIP package smoke."""
    return bool(_ZIP_SMOKE_SERVER_ID_RE.search(str(server_id or "").strip()))


def package_rebuild_blocked(status: Optional[str], *, force_rebuild: bool = False) -> bool:
    """Block duplicate build+smoke.

    ``force_rebuild`` only allows retry from terminal ``ready``; in-flight work is always blocked.
    """
    if package_build_in_flight(status):
        return True
    st = str(status or "").strip()
    if force_rebuild:
        return st != "ready"
    return st == "ready"

_DOCKERFILE_PATH_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_./-]*$")


def normalize_zip_build_mode(mode: Optional[str]) -> str:
    """Require an explicit transport mode from the client (no Dockerfile inference)."""
    raw = (mode or "").strip().lower()
    if raw == "http":
        raw = "streamable-http"
    if raw not in ALLOWED_ZIP_BUILD_MODES:
        raise HTTPException(
            status_code=400,
            detail="mode is required: choose stdio or streamable-http",
        )
    return raw


def validate_container_port_for_mode(mode: str, container_port: Optional[int]) -> int:
    if mode == "stdio":
        return 0
    if container_port is None:
        raise HTTPException(
            status_code=400,
            detail="container_port is required for streamable-http",
        )
    port = int(container_port)
    if not 1 <= port <= 65535:
        raise HTTPException(status_code=400, detail="container_port must be between 1 and 65535")
    return port


def normalize_endpoint_path(path: Optional[str]) -> str:
    raw = (path or "/mcp").strip()
    if not raw.startswith("/"):
        raw = f"/{raw}"
    if not re.match(r"^/[a-zA-Z0-9_./-]*$", raw):
        raise HTTPException(status_code=400, detail="endpoint_path must be a safe URL path")
    return raw


def validate_dockerfile_relative_path(rel: str) -> str:
    cleaned = (rel or "").strip().replace("\\", "/").lstrip("/")
    if not cleaned or ".." in cleaned.split("/"):
        raise HTTPException(status_code=400, detail="Invalid dockerfile_relative_path")
    if not _DOCKERFILE_PATH_RE.match(cleaned):
        raise HTTPException(status_code=400, detail="Invalid dockerfile_relative_path characters")
    return cleaned


def assert_dockerfile_in_package(
    extract_root: Path,
    dockerfile_relative_path: str,
    *,
    known_candidates: Optional[List[dict]] = None,
) -> Path:
    """Ensure Dockerfile exists under extract root and was listed at analyze (if known)."""
    root = extract_root.resolve()
    rel = validate_dockerfile_relative_path(dockerfile_relative_path)
    df = (root / rel).resolve()
    if not str(df).startswith(str(root)) or not df.is_file():
        raise HTTPException(status_code=400, detail=f"Dockerfile not found in package: {rel}")

    if known_candidates:
        allowed = {str(c.get("relative_path") or "") for c in known_candidates}
        if rel not in allowed:
            raise HTTPException(
                status_code=400,
                detail="dockerfile_relative_path was not in the last analyze result; run analyze again",
            )
    elif known_candidates is not None:
        raise HTTPException(
            status_code=400,
            detail="Run analyze on this package before build (no Dockerfile candidates yet)",
        )
    return df


def assert_dockerfile_allowed_for_build(extract_root: Path, dockerfile_relative_path: str) -> None:
    """Preflight + security scan; blocks privileged or host-escaping Dockerfiles."""
    rel = validate_dockerfile_relative_path(dockerfile_relative_path)
    pf = run_dockerfile_preflight(extract_root, rel)
    sec_issues = scan_dockerfile_security(
        (extract_root / rel).read_text(encoding="utf-8", errors="replace")
    )
    errors = [i for i in pf.issues if i.severity == "error"]
    errors.extend(sec_issues)
    if errors:
        first = errors[0].message
        raise HTTPException(
            status_code=400,
            detail=f"Dockerfile blocked: {first}",
        )
    if pf.status == "blocked":
        raise HTTPException(
            status_code=400,
            detail=pf.issues[0].message if pf.issues else "Dockerfile preflight blocked",
        )


def resolve_listen_port_explicit(mode: str, container_port: int) -> int:
    """Smoke/discover listen port — only from user input, never from EXPOSE heuristics."""
    if mode == "stdio":
        return 0
    return container_port
