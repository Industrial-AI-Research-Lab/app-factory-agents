"""Docker image build for MCP ZIP packages (host CLI)."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from sandbox.host_cli import run_host_cli
from tools.mcp_dockerfile_finder import resolve_docker_build_context
from tools.mcp_dockerfile_legacy_rewrite import (
    dockerfile_needs_legacy_rewrite,
    rewrite_dockerfile_for_legacy_builder,
)
from tools.mcp_tool_ids import (
    McpSegmentIdError,
    mcp_docker_tenant_slug,
    normalize_docker_repo_segment,
    validate_mcp_image_reference,
    validate_mcp_segment_id,
)

logger = logging.getLogger(__name__)

MAX_BUILD_LOG_CHARS = 512 * 1024


def append_build_log(current: str, stdout: str, stderr: str) -> str:
    chunk = ""
    if stdout:
        chunk += stdout
    if stderr:
        if chunk and not chunk.endswith("\n"):
            chunk += "\n"
        chunk += stderr
    combined = (current or "") + chunk
    if len(combined) > MAX_BUILD_LOG_CHARS:
        return combined[-MAX_BUILD_LOG_CHARS:]
    return combined


def mcp_image_tag(tenant_id: str, server_id: str, upload_id: str) -> str:
    tid = mcp_docker_tenant_slug(tenant_id)
    sid = normalize_docker_repo_segment(
        validate_mcp_segment_id(server_id, "server_id").lower(),
        fallback="mcp",
    )
    short = re.sub(r"[^a-zA-Z0-9]+", "", str(upload_id or ""))[:8] or "latest"
    return validate_mcp_image_reference(f"AppFactory-mcp/{tid}/{sid}:{short}")


def _mcp_build_env() -> Dict[str, str]:
    """Build env for ``docker build``.

    Default disables BuildKit so hosts without the buildx plugin still work.
    Set ``MCP_ZIP_DOCKER_BUILDKIT=1`` only when buildx is installed.
    """
    if os.environ.get("MCP_ZIP_DOCKER_BUILDKIT", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return {"DOCKER_BUILDKIT": "1"}
    return {"DOCKER_BUILDKIT": "0"}


def _buildkit_or_buildx_error(stderr: str, exit_code: int) -> bool:
    if exit_code == 0:
        return False
    msg = (stderr or "").lower()
    return "buildkit" in msg and ("buildx" in msg or "missing" in msg or "broken" in msg)


def _buildkit_mount_error(stderr: str, exit_code: int) -> bool:
    if exit_code == 0:
        return False
    msg = (stderr or "").lower()
    return "buildkit" in msg and "mount" in msg


def _prepare_legacy_dockerfile(dockerfile: Path) -> tuple[Path, str]:
    """Write ``.AppFactory-legacy.Dockerfile`` when BuildKit-only syntax is present."""
    try:
        original = dockerfile.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return dockerfile, ""
    if not dockerfile_needs_legacy_rewrite(original):
        return dockerfile, ""
    rewritten, notes = rewrite_dockerfile_for_legacy_builder(original)
    if rewritten.strip() == original.strip():
        return dockerfile, ""
    legacy_path = dockerfile.parent / ".AppFactory-legacy.Dockerfile"
    legacy_path.write_text(rewritten, encoding="utf-8")
    detail = ", ".join(dict.fromkeys(notes)) if notes else "removed RUN --mount / COPY --chown"
    banner = f"[AppFactory] Rewrote Dockerfile for legacy builder ({detail})\n"
    logger.info(
        "[MCP_BUILD] legacy rewrite dockerfile=%s -> %s (%s)",
        dockerfile,
        legacy_path,
        detail,
    )
    return legacy_path, banner


def _docker_build_args(
    *,
    dockerfile: Path,
    tag: str,
    tenant_id: str,
    server_id: str,
    upload_id: str,
    context: Path,
) -> list[str]:
    return [
        "-f",
        str(dockerfile),
        "-t",
        tag,
        "--label",
        "AppFactory.mcp_built=true",
        "--label",
        f"tenant_id={tenant_id}",
        "--label",
        f"upload_id={upload_id}",
        "--label",
        f"server_id={server_id}",
        str(context),
    ]


async def docker_build_mcp_image(
    *,
    tenant_id: str,
    server_id: str,
    upload_id: str,
    extract_root: Path,
    dockerfile_relative_path: str,
    timeout: int = 900,
) -> Dict[str, Any]:
    """Run ``docker build`` and return host_cli result dict plus ``image_tag``."""
    root = extract_root.resolve()
    rel = dockerfile_relative_path.replace("\\", "/").lstrip("/")
    dockerfile = (root / rel).resolve()
    if not dockerfile.is_file():
        raise FileNotFoundError(f"Dockerfile not found: {rel}")
    context = resolve_docker_build_context(root, dockerfile)
    tag = mcp_image_tag(tenant_id, server_id, upload_id)
    build_env = _mcp_build_env()
    build_dockerfile = dockerfile
    log_prefix = ""
    if build_env.get("DOCKER_BUILDKIT") == "0":
        build_dockerfile, log_prefix = _prepare_legacy_dockerfile(dockerfile)

    build_args = _docker_build_args(
        dockerfile=build_dockerfile,
        tag=tag,
        tenant_id=tenant_id,
        server_id=server_id,
        upload_id=upload_id,
        context=context,
    )
    logger.info(
        "[MCP_BUILD] tenant=%s server=%s upload=%s tag=%s dockerfile=%s buildkit=%s",
        tenant_id,
        server_id,
        upload_id,
        tag,
        build_dockerfile.relative_to(context) if str(build_dockerfile).startswith(str(context)) else build_dockerfile,
        build_env.get("DOCKER_BUILDKIT"),
    )

    legacy_cmd = ["docker", "build", *build_args]
    res = await run_host_cli(legacy_cmd, timeout=timeout, env=build_env)
    stderr = str(res.get("stderr") or "")
    exit_code = int(res.get("exit_code") or 1)
    if exit_code != 0 and _buildkit_mount_error(stderr, exit_code) and build_dockerfile == dockerfile:
        logger.warning(
            "[MCP_BUILD] tenant=%s upload=%s — RUN --mount needs legacy rewrite, retrying",
            tenant_id,
            upload_id,
        )
        build_dockerfile, log_prefix = _prepare_legacy_dockerfile(dockerfile)
        build_args = _docker_build_args(
            dockerfile=build_dockerfile,
            tag=tag,
            tenant_id=tenant_id,
            server_id=server_id,
            upload_id=upload_id,
            context=context,
        )
        legacy_cmd = ["docker", "build", *build_args]
        res = await run_host_cli(legacy_cmd, timeout=timeout, env={"DOCKER_BUILDKIT": "0"})
    elif exit_code != 0 and _buildkit_or_buildx_error(stderr, exit_code):
        logger.warning(
            "[MCP_BUILD] tenant=%s upload=%s — BuildKit/buildx error, retrying with DOCKER_BUILDKIT=0",
            tenant_id,
            upload_id,
        )
        res = await run_host_cli(
            legacy_cmd,
            timeout=timeout,
            env={"DOCKER_BUILDKIT": "0"},
        )

    if log_prefix:
        res["stdout"] = log_prefix + str(res.get("stdout") or "")
    res["image_tag"] = tag
    return res
