"""Find and rank Dockerfiles under an extracted MCP package tree."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

_DOCKERFILE_NAMES = ("Dockerfile", "dockerfile")
_EXPOSE_RE = re.compile(r"^\s*EXPOSE\s+(\d+)", re.MULTILINE | re.IGNORECASE)
_MCP_HINT_RE = re.compile(r"\b(mcp|stdio|streamable)\b", re.IGNORECASE)
# Prefer common MCP HTTP ports; avoid binding host :80/:443 (often taken on API hosts).
_PREFERRED_LISTEN_PORTS = (8080, 8000, 3000, 8888, 5000, 9000)
_REPO_ROOT_MARKERS = (
    "pnpm-lock.yaml",
    "pnpm-workspace.yaml",
    "package-lock.json",
    "yarn.lock",
    "go.work",
)


def resolve_docker_build_context(extract_root: Path, dockerfile: Path) -> Path:
    """Directory for ``docker build`` context (``-f`` dir parent may be too shallow in monorepos)."""
    root = extract_root.resolve()
    df = dockerfile.resolve()
    if not str(df).startswith(str(root)):
        raise ValueError("Dockerfile path escapes package root")
    chosen = df.parent
    current = df.parent
    while True:
        if any((current / name).is_file() for name in _REPO_ROOT_MARKERS):
            chosen = current
        if current == root or current.parent == current:
            break
        parent = current.parent
        if not str(parent).startswith(str(root)):
            break
        current = parent
    return chosen


def pick_listen_port(expose_ports: List[int], *, default: int = 8080) -> int:
    """Choose the TCP port the process listens on inside the container (from EXPOSE)."""
    if not expose_ports:
        return default
    for preferred in _PREFERRED_LISTEN_PORTS:
        if preferred in expose_ports:
            return preferred
    return expose_ports[0]


@dataclass
class DockerfileCandidate:
    relative_path: str
    context_dir: str
    score: int = 0
    expose_ports: List[int] = field(default_factory=list)
    suggested_mode: str = "http"
    suggested_container_port: int = 8080
    suggested_path: str = "/mcp"
    hints: List[str] = field(default_factory=list)
    preflight_status: str = "ready"
    preflight_issues: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _read_dockerfile_text(path: Path, limit: int = 32000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _parse_expose_ports(text: str) -> List[int]:
    ports: List[int] = []
    for m in _EXPOSE_RE.finditer(text):
        try:
            p = int(m.group(1))
            if 1 <= p <= 65535:
                ports.append(p)
        except (TypeError, ValueError):
            continue
    return ports


def _score_candidate(rel_path: str, text: str, expose_ports: List[int]) -> tuple[int, str, int, List[str]]:
    score = 0
    hints: List[str] = []
    parts = Path(rel_path).parts
    depth = len(parts) - 1
    if parts[-1] in _DOCKERFILE_NAMES and depth <= 1:
        score += 30
        hints.append("dockerfile near package root")
    elif "docker" in "/".join(parts).lower() or "deploy" in "/".join(parts).lower():
        score += 5
    if expose_ports:
        score += 20
        hints.append(f"EXPOSE {expose_ports[0]}")
    if _MCP_HINT_RE.search(text):
        score += 15
        hints.append("CMD/ENTRYPOINT mentions MCP")
    if Path(rel_path).name.lower() in ("dockerfile.uv",) or rel_path.lower().endswith(".uv"):
        score -= 50
        hints.append("uv/BuildKit variant — prefer plain Dockerfile for ZIP import")
    if "--mount=" in text.lower():
        score -= 80
        hints.append("uses RUN --mount (needs BuildKit or AppFactory legacy rewrite)")
    if depth > 4:
        score -= 10 * (depth - 4)
    port = pick_listen_port(expose_ports)
    return score, port, hints


def resolve_listen_port_for_dockerfile(extract_root: Path, dockerfile_relative: str) -> int:
    """Parse EXPOSE from the selected Dockerfile under *extract_root*."""
    path = (extract_root / dockerfile_relative).resolve()
    if not path.is_file():
        return 8080
    return pick_listen_port(_parse_expose_ports(_read_dockerfile_text(path)))


def find_dockerfile_candidates(
    extract_root: Path,
    *,
    max_depth: int = 8,
) -> List[DockerfileCandidate]:
    """Scan ``extract_root`` for Dockerfiles and return sorted by score descending."""
    root = extract_root.resolve()
    if not root.is_dir():
        return []

    found: List[DockerfileCandidate] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name not in _DOCKERFILE_NAMES:
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if len(Path(rel).parts) > max_depth:
            continue
        context = resolve_docker_build_context(root, path)
        try:
            ctx_rel = context.relative_to(root).as_posix()
        except ValueError:
            ctx_rel = "."
        if ctx_rel == ".":
            ctx_rel = ""
        text = _read_dockerfile_text(path)
        expose_ports = _parse_expose_ports(text)
        score, port, hints = _score_candidate(rel, text, expose_ports)
        from tools.mcp_dockerfile_preflight import run_dockerfile_preflight

        pf = run_dockerfile_preflight(root, rel)
        mode = pf.suggested_mode
        if pf.status == "ready":
            score += 25
            hints.append("preflight OK")
        elif pf.status == "warnings":
            score += 5
        else:
            score -= 100
        found.append(
            DockerfileCandidate(
                relative_path=rel,
                context_dir=ctx_rel,
                score=score,
                expose_ports=expose_ports,
                suggested_mode=mode,
                suggested_container_port=port,
                suggested_path=pf.suggested_path,
                hints=hints,
                preflight_status=pf.status,
                preflight_issues=[i.to_dict() for i in pf.issues],
            )
        )

    from tools.mcp_dockerfile_preflight import candidate_sort_key

    found.sort(key=lambda c: candidate_sort_key(c.to_dict()))
    return found


def autofill_from_candidates(
    server_id: str,
    candidates: List[DockerfileCandidate],
    *,
    image_tag: str | None = None,
) -> dict:
    """Minimal autofill; omitted runtime fields use ``McpPackageAutofill`` defaults."""
    _ = candidates
    return {
        "server_id": server_id,
        "image": image_tag,
        "container_port": None,
        "path": None,
        "mode": None,
        "docker_cmd_args": None,
    }
