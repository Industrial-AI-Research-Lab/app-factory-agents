"""Preflight checks for MCP ZIP Dockerfile candidates before ``docker build``."""

from __future__ import annotations

import re
import shlex
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Literal, Tuple

from tools.mcp_dockerfile_finder import _read_dockerfile_text, resolve_docker_build_context

PreflightStatus = Literal["ready", "warnings", "blocked"]

_COPY_ADD_LINE = re.compile(r"^\s*(?:COPY|ADD)\s+(.+)$", re.MULTILINE | re.IGNORECASE)
_LOCKFILE_NAMES = frozenset(
    {
        "pnpm-lock.yaml",
        "package-lock.json",
        "yarn.lock",
        "npm-shrinkwrap.json",
        "poetry.lock",
        "uv.lock",
    }
)


@dataclass
class PreflightIssue:
    code: str
    severity: str  # error | warning
    message: str
    path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DockerfilePreflightResult:
    status: PreflightStatus = "ready"
    issues: List[PreflightIssue] = field(default_factory=list)
    suggested_mode: str = "http"
    suggested_path: str = "/mcp"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "issues": [i.to_dict() for i in self.issues],
            "suggested_mode": self.suggested_mode,
            "suggested_path": self.suggested_path,
        }


def suggest_mode_from_dockerfile(text: str) -> str:
    """Infer MCP transport from Dockerfile instructions."""
    low = text.lower()
    if "streamable-http" in low or "streamable_http" in low:
        return "streamable-http"
    if re.search(r"--transport[\s\"']+stdio", low) or re.search(
        r"transport[\"']?\s*,\s*[\"']stdio", low
    ):
        return "stdio"
    if re.search(r"\bstdio\b", low) and not re.search(
        r"--transport[\s\"']+http", low
    ):
        if "expose" not in low and "8000" not in low and "8080" not in low:
            return "stdio"
    # MCP SDK CLI ``--transport http`` serves Streamable HTTP, not legacy JSON-RPC POST.
    if re.search(r"--transport[\s\"']+http", low) or re.search(
        r"transport[\"']?\s*,\s*[\"']http", low
    ):
        return "streamable-http"
    return "http"


_SECURITY_LINE_PATTERNS: Tuple[Tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"/var/run/docker\.sock", re.IGNORECASE), "docker_sock_mount", "References host Docker socket"),
    (re.compile(r"^\s*RUN\s+.*\b--privileged\b", re.IGNORECASE | re.MULTILINE), "privileged_run", "RUN uses --privileged"),
    (re.compile(r"^\s*RUN\s+.*\b--cap-add\b", re.IGNORECASE | re.MULTILINE), "cap_add_run", "RUN adds Linux capabilities"),
    (re.compile(r"^\s*RUN\s+.*\bSYS_ADMIN\b", re.IGNORECASE | re.MULTILINE), "sys_admin_cap", "RUN requests SYS_ADMIN"),
    (re.compile(r"^\s*RUN\s+.*\bnsenter\b", re.IGNORECASE | re.MULTILINE), "nsenter_run", "RUN uses nsenter"),
    (re.compile(r"^\s*RUN\s+.*\bchroot\b", re.IGNORECASE | re.MULTILINE), "chroot_run", "RUN uses chroot"),
)


def _iter_run_commands(text: str) -> List[str]:
    commands: List[str] = []
    current = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if current:
            current = f"{current} {line.lstrip()}"
        else:
            current = line
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        m = re.match(r"^\s*RUN\s+(.+)$", current, re.IGNORECASE)
        if m:
            commands.append(m.group(1))
        current = ""
    return commands


def _run_invokes_docker_cli(command: str) -> bool:
    # Split shell command into pipelines/chains; block only when docker binary
    # is used as the executable, not when "docker" appears in a filename/path.
    raw = command.strip()
    if raw.startswith("[") and raw.endswith("]"):
        try:
            arr = json.loads(raw)
            if isinstance(arr, list) and arr:
                exe = Path(str(arr[0])).name.lower()
                if exe == "docker":
                    return True
        except Exception:
            pass
    for chunk in re.split(r"(?:&&|\|\||[;|])", command):
        part = chunk.strip()
        if not part:
            continue
        try:
            argv = shlex.split(part, posix=True)
        except ValueError:
            argv = part.split()
        if not argv:
            continue
        exe = Path(argv[0]).name.lower()
        if exe == "docker":
            return True
    return False


def scan_dockerfile_security(text: str) -> List[PreflightIssue]:
    """Block Dockerfiles that try to escape the container or control the Docker host."""
    issues: List[PreflightIssue] = []
    if any(_run_invokes_docker_cli(cmd) for cmd in _iter_run_commands(text)):
        issues.append(
            PreflightIssue(
                code="docker_in_run",
                severity="error",
                message=(
                    "Dockerfile rejected for security: RUN invokes docker CLI. "
                    "Use a minimal MCP server image without host access."
                ),
                path="Dockerfile",
            )
        )
    for pattern, code, summary in _SECURITY_LINE_PATTERNS:
        if pattern.search(text):
            issues.append(
                PreflightIssue(
                    code=code,
                    severity="error",
                    message=(
                        f"Dockerfile rejected for security: {summary}. "
                        "Use a minimal MCP server image without host access."
                    ),
                    path="Dockerfile",
                )
            )
    return issues


def suggest_path_from_dockerfile(text: str, *, default: str = "/mcp") -> str:
    m = re.search(r"--path[\s=]+[\"']?(/[\w./-]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip() or default
    m = re.search(r"ENV\s+(?:MCP_)?PATH[\s=]+(/[\w./-]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip() or default
    return default


def _copy_line_uses_external_stage(line: str) -> bool:
    return bool(re.search(r"--from\s*=", line, re.IGNORECASE))


def _strip_copy_flags(body: str) -> str:
    rest = body.strip()
    while rest.startswith("--"):
        parts = rest.split(None, 1)
        if len(parts) < 2:
            return ""
        rest = parts[1].strip()
    return rest


def _parse_copy_sources(line: str) -> List[str]:
    m = _COPY_ADD_LINE.match(line.strip())
    if not m:
        return []
    if _copy_line_uses_external_stage(line):
        return []
    body = _strip_copy_flags(m.group(1))
    if not body or body.lstrip().startswith("["):
        return []
    tokens = body.split()
    if len(tokens) < 2:
        return []
    return tokens[:-1]


def _source_exists(context: Path, source: str) -> bool:
    src = source.strip().strip("\"'")
    if not src or src in (".", "/"):
        return True
    # ADD supports URL sources; they are fetched by docker, not read from context.
    if re.match(r"^[a-z][a-z0-9+.-]*://", src, re.IGNORECASE):
        return True
    # Build-time variable expansion can resolve the path later.
    if "${" in src:
        return True
    if "*" in src or "?" in src:
        return True
    path = (context / src).resolve()
    try:
        path.relative_to(context.resolve())
    except ValueError:
        return False
    return path.exists()


def _lockfile_hint(name: str) -> str:
    if name in ("pnpm-lock.yaml", "pnpm-workspace.yaml"):
        return (
            "GitHub ZIP archives often omit gitignored lockfiles. "
            "Use a full git clone or include pnpm-lock.yaml in the archive."
        )
    if name == "package-lock.json":
        return "Include package-lock.json in the ZIP or build from a full repository checkout."
    return "Add the lockfile to the archive or use a complete repository export."


def scan_archive_warnings(extract_root: Path) -> List[PreflightIssue]:
    """ZIP-wide warnings (not tied to one Dockerfile)."""
    issues: List[PreflightIssue] = []
    root = extract_root.resolve()
    for path in root.rglob(".env"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if "node_modules" in rel.split("/"):
            continue
        issues.append(
            PreflightIssue(
                code="env_file_in_zip",
                severity="warning",
                message=(
                    f"Found {rel} in the archive. Secrets are not imported automatically; "
                    "configure env vars in MCP tool settings after import."
                ),
                path=rel,
            )
        )
        break
    return issues


def run_dockerfile_preflight(
    extract_root: Path,
    dockerfile_relative: str,
) -> DockerfilePreflightResult:
    """Validate build context vs Dockerfile COPY/ADD before ``docker build``."""
    root = extract_root.resolve()
    df_path = (root / dockerfile_relative.replace("\\", "/").lstrip("/")).resolve()
    if not df_path.is_file():
        return DockerfilePreflightResult(
            status="blocked",
            issues=[
                PreflightIssue(
                    code="dockerfile_missing",
                    severity="error",
                    message=f"Dockerfile not found: {dockerfile_relative}",
                    path=dockerfile_relative,
                )
            ],
        )

    text = _read_dockerfile_text(df_path)
    context = resolve_docker_build_context(root, df_path)
    issues: List[PreflightIssue] = []

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for source in _parse_copy_sources(line):
            if _source_exists(context, source):
                continue
            hint = ""
            base = Path(source).name
            if base in _LOCKFILE_NAMES:
                hint = " " + _lockfile_hint(base)
            issues.append(
                PreflightIssue(
                    code="missing_copy_source",
                    severity="error",
                    message=(
                        f"Build context is missing {source!r} required by Dockerfile "
                        f"(context: {context.relative_to(root).as_posix() or '.'}).{hint}"
                    ).strip(),
                    path=source,
                )
            )

    if not _read_dockerfile_text(df_path).strip():
        issues.append(
            PreflightIssue(
                code="empty_dockerfile",
                severity="error",
                message="Dockerfile is empty.",
                path=dockerfile_relative,
            )
        )

    has_errors = any(i.severity == "error" for i in issues)
    has_warnings = any(i.severity == "warning" for i in issues)
    status: PreflightStatus = "blocked" if has_errors else ("warnings" if has_warnings else "ready")

    return DockerfilePreflightResult(
        status=status,
        issues=issues,
        suggested_mode=suggest_mode_from_dockerfile(text),
        suggested_path=suggest_path_from_dockerfile(text),
    )


def merge_preflight_status(
    current: PreflightStatus,
    new: PreflightStatus,
) -> PreflightStatus:
    order = {"ready": 0, "warnings": 1, "blocked": 2}
    return current if order[current] >= order[new] else new


def candidate_sort_key(candidate: dict) -> Tuple[int, int, str]:
    """Ready candidates first, then higher score, then path."""
    status = candidate.get("preflight_status") or "ready"
    rank = {"ready": 0, "warnings": 1, "blocked": 2}.get(status, 1)
    return (rank, -int(candidate.get("score") or 0), str(candidate.get("relative_path") or ""))
