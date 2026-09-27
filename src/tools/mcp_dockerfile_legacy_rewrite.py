"""Rewrite Dockerfiles that use BuildKit-only syntax for legacy ``docker build``."""

from __future__ import annotations

import re

_COPY_CHOWN_FLAG = re.compile(r"\s+--chown=[^\s]+", re.IGNORECASE)
_HAS_MOUNT = re.compile(r"--mount\s*=", re.IGNORECASE)
_RUN_LINE = re.compile(r"^(\s*)RUN\s+(.+)$", re.IGNORECASE)
_MOUNT_FLAG = re.compile(r"^--mount=\S+", re.IGNORECASE)


def dockerfile_needs_legacy_rewrite(text: str) -> bool:
    return bool(_HAS_MOUNT.search(text or ""))


def _iter_logical_lines(text: str) -> list[str]:
    """Merge backslash continuations into single Dockerfile instructions."""
    logical: list[str] = []
    current = ""
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if current:
            current = f"{current} {line.lstrip()}"
        else:
            current = line
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        logical.append(current)
        current = ""
    if current:
        logical.append(current)
    return logical


def _strip_mount_flags_from_run_body(body: str) -> tuple[str, str]:
    """Return (joined mount flags, remaining shell command)."""
    rest = body.strip()
    mounts: list[str] = []
    while rest:
        m = _MOUNT_FLAG.match(rest)
        if not m:
            break
        mounts.append(m.group(0))
        rest = rest[m.end() :].lstrip()
    return " ".join(mounts), rest


def rewrite_dockerfile_for_legacy_builder(text: str) -> tuple[str, list[str]]:
    """Return rewritten Dockerfile and human-readable change notes."""
    notes: list[str] = []
    out: list[str] = []
    for line in _iter_logical_lines(text):
        run_match = _RUN_LINE.match(line)
        if run_match and _HAS_MOUNT.search(run_match.group(2)):
            indent, body = run_match.group(1), run_match.group(2)
            mount_blob, cmd = _strip_mount_flags_from_run_body(body)
            if (
                "uv.lock" in mount_blob
                and "pyproject.toml" in mount_blob
                and cmd.strip().startswith("uv sync")
            ):
                out.append(f"{indent}COPY uv.lock pyproject.toml ./")
                notes.append("RUN --mount bind -> COPY uv.lock pyproject.toml")
            if cmd.strip():
                out.append(f"{indent}RUN {cmd.strip()}")
                notes.append("stripped RUN --mount (cache/bind)")
            continue

        if _COPY_CHOWN_FLAG.search(line) and line.lstrip().upper().startswith("COPY "):
            out.append(_COPY_CHOWN_FLAG.sub("", line))
            notes.append("removed COPY --chown")
            continue

        out.append(line)

    result = "\n".join(out)
    if text.endswith("\n"):
        result += "\n"
    return result, notes
