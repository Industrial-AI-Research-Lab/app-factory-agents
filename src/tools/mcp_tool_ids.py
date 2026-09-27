"""MCP tool IDs: tenant-visible ``server.tool`` vs opaque Mongo document keys."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Optional

_MCP_PUBLIC_SEP = "."
_STORAGE_TENANT_SEP = "."
MCP_SEGMENT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

OPENAI_FUNCTION_NAME_MAX_LEN: int = 64


def clamp_openai_function_name(name: str, *, max_len: int = OPENAI_FUNCTION_NAME_MAX_LEN) -> str:
    """Truncate an already-sanitized OpenAI function name to max_len characters."""
    return str(name or "")[:max_len]
TENANT_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
OPENAI_FUNCTION_NAME_MAX_LEN = 64


class McpSegmentIdError(ValueError):
    """Invalid MCP server key or bare tool name (dots/spaces not allowed)."""


def validate_mcp_segment_id(value: str, field_label: str) -> str:
    """Require ``[a-zA-Z0-9_-]+`` for ``mcp_server`` and MCP tool ``name`` fields."""
    raw = str(value or "").strip()
    if not raw:
        raise McpSegmentIdError(f"{field_label} must be non-empty")
    if not MCP_SEGMENT_ID_RE.match(raw):
        raise McpSegmentIdError(
            f"{field_label} '{value}' is invalid "
            "(use letters, numbers, hyphen, underscore only; no dots)"
        )
    return raw


def validate_tenant_slug(tenant_id: str) -> str:
    """Tenant ``_id`` must not normalize into another tenant's storage prefix."""
    raw = str(tenant_id or "").strip()
    if not raw:
        raise McpSegmentIdError("tenant id must be non-empty")
    if not TENANT_SLUG_RE.match(raw):
        raise McpSegmentIdError(
            f"tenant id '{tenant_id}' is invalid "
            "(must start with a lowercase letter, be 2-64 chars, and use lowercase letters, digits, underscores)"
        )
    return raw


_DOCKER_REPO_SEGMENT_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")


def normalize_docker_repo_segment(value: str, *, fallback: str = "mcp") -> str:
    """Normalize a string to a valid Docker image repository path segment."""
    fb = str(fallback or "mcp").strip().lower() or "mcp"
    slug = re.sub(r"[^a-z0-9._-]+", "-", str(value or "").strip().lower())
    slug = re.sub(r"-+", "-", slug)
    slug = re.sub(r"\.+", ".", slug)
    slug = slug.strip(".-_")
    while "__" in slug:
        slug = slug.replace("__", "-")
    if not slug or not slug[0].isalnum():
        slug = f"{fb}-{slug}" if slug else fb
        slug = re.sub(r"[^a-z0-9._-]+", "-", slug).strip(".-_")
        while "__" in slug:
            slug = slug.replace("__", "-")
        if not slug or not slug[0].isalnum():
            slug = fb
    if not _DOCKER_REPO_SEGMENT_RE.match(slug):
        slug = fb
    return slug[:128]


def mcp_docker_tenant_slug(tenant_id: str) -> str:
    """Docker image path segment for ``AppFactory-mcp/{slug}/...`` (not Mongo tenant id)."""
    tid = _normalize_tenant_key(tenant_id)
    aliases = {
        "__root__": "root",
        "__system__": "system",
        "__default__": "root",
    }
    slug = aliases.get(tid, tid).lower()
    return normalize_docker_repo_segment(slug, fallback="tenant")


def validate_mcp_image_reference(image_ref: str) -> str:
    """Ensure ``AppFactory-mcp/{tenant}/{server}:{tag}`` is valid for ``docker -t``."""
    ref = str(image_ref or "").strip()
    if not ref or ":" not in ref:
        raise McpSegmentIdError(f"invalid MCP image reference: {image_ref!r}")
    repo, _, tag_part = ref.partition(":")
    if not repo.startswith("AppFactory-mcp/") or not tag_part:
        raise McpSegmentIdError(f"invalid MCP image reference: {image_ref!r}")
    for segment in repo.split("/"):
        if not segment or not _DOCKER_REPO_SEGMENT_RE.match(segment):
            raise McpSegmentIdError(
                f"invalid MCP image reference segment '{segment}' in {image_ref!r}"
            )
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$", tag_part):
        raise McpSegmentIdError(f"invalid MCP image tag part: {tag_part!r}")
    return ref


def _normalize_tenant_key(tenant_id: str) -> str:
    tid = str(tenant_id or "").strip() or "__root__"
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", tid)


def clamp_openai_function_name(
    name: str,
    *,
    max_len: int = OPENAI_FUNCTION_NAME_MAX_LEN,
) -> str:
    """Clamp a wire/function name to OpenAI's max length with a stable suffix."""
    text = str(name or "").strip()
    if not text:
        return text
    if len(text) <= max_len:
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    keep = max_len - len(digest) - 1
    if keep < 1:
        return digest[:max_len]
    return f"{text[:keep]}_{digest}"


def _legacy_mcp_public_suffix(server_id: str, mcp_tool_name: str) -> str:
    sid = str(server_id or "").strip()
    name = str(mcp_tool_name or "").strip()
    return f"{sid}.{name}".replace(" ", "_").lower()


def _mcp_public_suffix(server_id: str, mcp_tool_name: str) -> str:
    """Tenant-visible id: ``{server_id}.{tool_name}`` (spaces → underscores, lowercased)."""
    return _legacy_mcp_public_suffix(server_id, mcp_tool_name)


def mcp_public_tool_id(server_id: str, mcp_tool_name: str) -> str:
    """Tenant-visible public id ``{server}.{tool}`` for allow-lists and UI."""
    return _mcp_public_suffix(server_id, mcp_tool_name)


def mcp_tool_document_id(tenant_id: str, server_id: str, mcp_tool_name: str) -> str:
    """Opaque Mongo ``_id``: ``{tenant}.{server}.{tool}`` (dot-separated)."""
    tid = _normalize_tenant_key(tenant_id)
    pub = mcp_public_tool_id(server_id, mcp_tool_name)
    return f"{tid}{_STORAGE_TENANT_SEP}{pub}"


def mcp_tool_document_id_candidates(
    tenant_id: str,
    server_id: str,
    mcp_tool_name: str,
) -> list[str]:
    """Storage id candidates: canonical public id first, then hash-clamped legacy."""
    tid = _normalize_tenant_key(tenant_id)
    canonical = mcp_public_tool_id(server_id, mcp_tool_name)
    clamped = clamp_openai_function_name(canonical)
    ids = [f"{tid}{_STORAGE_TENANT_SEP}{canonical}"]
    if clamped != canonical:
        ids.append(f"{tid}{_STORAGE_TENANT_SEP}{clamped}")
    return ids


def normalize_mcp_public_tool_ref(
    tool_ref: str,
    tenant_id: str | None = None,
) -> str:
    """Canonical unclamped MCP public id for lookups and LLM function names."""
    public = mcp_public_id_from_tool_ref(tool_ref, tenant_id)
    parsed = parse_mcp_public_tool_id(public)
    if parsed:
        return mcp_public_tool_id(parsed[0], parsed[1])
    return public


def mcp_tool_doc_id(tenant_id: str, server_id: str, tool_name: str) -> str:
    return mcp_tool_document_id(tenant_id, server_id, tool_name)


mcp_storage_tool_id = mcp_tool_document_id


def parse_mcp_public_tool_id(public_id: str) -> tuple[str, str] | None:
    s = str(public_id or "").strip()
    if _MCP_PUBLIC_SEP not in s:
        return None
    server_id, tool_name = s.split(_MCP_PUBLIC_SEP, 1)
    server_id = server_id.strip()
    tool_name = tool_name.strip()
    if not server_id or not tool_name:
        return None
    return server_id, tool_name


def mcp_public_tool_id_from_doc(doc: dict) -> Optional[str]:
    if not isinstance(doc, dict) or doc.get("source") != "mcp_server":
        return None
    server_id = str(doc.get("mcp_server") or "").strip()
    rpc_name = mcp_rpc_name_from_doc(doc)
    if not server_id or not rpc_name:
        return None
    return mcp_public_tool_id(server_id, rpc_name)


def mcp_public_id_from_tool_ref(tool_ref: str, tenant_id: str | None = None) -> str:
    """Normalize an allow-list ref to the tenant-visible ``server.tool`` form."""
    ref = str(tool_ref or "").strip()
    if not ref:
        return ref
    tid = str(tenant_id or "").strip()
    if tid and ref.startswith(f"{tid}."):
        rest = ref[len(tid) + 1:]
        if parse_mcp_public_tool_id(rest):
            return rest
    parts = ref.split(".")
    if len(parts) >= 3:
        candidate = ".".join(parts[1:])
        if parse_mcp_public_tool_id(candidate):
            return candidate
    if parse_mcp_public_tool_id(ref):
        return ref
    if "." in ref:
        _, _, rest = ref.partition(".")
        if parse_mcp_public_tool_id(rest):
            return rest
    return ref


def composite_tool_name_from_storage_id(server_id: str, tool_id: str) -> str:
    """Bare MCP JSON-RPC tool name from a storage or public id."""
    public_id = mcp_public_id_from_tool_ref(tool_id)
    parsed = parse_mcp_public_tool_id(public_id)
    if not parsed:
        return str(tool_id or "")
    sid, mcp_name = parsed
    if server_id and sid == server_id:
        return mcp_name
    if server_id and public_id.startswith(f"{server_id}."):
        return public_id[len(server_id) + 1:]
    return mcp_name


def mcp_rpc_name_from_doc(doc: dict) -> str:
    """MCP JSON-RPC tool name for ``tools/call`` (path A: ``rpc_name``; legacy: ``name``)."""
    if not isinstance(doc, dict):
        return ""
    rpc = str(doc.get("rpc_name") or "").strip()
    if rpc:
        return rpc
    return str(doc.get("name") or "").strip()


def mcp_rpc_tool_name(tool_config: dict) -> str:
    """Bare tool name for MCP ``tools/call`` (``rpc_name`` contract field)."""
    rpc = mcp_rpc_name_from_doc(tool_config)
    if rpc:
        return rpc
    tool_id = str(tool_config.get("_id") or tool_config.get("id") or "").strip()
    raise McpSegmentIdError(
        f"MCP tool document missing rpc_name (tool_id={tool_id!r})"
    )


def _tenant_owns_doc(doc: dict, tenant_id: str) -> bool:
    """Strict ownership: doc belongs to this tenant's own scope (no sharing)."""
    doc_tenant = str(doc.get("tenant_id") or "__root__")
    if doc_tenant == "__default__":
        doc_tenant = "__root__"
    req = str(tenant_id or "__root__")
    if req == "__default__":
        req = "__root__"
    return doc_tenant == req


def _tenant_matches(doc: dict, tenant_id: str) -> bool:
    """Match MCP/builtin doc to requesting tenant; __system__ docs are shared (ADR-0013)."""
    if str(doc.get("tenant_id") or "") == "__system__":
        return True
    return _tenant_owns_doc(doc, tenant_id)


def tenant_ids_for_storage_lookup(tenant_id: str) -> tuple[str, ...]:
    """Tenant id variants to query when resolving configuration by tenant scope."""
    tid = str(tenant_id or "__root__")
    if tid in ("__root__", "__default__"):
        return ("__root__", "__default__")
    return (tid,)


async def resolve_mcp_tool_doc(
    storage: Any,
    tenant_id: str,
    tool_ref: str,
) -> Optional[dict]:
    """Load MCP tool config by llm wire name, public id, or opaque document id.

    Resolution is tenant-first, then ``__system__`` (shared platform offering,
    ADR-0013) — a tenant doc with the same wire name shadows the system doc.
    """
    tid = str(tenant_id or "__root__")
    doc = await _resolve_mcp_tool_doc_in_scope(storage, tid, tool_ref)
    if doc is not None or tid == "__system__":
        return doc
    return await _resolve_mcp_tool_doc_in_scope(storage, "__system__", tool_ref)


async def _resolve_mcp_tool_doc_in_scope(
    storage: Any,
    tid: str,
    tool_ref: str,
) -> Optional[dict]:
    """Resolve strictly within one tenant scope (ownership, not sharing)."""
    raw_ref = str(tool_ref or "").strip()
    if not raw_ref or storage is None:
        return None

    llm_finder = getattr(storage, "find_mcp_tool_configuration_by_llm_function_name", None)
    if callable(llm_finder):
        doc = await llm_finder(tid, raw_ref)
        if isinstance(doc, dict) and doc.get("source") == "mcp_server" and _tenant_owns_doc(doc, tid):
            return doc

    by_name = getattr(storage, "find_mcp_tool_configuration_by_wire_name", None)
    if callable(by_name):
        doc = await by_name(tid, raw_ref)
        if isinstance(doc, dict) and doc.get("source") == "mcp_server" and _tenant_owns_doc(doc, tid):
            return doc

    ref = normalize_mcp_public_tool_ref(raw_ref, tid)

    getter = getattr(storage, "get_mcp_tool_configuration", None)
    if not callable(getter):
        getter = storage.get_tool_configuration

    for lookup_ref in (ref, raw_ref):
        doc = await getter(lookup_ref)
        if isinstance(doc, dict) and doc.get("source") == "mcp_server" and _tenant_owns_doc(doc, tid):
            return doc

    raw_public = mcp_public_id_from_tool_ref(raw_ref, tid)
    parsed = parse_mcp_public_tool_id(raw_public)
    if not parsed:
        return None

    server_id, mcp_name = parsed
    for doc_id in mcp_tool_document_id_candidates(tid, server_id, mcp_name):
        doc = await getter(doc_id)
        if isinstance(doc, dict) and doc.get("source") == "mcp_server" and _tenant_owns_doc(doc, tid):
            return doc

    # Bare rpc name match within tenant + server (discovery / legacy shape)
    if hasattr(storage, "get_mcp_tool_configurations"):
        docs = await storage.get_mcp_tool_configurations(
            enabled_only=False, tenant_id=tid,
        )
        for candidate in docs:
            if not isinstance(candidate, dict):
                continue
            cand_rpc = mcp_rpc_name_from_doc(candidate)
            if (
                str(candidate.get("mcp_server") or "") == server_id
                and cand_rpc == mcp_name
                and _tenant_owns_doc(candidate, tid)
            ):
                return candidate

    return None


async def resolve_mcp_tool_storage_id(
    storage: Any,
    tenant_id: str,
    tool_ref: str,
) -> Optional[str]:
    doc = await resolve_mcp_tool_doc(storage, tenant_id, tool_ref)
    if isinstance(doc, dict) and doc.get("_id"):
        return str(doc["_id"])
    return None
