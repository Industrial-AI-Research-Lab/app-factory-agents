"""Shared A2A artifact/response reconstruction helpers.

Pure, dependency-free transforms that turn an A2A agent's terminal artifacts (or a
bare Message reply) into (path, content) files for the ArtifactStore and into the
human-readable prose shown in chat. Extracted from WorkflowEngine so the static
workflow a2a node and the delegate path (AppFactory-182) share one implementation
instead of drifting copies. No I/O here — callers own the storage writes.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from integrations.a2a_output_modes import normalize_mime_type, own_media_type

# Characters that must never reach an artifact path segment: path traversal, Windows
# reserved chars, and control bytes. Applied to agent-supplied artifact names before
# they become file paths.
_UNSAFE_PATH_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def sanitize_a2a_path_segment(name: str) -> str:
    """Reduce an agent-supplied artifact name to one safe path segment."""
    cleaned = _UNSAFE_PATH_CHARS.sub("_", str(name))
    cleaned = cleaned.replace("..", "_").strip().strip(".").strip()
    return cleaned or "artifact"


def a2a_artifact_scope(run_id: Optional[str], node_id: Optional[str]) -> str:
    """Path segment that stops one node/run's artifacts from clobbering another's.

    save_file keys artifacts by (project, path) run-agnostically, so a same-named
    artifact from a later run or a different node would overwrite the earlier one and
    get_all_files would surface only the last. run_id + node_id together make the path
    unique per producing node execution. Empty (no ids) falls back to the flat layout.
    """
    segs = [
        sanitize_a2a_path_segment(str(s))
        for s in (run_id, node_id) if s
    ]
    return "/".join(segs)


def merge_a2a_artifact_update(
    artifacts: List[Dict[str, Any]], update: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Fold a streamed update in by the rule the A2A SDK server builds its Task with,
    so streaming yields the artifacts message/send would. An update with no id, or an
    append to an unseen id, is kept as its own artifact rather than lost. Returns a
    new list; neither argument is modified.
    """
    artifact_id = update.get("artifact_id")
    ids = [artifact.get("artifact_id") for artifact in artifacts]
    if not artifact_id or artifact_id not in ids:
        return [*artifacts, update]
    index = ids.index(artifact_id)
    merged = update
    if update.get("append"):
        existing = artifacts[index]
        merged = {
            **existing,
            "parts": [*(existing.get("parts") or []), *(update.get("parts") or [])],
        }
    return [*artifacts[:index], merged, *artifacts[index + 1:]]


_EXTENSION_BY_MEDIA_TYPE = {
    "text/markdown": "md",
    "text/csv": "csv",
    "application/json": "json",
    "application/geo+json": "geojson",
    "application/vnd.geo+json": "geojson",
}

_GEOJSON_OBJECT_TYPES = (
    "Feature",
    "FeatureCollection",
    "Point",
    "MultiPoint",
    "LineString",
    "MultiLineString",
    "Polygon",
    "MultiPolygon",
    "GeometryCollection",
)


def _part_media_types(part: Dict[str, Any]) -> Tuple[str, ...]:
    own = own_media_type(part)
    if own:
        return (own,)
    card = (normalize_mime_type(t) for t in part.get("card_media_types") or ())
    return tuple(t for t in card if t)


def _is_geojson_object(data: Any) -> bool:
    return isinstance(data, dict) and data.get("type") in _GEOJSON_OBJECT_TYPES


def _data_part_extension(part: Dict[str, Any]) -> str:
    # Written as serialised JSON whatever its type. Card types only list what the
    # part may be, so when they mix GeoJSON with another JSON type the data decides.
    is_geojson = [
        _EXTENSION_BY_MEDIA_TYPE.get(t) == "geojson" for t in _part_media_types(part)
    ]
    if any(is_geojson) and (all(is_geojson) or _is_geojson_object(part.get("data"))):
        return "geojson"
    return "json"


def _is_json_document(text: str) -> bool:
    try:
        return isinstance(json.loads(text), (dict, list))
    # json.loads raises RecursionError, not ValueError, on deeply nested input.
    except (ValueError, RecursionError):
        return False


def _text_file_extension(parts: List[Dict[str, Any]], content: str) -> str:
    # A part with no type adds None, which disagrees with any typed neighbour.
    extensions = {
        _EXTENSION_BY_MEDIA_TYPE.get(media_type, "txt") if media_type else None
        for part in parts
        for media_type in _part_media_types(part) or (None,)
    }
    declared = extensions.pop() if len(extensions) == 1 else None
    if declared in ("md", "csv"):
        return declared
    # A JSON-typed part can still carry prose and plain text can carry a JSON
    # document, so the content decides between .json/.geojson and .txt.
    if not _is_json_document(content):
        return "txt"
    return declared if declared in ("json", "geojson") else "json"


def a2a_artifact_to_files(
    artifact: Dict[str, Any], scope: str = ""
) -> List[Tuple[str, str]]:
    """Convert one A2A artifact's parts into (path, content) file pairs.

    A text, data or URL part carrying a ``filename`` (BlocksNet sends its CSV/JSON
    run files this way) is written under that name verbatim. The other parts are
    named after the artifact: all text and URL parts go into one file, each data
    part into its own, and any other part kind into a ``.parts.json`` sidecar so a
    contract surprise stays visible. Text parts are pieces of one text (the IDU
    agents send one per streamed token), so they join with nothing in between, as
    in the chat reply; each URL keeps a line of its own.

    ``scope`` (run/node) is inserted into the path so identically-named artifacts
    from different node executions coexist instead of overwriting each other.
    """
    name = sanitize_a2a_path_segment(
        artifact.get("name") or artifact.get("artifact_id") or "artifact"
    )
    prefix = f"a2a/{scope}" if scope else "a2a"
    parts = artifact.get("parts") or []
    texts: List[List[str]] = []
    text_parts: List[Dict[str, Any]] = []
    data_parts: List[Dict[str, Any]] = []
    other_parts: List[Any] = []
    named_files: List[Tuple[str, str]] = []
    after_url = False
    for part in parts:
        part = part or {}
        ptype = part.get("type")
        filename = part.get("filename")
        if filename and ptype in ("text", "data", "url"):
            if ptype == "data":
                content = json.dumps(
                    part.get("data") or {}, ensure_ascii=False, indent=2
                )
            else:
                content = (part.get("text") if ptype == "text" else part.get("url")) or ""
            named_files.append(
                (f"{prefix}/{sanitize_a2a_path_segment(filename)}", content)
            )
        elif ptype == "text":
            text = part.get("text") or ""
            if texts and not after_url:
                texts[-1].append(text)
            else:
                texts.append([text])
            after_url = False
            text_parts.append(part)
        elif ptype == "data":
            data_parts.append(part)
        elif ptype == "url":
            texts.append([part.get("url") or ""])
            after_url = True
        else:
            other_parts.append(part)

    files: List[Tuple[str, str]] = []
    if texts:
        content = "\n".join("".join(run) for run in texts)
        files.append((
            f"{prefix}/{name}.{_text_file_extension(text_parts, content)}",
            content,
        ))
    for i, part in enumerate(data_parts):
        stem = name if len(data_parts) == 1 else f"{name}.{i}"
        files.append((
            f"{prefix}/{stem}.{_data_part_extension(part)}",
            json.dumps(part.get("data") or {}, ensure_ascii=False, indent=2),
        ))
    if other_parts:
        files.append((
            f"{prefix}/{name}.parts.json",
            json.dumps(other_parts, ensure_ascii=False, indent=2),
        ))
    files.extend(named_files)
    return files


def dedupe_a2a_path(path: str, used: set) -> str:
    """Return a path not already used in this batch.

    Two A2A artifacts can share a name (a future agent may repeat one the
    way the restriction agent returns near-duplicate layer names). Without
    this, the second save_file would version-bump the first and
    get_all_files would surface only one, silently hiding the other.
    """
    if path not in used:
        used.add(path)
        return path
    stem, dot, ext = path.rpartition(".")
    i = 2
    while True:
        candidate = f"{stem}-{i}.{ext}" if dot else f"{path}-{i}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


def a2a_response_text(artifacts: List[Dict[str, Any]]) -> str:
    """Reconstruct the agent's human-readable reply from artifact text parts.

    Agents like IDU stream prose as many sub-word text parts ("выб","рали"),
    each already carrying its own leading space, so they rejoin with "" into
    readable text. Data parts are left for the Artifacts tab, and so are text
    parts carrying a filename: those are whole files (BlocksNet sends its run
    CSVs this way), not the reply.
    """
    chunks: List[str] = []
    for artifact in artifacts or []:
        texts = [
            part.get("text") or ""
            for part in (artifact.get("parts") or [])
            if isinstance(part, dict)
            and part.get("type") == "text"
            and not part.get("filename")
        ]
        joined = "".join(texts).strip()
        if joined:
            chunks.append(joined)
    return "\n\n".join(chunks)


def a2a_artifact_paths(artifacts: List[Dict[str, Any]], scope: str = "") -> List[str]:
    """The artifact-store paths an artifact list saves to, for linking in chat.

    Mirrors the save path computation (including dedup and the run/node ``scope``)
    without writing, so the links in the chat message match the Artifacts tab exactly.
    Pass the same ``scope`` the save used or the links break.
    """
    used: set = set()
    paths: List[str] = []
    for artifact in artifacts or []:
        for path, _content in a2a_artifact_to_files(artifact, scope):
            paths.append(dedupe_a2a_path(path, used))
    return paths
