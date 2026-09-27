"""Shared scrub + multimodal follow-ups for attachment_view image results.

Live tool results may carry ``image_data_url`` (data URL). Journal / SSE /
Inspector capture must not keep the bytes; resume must not promise a follow-up
that cannot be rebuilt from ``"<omitted>"``.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

IMAGE_DATA_URL_OMITTED = "<omitted>"
_VISION_FOLLOW = "image bytes follow in the next user message"
_VISION_OMITTED = (
    "vision omitted from journal/capture; call attachment_view again to re-load the image"
)


def journal_safe_tool_result(result: Any) -> Any:
    """Drop bulky vision payloads before persisting tool results / SSE."""
    if not isinstance(result, dict):
        return result
    scrubbed = None
    url = result.get("image_data_url")
    if url is not None:
        scrubbed = dict(result)
        scrubbed["image_data_url"] = IMAGE_DATA_URL_OMITTED
    preview = result.get("preview")
    if isinstance(preview, str) and "data:image/" in preview:
        scrubbed = scrubbed if scrubbed is not None else dict(result)
        # Archive spill head/tail can embed a data-URL prefix — never keep it.
        scrubbed["preview"] = IMAGE_DATA_URL_OMITTED
    return scrubbed if scrubbed is not None else result


def has_live_vision_bytes(result: Any) -> bool:
    """True when result still carries a data-URL image for a VL follow-up."""
    if not isinstance(result, dict):
        return False
    url = result.get("image_data_url")
    return isinstance(url, str) and url.startswith("data:image/")


def format_tool_output(result: Any) -> str:
    """Tool JSON for the LLM — never include raw image_data_url bytes."""
    if not isinstance(result, dict):
        return str(result)
    url = result.get("image_data_url")
    payload = {k: v for k, v in result.items() if k not in {"status", "tool_id", "image_data_url"}}
    if isinstance(url, str) and url.startswith("data:image/"):
        payload["vision"] = _VISION_FOLLOW
    elif url:
        # Journal scrubbed value or any non-data placeholder — do not promise bytes.
        payload["vision"] = _VISION_OMITTED
    return (
        json.dumps(payload, ensure_ascii=False, default=str)
        if payload
        else json.dumps(result, ensure_ascii=False, default=str)
    )


def vision_followups_from_tool_result(result: Any) -> List[Dict[str, Any]]:
    """Responses API multimodal items after a live attachment_view with bytes."""
    if not isinstance(result, dict):
        return []
    url = result.get("image_data_url")
    if not isinstance(url, str) or not url.startswith("data:image/"):
        return []
    filename = result.get("filename") or "image"
    aid = result.get("id") or ""
    return [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        f"[attachment_view vision] filename={filename} "
                        f"attachment_id={aid}. Read all visible text on this image."
                    ),
                },
                {"type": "input_image", "image_url": url},
            ],
        }
    ]


def vision_chat_followups_from_tool_result(result: Any) -> List[Dict[str, Any]]:
    """Chat Completions multimodal user messages (SimpleAgentRunner)."""
    items: List[Dict[str, Any]] = []
    for follow in vision_followups_from_tool_result(result):
        parts: List[Dict[str, Any]] = []
        for block in follow.get("content") or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in ("text", "input_text"):
                parts.append({"type": "text", "text": block.get("text", "")})
            elif btype in ("input_image", "image_url"):
                url = block.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url")
                if isinstance(url, str) and url.startswith("data:image/"):
                    parts.append({"type": "image_url", "image_url": {"url": url}})
        if parts:
            items.append({"role": "user", "content": parts})
    return items


def _scrub_image_url_value(url: Any) -> Any:
    if isinstance(url, str) and url.startswith("data:image/"):
        return IMAGE_DATA_URL_OMITTED
    if isinstance(url, dict):
        inner = url.get("url")
        if isinstance(inner, str) and inner.startswith("data:image/"):
            out = dict(url)
            out["url"] = IMAGE_DATA_URL_OMITTED
            return out
    return url


def _scrub_content_blocks(content: Any) -> Any:
    if isinstance(content, str) and "data:image/" in content:
        # Tool JSON that somehow still embeds a data URL — strip for capture only.
        try:
            parsed = json.loads(content)
        except Exception:
            return content
        if isinstance(parsed, dict) and isinstance(parsed.get("image_data_url"), str):
            parsed = journal_safe_tool_result(parsed)
            if parsed.get("image_data_url") == IMAGE_DATA_URL_OMITTED:
                parsed["vision"] = _VISION_OMITTED
            return json.dumps(parsed, ensure_ascii=False, default=str)
        return content
    if not isinstance(content, list):
        return content
    out = []
    for block in content:
        if not isinstance(block, dict):
            out.append(block)
            continue
        b = dict(block)
        btype = b.get("type")
        if btype in ("input_image", "image_url") or "image_url" in b:
            if "image_url" in b:
                b["image_url"] = _scrub_image_url_value(b["image_url"])
        out.append(b)
    return out


def scrub_for_capture(payload: Any) -> Any:
    """Deep-copy messages / Responses input_items with data: image URLs omitted."""
    data = copy.deepcopy(payload)
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            if "content" in item:
                item["content"] = _scrub_content_blocks(item.get("content"))
            # Responses function_call_output.output is already formatted text;
            # still scrub accidental data URLs in nested shapes.
            if isinstance(item.get("output"), str) and "data:image/" in item["output"]:
                item["output"] = _scrub_content_blocks(item["output"])
        return data
    if isinstance(data, dict) and "content" in data:
        data["content"] = _scrub_content_blocks(data.get("content"))
    return data
