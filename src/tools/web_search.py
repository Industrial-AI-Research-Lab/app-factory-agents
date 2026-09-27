from __future__ import annotations

import re
from typing import Any, Dict, List
from urllib.parse import quote_plus

import httpx


async def web_search(query: str, max_results: int = 5, timeout_seconds: float = 15.0) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        return {"status": "error", "error": "Missing query", "results": []}

    url = f"https://duckduckgo.com/html/?q={quote_plus(q)}"

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "AppFactory-websearch/0.1"})
            html = resp.text or ""
    except Exception as e:
        return {"status": "error", "error": str(e), "results": []}

    results: List[Dict[str, str]] = []

    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.IGNORECASE | re.DOTALL):
        href = m.group(1)
        title_html = m.group(2)
        title = _strip_html(title_html)
        if href and title:
            results.append({"title": title, "url": href})
        if len(results) >= max_results:
            break

    return {"status": "success", "query": q, "results": results}


def _strip_html(s: str) -> str:
    txt = re.sub(r"<[^>]+>", "", s or "")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt
