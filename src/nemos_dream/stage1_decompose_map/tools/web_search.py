"""Web search fallback via Tavily. Used when dict + retriever both miss."""

from __future__ import annotations

import os


def _is_placeholder(key: str | None) -> bool:
    return not key or key.endswith("...") or key in {"tvly-...", "your-key-here"}


def _build_query(term: str, *, ref_type: str | None, context: str | None) -> str:
    type_part = f"{ref_type} " if ref_type else ""
    base = f'Korean {type_part}equivalent for "{term}"'
    if context:
        snippet = context[:150].replace('"', "'")
        base += f' used in context: "{snippet}"'
    return base


def search(
    query: str,
    max_results: int = 3,
    *,
    ref_type: str | None = None,
    context: str | None = None,
) -> list[dict]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if _is_placeholder(api_key):
        return [{
            "title": f"[web_search disabled] {query}",
            "content": "TAVILY_API_KEY not set; returning empty.",
            "url": "",
        }]
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        full_query = _build_query(query, ref_type=ref_type, context=context)
        res = client.search(
            query=full_query,
            search_depth="basic",
            max_results=max_results,
        )
        return [
            {
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "url": r.get("url", ""),
            }
            for r in res.get("results", [])
        ]
    except Exception as exc:  # noqa: BLE001 — tavily may raise many error types
        return [{
            "title": f"[web_search error: {type(exc).__name__}]",
            "content": str(exc)[:300],
            "url": "",
        }]
