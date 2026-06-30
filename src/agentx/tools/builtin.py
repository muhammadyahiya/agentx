"""Built-in tools usable by agents. Currently: a keyless web search.

Uses the ``ddgs`` (DuckDuckGo) package if available; otherwise the tool returns
a friendly message instead of failing.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 5) -> str:
    """Run a DuckDuckGo text search and return a formatted string."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # older package name
        except ImportError:
            return "Web search unavailable (install `ddgs`)."
    try:
        lines = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "")
                body = r.get("body", "")
                url = r.get("href", r.get("url", ""))
                lines.append(f"- {title}\n  {body}\n  {url}")
        return "\n".join(lines) if lines else f"No results for '{query}'."
    except Exception as exc:  # noqa: BLE001
        logger.warning("web_search failed: %s", exc)
        return f"Web search error: {exc!r}"


def make_web_search_tool():
    """Return ``web_search`` wrapped as a LangChain ``@tool`` (lazy import)."""
    from langchain_core.tools import tool

    @tool
    def web_search_tool(query: str) -> str:
        """Search the public web for up-to-date information. Input a concise query."""
        return web_search(query)

    return web_search_tool
