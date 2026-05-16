"""DDG Search MCP Server.

Wraps the `ddgs` package as a FastMCP tool server.
Proxy is read from the HTTP_PROXY environment variable.

Run standalone (stdio mode):
    python -m mcp_servers.ddg_search

Or import tool functions directly in Sub-Agents:
    from mcp_servers.ddg_search import ddg_search, ddg_search_batch, fetch_page
"""

import asyncio
import logging
import os
from typing import Annotated

import httpx
import trafilatura
from ddgs import DDGS
from fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("ddg-search")

_PROXY = os.getenv("HTTP_PROXY", "") or None

_HTTP_CLIENT = httpx.AsyncClient(
    proxy=_PROXY,
    timeout=20,
    follow_redirects=True,
    headers={"User-Agent": "Mozilla/5.0 (compatible; PaperMind/0.3)"},
)


def _search_sync(query: str, max_results: int) -> list[dict]:
    try:
        with DDGS(proxy=_PROXY) as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        logger.warning("DDG search failed for %r: %s", query, e)
        return []


@mcp.tool()
async def ddg_search(
    query: Annotated[str, "Search query string"],
    max_results: Annotated[int, "Maximum number of results to return"] = 10,
) -> list[dict]:
    """Search DuckDuckGo and return a list of {title, href, body} dicts."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _search_sync, query, max_results)


@mcp.tool()
async def ddg_search_batch(
    queries: Annotated[list[str], "List of search query strings"],
    max_results_each: Annotated[int, "Maximum results per query"] = 5,
) -> dict[str, list[dict]]:
    """Run multiple DDG searches concurrently. Returns {query: results} dict."""
    tasks = [ddg_search(q, max_results_each) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return {
        q: (r if not isinstance(r, Exception) else [])
        for q, r in zip(queries, results)
    }


@mcp.tool()
async def fetch_page(
    url: Annotated[str, "URL to fetch"],
    max_chars: Annotated[int, "Maximum characters to return (0 = no limit)"] = 8000,
) -> dict:
    """Fetch a web page and extract its main text content.

    Returns {url, title, text, error}. Uses trafilatura for content extraction
    (strips navigation, ads, footers — returns the article body only).
    """
    try:
        resp = await _HTTP_CLIENT.get(url)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        logger.warning("fetch_page failed to download %s: %s", url, e)
        return {"url": url, "title": "", "text": "", "error": str(e)}

    # trafilatura extracts main body text, include_comments=False strips comment sections
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    ) or ""

    # Extract title separately
    meta = trafilatura.extract_metadata(html)
    title = meta.title if meta and meta.title else ""

    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"

    logger.debug("fetch_page %s: %d chars extracted", url, len(text))
    return {"url": url, "title": title, "text": text, "error": ""}


@mcp.tool()
async def fetch_pages_batch(
    urls: Annotated[list[str], "List of URLs to fetch"],
    max_chars_each: Annotated[int, "Maximum characters per page"] = 4000,
) -> list[dict]:
    """Fetch multiple pages concurrently. Returns list of {url, title, text, error}."""
    tasks = [fetch_page(url, max_chars_each) for url in urls]
    return list(await asyncio.gather(*tasks))


if __name__ == "__main__":
    mcp.run()
