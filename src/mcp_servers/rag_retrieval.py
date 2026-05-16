"""RAG Retrieval MCP Server.

Wraps the FAISS-based Retriever as FastMCP tools.
Retriever instances are cached per run_dir to avoid reloading the index.

Import tool functions directly in Sub-Agents:
    from mcp_servers.rag_retrieval import rag_search, rag_dual_search
"""

import logging
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP

from rag.retriever import Retriever, Section, format_chunks_for_llm
from shared.config import Config

logger = logging.getLogger(__name__)

mcp = FastMCP("rag-retrieval")

_config = Config()
_retriever_cache: dict[str, Retriever] = {}


def _get_retriever(run_dir: str) -> Retriever:
    if run_dir not in _retriever_cache:
        _retriever_cache[run_dir] = Retriever(
            Path(run_dir),
            embedding_model=_config.embedding_model,
        )
    return _retriever_cache[run_dir]


@mcp.tool()
def rag_search(
    run_dir: Annotated[str, "Path to the run directory containing faiss.index"],
    query: Annotated[str, "Search query"],
    top_k: Annotated[int, "Number of chunks to return"] = 6,
) -> str:
    """Single-path vector search. Returns formatted paper chunk text."""
    retriever = _get_retriever(run_dir)
    chunks = retriever.search(query, top_k=top_k)
    return format_chunks_for_llm(chunks)


@mcp.tool()
async def rag_dual_search(
    run_dir: Annotated[str, "Path to the run directory containing faiss.index"],
    section_title: Annotated[str, "Section title"],
    section_outline: Annotated[str, "Section outline text"],
    top_k: Annotated[int, "Number of chunks to return"] = 6,
) -> str:
    """Dual-path search (vector + LLM keyword expansion). Returns formatted chunk text."""
    retriever = _get_retriever(run_dir)
    section = Section(title=section_title, outline_text=section_outline, order=0)
    chunks = await retriever.dual_search(section, top_k=top_k)
    return format_chunks_for_llm(chunks)


@mcp.tool()
def rag_invalidate_cache(
    run_dir: Annotated[str, "Run directory to evict from cache"],
) -> str:
    """Evict a Retriever from the cache (call after rebuilding the index)."""
    _retriever_cache.pop(run_dir, None)
    return f"cache cleared for {run_dir}"


if __name__ == "__main__":
    mcp.run()
