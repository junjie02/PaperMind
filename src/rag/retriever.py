"""FAISS-based retriever with dual-path search.

Path A: direct vector search using section title + outline text.
Path B: LLM-expanded keyword search — LLM generates related phrases,
        each phrase is searched independently, results are merged.
"""

from __future__ import annotations

import asyncio
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from rag.chunker import Chunk

logger = logging.getLogger(__name__)

# Cache embedding model globally to avoid repeated loading
_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(embedding_model: str) -> SentenceTransformer:
    if embedding_model not in _model_cache:
        logger.info("Loading embedding model: %s", embedding_model)
        _model_cache[embedding_model] = SentenceTransformer(embedding_model)
    return _model_cache[embedding_model]

_KEYWORD_PROMPT = ChatPromptTemplate.from_messages([("human", """\
给定文献综述章节："{title}"
大纲描述：{outline}

列出 5-8 个用于检索相关论文的关键短语（英文或中文均可），每行一个，不要编号。""")])


@dataclass
class Section:
    title: str
    outline_text: str
    order: int


class Retriever:
    def __init__(
        self,
        run_dir: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm: ChatOpenAI | None = None,
    ):
        index_path = run_dir / "faiss.index"
        chunks_path = run_dir / "chunks.pkl"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"FAISS index not found in {run_dir}. Build the index first.")

        logger.info("Loading FAISS index: %s", index_path)
        self._index = faiss.read_index(str(index_path))

        with open(chunks_path, "rb") as f:
            self._chunks: list[Chunk] = pickle.load(f)

        self._model = _get_model(embedding_model)
        self._keyword_chain = (_KEYWORD_PROMPT | llm | StrOutputParser()) if llm else None

    def search(self, query: str, top_k: int = 6) -> list[Chunk]:
        vec = self._model.encode([query], show_progress_bar=False)
        vec = np.array(vec, dtype="float32")
        faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, min(top_k, len(self._chunks)))
        results = [(float(s), self._chunks[i]) for s, i in zip(scores[0], indices[0]) if i >= 0]
        return [c for _, c in sorted(results, key=lambda x: x[0], reverse=True)]

    def _search_vectors(self, vecs: np.ndarray, top_k: int) -> list[list[Chunk]]:
        faiss.normalize_L2(vecs)
        scores_batch, indices_batch = self._index.search(vecs, min(top_k, len(self._chunks)))
        results = []
        for scores, indices in zip(scores_batch, indices_batch):
            row = sorted(
                [(float(s), self._chunks[i]) for s, i in zip(scores, indices) if i >= 0],
                key=lambda x: x[0], reverse=True,
            )
            results.append([c for _, c in row])
        return results

    async def dual_search(self, section: Section, top_k: int = 6) -> list[Chunk]:
        path_a_query = f"{section.title} {section.outline_text[:300]}"
        loop = asyncio.get_running_loop()
        path_a_task = loop.run_in_executor(None, self.search, path_a_query, top_k)
        path_b_task = self._keyword_search(section, top_k)
        path_a_chunks, path_b_chunks = await asyncio.gather(path_a_task, path_b_task)
        return _merge_chunks(path_a_chunks, path_b_chunks, top_k * 2)

    async def _keyword_search(self, section: Section, top_k: int) -> list[Chunk]:
        if not self._keyword_chain:
            return []
        try:
            raw = await self._keyword_chain.ainvoke({
                "title": section.title,
                "outline": section.outline_text[:300],
            })
            phrases = [l.strip() for l in raw.splitlines() if l.strip()]
        except Exception as e:
            logger.warning("Keyword expansion failed: %s", e)
            return []

        if not phrases:
            return []

        loop = asyncio.get_running_loop()
        vecs = await loop.run_in_executor(
            None,
            lambda: np.array(self._model.encode(phrases, show_progress_bar=False), dtype="float32"),
        )
        per_phrase = self._search_vectors(vecs, top_k)

        merged: dict[str, tuple[float, Chunk]] = {}
        for phrase_chunks in per_phrase:
            for i, chunk in enumerate(phrase_chunks):
                key = f"{chunk.paper_id}:{chunk.chunk_index}"
                score = 1.0 - i / len(phrase_chunks)
                if key not in merged or merged[key][0] < score:
                    merged[key] = (score, chunk)

        return [c for _, c in sorted(merged.values(), key=lambda x: x[0], reverse=True)[:top_k]]


def _merge_chunks(path_a: list[Chunk], path_b: list[Chunk], max_total: int) -> list[Chunk]:
    seen: set[str] = set()
    merged: list[Chunk] = []
    for chunk in path_a + path_b:
        key = f"{chunk.paper_id}:{chunk.chunk_index}"
        if key not in seen:
            seen.add(key)
            merged.append(chunk)
    return merged[:max_total]


def format_chunks_for_llm(chunks: list[Chunk]) -> str:
    return "\n\n---\n\n".join(
        f"[{c.paper_title} | {c.section_title}]\n{c.text}" for c in chunks
    )
