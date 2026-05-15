"""FAISS-based retriever with dual-path search.

Path A: direct vector search using section title + outline text.
Path B: LLM-expanded keyword search — DeepSeek generates related phrases,
        each phrase is searched independently, results are merged.
"""

import asyncio
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from .chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class Section:
    title: str
    outline_text: str  # full text of this section in the outline (incl. subsections)
    order: int         # original position in outline (for merge ordering)


class Retriever:
    def __init__(
        self,
        run_dir: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_client: AsyncOpenAI | None = None,
        llm_model: str = "",
    ):
        index_path = run_dir / "faiss.index"
        chunks_path = run_dir / "chunks.pkl"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found in {run_dir}. Run 'write index --in {run_dir}' first."
            )

        logger.info("Loading FAISS index: %s", index_path)
        self._index = faiss.read_index(str(index_path))

        with open(chunks_path, "rb") as f:
            self._chunks: list[Chunk] = pickle.load(f)

        logger.info("Loading embedding model: %s", embedding_model)
        self._model = SentenceTransformer(embedding_model)
        self._llm_client = llm_client
        self._llm_model = llm_model

    def search(self, query: str, top_k: int = 6) -> list[Chunk]:
        """Single-path vector search. Returns top_k chunks sorted by score."""
        vec = self._model.encode([query], show_progress_bar=False)
        vec = np.array(vec, dtype="float32")
        faiss.normalize_L2(vec)

        scores, indices = self._index.search(vec, min(top_k, len(self._chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((float(score), self._chunks[idx]))
        results.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in results]

    def _search_vectors(self, vecs: np.ndarray, top_k: int) -> list[list[Chunk]]:
        """Batch search multiple query vectors. Returns one result list per vector."""
        faiss.normalize_L2(vecs)
        scores_batch, indices_batch = self._index.search(vecs, min(top_k, len(self._chunks)))
        results = []
        for scores, indices in zip(scores_batch, indices_batch):
            row = []
            for score, idx in zip(scores, indices):
                if idx >= 0:
                    row.append((float(score), self._chunks[idx]))
            row.sort(key=lambda x: x[0], reverse=True)
            results.append([c for _, c in row])
        return results

    async def dual_search(self, section: Section, top_k: int = 6) -> list[Chunk]:
        """Two-path search: direct match + LLM keyword expansion. Returns merged chunks."""
        path_a_query = f"{section.title} {section.outline_text[:300]}"

        loop = asyncio.get_running_loop()
        path_a_task = loop.run_in_executor(None, self.search, path_a_query, top_k)
        path_b_task = self._keyword_search(section, top_k)

        path_a_chunks, path_b_chunks = await asyncio.gather(path_a_task, path_b_task)

        return _merge_chunks(path_a_chunks, path_b_chunks, top_k * 2)

    async def _keyword_search(self, section: Section, top_k: int) -> list[Chunk]:
        """Path B: ask LLM for related search phrases, batch-encode, then search."""
        if not self._llm_client or not self._llm_model:
            return []

        prompt = (
            f"给定文献综述章节：\"{section.title}\"\n"
            f"大纲描述：{section.outline_text[:300]}\n\n"
            "列出 5-8 个用于检索相关论文的关键短语（英文或中文均可），每行一个，不要编号。"
        )
        try:
            resp = await self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            phrases = [
                line.strip()
                for line in resp.choices[0].message.content.strip().splitlines()
                if line.strip()
            ]
        except Exception as e:
            logger.warning("Keyword expansion failed: %s", e)
            return []

        if not phrases:
            return []

        logger.debug("Path B phrases for '%s': %s", section.title, phrases)

        loop = asyncio.get_running_loop()
        vecs = await loop.run_in_executor(
            None,
            lambda: np.array(
                self._model.encode(phrases, show_progress_bar=False), dtype="float32"
            ),
        )
        per_phrase = self._search_vectors(vecs, top_k)

        merged: dict[str, tuple[float, Chunk]] = {}
        for phrase_chunks in per_phrase:
            for i, chunk in enumerate(phrase_chunks):
                key = f"{chunk.paper_id}:{chunk.chunk_index}"
                score = 1.0 - i / len(phrase_chunks)
                if key not in merged or merged[key][0] < score:
                    merged[key] = (score, chunk)

        sorted_chunks = sorted(merged.values(), key=lambda x: x[0], reverse=True)
        return [c for _, c in sorted_chunks[:top_k]]


def _merge_chunks(
    path_a: list[Chunk],
    path_b: list[Chunk],
    max_total: int,
) -> list[Chunk]:
    """Merge two chunk lists, dedup by (paper_id, chunk_index), path_a takes priority."""
    seen: set[str] = set()
    merged: list[Chunk] = []

    for chunk in path_a:
        key = f"{chunk.paper_id}:{chunk.chunk_index}"
        if key not in seen:
            seen.add(key)
            merged.append(chunk)

    for chunk in path_b:
        key = f"{chunk.paper_id}:{chunk.chunk_index}"
        if key not in seen:
            seen.add(key)
            merged.append(chunk)

    return merged[:max_total]


def format_chunks_for_llm(chunks: list[Chunk]) -> str:
    """Format chunks as [paper_title | section_title]\ntext blocks."""
    parts = []
    for chunk in chunks:
        parts.append(f"[{chunk.paper_title} | {chunk.section_title}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)
