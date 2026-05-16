"""FAISS-based embedding and indexing for paper chunks."""

import logging
import pickle
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.chunker import Chunk

logger = logging.getLogger(__name__)


def build_index(
    run_dir: Path,
    chunks: list[Chunk],
    embedding_model: str = "all-MiniLM-L6-v2",
) -> str:
    """Embed chunks and save FAISS index to run_dir. Returns index file path."""
    index_path = run_dir / "faiss.index"
    chunks_path = run_dir / "chunks.pkl"

    logger.info("Loading embedding model: %s", embedding_model)
    model = SentenceTransformer(embedding_model)

    texts = [c.text for c in chunks]
    logger.info("Embedding %d chunks...", len(texts))
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    logger.info("FAISS index saved: %s (%d vectors, dim=%d)", index_path, len(chunks), dim)
    return str(index_path)


def _fallback_abstracts(run_dir: Path) -> dict[str, str]:
    """Use abstract + overview from papers.db as fallback text for indexing."""
    db_path = run_dir / "papers.db"
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT paper_id, title, abstract, overview FROM papers"
        ).fetchall()
    finally:
        conn.close()
    results: dict[str, str] = {}
    for paper_id, title, abstract, overview in rows:
        parts = [f"# {title}"]
        if abstract:
            parts.append(f"\n## Abstract\n{abstract}")
        if overview:
            parts.append(f"\n## Overview\n{overview}")
        text = "\n".join(parts)
        if len(text) > 50:
            results[paper_id] = text
    logger.info("Fallback abstracts: %d papers with text", len(results))
    return results


def build_index_from_run(run_dir: Path, embedding_model: str = "all-MiniLM-L6-v2") -> str:
    """Extract PDFs, chunk, embed, and save FAISS index for a run directory."""
    from rag.chunker import chunk_all
    from rag.db import load_paper_titles
    from rag.pdf_extractor import extract_all

    papers_text = extract_all(run_dir)
    if not papers_text:
        logger.warning("No paper text extracted from PDFs in %s, falling back to abstracts", run_dir)
        papers_text = _fallback_abstracts(run_dir)
    if not papers_text:
        raise ValueError(f"No paper text found in {run_dir}")

    paper_titles = load_paper_titles(run_dir / "papers.db")
    chunks = chunk_all(papers_text, paper_titles)
    logger.info("Chunked %d papers into %d chunks", len(papers_text), len(chunks))
    return build_index(run_dir, chunks, embedding_model=embedding_model)
