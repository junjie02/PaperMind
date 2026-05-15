"""FAISS-based embedding and indexing for paper chunks.

Saves faiss.index and chunks.pkl to the run directory.
No external services required.
"""

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .chunker import Chunk

logger = logging.getLogger(__name__)


def build_index(
    run_dir: Path,
    chunks: list[Chunk],
    embedding_model: str = "all-MiniLM-L6-v2",
    **_kwargs,  # absorb legacy qdrant_url kwarg if passed
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
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    logger.info("FAISS index saved: %s (%d vectors, dim=%d)", index_path, len(chunks), dim)
    return str(index_path)


def build_index_from_run(run_dir: Path, embedding_model: str = "all-MiniLM-L6-v2") -> str:
    """Extract PDFs, chunk, embed, and save FAISS index for a run directory."""
    from .chunker import chunk_all
    from .db import load_paper_titles
    from .pdf_converter import extract_all

    papers_text = extract_all(run_dir)
    if not papers_text:
        raise ValueError(f"No paper text found in {run_dir}")

    paper_titles = load_paper_titles(run_dir / "papers.db")
    chunks = chunk_all(papers_text, paper_titles)
    logger.info("Chunked %d papers into %d chunks", len(papers_text), len(chunks))
    return build_index(run_dir, chunks, embedding_model=embedding_model)
