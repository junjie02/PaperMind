"""PDF text extraction for RAG indexing.

Extracts text directly from PDFs using PyMuPDF — no intermediate MD files.
Non-arXiv papers that already have .md artifacts are read as-is.
"""

import logging
import sqlite3
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)


def extract_all(run_dir: Path) -> dict[str, str]:
    """Extract text from all paper artifacts.

    Returns a dict mapping paper_id → plain text content.
    """
    db_path = run_dir / "papers.db"

    if not db_path.exists():
        logger.error("papers.db not found: %s", db_path)
        return {}

    papers = _load_paper_artifacts(db_path)
    results: dict[str, str] = {}

    for paper_id, artifact_rel_path, title in papers:
        if not artifact_rel_path:
            logger.debug("No artifact for %s, skipping", paper_id)
            continue

        artifact_path = run_dir / artifact_rel_path
        if not artifact_path.exists():
            logger.warning("Artifact missing: %s", artifact_path)
            continue

        try:
            if artifact_path.suffix == ".pdf":
                text = _extract_pdf_text(artifact_path)
            elif artifact_path.suffix == ".md":
                text = artifact_path.read_text(encoding="utf-8")
            else:
                logger.debug("Unknown artifact type: %s", artifact_path.suffix)
                continue

            if text.strip():
                results[paper_id] = text
                logger.info("Extracted: %s (%d chars)", title[:50], len(text))
            else:
                logger.warning("Empty text: %s", paper_id)

        except Exception as e:
            logger.error("Failed to extract %s: %s", paper_id, e)

    logger.info("Text extraction complete: %d/%d papers", len(results), len(papers))
    return results


def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = pymupdf.open(str(pdf_path))
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def _load_paper_artifacts(db_path: Path) -> list[tuple[str, str | None, str]]:
    """Load paper_id, artifact_rel_path, title from SQLite."""
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT paper_id, artifact_rel_path, title FROM papers"
        ).fetchall()
        return rows
    finally:
        conn.close()
