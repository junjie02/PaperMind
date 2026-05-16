"""Section-aware chunking for paper markdown content.

Splits papers by heading boundaries, with fixed-size fallback for large sections.
"""

import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
MAX_CHUNK_CHARS = 6000  # ~1500 tokens
SPLIT_CHUNK_CHARS = 4000  # ~1000 tokens
OVERLAP_CHARS = 800  # ~200 tokens


@dataclass
class Chunk:
    paper_id: str
    paper_title: str
    section_title: str
    text: str
    chunk_index: int


def chunk_paper(paper_id: str, paper_title: str, markdown_text: str) -> list[Chunk]:
    """Split a paper's markdown into chunks by section boundaries."""
    sections = _split_by_headings(markdown_text)
    chunks: list[Chunk] = []
    idx = 0

    for section_title, section_text in sections:
        if len(section_text.strip()) < 50:
            continue

        if len(section_text) <= MAX_CHUNK_CHARS:
            chunks.append(Chunk(
                paper_id=paper_id,
                paper_title=paper_title,
                section_title=section_title,
                text=section_text.strip(),
                chunk_index=idx,
            ))
            idx += 1
        else:
            sub_chunks = _split_fixed_size(section_text, SPLIT_CHUNK_CHARS, OVERLAP_CHARS)
            for i, sub_text in enumerate(sub_chunks):
                chunks.append(Chunk(
                    paper_id=paper_id,
                    paper_title=paper_title,
                    section_title=f"{section_title} (part {i + 1})",
                    text=sub_text.strip(),
                    chunk_index=idx,
                ))
                idx += 1

    return chunks


def chunk_all(papers_md: dict[str, str], paper_titles: dict[str, str]) -> list[Chunk]:
    """Chunk all papers. paper_titles maps paper_id → title."""
    all_chunks: list[Chunk] = []
    for paper_id, md_text in papers_md.items():
        title = paper_titles.get(paper_id, paper_id)
        paper_chunks = chunk_paper(paper_id, title, md_text)
        all_chunks.extend(paper_chunks)
    return all_chunks


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split markdown into (section_title, section_text) pairs."""
    matches = list(HEADING_RE.finditer(text))

    if not matches:
        return [("Full Text", text)]

    sections: list[tuple[str, str]] = []

    if matches[0].start() > 0:
        preamble = text[:matches[0].start()]
        if preamble.strip():
            sections.append(("Preamble", preamble))

    for i, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end]
        sections.append((title, section_text))

    return sections


def _split_fixed_size(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping fixed-size chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break

        split_point = text.rfind("\n", start + chunk_size // 2, end)
        if split_point == -1:
            split_point = end

        chunks.append(text[start:split_point])
        start = split_point - overlap

    return chunks
