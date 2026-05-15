"""ReviewOrchestrator: parse outline, dispatch concurrent SectionWriters, merge results."""

import asyncio
import logging
import re
from pathlib import Path

from openai import AsyncOpenAI

from deepresearch.config import Config

from .retriever import Retriever, Section
from .section_writer import SectionWriter

logger = logging.getLogger(__name__)

_SECTION_RE = re.compile(r"^(##\s+.+)$", re.MULTILINE)


def parse_outline_sections(outline: str) -> list[Section]:
    """Split outline into ## sections. Each Section gets its full outline_text."""
    matches = list(_SECTION_RE.finditer(outline))
    if not matches:
        title_match = re.search(r"^#\s+(.+)$", outline, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "综述正文"
        return [Section(title=title, outline_text=outline, order=0)]

    sections: list[Section] = []
    for i, match in enumerate(matches):
        title = match.group(1).lstrip("#").strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(outline)
        sections.append(Section(title=title, outline_text=outline[start:end].strip(), order=i))

    return sections


def _extract_title(outline: str) -> str:
    m = re.search(r"^#\s+(.+)$", outline, re.MULTILINE)
    return m.group(1).strip() if m else "文献综述"


def _build_references_section(section_texts: list[str]) -> str:
    all_citations: dict[str, int] = {}
    for text in section_texts:
        for cite in re.findall(r"\[[^\[\]]+,\s*\d{4}[^\[\]]*\]", text):
            all_citations[cite] = all_citations.get(cite, 0) + 1

    if not all_citations:
        return ""

    lines = ["## 参考文献", ""]
    for cite in sorted(all_citations.keys()):
        lines.append(f"- {cite}")
    return "\n".join(lines)


def _merge_sections(sections: list[Section], results: list[str], outline: str) -> str:
    title = _extract_title(outline)
    parts = [f"# {title}", ""]

    section_texts = []
    for _, text in sorted(zip(sections, results), key=lambda x: x[0].order):
        if text.strip():
            parts.append(text.strip())
            parts.append("")
            section_texts.append(text)

    refs = _build_references_section(section_texts)
    if refs:
        parts.append(refs)

    return "\n".join(parts)


class ReviewOrchestrator:
    def __init__(self, config: Config, run_dir: Path):
        self.config = config
        self.run_dir = run_dir

    async def write_review(self, outline: str, n_papers: int) -> str:
        sections = parse_outline_sections(outline)
        logger.info("Parsed %d sections from outline", len(sections))

        client = AsyncOpenAI(
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
        )
        retriever = Retriever(
            self.run_dir,
            embedding_model=self.config.embedding_model,
            llm_client=client,
            llm_model=self.config.llm_model,
        )
        semaphore = asyncio.Semaphore(self.config.writer_concurrency)

        logger.info(
            "Starting %d concurrent writers (concurrency=%d, max_retries=%d)",
            len(sections),
            self.config.writer_concurrency,
            self.config.writer_max_retries,
        )

        writers = [SectionWriter(self.config, retriever, semaphore, client) for _ in sections]
        results = await asyncio.gather(*[
            writer.write_section(section)
            for writer, section in zip(writers, sections)
        ])

        return _merge_sections(sections, list(results), outline)
