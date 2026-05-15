import asyncio
import logging
from collections import deque
from pathlib import Path

import httpx

from deepresearch.config import Config
from deepresearch.database import Database
from deepresearch.models import PaperRecord
from deepresearch.text_utils import slugify

logger = logging.getLogger(__name__)


class ArtifactWriter:
    """Per-paper artifact writer.

    arXiv papers → PDF downloaded via streaming GET, with deque-based retry
    (failed downloads go to the back of the queue, up to ``max_attempts`` total
    tries per paper).

    Non-arXiv papers → a Markdown file rendered from the metadata is written
    instead, since we cannot reliably fetch a canonical PDF.
    """

    def __init__(self, artifact_dir: Path, config: Config):
        self.artifact_dir = Path(artifact_dir)
        self.max_attempts = max(config.pdf_max_attempts, 1)
        self.retry_sleep = max(config.pdf_retry_sleep, 0.0)
        extra = {"proxy": config.http_proxy} if config.http_proxy else {}
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.pdf_timeout, connect=min(config.pdf_timeout, 10.0)),
            follow_redirects=True,
            headers={"User-Agent": "PaperMind/0.1 (mailto:research@example.com)"},
            **extra,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def write_all(
        self, papers: list[PaperRecord], db: Database
    ) -> tuple[int, int, int]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        pdf_queue: deque[PaperRecord] = deque(p for p in papers if p.arxiv_id)
        md_papers = [p for p in papers if not p.arxiv_id]
        attempts: dict[str, int] = {}

        pdf_success = 0
        md_success = 0
        failed = 0

        logger.info(
            "准备产出：arxiv PDF %d 篇，非 arxiv MD %d 篇 → %s",
            len(pdf_queue), len(md_papers), self.artifact_dir,
        )

        for paper in md_papers:
            try:
                rel_path = await asyncio.to_thread(self._write_markdown, paper)
            except Exception as exc:
                logger.error("[%s] 写入 MD 失败: %s", paper.paper_id, exc)
                failed += 1
                continue

            paper.artifact_rel_path = rel_path
            await self._record_artifact_path(db, paper.paper_id, rel_path)
            md_success += 1
            logger.info("[%s] MD 已写 → %s", paper.paper_id, rel_path)

        while pdf_queue:
            paper = pdf_queue.popleft()
            pid = paper.paper_id

            if paper.artifact_rel_path:
                existing = self.artifact_dir.parent / paper.artifact_rel_path
                if existing.exists() and existing.stat().st_size > 0:
                    logger.info("跳过已存在: %s", paper.artifact_rel_path)
                    pdf_success += 1
                    continue

            pdf_url = paper.pdf_url or f"https://arxiv.org/pdf/{paper.arxiv_id}"
            attempts[pid] = attempts.get(pid, 0) + 1
            attempt_no = attempts[pid]

            try:
                rel_path = await self._download_pdf(paper, pdf_url)
            except Exception as exc:
                if attempt_no >= self.max_attempts:
                    logger.error(
                        "[%s] 下载失败放弃 (已尝试 %d 次): %s",
                        pid, attempt_no, exc,
                    )
                    failed += 1
                    continue

                logger.warning(
                    "[%s] 下载失败 (尝试 %d/%d): %s — 放回队尾",
                    pid, attempt_no, self.max_attempts, exc,
                )
                pdf_queue.append(paper)
                if self._only_failed_papers_remain(pdf_queue, attempts):
                    await asyncio.sleep(self.retry_sleep)
                continue

            paper.artifact_rel_path = rel_path
            await self._record_artifact_path(db, pid, rel_path)
            pdf_success += 1
            logger.info("[%s] PDF 下载完成 → %s", pid, rel_path)

        return pdf_success, md_success, failed

    @staticmethod
    def _only_failed_papers_remain(queue: deque[PaperRecord],
                                   attempts: dict[str, int]) -> bool:
        """True iff every paper still in queue has already failed at least once.

        Used to decide whether to back off before the next pop: if the queue
        only contains retries, hitting the same flaky URL immediately is wasted.
        """
        return all(attempts.get(p.paper_id, 0) > 0 for p in queue)

    @staticmethod
    async def _record_artifact_path(db: Database, paper_id: str, rel_path: str) -> None:
        try:
            await db.update_artifact_path(paper_id, rel_path)
        except Exception as exc:
            logger.error("[%s] 更新 artifact_rel_path 失败: %s", paper_id, exc)

    async def _download_pdf(self, paper: PaperRecord, pdf_url: str) -> str:
        target = self.artifact_dir / f"{paper.arxiv_id}.pdf"
        tmp = target.with_suffix(".pdf.tmp")

        async with self.client.stream("GET", pdf_url) as response:
            response.raise_for_status()

            def _open_tmp():
                return open(tmp, "wb")

            f = await asyncio.to_thread(_open_tmp)
            try:
                async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                    if chunk:
                        await asyncio.to_thread(f.write, chunk)
            finally:
                await asyncio.to_thread(f.close)

        await asyncio.to_thread(tmp.rename, target)
        return f"{self.artifact_dir.name}/{target.name}"

    def _write_markdown(self, paper: PaperRecord) -> str:
        stem = slugify(f"{paper.source}-{paper.title}", max_len=80,
                       fallback="paper", preserve_case=True)
        target = self.artifact_dir / f"{stem}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._render_markdown(paper), encoding="utf-8")
        return f"{self.artifact_dir.name}/{target.name}"

    def _render_markdown(self, p: PaperRecord) -> str:
        lines = [f"# {p.title}", ""]
        if p.authors:
            lines.append(f"**Authors:** {', '.join(p.authors)}")
        if p.venue:
            lines.append(f"**Venue:** {p.venue}")
        if p.published_at:
            lines.append(f"**Published:** {p.published_at}")
        lines.append(f"**Source:** {p.source}")
        if p.source_url:
            lines.append(f"**Source URL:** <{p.source_url}>")
        if p.pdf_url:
            lines.append(f"**PDF URL:** <{p.pdf_url}>")
        if p.categories:
            lines.append(f"**Categories:** {', '.join(p.categories)}")
        lines.append(f"**Relevance:** {p.relevance_score}/5")
        if p.search_direction:
            lines.append(f"**Search Direction:** {p.search_direction}")
        lines.extend(["", "## Overview (中文)", "", p.overview or "_无_", ""])
        lines.extend(["## Abstract", "", p.abstract or "_no abstract_", ""])
        if p.bibtex:
            lines.extend(["## BibTeX", "", "```bibtex", p.bibtex.strip(), "```", ""])
        return "\n".join(lines)
