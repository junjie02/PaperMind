import logging

from deepresearch.arxiv_client import ArxivClient
from deepresearch.dedup import DedupEngine
from deepresearch.models import PaperRecord

logger = logging.getLogger(__name__)


class ArxivSearchEngine:
    """Deterministic arXiv-backed search path.

    This is the default runtime path because it is observable, cheap, and does
    not require another agent process to browse on behalf of this project.
    """

    def __init__(self, proxy: str = "", timeout: float = 10.0, retries: int = 2):
        self.client = ArxivClient(proxy=proxy, timeout=timeout, retries=retries)
        self.dedup = DedupEngine()

    async def close(self) -> None:
        await self.client.close()

    async def search(
        self,
        question: str,
        target_count: int,
        exclude_ids: set[str],
    ) -> list[PaperRecord]:
        queries = self._build_queries(question)
        papers: dict[str, PaperRecord] = {}
        seen_ids = set(exclude_ids)

        per_query = max(10, min(50, target_count * 2))
        logger.info("使用 arXiv API 直接检索: %s", queries)

        for query in queries:
            if len(papers) >= target_count:
                break

            results = await self.client.search(query, max_results=per_query)
            logger.info("arXiv query=%r 返回 %d 条", query, len(results))

            for item in results:
                arxiv_id = self.dedup.normalize(item.get("arxiv_id", ""))
                if not arxiv_id or arxiv_id in seen_ids:
                    continue

                paper = self._to_record(item)
                if paper is None:
                    continue

                paper.arxiv_id = arxiv_id
                paper.search_round = 1
                paper.worker_id = "arxiv-api"
                papers[arxiv_id] = paper
                seen_ids.add(arxiv_id)

                if len(papers) >= target_count:
                    break

        return list(papers.values())

    def _build_queries(self, question: str) -> list[str]:
        cleaned = " ".join(question.split())
        queries = [
            cleaned,
            f"{cleaned} survey",
            f"{cleaned} review",
            f"{cleaned} benchmark",
            f"{cleaned} method",
        ]
        unique_queries = []
        seen = set()
        for query in queries:
            normalized = query.lower()
            if normalized not in seen:
                unique_queries.append(query)
                seen.add(normalized)
        return unique_queries

    def _to_record(self, item: dict) -> PaperRecord | None:
        try:
            return PaperRecord(
                arxiv_id=item["arxiv_id"],
                title=item.get("title", ""),
                authors=item.get("authors", []),
                abstract=item.get("abstract", ""),
                overview=self._make_overview(item),
                published_at=item.get("published_at"),
                categories=item.get("categories", []),
                primary_class=item.get("primary_class"),
                bibtex=item.get("bibtex", ""),
                abs_url=item.get("abs_url", ""),
                pdf_url=item.get("pdf_url", ""),
                relevance_score=3,
            )
        except Exception as exc:
            logger.warning("跳过无法解析的 arXiv 结果: %s | %s", exc, item)
            return None

    def _make_overview(self, item: dict) -> str:
        abstract = item.get("abstract", "")
        if not abstract:
            return "该论文来自 arXiv 检索结果；摘要暂缺。"
        first_sentence = abstract.split(". ", 1)[0].strip()
        if len(first_sentence) > 240:
            first_sentence = first_sentence[:237].rstrip() + "..."
        return f"该论文来自 arXiv 检索结果。摘要要点：{first_sentence}."
