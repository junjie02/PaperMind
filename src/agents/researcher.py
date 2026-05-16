"""Researcher Sub-Agent.

Single-prompt, memory-driven agent that iteratively searches DDG for papers,
fetches selected pages, extracts structured metadata, and stores results in SQLite.
Stops when target paper count is reached or no new papers are found.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import SimpleJsonOutputParser

from agents.base import SubAgentBase, parse_json_with_retry
from mcp_servers.ddg_search import ddg_search_batch, fetch_pages_batch
from shared.config import Config
from shared.database import Database
from shared.dedup import DedupEngine
from shared.llm import make_llm
from shared.models import AgentResult, AgentTask, PaperRecord

logger = logging.getLogger(__name__)

_PARSER = SimpleJsonOutputParser()

RESEARCHER_SYSTEM = """\
你是 PaperMind 文献收集专家，负责针对具体研究问题收集高质量学术论文。

## 你拥有的工具

- **DDG 搜索**：每轮可提交 1-3 个搜索词，系统返回每个词最多 3 条结果（title、url、snippet）
- **网页抓取**：从搜索结果中选择有价值的 URL，系统抓取正文内容返回给你

## 每轮工作流程

系统会给你研究问题、目标数量、已收集论文列表、上轮搜索结果和抓取内容，你需要输出本轮决策 JSON：

```json
{
  "queries": ["搜索词1", "搜索词2"],
  "fetch_urls": ["https://arxiv.org/abs/...", "https://..."],
  "papers": [
    {
      "title": "英文原标题",
      "authors": ["作者1", "作者2"],
      "abstract": "摘要，没有则空字符串",
      "overview": "用中文写的2-3句话概述",
      "source": "arxiv 或 web",
      "source_url": "论文页面 URL",
      "venue": "发表会议/期刊，不确定留空字符串",
      "arxiv_id": "arXiv ID 如 2301.12345，不在 arXiv 则空字符串",
      "pdf_url": "PDF 直链，没有则空字符串",
      "abs_url": "摘要页 URL",
      "published_at": "YYYY-MM-DD，不确定留空字符串",
      "categories": [],
      "primary_class": "",
      "bibtex": "",
      "relevance_score": 4
    }
  ],
  "complete": false
}
```

字段说明：
- `queries`：本轮搜索词，英文为主，可加 "arxiv"/"survey"/"2024" 提高精度；第一轮必须非空
- `fetch_urls`：从搜索结果中选择值得抓取的 URL；优先选 arXiv 摘要页（用 /abs/ 而非 /pdf/）、顶会论文页、高质量综述；跳过博客、新闻、非论文页面、PDF 直链（无法提取文本）；若无值得抓取的则为空数组
- `papers`：从本轮抓取内容中提取的高质量论文，立即输出，不要等到最后；第一轮无抓取内容时为空数组
- `complete`：已收集论文数 >= 目标数量，或连续两轮无新论文时设为 true

## 论文质量标准

- 与研究问题直接相关
- 顶会顶刊（NeurIPS、ICML、ICLR、ACL、CVPR 等）或高引用量
- 不重复已有论文（对比 title）
- relevance_score：1-5，5 为最相关

## 行为规范

- 第一轮：直接规划搜索词，papers 为空数组（还没看到内容）
- 后续轮：先从上轮抓取内容提取论文，再规划新搜索词
- 已达到目标数量时立即设 complete=true，不要继续搜索
- 直接输出 JSON，不要任何前言或说明
"""


def _write_iteration_log(run_dir: str, agent_id: str, records: list[dict]) -> None:
    if not run_dir:
        return
    try:
        data_dir = Path(run_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        safe_id = agent_id.replace("/", "-").replace(":", "-")
        path = data_dir / f"{safe_id}_iterations.json"
        path.write_text(json.dumps({"iterations": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("[%s] failed to write iteration log: %s", agent_id, e)


def _format_search_snippets(results: dict[str, list[dict]], max_snippet: int = 200) -> str:
    parts = []
    for query, hits in results.items():
        parts.append(f"### 搜索词: {query}")
        for h in hits:
            title = h.get("title", "")
            url = h.get("href", "")
            body = (h.get("body", "") or "")[:max_snippet]
            parts.append(f"- [{title}]({url})\n  {body}")
    return "\n".join(parts)


def _format_fetch_results(pages: list[dict], max_chars: int = 2000) -> str:
    parts = []
    for p in pages:
        if p.get("error"):
            parts.append(f"### {p['url']}\n抓取失败: {p['error']}")
        else:
            text = (p.get("text", "") or "")[:max_chars]
            title = p.get("title", p["url"])
            parts.append(f"### {title}\n{text}")
    return "\n\n".join(parts)


def _build_paper_record(raw: dict, iteration: int, agent_id: str, dedup: DedupEngine) -> PaperRecord | None:
    try:
        title = (raw.get("title") or "").strip()
        if not title:
            return None
        arxiv_id_raw = (raw.get("arxiv_id") or "").strip()
        normalized_arxiv = dedup.normalize_arxiv_id(arxiv_id_raw) if arxiv_id_raw else None
        source = (raw.get("source") or ("arxiv" if normalized_arxiv else "web")).strip().lower()
        source_url = (raw.get("source_url") or raw.get("abs_url") or "").strip()
        if not source_url and normalized_arxiv:
            source_url = f"https://arxiv.org/abs/{normalized_arxiv}"
        paper_id = dedup.compute_paper_id(
            arxiv_id=normalized_arxiv, source=source, source_url=source_url, title=title,
        )
        if not paper_id:
            return None
        return PaperRecord(
            paper_id=paper_id, title=title,
            authors=raw.get("authors", []), abstract=raw.get("abstract", ""),
            overview=raw.get("overview", ""), source=source, source_url=source_url,
            venue=raw.get("venue") or None, arxiv_id=normalized_arxiv,
            search_direction="", published_at=raw.get("published_at") or None,
            categories=raw.get("categories", []), primary_class=raw.get("primary_class") or None,
            bibtex=raw.get("bibtex", ""), abs_url=raw.get("abs_url", "") or source_url,
            pdf_url=raw.get("pdf_url", ""), relevance_score=raw.get("relevance_score", 3),
            search_round=iteration, worker_id=agent_id,
        )
    except Exception as e:
        logger.warning("build_paper_record failed: %s | %s", e, raw.get("title", "?"))
        return None


async def _download_pdf(paper: PaperRecord, run_dir: str) -> str | None:
    """Download PDF if pdf_url is available. Returns relative path or None."""
    pdf_url = paper.pdf_url
    if not pdf_url or not run_dir:
        return None
    # For arXiv, ensure we use the PDF URL
    if paper.arxiv_id and not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{paper.arxiv_id}"
    try:
        pdfs_dir = Path(run_dir) / "pdfs"
        pdfs_dir.mkdir(parents=True, exist_ok=True)
        safe_name = paper.paper_id.replace("/", "-").replace(":", "-")[:60]
        filename = f"{safe_name}.pdf"
        filepath = pdfs_dir / filename

        proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
        async with httpx.AsyncClient(proxy=proxy, timeout=30, follow_redirects=True) as client:
            resp = await client.get(pdf_url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and not resp.content[:5] == b"%PDF-":
                logger.debug("[download] not a PDF: %s (%s)", pdf_url, content_type)
                return None
            filepath.write_bytes(resp.content)

        rel_path = f"pdfs/{filename}"
        logger.info("[download] saved: %s (%d KB)", rel_path, len(resp.content) // 1024)
        return rel_path
    except Exception as e:
        logger.debug("[download] failed for %s: %s", pdf_url, e)
        return None


class ResearcherAgent(SubAgentBase):
    def __init__(self, config: Config, agent_id: str = "researcher"):
        super().__init__(agent_id, timeout=config.researcher_timeout)
        self._config = config
        self._dedup = DedupEngine()
        self._llm = make_llm(config, temperature=0.3)

    def _build_messages(self, history: list[dict]) -> list:
        msgs: list = [SystemMessage(content=RESEARCHER_SYSTEM)]
        for m in history:
            if m["role"] == "human":
                msgs.append(HumanMessage(content=m["content"]))
            else:
                msgs.append(AIMessage(content=m["content"]))
        return msgs

    async def _run_impl(self, task: AgentTask) -> AgentResult:
        params = json.loads(task.instruction) if task.instruction.strip().startswith("{") else {}
        question: str = params.get("question", task.instruction)
        target_papers: int = params.get("target_papers", 5)

        db = Database(task.db_path)
        await db.initialize()
        existing_ids = await db.get_all_ids()

        collected: list[PaperRecord] = []
        collected_ids: set[str] = set(existing_ids)
        messages: list[dict] = []
        iteration_records: list[dict] = []
        no_new_count = 0

        search_snippet_text = ""
        fetch_text = ""

        for iteration in range(task.max_iterations):
            logger.info("[%s] iteration %d/%d (collected %d/%d)",
                        self.agent_id, iteration + 1, task.max_iterations, len(collected), target_papers)

            collected_titles = "\n".join(f"- {p.title}" for p in collected) or "（尚无）"

            tool_block = ""
            if search_snippet_text:
                tool_block += f"## 上轮搜索结果（title + url + snippet）\n{search_snippet_text}\n\n"
            if fetch_text:
                tool_block += f"## 上轮抓取内容\n{fetch_text}\n\n"

            human_content = f"""\
## 研究问题
{question}

## 目标文献数量：{target_papers} 篇（当前已收集：{len(collected)} 篇）

## 已收集论文（不要重复）
{collected_titles}

{tool_block if tool_block else "## 提示\n第一轮，尚无工具结果，请直接规划搜索词。\n"}
请输出本轮决策 JSON。"""

            messages.append({"role": "human", "content": human_content})

            try:
                response = await self._llm.ainvoke(self._build_messages(messages))
                ai_content = response.content
                messages.append({"role": "ai", "content": ai_content})
                decision, ai_content = await parse_json_with_retry(
                    self._llm, self._build_messages, messages, ai_content
                )
            except Exception as e:
                logger.error("[%s] LLM decision failed: %s", self.agent_id, e)
                break

            queries: list[str] = decision.get("queries", [])
            fetch_urls: list[str] = decision.get("fetch_urls", [])
            papers_raw: list[dict] = decision.get("papers", []) or []
            complete: bool = decision.get("complete", False)

            # Store papers from this round immediately
            new_count = 0
            new_papers_this_round: list[PaperRecord] = []
            for raw in papers_raw:
                paper = _build_paper_record(raw, iteration + 1, self.agent_id, self._dedup)
                if paper and paper.paper_id and paper.paper_id not in collected_ids:
                    collected.append(paper)
                    collected_ids.add(paper.paper_id)
                    # Download PDF if available
                    rel_path = await _download_pdf(paper, task.run_dir)
                    if rel_path:
                        paper.artifact_rel_path = rel_path
                    await db.upsert(paper)
                    new_count += 1
                    new_papers_this_round.append(paper)
                    logger.info("[%s] +paper: %s", self.agent_id, paper.title[:60])

            record = {
                "iteration": iteration + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "human_content": human_content,
                "ai_content": ai_content,
                "queries": queries,
                "fetch_urls": fetch_urls,
                "papers_found": new_count,
                "papers": [p.model_dump() for p in new_papers_this_round],
                "complete": complete,
                "collected_so_far": len(collected),
            }

            # Execute tools for next round
            search_snippet_text = ""
            fetch_text = ""

            if queries:
                try:
                    sr = await ddg_search_batch(queries, max_results_each=8)
                    search_snippet_text = _format_search_snippets(sr)
                    record["search_results"] = sr
                except Exception as e:
                    logger.warning("[%s] search failed: %s", self.agent_id, e)
                    record["search_results"] = {}

            if fetch_urls:
                try:
                    pages = await fetch_pages_batch(fetch_urls[:3], max_chars_each=3000)
                    fetch_text = _format_fetch_results(pages)
                    record["fetch_results"] = pages
                except Exception as e:
                    logger.warning("[%s] fetch failed: %s", self.agent_id, e)
                    record["fetch_results"] = []

            iteration_records.append(record)
            logger.info("[%s] iter %d: +%d papers (total %d/%d) complete=%s",
                        self.agent_id, iteration + 1, new_count, len(collected), target_papers, complete)

            # Only count "no new papers" after we've had fetch results to work with
            if new_count == 0 and fetch_text:
                no_new_count += 1
                if no_new_count >= 2:
                    logger.info("[%s] 2 consecutive rounds with fetch content but no new papers, stopping", self.agent_id)
                    break
            elif new_count > 0:
                no_new_count = 0

            if complete or len(collected) >= target_papers:
                break

        await db.close()
        _write_iteration_log(task.run_dir, self.agent_id, iteration_records)

        status = "success" if collected else "partial"
        return AgentResult(
            task_id=task.task_id,
            status=status,
            summary=f"collected {len(collected)} papers for: {question[:80]}",
            papers=collected,
            metadata=[{"paper_id": p.paper_id, "title": p.title} for p in collected],
            suggested_followup=None if collected else f"No papers found for: {question}",
        )
