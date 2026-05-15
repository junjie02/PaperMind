import logging

from deepresearch.search_client import SearchClient
from deepresearch.models import PaperRecord, WorkerTask

logger = logging.getLogger(__name__)

WORKER_TASK_PROMPT = """你是一位文献调研研究员。你需要为以下研究方向收集相关论文。

## 研究方向
"{search_direction}"

## 所属更大的研究课题
"{research_question}"

## 你的任务
1. 联网搜索与上述方向相关的学术论文。每个搜索词必须包含 "arxiv" 关键字，确保搜到的是学术论文而非博客。
2. 对于有希望的搜索结果，抓取页面内容，阅读摘要和正文。
3. 通过 arXiv API 或 arXiv 页面获取每篇论文的完整元数据（标题、作者、摘要、分类、bibtex 等）。
4. 用中文为每篇论文写一段 2-3 句话的概述。
5. 只纳入你实际读过内容并确认相关性的论文。

## 排除列表
以下 arXiv ID 已经被收集过了，不要重复收集：
{exclude_ids}

## 要求
- 目标收集 {target_papers} 篇经过验证的相关论文；如果高质量结果不足，可以少于该数量，但不要为了凑数收录不相关论文。
- 优先收录来自顶会（CVPR、ICCV、NeurIPS、ICML、ACL 等）和 arXiv 高引论文。

## 输出格式
你必须只输出一个 JSON 对象，不要有任何额外的文字、解释。
JSON 必须严格符合以下结构，每个字段都必须填写：
{{
  "papers": [
    {{
      "arxiv_id": "论文 arXiv ID，如 1512.03385",
      "title": "论文标题",
      "authors": ["作者1", "作者2"],
      "abstract": "论文摘要全文",
      "overview": "用中文写的2-3句话概述",
      "published_at": "YYYY-MM-DD 格式发表日期",
      "categories": ["cs.CV", "cs.AI"],
      "primary_class": "主分类如 cs.CV",
      "bibtex": "@article{{...}} 完整 BibTeX 引用",
      "abs_url": "https://arxiv.org/abs/XXXX.XXXXX",
      "pdf_url": "https://arxiv.org/pdf/XXXX.XXXXX",
      "relevance_score": 5
    }}
  ],
  "summary": "本轮搜索的简短总结",
  "search_queries_used": ["实际使用的搜索词1", "实际使用的搜索词2"]
}}
"""


class Worker:

    def __init__(
        self,
        task: WorkerTask,
        searcher: SearchClient,
    ):
        self.task = task
        self.searcher = searcher

    async def run(self) -> list[PaperRecord]:
        wid = f"W{self.task.worker_index}-R{self.task.round_num}"
        exclude_str = ", ".join(sorted(self.task.exclude_ids)[:80]) or "（尚无）"

        prompt = WORKER_TASK_PROMPT.format(
            search_direction=self.task.search_direction,
            research_question=self.task.research_question,
            exclude_ids=exclude_str,
            target_papers=self.task.target_papers,
        )

        logger.info(f"[{wid}] 开始探索: {self.task.search_direction}")
        logger.info(f"[{wid}] 排除 {len(self.task.exclude_ids)} 篇已有论文")
        logger.info(f"[{wid}] 目标收集 {self.task.target_papers} 篇论文")

        task_info = {
            "round_num": self.task.round_num,
            "worker_index": self.task.worker_index,
            "search_direction": self.task.search_direction,
        }
        result = await self.searcher.exec(prompt, task_info=task_info)

        summary = result.get("summary", "")
        queries = result.get("search_queries_used", [])
        logger.info(f"[{wid}] 搜索完成 | 搜索词: {queries}")
        logger.info(f"[{wid}] 摘要: {summary}")

        papers = self.searcher.parse_papers(result, self.task)
        logger.info(f"[{wid}] 解析出 {len(papers)} 篇论文")
        for p in papers:
            logger.info(f"  [{p.relevance_score}/5] {p.title[:70]}")
            logger.info(f"  {p.arxiv_id} | {p.published_at or '?'}")

        return papers
