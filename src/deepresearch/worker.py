import logging

from deepresearch.search_client import SearchClient
from deepresearch.models import PaperRecord, WorkerTask

logger = logging.getLogger(__name__)

WORKER_TASK_PROMPT = """你是一位文献调研研究员。你需要为以下研究课题收集相关论文。

## 研究课题
"{research_question}"

## 你的任务
1. **优先用 arXiv 搜索**（搜索词加 "arxiv" 关键字）。arXiv 摘要页稳定可访问，元数据完整，是最高效的来源。
2. 若该课题的核心论文不在 arXiv（例如只发在 TCHES、S&P、NDSS、IACR ePrint、期刊），可以从搜索结果片段、DBLP、Google Scholar 缓存等渠道收集元数据；填上 `source` 和 `source_url` 即可，**不必**强行打开权威页面。
3. 如果论文来源是 arXiv 预印本，请填写 `arxiv_id`（仅 ID，不含版本号）；不在 arXiv 的论文留空 `arxiv_id`。
4. 只纳入你确认相关且质量过关（顶会顶刊 / 高引 / 公认重要工作）的论文，你必须通过其引用量级发表期刊、会议等信息确认其质量。
5. 自行决定从哪些子方向、用哪些关键词搜索；不需要被预先指定搜索词。

## 效率规则（严格遵守）
- **遇到 403 / 404 / 418 / Cloudflare 拦截**，不要重试同一 URL，也不要换站点反复找同一篇论文的元数据。
- 如果某次 web_search 工具返回 "I don't have web search tools" 之类的可疑回答，**忽略它，继续用下一个搜索词**，不要追问或重复同一查询。
- **当你已经收集到 {target_papers} 篇满足条件的论文，立即停止搜索并输出最终 JSON**，不要为了"再确认一下"而继续。
- 篇数不足时**不要凑数**：少给几篇可以接受。

## 排除列表
以下 paper_id 已经被收集过了，不要重复收集：
{exclude_ids}

## 要求
- 目标收集 {target_papers} 篇高质量、与本课题强相关的论文。
- 收集到的论文要尽量覆盖该课题的不同子方向，不要全是同一团队的不同年份连作。
- 必须填写 `source`（如 arxiv / openreview / acl / neurips / cvpr / tches / ieee / nature / web）和 `source_url`（权威页面 URL，即便没打开也可填）。

## 输出格式
你必须只输出一个 JSON 对象，不要有任何额外的文字、解释。
JSON 必须严格符合以下结构：
{{
  "papers": [
    {{
      "title": "论文标题",
      "authors": ["作者1", "作者2"],
      "abstract": "论文摘要全文",
      "overview": "用中文写的2-3句话概述",
      "source": "来源标识，如 arxiv / openreview / acl / cvpr / web",
      "source_url": "论文的权威页面 URL",
      "venue": "发表会议/期刊，如 'ICLR 2024'、'NeurIPS 2023'、'arXiv preprint'；不确定可留空字符串",
      "arxiv_id": "若也在 arXiv 上，填写如 1512.03385；否则填空字符串",
      "pdf_url": "PDF 直链；arXiv 论文填 https://arxiv.org/pdf/XXXX.XXXXX，其它论文如有公开 PDF 也可填，没有可留空",
      "abs_url": "摘要页 URL，可与 source_url 相同",
      "published_at": "YYYY-MM-DD，可留空字符串",
      "categories": ["可选的分类标签"],
      "primary_class": "可选的主分类",
      "bibtex": "@inproceedings{{...}} 完整 BibTeX 引用",
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
        wid = f"W{self.task.worker_index}"

        if self.task.exclude_ids:
            pairs = list(zip(self.task.exclude_ids, self.task.exclude_titles or []))
            lines = [
                f"- {pid}  {title}" if title else f"- {pid}"
                for pid, title in pairs
            ]
            exclude_str = "\n".join(lines)
        else:
            exclude_str = "（尚无）"

        prompt = WORKER_TASK_PROMPT.format(
            research_question=self.task.research_question,
            exclude_ids=exclude_str,
            target_papers=self.task.target_papers,
        )

        logger.info(f"[{wid}] 开始探索: {self.task.research_question}")
        logger.info(f"[{wid}] 排除 {len(self.task.exclude_ids)} 篇已有论文")
        logger.info(f"[{wid}] 目标收集 {self.task.target_papers} 篇论文")

        task_info = {
            "worker_index": self.task.worker_index,
            "research_question": self.task.research_question,
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
            logger.info(f"  {p.paper_id} | {p.source} | {p.published_at or '?'}")

        return papers
