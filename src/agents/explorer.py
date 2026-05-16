"""Explorer Sub-Agent.

Single-prompt, memory-driven agent that iteratively explores a research direction
through DDG searches and targeted page fetches. The agent manages its own context:
each round it decides what to search, which URLs to fetch, what to memorize, and
whether it has gathered enough information to stop.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import SimpleJsonOutputParser

from agents.base import SubAgentBase, parse_json_with_retry
from mcp_servers.ddg_search import ddg_search_batch, fetch_pages_batch
from shared.config import Config
from shared.llm import make_llm
from shared.models import AgentResult, AgentTask

logger = logging.getLogger(__name__)

_PARSER = SimpleJsonOutputParser()

EXPLORER_SYSTEM = """\
你是 PaperMind 文献调研专家，负责对一个研究方向进行深入探索，为后续文献综述写作提供知识基础。

## 你拥有的工具

- **DDG 搜索**：每轮可提交 1-3 个搜索词，系统会并发执行并返回结果（title、url、snippet）
- **网页抓取**：对搜索结果中有价值的页面，可提交 URL 列表，系统会抓取正文内容

## 每轮工作流程

系统会给你当前的 memory 和上一轮工具执行结果，你需要输出本轮决策 JSON：

```json
{
  "queries": ["搜索词1", "搜索词2"],
  "fetch_urls": ["https://...", "https://..."],
  "memory_update": "上一轮收获的核心信息摘要",
  "complete": false
}
```

字段说明：
- `queries`：本轮要搜索的关键词，英文为主，；若不需要搜索则为空数组
- `fetch_urls`：要抓取全文的 URL，优先选论文摘要页、综述文章、高质量博客；若不需要抓取则为空数组
- `memory_update`：本轮从搜索结果和网页内容中提炼的核心信息，会永久保存到 memory；若本轮无新收获则为空字符串
- `complete`：当你认为已充分了解该方向时设为 true，系统将停止迭代

## Memory 管理原则

memory 是你的知识积累，每轮的 `memory_update` 会追加进去。请确保 memory 最终覆盖：
- 该方向的主流方法和技术路线
- 关键争议或未解决问题
- 2022 年至今的新进展

当以上四个方面都有足够记录时，设 `complete: true`。

## 行为规范

- 第一轮：直接规划搜索词，queries 不为空，fetch_urls 可为空
- 后续轮：根据已有 memory 判断还缺什么，针对性搜索；对有价值的 URL 主动抓取
- 网页内容可能很长，memory_update 只保留核心信息，不要复制粘贴原文
- 不要重复搜索已覆盖的内容
- 直接输出 JSON，不要任何前言或说明
"""

_SUMMARIZE_USER = """\
## 总结任务

探索已完成。基于以上所有 memory，生成该研究方向的结构化报告。

直接输出 JSON，不要 markdown 代码块：
{{
  "direction": "{direction}",
  "mainstream_methods": ["主流方法1（简短描述）", "主流方法2"],
  "key_controversies": ["关键争议1", "关键争议2"],
  "recent_trends": "2022年至今的新进展（2-3句）",
  "summary": "整体现状总结（3-5句）"
}}"""


def _write_iteration_log(run_dir: str, agent_id: str, records: list[dict], report: dict) -> None:
    if not run_dir:
        return
    try:
        data_dir = Path(run_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        safe_id = agent_id.replace("/", "-").replace(":", "-")
        path = data_dir / f"{safe_id}_iterations.json"
        payload = {"iterations": records, "final_report": report}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("[%s] failed to write iteration log: %s", agent_id, e)


def _format_search_results(results: dict[str, list[dict]], max_snippet: int = 200) -> str:
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


class ExplorerAgent(SubAgentBase):
    def __init__(self, config: Config, agent_id: str = "explorer"):
        super().__init__(agent_id, timeout=config.explorer_timeout)

        self._llm = make_llm(config, temperature=0.3)

    def _build_messages(self, history: list[dict]) -> list:
        msgs: list = [SystemMessage(content=EXPLORER_SYSTEM)]
        for m in history:
            if m["role"] == "human":
                msgs.append(HumanMessage(content=m["content"]))
            else:
                msgs.append(AIMessage(content=m["content"]))
        return msgs

    async def _run_impl(self, task: AgentTask) -> AgentResult:
        direction = task.instruction
        memory: list[str] = []
        messages: list[dict] = []
        iteration_records: list[dict] = []

        search_results_text = ""
        fetch_results_text = ""

        for iteration in range(task.max_iterations):
            logger.info("[%s] iteration %d/%d", self.agent_id, iteration + 1, task.max_iterations)

            memory_block = "\n\n".join(
                f"[{i+1}] {m}" for i, m in enumerate(memory)
            ) if memory else "（尚无）"

            tool_block = ""
            if search_results_text:
                tool_block += f"### 搜索结果\n{search_results_text}\n\n"
            if fetch_results_text:
                tool_block += f"### 网页抓取结果\n{fetch_results_text}\n\n"

            human_content = f"""\
## 当前任务
研究方向：{direction}

## 已有 Memory（共 {len(memory)} 条）
{memory_block}

## 本轮工具执行结果
{tool_block if tool_block else "（第一轮，尚无工具结果）"}
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
            memory_update: str = decision.get("memory_update", "")
            complete: bool = decision.get("complete", False)

            if memory_update:
                memory.append(memory_update)

            record = {
                "iteration": iteration + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "queries": queries,
                "fetch_urls": fetch_urls,
                "memory_update": memory_update,
                "memory_snapshot": list(memory),
                "complete": complete,
            }

            # Execute tools
            search_results_text = ""
            fetch_results_text = ""

            if queries:
                try:
                    sr = await ddg_search_batch(queries, max_results_each=3)
                    search_results_text = _format_search_results(sr)
                    record["search_results"] = sr
                except Exception as e:
                    logger.warning("[%s] search failed: %s", self.agent_id, e)
                    record["search_results"] = {}

            if fetch_urls:
                try:
                    pages = await fetch_pages_batch(fetch_urls[:5], max_chars_each=3000)
                    fetch_results_text = _format_fetch_results(pages)
                    record["fetch_results"] = pages
                except Exception as e:
                    logger.warning("[%s] fetch failed: %s", self.agent_id, e)
                    record["fetch_results"] = []

            iteration_records.append(record)
            logger.info("[%s] iter %d: queries=%d fetch=%d memory=%d complete=%s",
                        self.agent_id, iteration + 1, len(queries), len(fetch_urls), len(memory), complete)

            if complete:
                break

        # Generate final structured report using the same conversation context
        memory_block = "\n\n".join(f"[{i+1}] {m}" for i, m in enumerate(memory)) if memory else "（无）"

        # Include last round's tool results if available — they were never fed back to the LLM
        last_tool_block = ""
        if search_results_text:
            last_tool_block += f"### 最后一轮搜索结果\n{search_results_text}\n\n"
        if fetch_results_text:
            last_tool_block += f"### 最后一轮网页抓取结果\n{fetch_results_text}\n\n"

        summarize_prompt = f"""\
## 当前任务
研究方向：{direction}

## 已有 Memory（共 {len(memory)} 条）
{memory_block}

{f"## 最后一轮工具结果（尚未消化入 Memory）{chr(10)}{last_tool_block}" if last_tool_block else ""}
{_SUMMARIZE_USER.format(direction=direction)}"""

        messages.append({"role": "human", "content": summarize_prompt})
        report: dict = {}
        try:
            response = await self._llm.ainvoke(self._build_messages(messages))
            messages.append({"role": "ai", "content": response.content})
            report = _PARSER.parse(response.content)
            if not isinstance(report, dict):
                raise ValueError(f"unexpected report type: {type(report)}")
        except Exception as e:
            logger.error("[%s] synthesize failed: %s", self.agent_id, e)
            report = {
                "direction": direction,
                "mainstream_methods": [],
                "key_controversies": [],
                "representative_papers": [],
                "recent_trends": "",
                "summary": "; ".join(memory) if memory else "exploration failed",
            }

        _write_iteration_log(task.run_dir, self.agent_id, iteration_records, report)

        return AgentResult(
            task_id=task.task_id,
            status="success",
            summary=report.get("summary", ""),
            metadata=[report],
        )
