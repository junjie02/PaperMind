"""Reviewer Sub-Agent.

Single-prompt, memory-driven agent that iteratively verifies factual claims
in a draft section by selecting statements to check via RAG retrieval.
Each round it picks up to 3 claims to verify, system retrieves evidence,
and the agent judges accuracy. Continues until all claims are checked.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import SimpleJsonOutputParser

from agents.base import SubAgentBase, parse_json_with_retry
from rag.retriever import Retriever, format_chunks_for_llm
from shared.config import Config
from shared.llm import make_llm
from shared.models import AgentResult, AgentTask

logger = logging.getLogger(__name__)

_PARSER = SimpleJsonOutputParser()

REVIEWER_SYSTEM = """\
你是 PaperMind 文献综述审核专家，负责逐句核查章节中的事实性论断。

## 你拥有的工具
- **RAG 检索**：每轮可提交最多 3 个查询句子，系统会并行检索每句对应的 top-3 论文片段返回给你

## 每轮工作流程
系统会给你待审核的完整章节、你已检索过的句子列表、以及本轮检索结果。你需要输出：
{
  "verify_queries": ["要核查的句子1", "要核查的句子2", "要核查的句子3"],
  "verified_ok": ["已确认有据可查的句子"],
  "issues": ["发现的问题描述"],
  "complete": false
}

## 字段说明
- `verify_queries`：本轮要提交给 RAG 检索的句子（从 draft 中摘出带引用的论断），最多 3 个；若不需要继续检索则为空数组
- `verified_ok`：本轮确认有据可查的句子（基于上轮检索结果判断）
- `issues`：本轮发现的问题（引用无据、内容编造、格式错误等）
- `complete`：所有带引用的论断都已检索验证完毕时设为 true

## 审核标准
- 引用格式：[论文完整标题] 或多篇 [标题1; 标题2]
- 每个带引用标注的事实性论断必须能在检索到的论文片段中找到依据
- 不带引用的一般性描述（如背景介绍、过渡句）不需要核查
- 逻辑连贯性和学术表达质量也需关注

## 行为规范
- 第一轮：从 draft 中选取前 3 个带引用的论断提交检索，verified_ok 和 issues 为空
- 后续轮：根据上轮检索结果判断哪些有据（加入 verified_ok）、哪些有问题（加入 issues），同时选取下一批待检索的句子
- 按顺序从头到尾检索，确保不遗漏
- 当所有带引用的论断都已验证完毕，设 complete=true
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


class ReviewerAgent(SubAgentBase):
    def __init__(self, config: Config, agent_id: str = "reviewer"):
        super().__init__(agent_id, timeout=config.reviewer_timeout)
        self._config = config
        self._llm = make_llm(config, temperature=0)

    def _build_messages(self, history: list[dict]) -> list:
        msgs: list = [SystemMessage(content=REVIEWER_SYSTEM)]
        for m in history:
            if m["role"] == "human":
                msgs.append(HumanMessage(content=m["content"]))
            else:
                msgs.append(AIMessage(content=m["content"]))
        return msgs

    async def _run_impl(self, task: AgentTask) -> AgentResult:
        params = json.loads(task.instruction)
        section_title: str = params["section_title"]
        draft_text: str = params["draft_text"]

        retriever = Retriever(Path(task.run_dir), embedding_model=self._config.embedding_model)
        messages: list[dict] = []
        iteration_records: list[dict] = []
        all_issues: list[str] = []
        all_verified: list[str] = []
        retrieval_results_text = ""

        for iteration in range(task.max_iterations):
            logger.info("[%s] iteration %d/%d", self.agent_id, iteration + 1, task.max_iterations)

            verified_block = "\n".join(f"- {v}" for v in all_verified) if all_verified else "（尚无）"
            retrieval_block = retrieval_results_text if retrieval_results_text else "（首轮，尚无检索结果）"

            human_content = f"""\
## 待审核章节：{section_title}

{draft_text}

## 已验证通过的句子（共 {len(all_verified)} 句）
{verified_block}

## 本轮检索结果
{retrieval_block}

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

            verify_queries: list[str] = decision.get("verify_queries", [])[:3]
            verified_ok: list[str] = decision.get("verified_ok", [])
            issues: list[str] = decision.get("issues", [])
            complete: bool = decision.get("complete", False)

            all_verified.extend(verified_ok)
            all_issues.extend(issues)

            record = {
                "iteration": iteration + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "verify_queries": verify_queries,
                "verified_ok": verified_ok,
                "issues": issues,
                "complete": complete,
                "total_verified": len(all_verified),
                "total_issues": len(all_issues),
            }

            if complete or not verify_queries:
                iteration_records.append(record)
                break

            # Execute parallel RAG retrieval for each query
            retrieval_parts = []

            async def _search(query: str) -> tuple[str, str]:
                chunks = retriever.search(query, top_k=3)
                text = format_chunks_for_llm(chunks) if chunks else "（未检索到相关片段）"
                return query, text

            search_results = await asyncio.gather(*[_search(q) for q in verify_queries])
            for query, text in search_results:
                retrieval_parts.append(f"### 查询：{query}\n{text}")
            retrieval_results_text = "\n\n".join(retrieval_parts)

            record["retrieval_queries_count"] = len(verify_queries)
            iteration_records.append(record)

        _write_iteration_log(task.run_dir, self.agent_id, iteration_records)

        passed = len(all_issues) == 0
        return AgentResult(
            task_id=task.task_id,
            status="success" if passed else "partial",
            summary=f"review {'passed' if passed else f'found {len(all_issues)} issues'} for '{section_title}' (verified {len(all_verified)} claims)",
            issues=all_issues,
        )
