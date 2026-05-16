"""Writer Sub-Agent.

Single-prompt, memory-driven agent that iteratively writes a literature review
section using RAG retrieval. Each round it outputs a complete draft and optionally
requests more information via a hypothetical query.
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
from rag.retriever import Retriever, Section, format_chunks_for_llm
from shared.config import Config
from shared.llm import make_llm
from shared.models import AgentResult, AgentTask

logger = logging.getLogger(__name__)

_PARSER = SimpleJsonOutputParser()

WRITER_SYSTEM = """\
你是 PaperMind 文献综述写作专家，负责严格基于 RAG 检索到的真实论文片段撰写学术综述章节。

## 你拥有的工具
- **RAG 检索**：每轮可提交一个查询，系统从论文向量库中检索最相关的片段返回给你

## 每轮工作流程
系统会给你章节大纲、已检索到的论文片段、以及你上一轮的草稿，你需要输出：
{
  "queries": ["RAG 检索查询1", "查询2", ...],
  "draft": "当前版本的完整章节正文（每轮都要输出完整版本，不是增量）",
  "complete": false
}

字段说明：
- `queries`：本轮要提交的 RAG 检索查询列表，系统会并行检索每个查询的 top-2 相关片段；不需要更多信息时为空数组
- `draft`：当前最佳版本的完整正文
- `complete`：内容已充分覆盖大纲要求时设为 true

## 写作规范
- 严格基于检索到的论文片段写作，不得编造未出现的内容
- 每个事实性论断后标注引用，使用方括号包裹论文完整标题：[论文标题]
- 多篇引用用分号分隔：[标题1; 标题2]
- 标题必须与检索到的论文片段中显示的**完整标题**完全一致，不要缩写或修改
- 文献引用禁止使用圆括号引用格式如 (Author et al., Year)，必须用方括号 [论文标题]
- 每一段都必须至少有一个引用标注，没有引用支撑的段落不要写
- 如果你写了某个具体数据、方法名称或实验结论，必须标注来源论文；无法标注来源的内容直接删除
- 禁止使用你自己的知识补充内容，只能使用检索片段中明确出现的信息
- 学术综述体，段落式论述，注重论文间的对比、联系和发展脉络
- 每轮输出的 draft 是当前最佳版本（完善上一轮，不是追加）
- 若检索内容不足以支撑某个子节，简短说明"相关研究有限"
- 全文markdown格式，公式请用$$符号包裹
- 当你认为内容已充分覆盖大纲要求，设 complete=true


## 写作风格要求
- 严格按照文献综述的风格写作，应该先描述技术/方法/发现，再在句末标注引用
- 不要照搬相邻章节参考中的原文或标题，那只是帮你了解上下文衔接的
- 三级标题不要带序号，系统会自动编号；直接写 `### 子话题名称`

## 写作示例
错误写法：GPTQ: Accurate Post-Training Quantization提出了一种基于二阶信息的量化方法[Wang. 2023]。
正确写法：基于近似二阶信息的一次性权重量化方法能够在约4 GPU小时内将1750亿参数模型量化到3-4位精度，且精度损失可忽略不计[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers]。

错误写法：(Xiao et al., 2023) 提出了SmoothQuant方法。
正确写法：SmoothQuant通过数学等价变换将激活值的量化难度迁移到权重，实现了INT8权重和激活的同时量化[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models]。

## 格式规范
你写的内容将作为综述中某个章节（二级标题）下的一个小节。整体文档结构为：
  # 研究课题（一级标题）
  ## 章节名（二级标题）——由系统生成，你不需要写
  ### 子话题（三级标题）——你可以使用，用来组织内容
  正文段落

因此：
- 不要输出一级标题（#）或二级标题（##）
- 如果内容较长或涉及多个子话题，用三级标题（###）分段组织
- 如果内容简短集中，可以不用三级标题，直接写段落

## 行为规范
- 第一轮：基于系统提供的初始检索结果写初稿，如需更多信息在 queries 中列出多个查询
- 后续轮：根据新检索结果完善 draft，queries 为空数组表示不再需要检索
- complete=true 时 draft 为最终版本
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


class WriterAgent(SubAgentBase):
    def __init__(self, config: Config, agent_id: str = "writer"):
        super().__init__(agent_id, timeout=config.write_timeout)
        self._config = config
        self._llm = make_llm(config, temperature=0.4)
        self._messages: list[dict] = []
        self._draft: str = ""

    def _build_messages(self, history: list[dict]) -> list:
        msgs: list = [SystemMessage(content=WRITER_SYSTEM)]
        for m in history:
            if m["role"] == "human":
                msgs.append(HumanMessage(content=m["content"]))
            else:
                msgs.append(AIMessage(content=m["content"]))
        return msgs

    async def _run_impl(self, task: AgentTask) -> AgentResult:
        params = json.loads(task.instruction)
        section_title: str = params["section_title"]
        outline_text: str = params["outline_text"]
        adjacent_context: str = params.get("adjacent_context", "")
        feedback: str = params.get("feedback", "")
        available_papers: list[dict] = params.get("available_papers", [])

        run_dir = Path(task.run_dir)
        retriever = Retriever(
            run_dir,
            embedding_model=self._config.embedding_model,
            llm=make_llm(self._config, temperature=0),
        )

        section = Section(title=section_title, outline_text=outline_text, order=0)
        iteration_records: list[dict] = []
        draft = self._draft
        chunks_text = ""

        # Initial RAG retrieval
        try:
            chunks = await retriever.dual_search(section, top_k=3)
            chunks_text = format_chunks_for_llm(chunks) if chunks else "（未检索到相关片段）"
        except Exception as e:
            logger.warning("[%s] initial RAG failed: %s", self.agent_id, e)
            chunks_text = "（检索失败）"

        for iteration in range(task.max_iterations):
            logger.info("[%s] iteration %d/%d", self.agent_id, iteration + 1, task.max_iterations)

            adjacent_block = f"## 相邻章节参考（保持连贯性）\n{adjacent_context[:1000]}\n\n" if adjacent_context else ""
            feedback_block = f"## 审核反馈\n{feedback}\n\n请根据反馈修改 draft。\n\n" if feedback else ""
            previous_draft_block = draft if draft else "（首轮，尚无草稿）"
            papers_block = ""
            if available_papers:
                papers_list = "\n".join(f"- {p['title']}：{p['overview']}" for p in available_papers)
                papers_block = f"## 可用论文列表（请尽量全部引用，通过 queries 检索它们）\n{papers_list}\n\n"

            human_content = f"""\
## 章节大纲
{outline_text}

{adjacent_block}{papers_block}## 本轮检索到的论文片段
{chunks_text}

## 你上一轮的草稿
{previous_draft_block}

{feedback_block}请输出本轮决策 JSON。"""

            self._messages.append({"role": "human", "content": human_content})

            try:
                response = await self._llm.ainvoke(self._build_messages(self._messages))
                ai_content = response.content
                self._messages.append({"role": "ai", "content": ai_content})
                decision, ai_content = await parse_json_with_retry(
                    self._llm, self._build_messages, self._messages, ai_content
                )
            except Exception as e:
                logger.error("[%s] LLM decision failed: %s", self.agent_id, e)
                break

            queries: list[str] = decision.get("queries", [])
            draft = decision.get("draft", draft)
            complete: bool = decision.get("complete", False)
            # Clear feedback after first use
            feedback = ""

            record = {
                "iteration": iteration + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "queries": queries,
                "draft_length": len(draft),
                "complete": complete,
            }
            iteration_records.append(record)

            if complete:
                break

            # Execute parallel RAG queries for next round
            chunks_text = ""
            if queries:
                try:
                    async def _search(q: str) -> tuple[str, list]:
                        results = retriever.search(q, top_k=2)
                        return q, results

                    search_results = await asyncio.gather(*[_search(q) for q in queries])
                    parts = []
                    total_chunks = 0
                    for q, chunks in search_results:
                        if chunks:
                            total_chunks += len(chunks)
                            parts.append(f"### 查询：{q}\n{format_chunks_for_llm(chunks)}")
                        else:
                            parts.append(f"### 查询：{q}\n（未检索到相关片段）")
                    chunks_text = "\n\n".join(parts)
                    record["query_results_count"] = total_chunks
                except Exception as e:
                    logger.warning("[%s] RAG query failed: %s", self.agent_id, e)
                    chunks_text = "（检索失败）"

        self._draft = draft
        _write_iteration_log(task.run_dir, self.agent_id, iteration_records)

        return AgentResult(
            task_id=task.task_id,
            status="success" if draft else "failed",
            summary=f"wrote '{section_title}' ({len(draft)} chars)",
            draft_text=draft,
        )

    async def revise(self, task: AgentTask, feedback: str) -> AgentResult:
        """Continue writing with reviewer feedback, preserving conversation memory."""
        params = json.loads(task.instruction)
        params["feedback"] = feedback
        task = AgentTask(
            task_id=task.task_id,
            instruction=json.dumps(params, ensure_ascii=False),
            max_iterations=task.max_iterations,
            run_dir=task.run_dir,
        )
        return await self._run_impl(task)
