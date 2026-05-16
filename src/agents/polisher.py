"""Polisher Sub-Agents: per-section polishing + global consistency check."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import SimpleJsonOutputParser, StrOutputParser

from agents.base import SubAgentBase
from shared.config import Config
from shared.llm import make_llm
from shared.models import AgentResult, AgentTask

logger = logging.getLogger(__name__)

_PARSER_JSON = SimpleJsonOutputParser()
_PARSER_STR = StrOutputParser()

POLISHER_SYSTEM = """\
你是学术写作润色专家。

## 任务
润色学术综述章节，改善表达流畅性和逻辑连贯性。

## 规范
- 保持所有引用标注（如 [论文标题] 或 [标题1; 标题2]）完全不变，不要修改标题内容
- 改善句子流畅性和段落衔接
- 统一术语使用（如全文统一用英文缩写或中文全称）
- 不增删实质性内容，不改变论点
- 直接输出润色后的章节正文，不要任何前言或说明
"""

CONSISTENCY_SYSTEM = """\
你是学术综述一致性检查专家。

## 任务
检查综述各章节之间的一致性问题。

## 重要限制
- 不要把专有技术名称（如 FlashAttention、PagedAttention、SmoothQuant 等）替换为中文泛称
- 不要把论文标题或论文标题的一部分列为需要替换的变体
- 只处理同一概念的不同中文表述（如"大语言模型"vs"大型语言模型"），或同一缩写的不同写法
- variants 中不要包含引用标题 [...] 内出现的文本

## 输出格式
直接输出 JSON，不要任何前言或说明：
{
  "terminology_issues": [
    {"term": "概念名称", "variants": ["不统一的写法1", "不统一的写法2"], "recommended": "统一替换为的词（只写一个词或短语，不要写解释）"}
  ],
  "transition_issues": [
    "章节X结尾与章节Y开头衔接问题描述"
  ],
  "citation_issues": [
    "引用编号或格式异常描述"
  ]
}

## 示例
正确：{"term": "大语言模型", "variants": ["大型语言模型", "超大语言模型"], "recommended": "大语言模型"}
错误：{"term": "注意力优化", "variants": ["FlashAttention", "FlashAttention-2"], "recommended": "注意力优化"}
错误原因：FlashAttention 是专有技术名称，不应被替换为中文泛称

如果没有问题，对应字段输出空数组。
"""


class PolisherAgent(SubAgentBase):
    def __init__(self, config: Config, agent_id: str = "polisher"):
        super().__init__(agent_id, timeout=config.polisher_timeout)
        self._llm = make_llm(config, temperature=0.3)

    async def _run_impl(self, task: AgentTask) -> AgentResult:
        params = json.loads(task.instruction)
        draft_text: str = params["draft_text"]
        section_title: str = params.get("section_title", "")

        human_content = f"""\
## 待润色章节：{section_title}

{draft_text}

请直接输出润色后的正文。"""

        messages = [
            SystemMessage(content=POLISHER_SYSTEM),
            HumanMessage(content=human_content),
        ]

        try:
            response = await self._llm.ainvoke(messages)
            polished = response.content
        except Exception as e:
            logger.error("[%s] polish failed: %s", self.agent_id, e)
            polished = draft_text

        return AgentResult(
            task_id=task.task_id,
            status="success",
            summary=f"polished '{section_title}' ({len(draft_text)} → {len(polished)} chars)",
            polished_text=polished,
        )


class ConsistencyCheckerAgent(SubAgentBase):
    def __init__(self, config: Config):
        super().__init__("consistency-checker", timeout=config.polisher_timeout)
        self._llm = make_llm(config, temperature=0)

    async def _run_impl(self, task: AgentTask) -> AgentResult:
        params = json.loads(task.instruction)
        sections = params.get("sections", [])

        digest_parts = []
        for s in sections:
            title = s.get("title", "")
            head = s.get("head", "")[:300]
            tail = s.get("tail", "")[-300:]
            digest_parts.append(f"### {title}\n[开头] {head}\n...\n[结尾] {tail}")
        sections_digest = "\n\n".join(digest_parts)

        human_content = f"""\
## 综述各章节摘要与边界段落

{sections_digest[:5000]}

请检查一致性问题并输出 JSON。"""

        messages = [
            SystemMessage(content=CONSISTENCY_SYSTEM),
            HumanMessage(content=human_content),
        ]

        try:
            response = await self._llm.ainvoke(messages)
            report = _PARSER_JSON.parse(response.content)
            if not isinstance(report, dict):
                report = {"terminology_issues": [], "transition_issues": [], "citation_issues": []}
        except Exception as e:
            logger.error("[consistency-checker] failed: %s", e)
            report = {"terminology_issues": [], "transition_issues": [], "citation_issues": []}

        total_issues = (
            len(report.get("terminology_issues", []))
            + len(report.get("transition_issues", []))
            + len(report.get("citation_issues", []))
        )

        return AgentResult(
            task_id=task.task_id,
            status="success",
            summary=f"consistency check: {total_issues} issues found",
            metadata=[report],
        )
