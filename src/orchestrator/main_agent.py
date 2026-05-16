"""Main orchestrator Agent with persistent conversation memory.

A single LLM instance with one system prompt drives all orchestration decisions.
Conversation history accumulates across phases so the agent always knows what
it has done and what comes next.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import SimpleJsonOutputParser, StrOutputParser

from shared.config import Config
from shared.llm import make_llm

logger = logging.getLogger(__name__)

# ── Single system prompt describing the agent's full mission ────────────────

MAIN_AGENT_SYSTEM = """\
你是 PaperMind 文献综述主导 Agent，负责端到端地完成一篇高质量学术文献综述。

## 你的完整工作流程

你将按以下阶段推进。每个阶段结束后，系统会把执行结果反馈给你，你可以根据内容质量决定是否进入下一阶段。

---

### 阶段 1 — 方向拆分
**触发**：收到研究课题后首次被调用。
**任务**：将课题拆分为 3-8 个相对独立的子研究方向，并决定同时启动多少个 Explorer Sub-Agent 并发探索。
**输出格式**：JSON，直接输出，不要 markdown 代码块：
{
  "directions": ["子方向1", "子方向2", "子方向3", ...],
  "explorer_concurrency": <1-3> #并发的Sub-Agent数量
}

---

### 阶段 2 — 大纲生成
**触发**：收到各子方向的 Explorer 探索结果后。
**任务**：
1. 汇总探索结果，生成文献综述格式结构化的研究大纲（含 Introduction 和 Conclusion，章数根据研究方向实际情况规划，每章 2-4 个 sub_questions）
2. 决定后续 Researcher / Writer / Reviewer / Polisher 阶段的并发 Sub-Agent 数量
3. 为每个 sub_question 分配文献收集数量，总和等于用户指定的总文献数 {total_papers}；根据每个 sub_question 的重要性和研究深度自主分配，核心问题多分配，次要问题少分配

**重要约束**：
- Introduction 的 `sub_questions` 必须为空数组 `[]`，它将在所有正文章节写完后，根据正文内容自动生成
- Conclusion 的 `sub_questions` 必须为空数组 `[]`，它将在所有正文章节写完后，汇总全文内容生成

**输出格式**：JSON，直接输出，不要 markdown 代码块：
{
  "concurrency": <1-3>,
  "chapters": [
    {
      "title": "章节标题",
      "description": "该章节研究内容（2-3句）",
      "sub_questions": ["具体研究问题1", "具体研究问题2"]
    }
  ],
  "papers_per_question": {
    "具体研究问题1": <分配数量>,
    "具体研究问题2": <分配数量>
  }
}

---

### 阶段 3 — 覆盖度评估
**触发**：收到各 sub_question 的文献收集结果后。
**任务**：判断已收集文献是否足以支撑大纲写作；若不足，指出薄弱方向。
**输出格式**：JSON，直接输出，不要 markdown 代码块：
{
  "sufficient": true 或 false,
  "weak_areas": ["覆盖不足的方向1", "覆盖不足的方向2"],
  "reason": "评估理由（2-3句）"
}
"""

# ── Serializable message format stored in LangGraph state ───────────────────
# {"role": "human" | "ai", "content": "..."}

_PARSER_JSON = SimpleJsonOutputParser()
_PARSER_STR = StrOutputParser()


def _write_main_agent_log(run_dir: str, human: str, ai: str, full_history: list[dict]) -> None:
    if not run_dir:
        return
    try:
        data_dir = Path(run_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "main_agent.json"
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_prompt": MAIN_AGENT_SYSTEM,
            "history": full_history,
            "input": human,
            "output": ai,
        }
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        existing.append(record)
        path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("[main_agent] failed to write log: %s", e)


class MainAgent:
    """Stateless wrapper — conversation history lives in LangGraph state."""

    def __init__(self, config: Config):
        self._llm = make_llm(config, temperature=0.3)

    def _build_messages(self, history: list[dict]) -> list:
        msgs: list = [SystemMessage(content=MAIN_AGENT_SYSTEM)]
        for m in history:
            if m["role"] == "human":
                msgs.append(HumanMessage(content=m["content"]))
            else:
                msgs.append(AIMessage(content=m["content"]))
        return msgs

    async def chat(
        self,
        history: list[dict],
        user_content: str,
        run_dir: str = "",
    ) -> tuple[str, list[dict]]:
        """Send a message, return (ai_content, updated_history)."""
        messages = self._build_messages(history)
        messages.append(HumanMessage(content=user_content))
        response = await self._llm.ainvoke(messages)
        content: str = response.content
        _write_main_agent_log(run_dir, user_content, content, history)
        new_history = history + [
            {"role": "human", "content": user_content},
            {"role": "ai", "content": content},
        ]
        return content, new_history

    async def chat_json(
        self,
        history: list[dict],
        user_content: str,
        run_dir: str = "",
    ) -> tuple[dict | list, list[dict]]:
        """chat() + parse JSON from response. Falls back to {} on parse error."""
        content, new_history = await self.chat(history, user_content, run_dir=run_dir)
        try:
            data = _PARSER_JSON.parse(content)
        except Exception as e:
            logger.warning("[main_agent] JSON parse failed: %s | raw: %.200s", e, content)
            data = {}
        return data, new_history
