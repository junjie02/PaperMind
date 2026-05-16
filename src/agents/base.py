"""SubAgentBase: unified timeout/retry wrapper for all Sub-Agents."""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.output_parsers import SimpleJsonOutputParser

from shared.models import AgentResult, AgentTask

logger = logging.getLogger(__name__)

_JSON_PARSER = SimpleJsonOutputParser()
_MAX_PARSE_RETRIES = 2


async def parse_json_with_retry(llm, messages_builder, messages: list[dict], ai_content: str) -> tuple[dict, str]:
    """Try to parse JSON from ai_content. On failure, ask LLM to fix it (up to 2 retries).

    Returns (parsed_dict, final_ai_content). Raises ValueError if all retries fail.
    """
    for retry in range(_MAX_PARSE_RETRIES + 1):
        try:
            result = _JSON_PARSER.parse(ai_content)
            if not isinstance(result, dict):
                raise ValueError(f"expected dict, got {type(result)}")
            return result, ai_content
        except Exception as e:
            if retry >= _MAX_PARSE_RETRIES:
                raise ValueError(f"JSON parse failed after {_MAX_PARSE_RETRIES} retries: {e}") from e
            logger.warning("JSON parse retry %d: %s", retry + 1, e)
            messages.append({"role": "ai", "content": ai_content})
            messages.append({"role": "human", "content": f"你的输出 JSON 解析失败：{e}\n请重新输出正确的 JSON，不要 markdown 代码块。"})
            response = await llm.ainvoke(messages_builder(messages))
            ai_content = response.content
    raise ValueError("unreachable")


def _write_agent_log(run_dir: str, agent_id: str, task: AgentTask, result: AgentResult) -> None:
    if not run_dir:
        return
    try:
        data_dir = Path(run_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        safe_id = agent_id.replace("/", "-").replace(":", "-")
        path = data_dir / f"{safe_id}.json"
        record = {
            "agent_id": agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": {
                "task_id": task.task_id,
                "instruction": task.instruction,
                "max_iterations": task.max_iterations,
            },
            "output": {
                "status": result.status,
                "summary": result.summary,
                "papers": [p.model_dump() for p in result.papers],
                "draft_text": result.draft_text,
                "issues": result.issues,
                "polished_text": result.polished_text,
                "metadata": result.metadata,
            },
        }
        # Append to array — read existing if present
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        else:
            existing = []
        existing.append(record)
        path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("[%s] failed to write agent log: %s", agent_id, e)


class SubAgentBase(ABC):
    def __init__(self, agent_id: str, timeout: int = 600):
        self.agent_id = agent_id
        self.timeout = timeout

    async def run(self, task: AgentTask) -> AgentResult:
        """Unified entry point with timeout, exception handling, and I/O logging."""
        try:
            result = await asyncio.wait_for(self._run_impl(task), timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.error("[%s] timeout after %ds for task %s", self.agent_id, self.timeout, task.task_id)
            result = AgentResult(
                task_id=task.task_id,
                status="failed",
                summary=f"timeout after {self.timeout}s",
            )
        except Exception as e:
            logger.error("[%s] error for task %s: %s", self.agent_id, task.task_id, e, exc_info=True)
            result = AgentResult(
                task_id=task.task_id,
                status="failed",
                summary=f"error: {e}",
            )
        _write_agent_log(task.run_dir, self.agent_id, task, result)
        return result

    @abstractmethod
    async def _run_impl(self, task: AgentTask) -> AgentResult: ...
