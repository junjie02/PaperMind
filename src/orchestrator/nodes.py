"""Orchestrator nodes for Phase 1-3 (exploration, outline, research)."""

from __future__ import annotations

import asyncio
import json
import logging

from langgraph.types import Command

from agents.explorer import ExplorerAgent
from agents.researcher import ResearcherAgent
from orchestrator.main_agent import MainAgent
from orchestrator.state import PaperMindState
from shared.config import Config
from shared.models import AgentTask

logger = logging.getLogger(__name__)


def _save_outline_md(run_dir: str, chapters: list[dict], papers_per_question: dict) -> None:
    if not run_dir:
        return
    from pathlib import Path
    path = Path(run_dir) / "outline.md"
    lines = ["# Research Outline\n"]
    for ch in chapters:
        lines.append(f"## {ch['title']}\n")
        if desc := ch.get("description", ""):
            lines.append(f"{desc}\n")
        for sq in ch.get("sub_questions", []):
            target = papers_per_question.get(sq, "")
            suffix = f" （目标 {target} 篇）" if target else ""
            lines.append(f"- {sq}{suffix}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_nodes(config: Config):
    max_concurrent = config.max_concurrent_agents
    agent_max_retries = config.agent_max_retries
    main_agent = MainAgent(config)

    # ── Node: explore_directions ────────────────────────────────────────────

    async def explore_directions(state: PaperMindState) -> Command:
        topic = state["research_topic"]
        history = state.get("agent_messages", [])
        logger.info("[orchestrator] explore_directions: %s", topic)

        prompt = f"""\
## 阶段 1 — 方向拆分

研究课题：{topic}

请将该课题拆分为 3-5 个子研究方向，并决定并发启动多少个 Explorer Sub-Agent。
按阶段 1 的输出格式输出 JSON，直接输出，不要 markdown 代码块。"""

        try:
            data, new_history = await main_agent.chat_json(history, prompt, run_dir=state.get("run_dir", ""))
            sub_directions = [d.strip() for d in data.get("directions", []) if d.strip()][:5]
            explorer_concurrency = max(1, min(3, int(data.get("explorer_concurrency", max_concurrent))))
        except Exception as e:
            logger.error("[orchestrator] split_directions failed: %s", e)
            sub_directions = []
            explorer_concurrency = min(3, max_concurrent)
            new_history = history

        if not sub_directions:
            sub_directions = [topic]
        logger.info("[orchestrator] sub_directions=%s concurrency=%d", sub_directions, explorer_concurrency)

        semaphore = asyncio.Semaphore(explorer_concurrency)

        async def _explore(direction: str) -> dict:
            async with semaphore:
                agent = ExplorerAgent(config, agent_id=f"explorer-{direction[:20]}")
                task = AgentTask(instruction=direction, max_iterations=3, run_dir=state.get("run_dir", ""))
                result = await agent.run(task)
                return result.metadata[0] if result.metadata else {"direction": direction, "summary": result.summary}

        explorer_results = await asyncio.gather(*[_explore(d) for d in sub_directions])

        return Command(
            update={
                "sub_directions": sub_directions,
                "explorer_results": list(explorer_results),
                "agent_messages": new_history,
            },
            goto="synthesize_outline",
        )

    # ── Node: synthesize_outline ────────────────────────────────────────────

    async def synthesize_outline(state: PaperMindState) -> Command:
        topic = state["research_topic"]
        explorer_results = state.get("explorer_results", [])
        history = state.get("agent_messages", [])
        logger.info("[orchestrator] synthesize_outline")

        summaries = []
        for r in explorer_results:
            direction = r.get("direction", "")
            summary = r.get("summary", "")
            methods = r.get("mainstream_methods", [])
            summaries.append(f"### {direction}\n{summary}\n主流方法: {', '.join(methods[:3])}")
        explorer_summaries = "\n\n".join(summaries)

        total_papers = state.get("target_papers", 20)
        prompt = f"""\
## 阶段 2 — 大纲生成

各子方向的 Explorer 探索结果如下：

{explorer_summaries}

请根据以上结果，为课题「{topic}」生成结构化研究大纲，并决定后续并发 Sub-Agent 数量。
用户指定总文献数量为 {total_papers} 篇，请为每个 sub_question 分配合适的文献数量，总和等于 {total_papers}。
按阶段 2 的输出格式输出 JSON，直接输出，不要 markdown 代码块。"""

        try:
            data, new_history = await main_agent.chat_json(history, prompt, run_dir=state.get("run_dir", ""))
            chapters = data.get("chapters", []) if isinstance(data, dict) else []
            raw_concurrency = int(data.get("concurrency", max_concurrent)) if isinstance(data, dict) else max_concurrent
            agent_concurrency = max(1, min(3, raw_concurrency))
            papers_per_question: dict = data.get("papers_per_question", {}) if isinstance(data, dict) else {}
        except Exception as e:
            logger.error("[orchestrator] synthesize_outline failed: %s", e)
            chapters = [
                {"title": "Introduction", "description": "", "sub_questions": []},
                {"title": topic, "description": "", "sub_questions": [topic]},
                {"title": "Conclusion", "description": "", "sub_questions": []},
            ]
            agent_concurrency = min(3, max_concurrent)
            papers_per_question = {}
            new_history = history

        logger.info("[orchestrator] outline: %d chapters, concurrency=%d", len(chapters), agent_concurrency)

        # Write outline as readable markdown to run directory
        _save_outline_md(state.get("run_dir", ""), chapters, papers_per_question)

        return Command(
            update={
                "research_outline": chapters,
                "coverage_check_count": 0,
                "agent_concurrency": agent_concurrency,
                "papers_per_question": papers_per_question,
                "agent_messages": new_history,
            },
            goto="research_sections",
        )

    # ── Node: research_sections ─────────────────────────────────────────────

    async def research_sections(state: PaperMindState) -> Command:
        outline = state.get("research_outline", [])
        db_path = state["db_path"]
        run_dir = state["run_dir"]
        existing_results: dict = state.get("researcher_results", {})
        papers_per_question: dict = state.get("papers_per_question", {})
        default_target = max(3, state.get("target_papers", 20) // max(1, sum(len(ch.get("sub_questions", [])) for ch in outline)))
        logger.info("[orchestrator] research_sections")

        tasks_to_run: list[tuple[str, AgentTask]] = []
        for chapter in outline:
            for sq in chapter.get("sub_questions", []):
                if sq not in existing_results:
                    target = papers_per_question.get(sq, default_target)
                    instruction = json.dumps({"question": sq, "target_papers": target}, ensure_ascii=False)
                    tasks_to_run.append((sq, AgentTask(
                        instruction=instruction, max_iterations=12, db_path=db_path, run_dir=run_dir,
                    )))

        if not tasks_to_run:
            return Command(update={}, goto="check_coverage")

        concurrency = min(3, state.get("agent_concurrency", max_concurrent))
        semaphore = asyncio.Semaphore(concurrency)

        async def _research(sq: str, task: AgentTask) -> tuple[str, dict]:
            async with semaphore:
                for attempt in range(agent_max_retries):
                    agent = ResearcherAgent(config, agent_id=f"researcher-{sq[:20]}")
                    result = await agent.run(task)
                    if result.status != "failed":
                        return sq, result.model_dump()
                    logger.warning("[orchestrator] researcher attempt %d failed for: %s", attempt + 1, sq)
                return sq, result.model_dump()

        new_results = await asyncio.gather(*[_research(sq, t) for sq, t in tasks_to_run])
        updated_results = {**existing_results, **dict(new_results)}

        return Command(update={"researcher_results": updated_results}, goto="check_coverage")

    # ── Node: check_coverage ────────────────────────────────────────────────

    async def check_coverage(state: PaperMindState) -> Command:
        topic = state["research_topic"]
        outline = state.get("research_outline", [])
        researcher_results = state.get("researcher_results", {})
        check_count = state.get("coverage_check_count", 0)
        history = state.get("agent_messages", [])
        logger.info("[orchestrator] check_coverage (attempt %d)", check_count + 1)

        outline_summary = "\n".join(
            f"- {ch['title']}: {', '.join(ch.get('sub_questions', []))}" for ch in outline
        )
        research_summary = "\n".join(
            f"- {sq}: {len(r.get('papers', []))} 篇论文"
            for sq, r in researcher_results.items()
        ) or "（尚无）"

        prompt = f"""\
## 阶段 3 — 覆盖度评估（第 {check_count + 1} 次）

研究大纲：
{outline_summary}

已收集文献情况：
{research_summary}

请评估当前文献是否足以支撑大纲写作。
按阶段 3 的输出格式输出 JSON，直接输出，不要 markdown 代码块。"""

        try:
            data, new_history = await main_agent.chat_json(history, prompt, run_dir=state.get("run_dir", ""))
            sufficient = data.get("sufficient", True)
            weak_areas = data.get("weak_areas", [])
        except Exception as e:
            logger.warning("[orchestrator] check_coverage failed: %s", e)
            sufficient = True
            weak_areas = []
            new_history = history

        logger.info("[orchestrator] coverage sufficient=%s, weak_areas=%s", sufficient, weak_areas)

        if sufficient or check_count >= 2:
            return Command(
                update={"coverage_ok": True, "agent_messages": new_history},
                goto="build_index",
            )

        updated_outline = list(outline)
        for area in weak_areas[:3]:
            if not any(area.lower() in ch["title"].lower() for ch in updated_outline):
                for ch in updated_outline:
                    if ch.get("sub_questions"):
                        ch["sub_questions"].append(f"{area} (supplementary)")
                        break

        return Command(
            update={
                "research_outline": updated_outline,
                "coverage_check_count": check_count + 1,
                "agent_messages": new_history,
            },
            goto="research_sections",
        )

    return {
        "explore_directions": explore_directions,
        "synthesize_outline": synthesize_outline,
        "research_sections": research_sections,
        "check_coverage": check_coverage,
    }
