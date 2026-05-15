"""LangGraph node functions for the outline-drafting pipeline.

All node functions are sync and return ``Command(update=..., goto=...)``.
``build_nodes`` is a factory that closes over Config and CLI args so node
functions don't need to read them off the state object.
"""

import json
import logging
import os
import re
from argparse import Namespace
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command

from deepresearch.config import Config

from . import db
from .llm import build_chat_model
from .prompts import (
    CLUSTER_PROMPT,
    DRAFT_OUTLINE_PROMPT,
    MERGE_OUTLINES_PROMPT,
    WRITER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


CITATION_RE = re.compile(r"\[([^\[\]]+?)\]")


def _f(env: str, default: float) -> float:
    raw = os.getenv(env)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _truncate(text: str, n: int) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "..."


def build_nodes(config: Config, args: Namespace) -> dict:
    cluster_llm = build_chat_model(
        config, temperature=_f("OUTLINE_TEMP_CLUSTER", 0.1), json_mode=True
    )
    draft_llm = build_chat_model(config, temperature=_f("OUTLINE_TEMP_DRAFT", 0.6))
    merge_llm = build_chat_model(config, temperature=_f("OUTLINE_TEMP_MERGE", 0.3))

    def load_papers(state: dict) -> Command:
        db_path = Path(state["in_dir"]) / "papers.db"
        papers = db.load_papers(db_path)
        if not papers:
            raise RuntimeError(f"papers.db 中没有论文: {db_path}")
        logger.info("加载 %d 篇论文", len(papers))
        return Command(
            update={
                "papers": papers,
                "messages": [AIMessage(content=f"loaded {len(papers)} papers")],
            },
            goto="cluster_papers",
        )

    def cluster_papers(state: dict) -> Command:
        papers = state["papers"]
        db_path = Path(state["in_dir"]) / "papers.db"
        existing = db.distinct_directions(db_path)

        if len(papers) <= 4:
            reason = f"papers≤4 ({len(papers)} 篇)"
        elif len(existing) >= 2 and not args.recluster:
            reason = f"DB 已有 {len(existing)} 个 direction（用 --recluster 强制）"
        else:
            reason = ""

        if reason:
            logger.info("跳过聚类: %s", reason)
            assignments = [
                {
                    "paper_id": p["paper_id"],
                    "direction": p["search_direction"] or state["topic"],
                }
                for p in papers
            ]
            return Command(
                update={
                    "cluster_skipped_reason": reason,
                    "cluster_assignments": assignments,
                },
                goto="group_by_direction",
            )

        paper_lines = "\n".join(
            f"- {p['paper_id']} | {p['title']} | "
            f"overview: {_truncate(p.get('overview', ''), 200)} | "
            f"abstract: {_truncate(p.get('abstract', ''), 200)}"
            for p in papers
        )
        user_msg = CLUSTER_PROMPT.format(
            topic=state["topic"],
            n_papers=len(papers),
            min_groups=3,
            max_groups=int(os.getenv("OUTLINE_CLUSTER_MAX", "6")),
            paper_lines=paper_lines,
        )
        logger.info("调用 LLM 聚类 %d 篇论文", len(papers))
        resp = cluster_llm.invoke(
            [
                SystemMessage(content="你是文献分类助手。仅返回合法 JSON。"),
                HumanMessage(content=user_msg),
            ]
        )
        assignments, group_names = _parse_cluster_json(resp.content, papers, state["topic"])

        for a in assignments:
            db.update_search_direction(db_path, a["paper_id"], a["direction"])
        logger.info("聚类完成: %d 组 → %s", len(group_names), group_names)

        return Command(
            update={
                "cluster_assignments": assignments,
                "messages": [resp],
            },
            goto="group_by_direction",
        )

    def group_by_direction(state: dict) -> Command:
        papers = state["papers"]
        by_id = {p["paper_id"]: p for p in papers}
        direction_for: dict[str, str] = {
            a["paper_id"]: a["direction"] for a in state["cluster_assignments"]
        }

        groups: dict[str, list[dict]] = {}
        for p in papers:
            direction = direction_for.get(p["paper_id"]) or state["topic"]
            groups.setdefault(direction, []).append(by_id[p["paper_id"]])

        result = []
        for direction, group_papers in groups.items():
            group_papers.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            result.append({"direction": direction, "papers": group_papers})

        logger.info("分组完成: %d 组", len(result))
        for g in result:
            logger.info("  · [%s] %d 篇", g["direction"], len(g["papers"]))

        return Command(update={"paper_groups": result}, goto="draft_per_group")

    def draft_per_group(state: dict) -> Command:
        outlines: list[str] = []
        sections_per_group = max(2, state["n_sections"] // max(len(state["paper_groups"]), 1))

        for grp in state["paper_groups"]:
            paper_blocks = "\n\n".join(
                f"### {p['title']}\n"
                f"- overview: {_truncate(p.get('overview', ''), 300)}\n"
                f"- abstract: {_truncate(p.get('abstract', ''), 400)}"
                for p in grp["papers"]
            )
            user_msg = DRAFT_OUTLINE_PROMPT.format(
                topic=state["topic"],
                direction=grp["direction"],
                n_sections=sections_per_group,
                n_subsections=state["n_subsections"],
                paper_blocks=paper_blocks,
            )
            messages = [
                SystemMessage(content=WRITER_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ]

            outline = _invoke_with_one_retry(
                draft_llm, messages, label=f"draft[{grp['direction']}]"
            )
            if outline is None:
                outline = (
                    f"# {grp['direction']}\n\n_本组未能成功起草（LLM 调用失败）_\n"
                )
            outlines.append(outline)

        logger.info("起草完成: %d 段子大纲", len(outlines))
        return Command(update={"group_outlines": outlines}, goto="merge_outlines")

    def merge_outlines(state: dict) -> Command:
        outlines = state["group_outlines"]
        outlines_joined = "\n\n---\n\n".join(outlines)
        user_msg = MERGE_OUTLINES_PROMPT.format(
            n_groups=len(outlines),
            topic=state["topic"],
            n_sections=state["n_sections"],
            n_subsections=state["n_subsections"],
            outlines_joined=outlines_joined,
        )
        messages = [
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
        logger.info("合并 %d 段子大纲", len(outlines))
        merged = _invoke_with_one_retry(merge_llm, messages, label="merge")
        if merged is None:
            logger.error("合并失败,直接拼接子大纲作为兜底")
            merged = (
                f"# {state['topic']}（自动合并失败的兜底版）\n\n"
                + outlines_joined
            )
        return Command(update={"final_outline": merged}, goto="render_references")

    def render_references(state: dict) -> Command:
        papers_by_title = {p["title"]: p for p in state["papers"]}
        cited_titles: list[str] = []
        seen: set[str] = set()
        for raw in CITATION_RE.findall(state["final_outline"]):
            for chunk in raw.split("|"):
                title = chunk.strip()
                if title and title not in seen:
                    seen.add(title)
                    cited_titles.append(title)

        lines = ["## References", ""]
        missing: list[str] = []
        for i, title in enumerate(cited_titles, 1):
            paper = papers_by_title.get(title)
            if paper is None:
                missing.append(title)
                lines.append(f"{i}. **{title}** _(not found in papers.db)_")
                continue
            lines.append(f"{i}. **{paper['title']}**")
            authors = paper.get("authors") or []
            if authors:
                lines.append(f"   - Authors: {', '.join(authors)}")
            if paper.get("venue"):
                lines.append(f"   - Venue: {paper['venue']}")
            if paper.get("source_url"):
                lines.append(f"   - URL: {paper['source_url']}")
            if paper.get("artifact_rel_path"):
                lines.append(f"   - Local: `{paper['artifact_rel_path']}`")
            if paper.get("bibtex"):
                lines.append("")
                lines.append("   ```bibtex")
                for ln in paper["bibtex"].strip().splitlines():
                    lines.append(f"   {ln}")
                lines.append("   ```")
            lines.append("")
        references_md = "\n".join(lines)

        if missing:
            logger.warning("有 %d 个引用在 papers.db 中未找到: %s", len(missing), missing)

        out_path = Path(args.out) if args.out else Path(state["in_dir"]) / "outline.md"
        full = (
            f"# {state['topic']}\n\n"
            + state["final_outline"].strip()
            + "\n\n"
            + references_md
            + "\n"
        )
        out_path.write_text(full, encoding="utf-8")
        logger.info("已写出大纲: %s (%d 个引用,%d 篇缺失)", out_path, len(cited_titles), len(missing))

        return Command(
            update={
                "references_md": references_md,
                "output_path": str(out_path),
            },
            goto=END,
        )

    return {
        "load_papers": load_papers,
        "cluster_papers": cluster_papers,
        "group_by_direction": group_by_direction,
        "draft_per_group": draft_per_group,
        "merge_outlines": merge_outlines,
        "render_references": render_references,
    }


def _invoke_with_one_retry(llm, messages, *, label: str) -> str | None:
    try:
        return llm.invoke(messages).content
    except Exception as e:
        logger.warning("[%s] 第一次失败: %s — 重试", label, e)
    try:
        return llm.invoke(messages).content
    except Exception as e:
        logger.error("[%s] 重试仍失败: %s", label, e)
        return None


def _parse_cluster_json(
    raw: str, papers: list[dict], fallback_direction: str
) -> tuple[list[dict], list[str]]:
    """Parse the cluster LLM output. On any failure, return one big group."""
    try:
        data = json.loads(raw)
        groups = data.get("groups", [])
        valid_ids = {p["paper_id"] for p in papers}
        assignments: list[dict] = []
        names: list[str] = []
        seen: set[str] = set()
        for g in groups:
            name = (g.get("name") or "").strip() or fallback_direction
            for pid in g.get("paper_ids", []):
                if pid in valid_ids and pid not in seen:
                    assignments.append({"paper_id": pid, "direction": name})
                    seen.add(pid)
            names.append(name)
        # Catch papers the LLM forgot to assign
        for p in papers:
            if p["paper_id"] not in seen:
                assignments.append(
                    {"paper_id": p["paper_id"], "direction": fallback_direction}
                )
        if not assignments:
            raise ValueError("空的 groups")
        return assignments, names
    except Exception as e:
        logger.error("聚类 JSON 解析失败: %s,回退到单组", e)
        return (
            [{"paper_id": p["paper_id"], "direction": fallback_direction} for p in papers],
            [fallback_direction],
        )
