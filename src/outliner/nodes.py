"""LangGraph node functions for the outline-drafting pipeline.

All node functions are sync and return ``Command(update=..., goto=...)``.
``build_nodes`` is a factory that closes over Config and CLI args so node
functions don't need to read them off the state object.
"""

import json
import logging
import os
import re
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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
    PLAN_CHAPTERS_PROMPT,
    REVIEW_OUTLINE_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    REVISE_OUTLINE_PROMPT,
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
        config, temperature=_f("OUTLINE_TEMP_CLUSTER", 0.1)
    )
    plan_llm = build_chat_model(config, temperature=_f("OUTLINE_TEMP_PLAN", 0.1))
    draft_llm = build_chat_model(config, temperature=_f("OUTLINE_TEMP_DRAFT", 0.6))
    merge_llm = build_chat_model(config, temperature=_f("OUTLINE_TEMP_MERGE", 0.3))
    review_llm = build_chat_model(
        config, temperature=_f("OUTLINE_TEMP_REVIEW", 0.2), json_mode=True
    )
    revise_llm = build_chat_model(config, temperature=_f("OUTLINE_TEMP_REVISE", 0.4))

    max_revisions = int(os.getenv("OUTLINE_MAX_REVISIONS", "2"))

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
        messages = [
            SystemMessage(content="你是文献分类助手。仅返回合法 JSON。非MarkDown格式"),
            HumanMessage(content=user_msg),
        ]
        assignments = group_names = None
        started_at = time.monotonic()
        for attempt in range(1, 4):
            resp = cluster_llm.invoke(messages)
            assignments, group_names = _parse_cluster_json(resp.content, papers, state["topic"])
            if len(group_names) > 1:
                break
            if attempt < 3:
                logger.warning("聚类第 %d 次尝试解析失败，重试", attempt)
        assignments = assignments or [{"paper_id": p["paper_id"], "direction": state["topic"]} for p in papers]
        group_names = group_names or [state["topic"]]

        _save_cluster_log(
            topic=state["topic"],
            model=config.llm_model,
            prompt=user_msg,
            response=resp.content,
            group_names=group_names,
            duration_ms=int((time.monotonic() - started_at) * 1000),
        )

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

        return Command(update={"paper_groups": result}, goto="plan_chapters")

    def plan_chapters(state: dict) -> Command:
        groups = state["paper_groups"]
        groups_block = "\n\n".join(
            "## {}\n{}".format(
                g["direction"],
                "\n".join(
                    f"- {p['paper_id']} | {p['title']} | {_truncate(p.get('overview', ''), 150)}"
                    for p in g["papers"]
                ),
            )
            for g in groups
        )
        user_msg = PLAN_CHAPTERS_PROMPT.format(
            topic=state["topic"],
            n_sections=state["n_sections"],
            n_subsections=state["n_subsections"],
            groups_block=groups_block,
        )
        logger.info("规划章节结构 (%d 个子主题)", len(groups))
        messages = [
            SystemMessage(content="你是文献综述结构规划专家。仅返回合法 JSON。"),
            HumanMessage(content=user_msg),
        ]
        chapter_plan: list[dict] = []
        survey_title = state["topic"]
        for attempt in range(1, 4):
            resp = _invoke_with_one_retry(plan_llm, messages, label="plan_chapters")
            if not resp:
                if attempt < 3:
                    logger.warning("章节规划第 %d 次调用失败，重试", attempt)
                continue
            try:
                text = resp.strip()
                if text.startswith("```"):
                    text = re.sub(r"^```[^\n]*\n?", "", text)
                    text = re.sub(r"\n?```\s*$", "", text)
                data = json.loads(text)
                chapter_plan = data.get("chapters", [])
                survey_title = data.get("title", state["topic"]) or state["topic"]
                if chapter_plan:
                    break
            except (json.JSONDecodeError, KeyError):
                if attempt < 3:
                    logger.warning("章节规划 JSON 解析失败 (第 %d 次)，重试", attempt)

        if not chapter_plan:
            logger.warning("章节规划失败，回退到按子主题直接起草")
            chapter_plan = [
                {"title": g["direction"], "directions": [g["direction"]]}
                for g in groups
            ]

        logger.info("章节规划完成: %d 章，综述标题: %s", len(chapter_plan), survey_title)
        for ch in chapter_plan:
            logger.info("  · %s → %s", ch["title"], ch.get("directions", []))

        return Command(update={"chapter_plan": chapter_plan, "survey_title": survey_title}, goto="draft_per_group")

    def draft_per_group(state: dict) -> Command:
        groups = state["paper_groups"]
        chapter_plan = state["chapter_plan"]
        groups_by_direction = {g["direction"]: g for g in groups}

        # Map each chapter to its primary paper group (first matching direction)
        def _find_group(direction: str) -> dict | None:
            if direction in groups_by_direction:
                return groups_by_direction[direction]
            # fuzzy fallback: find group whose name contains or is contained by direction
            dl = direction.lower()
            for name, grp in groups_by_direction.items():
                nl = name.lower()
                if dl in nl or nl in dl:
                    return grp
            # last resort: longest common substring match
            best, best_len = None, 0
            for name, grp in groups_by_direction.items():
                common = sum(1 for a, b in zip(direction, name) if a == b)
                if common > best_len:
                    best, best_len = grp, common
            return best

        def _papers_for_chapter(chapter: dict) -> list[dict]:
            for direction in chapter.get("directions", []):
                grp = _find_group(direction)
                if grp:
                    return grp["papers"]
            # collect from all matching directions, dedup
            seen_ids: set[str] = set()
            papers: list[dict] = []
            for direction in chapter.get("directions", []):
                grp = _find_group(direction)
                if grp:
                    for p in grp["papers"]:
                        if p["paper_id"] not in seen_ids:
                            seen_ids.add(p["paper_id"])
                            papers.append(p)
            return papers

        # Only draft non-Introduction/Conclusion chapters
        draft_chapters = [
            ch for ch in chapter_plan
            if ch["title"].lower() not in {"introduction", "conclusion"}
        ]

        def _draft_one(chapter: dict) -> tuple[int, str]:
            idx = draft_chapters.index(chapter)
            papers = _papers_for_chapter(chapter)
            paper_blocks = "\n\n".join(
                f"### {p['title']}\n"
                f"- overview: {_truncate(p.get('overview', ''), 300)}\n"
                f"- abstract: {_truncate(p.get('abstract', ''), 400)}"
                for p in papers
            ) or "（本章节暂无对应论文）"
            subsections = chapter.get("subsections") or []
            subsections_block = "\n".join(f"- {s}" for s in subsections) or "（由你自行规划）"
            user_msg = DRAFT_OUTLINE_PROMPT.format(
                topic=state["topic"],
                chapter_title=chapter["title"],
                subsections_block=subsections_block,
                paper_blocks=paper_blocks,
            )
            messages = [
                SystemMessage(content=WRITER_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ]
            started_at = time.monotonic()
            outline = _invoke_with_one_retry(
                draft_llm, messages, label=f"draft[{chapter['title']}]"
            )
            if outline is None:
                outline = f"# {chapter['title']}\n\n_本章节未能成功起草（LLM 调用失败）_\n"
            _save_draft_log(
                chapter_title=chapter["title"],
                model=config.llm_model,
                prompt=user_msg,
                response=outline,
                duration_ms=int((time.monotonic() - started_at) * 1000),
            )
            return idx, outline

        outlines_map: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=len(draft_chapters) or 1) as pool:
            futures = {pool.submit(_draft_one, ch): ch for ch in draft_chapters}
            for fut in as_completed(futures):
                idx, outline = fut.result()
                outlines_map[idx] = outline

        outlines = [outlines_map[i] for i in range(len(draft_chapters))]
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
        return Command(update={"final_outline": merged}, goto="review_outline")

    def review_outline(state: dict) -> Command:
        outline = state["final_outline"]
        revision_count = state.get("revision_count", 0)
        paper_titles = "\n".join(f"- {p['title']}" for p in state["papers"])

        user_msg = REVIEW_OUTLINE_PROMPT.format(
            topic=state["topic"],
            revision_round=revision_count + 1,
            outline=outline,
            n_papers=len(state["papers"]),
            paper_titles=paper_titles,
        )
        review_messages = [
            SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
        logger.info("评审大纲 (第 %d 轮)", revision_count + 1)

        approved = True
        feedback = ""
        resp = None

        started_at = time.monotonic()
        for attempt in range(1, 4):
            resp = _invoke_with_one_retry(review_llm, review_messages, label="review")
            if not resp:
                if attempt < 3:
                    logger.warning("评审第 %d 次调用失败，重试", attempt)
                continue
            sufficient, feedback = _parse_review_json(resp)
            if sufficient is not None:
                approved = sufficient
                logger.info("评审结果: %s", "通过" if approved else "不通过")
                if feedback:
                    logger.info("  反馈: %s", feedback[:200])
                break
            if attempt < 3:
                logger.warning("评审 JSON 解析失败 (第 %d 次)，重试", attempt)
            else:
                logger.warning("评审 JSON 解析失败，视为通过")
                approved = True

        _save_review_log(
            topic=state["topic"],
            model=config.llm_model,
            revision_count=revision_count,
            prompt=user_msg,
            response=resp or "",
            approved=approved,
            score=0,
            feedback=feedback,
            duration_ms=int((time.monotonic() - started_at) * 1000),
        )

        if approved or revision_count >= max_revisions:
            if not approved:
                logger.warning("已达最大修订次数 (%d),强制通过", max_revisions)
            return Command(
                update={
                    "review_feedback": feedback,
                    "messages": [AIMessage(content=f"review: approved={approved}")],
                },
                goto="render_references",
            )

        # Rejected: revise the outline
        logger.info("大纲未通过评审,开始修订 (第 %d → %d 轮)", revision_count + 1, revision_count + 2)

        revise_msg = REVISE_OUTLINE_PROMPT.format(
            topic=state["topic"],
            n_sections=state["n_sections"],
            n_subsections=state["n_subsections"],
            current_outline=outline,
            feedback=feedback or "无具体反馈",
        )
        revised = _invoke_with_one_retry(
            revise_llm,
            [
                SystemMessage(content=WRITER_SYSTEM_PROMPT),
                HumanMessage(content=revise_msg),
            ],
            label="revise",
        )
        if revised is None:
            logger.error("修订失败,保留原大纲继续")
            revised = outline

        return Command(
            update={
                "final_outline": revised,
                "revision_count": revision_count + 1,
                "review_feedback": feedback,
                "messages": [AIMessage(content=f"revised outline (round {revision_count + 2})")],
            },
            goto="review_outline",
        )

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
        survey_title = state.get("survey_title") or state["topic"]
        full = (
            f"# {survey_title}\n\n"
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
        "plan_chapters": plan_chapters,
        "draft_per_group": draft_per_group,
        "merge_outlines": merge_outlines,
        "review_outline": review_outline,
        "render_references": render_references,
    }


_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _save_cluster_log(
    topic: str,
    model: str,
    prompt: str,
    response: str,
    group_names: list[str],
    duration_ms: int,
) -> None:
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        filename = _LOG_DIR / f"cluster-{ts}.json"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "duration_ms": duration_ms,
            "input": {"model": model, "prompt": prompt},
            "output": {"raw": response, "group_names": group_names},
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
        logger.info("聚类日志已保存: %s", filename)
    except Exception as e:
        logger.warning("保存聚类日志失败: %s", e)


def _save_draft_log(
    chapter_title: str,
    model: str,
    prompt: str,
    response: str,
    duration_ms: int,
) -> None:
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe_title = re.sub(r"[^\w一-鿿-]", "_", chapter_title)[:30]
        filename = _LOG_DIR / f"draft-{ts}-{safe_title}.json"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chapter_title": chapter_title,
            "duration_ms": duration_ms,
            "input": {"model": model, "prompt": prompt},
            "output": {"draft": response},
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
        logger.info("起草日志已保存: %s", filename)
    except Exception as e:
        logger.warning("保存起草日志失败: %s", e)


def _save_review_log(
    topic: str,
    model: str,
    revision_count: int,
    prompt: str,
    response: str,
    approved: bool,
    score: int,
    feedback: str,
    duration_ms: int,
) -> None:
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        filename = _LOG_DIR / f"review-{ts}-r{revision_count}.json"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "revision_count": revision_count,
            "duration_ms": duration_ms,
            "input": {"model": model, "prompt": prompt},
            "output": {
                "raw": response,
                "approved": approved,
                "score": score,
                "feedback": feedback,
            },
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
        logger.info("评审日志已保存: %s", filename)
    except Exception as e:
        logger.warning("保存评审日志失败: %s", e)


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


def _parse_review_json(text: str) -> tuple[bool | None, str]:
    """Parse new-format review JSON. Returns (sufficient, feedback) or (None, '') on failure."""
    try:
        raw = text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[^\n]*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw)
        review = json.loads(raw)
        sufficient = str(review.get("SUFFICIENT", "yes")).strip().lower() == "yes"
        feedback = review.get("FEEDBACK", "")
        return sufficient, feedback
    except Exception:
        return None, ""


def _parse_cluster_json(
    raw: str, papers: list[dict], fallback_direction: str
) -> tuple[list[dict], list[str]]:
    """Parse the cluster LLM output. On any failure, return one big group."""
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[^\n]*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        data = json.loads(text)
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
