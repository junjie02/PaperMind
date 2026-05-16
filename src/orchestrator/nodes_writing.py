"""Orchestrator nodes for Phase 4-6 (writing, review, polish, merge)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
from pathlib import Path

import httpx
from langgraph.types import Command

from agents.polisher import ConsistencyCheckerAgent, PolisherAgent
from agents.reviewer import ReviewerAgent
from agents.writer import WriterAgent
from orchestrator.state import PaperMindState
from rag.indexer import build_index_from_run
from shared.config import Config
from shared.llm import make_llm
from shared.models import AgentTask

logger = logging.getLogger(__name__)


async def _backfill_artifacts(run_dir: Path) -> None:
    """Scan papers.db for papers without artifacts; download PDF or generate MD."""
    db_path = run_dir / "papers.db"
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT paper_id, title, authors, abstract, overview, pdf_url, "
        "source_url, venue, published_at, arxiv_id, artifact_rel_path FROM papers"
    ).fetchall()
    conn.close()

    pdfs_dir = run_dir / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    updates: list[tuple[str, str]] = []

    async with httpx.AsyncClient(proxy=proxy, timeout=30, follow_redirects=True) as client:
        for row in rows:
            paper_id, title, authors, abstract, overview, pdf_url, source_url, venue, published_at, arxiv_id, artifact_rel_path = row
            if artifact_rel_path:
                if (run_dir / artifact_rel_path).exists():
                    continue

            safe_name = paper_id.replace("/", "-").replace(":", "-")[:60]

            # Try downloading PDF
            url_to_try = pdf_url
            if not url_to_try and arxiv_id:
                url_to_try = f"https://arxiv.org/pdf/{arxiv_id}"

            if url_to_try:
                try:
                    resp = await client.get(url_to_try)
                    resp.raise_for_status()
                    if b"%PDF-" in resp.content[:20]:
                        filepath = pdfs_dir / f"{safe_name}.pdf"
                        filepath.write_bytes(resp.content)
                        rel_path = f"pdfs/{safe_name}.pdf"
                        updates.append((rel_path, paper_id))
                        logger.info("[backfill] downloaded PDF: %s", title[:50])
                        continue
                except Exception as e:
                    logger.debug("[backfill] PDF download failed for %s: %s", paper_id, e)

            # Fallback: generate MD from metadata
            md_lines = [f"# {title}\n"]
            if authors:
                md_lines.append(f"**Authors:** {authors}\n")
            if venue:
                md_lines.append(f"**Venue:** {venue}\n")
            if published_at:
                md_lines.append(f"**Published:** {published_at}\n")
            if arxiv_id:
                md_lines.append(f"**arXiv:** {arxiv_id}\n")
            if source_url:
                md_lines.append(f"**URL:** {source_url}\n")
            if abstract:
                md_lines.append(f"\n## Abstract\n\n{abstract}\n")
            if overview:
                md_lines.append(f"\n## Overview\n\n{overview}\n")

            md_path = pdfs_dir / f"{safe_name}.md"
            md_path.write_text("\n".join(md_lines), encoding="utf-8")
            rel_path = f"pdfs/{safe_name}.md"
            updates.append((rel_path, paper_id))
            logger.info("[backfill] generated MD: %s", title[:50])

    # Batch update artifact_rel_path in DB
    if updates:
        conn = sqlite3.connect(str(db_path))
        conn.executemany(
            "UPDATE papers SET artifact_rel_path = ? WHERE paper_id = ?", updates
        )
        conn.commit()
        conn.close()
        logger.info("[backfill] updated %d papers with artifacts", len(updates))


def _resolve_citations(full_text: str, db_path: str) -> tuple[str, list[str]]:
    """Replace [paper title] citations with [N] and generate references list.

    Uses embedding similarity for robust matching when exact/fuzzy match fails.
    """
    if not Path(db_path).exists():
        return full_text, []

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT title, authors, venue, published_at, bibtex FROM papers").fetchall()
    conn.close()

    if not rows:
        return full_text, []

    # Build title lookup
    title_db: dict[str, tuple] = {}
    db_titles: list[str] = []
    for title, authors, venue, published_at, bibtex in rows:
        key = title.strip().lower()
        year = (published_at or "")[:4]
        title_db[key] = (title, authors, venue, year, bibtex)
        db_titles.append(key)

    # Build embedding index for all DB titles
    import numpy as np
    from rag.retriever import _get_model
    model = _get_model("all-MiniLM-L6-v2")
    db_embeddings = model.encode(db_titles, show_progress_bar=False)
    db_embeddings = np.array(db_embeddings, dtype="float32")
    norms = np.linalg.norm(db_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    db_embeddings = db_embeddings / norms

    # Find all [...] citations
    citation_pattern = re.compile(r"\[([^\[\]]{5,})\]")
    citation_map: dict[str, int] = {}  # matched_key → number
    citation_info: dict[int, tuple] = {}  # number → (title, authors, venue, year, bibtex)
    counter = [0]
    unmatched: list[str] = []

    def _match_title(raw_title: str) -> int | None:
        t = raw_title.strip().lower()
        # 1. Exact match
        if t in title_db:
            if t not in citation_map:
                counter[0] += 1
                citation_map[t] = counter[0]
                citation_info[counter[0]] = title_db[t]
            return citation_map[t]
        # 2. Substring match
        for db_key, db_val in title_db.items():
            if t[:30] in db_key or db_key[:30] in t:
                if db_key not in citation_map:
                    counter[0] += 1
                    citation_map[db_key] = counter[0]
                    citation_info[counter[0]] = db_val
                return citation_map[db_key]
        # 3. Embedding similarity match (threshold 0.75)
        query_emb = model.encode([t], show_progress_bar=False)
        query_emb = np.array(query_emb, dtype="float32")
        query_emb = query_emb / np.linalg.norm(query_emb)
        scores = (db_embeddings @ query_emb.T).flatten()
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score >= 0.75:
            db_key = db_titles[best_idx]
            if db_key not in citation_map:
                counter[0] += 1
                citation_map[db_key] = counter[0]
                citation_info[counter[0]] = title_db[db_key]
            return citation_map[db_key]
        return None

    def _replace(m: re.Match) -> str:
        content = m.group(1)
        if content.strip().isdigit():
            return m.group(0)
        titles = [t.strip() for t in content.split(";") if t.strip()]
        nums = []
        for title in titles:
            num = _match_title(title)
            if num is not None:
                nums.append(str(num))
            else:
                unmatched.append(title)
                logger.warning("[merge_final] unmatched citation: %s", title[:60])
                nums.append("?")
        return f"[{', '.join(nums)}]"

    processed = citation_pattern.sub(_replace, full_text)

    # Generate references list (unified plain text format)
    ref_lines: list[str] = []
    for num in sorted(citation_info.keys()):
        title, authors_json, venue, year, bibtex = citation_info[num]
        try:
            authors_list = json.loads(authors_json) if authors_json else []
        except (json.JSONDecodeError, TypeError):
            authors_list = []
        authors_str = ", ".join(authors_list[:3])
        if len(authors_list) > 3:
            authors_str += " et al."
        if not authors_str:
            authors_str = "Unknown"
        venue_str = f", {venue}" if venue else ""
        year_str = f", {year}" if year else ""
        ref_lines.append(f"[{num}] {authors_str}, \"{title}\"{venue_str}{year_str}.")

    if unmatched:
        logger.warning("[merge_final] %d citations could not be matched to database", len(unmatched))

    return processed, ref_lines


_TRANSITION_SYSTEM = "你是学术写作专家，负责生成章节间的过渡句。"
_TRANSITION_USER = """\
上一章节结尾：
{prev_tail}

下一章节开头：
{next_head}

请生成一句自然的过渡句（中文，学术风格），连接两个章节。只输出这一句话，不要其他内容。
"""


def _save_section_md(run_dir: str, key: str, draft: str) -> None:
    if not run_dir or not draft:
        return
    sections_dir = Path(run_dir) / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)
    safe_name = key.replace("/", "-").replace(":", "-")[:60]
    path = sections_dir / f"{safe_name}.md"
    path.write_text(f"# {key}\n\n{draft}\n", encoding="utf-8")


def _save_draft_md(run_dir: str, outline: list[dict], drafts: dict) -> None:
    if not run_dir:
        return
    parts = []
    for chapter in outline:
        sqs = chapter.get("sub_questions", [])
        if sqs:
            parts.append(f"# {chapter['title']}\n")
            for sq in sqs:
                if sq in drafts:
                    parts.append(f"## {sq}\n\n{drafts[sq]}\n")
        else:
            title = chapter["title"]
            if title in drafts:
                parts.append(f"# {title}\n\n{drafts[title]}\n")
    path = Path(run_dir) / "draft.md"
    path.write_text("\n".join(parts), encoding="utf-8")


def _strip_draft_headings(text: str) -> str:
    """Remove H1/H2 heading lines that the LLM may have added. Keep H3+ intact."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Remove lines starting with # or ## but not ### or deeper
        if stripped.startswith("##") and not stripped.startswith("###"):
            continue
        if stripped.startswith("#") and not stripped.startswith("##"):
            continue
        cleaned.append(line)
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    return "\n".join(cleaned)


def _renumber_h3(text: str, chapter_num: int, section_num: int) -> str:
    """Renumber ### headings to ### {chapter}.{sub}.{seq} format."""
    lines = text.split("\n")
    result = []
    h3_counter = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("### "):
            h3_counter += 1
            # Remove any existing numbering (e.g., "### 3.1 Title" → "Title")
            title_part = stripped[4:].strip()
            # Strip leading numbers like "3.1 " or "1.2.3 "
            import re as _re
            title_part = _re.sub(r"^[\d.]+\s*", "", title_part)
            if section_num > 0:
                line = f"### {chapter_num}.{section_num}.{h3_counter} {title_part}"
            else:
                line = f"### {chapter_num}.{h3_counter} {title_part}"
        result.append(line)
    return "\n".join(result)


async def _generate_intro_conclusion(
    topic: str, body_summaries: list[str], section_type: str, llm
) -> str:
    """Generate Introduction or Conclusion based on body content summaries."""
    from langchain_core.messages import HumanMessage, SystemMessage

    summaries_text = "\n".join(body_summaries[:20])

    if section_type == "introduction":
        prompt = f"""\
你是学术文献综述写作专家。请为以下综述撰写引言部分（1-3段）。

研究课题：{topic}

各章节摘要：
{summaries_text}

要求：
- 介绍研究背景和意义
- 概述本综述涵盖的主要技术方向
- 不要包含具体的技术细节或实验数据
- 不要使用标题或编号
- 直接输出引言正文"""
    else:
        prompt = f"""\
你是学术文献综述写作专家。请为以下综述撰写结论与展望部分（2-4段）。

研究课题：{topic}

各章节摘要：
{summaries_text}

要求：
- 总结各技术方向的核心进展和贡献
- 指出当前的局限性和未解决的挑战
- 展望未来研究方向
- 不要重复正文中的具体技术细节和实验数据
- 不要使用标题或编号
- 直接输出结论正文"""

    try:
        messages = [
            SystemMessage(content="你是学术文献综述写作专家。"),
            HumanMessage(content=prompt),
        ]
        response = await llm.ainvoke(messages)
        return response.content.strip()
    except Exception as e:
        logger.warning("[merge_final] generate %s failed: %s", section_type, e)
        return ""


def build_writing_nodes(config: Config):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    transition_chain = (
        ChatPromptTemplate.from_messages([("system", _TRANSITION_SYSTEM), ("human", _TRANSITION_USER)])
        | make_llm(config, temperature=0.3)
        | StrOutputParser()
    )
    max_concurrent = config.max_concurrent_agents
    agent_max_retries = config.agent_max_retries

    # ── Node: build_index ───────────────────────────────────────────────────

    async def build_index(state: PaperMindState) -> Command:
        run_dir = Path(state["run_dir"])
        logger.info("[orchestrator] build_index: %s", run_dir)

        # Backfill: download missing PDFs or generate MD for papers without artifacts
        await _backfill_artifacts(run_dir)

        index_path = run_dir / "faiss.index"
        if not index_path.exists():
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                build_index_from_run,
                run_dir,
                config.embedding_model,
            )
            logger.info("[orchestrator] FAISS index built")
        else:
            logger.info("[orchestrator] FAISS index already exists, skipping")

        return Command(update={"faiss_built": True}, goto="write_sections")

    # ── Node: write_sections ────────────────────────────────────────────────

    async def write_sections(state: PaperMindState) -> Command:
        outline = state.get("research_outline", [])
        run_dir = state["run_dir"]
        existing_drafts: dict = state.get("section_drafts", {})
        max_revisions = 0 if state.get("skip_review") else state.get("max_revisions", 2)
        logger.info("[orchestrator] write_sections (skip_review=%s)", state.get("skip_review", False))

        # Collect body sections (skip intro/conclusion)
        body_sections: list[dict] = []
        for chapter in outline:
            chapter_title = chapter["title"]
            sub_questions = chapter.get("sub_questions", [])
            title_lower = chapter_title.lower()

            if not sub_questions and (title_lower.startswith("intro") or title_lower.startswith("conclusion")):
                continue
            elif not sub_questions:
                body_sections.append({
                    "key": chapter_title,
                    "section_title": chapter_title,
                    "outline_text": chapter.get("description", chapter_title),
                })
            else:
                for sq in sub_questions:
                    body_sections.append({
                        "key": sq,
                        "section_title": sq,
                        "outline_text": f"## {chapter_title}\n\n{sq}\n\n{chapter.get('description', '')}",
                    })

        tasks_to_run = [s for s in body_sections if s["key"] not in existing_drafts]
        if not tasks_to_run:
            return Command(update={}, goto="polish_sections")

        semaphore = asyncio.Semaphore(min(3, state.get("agent_concurrency", max_concurrent)))

        # Build adjacent context map
        all_keys = [s["key"] for s in body_sections]

        def _adjacent_context(key: str, drafts: dict) -> str:
            try:
                idx = all_keys.index(key)
            except ValueError:
                return ""
            parts = []
            if idx > 0 and all_keys[idx - 1] in drafts:
                parts.append(f"[前一节结尾]\n{drafts[all_keys[idx - 1]][-500:]}")
            if idx < len(all_keys) - 1 and all_keys[idx + 1] in drafts:
                parts.append(f"[后一节开头]\n{drafts[all_keys[idx + 1]][:500]}")
            return "\n\n".join(parts)

        async def _write_and_review(section_info: dict) -> tuple[str, str]:
            """Write a section, then review+revise loop (max 2 revisions)."""
            key = section_info["key"]
            adjacent = _adjacent_context(key, existing_drafts)
            instruction = json.dumps({
                "section_title": section_info["section_title"],
                "outline_text": section_info["outline_text"],
                "adjacent_context": adjacent,
            }, ensure_ascii=False)
            task = AgentTask(instruction=instruction, max_iterations=12, run_dir=run_dir)

            async with semaphore:
                # Initial write
                writer = WriterAgent(config, agent_id=f"writer-{key[:20]}")
                result = await writer.run(task)
                draft = result.draft_text or ""

                # Review + revise loop
                for rev in range(max_revisions):
                    if not draft:
                        break
                    review_instruction = json.dumps({"section_title": key, "draft_text": draft}, ensure_ascii=False)
                    review_task = AgentTask(instruction=review_instruction, run_dir=run_dir)
                    reviewer = ReviewerAgent(config, agent_id=f"reviewer-{key[:20]}")
                    review_result = await reviewer.run(review_task)

                    if not review_result.issues:
                        logger.info("[orchestrator] %s passed review on revision %d", key[:30], rev)
                        break

                    # Feed issues back to writer (preserving memory)
                    feedback = "\n".join(f"- {i}" for i in review_result.issues)
                    logger.info("[orchestrator] %s revision %d: %d issues", key[:30], rev + 1, len(review_result.issues))
                    revise_result = await writer.revise(task, feedback)
                    draft = revise_result.draft_text or draft

            # Save individual section md
            _save_section_md(run_dir, key, draft)
            return key, draft

        results = await asyncio.gather(*[_write_and_review(s) for s in tasks_to_run])

        updated_drafts = dict(existing_drafts)
        for key, draft in results:
            if draft:
                updated_drafts[key] = draft

        # Save combined draft.md
        _save_draft_md(run_dir, outline, updated_drafts)

        return Command(update={"section_drafts": updated_drafts}, goto="polish_sections")

    # ── Node: polish_sections ───────────────────────────────────────────────

    async def polish_sections(state: PaperMindState) -> Command:
        section_drafts = state.get("section_drafts", {})
        run_dir = state.get("run_dir", "")
        logger.info("[orchestrator] polish_sections: %d sections", len(section_drafts))

        semaphore = asyncio.Semaphore(min(3, state.get("agent_concurrency", max_concurrent)))

        async def _polish(key: str, draft: str) -> tuple[str, str]:
            instruction = json.dumps({"section_title": key, "draft_text": draft}, ensure_ascii=False)
            task = AgentTask(instruction=instruction, run_dir=run_dir)
            async with semaphore:
                agent = PolisherAgent(config, agent_id=f"polisher-{key[:20]}")
                result = await agent.run(task)
            return key, result.polished_text or draft

        polished = await asyncio.gather(*[_polish(k, v) for k, v in section_drafts.items() if v])

        polished_sections = {k: v for k, v in polished}
        return Command(update={"polished_sections": polished_sections}, goto="check_consistency")

    # ── Node: check_consistency ─────────────────────────────────────────────

    async def check_consistency(state: PaperMindState) -> Command:
        polished_sections = state.get("polished_sections", {})
        outline = state.get("research_outline", [])
        logger.info("[orchestrator] check_consistency")

        # Build ordered section list
        ordered_keys: list[str] = []
        for chapter in outline:
            sqs = chapter.get("sub_questions", [])
            if sqs:
                ordered_keys.extend(sqs)
            else:
                ordered_keys.append(chapter["title"])

        sections_for_check = []
        for key in ordered_keys:
            if key in polished_sections:
                text = polished_sections[key]
                sections_for_check.append({
                    "title": key,
                    "head": text[:300],
                    "tail": text[-300:],
                })

        instruction = json.dumps({"sections": sections_for_check}, ensure_ascii=False)
        task = AgentTask(instruction=instruction, run_dir=state.get("run_dir", ""))
        agent = ConsistencyCheckerAgent(config)
        result = await agent.run(task)
        report = result.metadata[0] if result.metadata else {}

        return Command(update={"consistency_report": report}, goto="merge_final")

    # ── Node: merge_final ───────────────────────────────────────────────────

    async def merge_final(state: PaperMindState) -> Command:
        polished_sections = state.get("polished_sections", {})
        outline = state.get("research_outline", [])
        consistency_report = state.get("consistency_report", {})
        topic = state["research_topic"]
        run_dir = Path(state["run_dir"])
        logger.info("[orchestrator] merge_final")

        # Build ordered section list
        ordered_keys: list[str] = []
        for chapter in outline:
            sqs = chapter.get("sub_questions", [])
            if sqs:
                ordered_keys.extend(sqs)
            else:
                ordered_keys.append(chapter["title"])

        # Apply terminology replacements
        terminology_issues = consistency_report.get("terminology_issues", [])
        sections = dict(polished_sections)
        for issue in terminology_issues:
            recommended = issue.get("recommended", "")
            variants = issue.get("variants", [])
            # Skip if recommended looks like a sentence rather than a term
            if not recommended or len(recommended) > 20 or "，" in recommended or "。" in recommended:
                continue
            if recommended and variants:
                for key in sections:
                    for variant in variants:
                        if variant != recommended:
                            sections[key] = sections[key].replace(variant, recommended)

        # Insert transition sentences for flagged transitions
        transition_issues = consistency_report.get("transition_issues", [])
        if transition_issues:
            for i in range(len(ordered_keys) - 1):
                prev_key = ordered_keys[i]
                next_key = ordered_keys[i + 1]
                if prev_key in sections and next_key in sections:
                    prev_tail = sections[prev_key][-200:]
                    next_head = sections[next_key][:200]
                    # Only add transition if flagged
                    flagged = any(
                        prev_key in issue or next_key in issue
                        for issue in transition_issues
                    )
                    if flagged:
                        try:
                            transition = await transition_chain.ainvoke({
                                "prev_tail": prev_tail,
                                "next_head": next_head,
                            })
                            sections[next_key] = transition.strip() + "\n\n" + sections[next_key]
                        except Exception as e:
                            logger.warning("[orchestrator] transition generation failed: %s", e)

        # Assemble final document with chapter grouping and auto-numbering
        parts = [f"# {topic}\n"]
        chapter_num = 0
        body_summaries: list[str] = []

        for chapter in outline:
            chapter_title = chapter["title"]
            title_lower = chapter_title.lower()
            sqs = chapter.get("sub_questions", [])

            # Skip intro/conclusion — will be generated separately
            if not sqs and (title_lower.startswith("intro") or title_lower.startswith("conclusion")):
                continue

            chapter_num += 1
            if sqs:
                parts.append(f"\n## {chapter_num} {chapter_title}\n")
                sub_num = 0
                for sq in sqs:
                    if sq in sections:
                        sub_num += 1
                        cleaned = _strip_draft_headings(sections[sq])
                        # Renumber ### headings within this section
                        cleaned = _renumber_h3(cleaned, chapter_num, sub_num)
                        parts.append(f"\n{cleaned}")
                        body_summaries.append(f"[{chapter_title}/{sq}] {cleaned[:300]}")
            else:
                if chapter_title in sections:
                    cleaned = _strip_draft_headings(sections[chapter_title])
                    cleaned = _renumber_h3(cleaned, chapter_num, 0)
                    parts.append(f"\n## {chapter_num} {chapter_title}\n\n{cleaned}")
                    body_summaries.append(f"[{chapter_title}] {cleaned[:300]}")

        body_text = "\n".join(parts)

        # Generate Introduction and Conclusion via LLM
        intro_text = await _generate_intro_conclusion(
            topic, body_summaries, "introduction", make_llm(config, temperature=0.3)
        )
        conclusion_text = await _generate_intro_conclusion(
            topic, body_summaries, "conclusion", make_llm(config, temperature=0.3)
        )

        # Assemble full document
        full_parts = [f"# {topic}\n"]
        if intro_text:
            full_parts.append(f"\n## 引言\n\n{intro_text}")
        full_parts.append(body_text.replace(f"# {topic}\n", "", 1))
        if conclusion_text:
            full_parts.append(f"\n## {chapter_num + 1} 结论与展望\n\n{conclusion_text}")

        full_text = "\n".join(full_parts)

        # Resolve citations: [paper title] → [N], with DB matching
        db_path = Path(state["run_dir"]) / "papers.db"
        full_text, ref_lines = _resolve_citations(full_text, str(db_path))
        if ref_lines:
            full_text += "\n\n## 参考文献\n\n" + "\n".join(ref_lines)

        # Write output
        output_path = run_dir / "survey.md"
        output_path.write_text(full_text, encoding="utf-8")
        logger.info("[orchestrator] survey written to %s", output_path)

        return Command(
            update={"final_output": full_text, "output_path": str(output_path)},
            goto="__end__",
        )

    return {
        "build_index": build_index,
        "write_sections": write_sections,
        "polish_sections": polish_sections,
        "check_consistency": check_consistency,
        "merge_final": merge_final,
    }
