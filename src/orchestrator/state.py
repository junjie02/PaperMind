"""PaperMind LangGraph state definition."""

from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict

from shared.models import AgentResult


class PaperMindState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────
    research_topic: str
    run_dir: str
    db_path: str
    target_papers: int               # user-specified total paper count (-n)
    skip_review: bool                # skip reviewer phase (--skip-review)

    # ── Phase 1: Direction exploration ─────────────────────────────────────
    sub_directions: list[str]           # topic split into sub-directions
    explorer_results: list[dict]        # one ExplorerAgent result per direction

    # ── Main agent conversation memory ─────────────────────────────────────
    agent_messages: list[dict]          # [{"role": "human"|"ai", "content": "..."}]

    # ── Phase 2: Research outline ──────────────────────────────────────────
    research_outline: list[dict]        # [{title, description, sub_questions:[]}]
    agent_concurrency: int              # LLM-recommended concurrency (capped at 3)
    papers_per_question: dict           # sub_question → target paper count

    # ── Phase 3: Deep research ─────────────────────────────────────────────
    researcher_results: dict[str, Any]  # sub_question → AgentResult dict
    coverage_ok: bool
    coverage_check_count: int

    # ── Phase 4: Writing ───────────────────────────────────────────────────
    section_drafts: dict[str, str]      # section_title → draft_text
    faiss_built: bool

    # ── Phase 5: Review ────────────────────────────────────────────────────
    review_issues: dict[str, list[str]] # section_title → issues list
    revision_count: int
    max_revisions: int

    # ── Phase 6: Polish ────────────────────────────────────────────────────
    polished_sections: dict[str, str]
    consistency_report: dict
    final_output: str
    output_path: str
