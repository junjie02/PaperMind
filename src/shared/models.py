from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class PaperRecord(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    overview: str = ""
    source: str = "web"
    source_url: str = ""
    venue: str | None = None
    arxiv_id: str | None = None
    search_direction: str = ""
    published_at: str | None = None
    categories: list[str] = Field(default_factory=list)
    primary_class: str | None = None
    bibtex: str = ""
    abs_url: str = ""
    pdf_url: str = ""
    artifact_rel_path: str | None = None
    search_round: int = 0
    worker_id: str = ""
    relevance_score: int = 3


class AgentTask(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instruction: str
    max_iterations: int = 12
    stop_conditions: list[str] = Field(
        default_factory=lambda: ["no_new_info", "target_hit", "max_results_reached"]
    )
    previous_results: dict[str, Any] | None = None
    run_dir: str = ""
    db_path: str = ""


class AgentResult(BaseModel):
    task_id: str
    status: Literal["success", "partial", "failed"]
    summary: str
    metadata: list[dict[str, Any]] = Field(default_factory=list)
    suggested_followup: str | None = None
    papers: list[PaperRecord] = Field(default_factory=list)
    draft_text: str = ""
    issues: list[str] = Field(default_factory=list)
    polished_text: str = ""
