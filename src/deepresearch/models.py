from __future__ import annotations

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


class WorkerTask(BaseModel):
    research_question: str
    exclude_ids: list[str] = Field(default_factory=list)
    exclude_titles: list[str] = Field(default_factory=list)
    worker_index: int = 0
    target_papers: int = 3
