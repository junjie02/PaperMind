from __future__ import annotations

from pydantic import BaseModel, Field


class PaperRecord(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    overview: str = ""
    published_at: str | None = None
    categories: list[str] = Field(default_factory=list)
    primary_class: str | None = None
    bibtex: str = ""
    abs_url: str = ""
    pdf_url: str = ""
    search_round: int = 0
    worker_id: str = ""
    relevance_score: int = 3


class SearchDirection(BaseModel):
    direction: str
    search_queries: list[str]


class WorkerTask(BaseModel):
    research_question: str
    search_direction: str
    exclude_ids: list[str] = Field(default_factory=list)
    round_num: int = 0
    worker_index: int = 0
    target_papers: int = 3
