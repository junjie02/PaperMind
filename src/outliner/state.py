from langgraph.graph import MessagesState


class LiRAState(MessagesState):
    """LangGraph state for the outline-drafting pipeline.

    Inherits ``messages`` from MessagesState — used now only as a transcript
    sink, but reserved for the future reviewer loop where reviewer feedback
    accumulates across rounds.
    """

    in_dir: str
    topic: str
    n_sections: int
    n_subsections: int

    papers: list[dict]
    cluster_assignments: list[dict]
    cluster_skipped_reason: str
    paper_groups: list[dict]
    group_outlines: list[str]
    chapter_plan: list[dict]
    survey_title: str
    final_outline: str
    references_md: str
    output_path: str

    revision_count: int
    review_feedback: str
