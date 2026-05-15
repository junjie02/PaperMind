"""LangGraph workflow assembly for the outline-drafting pipeline.

All inter-node transitions go through ``Command(goto=...)`` returned by each
node. This file only registers the nodes — no static ``add_edge`` between
business steps. That makes it cheap to insert future nodes (reviewer, RAG
content writer, polish) at the marked seams without touching the rest.
"""

from argparse import Namespace

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from deepresearch.config import Config

from .nodes import build_nodes
from .state import LiRAState


def build_graph(config: Config, args: Namespace):
    nodes = build_nodes(config, args)

    workflow = StateGraph(LiRAState)
    workflow.add_node("load_papers", nodes["load_papers"])
    workflow.add_node("cluster_papers", nodes["cluster_papers"])
    workflow.add_node("group_by_direction", nodes["group_by_direction"])
    workflow.add_node("draft_per_group", nodes["draft_per_group"])
    # SEAM #1: future ``review_outline`` slots in here.
    # draft_per_group will then goto="review_outline" instead of "merge_outlines";
    # review_outline returns Command(goto="merge_outlines" if approved else
    # "draft_per_group", update={"revision_count": +1, "review_feedback": ...}).
    workflow.add_node("merge_outlines", nodes["merge_outlines"])
    # SEAM #2: future ``write_content`` and ``polish`` slot in here, before
    # render_references. They will pull RAG context per section.
    workflow.add_node("render_references", nodes["render_references"])

    workflow.set_entry_point("load_papers")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
