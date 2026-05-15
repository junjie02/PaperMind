"""LangGraph workflow assembly for the outline pipeline.

All inter-node transitions go through ``Command(goto=...)`` returned by each
node. This file only registers the nodes — no static ``add_edge`` between
business steps.

Current flow:
  load_papers → cluster_papers → group_by_direction → draft_per_group
  → merge_outlines → review_outline ⟲ (loop if rejected, max N revisions)
  → render_references → END
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
    workflow.add_node("merge_outlines", nodes["merge_outlines"])
    workflow.add_node("review_outline", nodes["review_outline"])
    # SEAM: future ``write_content`` and ``polish`` slot in here, before
    # render_references. They will pull RAG context per section.
    workflow.add_node("render_references", nodes["render_references"])

    workflow.set_entry_point("load_papers")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
