"""PaperMind LangGraph orchestration graph.

Flow:
  explore_directions → synthesize_outline → research_sections → check_coverage
    → build_index → write_sections (includes review loop)
    → polish_sections → check_consistency → final_review → merge_final → END
"""

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from orchestrator.nodes import build_nodes
from orchestrator.nodes_writing import build_writing_nodes
from orchestrator.state import PaperMindState
from shared.config import Config

logger = logging.getLogger(__name__)


def build_graph(config: Config):
    research_nodes = build_nodes(config)
    writing_nodes = build_writing_nodes(config)

    workflow = StateGraph(PaperMindState)

    # Register all nodes
    for name, fn in {**research_nodes, **writing_nodes}.items():
        workflow.add_node(name, fn)

    workflow.set_entry_point("explore_directions")

    # All transitions are handled via Command(goto=...) inside each node.
    # No static edges needed.

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def build_graph_from(config: Config, entry_point: str):
    """Build graph with a custom entry point for resuming from a specific phase."""
    research_nodes = build_nodes(config)
    writing_nodes = build_writing_nodes(config)

    all_nodes = {**research_nodes, **writing_nodes}
    if entry_point not in all_nodes:
        valid = ", ".join(sorted(all_nodes.keys()))
        raise ValueError(f"Unknown entry point '{entry_point}'. Valid: {valid}")

    workflow = StateGraph(PaperMindState)
    for name, fn in all_nodes.items():
        workflow.add_node(name, fn)

    workflow.set_entry_point(entry_point)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
