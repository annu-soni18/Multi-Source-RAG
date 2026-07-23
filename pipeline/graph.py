"""
graph.py

This file's ONE job: wire all 6 nodes from nodes.py together into an
actual LangGraph StateGraph - the diagram we drew earlier, now made
real.

Flow:
    rewrite_and_classify
            |
      is_complex? ---false---> retrieve_simple ----+
            |                                        |--> rerank --> generate --> persist
            +--------true-----> retrieve_decomposed -+
"""

from langgraph.graph import StateGraph, START, END

from pipeline.state import PipelineState
from pipeline.nodes import (
    rewrite_and_classify,
    retrieve_simple,
    retrieve_decomposed,
    rerank,
    generate,
    persist,
)


def _route_after_classify(state: PipelineState) -> str:
    """
    This is the "router" function for our conditional edge. LangGraph
    calls this right after rewrite_and_classify finishes, and
    whatever STRING it returns tells LangGraph which node to go to
    next.
    """
    if state["is_complex"]:
        return "retrieve_decomposed"
    return "retrieve_simple"


def build_graph():
    """
    Builds and compiles the full pipeline graph.

    Returns:
        A compiled graph, ready to be run with .invoke(initial_state).
    """

    builder = StateGraph(PipelineState)

    # ---- Step 1: register every node ----
    # Each node here is just the plain function from nodes.py - we're
    # not calling them yet, just telling the graph they exist and
    # what to call them.
    builder.add_node("rewrite_and_classify", rewrite_and_classify)
    builder.add_node("retrieve_simple", retrieve_simple)
    builder.add_node("retrieve_decomposed", retrieve_decomposed)
    builder.add_node("rerank", rerank)
    builder.add_node("generate", generate)
    builder.add_node("persist", persist)

    # ---- Step 2: wire the edges (the arrows in our diagram) ----

    # Every run starts at rewrite_and_classify.
    builder.add_edge(START, "rewrite_and_classify")

    # This is the ONE conditional branch in the whole graph - it
    # reads is_complex from the state and picks a path accordingly.
    builder.add_conditional_edges(
        "rewrite_and_classify",
        _route_after_classify,
        {
            "retrieve_simple": "retrieve_simple",
            "retrieve_decomposed": "retrieve_decomposed",
        },
    )

    # Both retrieval paths converge back into the SAME rerank node -
    # this is the "convergence point" we designed earlier, so
    # reranking logic only needs to be written once.
    builder.add_edge("retrieve_simple", "rerank")
    builder.add_edge("retrieve_decomposed", "rerank")

    # From here on, it's a simple straight line: rerank -> generate -> persist -> done.
    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", "persist")
    builder.add_edge("persist", END)

    return builder.compile()


# Build the graph ONCE when this module is first imported, so every
# other file just imports this ready-to-use compiled graph instead of
# rebuilding it on every single query.
compiled_graph = build_graph()
