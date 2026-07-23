"""
state.py

This file defines the SHAPE of the data that flows through our
LangGraph pipeline - from the moment a user asks a question, all the
way to the final saved answer.

LangGraph expects this "state" to be a TypedDict - think of it as a
dictionary where we PROMISE ahead of time exactly which keys will
exist and what type of value each one holds. Every node in our graph
(rewrite, retrieve, rerank, generate, persist) will receive this same
state, read what it needs from it, and add its own results back into
it before passing it to the next node.
"""

from typing import TypedDict, Optional


class PipelineState(TypedDict):
    # ---- Filled in at the very start, before the graph even runs ----
    session_id: str
    conversation_id: str        # which conversation thread this message belongs to
    raw_query: str              # exactly what the user typed, untouched

    # ---- Filled in by the "rewrite_and_classify" node ----
    rewritten_query: str        # the cleaned-up, unambiguous version of the query
    is_complex: bool             # decides which retrieval path we take
    sub_questions: list[str]    # only non-empty when is_complex is True

    # ---- Filled in by the retrieval nodes ----
    retrieved_chunks: list       # list[Chunk] - the wide net from hybrid search

    # ---- Filled in by the "rerank" node ----
    reranked_chunks: list        # list[Chunk] - the narrowed, best-first shortlist

    # ---- Filled in by the "generate" node ----
    answer: str
    citations: list              # list of dicts, e.g. {"source_type": "pdf", "page_number": 4}
