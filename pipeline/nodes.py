"""
nodes.py

This file holds every "step" (node) in our LangGraph pipeline. Each
function here takes the current PipelineState, does ONE job, and
returns the parts of the state it changed.

We're building this one node at a time - starting with
rewrite_and_classify, the combined call that does THREE jobs in one
LLM request: cleans up the query, decides if it's complex, and (if
so) breaks it into sub-questions.
"""

import json
from groq import Groq

from config.settings import GROQ_API_KEY, GROQ_REWRITE_MODEL, GROQ_GENERATION_MODEL
from pipeline.state import PipelineState
from ingestion.schema import Chunk
from processing.embeddings import embed_query
from retrieval.vector_store import hybrid_search
from retrieval.reranker import rerank_chunks
from storage.db import save_message
from storage.models import MessageRecord


_groq_client = Groq(api_key=GROQ_API_KEY)


REWRITE_PROMPT = """You are a query understanding assistant for a document search system.

Given the user's question, do three things:
1. Rewrite it to be clear and unambiguous on its own (fix typos, resolve vague pronouns if the meaning is obvious, keep it short).
2. Decide if it's a COMPLEX question - meaning it genuinely asks about two or more separate things (e.g. uses "and", "also", "compare", "versus", or asks multiple things at once).
3. If it IS complex, break it into 2-4 clear, standalone sub-questions. If it is NOT complex, return an empty list for sub-questions.

Respond with ONLY valid JSON, in exactly this shape, and nothing else:
{"rewritten_query": "...", "is_complex": true or false, "sub_questions": ["...", "..."]}
"""


def rewrite_and_classify(state: PipelineState) -> dict:
    """
    Node 1 of the pipeline.

    Reads:
        state["raw_query"] - exactly what the user typed

    Returns (added back into the state):
        rewritten_query, is_complex, sub_questions
    """

    user_query = state["raw_query"]

    response = _groq_client.chat.completions.create(
        model=GROQ_REWRITE_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": user_query},
        ],
        # This tells Groq to guarantee the response is valid JSON,
        # so we don't have to worry about the model adding extra
        # words before or after the JSON object.
        response_format={"type": "json_object"},
        temperature=0.2,  # low temperature = more consistent, less "creative" output
    )

    raw_output = response.choices[0].message.content
    parsed = json.loads(raw_output)

    # A small safety net: if the model ever forgets a field (rare,
    # but possible), we fall back to sensible defaults instead of
    # crashing the whole pipeline.
    return {
        "rewritten_query": parsed.get("rewritten_query", user_query),
        "is_complex": parsed.get("is_complex", False),
        "sub_questions": parsed.get("sub_questions", []),
    }


def _payload_to_chunk(payload: dict) -> Chunk:
    """
    Converts a raw Qdrant payload dictionary back into a proper Chunk
    object. This works cleanly because we designed Chunk.to_payload()
    (back in schema.py) to use the EXACT SAME field names as Chunk
    itself - so we can rebuild one from the other with no translation
    needed.
    """
    return Chunk(**payload)


def retrieve_simple(state: PipelineState) -> dict:
    """
    Retrieval path for SIMPLE queries (is_complex is False).

    Reads:
        state["rewritten_query"], state["session_id"]

    Returns:
        retrieved_chunks - the wide net from one hybrid search
    """

    query_embedding = embed_query(state["rewritten_query"])

    results = hybrid_search(
        dense_query_vector=query_embedding["dense_vector"],
        sparse_query_indices=query_embedding["sparse_indices"],
        sparse_query_values=query_embedding["sparse_values"],
        session_id=state["session_id"],
        limit=10,
    )

    chunks = [_payload_to_chunk(r["payload"]) for r in results]

    return {"retrieved_chunks": chunks}


def retrieve_decomposed(state: PipelineState) -> dict:
    """
    Retrieval path for COMPLEX queries (is_complex is True).

    Runs a SEPARATE hybrid search for each sub-question, then merges
    all the results together - while removing duplicates, since two
    different sub-questions can easily both match the same chunk.

    Reads:
        state["sub_questions"], state["session_id"]

    Returns:
        retrieved_chunks - the merged, deduplicated results from all
        sub-questions combined
    """

    # We use a dictionary keyed by (source_id, chunk_index) to
    # naturally deduplicate: if the same chunk gets matched by two
    # different sub-questions, the second one just overwrites the
    # first instead of creating a duplicate entry.
    merged_chunks_by_key = {}

    for sub_question in state["sub_questions"]:
        query_embedding = embed_query(sub_question)

        results = hybrid_search(
            dense_query_vector=query_embedding["dense_vector"],
            sparse_query_indices=query_embedding["sparse_indices"],
            sparse_query_values=query_embedding["sparse_values"],
            session_id=state["session_id"],
            limit=5,  # fewer per sub-question, since we're running several searches
        )

        for r in results:
            chunk = _payload_to_chunk(r["payload"])
            key = (chunk.source_id, chunk.chunk_index)
            merged_chunks_by_key[key] = chunk

    return {"retrieved_chunks": list(merged_chunks_by_key.values())}


def rerank(state: PipelineState) -> dict:
    """
    Node: re-scores retrieved_chunks against the query, keeping only
    the best few before they go to the LLM.

    Reads:
        state["rewritten_query"], state["retrieved_chunks"]

    Returns:
        reranked_chunks - the narrowed, best-first shortlist
    """

    best_chunks = rerank_chunks(
        query=state["rewritten_query"],
        chunks=state["retrieved_chunks"],
        top_k=5,
    )

    return {"reranked_chunks": best_chunks}


GENERATION_PROMPT = """You are a helpful assistant answering questions using ONLY the provided context below.

Rules:
- Write one clear, natural, flowing answer - like explaining it to a person, not a report.
- Do NOT add citation markers, footnotes, source names, or bracketed numbers inside your answer. Sources will be shown separately, after your answer - so just focus on writing a clean, complete answer.
- If the context does not contain enough information to answer, say so honestly instead of guessing.

Context:
{context}
"""


def _build_context_string(chunks: list) -> str:
    """
    Joins the reranked chunks into one big text block to hand to the
    LLM as its "context" - the material it's allowed to answer from.
    """
    return "\n\n---\n\n".join(chunk.chunk_text for chunk in chunks)


def _build_citations(chunks: list) -> list[dict]:
    """
    Builds the "Sources" list shown AFTER the answer, straight from
    the chunk metadata - not from anything the LLM writes. This is
    the design choice we discussed: citations come from our own
    retrieval data (always accurate), never from the LLM guessing.

    We deduplicate by source_id, since multiple reranked chunks often
    come from the same PDF/video - we only want to list each source
    once, not once per chunk.
    """
    seen_source_ids = set()
    citations = []

    for chunk in chunks:
        if chunk.source_id in seen_source_ids:
            continue
        seen_source_ids.add(chunk.source_id)

        citations.append({
            "source_type": chunk.source_type,
            "page_number": chunk.page_number,
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
            "source_url": chunk.source_url,
        })

    return citations


def generate(state: PipelineState) -> dict:
    """
    Node: generates the final answer using the reranked chunks as
    context, and separately builds the citation list from the same
    chunks.

    Reads:
        state["rewritten_query"], state["reranked_chunks"]

    Returns:
        answer, citations
    """

    context = _build_context_string(state["reranked_chunks"])

    response = _groq_client.chat.completions.create(
        model=GROQ_GENERATION_MODEL,
        messages=[
            {"role": "system", "content": GENERATION_PROMPT.format(context=context)},
            {"role": "user", "content": state["rewritten_query"]},
        ],
        temperature=0.3,
    )

    answer_text = response.choices[0].message.content
    citations = _build_citations(state["reranked_chunks"])

    return {
        "answer": answer_text,
        "citations": citations,
    }


def persist(state: PipelineState) -> dict:
    """
    Node: saves this full turn (the user's question AND the
    assistant's answer) into Supabase, with the full pipeline trace
    attached to the assistant's row.

    This is deliberately the LAST node in the graph - if anything
    earlier fails, this node never runs, so we never save a
    half-finished or broken record.

    Reads:
        almost everything in the state - this node's whole job is
        to write it all down

    Returns:
        an empty dict - persist doesn't add anything new to the
        state, it just saves what's already there
    """

    # Save the user's turn first - a simple record, no trace data
    # needed since the user's message IS the raw input, not a result.
    save_message(MessageRecord(
        conversation_id=state["conversation_id"],
        role="user",
        content=state["raw_query"],
    ))

    # Save the assistant's turn, with the full trace attached - this
    # is what makes every answer debuggable and auditable later.
    save_message(MessageRecord(
        conversation_id=state["conversation_id"],
        role="assistant",
        content=state["answer"],
        rewritten_query=state["rewritten_query"],
        sub_questions=state["sub_questions"],
        retrieved_chunks=[chunk.to_payload() for chunk in state["reranked_chunks"]],
        citations=state["citations"],
    ))

    return {}
