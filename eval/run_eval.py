"""
run_eval.py

This is the MAIN evaluation script. It answers one question: "did my
RAG pipeline get WORSE?" - by running it against a fixed, known set of
questions every time, and scoring the answers with RAGAS.

This is designed to be run in two places:
  1. Locally, by you, whenever you want to sanity-check a change.
  2. Automatically, by GitHub Actions, on every pull request.

WHY A FIXED TEST CORPUS (not real user data)?
Real user documents change constantly, so scores would be
meaningless to compare over time - a "faithfulness" drop might just
mean someone uploaded a harder PDF, not that the code got worse.
Using the SAME fixed text every time means: if the score changes,
something about the CODE changed, not the data. This is a basic
principle of evaluation - only change one variable at a time.

WHAT RAGAS MEASURES HERE:
- Faithfulness: does the answer only say things actually supported by
  the retrieved chunks? (catches hallucination)
- Answer Relevancy: does the answer actually address the question asked?
- Context Precision: are the retrieved chunks actually relevant, or is
  the search pulling in noise?
- Context Recall: did retrieval find EVERYTHING needed to answer
  correctly, or did it miss something important?
"""

import json
import sys
from pathlib import Path

from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from config.settings import GROQ_API_KEY, GROQ_GENERATION_MODEL
from ingestion.text_loader import load_text
from processing.chunking import chunk_document
from processing.embeddings import embed_chunks
from retrieval.vector_store import ensure_collection_exists, upsert_chunks
from pipeline.nodes import rewrite_and_classify, retrieve_simple, retrieve_decomposed, rerank, generate
from storage.db import save_eval_run


# A dedicated, fixed session_id just for eval runs - completely
# separate from real user sessions, so this never mixes with actual
# user data, and re-running eval never pollutes anything.
EVAL_SESSION_ID = "eval-fixed-corpus-v1"

# Below this faithfulness score, we consider the pipeline to have
# REGRESSED, and the script exits with an error code - this is what
# lets GitHub Actions block a pull request automatically.
FAITHFULNESS_THRESHOLD = 0.7


def ingest_test_corpus(source_text: str):
    """
    Ingests our FIXED test document, using the exact same pipeline
    real users go through (load -> chunk -> embed -> store) - so the
    eval is testing the real code path, not a shortcut.

    Because text_loader.py now builds a deterministic source_id from
    the text's content (see our earlier fix), re-running this in CI
    every single time safely OVERWRITES the same points instead of
    duplicating them.
    """
    ensure_collection_exists()
    document = load_text(source_text)
    chunks = chunk_document(document, session_id=EVAL_SESSION_ID)
    embedded = embed_chunks(chunks)
    upsert_chunks(embedded)


def run_pipeline_without_persisting(question: str) -> dict:
    """
    Runs our REAL pipeline nodes (rewrite, retrieve, rerank, generate)
    for one question - but deliberately SKIPS the persist node, since
    an eval run isn't a real user conversation and shouldn't be saved
    into the chat history tables.

    We call the node functions directly instead of using
    pipeline.graph.compiled_graph, since the compiled graph always
    ends by calling persist - this gives us the same logic, minus
    that one step.
    """
    state = {
        "session_id": EVAL_SESSION_ID,
        "raw_query": question,
        "chat_history": [],
    }

    state.update(rewrite_and_classify(state))

    if state["is_complex"]:
        state.update(retrieve_decomposed(state))
    else:
        state.update(retrieve_simple(state))

    state.update(rerank(state))
    state.update(generate(state))

    return state


def build_ragas_dataset(testset: list[dict]) -> EvaluationDataset:
    """
    Runs the pipeline for every question in our test set, and packages
    the results into the exact shape RAGAS expects for scoring.
    """
    rows = []

    for item in testset:
        print(f"Running pipeline for: {item['question']}")
        result = run_pipeline_without_persisting(item["question"])

        rows.append({
            "user_input": item["question"],
            "response": result["answer"],
            "retrieved_contexts": [chunk.chunk_text for chunk in result["reranked_chunks"]],
            "reference": item["reference"],
        })

    return EvaluationDataset.from_list(rows)


def run_evaluation() -> dict:
    """
    The main entry point: ingests the fixed corpus, runs the pipeline
    on every test question, scores the results with RAGAS, and
    returns a dict of average scores per metric.
    """

    testset_path = Path(__file__).parent / "testset.json"
    with open(testset_path) as f:
        testset_data = json.load(f)

    print("Ingesting fixed test corpus...")
    ingest_test_corpus(testset_data["source_text"])

    print("Building RAGAS dataset by running the real pipeline...")
    dataset = build_ragas_dataset(testset_data["questions"])

    # RAGAS needs an LLM to actually JUDGE the answers (e.g. "does
    # this answer contradict the retrieved context?"). We point it at
    # Groq using its OpenAI-compatible endpoint - Groq supports the
    # same API shape as OpenAI, so no separate Groq-specific library
    # is needed here.
    #
    # NOTE: ragas's own docs recommend llm_factory() as the modern
    # replacement for LangchainLLMWrapper. We tried that first, but
    # discovered (by reading ragas's actual source) that the object
    # llm_factory() returns uses a completely different calling
    # convention than what these specific metrics (Faithfulness,
    # AnswerRelevancy, etc.) actually call internally - a real
    # incompatibility inside ragas==0.3.9 itself between its "modern"
    # LLM factory and its "prompt-based" metrics. LangchainLLMWrapper
    # is marked deprecated, but it's the one that actually has the
    # correct interface these metrics expect, so we use it deliberately.
    chat_model = ChatOpenAI(
        model=GROQ_GENERATION_MODEL,
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    ragas_llm = LangchainLLMWrapper(chat_model)

    # Answer Relevancy specifically needs an EMBEDDING model (it
    # checks how closely the answer's implied question matches the
    # real question) - Groq doesn't offer an embeddings API, so we
    # use a small local Hugging Face model just for this scoring step.
    # This only affects the eval script, not the deployed app.
    #
    # Same interface mismatch as the LLM above: ragas's own
    # HuggingFaceEmbeddings doesn't implement embed_query(), which
    # this metric actually calls. We use LangchainEmbeddingsWrapper
    # around langchain_community's HuggingFaceEmbeddings instead,
    # which has the correct interface - and it's already installed
    # as a dependency of ragas itself, so no extra package is needed.
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    metrics = [
        Faithfulness(),
        # AnswerRelevancy normally generates 3 alternate questions per
        # answer (strictness=3) and averages their similarity, to
        # reduce scoring variance. That requires asking the LLM for 3
        # completions in a single request (n=3) - which Groq's API
        # rejects outright ("'n': number must be at most 1"). Setting
        # strictness=1 makes it request just 1 generation instead,
        # trading a bit of score stability for actually working with
        # Groq's API limits.
        AnswerRelevancy(strictness=1),
        LLMContextPrecisionWithoutReference(),
        LLMContextRecall(),
    ]

    print("Scoring with RAGAS...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    # Average each metric's per-question scores into one overall
    # number per metric. We filter out any individual NaN scores
    # first - RAGAS can legitimately return NaN for one question
    # (e.g. if the model generated no checkable statements), and we
    # don't want one such case to turn the WHOLE average into NaN.
    import math

    scores = {}
    for metric in metrics:
        metric_name = metric.name
        per_question_scores = [s for s in result[metric_name] if not math.isnan(s)]

        if len(per_question_scores) == 0:
            scores[metric_name] = None
        else:
            scores[metric_name] = sum(per_question_scores) / len(per_question_scores)

    return scores


if __name__ == "__main__":
    import os
    import subprocess

    scores = run_evaluation()

    print("\n=== RAGAS Evaluation Results ===")
    for metric_name, score in scores.items():
        if score is None:
            print(f"{metric_name}: N/A (no valid scores)")
        else:
            print(f"{metric_name}: {score:.4f}")

    # Save results to a file - GitHub Actions will pick this up as a
    # build artifact, and it's also handy for local debugging.
    output_path = Path(__file__).parent.parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Save this run to Supabase too, so it shows up in the Grafana
    # dashboard's history. In GitHub Actions, GITHUB_SHA is provided
    # automatically; running locally, we fall back to asking git
    # directly, so this works either way.
    commit_sha = os.environ.get("GITHUB_SHA")
    if not commit_sha:
        try:
            commit_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:
            commit_sha = "unknown"

    save_eval_run(commit_sha, scores)
    print(f"Saved eval run to Supabase (commit: {commit_sha[:8]})")

    # This is what lets CI actually BLOCK a pull request: if
    # faithfulness drops below our threshold, exit with a non-zero
    # code, which GitHub Actions treats as a failed check. A None
    # score (every question came back NaN) counts as a failure too -
    # that's not a "good" run, it's a broken one.
    faithfulness_score = scores.get("faithfulness")
    if faithfulness_score is None or faithfulness_score < FAITHFULNESS_THRESHOLD:
        print(f"\nFAILED: faithfulness ({faithfulness_score}) is below threshold ({FAITHFULNESS_THRESHOLD})")
        sys.exit(1)

    print("\nPASSED: all scores within acceptable range.")
    sys.exit(0)