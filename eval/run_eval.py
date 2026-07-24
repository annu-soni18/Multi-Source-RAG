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
from ragas.run_config import RunConfig
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


EVAL_SESSION_ID = "eval-fixed-corpus-v1"
FAITHFULNESS_THRESHOLD = 0.7


def ingest_test_corpus(source_text: str):
    ensure_collection_exists()
    document = load_text(source_text)
    chunks = chunk_document(document, session_id=EVAL_SESSION_ID)
    embedded = embed_chunks(chunks)
    upsert_chunks(embedded)


def run_pipeline_without_persisting(question: str) -> dict:
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
    testset_path = Path(__file__).parent / "testset.json"
    with open(testset_path) as f:
        testset_data = json.load(f)

    print("Ingesting fixed test corpus...")
    ingest_test_corpus(testset_data["source_text"])

    print("Building RAGAS dataset by running the real pipeline...")
    dataset = build_ragas_dataset(testset_data["questions"])

    chat_model = ChatOpenAI(
        model=GROQ_GENERATION_MODEL,
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    ragas_llm = LangchainLLMWrapper(chat_model)

    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    metrics = [
        Faithfulness(),
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
        show_progress=False,
        run_config=RunConfig(max_workers=2, timeout=300),
    )

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

    output_path = Path(__file__).parent.parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nResults saved to: {output_path}")

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

    faithfulness_score = scores.get("faithfulness")
    if faithfulness_score is None or faithfulness_score < FAITHFULNESS_THRESHOLD:
        print(f"\nFAILED: faithfulness ({faithfulness_score}) is below threshold ({FAITHFULNESS_THRESHOLD})")
        sys.exit(1)

    print("\nPASSED: all scores within acceptable range.")
    sys.exit(0)