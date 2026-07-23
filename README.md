# Multi-source RAG chatbot

Paste a PDF, YouTube link, web link, or plain text (any combination, at once)
and ask questions across all of them together.

## Pipeline

Ingestion -> unified schema -> chunking -> dense + sparse embeddings ->
Qdrant hybrid index -> query rewrite + conditional decomposition ->
hybrid retrieval -> cross-encoder rerank -> LLM generation -> persisted
to Supabase with full trace.

## Known limitation

YouTube transcript extraction can get blocked on cloud-hosted deployments
(YouTube blocks known cloud provider IP ranges). Ingestion fails gracefully
per source with a clear reason shown in the UI instead of crashing the
session.

## Stack

Streamlit, Qdrant Cloud, Supabase Postgres, Groq (Llama 3.1 / 3.3),
LangGraph, RAGAS, GitHub Actions, Grafana Cloud.
