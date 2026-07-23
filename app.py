"""
app.py

This is the ONLY file that builds the user interface. Everything else
in the project (ingestion, chunking, embeddings, retrieval, the
LangGraph pipeline, storage) is plain Python logic that this file
calls into - this file itself stays "thin", just wiring buttons and
text boxes to the real work happening elsewhere.
"""

import uuid
import tempfile

import streamlit as st

from config.settings import check_required_settings
from ingestion.coordinator import ingest_sources
from processing.chunking import chunk_document
from processing.embeddings import embed_chunks
from retrieval.vector_store import ensure_collection_exists, upsert_chunks, list_sources
from storage.db import (
    get_or_create_session,
    create_conversation,
    get_latest_conversation_id,
    get_messages,
    list_conversations,
    update_conversation_title,
)
from pipeline.graph import compiled_graph


st.set_page_config(page_title="Multi-source RAG Chatbot", layout="wide")

check_required_settings()
ensure_collection_exists()


# ---------------------------------------------------------------------
# SESSION SETUP
# ---------------------------------------------------------------------

if "session_id" not in st.query_params:
    st.query_params["session_id"] = str(uuid.uuid4())

session_id = st.query_params["session_id"]
get_or_create_session(session_id)

if "conversation_id" not in st.session_state:
    existing_conversation_id = get_latest_conversation_id(session_id)

    if existing_conversation_id is not None:
        st.session_state.conversation_id = existing_conversation_id
        st.session_state.chat_history = get_messages(existing_conversation_id)
    else:
        st.session_state.conversation_id = create_conversation(
            session_id=session_id,
            title="New conversation",
        )
        st.session_state.chat_history = []


# ---------------------------------------------------------------------
# SIDEBAR: chat history (like ChatGPT's conversation list)
# ---------------------------------------------------------------------

st.sidebar.header("Chats")

if st.sidebar.button("+ New Chat", use_container_width=True):
    new_conversation_id = create_conversation(session_id=session_id, title="New conversation")
    st.session_state.conversation_id = new_conversation_id
    st.session_state.chat_history = []
    st.rerun()

past_conversations = list_conversations(session_id)

for convo in past_conversations:
    is_current = convo["conversation_id"] == st.session_state.conversation_id
    label = convo["title"] or "New conversation"
    button_type = "primary" if is_current else "secondary"

    if st.sidebar.button(label, key=f"convo_{convo['conversation_id']}", use_container_width=True, type=button_type):
        st.session_state.conversation_id = convo["conversation_id"]
        st.session_state.chat_history = get_messages(convo["conversation_id"])
        st.rerun()

st.sidebar.divider()

if "ingested_source_keys" not in st.session_state:
    st.session_state.ingested_source_keys = set()


# ---------------------------------------------------------------------
# SIDEBAR: adding sources
# ---------------------------------------------------------------------

st.sidebar.header("Add your sources")

existing_sources = list_sources(session_id)
if existing_sources:
    with st.sidebar.expander(f"Already added ({len(existing_sources)})", expanded=False):
        for source in existing_sources:
            if source["source_type"] == "pdf":
                st.caption("📄 PDF")
            elif source["source_type"] == "youtube":
                st.caption(f"▶️ {source['source_url']}")
            elif source["source_type"] == "web":
                st.caption(f"🌐 {source['source_url']}")
            else:
                st.caption("📝 Pasted text")

pdf_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
youtube_url = st.sidebar.text_input("YouTube video URL")
web_url = st.sidebar.text_input("Web page URL")
pasted_text = st.sidebar.text_area("Paste plain text")

if st.session_state.ingested_source_keys:
    st.sidebar.caption(f"{len(st.session_state.ingested_source_keys)} source(s) already added this session.")

if st.sidebar.button("Ingest sources"):
    sources_to_process = []

    if pdf_file is not None:
        pdf_key = f"pdf:{pdf_file.name}:{pdf_file.size}"
        if pdf_key not in st.session_state.ingested_source_keys:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.read())
                sources_to_process.append({"type": "pdf", "reference": tmp.name, "key": pdf_key})

    if youtube_url.strip():
        yt_key = f"youtube:{youtube_url.strip()}"
        if yt_key not in st.session_state.ingested_source_keys:
            sources_to_process.append({"type": "youtube", "reference": youtube_url.strip(), "key": yt_key})

    if web_url.strip():
        web_key = f"web:{web_url.strip()}"
        if web_key not in st.session_state.ingested_source_keys:
            sources_to_process.append({"type": "web", "reference": web_url.strip(), "key": web_key})

    if pasted_text.strip():
        text_key = f"text:{pasted_text.strip()}"
        if text_key not in st.session_state.ingested_source_keys:
            sources_to_process.append({"type": "text", "reference": pasted_text.strip(), "key": text_key})

    if len(sources_to_process) == 0:
        st.sidebar.warning("Add at least one NEW source before ingesting - everything currently filled in has already been added.")
    else:
        with st.spinner("Processing your sources..."):
            result = ingest_sources([
                {"type": s["type"], "reference": s["reference"]} for s in sources_to_process
            ])

            for document in result.successful_documents:
                chunks = chunk_document(document, session_id=session_id)
                embedded = embed_chunks(chunks)
                upsert_chunks(embedded)

            for s in sources_to_process:
                st.session_state.ingested_source_keys.add(s["key"])

        if result.successful_documents:
            st.sidebar.success(f"{len(result.successful_documents)} new source(s) added successfully.")

        for failure in result.failures:
            st.sidebar.error(f"{failure.source_type}: {failure.reason}")


def _format_citation(citation: dict) -> str:
    source_type = citation["source_type"]

    if source_type == "pdf" and citation.get("page_number"):
        return f"PDF - page {citation['page_number']}"

    if source_type == "youtube" and citation.get("start_time") is not None:
        start = int(citation["start_time"])
        end = int(citation["end_time"])
        return f"YouTube - {start // 60}:{start % 60:02d} to {end // 60}:{end % 60:02d}"

    if source_type == "web" and citation.get("source_url"):
        return f"Web page - {citation['source_url']}"

    return source_type.capitalize()


# ---------------------------------------------------------------------
# MAIN AREA: the chat itself
# ---------------------------------------------------------------------

st.title("Ask questions across your sources")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message.get("citations"):
            with st.expander("Sources"):
                for citation in message["citations"]:
                    st.caption(_format_citation(citation))


user_question = st.chat_input("Ask a question about your sources...")

if user_question:
    is_first_message = len(st.session_state.chat_history) == 0
    if is_first_message:
        auto_title = user_question[:50] + ("..." if len(user_question) > 50 else "")
        update_conversation_title(st.session_state.conversation_id, auto_title)

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            recent_history = st.session_state.chat_history[-6:]

            result = compiled_graph.invoke({
                "session_id": session_id,
                "conversation_id": st.session_state.conversation_id,
                "raw_query": user_question,
                "chat_history": recent_history,
            })

        st.write(result["answer"])

        if result["citations"]:
            with st.expander("Sources"):
                for citation in result["citations"]:
                    st.caption(_format_citation(citation))

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "citations": result["citations"],
    })