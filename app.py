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
from streamlit_cookies_controller import CookieController

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

# Fail loudly, right at startup, if any required API key is missing -
# instead of a confusing crash deep inside some function later.
check_required_settings()

# Qdrant's collection only needs to be created once ever, but calling
# this every time the app starts is safe and cheap - it just checks
# and does nothing if it already exists.
ensure_collection_exists()


# ---------------------------------------------------------------------
# SESSION SETUP
# ---------------------------------------------------------------------
# We used to store session_id in the URL - but that meant the user
# had to keep using the EXACT same link, or the app would treat them
# as a brand new visitor. A real browser COOKIE fixes this properly:
# it's set once, silently, and the browser sends it back automatically
# on every future visit - even from a completely fresh tab, with no
# special URL needed. This is exactly how real websites recognize
# returning visitors without requiring a login.

cookie_controller = CookieController()
session_id = cookie_controller.get("session_id")

if session_id is None:
    # First-ever visit from this browser - mint a new identity and
    # store it in a cookie that lasts a full year, so it survives
    # closing the browser, restarting the app, everything.
    session_id = str(uuid.uuid4())
    cookie_controller.set(
        "session_id",
        session_id,
        max_age=60 * 60 * 24 * 365,  # 1 year, in seconds
    )
    # Setting a cookie takes one round-trip to the browser before it's
    # actually readable - we rerun once immediately so the rest of
    # this script (and every future run) sees it correctly from here on.
    st.rerun()

get_or_create_session(session_id)

# The conversation_id and chat_history only need to be set up ONCE
# per browser tab session - if they're already in session_state, a
# previous run (or a click on a past chat / "New Chat") already
# handled this.
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
# This is what removes the need for the user to remember a session_id
# or URL just to get back to an old conversation - every past chat
# for this session is listed here, clickable, with a "+ New Chat"
# button to start a fresh one at any time.

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

    # Highlight whichever conversation is currently open, so the user
    # always knows which chat they're looking at - same idea as
    # ChatGPT bolding the active chat in its sidebar list.
    button_type = "primary" if is_current else "secondary"

    if st.sidebar.button(label, key=f"convo_{convo['conversation_id']}", use_container_width=True, type=button_type):
        st.session_state.conversation_id = convo["conversation_id"]
        st.session_state.chat_history = get_messages(convo["conversation_id"])
        st.rerun()

st.sidebar.divider()

# Tracks which sources have ALREADY been successfully ingested this
# session, so clicking "Ingest sources" again never re-processes the
# same source twice. Without this, every click would re-ingest
# whatever is still sitting in the sidebar fields, creating duplicate
# chunks in Qdrant (since each ingestion generates a fresh source_id).
if "ingested_source_keys" not in st.session_state:
    st.session_state.ingested_source_keys = set()


# ---------------------------------------------------------------------
# SIDEBAR: adding sources
# ---------------------------------------------------------------------

st.sidebar.header("Add your sources")

# Show sources that ALREADY exist in Qdrant for this session_id - this
# is what makes the sidebar reflect reality even after reopening the
# app, instead of always looking empty regardless of what's actually
# been ingested and is still searchable.
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

# Show what's already been added, so it's obvious at a glance instead
# of having to remember or scroll back through old messages.
if st.session_state.ingested_source_keys:
    st.sidebar.caption(f"{len(st.session_state.ingested_source_keys)} source(s) already added this session.")

if st.sidebar.button("Ingest sources"):
    sources_to_process = []

    # Only include a source if the user actually filled it in - the
    # user might only paste ONE type of source, and that's fine (we
    # designed for this case earlier).
    if pdf_file is not None:
        # We build a stable key from the filename + size, so the SAME
        # uploaded file (which stays in the widget across reruns)
        # isn't re-ingested every time the button is clicked.
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
            # ingest_sources() only needs "type" and "reference" - we
            # strip the "key" field back out before calling it, since
            # that field only exists for our own tracking above.
            result = ingest_sources([
                {"type": s["type"], "reference": s["reference"]} for s in sources_to_process
            ])

            # For every successfully ingested document, chunk it,
            # embed it, and store it in Qdrant - this is the full
            # ingestion pipeline from our earlier diagram, running
            # end to end for each source.
            for document in result.successful_documents:
                chunks = chunk_document(document, session_id=session_id)
                embedded = embed_chunks(chunks)
                upsert_chunks(embedded)

            # Mark every source we ATTEMPTED as tracked - both
            # successes and failures. We don't want a failure (like a
            # bad YouTube link) to keep retrying silently forever
            # every time the button is clicked; the error message
            # already told the user what happened.
            for s in sources_to_process:
                st.session_state.ingested_source_keys.add(s["key"])

        # Show the user exactly what worked and what didn't - the
        # partial-failure behavior we designed earlier.
        if result.successful_documents:
            st.sidebar.success(f"{len(result.successful_documents)} new source(s) added successfully.")

        for failure in result.failures:
            st.sidebar.error(f"{failure.source_type}: {failure.reason}")


def _format_citation(citation: dict) -> str:
    """
    Turns one citation dict into a short, readable line, e.g.:
      "PDF - page 4"
      "YouTube - 2:05 to 2:35"
      "Web page - https://example.com/article"
    This is plain formatting logic, not something the LLM writes -
    matching the design decision that citations are built from our
    own retrieval data, always accurate.
    """
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

# Redraw the whole conversation so far, on every rerun - this is the
# normal Streamlit pattern for chat interfaces.
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # This is where the "citations shown after completion, not
        # per line" design actually happens: the answer text renders
        # first, and only then, separately, do we render the sources.
        if message.get("citations"):
            with st.expander("Sources"):
                for citation in message["citations"]:
                    st.caption(_format_citation(citation))


user_question = st.chat_input("Ask a question about your sources...")

if user_question:
    # If this is the FIRST message in this conversation, rename it
    # from "New conversation" to something readable, based on the
    # question itself - this is what makes the sidebar chat list
    # actually useful instead of a wall of identical labels.
    is_first_message = len(st.session_state.chat_history) == 0
    if is_first_message:
        auto_title = user_question[:50] + ("..." if len(user_question) > 50 else "")
        update_conversation_title(st.session_state.conversation_id, auto_title)

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # We pass the last few turns of chat history so the
            # rewrite_and_classify node can resolve pronouns like
            # "her", "that", "it" using what was actually discussed
            # just before this question - without this, every
            # question would be treated as if it's the very first
            # one ever asked.
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