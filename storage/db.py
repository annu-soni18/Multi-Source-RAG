"""
db.py

This file's ONE job: talk to Supabase (our Postgres database) to save
and load chat history. Every other file that needs to read or write
chat data goes through this file - no other file should import the
supabase client directly.
"""

import math
import uuid
from datetime import datetime, timezone

from supabase import create_client

from config.settings import SUPABASE_URL, SUPABASE_KEY
from storage.models import MessageRecord


_client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_or_create_session(session_id: str):
    """
    Makes sure a row exists in the "sessions" table for this
    session_id. If it already exists, we just update its
    last_active_at timestamp instead of creating a duplicate.

    We use "upsert" here - a combined insert-or-update operation -
    which is exactly what we want: create it the first time, refresh
    it every time after that.
    """
    _client.table("sessions").upsert({
        "session_id": session_id,
        "last_active_at": datetime.now(timezone.utc).isoformat(),
    }).execute()


def create_conversation(session_id: str, title: str) -> str:
    """
    Creates a new conversation row, linked to a session.

    Returns:
        The new conversation_id (as a string), so the caller can use
        it right away when saving messages into this conversation.
    """
    conversation_id = str(uuid.uuid4())

    _client.table("conversations").insert({
        "conversation_id": conversation_id,
        "session_id": session_id,
        "title": title,
    }).execute()

    return conversation_id


def save_message(message: MessageRecord):
    """
    Saves ONE message row (either a user question or an assistant
    answer) into the "messages" table.

    We take a MessageRecord (defined in storage/models.py) rather
    than a raw dictionary, so every caller is forced to provide data
    in the correct shape - Python will complain immediately if a
    required field is missing, instead of Supabase quietly rejecting
    a malformed insert later.
    """
    _client.table("messages").insert({
        "message_id": str(uuid.uuid4()),
        "conversation_id": message.conversation_id,
        "role": message.role,
        "content": message.content,
        "rewritten_query": message.rewritten_query,
        "sub_questions": message.sub_questions,
        "retrieved_chunks": message.retrieved_chunks,
        "citations": message.citations,
        "latency_ms": message.latency_ms,
    }).execute()


def get_latest_conversation_id(session_id: str) -> str | None:
    """
    Finds the most recent conversation for this session, if one
    exists - so we can pick up where the user left off instead of
    always starting a brand new (empty-looking) conversation.

    Returns:
        The conversation_id (str) of the most recent conversation, or
        None if this session has never had a conversation before.
    """
    response = (
        _client.table("conversations")
        .select("conversation_id")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if len(response.data) == 0:
        return None

    return response.data[0]["conversation_id"]


def get_messages(conversation_id: str) -> list[dict]:
    """
    Fetches every message in a conversation, in the order they
    happened - used to rebuild the chat history shown on screen when
    a user reopens the app.

    Returns:
        A list of dicts, each shaped like:
        {"role": "user" or "assistant", "content": "...", "citations": [...]}
        - matching exactly the shape app.py already uses for
        st.session_state.chat_history, so no translation is needed.
    """
    response = (
        _client.table("messages")
        .select("role, content, citations")
        .eq("conversation_id", conversation_id)
        .order("created_at")
        .execute()
    )

    return response.data


def list_conversations(session_id: str) -> list[dict]:
    """
    Fetches every conversation for this session, most recent first -
    this is what powers the "chat history" list in the sidebar, so
    the user can see and switch between past conversations instead of
    only ever seeing the latest one.

    Returns:
        A list of dicts: [{"conversation_id": ..., "title": ...}, ...]
    """
    response = (
        _client.table("conversations")
        .select("conversation_id, title")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .execute()
    )

    return response.data


def update_conversation_title(conversation_id: str, title: str):
    """
    Renames a conversation - we use this to auto-title a conversation
    after its FIRST question, so the sidebar list shows something
    readable ("What is UDP?") instead of "New conversation" for every
    single entry.
    """
    _client.table("conversations").update({"title": title}).eq(
        "conversation_id", conversation_id
    ).execute()


def save_eval_run(commit_sha: str, scores: dict):
    """
    Saves ONE RAGAS evaluation run's scores, tagged with the git
    commit that produced them. This is what powers the Grafana
    dashboard later - each CI run adds one row here, and Grafana
    reads this table directly to plot scores over time and detect
    regressions.
    """

    def _safe_score(value):
        # Supabase's API sends data as JSON, and JSON has no way to
        # represent NaN - trying to send one crashes with a cryptic
        # error. We convert any NaN or missing score to a plain
        # database NULL instead, which Postgres and Grafana both
        # handle gracefully (shown as a gap, not a crash).
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    _client.table("eval_runs").insert({
        "run_id": str(uuid.uuid4()),
        "commit_sha": commit_sha,
        "faithfulness": _safe_score(scores.get("faithfulness")),
        "answer_relevancy": _safe_score(scores.get("answer_relevancy")),
        "llm_context_precision_without_reference": _safe_score(scores.get("llm_context_precision_without_reference")),
        "context_recall": _safe_score(scores.get("context_recall")),
    }).execute()