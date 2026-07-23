"""
models.py

This file holds the actual database table definitions (as SQL), and
matching Python dataclasses for the data we'll save.

IMPORTANT: Supabase doesn't let us create tables from Python code the
way Qdrant lets us create a collection. Instead, you run this SQL
ONCE, yourself, in the Supabase dashboard's "SQL Editor" tab.
"""

from dataclasses import dataclass
from typing import Optional


SETUP_SQL = """
create table if not exists sessions (
    session_id uuid primary key,
    created_at timestamptz default now(),
    last_active_at timestamptz default now()
);

create table if not exists conversations (
    conversation_id uuid primary key,
    session_id uuid references sessions(session_id),
    title text,
    created_at timestamptz default now()
);

create table if not exists messages (
    message_id uuid primary key,
    conversation_id uuid references conversations(conversation_id),
    role text not null,
    content text not null,
    rewritten_query text,
    sub_questions jsonb,
    retrieved_chunks jsonb,
    citations jsonb,
    latency_ms integer,
    created_at timestamptz default now()
);

create index if not exists idx_conversations_session
    on conversations(session_id);

create index if not exists idx_messages_conversation
    on messages(conversation_id, created_at);

create table if not exists eval_runs (
    run_id uuid primary key,
    commit_sha text,
    faithfulness float,
    answer_relevancy float,
    llm_context_precision_without_reference float,
    context_recall float,
    created_at timestamptz default now()
);

create index if not exists idx_eval_runs_created
    on eval_runs(created_at);
"""


@dataclass
class MessageRecord:
    conversation_id: str
    role: str
    content: str
    rewritten_query: Optional[str] = None
    sub_questions: Optional[list] = None
    retrieved_chunks: Optional[list] = None
    citations: Optional[list] = None
    latency_ms: Optional[int] = None