"""
settings.py

This file's ONE job: read all our secret keys and config values from
environment variables, ONE TIME, in ONE place.

Why not just call os.getenv("QDRANT_URL") directly inside
vector_store.py, groq_client.py, etc.? Because if we did that, the
same environment variable name would be typed out in five different
files. If we ever renamed it, or made a typo, we'd have to hunt
through the whole project. Keeping it here means every other file
just imports these ready-made values instead.
"""

import os
from dotenv import load_dotenv

# This reads the .env file (if one exists locally) and loads its
# values into the environment. On Streamlit Cloud, these same values
# come from the "Secrets" settings instead of a .env file - but the
# rest of our code doesn't need to know the difference.
load_dotenv()


# ---- Qdrant (our vector database) ----
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "docs"

# ---- Groq (our LLM provider) ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_GENERATION_MODEL = "llama-3.3-70b-versatile"
GROQ_REWRITE_MODEL = "llama-3.1-8b-instant"

# ---- Supabase (our chat history database) ----
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def check_required_settings():
    """
    Checks that every required environment variable was actually set.

    Why do we need this? Without it, a missing API key would cause a
    confusing error much later, deep inside some unrelated function,
    making it hard to figure out what actually went wrong. This
    function fails LOUDLY and EARLY, right when the app starts, with
    a message that clearly says what's missing.
    """
    required = {
        "QDRANT_URL": QDRANT_URL,
        "QDRANT_API_KEY": QDRANT_API_KEY,
        "GROQ_API_KEY": GROQ_API_KEY,
        "SUPABASE_URL": SUPABASE_URL,
        "SUPABASE_KEY": SUPABASE_KEY,
    }

    missing = [name for name, value in required.items() if not value]

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Check your .env file (local) or Secrets settings (deployed)."
        )
