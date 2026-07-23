"""
clear_session.py

A ONE-TIME cleanup script: deletes ALL chunks in Qdrant belonging to
a specific session_id. Use this to wipe out duplicate data that got
created before the re-ingestion-duplication bug was fixed in app.py.

Run it like this:
    python clear_session.py YOUR_SESSION_ID
"""

import sys

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config.settings import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME

if len(sys.argv) != 2:
    print("Usage: python clear_session.py YOUR_SESSION_ID")
    sys.exit(1)

session_id = sys.argv[1]

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

print(f"Deleting all chunks for session_id: {session_id}")
client.delete(
    collection_name=QDRANT_COLLECTION_NAME,
    points_selector=Filter(
        must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
    ),
)
print("Done. All chunks for this session have been removed.")