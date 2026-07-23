"""
create_index.py

A ONE-TIME, standalone script - run this directly (not through
Streamlit) to create the missing session_id index on your Qdrant
collection.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

from config.settings import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME

print(f"Connecting to Qdrant collection: {QDRANT_COLLECTION_NAME}")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

existing_collections = [c.name for c in client.get_collections().collections]
print(f"Collections found: {existing_collections}")

if QDRANT_COLLECTION_NAME not in existing_collections:
    print(f"ERROR: collection '{QDRANT_COLLECTION_NAME}' does not exist yet.")
    print("Ingest at least one source through the app first, then run this script.")
else:
    print(f"Creating index on 'session_id' for collection '{QDRANT_COLLECTION_NAME}'...")
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="session_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print("SUCCESS: index created (or already existed). You can now ask questions in the app.")