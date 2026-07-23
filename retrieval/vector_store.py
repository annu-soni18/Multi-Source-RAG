"""
vector_store.py

This file's ONE job: talk to Qdrant - our vector database. It handles:
  1. Setting up the collection (the "box" we store all chunks in)
  2. Saving embedded chunks into it
  3. Searching it using HYBRID search (dense + sparse together)

Everything else in the project that needs Qdrant goes through this
file - no other file should import qdrant_client directly.
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
    PointStruct,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    FusionQuery,
    Fusion,
    PayloadSchemaType,
)

from config.settings import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME


# Just like the embedding models in embeddings.py, we create ONE
# client connection here and reuse it everywhere, instead of opening
# a new connection every time we need to talk to Qdrant.
#
# timeout=60 (seconds) is higher than the client's default. Free-tier
# Qdrant Cloud clusters can be slow to respond, especially on the
# first request after being idle - the default timeout was too short
# and caused write operations to fail with a timeout error.
_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)


def ensure_collection_exists():
    """
    Creates our Qdrant collection if it doesn't already exist, AND
    makes sure the session_id field has an index.

    Why do we need an index? Qdrant needs to know ahead of time which
    payload fields you plan to FILTER by (like we do in hybrid_search()
    with session_id) - without an index, it refuses the filtered query
    entirely with a "Bad Request" error, which is exactly the error
    that led to this fix.
    """
    existing_collections = [c.name for c in _client.get_collections().collections]

    if QDRANT_COLLECTION_NAME not in existing_collections:
        _client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            # "dense" holds our meaning-based vectors (384 numbers each,
            # since that's the output size of the MiniLM model we chose).
            vectors_config={
                "dense": VectorParams(size=384, distance=Distance.COSINE),
            },
            # "sparse" holds our keyword-based BM25 vectors.
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )

    # This is the actual fix: create an index on session_id so Qdrant
    # allows us to filter by it in hybrid_search(). We call this every
    # time regardless of whether the collection is brand new or already
    # existed, since older collections (created before this fix) won't
    # have the index yet either. Qdrant safely does nothing if the
    # index already exists, so this is safe to call repeatedly.
    _client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="session_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def _make_point_id(session_id: str, source_id: str, chunk_index: int) -> str:
    """
    Builds a DETERMINISTIC id for a chunk, instead of a random one.

    Why deterministic? If the same chunk ever gets ingested twice
    (e.g. a user accidentally pastes the same PDF again, or our code
    retries after a failure), this will produce the EXACT SAME id
    both times - so Qdrant just overwrites the old point instead of
    creating a duplicate.
    """
    raw_key = f"{session_id}:{source_id}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_key))


def upsert_chunks(embedded_chunks: list[dict]):
    """
    Saves a list of embedded chunks into Qdrant.

    Parameters:
        embedded_chunks: the output of embed_chunks() from
            processing/embeddings.py - each item has a "chunk" object
            plus its dense_vector, sparse_indices, and sparse_values.
    """

    points = []
    for item in embedded_chunks:
        chunk = item["chunk"]

        point = PointStruct(
            id=_make_point_id(chunk.session_id, chunk.source_id, chunk.chunk_index),
            vector={
                "dense": item["dense_vector"],
                "sparse": SparseVector(
                    indices=item["sparse_indices"],
                    values=item["sparse_values"],
                ),
            },
            payload=chunk.to_payload(),
        )
        points.append(point)

    # We send points in smaller BATCHES instead of all at once. A long
    # YouTube video or a big PDF can produce hundreds of chunks - trying
    # to upload all of them in a single request is what caused the
    # timeout error. Smaller batches complete faster individually, and
    # if one batch fails, we don't lose everything that came before it.
    BATCH_SIZE = 64

    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        _client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=batch)


def hybrid_search(
    dense_query_vector: list[float],
    sparse_query_indices: list[int],
    sparse_query_values: list[float],
    session_id: str,
    limit: int = 10,
) -> list[dict]:
    """
    Searches for the best matching chunks using BOTH dense (meaning)
    and sparse (keyword) search at once, combined using RRF (Reciprocal
    Rank Fusion) - this is the "hybrid search" we designed earlier.

    Parameters:
        dense_query_vector: the user's query, embedded as a dense vector
        sparse_query_indices / sparse_query_values: the user's query,
            embedded as a sparse vector
        session_id: only search chunks belonging to this session -
            this is what keeps different users' data separate
        limit: how many chunks to return

    Returns:
        A list of dicts, each containing a chunk's payload plus its
        match score.
    """

    results = _client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        # "prefetch" runs BOTH searches - dense and sparse - and Qdrant
        # then fuses their results together into one ranked list.
        prefetch=[
            Prefetch(query=dense_query_vector, using="dense", limit=limit * 2),
            Prefetch(
                query=SparseVector(indices=sparse_query_indices, values=sparse_query_values),
                using="sparse",
                limit=limit * 2,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        # This filter is what scopes the search to just THIS session's
        # sources - so one user's PDF never leaks into another user's
        # search results.
        query_filter=Filter(
            must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
        ),
        limit=limit,
    )

    return [
        {"payload": point.payload, "score": point.score}
        for point in results.points
    ]


def list_sources(session_id: str) -> list[dict]:
    """
    Returns ONE entry per unique source (not per chunk) that's
    already been ingested for this session - used to show "you
    already have these sources added" in the sidebar when the app
    reopens, instead of it looking empty.

    We use Qdrant's "scroll" - basically paging through every point
    that matches our filter - since Qdrant doesn't have a built-in
    "list distinct sources" query. We then deduplicate by source_id
    ourselves, since a single source produces many chunks/points.
    """
    seen_source_ids = set()
    sources = []
    next_page_offset = None

    while True:
        points, next_page_offset = _client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            ),
            limit=200,
            with_payload=True,
            with_vectors=False,
            offset=next_page_offset,
        )

        for point in points:
            payload = point.payload
            source_id = payload.get("source_id")

            if source_id not in seen_source_ids:
                seen_source_ids.add(source_id)
                sources.append({
                    "source_type": payload.get("source_type"),
                    "source_url": payload.get("source_url"),
                })

        # scroll() returns None for next_page_offset once there are no
        # more pages left - that's our signal to stop.
        if next_page_offset is None:
            break

    return sources