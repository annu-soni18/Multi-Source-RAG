"""
embeddings.py

This file's ONE job: take a list of Chunks (from chunking.py) and turn
each one into TWO kinds of numbers:

  1. A "dense" vector - the meaning-based embedding we talked about
     earlier (using a MiniLM model).
  2. A "sparse" vector - the keyword-based fingerprint (using BM25).

Both come from the "fastembed" library, which runs on CPU with no
PyTorch needed - important for keeping our deployment free and light.

Note: the very first time these models run, fastembed downloads their
weight files from the internet (a few hundred MB) and caches them
locally. After that first download, it's fast.
"""

from fastembed import TextEmbedding, SparseTextEmbedding

from ingestion.schema import Chunk


# We only want to load each model ONCE (loading them is slow), not
# every time we embed something. So we create them here, at the
# module level, and every other file that imports from this module
# will reuse the same loaded models.
_dense_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
_sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def embed_chunks(chunks: list[Chunk]) -> list[dict]:
    """
    Takes a list of Chunk objects and returns a list of dictionaries,
    where each dictionary bundles together:
      - the original chunk
      - its dense vector (a list of ~384 numbers)
      - its sparse vector (a smaller list of "important word" scores)

    We process all chunks together in one call to embed(), rather
    than one at a time in a loop - this is much faster, since the
    model can work on a whole batch at once instead of restarting for
    each individual chunk.
    """

    texts = [chunk.chunk_text for chunk in chunks]

    # .embed() returns a generator, so we wrap it in list() to
    # actually run it and get real results back.
    dense_vectors = list(_dense_model.embed(texts))
    sparse_vectors = list(_sparse_model.embed(texts))

    results = []
    for chunk, dense_vector, sparse_vector in zip(chunks, dense_vectors, sparse_vectors):
        results.append({
            "chunk": chunk,
            "dense_vector": dense_vector.tolist(),  # convert from numpy array to a plain list
            "sparse_indices": sparse_vector.indices.tolist(),
            "sparse_values": sparse_vector.values.tolist(),
        })

    return results


def embed_query(query_text: str) -> dict:
    """
    Embeds ONE piece of text (a user's question) instead of a whole
    batch of chunks. We reuse the exact same models as embed_chunks(),
    since the query and the stored chunks MUST be embedded the same
    way for search to work correctly - comparing vectors from two
    different models would give meaningless results.

    Returns a dict with dense_vector, sparse_indices, and sparse_values -
    ready to hand straight to vector_store.hybrid_search().
    """
    dense_vector = list(_dense_model.embed([query_text]))[0]
    sparse_vector = list(_sparse_model.embed([query_text]))[0]

    return {
        "dense_vector": dense_vector.tolist(),
        "sparse_indices": sparse_vector.indices.tolist(),
        "sparse_values": sparse_vector.values.tolist(),
    }
