"""
reranker.py

This file's ONE job: take the chunks that hybrid search already
found, and re-score them MORE ACCURATELY before they get sent to the
LLM.

Why do we need this extra step, if we already searched in
vector_store.py? Because embedding-based search (dense + sparse) is
fast but approximate - it compares the query and each chunk
SEPARATELY, then measures how close their vectors are. A cross-encoder
reranker instead looks at the query AND the chunk TOGETHER, at the
same time, which is slower but noticeably more accurate. So the
pattern is: cast a wide net with fast search, then use the slower,
smarter reranker to pick the best few from that net.

We use fastembed's cross-encoder here too, for the same reason as our
other models - it's CPU-only and doesn't need PyTorch.
"""

from fastembed.rerank.cross_encoder import TextCrossEncoder

from ingestion.schema import Chunk


# Same pattern as embeddings.py: load the model ONCE, here at the top
# of the file, and reuse it everywhere - loading it fresh every time
# would be slow.
_reranker_model = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")


def rerank_chunks(query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
    """
    Re-scores a list of chunks against the query, and returns only the
    best "top_k" of them, in order from most to least relevant.

    Parameters:
        query: the user's (rewritten) question
        chunks: the chunks that came back from hybrid_search()
        top_k: how many of the best chunks to keep

    Returns:
        A list of Chunk objects - the best matches, trimmed down from
        whatever hybrid search originally returned.
    """

    if len(chunks) == 0:
        return []

    chunk_texts = [chunk.chunk_text for chunk in chunks]

    # This is the actual reranking step: the model looks at the query
    # PAIRED with each individual chunk's text, and returns one
    # relevance score per chunk.
    scores = list(_reranker_model.rerank(query, chunk_texts))

    # We now have two matching lists - chunks and their scores - so we
    # pair them up together, then sort by score with the HIGHEST
    # (most relevant) first.
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda pair: pair[1], reverse=True)

    # Keep only the top_k best ones, and drop the scores - we just
    # need the chunks themselves now, in their new best-first order.
    best_chunks = [chunk for chunk, score in scored_chunks[:top_k]]

    return best_chunks
