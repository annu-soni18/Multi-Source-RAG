"""
chunking.py

This file's ONE job: take one IngestedDocument (now a LIST of location-
tagged Segments - see schema.py) and cut it into smaller Chunk pieces,
ready to be embedded - while keeping each chunk's page number or
timestamp attached, so citations can point to an exact location later.
"""

from ingestion.schema import IngestedDocument, Chunk


CHUNK_CONFIG = {
    "pdf":     {"size": 800, "overlap": 100},
    "youtube": {"size": 500, "overlap": 50},
    "web":     {"size": 800, "overlap": 100},
    "text":    {"size": 600, "overlap": 80},
}


def split_text_into_windows(text: str, size: int, overlap: int) -> list[str]:
    """
    Cuts one string into a list of smaller overlapping pieces.
    (See the earlier version's docstring for the full explanation of
    why we use overlap - same idea here, just applied per-segment now
    instead of on one giant flattened string.)
    """
    windows = []
    step = size - overlap
    start = 0

    while start < len(text):
        end = start + size
        windows.append(text[start:end])
        start += step

    return windows


def chunk_document(document: IngestedDocument, session_id: str) -> list[Chunk]:
    """
    Turns one IngestedDocument into a list of Chunk objects.

    The key change from before: we now loop over each SEGMENT first
    (each page, for a PDF; each timestamped group, for YouTube), and
    chunk WITHIN that segment. This means every resulting chunk
    naturally inherits that segment's page_number or start_time/
    end_time - we don't have to guess it after the fact.
    """

    config = CHUNK_CONFIG[document.source_type]
    chunks = []
    chunk_index = 0  # counts UP across the whole document, not reset per segment

    for segment in document.segments:
        text_windows = split_text_into_windows(
            text=segment.text,
            size=config["size"],
            overlap=config["overlap"],
        )

        for window_text in text_windows:
            chunk = Chunk(
                chunk_text=window_text,
                session_id=session_id,
                source_id=document.source_id,
                source_type=document.source_type,
                chunk_index=chunk_index,
                source_url=document.source_url,
                # These three lines are the whole point of today's
                # change: each chunk now carries the EXACT location
                # it came from, straight from its parent segment.
                page_number=segment.page_number,
                start_time=segment.start_time,
                end_time=segment.end_time,
            )
            chunks.append(chunk)
            chunk_index += 1

    return chunks
