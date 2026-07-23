"""
schema.py

This file defines the "shape" of our data - like a blueprint.

Every piece of text we store (whether it came from a PDF, a YouTube
video, a web page, or plain text) will follow this SAME shape. This is
what lets us keep everything in one place instead of needing separate
code for each source type.

If you're new to Python: a "dataclass" is just a quick way to create a
class that mainly holds data, without writing __init__ by hand.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Segment:
    """
    One small, LOCATION-TAGGED piece of a document, before chunking.

    Why do we need this extra layer, instead of going straight from
    "whole document" to "chunks"? Because a PDF naturally comes in
    pages, and a YouTube transcript naturally comes in timestamped
    lines. If we squash everything into one giant string too early,
    we lose that location information forever. Segment is how we
    carry it forward until chunking happens.
    """
    text: str
    page_number: Optional[int] = None      # filled in for PDF segments
    start_time: Optional[float] = None     # filled in for YouTube segments
    end_time: Optional[float] = None       # filled in for YouTube segments


@dataclass
class IngestedDocument:
    """
    Represents ONE full source, right after we've pulled its content
    out - but BEFORE we've cut it into embedding-sized chunks.

    Instead of one big flat string, this now holds a LIST of Segments,
    so location information (page number / timestamp) survives all
    the way through to chunking.
    """

    source_id: str        # a unique ID we generate for this source
    source_type: str      # one of: "pdf", "youtube", "web", "text"
    segments: list        # list[Segment] - see the Segment class above
    source_url: Optional[str] = None
    # ^ only filled in for "web" and "youtube" sources


@dataclass
class Chunk:
    """
    One small piece of text, ready to be turned into embeddings and
    stored. Think of this as one "index card" from our earlier example.
    """

    # ---- Fields every chunk has, no matter the source ----
    chunk_text: str        # the actual text on this "card"
    session_id: str        # which user session this belongs to
    source_id: str         # which source (this exact PDF/video/etc.) this came from
    source_type: str       # "pdf", "youtube", "web", or "text"
    chunk_index: int       # position of this chunk within its source (0, 1, 2, ...)

    # ---- Fields that only some source types use ----
    # We use "Optional" here, which means: this can either hold a real
    # value, or be empty (None). A "text" chunk, for example, will never
    # have a page_number, so it just stays None for that one.
    page_number: Optional[int] = None      # only used for PDF chunks
    start_time: Optional[float] = None     # only used for YouTube chunks (in seconds)
    end_time: Optional[float] = None       # only used for YouTube chunks (in seconds)
    source_url: Optional[str] = None       # only used for web page chunks

    def to_payload(self) -> dict:
        """
        Converts this Chunk object into a plain Python dictionary.

        Why do we need this? Qdrant (our vector database) doesn't
        understand Python classes - it only understands plain
        dictionaries (it calls this a "payload"). So whenever we're
        about to save a chunk into Qdrant, we call this function first.
        """
        return {
            "chunk_text": self.chunk_text,
            "session_id": self.session_id,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "source_url": self.source_url,
        }
