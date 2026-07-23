"""
text_loader.py

This file's ONE job: take plain text the user pasted directly (no
file, no link - just typed or pasted text), and turn it into an
IngestedDocument.

This is the simplest loader of the four, because there's no file to
open and no network request that can fail - the text is already right
there in front of us.
"""

import hashlib

from ingestion.schema import IngestedDocument, Segment
from ingestion.errors import EmptyTextError


def load_text(raw_text: str) -> IngestedDocument:
    """
    Wraps plain pasted text into an IngestedDocument.

    Parameters:
        raw_text: the text the user typed or pasted directly

    Returns:
        An IngestedDocument containing that text, unchanged.

    Raises:
        EmptyTextError: if the user submitted nothing (or only spaces)
    """

    # Even though this loader is simple, we still check for the one
    # thing that CAN go wrong: the user submitting empty text
    # (e.g. they clicked "add" without typing anything).
    if len(raw_text.strip()) == 0:
        raise EmptyTextError("Submitted text is empty.")

    # source_id is a hash of the text itself - pasting the EXACT same
    # text again always produces the same source_id, so it overwrites
    # instead of duplicating.
    source_id = hashlib.sha256(raw_text.strip().encode()).hexdigest()

    return IngestedDocument(
        source_id=source_id,
        source_type="text",
        segments=[Segment(text=raw_text)],
        source_url=None,  # pasted text has no URL
    )