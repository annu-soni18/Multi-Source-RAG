"""
pdf_loader.py

This file's ONE job: take a PDF file, and turn it into an
IngestedDocument (the "whole document, not yet chunked" object we
defined in schema.py).

We use a library called PyMuPDF (imported as "fitz") to actually read
the PDF - it's fast and handles most real-world PDFs well.
"""

import hashlib
import fitz  # this is PyMuPDF's import name - a bit unusual, but that's how it works

from ingestion.schema import IngestedDocument, Segment
from ingestion.errors import (
    EmptyPDFError,
    CorruptedPDFError,
    PasswordProtectedPDFError,
)


def load_pdf(file_path: str) -> IngestedDocument:
    """
    Reads a PDF file from disk and returns an IngestedDocument.

    Parameters:
        file_path: where the uploaded PDF is saved on disk

    Returns:
        An IngestedDocument holding one Segment PER PAGE, so later
        steps (chunking, citations) know exactly which page any given
        piece of text came from.

    Raises:
        CorruptedPDFError: if the file can't be opened at all
        PasswordProtectedPDFError: if the PDF is locked
        EmptyPDFError: if we open it fine but find no readable text
    """

    # ---- Step 1: try to open the file ----
    try:
        doc = fitz.open(file_path)
    except Exception:
        raise CorruptedPDFError(f"Could not open PDF: {file_path}")

    # ---- Step 2: check if it's password-protected ----
    if doc.is_encrypted:
        raise PasswordProtectedPDFError(f"PDF is password-protected: {file_path}")

    # ---- Step 3: pull text out PAGE BY PAGE, keeping the page number ----
    # This is the key change from before: instead of joining every
    # page into one giant string, we keep one Segment per page, with
    # that page's number attached to it.
    segments = []
    for page_index, page in enumerate(doc):
        page_text = page.get_text()

        # Skip pages that came back empty (e.g. a blank page in the
        # middle of an otherwise normal PDF) - no point keeping an
        # empty segment around.
        if len(page_text.strip()) > 0:
            segments.append(Segment(
                text=page_text,
                page_number=page_index + 1,  # +1 so pages are numbered from 1, not 0
            ))

    # ---- Step 4: check if we actually got any real text at all ----
    if len(segments) == 0:
        raise EmptyPDFError(f"No extractable text found in PDF: {file_path}")

    # ---- Step 5: build a DETERMINISTIC source_id from the file's
    # content, not a random one ----
    # Why? If the user uploads the EXACT SAME PDF twice (accidentally,
    # or because they revisit the app later), we want it to produce
    # the SAME source_id both times - so it overwrites the same
    # points in Qdrant instead of creating a genuine duplicate. We
    # hash the actual file bytes, not the filename - so even if the
    # file gets renamed, the same content still produces the same ID.
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    content_hash = hashlib.sha256(file_bytes).hexdigest()

    return IngestedDocument(
        source_id=content_hash,
        source_type="pdf",
        segments=segments,
        source_url=None,  # PDFs don't have a URL - this stays empty
    )