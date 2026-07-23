"""
web_loader.py

This file's ONE job: take a web page URL, download and clean its
content, and turn it into an IngestedDocument.

We use a library called "trafilatura" - it's built specifically for
pulling out the MAIN article text from a page, while automatically
throwing away navigation menus, ads, footers, and other clutter that
a basic HTML parser would keep.
"""

import hashlib
import trafilatura

from ingestion.schema import IngestedDocument, Segment
from ingestion.errors import WebPageUnreachableError


def load_web(url: str) -> IngestedDocument:
    """
    Downloads a web page and extracts its main readable text.

    Parameters:
        url: the web page link pasted by the user

    Returns:
        An IngestedDocument containing the page's cleaned text.

    Raises:
        WebPageUnreachableError: if the page can't be downloaded,
            or if we get it but there's no real article text in it
    """

    # ---- Step 1: download the raw page ----
    # trafilatura.fetch_url() returns None if the download fails
    # (bad URL, site is down, blocked, etc.) instead of raising an error,
    # so we check for that ourselves.
    downloaded = trafilatura.fetch_url(url)

    if downloaded is None:
        raise WebPageUnreachableError(f"Could not download page: {url}")

    # ---- Step 2: extract just the clean article text ----
    # This is the part that strips out ads, nav bars, footers, etc.
    extracted_text = trafilatura.extract(downloaded)

    # extract() also returns None if it couldn't find real article
    # content on the page (e.g. it was just a login screen or an
    # empty page).
    if extracted_text is None or len(extracted_text.strip()) == 0:
        raise WebPageUnreachableError(f"No readable content found on page: {url}")

    # ---- Step 3: build and return the IngestedDocument ----
    # A web page has no natural "page number" or "timestamp" the way
    # a PDF or video does, so we just keep it as ONE segment covering
    # the whole page.
    #
    # The source_id is a hash of the URL itself, normalized to
    # lowercase with no trailing slash - so pasting the same link
    # again (even with small formatting differences) still produces
    # the same source_id, and overwrites instead of duplicating.
    normalized_url = url.strip().lower().rstrip("/")
    source_id = hashlib.sha256(normalized_url.encode()).hexdigest()

    return IngestedDocument(
        source_id=source_id,
        source_type="web",
        segments=[Segment(text=extracted_text)],
        source_url=url,
    )