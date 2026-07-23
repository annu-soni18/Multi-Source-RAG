"""
coordinator.py

This file's ONE job: take the list of sources a user pasted (could be
1 PDF, or a PDF + a YouTube link + a website, in any combination), and
run each one through its matching loader - WITHOUT letting one
failure stop the others.

This is the piece that makes the "partial failure" behavior we
designed earlier actually happen: if 1 of 3 sources fails, the other
2 still get processed normally.
"""

from dataclasses import dataclass

from ingestion.schema import IngestedDocument
from ingestion.errors import IngestionError, get_friendly_message

from ingestion.pdf_loader import load_pdf
from ingestion.youtube_loader import load_youtube
from ingestion.web_loader import load_web
from ingestion.text_loader import load_text


@dataclass
class SourceFailure:
    """Holds info about one source that failed to ingest."""
    source_reference: str   # the file name, URL, or a short text preview
    source_type: str        # "pdf", "youtube", "web", or "text"
    reason: str              # the friendly message to show the user


@dataclass
class IngestionResult:
    """
    The final outcome after trying to ingest ALL the sources the user
    submitted - some may have worked, some may not have.
    """
    successful_documents: list  # list of IngestedDocument objects
    failures: list               # list of SourceFailure objects

    def all_failed(self) -> bool:
        """True only if NOTHING was successfully ingested."""
        return len(self.successful_documents) == 0 and len(self.failures) > 0


# This dictionary connects each source type to the function that
# knows how to load it. Using a dictionary here (instead of a long
# if/elif/elif chain) makes it easy to add a 5th source type later -
# you'd just add one more line here, nothing else changes.
LOADERS = {
    "pdf": load_pdf,
    "youtube": load_youtube,
    "web": load_web,
    "text": load_text,
}


def ingest_sources(sources: list[dict]) -> IngestionResult:
    """
    Processes a list of sources, one at a time, and collects both the
    successes and the failures separately.

    Parameters:
        sources: a list of dicts, each shaped like:
            {"type": "pdf", "reference": "/path/to/file.pdf"}
            {"type": "youtube", "reference": "https://youtube.com/..."}
            {"type": "web", "reference": "https://example.com/article"}
            {"type": "text", "reference": "some pasted text..."}

    Returns:
        An IngestionResult holding everything that succeeded and
        everything that failed.
    """

    successful_documents = []
    failures = []

    for source in sources:
        source_type = source["type"]
        reference = source["reference"]

        loader_function = LOADERS.get(source_type)

        # Safety check: if somehow an unknown source type sneaks in,
        # record it as a failure instead of crashing with a KeyError.
        if loader_function is None:
            failures.append(SourceFailure(
                source_reference=reference,
                source_type=source_type,
                reason=f"Unsupported source type: {source_type}",
            ))
            continue  # move on to the next source

        # This is the key part: EACH source gets its OWN try/except.
        # If this one fails, we record it and move to the next source -
        # we do NOT stop the whole loop.
        try:
            document = loader_function(reference)
            successful_documents.append(document)

        except IngestionError as e:
            # One of OUR known, expected errors - we already have a
            # clean message for it.
            failures.append(SourceFailure(
                source_reference=reference,
                source_type=source_type,
                reason=get_friendly_message(e),
            ))

        except Exception as e:
            # Something we did NOT plan for. We still don't crash -
            # we just record it with the generic fallback message.
            # (In a real deployment, you'd also log the real error
            # here for yourself, e.g. using Python's logging module.)
            failures.append(SourceFailure(
                source_reference=reference,
                source_type=source_type,
                reason=get_friendly_message(e),
            ))

    return IngestionResult(
        successful_documents=successful_documents,
        failures=failures,
    )
