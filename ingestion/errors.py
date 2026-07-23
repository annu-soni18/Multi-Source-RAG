"""
errors.py

This file defines what can go wrong during ingestion (PDF/YouTube/
web/text processing), and what message we should show the USER for
each problem.

Why not just use Python's built-in errors everywhere?
Because a generic error like "Exception: 'NoneType' object has no
attribute 'text'" means nothing to a normal user. Instead, we create
our OWN named errors (like "TranscriptsDisabledError") so that:
  1. Our code can catch a SPECIFIC problem, not just "something broke"
  2. We can show a clear, human-friendly message for each one
"""


# ---------------------------------------------------------------------
# STEP 1: Define one custom error class per "thing that can go wrong"
# ---------------------------------------------------------------------
# Each of these is just a named box to put a specific failure in.
# They all inherit from Python's built-in Exception class.

class IngestionError(Exception):
    """Base class for every ingestion error. Other errors below build on this."""
    pass


# --- PDF-specific problems ---
class EmptyPDFError(IngestionError):
    """Raised when a PDF has no readable text (e.g. it's just scanned images)."""
    pass


class CorruptedPDFError(IngestionError):
    """Raised when the PDF file can't even be opened."""
    pass


class PasswordProtectedPDFError(IngestionError):
    """Raised when a PDF is locked with a password."""
    pass


# --- YouTube-specific problems ---
class YouTubeBlockedError(IngestionError):
    """Raised when YouTube blocks our server's request (common on cloud hosting)."""
    pass


class TranscriptsDisabledError(IngestionError):
    """Raised when the video's owner has turned off captions/transcripts."""
    pass


class VideoUnavailableError(IngestionError):
    """Raised when the video is private, deleted, or region-locked."""
    pass


# --- Web page-specific problems ---
class WebPageUnreachableError(IngestionError):
    """Raised when we can't load the web page at all (bad URL, site down, etc.)."""
    pass


# --- Plain text-specific problems ---
class EmptyTextError(IngestionError):
    """Raised when the user submits text that's empty or just whitespace."""
    pass


# ---------------------------------------------------------------------
# STEP 2: Map each error type to a friendly message
# ---------------------------------------------------------------------
# This dictionary is the ONLY place in the whole project where these
# messages live. If you ever want to reword one, you only change it
# here - not in five different files.

ERROR_MESSAGES = {
    EmptyPDFError: "This PDF appears to be scanned images with no readable text. Try a text-based PDF, or one with OCR applied.",
    CorruptedPDFError: "This file couldn't be opened - it may be corrupted or not a valid PDF.",
    PasswordProtectedPDFError: "This PDF is password-protected and can't be read. Please upload an unlocked version.",

    YouTubeBlockedError: "YouTube blocked this request from our server. This is a known limitation of free hosting - try a different video, or run this locally.",
    TranscriptsDisabledError: "This video doesn't have captions available.",
    VideoUnavailableError: "This video can't be accessed - it may be private or region-restricted.",

    WebPageUnreachableError: "This web page couldn't be loaded. Please check the link and try again.",

    EmptyTextError: "The text you submitted was empty. Please paste some content.",
}


def get_friendly_message(error: Exception) -> str:
    """
    Given any error that was raised, return the clean message we
    should show the user.

    If it's one of OUR known errors (defined above), we return the
    matching friendly message. If it's some other, unexpected error
    (a bug, a typo, anything we didn't plan for), we return a generic
    safe message instead of leaking a raw error to the user.
    """
    error_type = type(error)

    if error_type in ERROR_MESSAGES:
        return ERROR_MESSAGES[error_type]

    # Fallback for anything we didn't specifically plan for.
    return "Something went wrong while processing this source. Please try a different one."
