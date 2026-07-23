"""
youtube_loader.py

This file's ONE job: take a YouTube video URL, fetch its transcript,
and turn it into an IngestedDocument.

We use the "youtube-transcript-api" library. IMPORTANT: version 1.x
of this library changed its API significantly - the old
YouTubeTranscriptApi.get_transcript(video_id) classmethod was REMOVED
entirely. The new way is instance-based: create a YouTubeTranscriptApi
object, then call .fetch(video_id) on it. This bit us once already -
always verify a library's actual current API rather than trusting
memory of an older version.
"""

import re

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

from ingestion.schema import IngestedDocument, Segment
from ingestion.errors import (
    TranscriptsDisabledError,
    VideoUnavailableError,
    YouTubeBlockedError,
)


# We create ONE instance of the API client here, at the top of the
# file, and reuse it for every video - same pattern as our embedding
# models and Qdrant client elsewhere in the project.
_api = YouTubeTranscriptApi()


def extract_video_id(youtube_url: str) -> str:
    """
    YouTube URLs come in different shapes, e.g.:
      https://www.youtube.com/watch?v=abc123
      https://youtu.be/abc123

    This function pulls out just the "abc123" part (the video ID),
    which is what the transcript API actually needs.
    """
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", youtube_url)
    if not match:
        raise VideoUnavailableError(f"Could not find a video ID in URL: {youtube_url}")
    return match.group(1)


def load_youtube(youtube_url: str) -> IngestedDocument:
    """
    Fetches the transcript for a YouTube video and returns an
    IngestedDocument.

    Parameters:
        youtube_url: the full YouTube video URL pasted by the user

    Returns:
        An IngestedDocument containing the full transcript text.

    Raises:
        TranscriptsDisabledError: the video owner turned off captions
        VideoUnavailableError: the video is private, deleted, or region-locked
        YouTubeBlockedError: our server's IP got blocked by YouTube
    """

    video_id = extract_video_id(youtube_url)

    try:
        # .fetch() returns a FetchedTranscript object, not a plain
        # list - .to_raw_data() converts it into the same list-of-
        # dicts shape the rest of this function expects:
        # [{"text": "...", "start": 0.0, "duration": 2.5}, ...]
        transcript_list = _api.fetch(video_id).to_raw_data()

    except TranscriptsDisabled:
        raise TranscriptsDisabledError(f"Captions disabled for video: {video_id}")

    except (VideoUnavailable, NoTranscriptFound):
        raise VideoUnavailableError(f"Video unavailable or has no transcript: {video_id}")

    except Exception as e:
        # Anything else - most commonly on free cloud hosting - is very
        # likely YouTube blocking our server's IP address. We check the
        # error's class name as text, since the exact exception type can
        # vary between library versions. (Confirmed names in this
        # version include IpBlocked and RequestBlocked.)
        error_name = type(e).__name__
        if "Blocked" in error_name or "TooManyRequests" in error_name:
            raise YouTubeBlockedError(f"YouTube blocked this request for video: {video_id}")
        # If it's something we truly didn't expect, let it bubble up
        # so we can see the real error in our own logs.
        raise

    # transcript_list looks like: [{"text": "...", "start": 0.0, "duration": 2.5}, ...]
    # Instead of joining these into one big string (which would throw
    # away the timestamps), we keep each one as its own Segment. We
    # group every few raw transcript lines together first, though -
    # a single caption line is often only 2-3 seconds long, which
    # would be too small and choppy to be a useful segment on its own.
    segments = []
    GROUP_SIZE = 8  # how many raw transcript lines to combine into one segment

    for i in range(0, len(transcript_list), GROUP_SIZE):
        group = transcript_list[i : i + GROUP_SIZE]
        combined_text = " ".join(line["text"] for line in group)

        segments.append(Segment(
            text=combined_text,
            start_time=group[0]["start"],                              # when this group starts
            end_time=group[-1]["start"] + group[-1]["duration"],       # when this group ends
        ))

    return IngestedDocument(
        # The video_id itself is already a stable, unique identifier
        # for this exact video - using it directly means re-ingesting
        # the same video (even in a different browser session, even
        # after restarting the app) always produces the SAME
        # source_id, so it overwrites instead of duplicating.
        source_id=f"youtube-{video_id}",
        source_type="youtube",
        segments=segments,
        source_url=youtube_url,
    )