"""Download a YouTube video's transcript (with timestamps) given its URL or video ID.

Usage:
    python fetch_transcript.py <url_or_id> [--gameweek N] [--out PATH] [--name NAME]
    python fetch_transcript.py --gameweek N --method text_file

Without --gameweek/--out, the transcript is printed to stdout.
With --gameweek N, it's saved to gw/gwN/transcripts/<NAME>.txt (NAME defaults to the video ID).
With --method text_file, URLs are read from that gameweek's transcripts/transcript_urls.txt.

Prefers manually-created captions over auto-generated ones, when both exist.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url_or_id: str) -> str:
    """Pull an 11-character YouTube video ID out of a URL, or return it unchanged if
    it already looks like a bare ID."""
    candidate = url_or_id.strip()
    if VIDEO_ID_RE.match(candidate):
        return candidate

    parsed = urlparse(candidate)
    host = parsed.netloc.lower()

    if "youtu.be" in host:
        video_id = parsed.path.lstrip("/")
        if VIDEO_ID_RE.match(video_id):
            return video_id

    if "youtube.com" in host:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
            if VIDEO_ID_RE.match(video_id):
                return video_id
        for prefix in ("/embed/", "/shorts/", "/live/"):
            if parsed.path.startswith(prefix):
                video_id = parsed.path[len(prefix):].split("/")[0]
                if VIDEO_ID_RE.match(video_id):
                    return video_id

    raise ValueError(f"Could not extract a YouTube video ID from: {url_or_id!r}")


def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def fetch_transcript_text(
    video_id: str, languages: list[str], paragraph_seconds: float = 15.0
) -> str:
    """Fetch a transcript and group snippets into paragraphs spanning roughly
    `paragraph_seconds` each, rather than emitting one timestamped line per raw
    caption snippet. This turns choppy, mid-sentence fragments into flowing prose
    with one timestamp per paragraph, which is cheaper and easier for an LLM (or a
    human) to read downstream."""
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)

    try:
        transcript = transcript_list.find_manually_created_transcript(languages)
    except NoTranscriptFound:
        transcript = transcript_list.find_generated_transcript(languages)

    fetched = transcript.fetch()

    paragraphs: list[str] = []
    paragraph_start: float | None = None
    paragraph_words: list[str] = []
    for snippet in fetched:
        if paragraph_start is None:
            paragraph_start = snippet.start
        paragraph_words.append(snippet.text)
        if snippet.start - paragraph_start >= paragraph_seconds:
            paragraphs.append(f"[{format_timestamp(paragraph_start)}] {' '.join(paragraph_words)}")
            paragraph_start = None
            paragraph_words = []
    if paragraph_words:
        assert paragraph_start is not None
        paragraphs.append(f"[{format_timestamp(paragraph_start)}] {' '.join(paragraph_words)}")

    return "\n\n".join(paragraphs)


def fetch_to_file(url_or_id: str, languages: list[str], out_path: Path) -> bool:
    try:
        video_id = extract_video_id(url_or_id)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return False

    try:
        text = fetch_transcript_text(video_id, languages)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as exc:
        print(f"Error: no transcript available for {video_id}: {exc}", file=sys.stderr)
        return False

    out_path.mkdir(parents=True, exist_ok=True)
    transcript_path = out_path / f"{video_id}.txt"
    transcript_path.write_text(text, encoding="utf-8")
    print(f"Transcript written to {transcript_path.relative_to(PROJECT_ROOT)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url_or_id", nargs="?", help="YouTube video URL or bare video ID")
    parser.add_argument("--gameweek", type=int, help="Save into gw/gw{N}/transcripts/")
    parser.add_argument(
        "--method",
        choices=["single", "text_file"],
        default="single",
        help="Fetch one URL or all URLs in the gameweek's transcript_urls.txt",
    )
    parser.add_argument("--out", type=Path, help="Explicit output file path")
    parser.add_argument("--name", help="Output filename stem (default: video ID)")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="Preferred transcript language codes, in priority order (default: en)",
    )
    args = parser.parse_args()

    if args.method == "text_file":
        if args.gameweek is None:
            parser.error("--method text_file requires --gameweek")
        if args.url_or_id or args.out or args.name:
            parser.error("--method text_file cannot be combined with url_or_id, --out, or --name")

        transcripts_dir = PROJECT_ROOT / "gw" / f"gw{args.gameweek}" / "transcripts"
        urls_path = transcripts_dir / "transcript_urls.txt"
        if not urls_path.is_file():
            parser.error(f"URL list not found: {urls_path}")

        urls = [
            line.strip()
            for line in urls_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not urls:
            parser.error(f"No URLs found in {urls_path}")

        successful = sum(fetch_to_file(url, args.languages, transcripts_dir) for url in urls)
        print(f"Fetched {successful} of {len(urls)} transcripts.")
        if successful == 0:
            sys.exit(1)
        return

    if not args.url_or_id:
        parser.error("url_or_id is required unless --method text_file is used")

    try:
        video_id = extract_video_id(args.url_or_id)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        text = fetch_transcript_text(video_id, args.languages)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as exc:
        print(f"Error: no transcript available for {video_id}: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.out:
        out_path = args.out
    elif args.gameweek:
        transcripts_dir = PROJECT_ROOT / "gw" / f"gw{args.gameweek}" / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        out_path = transcripts_dir / f"{args.name or video_id}.txt"
    else:
        print(text)
        return

    out_path.write_text(text, encoding="utf-8")
    print(f"Transcript written to {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
