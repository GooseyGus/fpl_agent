from __future__ import annotations

# Segmentation work.

import hashlib
import json
import re
from pathlib import Path
from typing import Any

TRANSCRIPT_CHAR_LIMIT = 80_000
PROMPT_VERSION = "v15"
MAX_CONCURRENT_TRANSCRIPTS = 6
SEGMENTATION_TIMEOUT_SECONDS = 90
SEGMENTATION_MAX_RETRIES = 0
KEEP_SEGMENT_TYPES = {"FPL_ANALYSIS"}
VALID_SEGMENT_TYPES = KEEP_SEGMENT_TYPES | {"INTRO", "ADVERTISEMENT", "BANTER", "OTHER"}
model = "gpt-4.1-mini"

SEGMENT_PROMPT_TEMPLATE = """You divide a raw auto-generated YouTube transcript about Fantasy Premier League (FPL) into broad, coherent topic sections.

Each paragraph below is numbered and starts with a [MM:SS] timestamp marker. Group consecutive paragraphs that belong to the same broader discussion, such as `Arsenal`, `Defenders`, `Captaincy`, `Fixture strategy`, or `Advertisement`. Prefer a small number of substantial sections over many timestamp-sized sections. Start a new section only when the main subject or content type clearly changes.

Most content is substantive FPL analysis (player names, stats, transfers, captaincy, fixtures, chip strategy, squad structure, or expert opinions). Classify those sections as `FPL_ANALYSIS`. Classify genuinely non-analysis sections as `INTRO`, `ADVERTISEMENT`, `BANTER`, or `OTHER`.

Return valid JSON only with exactly one key: `sections`.

`sections` must cover every paragraph exactly once, in order, with no gaps or overlaps. Each section must have:
- `start_index`: the first paragraph number in the section (inclusive integer)
- `end_index`: the last paragraph number in the section (inclusive integer)
- `type`: one of `FPL_ANALYSIS`, `INTRO`, `ADVERTISEMENT`, `BANTER`, `OTHER`
- `topic`: a concise label for the broader subject, preferably a club, position group, player comparison, or strategy topic

Do not reproduce, rewrite, or quote paragraph text. Only report section boundaries, types, and topic labels.

PARAGRAPHS:
{numbered_text}"""


def _cache_key(*parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:24]


def _read_cache(cache_dir: Path, key: str):
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_cache(cache_dir: Path, key: str, data) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def collect_transcripts(transcript_dir: Path) -> list[dict[str, str]]:
    transcripts = []
    for transcript_path in sorted(transcript_dir.glob("*.txt")):
        if transcript_path.name == "transcript_urls.txt":
            continue
        text = transcript_path.read_text(encoding="utf-8", errors="replace")
        transcripts.append({"source": transcript_path.stem, "text": text[:TRANSCRIPT_CHAR_LIMIT]})
    return transcripts


def split_paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"(?=\[\d{1,2}:\d{2}\])", text) if paragraph.strip()]


def _section_timestamp(text: str, *, last: bool = False) -> str | None:
    timestamps = re.findall(r"\[(\d{1,2}:\d{2})\]", text)
    if not timestamps:
        return None
    return timestamps[-1] if last else timestamps[0]


def _normalise_section_specs(raw_sections: Any, paragraph_count: int) -> list[dict[str, Any]]:
    candidates = []
    if isinstance(raw_sections, list):
        for item in raw_sections:
            if not isinstance(item, dict):
                continue
            raw_start_index = item.get("start_index")
            raw_end_index = item.get("end_index")
            if not isinstance(raw_start_index, (int, str)) or not isinstance(raw_end_index, (int, str)):
                continue
            try:
                start_index = int(raw_start_index)
                end_index = int(raw_end_index)
            except (TypeError, ValueError):
                continue
            if start_index < 0 or end_index < start_index or start_index >= paragraph_count:
                continue
            segment_type = str(item.get("type", "OTHER")).upper()
            if segment_type not in VALID_SEGMENT_TYPES:
                segment_type = "OTHER"
            topic = str(item.get("topic", "")).strip() or segment_type.replace("_", " ").title()
            candidates.append(
                {
                    "start_index": start_index,
                    "end_index": min(end_index, paragraph_count - 1),
                    "type": segment_type,
                    "topic": topic,
                }
            )

    normalised = []
    next_index = 0
    for candidate in sorted(candidates, key=lambda item: (item["start_index"], item["end_index"])):
        if candidate["end_index"] < next_index:
            continue
        if candidate["start_index"] > next_index:
            normalised.append(
                {
                    "start_index": next_index,
                    "end_index": candidate["start_index"] - 1,
                    "type": "FPL_ANALYSIS",
                    "topic": "General FPL discussion",
                }
            )
        candidate["start_index"] = max(candidate["start_index"], next_index)
        normalised.append(candidate)
        next_index = candidate["end_index"] + 1
        if next_index >= paragraph_count:
            break

    if next_index < paragraph_count:
        normalised.append(
            {
                "start_index": next_index,
                "end_index": paragraph_count - 1,
                "type": "FPL_ANALYSIS",
                "topic": "General FPL discussion",
            }
        )
    return normalised


def build_analysis_text(segments: list[dict[str, Any]]) -> str:
    retained_sections = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if segment.get("type") not in KEEP_SEGMENT_TYPES or not text:
            continue
        topic = str(segment.get("topic", "General FPL discussion")).strip()
        start_timestamp = _section_timestamp(text)
        end_timestamp = _section_timestamp(text, last=True)
        timestamp_range = f"{start_timestamp}-{end_timestamp}" if start_timestamp and end_timestamp else "unknown"
        retained_sections.append(f"[SECTION: {topic} | {timestamp_range}]\n{text}")
    return "\n\n".join(retained_sections)


def segment_transcript(client: Any, model: str, cache_dir: Path, text: str) -> list[dict[str, Any]]:
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    key = _cache_key("segment", PROMPT_VERSION, model, text)
    cached = _read_cache(cache_dir, key)
    if cached is not None:
        section_specs = _normalise_section_specs(cached.get("sections", []), len(paragraphs))
    else:
        numbered_text = "\n".join(f"{index}: {paragraph}" for index, paragraph in enumerate(paragraphs))
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": SEGMENT_PROMPT_TEMPLATE.format(numbered_text=numbered_text)},
            ],
            temperature=0,
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
        section_specs = _normalise_section_specs(parsed.get("sections", []), len(paragraphs))
        _write_cache(cache_dir, key, {"sections": section_specs})

    return [
        {
            **section,
            "text": "\n\n".join(paragraphs[section["start_index"] : section["end_index"] + 1]),
        }
        for section in section_specs
    ]