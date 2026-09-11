from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EXTRACTION_PROMPT_VERSION = "v1"

EXTRACTION_PROMPT_TEMPLATE = """You extract structured Fantasy Premier League (FPL) insights from a retained transcript analysis.

The transcript is grouped under `[SECTION: topic | start-end]` metadata headings. Use these broader topics to connect related analysis across consecutive paragraphs. The headings were generated during segmentation: use them as context, but never return them as supporting quotes.

You are given:
- a transcript source id
- the retained analysis text after segmentation
- name-correction decisions that may help normalize player references

Return strict JSON only with exactly three keys: `recommendations`, `captaincy`, and `themes`.

`recommendations` must be an array of objects. Each object must include:
- `player_name`: the player being discussed
- `stance`: one of `buy`, `consider`, `hold`, `avoid`, `sell`
- `sentiment`: a short sentiment label or summary
- `confidence`: a number from 1 to 5
- `recommendation_strength`: an optional numeric or string strength indicator
- `rationale`: a short explanation grounded in the transcript
- `supporting_quote`: a direct quote from the transcript or analysis text
- `reasons`: an array of short reason strings
- `risks`: an array of short risk strings
- `time_horizon`: a short horizon label such as `short-term`, `medium-term`, or `long-term`

`captaincy` must be an array of objects. Each object must include:
- `player_name`
- `mention_type`: one of `triple_captain`, `captain`, `vice_captain`, `avoid_captain`
- `confidence`: a number from 1 to 5
- `expert_percentage`: an optional number from 0 to 100
- `rationale`
- `supporting_quote`

`themes` must be an array of short strings.

Keep supporting quotes faithful to the source. Do not invent players or quotes.

SOURCE:
{source}

NAME_CORRECTIONS:
{name_corrections}

ANALYSIS_TEXT:
{analysis_text}
"""


def _cache_key(*parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:24]


def _read_cache(cache_dir: Path, key: str):
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_cache(cache_dir: Path, key: str, payload: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _normalise_response(parsed: dict[str, Any]) -> dict[str, Any]:
    recommendations = [item for item in parsed.get("recommendations", []) if isinstance(item, dict)]
    captaincy = [item for item in parsed.get("captaincy", []) if isinstance(item, dict)]
    themes = [str(item).strip() for item in parsed.get("themes", []) if str(item).strip()]
    return {
        "recommendations": recommendations,
        "captaincy": captaincy,
        "themes": themes,
    }


def extract_from_transcript(
    client: Any,
    model: str,
    cache_dir: Path,
    source: str,
    analysis_text: str,
    source_name_corrections: list[dict],
) -> dict[str, Any]:
    name_corrections_context = json.dumps(source_name_corrections, ensure_ascii=False, sort_keys=True)
    key = _cache_key(
        "extraction",
        EXTRACTION_PROMPT_VERSION,
        model,
        source,
        analysis_text,
        name_corrections_context,
    )
    cached = _read_cache(cache_dir, key)
    if cached is not None:
        return _normalise_response(cached)

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {
                "role": "user",
                "content": EXTRACTION_PROMPT_TEMPLATE.format(
                    source=source,
                    name_corrections=name_corrections_context,
                    analysis_text=analysis_text,
                ),
            },
        ],
        temperature=0,
    )
    parsed = json.loads(response.choices[0].message.content or "{}")
    normalized = _normalise_response(parsed if isinstance(parsed, dict) else {})
    _write_cache(cache_dir, key, normalized)
    return normalized


def build_transcript_extractions_output(
    *,
    sources: list[str],
    recommendations: list[dict[str, Any]],
    captaincy: list[dict[str, Any]],
    themes: list[tuple[str, str]],
) -> dict[str, Any]:
    by_source: dict[str, dict[str, Any]] = {
        source: {
            "source": source,
            "recommendations": [],
            "captaincy": [],
            "themes": [],
        }
        for source in sources
    }

    for recommendation in recommendations:
        source = str(recommendation.get("source", "")).strip()
        if not source:
            continue
        entry = by_source.setdefault(
            source,
            {
                "source": source,
                "recommendations": [],
                "captaincy": [],
                "themes": [],
            },
        )
        entry["recommendations"].append(recommendation)

    for mention in captaincy:
        source = str(mention.get("source", "")).strip()
        if not source:
            continue
        entry = by_source.setdefault(
            source,
            {
                "source": source,
                "recommendations": [],
                "captaincy": [],
                "themes": [],
            },
        )
        entry["captaincy"].append(mention)

    theme_records: list[dict[str, str]] = []
    for theme, source in themes:
        theme_text = str(theme).strip()
        source_name = str(source).strip()
        if not theme_text or not source_name:
            continue
        theme_record = {"theme": theme_text, "source": source_name}
        theme_records.append(theme_record)
        entry = by_source.setdefault(
            source_name,
            {
                "source": source_name,
                "recommendations": [],
                "captaincy": [],
                "themes": [],
            },
        )
        entry["themes"].append(theme_text)

    ordered_sources = []
    for source in sources + sorted(by_source):
        if source not in ordered_sources:
            ordered_sources.append(source)

    source_records = [by_source[source] for source in ordered_sources if source in by_source]
    return {
        "sources": ordered_sources,
        "counts": {
            "sources": len(ordered_sources),
            "recommendations": len(recommendations),
            "captaincy": len(captaincy),
            "themes": len(theme_records),
        },
        "recommendations": recommendations,
        "captaincy": captaincy,
        "themes": theme_records,
        "source_records": source_records,
    }


def write_transcript_extractions_output(
    output_path: Path,
    *,
    sources: list[str],
    recommendations: list[dict[str, Any]],
    captaincy: list[dict[str, Any]],
    themes: list[tuple[str, str]],
) -> dict[str, Any]:
    output = build_transcript_extractions_output(
        sources=sources,
        recommendations=recommendations,
        captaincy=captaincy,
        themes=themes,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
