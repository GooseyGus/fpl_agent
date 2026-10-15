from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any
from fpl_hq.consensus.llm import DETERMINISTIC

EXTRACTION_PROMPT_VERSION = "v4"
MAX_CONCURRENT_EXTRACTIONS = 6

EXTRACTION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "transcript_player_recommendations",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "player_name": {"type": "string"},
                            "stance": {"type": "integer", "minimum": 1, "maximum": 5},
                            "sentiment": {"type": "string"},
                            "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
                            "supporting_quote": {"type": "string"},
                        },
                        "required": [
                            "player_name",
                            "stance",
                            "sentiment",
                            "confidence",
                            "supporting_quote",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["recommendations"],
            "additionalProperties": False,
        },
    },
}

EXTRACTION_PROMPT_TEMPLATE = """You extract structured Fantasy Premier League (FPL) player recommendations from retained transcript analysis.

The transcript is grouped under `[SECTION: topic | start-end]` metadata headings. FPL videos are organised by topic (position groups, price brackets, teams, fixtures, transfers, captaincy), so a player's assessment may be spread across a section and expressed relative to the other players discussed there. Use the headings as context, but never return them as supporting quotes.

You are given:
- a transcript source id
- the retained analysis text after segmentation
- name-correction decisions that may help normalize player references

Return strict JSON with exactly one key: `recommendations`.

`recommendations` must be an array of objects. Each object must include:
- `player_name`: the canonical player name when supported by NAME_CORRECTIONS; otherwise the name as spoken
- `stance`: an integer answering "should an FPL manager bring this player into their team now?", covering the next deadline and the next few gameweeks:
    5 = top priority: a must-have, the speaker's captain or triple-captain pick, or the clear best option in a position or price bracket
    4 = recommended: a good buy or strong option to bring in now or for an upcoming fixture run
    3 = neutral: fine to keep but not worth buying, a close call, or a genuinely mixed view
    2 = not recommended: do not buy; owners could move him on
    1 = sell or avoid: get rid now, or do not go near
- `sentiment`: a faithful explanation of the speaker's full opinion and reasoning; it may be several sentences when the nuance needs it
- `confidence`: an integer from 1 to 5 measuring confidence that you interpreted the transcript correctly, not confidence in the player's future performance
- `supporting_quote`: a direct, verbatim quote that identifies the player and best supports the stance; include multiple consecutive sentences when the player's name and the opinion are separated

Rules:
1. Return at most one record per player: the speaker's overall view for the next deadline onwards, considering everything said about the player across the transcript. If the speaker changes their mind, use the final view. Put any other relevant points in `sentiment`.
2. Rate the advice, not the speaker's ownership. "He's an easy hold but I wouldn't buy him" is 3. Keeping a player only because selling costs a transfer is 3.
3. Past performance alone is not a recommendation. Points scored last gameweek, hauls, blanks, captain results, or calling a past pick a good or bad decision ("good wildcard pick", "he let me down") produce no record unless the speaker also gives a view on the player from now on.
4. When a recommendation depends on something unknown, such as fitness or minutes, rate the stated recommendation, but no higher than 4, and note the condition in `sentiment`.
5. A long-term target ("I want him for the run from gameweek 10") is a recommendation: usually 4.
6. When two speakers disagree, use their net view and describe the disagreement in `sentiment`.
7. Do not create a record from a name-only mention, a fixture list, or neutral factual description with no evaluative opinion. Capture negative opinions as carefully as positive ones.

Use NAME_CORRECTIONS only to normalize `player_name`. Never apply corrections to `supporting_quote`, which must remain faithful to ANALYSIS_TEXT. Do not invent players, opinions, or quotes.

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


def _name_key(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in decomposed if character.isalnum())


def _canonical_name_map(source_name_corrections: list[dict[str, Any]]) -> dict[str, str]:
    candidates: dict[str, set[str]] = {}
    for correction in source_name_corrections:
        raw_name_value = correction.get("raw_name")
        selected_name_value = correction.get("selected_player_name")
        raw_name = str(raw_name_value).strip() if raw_name_value is not None else ""
        selected_name = str(selected_name_value).strip() if selected_name_value is not None else ""
        if not raw_name or not selected_name:
            continue
        candidates.setdefault(_name_key(raw_name), set()).add(selected_name)
        candidates.setdefault(_name_key(selected_name), set()).add(selected_name)
    return {
        raw_name: next(iter(selected_names))
        for raw_name, selected_names in candidates.items()
        if len(selected_names) == 1
    }


def _normalise_response(
    parsed: dict[str, Any],
    source_name_corrections: list[dict[str, Any]],
) -> dict[str, Any]:
    canonical_names = _canonical_name_map(source_name_corrections)
    supported_names = set(canonical_names.values())
    recommendations = []
    for item in parsed.get("recommendations", []):
        if not isinstance(item, dict):
            continue
        player_name = str(item.get("player_name", "")).strip()
        player_name = canonical_names.get(_name_key(player_name), player_name)
        sentiment = str(item.get("sentiment", "")).strip()
        supporting_quote = str(item.get("supporting_quote", "")).strip()
        raw_stance = item.get("stance")
        raw_confidence = item.get("confidence")
        if not isinstance(raw_stance, (int, str)) or not isinstance(raw_confidence, (int, str)):
            continue
        try:
            stance = int(raw_stance)
            confidence = int(raw_confidence)
        except (TypeError, ValueError):
            continue
        if player_name not in supported_names or not sentiment or not supporting_quote:
            continue
        if stance not in range(1, 6) or confidence not in range(1, 6):
            continue
        recommendations.append(
            {
                "player_name": player_name,
                "stance": stance,
                "sentiment": sentiment,
                "confidence": confidence,
                "supporting_quote": supporting_quote,
            }
        )
    return {"recommendations": recommendations}


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
        return _normalise_response(cached, source_name_corrections)

    response = client.chat.completions.create(
        model=model,
        response_format=EXTRACTION_RESPONSE_FORMAT,
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
        **DETERMINISTIC,
    )
    parsed = json.loads(response.choices[0].message.content or "{}")
    normalized = _normalise_response(
        parsed if isinstance(parsed, dict) else {},
        source_name_corrections,
    )
    _write_cache(cache_dir, key, normalized)
    return normalized


def build_transcript_extractions_output(
    *,
    sources: list[str],
    recommendations: list[dict[str, Any]],
) -> dict[str, Any]:
    by_source: dict[str, dict[str, Any]] = {
        source: {
            "source": source,
            "recommendations": [],
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
            },
        )
        entry["recommendations"].append(recommendation)

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
        },
        "recommendations": recommendations,
        "source_records": source_records,
    }


def write_transcript_extractions_output(
    output_path: Path,
    *,
    sources: list[str],
    recommendations: list[dict[str, Any]],
) -> dict[str, Any]:
    output = build_transcript_extractions_output(
        sources=sources,
        recommendations=recommendations,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
