from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from fpl_hq.consensus.llm import DETERMINISTIC

FILTER_PROMPT_VERSION = "v2"
MAX_CONCURRENT_SELECTION_FILTERS = 6

FILTER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "transcript_section_filter",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "section_index": {"type": "integer", "minimum": 0},
                            "retain": {"type": "boolean"},
                            "reason": {"type": "string"},
                        },
                        "required": ["section_index", "retain", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["sections"],
            "additionalProperties": False,
        },
    },
}

FILTER_PROMPT_TEMPLATE = """You filter Fantasy Premier League (FPL) transcript sections before player-recommendation extraction.

Each input item is one section from a previous segmentation pass. Decide only whether the section contains advice about upcoming decisions: from the next deadline onwards.

Keep a section when it contains forward-looking advice:
- buy, sell, hold, start, bench, captain, transfer, wildcard, chip, punt, or team selection for the next deadline or later
- an assessment of a player's or team's prospects, fixtures, minutes, or role from now on
- price-bracket or position comparisons, tier lists, or shortlists of options

Drop a section when it only reviews what already happened:
- the previous gameweek's points, hauls, blanks, captain results, rank or green arrows
- praise or criticism of past picks ("good wildcard decision", "he let me down") with no decision for the next deadline
- hindsight, banter, narrative filler or historical context

A previous-gameweek review stays dropped even when it praises or criticises players. Keep a mixed section only when it contains a clear decision or assessment for the next deadline or later.

Return strict JSON with exactly one key: `sections`.

Each item in `sections` must include:
- `section_index`: zero-based index of the input section
- `retain`: boolean
- `reason`: a short explanation

Do not rewrite section text. Do not invent content. Preserve the input order exactly.

SOURCE:
{source}

ANALYSIS_SECTIONS:
{analysis_sections}
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


def _split_analysis_sections(analysis_text: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for block in re.split(r"\n\n(?=\[SECTION: )", analysis_text.strip()):
        block = block.strip()
        if not block:
            continue
        heading, separator, body = block.partition("\n")
        heading = heading.strip()
        body = body.strip() if separator else ""
        if not heading.startswith("[SECTION:") or not body:
            continue
        sections.append(
            {
                "heading": heading,
                "text": body,
                "block": f"{heading}\n{body}",
            }
        )
    return sections


def _normalise_response(parsed: dict[str, Any], section_count: int) -> list[dict[str, Any]]:
    decisions = [
        {
            "section_index": index,
            "retain": True,
            "reason": "Defaulted to keep because no valid decision was returned.",
        }
        for index in range(section_count)
    ]
    if section_count <= 0:
        return decisions

    for item in parsed.get("sections", []):
        if not isinstance(item, dict):
            continue
        raw_section_index = item.get("section_index")
        if not isinstance(raw_section_index, (int, str)):
            continue
        try:
            section_index = int(raw_section_index)
        except (TypeError, ValueError):
            continue
        if section_index < 0 or section_index >= section_count:
            continue
        retain = bool(item.get("retain"))
        reason = str(item.get("reason", "")).strip()
        if not reason:
            reason = "Retain because the section is actionable." if retain else "Drop because the section is retrospective only."
        decisions[section_index] = {
            "section_index": section_index,
            "retain": retain,
            "reason": reason,
        }
    return decisions


def filter_transcript_analysis(
    client: Any,
    model: str,
    cache_dir: Path,
    source: str,
    analysis_text: str,
) -> dict[str, Any]:
    sections = _split_analysis_sections(analysis_text)
    if not sections:
        return {
            "source": source,
            "analysis_text": analysis_text,
            "filtered_analysis_text": "",
            "sections": [],
            "retained_sections": [],
            "removed_sections": [],
            "counts": {
                "sections": 0,
                "retained_sections": 0,
                "removed_sections": 0,
            },
        }

    prompt_sections = [
        {
            "section_index": index,
            "heading": section["heading"],
            "text": section["text"],
        }
        for index, section in enumerate(sections)
    ]
    prompt_sections_context = json.dumps(prompt_sections, ensure_ascii=False, indent=2)
    key = _cache_key(
        "selection_filter",
        FILTER_PROMPT_VERSION,
        model,
        source,
        analysis_text,
    )
    cached = _read_cache(cache_dir, key)
    if cached is not None:
        decisions = _normalise_response(cached if isinstance(cached, dict) else {}, len(sections))
    else:
        response = client.chat.completions.create(
            model=model,
            response_format=FILTER_RESPONSE_FORMAT,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {
                    "role": "user",
                    "content": FILTER_PROMPT_TEMPLATE.format(
                        source=source,
                        analysis_sections=prompt_sections_context,
                    ),
                },
            ],
            **DETERMINISTIC,
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
        decisions = _normalise_response(parsed if isinstance(parsed, dict) else {}, len(sections))
        _write_cache(cache_dir, key, {"sections": decisions})

    retained_sections = []
    removed_sections = []
    for section, decision in zip(sections, decisions):
        section_record = {
            "section_index": decision["section_index"],
            "heading": section["heading"],
            "text": section["text"],
            "retain": decision["retain"],
            "reason": decision["reason"],
        }
        if decision["retain"]:
            retained_sections.append(section_record)
        else:
            removed_sections.append(section_record)

    filtered_analysis_text = "\n\n".join(f"{section['heading']}\n{section['text']}" for section in retained_sections)
    return {
        "source": source,
        "analysis_text": analysis_text,
        "filtered_analysis_text": filtered_analysis_text,
        "sections": [
            {
                "section_index": section["section_index"],
                "heading": section["heading"],
                "text": section["text"],
                "retain": section["retain"],
                "reason": section["reason"],
            }
            for section in retained_sections + removed_sections
        ],
        "retained_sections": retained_sections,
        "removed_sections": removed_sections,
        "counts": {
            "sections": len(sections),
            "retained_sections": len(retained_sections),
            "removed_sections": len(removed_sections),
        },
    }


def build_transcript_selection_output(
    *,
    sources: list[str],
    selections: list[dict[str, Any]],
) -> dict[str, Any]:
    by_source: dict[str, dict[str, Any]] = {
        source: {
            "source": source,
            "analysis_text": "",
            "filtered_analysis_text": "",
            "sections": [],
            "retained_sections": [],
            "removed_sections": [],
        }
        for source in sources
    }

    for selection in selections:
        source = str(selection.get("source", "")).strip()
        if not source:
            continue
        entry = by_source.setdefault(
            source,
            {
                "source": source,
                "analysis_text": "",
                "filtered_analysis_text": "",
                "sections": [],
                "retained_sections": [],
                "removed_sections": [],
            },
        )
        entry["analysis_text"] = selection.get("analysis_text", "")
        entry["filtered_analysis_text"] = selection.get("filtered_analysis_text", "")
        entry["sections"] = selection.get("sections", [])
        entry["retained_sections"] = selection.get("retained_sections", [])
        entry["removed_sections"] = selection.get("removed_sections", [])

    ordered_sources = []
    for source in sources + sorted(by_source):
        if source not in ordered_sources:
            ordered_sources.append(source)

    source_records = [by_source[source] for source in ordered_sources if source in by_source]
    return {
        "sources": ordered_sources,
        "counts": {
            "sources": len(ordered_sources),
            "sections": sum(len(selection.get("sections", [])) for selection in selections),
            "retained_sections": sum(len(selection.get("retained_sections", [])) for selection in selections),
            "removed_sections": sum(len(selection.get("removed_sections", [])) for selection in selections),
        },
        "selections": selections,
        "source_records": source_records,
    }


def write_transcript_selection_output(
    output_path: Path,
    *,
    sources: list[str],
    selections: list[dict[str, Any]],
) -> dict[str, Any]:
    output = build_transcript_selection_output(
        sources=sources,
        selections=selections,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output