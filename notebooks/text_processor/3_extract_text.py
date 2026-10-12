"""Extract structured FPL insights from filtered text sources.

This module batches cleaned article text and asks the LLM to return structured
player recommendations in the same core shape as the transcript extractor.

Example:
    python notebooks/text_processor/3_extract_text.py --gameweek 3
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "gpt-4.1-mini"
EXTRACTION_PROMPT_VERSION = "v3"
DEFAULT_MAX_WORKERS = 4
DEFAULT_BATCH_SIZE = 4

EXTRACTION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "website_player_recommendations",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
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
                                    "required": ["player_name", "stance", "sentiment", "confidence", "supporting_quote"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["source", "recommendations"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["results"],
            "additionalProperties": False,
        },
    },
}

load_dotenv(PROJECT_ROOT / ".env")

EXTRACTION_PROMPT_TEMPLATE = """You extract structured Fantasy Premier League (FPL) player recommendations from filtered website text.

You are given a JSON array of website sources. Each item contains:
- `source`: a website source id
- `filtered_text`: article text that already had non-FPL filler removed

Return strict JSON only with exactly one key: `results`.

`results` must be an array with one item for every input source, in the same order. Each result item must contain exactly two keys:
- `source`
- `recommendations`

For each result item:

`recommendations` must be an array of objects. Each object must include:
- `player_name`: the player being discussed
- `stance`: an integer from 1 to 5 using the same scale as the transcript extractor, where 1 is strongest avoid/sell and 5 is strongest buy
- `sentiment`: a short phrase that captures the recommendation's nuance
- `confidence`: an integer from 1 to 5
- `supporting_quote`: a direct quote from the article text

Rules:
- Keep supporting quotes faithful to the source.
- Do not invent players, quotes, injuries, or stats.
- If a player is mentioned but there is no actionable recommendation, omit them.

INPUT_SOURCES:
{sources_json}
"""


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def _cache_key(*parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:24]


def _read_cache(cache_dir: Path, key: str) -> Any | None:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_cache(cache_dir: Path, key: str, payload: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _normalize_response(parsed: dict[str, Any], source_name: str) -> dict[str, Any]:
    recommendations: list[dict[str, Any]] = []
    for item in parsed.get("recommendations", []):
        if not isinstance(item, dict):
            continue
        player_name = str(item.get("player_name", "")).strip()
        raw_stance = item.get("stance")
        supporting_quote = str(item.get("supporting_quote", "")).strip()
        if not player_name or raw_stance is None or not supporting_quote:
            continue
        try:
            stance = int(raw_stance)
        except (TypeError, ValueError):
            continue
        if stance < 1 or stance > 5:
            continue
        recommendations.append(
            {
                "source": source_name,
                "player_name": player_name,
                "stance": stance,
                "sentiment": str(item.get("sentiment", "")).strip(),
                "confidence": int(item.get("confidence", 3) or 3),
                "supporting_quote": supporting_quote,
            }
        )
    return {"recommendations": recommendations}


_thread_local = threading.local()


def _get_client() -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenAI()
        _thread_local.client = client
    return client


def extract_from_text_batch(
    model: str,
    cache_dir: Path,
    batch: list[tuple[str, str]],
    refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    payload = [{"source": source, "filtered_text": analysis_text} for source, analysis_text in batch]
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    key = _cache_key("text-extraction-batch", EXTRACTION_PROMPT_VERSION, model, payload_json)
    if not refresh:
        cached = _read_cache(cache_dir, key)
        if cached is not None:
            return {
                str(source): _normalize_response(result if isinstance(result, dict) else {}, str(source))
                for source, result in cached.get("results_by_source", {}).items()
            }

    response = _get_client().chat.completions.create(
        model=model,
        response_format=EXTRACTION_RESPONSE_FORMAT,
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {
                "role": "user",
                "content": EXTRACTION_PROMPT_TEMPLATE.format(sources_json=payload_json),
            },
        ],
        temperature=0,
    )
    parsed = json.loads(response.choices[0].message.content or "{}")
    raw_results = parsed.get("results", []) if isinstance(parsed, dict) else []

    normalized_by_source: dict[str, dict[str, Any]] = {}
    expected_sources = [source for source, _ in batch]
    raw_results_by_source = {
        str(item.get("source", "")).strip(): item
        for item in raw_results
        if isinstance(item, dict) and str(item.get("source", "")).strip()
    }
    for source in expected_sources:
        normalized_by_source[source] = _normalize_response(raw_results_by_source.get(source, {}), source)

    _write_cache(
        cache_dir,
        key,
        {
            "sources": expected_sources,
            "results_by_source": normalized_by_source,
        },
    )
    return normalized_by_source


def process_filtered_text_sources(
    filtered_text_payload: dict[str, str],
    cache_dir: Path,
    model: str = DEFAULT_MODEL,
    max_workers: int = DEFAULT_MAX_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    items = [(source, text.strip()) for source, text in sorted(filtered_text_payload.items()) if text.strip()]
    if not items:
        return {}

    results: dict[str, dict[str, Any]] = {}
    batches = _chunked(items, batch_size)
    worker_count = min(max_workers, len(batches))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(extract_from_text_batch, model, cache_dir, batch, refresh): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(future_map):
            results.update(future.result())
    return dict(sorted(results.items()))


def build_combined_output(gameweek: int, per_source_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    combined_recommendations: list[dict[str, Any]] = []
    for source, result in per_source_results.items():
        combined_recommendations.extend(result.get("recommendations", []))
    return {
        "gameweek": gameweek,
        "sources": sorted(per_source_results),
        "counts": {
            "sources": len(per_source_results),
            "recommendations": len(combined_recommendations),
        },
        "recommendations": combined_recommendations,
        "source_records": [
            {
                "source": source,
                "recommendations": result.get("recommendations", []),
            }
            for source, result in per_source_results.items()
        ],
        "per_source": per_source_results,
    }