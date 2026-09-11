"""Extract structured FPL insights from filtered text sources.

This module batches cleaned article text and asks the LLM to return structured
recommendations, captaincy mentions, and themes.

Example:
    python notebooks/text_processor/2_extract_text.py --gameweek 3
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
EXTRACTION_PROMPT_VERSION = "v2"
DEFAULT_MAX_WORKERS = 4
DEFAULT_BATCH_SIZE = 4

load_dotenv(PROJECT_ROOT / ".env")

EXTRACTION_PROMPT_TEMPLATE = """You extract structured Fantasy Premier League (FPL) insights from filtered website text.

You are given a JSON array of website sources. Each item contains:
- `source`: a website source id
- `filtered_text`: article text that already had non-FPL filler removed

Return strict JSON only with exactly one key: `results`.

`results` must be an array with one item for every input source, in the same order. Each result item must contain exactly four keys:
- `source`
- `recommendations`
- `captaincy`
- `themes`

For each result item:

`recommendations` must be an array of objects. Each object must include:
- `player_name`: the player being discussed
- `stance`: one of `buy`, `consider`, `hold`, `avoid`, `sell`
- `sentiment`: one of `positive`, `neutral`, `negative`, `mixed`
- `recommendation_strength`: an integer from 1 to 5
- `confidence`: an integer from 1 to 5
- `rationale`: a short explanation grounded in the article text
- `supporting_quote`: a direct quote from the article text
- `reasons`: an array of objects, where each object has `category`, `sentiment`, and `text`
- `risks`: an array of objects, where each object has `category`, `sentiment`, and `text`
- `time_horizon`: a short label such as `immediate`, `short-term`, `medium-term`, or `long-term`
- `timestamp`: use an empty string unless the source text includes a clear timestamp-like marker

`captaincy` must be an array of objects. Each object must include:
- `player_name`
- `mention_type`: one of `triple_captain`, `captain`, `vice_captain`, `avoid_captain`
- `confidence`: an integer from 1 to 5
- `expert_percentage`: an optional number from 0 to 100
- `rationale`
- `supporting_quote`
- `timestamp`: use an empty string unless the source text includes a clear timestamp-like marker

`themes` must be an array of short strings.

Rules:
- Keep supporting quotes faithful to the source.
- Do not invent players, quotes, injuries, or stats.
- If the article discusses multiple analysts or draft paths, extract each distinct player recommendation separately.
- If a reason or risk is implied but not explicit enough to quote, keep it short and grounded in the article.
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


def _normalize_reason_items(items: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if not isinstance(items, list):
        return normalized
    for item in items:
        if isinstance(item, dict):
            category = str(item.get("category", "other")).strip() or "other"
            sentiment = str(item.get("sentiment", "neutral")).strip() or "neutral"
            text = str(item.get("text", "")).strip()
            if text:
                normalized.append({"category": category, "sentiment": sentiment, "text": text})
            continue
        text = str(item).strip()
        if text:
            normalized.append({"category": "other", "sentiment": "neutral", "text": text})
    return normalized


def _normalize_response(parsed: dict[str, Any], source_name: str) -> dict[str, Any]:
    recommendations: list[dict[str, Any]] = []
    for item in parsed.get("recommendations", []):
        if not isinstance(item, dict):
            continue
        player_name = str(item.get("player_name", "")).strip()
        stance = str(item.get("stance", "")).strip()
        supporting_quote = str(item.get("supporting_quote", "")).strip()
        if not player_name or not stance or not supporting_quote:
            continue
        recommendations.append(
            {
                "source": source_name,
                "player_name": player_name,
                "stance": stance,
                "sentiment": str(item.get("sentiment", "neutral")).strip() or "neutral",
                "recommendation_strength": int(item.get("recommendation_strength", 3) or 3),
                "confidence": int(item.get("confidence", 3) or 3),
                "rationale": str(item.get("rationale", "")).strip(),
                "supporting_quote": supporting_quote,
                "reasons": _normalize_reason_items(item.get("reasons", [])),
                "risks": _normalize_reason_items(item.get("risks", [])),
                "time_horizon": str(item.get("time_horizon", "short-term")).strip() or "short-term",
                "timestamp": str(item.get("timestamp", "")).strip(),
            }
        )

    captaincy: list[dict[str, Any]] = []
    for item in parsed.get("captaincy", []):
        if not isinstance(item, dict):
            continue
        player_name = str(item.get("player_name", "")).strip()
        mention_type = str(item.get("mention_type", "")).strip()
        supporting_quote = str(item.get("supporting_quote", "")).strip()
        if not player_name or not mention_type or not supporting_quote:
            continue
        captaincy.append(
            {
                "source": source_name,
                "player_name": player_name,
                "mention_type": mention_type,
                "confidence": int(item.get("confidence", 3) or 3),
                "expert_percentage": item.get("expert_percentage"),
                "rationale": str(item.get("rationale", "")).strip(),
                "supporting_quote": supporting_quote,
                "timestamp": str(item.get("timestamp", "")).strip(),
            }
        )

    themes = [str(item).strip() for item in parsed.get("themes", []) if str(item).strip()]
    return {
        "recommendations": recommendations,
        "captaincy": captaincy,
        "themes": themes,
    }


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
        response_format={"type": "json_object"},
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
    combined_captaincy: list[dict[str, Any]] = []
    combined_themes: list[dict[str, str]] = []
    for source, result in per_source_results.items():
        combined_recommendations.extend(result.get("recommendations", []))
        combined_captaincy.extend(result.get("captaincy", []))
        combined_themes.extend({"source": source, "theme": theme} for theme in result.get("themes", []))
    return {
        "gameweek": gameweek,
        "sources": sorted(per_source_results),
        "recommendations": combined_recommendations,
        "captaincy": combined_captaincy,
        "themes": combined_themes,
        "per_source": per_source_results,
    }