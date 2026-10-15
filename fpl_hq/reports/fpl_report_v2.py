#!/usr/bin/env python3
"""Assemble an FPL consideration and avoidance report for one upcoming gameweek.

This script combines the FPL data aggregate with qualitative analysis output from
transcripts and filtered website text into a markdown report. It does not do any
NLP itself.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
from pathlib import Path
from typing import Any

from fpl_hq.paths import ENV_FILE, PROJECT_ROOT, REPORTS_DIR, gw_dir
from fpl_hq.consensus.llm import MODEL, DETERMINISTIC
REPORTS_DIR = REPORTS_DIR


PROMPT_VERSION = "v4"
MIN_MENTIONS_FOR_FULL_CONFIDENCE = 3
STANCE_VALUES = {
    "buy": 1.0,
    "consider": 0.6,
    "hold": 0.2,
    "avoid": -0.8,
    "sell": -1.0,
}
NUMERIC_STANCE_VALUES = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}
CAPTAINCY_WEIGHTS = {
    "triple_captain": 1.0,
    "captain": 0.8,
    "vice_captain": 0.4,
    "avoid_captain": -1.0,
}

# How many summary batches to synthesize concurrently.
MAX_CONCURRENT_SUMMARY_BATCHES = 6
TRANSCRIPT_PIPELINE_MODEL = MODEL


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def clamp(value: float, lower: float = 0, upper: float = 100) -> float:
    return max(lower, min(value, upper))


def score_player(player: dict[str, Any]) -> tuple[float, list[str]]:
    form = player["recent_form"]["last_3_gameweeks"]
    appearances = form["appearances"]
    totals = form["totals"]
    per_90 = form["per_90"]
    fixture = player["upcoming_fixture"]
    fixture_run = player.get("fixture_run")

    points_per_appearance = totals["total_points"] / appearances if appearances else 0
    minutes_per_appearance = totals["minutes"] / appearances if appearances else 0
    xgi_per_90 = per_90["expected_goal_involvements"]
    fixture_difficulty = fixture["difficulty"] if fixture else 5
    opponent_goals_against = fixture["opponent_form"]["goals_against"] if fixture else 0
    gameweeks_included = max(1, len(form["gameweeks_included"]))

    run_difficulty = (
        fixture_run["average_difficulty"]
        if fixture_run and fixture_run.get("average_difficulty") is not None
        else fixture_difficulty
    )

    score = (
        clamp(points_per_appearance * 4, 0, 25)
        + clamp(xgi_per_90 * 20, 0, 20)
        + clamp(minutes_per_appearance / 4, 0, 20)
        + (6 - fixture_difficulty) * 4
        + (6 - run_difficulty) * 3
        + clamp(opponent_goals_against / gameweeks_included * 3, 0, 10)
    )

    chance = player["chance_of_playing_next_round"]
    if chance is not None:
        score += clamp(float(chance) / 10, 0, 10)

    evidence = [
        f"{points_per_appearance:.1f} points per appearance across GW{form['gameweeks_included'][0] if form['gameweeks_included'] else '-'}-GW{form['gameweeks_included'][-1] if form['gameweeks_included'] else '-'}",
        f"{xgi_per_90:.2f} xGI per 90",
        f"{fixture_difficulty}/5 fixture difficulty against {fixture['opponent'] if fixture else 'no listed opponent'}",
    ]
    if fixture_run and fixture_run.get("num_fixtures"):
        evidence.append(
            f"{run_difficulty:.1f}/5 average difficulty over next {fixture_run['num_fixtures']} fixtures"
        )
    return round(clamp(score), 1), evidence


def load_transcript_analysis(path: Path) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], list[tuple[str, str]], list[str]]:
    data = load_json(path)
    sentiment = {int(player_id): entry for player_id, entry in data.get("sentiment", {}).items()}
    captaincy = {int(player_id): entry for player_id, entry in data.get("captaincy", {}).items()}
    themes = [(text, source) for text, source in data.get("themes", [])]
    sources = list(data.get("sources", []))
    return sentiment, captaincy, themes, sources


def load_transcript_extractions(
    path: Path,
    players: dict[str, dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], list[tuple[str, str]], list[str]]:
    data = load_json(path)
    recommendations = [item for item in data.get("recommendations", []) if isinstance(item, dict)]
    captaincy_mentions = [item for item in data.get("captaincy", []) if isinstance(item, dict)]

    raw_themes = data.get("themes", [])
    themes: list[tuple[str, str]] = []
    if isinstance(raw_themes, list):
        for item in raw_themes:
            if isinstance(item, dict):
                theme = str(item.get("theme", "")).strip()
                source = str(item.get("source", "")).strip()
                if theme:
                    themes.append((theme, source))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                theme = str(item[0]).strip()
                source = str(item[1]).strip()
                if theme:
                    themes.append((theme, source))

    sentiment, captaincy, aggregated_themes, inferred_sources = aggregate_source_extractions(
        recommendations,
        captaincy_mentions,
        themes,
        players,
    )
    sources = _source_list(data.get("sources")) or inferred_sources
    return sentiment, captaincy, aggregated_themes, sources


class _CacheOnlyClient:
    class _Chat:
        class _Completions:
            @staticmethod
            def create(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("Transcript cache miss during report generation.")

        completions = _Completions()

    chat = _Chat()


def load_transcript_analysis_from_cache(
    gameweek: int,
    players: dict[str, dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], list[tuple[str, str]], list[str]]:
    transcript_dir = gw_dir(gameweek) / "transcripts"
    cache_root = gw_dir(gameweek) / "transcript_cache"
    segmentation_cache_dir = cache_root / "segmentation_cache"
    name_resolution_cache_dir = cache_root / "name_resolution_cache"
    extraction_cache_dir = cache_root / "extraction_cache"

    if not transcript_dir.exists() or not extraction_cache_dir.exists():
        return {}, {}, [], []

    segmentation_module = importlib.import_module("fpl_hq.consensus.transcripts.1_segmentation")
    name_resolution_module = importlib.import_module("fpl_hq.consensus.transcripts.2_name_resolution")
    extraction_module = importlib.import_module("fpl_hq.consensus.transcripts.4_extraction")

    collect_transcripts = segmentation_module.collect_transcripts
    build_analysis_text = segmentation_module.build_analysis_text
    segment_transcript = segmentation_module.segment_transcript
    build_name_resolution_helpers = name_resolution_module.build_name_resolution_helpers
    load_name_aliases = name_resolution_module.load_name_aliases
    resolve_transcript_names = build_name_resolution_helpers(
        name_aliases=load_name_aliases(),
        canonical_player_names={player["name"]: int(player_id) for player_id, player in players.items()},
        canonical_player_id_to_name={int(player_id): player["name"] for player_id, player in players.items()},
        player_metadata={
            int(player_id): {
                "team": str(player.get("team", "")),
                "position": int(player.get("position", 0) or 0),
            }
            for player_id, player in players.items()
        },
    ).resolve_transcript_names
    extract_from_transcript = extraction_module.extract_from_transcript

    cache_client = _CacheOnlyClient()
    recommendations: list[dict[str, Any]] = []
    captaincy_mentions: list[dict[str, Any]] = []
    themes: list[tuple[str, str]] = []
    sources: list[str] = []

    for transcript in collect_transcripts(transcript_dir):
        source = str(transcript.get("source", "")).strip()
        text = str(transcript.get("text", ""))
        if not source or not text:
            continue
        try:
            segments = segment_transcript(
                cache_client,
                TRANSCRIPT_PIPELINE_MODEL,
                segmentation_cache_dir,
                text,
                gameweek,
            )
            analysis_text = build_analysis_text(segments)
            if not analysis_text.strip():
                continue
            source_name_corrections = resolve_transcript_names(
                cache_client,
                TRANSCRIPT_PIPELINE_MODEL,
                name_resolution_cache_dir,
                source,
                analysis_text,
            )
            parsed = extract_from_transcript(
                cache_client,
                TRANSCRIPT_PIPELINE_MODEL,
                extraction_cache_dir,
                source,
                analysis_text,
                source_name_corrections,
            )
        except RuntimeError:
            continue

        has_source_evidence = False
        for recommendation in parsed.get("recommendations", []):
            if not isinstance(recommendation, dict):
                continue
            recommendations.append({**recommendation, "source": source})
            has_source_evidence = True
        for mention in parsed.get("captaincy", []):
            if not isinstance(mention, dict):
                continue
            captaincy_mentions.append({**mention, "source": source})
            has_source_evidence = True
        for theme in parsed.get("themes", []):
            text_value = str(theme).strip()
            if not text_value:
                continue
            themes.append((text_value, source))
            has_source_evidence = True
        if has_source_evidence and source not in sources:
            sources.append(source)

    if not recommendations and not captaincy_mentions and not themes:
        return {}, {}, [], []

    return aggregate_source_extractions(recommendations, captaincy_mentions, themes, players)


def _cache_key(*parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:24]


def _read_cache(cache_dir: Path, key: str) -> Any | None:
    path = cache_dir / f"{key}.json"
    return load_json(path) if path.exists() else None


def _write_cache(cache_dir: Path, key: str, data: Any) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


SUMMARY_PROMPT_TEMPLATE = """You write concise decision-support summaries for Fantasy Premier League (FPL) players, combining each player's statistical data profile with qualitative source opinions collected this gameweek.

For each player in the JSON array below, return:
- summary: 2-3 plain-English sentences weighing data_evidence and analyst_evidence together. If analyst_evidence is empty, say so explicitly and summarise data_evidence only. Do not invent facts beyond what is given. Where useful, cite analyst evidence using its bracketed source letter already present in the text (e.g. "per [B]").
- verdict: a short 3-6 word label capturing the overall takeaway (e.g. "Strong buy consensus", "Split opinion, monitor", "Data-only, no coverage", "Injury concern flagged").

Return strict JSON: {{"players": [{{"id": <int>, "summary": <string>, "verdict": <string>}}, ...]}} covering every player id below, in the same order, with no extra players.

PLAYERS:
{players_json}"""


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def synthesize_player_summaries_batch(
    client: Any, model: str, cache_dir: Path, batch: list[dict[str, Any]]
) -> dict[int, dict[str, str]]:
    payload = json.dumps(batch, sort_keys=True)
    key = _cache_key("summary", PROMPT_VERSION, model, payload)
    cached = _read_cache(cache_dir, key)
    if cached is not None:
        return {int(player_id): value for player_id, value in cached.items()}

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": SUMMARY_PROMPT_TEMPLATE.format(players_json=payload)},
        ],
        **DETERMINISTIC,
    )
    parsed = json.loads(response.choices[0].message.content or "{}")
    result: dict[int, dict[str, str]] = {}
    for item in parsed.get("players", []):
        try:
            player_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        result[player_id] = {
            "summary": str(item.get("summary", "")).strip(),
            "verdict": str(item.get("verdict", "")).strip(),
        }
    _write_cache(cache_dir, key, {str(player_id): value for player_id, value in result.items()})
    return result


def generate_player_summaries(
    client: Any, model: str, cache_dir: Path, items: list[dict[str, Any]], batch_size: int = 12
) -> dict[int, dict[str, str]]:
    batches = _chunked(items, batch_size)
    if not batches:
        return {}

    results: dict[int, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_SUMMARY_BATCHES, len(batches))) as executor:
        futures = [
            executor.submit(synthesize_player_summaries_batch, client, model, cache_dir, batch)
            for batch in batches
        ]
        for future in as_completed(futures):
            results.update(future.result())
    return results


def latest_public_squad(gameweek: int) -> tuple[int, list[int]] | None:
    candidates = []
    for picks_path in PROJECT_ROOT.glob("gw/gw*/data/gw_*_my_team_picks.json"):
        match = re.search(r"gw_(\d+)_my_team_picks\.json$", picks_path.name)
        if match and int(match.group(1)) < gameweek:
            candidates.append((int(match.group(1)), picks_path))
    if not candidates:
        return None

    source_gameweek, picks_path = max(candidates)
    picks_data = load_json(picks_path)
    return source_gameweek, [pick["element"] for pick in picks_data["picks"]]


def markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        output.append("| " + " | ".join(value.replace("|", "/") for value in row) + " |")
    return "\n".join(output)


def html_table(rows: list[list[str]], headers: list[str]) -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>")
    body = "".join(body_rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


POSITION_LABELS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def format_fixture(player: dict[str, Any]) -> str:
    fixture = player["upcoming_fixture"]
    if not fixture:
        return "-"
    venue = "H" if fixture["is_home"] else "A"
    return f"{fixture['opponent']} ({venue}, FDR {fixture['difficulty']})"


def format_fixture_run(player: dict[str, Any]) -> str:
    fixture_run = player.get("fixture_run")
    if not fixture_run or not fixture_run.get("num_fixtures"):
        return "-"
    return f"{fixture_run['average_difficulty']:.1f} avg ({fixture_run['num_fixtures']} fixtures)"


def _source_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def load_name_resolution_helpers(players: dict[str, dict[str, Any]]) -> Any:
    module = importlib.import_module("fpl_hq.consensus.transcripts.2_name_resolution")
    load_name_aliases = module.load_name_aliases
    build_name_resolution_helpers = module.build_name_resolution_helpers

    player_names = {player["name"]: int(player_id) for player_id, player in players.items()}
    canonical_player_names = dict(player_names)
    canonical_player_id_to_name = {player_id: name for name, player_id in player_names.items()}
    player_metadata = {
        int(player_id): {
            "team": str(player.get("team", "")),
            "position": int(player.get("position", 0) or 0),
        }
        for player_id, player in players.items()
    }
    return build_name_resolution_helpers(
        name_aliases=load_name_aliases(),
        canonical_player_names=canonical_player_names,
        canonical_player_id_to_name=canonical_player_id_to_name,
        player_metadata=player_metadata,
    )


def aggregate_source_extractions(
    recommendations: list[dict[str, Any]],
    captaincy_mentions: list[dict[str, Any]],
    themes: list[tuple[str, str]],
    players: dict[str, dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], list[tuple[str, str]], list[str]]:
    helpers = load_name_resolution_helpers(players)
    resolve_supported_player_id = helpers.resolve_supported_player_id

    player_names = {player["name"]: int(player_id) for player_id, player in players.items()}
    player_id_to_name = {int(player_id): player["name"] for player_id, player in players.items()}

    sentiment: dict[int, dict[str, Any]] = {}
    for recommendation in recommendations:
        player_id = resolve_supported_player_id(recommendation, player_names)
        if player_id is None:
            continue
        raw_stance = recommendation.get("stance")
        try:
            numeric_stance = int(raw_stance)
        except (TypeError, ValueError):
            numeric_stance = None
        if numeric_stance in NUMERIC_STANCE_VALUES:
            stance = numeric_stance
            stance_value = NUMERIC_STANCE_VALUES[numeric_stance]
        else:
            stance = str(raw_stance or "").lower()
            if stance not in STANCE_VALUES:
                continue
            stance_value = STANCE_VALUES[stance]
        confidence = clamp(float(recommendation.get("confidence", 1)), 1, 5)
        entry = sentiment.setdefault(
            player_id,
            {
                "player_name": player_id_to_name[player_id],
                "weighted_stances": [],
                "evidence": [],
            },
        )
        entry["weighted_stances"].append(stance_value * confidence / 5)
        entry["evidence"].append({
            "stance": stance,
            "sentiment": str(recommendation.get("sentiment", "")),
            "supporting_quote": str(recommendation.get("supporting_quote", "")),
            "source": str(recommendation.get("source", "")),
            "timestamp": recommendation.get("timestamp"),
            "confidence": confidence,
        })

    for entry in sentiment.values():
        weighted_stances = entry["weighted_stances"]
        mention_count = len(weighted_stances)
        average_stance = sum(weighted_stances) / mention_count
        decay = min(1.0, mention_count / MIN_MENTIONS_FOR_FULL_CONFIDENCE)
        entry["score"] = round(clamp(50 + average_stance * 50 * decay), 1)
        entry["mention_count"] = mention_count
        supportive = sum(1 for value in weighted_stances if value > 0)
        bearish = sum(1 for value in weighted_stances if value < 0)
        entry["disagreement"] = f"{supportive} supportive vs {bearish} bearish" if supportive and bearish else None
        del entry["weighted_stances"]

    captaincy: dict[int, dict[str, Any]] = {}
    for mention in captaincy_mentions:
        player_id = resolve_supported_player_id(mention, player_names)
        if player_id is None:
            continue
        mention_type = str(mention.get("mention_type", "")).lower()
        if mention_type not in CAPTAINCY_WEIGHTS:
            continue
        confidence = clamp(float(mention.get("confidence", 1)), 1, 5)
        entry = captaincy.setdefault(
            player_id,
            {
                "player_name": player_id_to_name[player_id],
                "weighted_mentions": [],
                "evidence": [],
            },
        )
        entry["weighted_mentions"].append(CAPTAINCY_WEIGHTS[mention_type] * confidence / 5)
        entry["evidence"].append({
            "mention_type": mention_type,
            "expert_percentage": mention.get("expert_percentage"),
            "rationale": str(mention.get("rationale", "")),
            "supporting_quote": str(mention.get("supporting_quote", "")),
            "source": str(mention.get("source", "")),
            "timestamp": mention.get("timestamp"),
            "confidence": confidence,
        })

    for entry in captaincy.values():
        weighted_mentions = entry["weighted_mentions"]
        mention_count = len(weighted_mentions)
        average_weight = sum(weighted_mentions) / mention_count if mention_count else 0
        entry["score"] = round(clamp(50 + average_weight * 50), 1)
        entry["mention_count"] = mention_count
        del entry["weighted_mentions"]

    source_names = sorted({str(item.get("source", "")).strip() for item in recommendations + captaincy_mentions if str(item.get("source", "")).strip()})
    return sentiment, captaincy, themes, source_names


def load_text_analysis(
    path: Path,
    players: dict[str, dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], list[tuple[str, str]], list[str]]:
    data = load_json(path)
    recommendations = [item for item in data.get("recommendations", []) if isinstance(item, dict)]
    captaincy_mentions = [item for item in data.get("captaincy", []) if isinstance(item, dict)]

    raw_themes = data.get("themes", [])
    themes: list[tuple[str, str]] = []
    if isinstance(raw_themes, list):
        for item in raw_themes:
            if isinstance(item, dict):
                theme = str(item.get("theme", "")).strip()
                source = str(item.get("source", "")).strip()
                if theme:
                    themes.append((theme, source))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                theme = str(item[0]).strip()
                source = str(item[1]).strip()
                if theme:
                    themes.append((theme, source))

    sentiment, captaincy, aggregated_themes, inferred_sources = aggregate_source_extractions(
        recommendations,
        captaincy_mentions,
        themes,
        players,
    )
    sources = _source_list(data.get("sources")) or inferred_sources
    return sentiment, captaincy, aggregated_themes, sources


def _sentiment_weight_total(entry: dict[str, Any]) -> tuple[float, int]:
    mention_count = int(entry.get("mention_count", len(entry.get("evidence", []))) or 0)
    if mention_count <= 0:
        return 0.0, 0
    decay = min(1.0, mention_count / MIN_MENTIONS_FOR_FULL_CONFIDENCE)
    if decay <= 0:
        return 0.0, mention_count
    score = float(entry.get("score", 50.0))
    average_stance = (score - 50.0) / (50.0 * decay)
    return average_stance * mention_count, mention_count


def merge_sentiment_maps(*maps: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for mapping in maps:
        for player_id, entry in mapping.items():
            total_weight, mention_count = _sentiment_weight_total(entry)
            merged_entry = merged.setdefault(
                player_id,
                {
                    "player_name": entry.get("player_name", ""),
                    "weighted_total": 0.0,
                    "mention_count": 0,
                    "evidence": [],
                },
            )
            merged_entry["weighted_total"] += total_weight
            merged_entry["mention_count"] += mention_count
            merged_entry["evidence"].extend(entry.get("evidence", []))

    result: dict[int, dict[str, Any]] = {}
    for player_id, entry in merged.items():
        mention_count = entry["mention_count"]
        if mention_count <= 0:
            continue
        average_stance = entry["weighted_total"] / mention_count
        decay = min(1.0, mention_count / MIN_MENTIONS_FOR_FULL_CONFIDENCE)
        evidence = entry["evidence"]
        supportive = sum(1 for item in evidence if str(item.get("stance", "")).lower() in {"buy", "consider", "hold"})
        bearish = sum(1 for item in evidence if str(item.get("stance", "")).lower() in {"avoid", "sell"})
        result[player_id] = {
            "player_name": entry.get("player_name", ""),
            "score": round(clamp(50 + average_stance * 50 * decay), 1),
            "mention_count": mention_count,
            "disagreement": f"{supportive} supportive vs {bearish} bearish" if supportive and bearish else None,
            "evidence": evidence,
        }
    return result


def _captaincy_weight_total(entry: dict[str, Any]) -> tuple[float, int]:
    evidence = entry.get("evidence", [])
    mention_count = int(entry.get("mention_count", len(evidence)) or 0)
    if mention_count <= 0:
        return 0.0, 0
    weights = []
    for item in evidence:
        mention_type = str(item.get("mention_type", "")).lower()
        if mention_type not in CAPTAINCY_WEIGHTS:
            continue
        confidence = clamp(float(item.get("confidence", 3)), 1, 5)
        weights.append(CAPTAINCY_WEIGHTS[mention_type] * confidence / 5)
    if weights:
        return sum(weights), len(weights)
    score = float(entry.get("score", 50.0))
    average_weight = (score - 50.0) / 50.0
    return average_weight * mention_count, mention_count


def merge_captaincy_maps(*maps: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for mapping in maps:
        for player_id, entry in mapping.items():
            total_weight, mention_count = _captaincy_weight_total(entry)
            merged_entry = merged.setdefault(
                player_id,
                {
                    "player_name": entry.get("player_name", ""),
                    "weighted_total": 0.0,
                    "mention_count": 0,
                    "evidence": [],
                },
            )
            merged_entry["weighted_total"] += total_weight
            merged_entry["mention_count"] += mention_count
            merged_entry["evidence"].extend(entry.get("evidence", []))

    result: dict[int, dict[str, Any]] = {}
    for player_id, entry in merged.items():
        mention_count = entry["mention_count"]
        if mention_count <= 0:
            continue
        average_weight = entry["weighted_total"] / mention_count
        result[player_id] = {
            "player_name": entry.get("player_name", ""),
            "score": round(clamp(50 + average_weight * 50), 1),
            "mention_count": mention_count,
            "evidence": entry["evidence"],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an FPL weekly recommendation report.")
    parser.add_argument("--gameweek", type=int, required=True, help="Upcoming gameweek number.")
    parser.add_argument("--model", default=MODEL, help="OpenAI model for summary synthesis.")
    parser.add_argument("--no-llm", action="store_true", help="Generate a data-only report without OpenAI.")
    args = parser.parse_args()

    data_dir = gw_dir(args.gameweek) / "data"
    aggregate_path = data_dir / f"gw_{args.gameweek}_aggregated_player_form.json"
    transcript_analysis_path = data_dir / f"gw_{args.gameweek}_transcript_analysis.json"
    transcript_extractions_path = data_dir / f"gw_{args.gameweek}_transcript_extractions.json"
    text_analysis_path = data_dir / f"gw_{args.gameweek}_text_extractions.json"
    cache_dir = gw_dir(args.gameweek) / "transcript_cache"
    if not aggregate_path.exists():
        raise FileNotFoundError(
            f"Missing {aggregate_path.relative_to(PROJECT_ROOT)}. Run data_generator.ipynb for GW{args.gameweek} first."
        )

    players = load_json(aggregate_path)
    scored_players = []
    for player_id, player in players.items():
        data_score, data_evidence = score_player(player)
        scored_players.append({
            **player,
            "data_score": data_score,
            "data_evidence": data_evidence,
            "sentiment_score": 50.0,
            "sentiment_evidence": [],
            "mention_count": 0,
            "disagreement": None,
            "captaincy_score": 0.0,
            "captaincy_evidence": [],
        })

    transcript_sentiment: dict[int, dict[str, Any]] = {}
    transcript_captaincy: dict[int, dict[str, Any]] = {}
    transcript_themes: list[tuple[str, str]] = []
    transcript_sources: list[str] = []
    transcript_input_note: str | None = None
    if transcript_analysis_path.exists():
        transcript_sentiment, transcript_captaincy, transcript_themes, transcript_sources = load_transcript_analysis(transcript_analysis_path)
        transcript_input_note = (
            f"transcript evidence from {transcript_analysis_path.relative_to(PROJECT_ROOT)} "
            f"({len(transcript_sources)} source(s))"
        )
    elif transcript_extractions_path.exists():
        transcript_sentiment, transcript_captaincy, transcript_themes, transcript_sources = load_transcript_extractions(
            transcript_extractions_path,
            players,
        )
        transcript_input_note = (
            f"transcript evidence from {transcript_extractions_path.relative_to(PROJECT_ROOT)} "
            f"({len(transcript_sources)} source(s))"
        )
    else:
        transcript_sentiment, transcript_captaincy, transcript_themes, transcript_sources = load_transcript_analysis_from_cache(
            args.gameweek,
            players,
        )
        if transcript_sources:
            transcript_input_note = (
                f"transcript evidence from data/2026-27/gameweeks/gw{args.gameweek}/transcript_cache/extraction_cache "
                f"({len(transcript_sources)} source(s), cache fallback)"
            )

    text_sentiment: dict[int, dict[str, Any]] = {}
    text_captaincy: dict[int, dict[str, Any]] = {}
    text_themes: list[tuple[str, str]] = []
    text_sources: list[str] = []
    if text_analysis_path.exists():
        text_sentiment, text_captaincy, text_themes, text_sources = load_text_analysis(text_analysis_path, players)

    sentiment = merge_sentiment_maps(transcript_sentiment, text_sentiment)
    captaincy = merge_captaincy_maps(transcript_captaincy, text_captaincy)
    themes = transcript_themes + text_themes
    sources = []
    for source in transcript_sources + text_sources:
        if source not in sources:
            sources.append(source)

    evidence_notes = []
    if transcript_input_note:
        evidence_notes.append(transcript_input_note)
    if text_analysis_path.exists():
        evidence_notes.append(
            f"website text evidence from {text_analysis_path.relative_to(PROJECT_ROOT)} ({len(text_sources)} source(s))"
        )
    llm_note = "; ".join(evidence_notes) if evidence_notes else "No qualitative analysis found; report uses FPL data only."

    source_to_letter = {source: chr(65 + index) for index, source in enumerate(sources)}

    for player in scored_players:
        player_sentiment = sentiment.get(player["id"])
        if player_sentiment:
            player["sentiment_score"] = player_sentiment["score"]
            player["sentiment_evidence"] = player_sentiment["evidence"]
            player["mention_count"] = player_sentiment["mention_count"]
            player["disagreement"] = player_sentiment["disagreement"]
        player_captaincy = captaincy.get(player["id"])
        if player_captaincy:
            player["captaincy_score"] = player_captaincy["score"]
            player["captaincy_evidence"] = player_captaincy["evidence"]

    for player in scored_players:
        player["combined_score"] = round(0.5 * player["data_score"] + 0.5 * player["sentiment_score"], 1)

    public_squad = latest_public_squad(args.gameweek)
    squad_ids: set[int] = set()
    squad_source_gameweek = None
    if public_squad:
        squad_source_gameweek, squad_ids_list = public_squad
        squad_ids = set(squad_ids_list)

    mentioned_ids = {
        player["id"] for player in scored_players
        if player["sentiment_evidence"] or player["captaincy_evidence"]
    }
    other_ids = mentioned_ids - squad_ids

    data_only_candidates = sorted(
        (
            player for player in scored_players
            if player["id"] not in squad_ids
            and player["id"] not in mentioned_ids
            and player["appearances"] > 0
            and player["data_score"] >= 65
        ),
        key=lambda player: player["data_score"],
        reverse=True,
    )[:10]
    data_only_ids = {player["id"] for player in data_only_candidates}

    players_by_id = {player["id"]: player for player in scored_players}

    def build_summary_item(player: dict[str, Any]) -> dict[str, Any]:
        analyst_evidence = []
        for evidence in player["sentiment_evidence"]:
            letter = source_to_letter.get(evidence["source"], "?")
            analyst_evidence.append(f"[{letter}] {evidence['stance']}: {evidence['rationale']}")
        for evidence in player["captaincy_evidence"]:
            letter = source_to_letter.get(evidence["source"], "?")
            percentage = f" ({evidence['expert_percentage']}%)" if evidence.get("expert_percentage") else ""
            analyst_evidence.append(f"[{letter}] {evidence['mention_type']}{percentage}: {evidence['rationale']}")
        return {
            "id": player["id"],
            "name": player["name"],
            "team": player["team"],
            "position": player["position"],
            "data_evidence": player["data_evidence"],
            "analyst_evidence": analyst_evidence,
            "disagreement": player["disagreement"],
        }

    summary_ids = squad_ids | other_ids | data_only_ids
    summaries: dict[int, dict[str, str]] = {}
    if not args.no_llm and summary_ids:
        from dotenv import load_dotenv

        load_dotenv(ENV_FILE)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env, or run with --no-llm.")
        from openai import OpenAI

        summary_client = OpenAI(api_key=api_key)
        summary_items = [
            build_summary_item(players_by_id[player_id])
            for player_id in sorted(summary_ids) if player_id in players_by_id
        ]
        summaries = generate_player_summaries(summary_client, args.model, cache_dir, summary_items)

    def fallback_verdict(player: dict[str, Any]) -> str:
        if player["sentiment_evidence"] or player["captaincy_evidence"]:
            return "Qualitative + data coverage (summary unavailable)"
        if player["data_score"] >= 65:
            return "Data-only, no qualitative coverage"
        return "Data-only, weak signal"

    def render_player_summary(player: dict[str, Any]) -> tuple[str, str]:
        summary = summaries.get(player["id"], {})
        verdict = summary.get("verdict") or fallback_verdict(player)
        summary_text = summary.get("summary")
        if not summary_text:
            bullets = player["data_evidence"][:3]
            summary_text = "; ".join(bullets) if bullets else "No data available."
        return verdict, summary_text

    def render_player_card(player: dict[str, Any], section_label: str) -> str:
        verdict, summary_text = render_player_summary(player)
        position_label = POSITION_LABELS.get(player["position"], str(player["position"]))
        sources = sorted(
            {
                source_to_letter.get(evidence["source"], "?")
                for evidence in player["sentiment_evidence"] + player["captaincy_evidence"]
                if source_to_letter.get(evidence["source"], "?")
            }
        )
        evidence_bits: list[str] = []
        if player["sentiment_evidence"]:
            for evidence in player["sentiment_evidence"][:4]:
                source = source_to_letter.get(evidence["source"], "?")
                evidence_bits.append(
                    f"<li><strong>[{escape(source)}] {escape(str(evidence.get('stance', '')))}</strong>: "
                    f"{escape(str(evidence.get('rationale', '')))}"
                    + (
                        f"<div class='quote'>\u201c{escape(str(evidence.get('supporting_quote', '')))}\u201d</div>"
                        if evidence.get("supporting_quote") else ""
                    )
                    + "</li>"
                )
        if player["captaincy_evidence"]:
            for evidence in player["captaincy_evidence"][:4]:
                source = source_to_letter.get(evidence["source"], "?")
                mention = str(evidence.get("mention_type", ""))
                if evidence.get("expert_percentage"):
                    mention += f" ({evidence['expert_percentage']}%)"
                evidence_bits.append(
                    f"<li><strong>[{escape(source)}] {escape(mention)}</strong>: "
                    f"{escape(str(evidence.get('rationale', '')))}"
                    + (
                        f"<div class='quote'>\u201c{escape(str(evidence.get('supporting_quote', '')))}\u201d</div>"
                        if evidence.get("supporting_quote") else ""
                    )
                    + "</li>"
                )
        if player["disagreement"]:
            evidence_bits.append(f"<li><strong>Disagreement:</strong> {escape(str(player['disagreement']))}</li>")

        fixture_run = format_fixture_run(player)
        fixture_line = format_fixture(player)
        metrics = [
            f"<span class='metric'><strong>Data:</strong> {player['data_score']:.1f}</span>",
            f"<span class='metric'><strong>Sentiment:</strong> {player['sentiment_score']:.1f}</span>",
            f"<span class='metric'><strong>Combined:</strong> {player['combined_score']:.1f}</span>",
            f"<span class='metric'><strong>Mentions:</strong> {player['mention_count']}</span>",
        ]
        if player["captaincy_evidence"]:
            metrics.append(f"<span class='metric'><strong>Captaincy:</strong> {player['captaincy_score']:.1f}</span>")

        source_pills = "".join(f"<span class='pill'>{escape(letter)}</span>" for letter in sources)
        evidence_html = "".join(evidence_bits) if evidence_bits else "<li>No qualitative evidence for this player.</li>"

        return f"""
        <article class="player-card" data-name="{escape(player['name'].lower())}" data-team="{escape(player['team'].lower())}" data-verdict="{escape(verdict.lower())}" data-section="{escape(section_label.lower())}">
          <div class="player-card__header">
            <div>
              <h3>{escape(player['name'])}</h3>
              <p class="subtitle">{escape(player['team'])} · {escape(position_label)}</p>
            </div>
            <div class="verdict">{escape(verdict)}</div>
          </div>
          <p class="summary">{escape(summary_text)}</p>
          <div class="meta-row">
            <span class="metric"><strong>Fixture:</strong> {escape(fixture_line)}</span>
            {f"<span class='metric'><strong>Next 5 FDR:</strong> {escape(fixture_run)}</span>" if fixture_run != '-' else ''}
          </div>
          <div class="meta-row">{''.join(metrics)}</div>
          <div class="source-row">{source_pills}</div>
          <details open>
            <summary>Evidence</summary>
            <ul class="evidence-list">{evidence_html}</ul>
          </details>
        </article>
        """

    def render_player_group(title: str, description: str | None, players: list[dict[str, Any]], section_label: str) -> str:
        if players:
            cards = "".join(render_player_card(player, section_label) for player in players)
        else:
            cards = f"<p class='empty'>{escape(description or 'No players found.')}</p>"
        return f"""
        <section class="report-section" id="{escape(section_label.lower().replace(' ', '-'))}">
          <div class="section-heading">
            <div>
              <h2>{escape(title)}</h2>
              {f"<p>{escape(description)}</p>" if description else ''}
            </div>
          </div>
          <div class="card-grid">{cards}</div>
        </section>
        """

    captain_candidates = sorted(
        [player for player in scored_players if player["captaincy_evidence"]],
        key=lambda player: player["captaincy_score"],
        reverse=True,
    )[:8]
    headline_players = sorted(
        scored_players,
        key=lambda player: (player["combined_score"], player["mention_count"], player["captaincy_score"]),
        reverse=True,
    )[:8]

    if squad_ids:
        squad_players = sorted(
            (players_by_id[player_id] for player_id in squad_ids if player_id in players_by_id),
            key=lambda player: player["combined_score"],
            reverse=True,
        )
        my_team_section = render_player_group(
            "My Team",
            f"Your latest public squad is from GW{squad_source_gameweek}. Public FPL data does not expose the current pre-deadline squad.",
            squad_players,
            "My Team",
        )
    else:
        squad_players = []
        my_team_section = render_player_group(
            "My Team",
            "No completed-gameweek team-picks file is available yet.",
            [],
            "My Team",
        )

    other_players = sorted(
        (players_by_id[player_id] for player_id in other_ids if player_id in players_by_id),
        key=lambda player: player["combined_score"],
        reverse=True,
    )
    other_players_section = render_player_group(
        "Other Players Worth Knowing About",
        "No further source-mentioned players outside your squad.",
        other_players,
        "Other Players Worth Knowing About",
    )
    data_only_section = render_player_group(
        "Data Also Flags (No Analyst Coverage)",
        "No additional high-data players outside qualitative coverage.",
        data_only_candidates,
        "Data Also Flags",
    )

    if True:
        headline_cards_html = "".join(
                f'<div class="headline-card">'
                f'<strong>{escape(player["name"])} · {escape(player["team"])} · {escape(POSITION_LABELS.get(player["position"], str(player["position"])))}</strong>'
                f'<p>{escape(render_player_summary(player)[1])}</p>'
                f'<p><strong>Verdict:</strong> {escape(render_player_summary(player)[0])}</p>'
                f'</div>'
                for player in headline_players
        )
        captain_cards_html = "".join(render_player_card(player, 'Captain Picks') for player in captain_candidates)
        captain_section_html = (
                f'<div class="card-grid">{captain_cards_html}</div>'
                if captain_cards_html else '<p class="empty">No captaincy-specific mentions were found in the qualitative sources.</p>'
        )
        source_cards_html = "".join(
                f'<div class="headline-card"><strong>{escape(letter)}</strong><p>{escape(source)}</p></div>'
                for source, letter in sorted(source_to_letter.items(), key=lambda item: item[1])
        )
        sources_section_html = (
                f'<div class="headline-grid">{source_cards_html}</div>'
                if source_cards_html else '<p class="empty">No qualitative sources were loaded.</p>'
        )

        html_report = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>FPL Gameweek {args.gameweek} Report</title>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f5f1e8;
            --panel: #fffaf2;
            --panel-strong: #f2e7d8;
            --text: #1f2328;
            --muted: #6b7280;
            --line: #d8cbb8;
            --accent: #0f766e;
            --accent-2: #b45309;
            --pill: #ecfdf5;
            --shadow: 0 18px 50px rgba(31, 35, 40, 0.08);
        }}
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; font-family: Georgia, 'Times New Roman', serif; color: var(--text); background: radial-gradient(circle at top, #fffdf8 0, var(--bg) 45%, #efe4d3 100%); }}
        .page {{ max-width: 1440px; margin: 0 auto; padding: 32px 20px 64px; }}
        .hero {{ background: linear-gradient(135deg, #163a39, #215f5d 55%, #2f756f); color: #fff; border-radius: 28px; padding: 28px; box-shadow: var(--shadow); }}
        .hero h1 {{ margin: 0 0 10px; font-size: clamp(2rem, 4vw, 3.5rem); letter-spacing: -0.03em; }}
        .hero p {{ margin: 0; max-width: 84ch; color: rgba(255,255,255,0.88); line-height: 1.55; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 20px; }}
        .stat {{ background: rgba(255,255,255,0.11); border: 1px solid rgba(255,255,255,0.18); border-radius: 18px; padding: 14px 16px; }}
        .stat span {{ display: block; color: rgba(255,255,255,0.7); font-size: 0.84rem; text-transform: uppercase; letter-spacing: 0.08em; }}
        .stat strong {{ display: block; margin-top: 6px; font-size: 1.05rem; }}
        .toolbar {{ position: sticky; top: 0; z-index: 10; background: rgba(245,241,232,0.88); backdrop-filter: blur(14px); padding: 14px 0; margin: 18px 0; }}
        .toolbar-inner {{ display: grid; gap: 10px; grid-template-columns: 1fr auto; align-items: center; background: var(--panel); border: 1px solid var(--line); border-radius: 20px; padding: 14px 16px; box-shadow: var(--shadow); }}
        .toolbar input {{ width: 100%; border: 0; outline: 0; font-size: 1rem; background: transparent; color: var(--text); }}
        .toolbar .hint {{ color: var(--muted); font-size: 0.92rem; }}
        .section-heading {{ display: flex; justify-content: space-between; align-items: end; gap: 16px; margin: 30px 4px 14px; }}
        .section-heading h2 {{ margin: 0; font-size: 1.7rem; }}
        .section-heading p {{ margin: 6px 0 0; color: var(--muted); max-width: 80ch; }}
        .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
        .player-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 22px; padding: 18px; box-shadow: var(--shadow); }}
        .player-card h3 {{ margin: 0; font-size: 1.25rem; }}
        .subtitle {{ margin: 4px 0 0; color: var(--muted); font-size: 0.92rem; }}
        .player-card__header {{ display: flex; justify-content: space-between; gap: 12px; align-items: start; }}
        .verdict {{ background: #183f3d; color: white; padding: 8px 12px; border-radius: 999px; font-size: 0.82rem; max-width: 44%; text-align: right; }}
        .summary {{ margin: 14px 0; line-height: 1.6; }}
        .meta-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }}
        .metric {{ background: var(--panel-strong); border-radius: 999px; padding: 8px 10px; font-size: 0.88rem; }}
        .source-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 0; }}
        .pill {{ display: inline-flex; align-items: center; justify-content: center; min-width: 30px; height: 28px; border-radius: 999px; background: var(--pill); border: 1px solid #cfe9db; color: var(--accent); font-weight: 700; font-size: 0.82rem; }}
        details {{ margin-top: 14px; }}
        summary {{ cursor: pointer; color: var(--accent-2); font-weight: 700; }}
        .evidence-list {{ margin: 12px 0 0; padding-left: 18px; display: grid; gap: 10px; }}
        .quote {{ margin-top: 6px; padding: 10px 12px; border-left: 3px solid var(--line); background: #fcfaf5; color: #4b5563; font-style: italic; }}
        .empty {{ color: var(--muted); margin: 16px 4px; }}
        .headline-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
        .headline-card {{ background: linear-gradient(180deg, #fffdf8, #f8efe0); border: 1px solid var(--line); border-radius: 18px; padding: 14px; }}
        .headline-card strong {{ display: block; margin-bottom: 6px; }}
        .headline-card p {{ margin: 0; color: var(--muted); line-height: 1.45; }}
        .footer {{ margin-top: 36px; color: var(--muted); font-size: 0.92rem; }}
        @media (max-width: 760px) {{
            .stats, .toolbar-inner {{ grid-template-columns: 1fr; }}
            .verdict {{ max-width: none; }}
        }}
    </style>
</head>
<body>
    <div class="page">
        <section class="hero">
            <h1>FPL Gameweek {args.gameweek} Report</h1>
            <p>This is decision support, not a points prediction. It combines recent form, fixture context, and the current qualitative source set into a searchable static report.</p>
            <div class="stats">
                <div class="stat"><span>Data input</span><strong>{escape(str(aggregate_path.relative_to(PROJECT_ROOT)))}</strong></div>
                <div class="stat"><span>Transcript input</span><strong>{len(transcript_sources)} source(s)</strong></div>
                <div class="stat"><span>Text input</span><strong>{len(text_sources)} source(s)</strong></div>
                <div class="stat"><span>Analysis</span><strong>{escape(llm_note)}</strong></div>
            </div>
        </section>

        <div class="toolbar">
            <div class="toolbar-inner">
                <input id="player-search" type="search" placeholder="Search players, teams, verdicts, sources..." aria-label="Search players" />
                <div class="hint">Filter cards instantly in the browser.</div>
            </div>
        </div>

        <section class="report-section">
            <div class="section-heading">
                <div>
                    <h2>Headline Players</h2>
                    <p>Top names by the current combined signal, with captaincy options surfaced separately for quick review.</p>
                </div>
            </div>
            <div class="headline-grid">
                {headline_cards_html}
            </div>
        </section>

        <section class="report-section">
            <div class="section-heading">
                <div>
                    <h2>Captain Picks</h2>
                    <p>Ordered by the combined captaincy signal across all qualitative mentions.</p>
                </div>
            </div>
            {captain_section_html}
        </section>

        <section class="report-section">
            <div class="section-heading">
                <div>
                    <h2>Sources</h2>
                    <p>Letters are used throughout the page so you can jump between the summary and the source evidence quickly.</p>
                </div>
            </div>
            {sources_section_html}
        </section>

        {my_team_section}
        {other_players_section}
        {data_only_section}

        <div class="footer">Generated from the same scoring pipeline as the markdown report, but rendered as a searchable HTML page.</div>
    </div>

    <script>
        const search = document.getElementById('player-search');
        const cards = Array.from(document.querySelectorAll('.player-card'));
        function applyFilter() {{
            const query = search.value.trim().toLowerCase();
            cards.forEach((card) => {{
                const haystack = [card.dataset.name, card.dataset.team, card.dataset.verdict, card.dataset.section, card.innerText].join(' ').toLowerCase();
                card.style.display = !query || haystack.includes(query) ? '' : 'none';
            }});
        }}
        search.addEventListener('input', applyFilter);
    </script>
</body>
</html>
"""

    reports_dir = REPORTS_DIR / f"gw_{args.gameweek}"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"gw_{args.gameweek}_report.html"
    report_path.write_text(html_report, encoding="utf-8")
    print(f"Report written to {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
