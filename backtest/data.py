"""Load point-in-time snapshots, consensus, and final points for one gameweek."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import requests

from reports.report_scripts.deterministic_report import _combined_ranked, _source_scores
from reports.report_scripts.fpl_report_v2 import load_name_resolution_helpers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPL_API = "https://fantasy.premierleague.com/api"


def data_dir(gameweek: int) -> Path:
    return PROJECT_ROOT / "gw" / f"gw{gameweek}" / "data"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_snapshot(gameweek: int) -> dict[str, dict[str, Any]]:
    """Player form as it stood before the gameweek deadline."""
    return load_json(data_dir(gameweek) / f"gw_{gameweek}_aggregated_player_form.json")


def has_numeric_extractions(gameweek: int) -> bool:
    """True when extractions exist and use the 1-5 stance scale (early gameweeks used text labels)."""
    paths = [data_dir(gameweek) / f"gw_{gameweek}_{kind}_extractions.json" for kind in ("transcript", "text")]
    recommendations = [
        item for path in paths if path.exists() for item in load_json(path).get("recommendations", [])
    ]
    return bool(recommendations) and all(isinstance(item.get("stance"), int) for item in recommendations)


def finished_gameweeks() -> set[int]:
    """Gameweeks whose points the FPL API has finalised (finished and bonus-checked)."""
    response = requests.get(f"{FPL_API}/bootstrap-static/", timeout=30)
    response.raise_for_status()
    return {
        int(event["id"])
        for event in response.json()["events"]
        if event.get("finished") and event.get("data_checked")
    }


def load_final_points(gameweek: int) -> dict[int, int]:
    """Final points per player, summed across fixtures, cached after the first fetch.

    The saved `api_data_player_stats` files can be captured mid-gameweek and hold a
    single fixture per player, so the live endpoint is used instead.
    """
    cache_path = data_dir(gameweek) / f"gw_{gameweek}_final_points.json"
    if cache_path.exists():
        return {int(player_id): points for player_id, points in load_json(cache_path).items()}

    response = requests.get(f"{FPL_API}/event/{gameweek}/live/", timeout=30)
    response.raise_for_status()
    points = {int(item["id"]): int(item["stats"]["total_points"]) for item in response.json()["elements"]}
    cache_path.write_text(json.dumps(points, indent=1), encoding="utf-8")
    return points


def _resolvable(extraction: dict[str, Any], players: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Drop recommendations whose names the resolver cannot match, warning about each."""
    helpers = load_name_resolution_helpers(players)
    player_names = {player["name"]: int(player_id) for player_id, player in players.items()}
    kept, dropped = [], []
    for item in extraction.get("recommendations", []):
        if not isinstance(item, dict):
            continue
        if helpers.resolve_supported_player_id(dict(item), player_names) is None:
            dropped.append(str(item.get("player_name", "")))
        else:
            kept.append(item)
    if dropped:
        print(f"  dropped {len(dropped)} unresolved recommendation(s): {', '.join(sorted(set(dropped)))}", file=sys.stderr)
    return {**extraction, "recommendations": kept}


def load_consensus(gameweek: int, players: dict[str, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Combined transcript + text consensus, computed exactly as the deterministic report does."""
    ranked_by_kind = []
    for kind in ("transcript", "text"):
        path = data_dir(gameweek) / f"gw_{gameweek}_{kind}_extractions.json"
        extraction = load_json(path) if path.exists() else {"recommendations": [], "sources": []}
        ranked, _ = _source_scores(_resolvable(extraction, players), players)
        ranked_by_kind.append(ranked)
    return {int(item["player_id"]): item for item in _combined_ranked(*ranked_by_kind)}
