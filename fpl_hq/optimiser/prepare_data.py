from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from fpl_hq.paths import vaastav_season_dir

SELECTED_COLUMNS = [
    "name",
    "position",
    "team",
    "xP",
    "assists",
    "bonus",
    "bps",
    "clean_sheets",
    "clearances_blocks_interceptions",
    "creativity",
    "defensive_contribution",
    "element",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals",
    "expected_goals_conceded",
    "fixture",
    "goals_conceded",
    "goals_scored",
    "ict_index",
    "influence",
    "kickoff_time",
    "minutes",
    "modified",
    "opponent_team",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "recoveries",
    "red_cards",
    "round",
    "saves",
    "selected",
    "starts",
    "tackles",
    "team_a_score",
    "team_h_score",
    "threat",
    "total_points",
    "transfers_balance",
    "transfers_in",
    "transfers_out",
    "value",
    "was_home",
    "yellow_cards",
]

POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


def _default_gws_path(season: str = "2025-26") -> Path:
    return vaastav_season_dir(season) / "gws"


def _gw_sort_key(path: Path) -> int:
    match = re.fullmatch(r"gw(\d+)\.csv", path.name)
    if not match:
        return 10_000
    return int(match.group(1))


def load_and_prepare_gw_data(
    gws_path: Path | None = None,
    season: str = "2025-26",
    min_gw: int = 1,
    max_gw: int = 38,
) -> pd.DataFrame:
    if gws_path is None:
        gws_path = _default_gws_path(season=season)

    gw_files = []
    for candidate in gws_path.glob("gw*.csv"):
        match = re.fullmatch(r"gw(\d+)\.csv", candidate.name)
        if match is None:
            continue
        gw_number = int(match.group(1))
        if min_gw <= gw_number <= max_gw:
            gw_files.append(candidate)

    gw_files.sort(key=_gw_sort_key)

    if not gw_files:
        raise FileNotFoundError(f"No GW CSV files found in {gws_path}")

    frames = []
    for gw_file in gw_files:
        gw_df = pd.read_csv(gw_file)
        missing_columns = [col for col in SELECTED_COLUMNS if col not in gw_df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {gw_file.name}: {', '.join(missing_columns)}"
            )
        frames.append(gw_df[SELECTED_COLUMNS])

    prepared = pd.concat(frames, ignore_index=True)
    prepared["_position_order"] = (
        prepared["position"].map(POSITION_ORDER).fillna(len(POSITION_ORDER))
    )
    prepared = prepared.sort_values(
        by=["team", "_position_order", "name"], kind="mergesort"
    ).drop(columns="_position_order")
    prepared = prepared.reset_index(drop=True)
    return prepared