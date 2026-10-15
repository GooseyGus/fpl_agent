import sys
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional

from fpl_hq.paths import SEASON_DATA_DIR


BASE_URL = "https://fantasy.premierleague.com/api/"
DATA_DIR = SEASON_DATA_DIR
CSV_PATH = DATA_DIR / "player_data_26_27.csv"

SELECTED_COLUMNS = [
    "id",
    "code",
    "web_name",
    "first_name",
    "second_name",
    "team",
    "element_type",
    "now_cost",
    "ep_this",
    "ep_next",
    "event_points",
    "form",
    "status",
    "total_points",
    "points_per_game",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "starts",
]


def get_bootstrap_data() -> dict:
    """Fetch bootstrap data from API."""
    return requests.get(BASE_URL + "bootstrap-static/").json()


def get_gameweek_info(gameweek: int, bootstrap: dict) -> Optional[dict]:
    """Get info about a specific gameweek."""
    for event in bootstrap["events"]:
        if event["id"] == gameweek:
            return event
    return None


def is_gameweek_finished(gameweek: int, bootstrap: dict) -> bool:
    """Check if a gameweek is marked as finished in the API."""
    gw_info = get_gameweek_info(gameweek, bootstrap)
    if not gw_info:
        return False
    return gw_info["finished"]


def get_last_fetched_gw(csv_path: Path) -> Optional[int]:
    """Get the last gameweek we successfully fetched."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    df = pd.read_csv(csv_path)
    return int(df["round"].max()) if len(df) > 0 else None


def fetch_gw0_data(bootstrap: dict) -> pd.DataFrame:
    """Fetch pre-season snapshot (GW0) with player info and previous season stats."""
    elements = bootstrap["elements"]

    rows = []
    for player in elements:
        row = {col: player.get(col) for col in SELECTED_COLUMNS}
        row["round"] = 0
        row["season"] = "2026-27"
        rows.append(row)

    return pd.DataFrame(rows)


def fetch_gameweek_data(gameweek: int, bootstrap: dict) -> pd.DataFrame:
    """Fetch gameweek performance data (only works for finished gameweeks)."""
    if not is_gameweek_finished(gameweek, bootstrap):
        return None  # Signal that GW isn't ready yet

    elements = bootstrap["elements"]
    rows = []

    for player in elements:
        row = {col: player.get(col) for col in SELECTED_COLUMNS}
        row["round"] = gameweek
        row["season"] = "2026-27"
        rows.append(row)

    return pd.DataFrame(rows)


def init_or_update_player_data(force_gw: Optional[int] = None) -> None:
    """Initialize or update player_data_26_27.csv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bootstrap = get_bootstrap_data()

    # First run: Initialize with GW0
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        print("📝 Initializing with GW0 (pre-season snapshot)...")
        gw0_data = fetch_gw0_data(bootstrap)
        gw0_data = gw0_data.sort_values(by=["round", "id"], ignore_index=True)
        gw0_data.to_csv(CSV_PATH, index=False)
        print(f"✅ Saved {len(gw0_data)} players at GW0")
        print("⏳ Waiting for GW1 to finish before fetching live data...")
        return

    # Subsequent runs: Check what GW to fetch
    if force_gw:
        target_gw = force_gw
        print(f"🎯 Force fetching GW{target_gw}...")
    else:
        last_fetched = get_last_fetched_gw(CSV_PATH)
        target_gw = (last_fetched or 0) + 1
        print(f"📊 Last fetched: GW{last_fetched}, attempting GW{target_gw}...")

    gw_info = get_gameweek_info(target_gw, bootstrap)
    
    if not gw_info:
        print(f"❌ GW{target_gw} doesn't exist in the API yet.")
        return

    if not is_gameweek_finished(target_gw, bootstrap):
        deadline = gw_info["deadline_time"]
        deadline_dt = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
        now = datetime.now(deadline_dt.tzinfo)

        if deadline_dt > now:
            print(f"⏳ GW{target_gw} hasn't started yet (deadline: {deadline})")
            print("   Check back after all matches finish.")
        else:
            print(f"⏳ GW{target_gw} is in progress (deadline was: {deadline})")
            print("   Check back after all matches finish to avoid partial data.")
        return

    # Fetch the finished gameweek
    new_gw_data = fetch_gameweek_data(target_gw, bootstrap)
    
    if new_gw_data is None or new_gw_data.empty:
        print(f"❌ No data for GW{target_gw}")
        return

    # Load existing and update
    existing_data = pd.read_csv(CSV_PATH)
    existing_data = existing_data[existing_data["round"] != target_gw]
    combined_data = pd.concat([existing_data, new_gw_data], ignore_index=True)
    combined_data = combined_data.sort_values(by=["round", "id"], ignore_index=True)
    combined_data.to_csv(CSV_PATH, index=False)
    print(f"✅ Fetched GW{target_gw}: {len(combined_data)} total rows")


if __name__ == "__main__":
    force_gw = None
    if len(sys.argv) > 1:
        force_gw = int(sys.argv[1])

    init_or_update_player_data(force_gw)
