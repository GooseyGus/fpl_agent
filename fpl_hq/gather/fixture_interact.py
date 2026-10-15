import sys
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional

from fpl_hq.paths import SEASON_DATA_DIR


BASE_URL = "https://fantasy.premierleague.com/api/"
DATA_DIR = SEASON_DATA_DIR
CSV_PATH = DATA_DIR / "fixture_data_26_27.csv"

FIXTURE_COLUMNS = [
    "id",
    "code",
    "event",
    "kickoff_time",
    "team_h",
    "team_a",
    "team_h_score",
    "team_a_score",
    "team_h_difficulty",
    "team_a_difficulty",
    "minutes",
    "finished",
    "started",
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


def fetch_gw0_fixtures(bootstrap: dict) -> pd.DataFrame:
    """Fetch GW0 fixture preview (fixtures for upcoming gameweeks, not yet started)."""
    fixtures_data = requests.get(BASE_URL + "fixtures/").json()

    rows = []
    for fixture in fixtures_data:
        # Only include fixtures that haven't started yet (GW0 = preview state)
        if not fixture.get("started"):
            row = {col: fixture.get(col) for col in FIXTURE_COLUMNS}
            row["round"] = 0  # Mark as GW0 (pre-gameweek preview)
            row["season"] = "2026-27"
            rows.append(row)

    return pd.DataFrame(rows)


def fetch_gameweek_fixtures(gameweek: int, bootstrap: dict) -> pd.DataFrame:
    """Fetch fixtures for a specific gameweek."""
    if not is_gameweek_finished(gameweek, bootstrap):
        return None  # Signal that GW isn't ready yet

    fixtures_data = requests.get(BASE_URL + f"fixtures?event={gameweek}").json()
    
    rows = []
    for fixture in fixtures_data:
        row = {col: fixture.get(col) for col in FIXTURE_COLUMNS}
        row["round"] = gameweek
        row["season"] = "2026-27"
        rows.append(row)
    
    return pd.DataFrame(rows)


def init_or_update_fixture_data(force_gw: Optional[int] = None) -> None:
    """Initialize or update fixture_data_26_27.csv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bootstrap = get_bootstrap_data()

    # First run: Initialize with all fixtures (GW0)
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        print("📋 Initializing fixtures (all gameweeks)...")
        fixtures_data = fetch_gw0_fixtures(bootstrap)
        fixtures_data = fixtures_data.sort_values(by=["round", "id"], ignore_index=True)
        fixtures_data.to_csv(CSV_PATH, index=False)
        print(f"✅ Saved {len(fixtures_data)} fixtures")
        return

    # Subsequent runs: Update finished gameweeks
    if force_gw:
        target_gw = force_gw
        print(f"🎯 Force fetching GW{target_gw} fixtures...")
    else:
        last_fetched = get_last_fetched_gw(CSV_PATH)
        target_gw = (last_fetched or 0) + 1
        print(f"📋 Last fetched: GW{last_fetched}, attempting GW{target_gw}...")

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
    new_gw_fixtures = fetch_gameweek_fixtures(target_gw, bootstrap)
    
    if new_gw_fixtures is None or new_gw_fixtures.empty:
        print(f"❌ No fixtures for GW{target_gw}")
        return

    # Load existing and update
    existing_data = pd.read_csv(CSV_PATH)
    existing_data = existing_data[existing_data["round"] != target_gw]
    combined_data = pd.concat([existing_data, new_gw_fixtures], ignore_index=True)
    combined_data = combined_data.sort_values(by=["round", "id"], ignore_index=True)
    combined_data.to_csv(CSV_PATH, index=False)
    print(f"✅ Fetched GW{target_gw} fixtures: {len(combined_data)} total rows")


if __name__ == "__main__":
    force_gw = None
    if len(sys.argv) > 1:
        force_gw = int(sys.argv[1])

    init_or_update_fixture_data(force_gw)
