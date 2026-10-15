import pandas as pd
import requests
from pathlib import Path

from fpl_hq.paths import SEASON_DATA_DIR

BASE_URL = "https://fantasy.premierleague.com/api/"
CSV_PATH = SEASON_DATA_DIR / "overall_player_data_25_26.csv"

COLUMNS = [
    "id",
    "code",
    "web_name",
    "first_name",
    "second_name",
    "team",
    "element_type",
    "total_points",
    "points_per_game",
    "minutes",
    "starts",
    "now_cost",
    "goals_scored",
    "assists",
    "clean_sheets",
    "bonus",
    "bps",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "ict_index",
    "influence",
    "creativity",
    "threat",
    "form",
    "selected_by_percent",
]


def fetch_and_save_player_data() -> None:
    """Fetch bootstrap player data and save to CSV."""
    bootstrap = requests.get(BASE_URL + "bootstrap-static/").json()
    elements = bootstrap["elements"]

    df = pd.DataFrame(elements)
    df = df[COLUMNS]
    df = df.sort_values(by="total_points", ascending=False).reset_index(drop=True)

    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(df)} players to {CSV_PATH}")


if __name__ == "__main__":
    fetch_and_save_player_data()
