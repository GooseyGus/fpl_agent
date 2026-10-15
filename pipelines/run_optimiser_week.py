"""Update player and fixture data before running optimiser."""
import sys
import subprocess
from pathlib import Path

# Launched as a script, so put the repo root on the path for `fpl_hq`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_hq.gather.api_interact import init_or_update_player_data
from fpl_hq.gather.fixture_interact import init_or_update_fixture_data
from fpl_hq.paths import OPTIMISER_OUTPUT_DIR


def get_next_gameweek() -> int | None:
    """Determine which gameweek to optimize.

    Gets the current gameweek from the API.
    """
    import requests

    BASE_URL = "https://fantasy.premierleague.com/api/"
    bootstrap = requests.get(BASE_URL + "bootstrap-static/").json()
    events = bootstrap["events"]

    # Find current gameweek
    for event in events:
        if event["is_current"]:
            return event["id"]

    # If no current gameweek, find the next unfinished one
    for event in events:
        if not event["finished"]:
            return event["id"]

    # All 38 gameweeks done
    print("✅ All 38 gameweeks completed!")
    return None


if __name__ == "__main__":
    # Get the gameweek to optimize
    if len(sys.argv) > 1:
        gameweek = int(sys.argv[1])
    else:
        gameweek = get_next_gameweek()

    if gameweek is None:
        sys.exit(0)

    print(f"🎯 Running GW{gameweek}...")

    # Update data
    init_or_update_player_data(gameweek)
    init_or_update_fixture_data(gameweek)

    # Run dual optimizer (works for GW1+ with different logic)
    if gameweek == 1:
        # GW1: simple squad selection
        subprocess.run([
            sys.executable,
            "-m", "fpl_hq.optimiser.run_dual_optimiser",
            "--gameweek", "1"
        ], check=True)
    else:
        # GW2+: transfer optimization
        prev_squad_file = OPTIMISER_OUTPUT_DIR / f"squad_gw{gameweek - 1}_api.json"
        if not prev_squad_file.exists():
            print(f"❌ Previous squad file not found: {prev_squad_file}")
            sys.exit(1)

        subprocess.run([
            sys.executable,
            "-m", "fpl_hq.optimiser.run_dual_optimiser",
            "--gameweek", str(gameweek),
            "--prev-squad", str(prev_squad_file)
        ], check=True)