"""Single source of truth for where code, data, and outputs live.

Every module builds file locations from here, so moving a folder means
changing one line in this file rather than hunting through the codebase.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEASON = "2026-27"

DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"  # hand-curated + generated lookups (tracked in git)
HISTORY_DIR = DATA_DIR / "history"  # past seasons: vaastav archive, FPL API snapshots
SEASON_DIR = DATA_DIR / SEASON

GAMEWEEKS_DIR = SEASON_DIR / "gameweeks"  # per-GW inputs, caches, and derived JSON
SEASON_DATA_DIR = SEASON_DIR / "season_data"  # season-long player and fixture tables
REPORTS_DIR = SEASON_DIR / "reports"
OPTIMISER_OUTPUT_DIR = SEASON_DIR / "optimiser"
BACKTEST_OUTPUT_DIR = SEASON_DIR / "backtest"

ENV_FILE = PROJECT_ROOT / ".env"


def gw_dir(gameweek: int) -> Path:
    return GAMEWEEKS_DIR / f"gw{gameweek}"


def gw_data_dir(gameweek: int) -> Path:
    return gw_dir(gameweek) / "data"


def gw_report_dir(gameweek: int) -> Path:
    return REPORTS_DIR / f"gw_{gameweek}"


def vaastav_season_dir(season: str) -> Path:
    return HISTORY_DIR / "vaastav" / season
