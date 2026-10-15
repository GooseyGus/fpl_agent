"""Run two optimizers side-by-side: API ep_next vs custom gus_ep_next."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests
from pulp import LpMaximize, LpProblem, LpStatus, LpVariable, PULP_CBC_CMD, lpSum, value

from fpl_hq.optimiser.constraints import (
    add_budget_constraint,
    add_gw1_bench_quality_constraint,
    GK_MIN_PPG,
    GK_MIN_TOTAL_POINTS,
    OUTFIELD_MIN_PPG,
    OUTFIELD_MIN_TOTAL_POINTS,
    OUTFIELD_MIN_AVG_PPG,
    OUTFIELD_MIN_AVG_TOTAL,
    add_max_3_from_same_team_constraint,
    add_squad_position_constraints,
    add_starter_position_constraints,
    add_team_size_constraints,
)
from fpl_hq.optimiser.prepare_data import load_and_prepare_gw_data
from fpl_hq.optimiser.transfer_optimizer import build_transfer_model

from fpl_hq.optimiser.squad_state import SquadState
from fpl_hq.paths import OPTIMISER_OUTPUT_DIR, SEASON_DATA_DIR

POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
REPORT_COLUMNS = [
    "web_name",
    "position",
    "team_name",
    "opp_team_name",
    "opp_difficulty",
    "ep_next",
    "now_cost",
    "form",
    "points_per_game",
    "total_points",
]


def _coerce_numeric_columns(gameweek_pool: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = ["ep_next", "now_cost", "minutes", "total_points", "fixture", "opponent_team"]
    for column in numeric_columns:
        if column in gameweek_pool.columns:
            gameweek_pool[column] = pd.to_numeric(gameweek_pool[column], errors="coerce")
    if "ep_next" in gameweek_pool.columns:
        gameweek_pool["ep_next"] = gameweek_pool["ep_next"].fillna(0.0)
    if "now_cost" in gameweek_pool.columns:
        gameweek_pool["now_cost"] = gameweek_pool["now_cost"].fillna(0.0)
    return gameweek_pool


def _reduce_to_one_row_per_player(gameweek_pool: pd.DataFrame) -> pd.DataFrame:
    if not gameweek_pool["id"].duplicated().any():
        return gameweek_pool

    exclude_cols = {"ep_next", "fixture"}
    first_columns = {
        column: "first" for column in gameweek_pool.columns if column not in exclude_cols
    }
    aggregation = {
        **first_columns,
        "ep_next": "sum" if "ep_next" in gameweek_pool.columns else None,
        "fixture": lambda values: ",".join(str(v) for v in sorted(set(values.dropna()))),
    }
    aggregation = {k: v for k, v in aggregation.items() if v is not None}
    return gameweek_pool.groupby("id", as_index=False).agg(aggregation)


def _load_gameweek_pool_from_csv(gameweek: int) -> pd.DataFrame | None:
    """Load gameweek data from the 2026-27 CSV files."""
    data_dir = SEASON_DATA_DIR
    csv_path = data_dir / "player_data_26_27.csv"

    if not csv_path.exists():
        return None

    data = pd.read_csv(csv_path)
    gameweek_pool = data[data["round"] == gameweek].copy()

    if gameweek_pool.empty:
        return None

    # Map element_type to position
    element_type_to_position = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    if "element_type" in gameweek_pool.columns and "position" not in gameweek_pool.columns:
        gameweek_pool["position"] = gameweek_pool["element_type"].map(element_type_to_position)

    # Rename/map columns for compatibility with optimiser
    if "id" in gameweek_pool.columns and "element" not in gameweek_pool.columns:
        gameweek_pool["element"] = gameweek_pool["id"]

    if "web_name" in gameweek_pool.columns and "name" not in gameweek_pool.columns:
        gameweek_pool["name"] = gameweek_pool["web_name"]

    if "now_cost" in gameweek_pool.columns and "value" not in gameweek_pool.columns:
        gameweek_pool["value"] = gameweek_pool["now_cost"] / 10

    if "ep_next" in gameweek_pool.columns and "xP" not in gameweek_pool.columns:
        gameweek_pool["xP"] = gameweek_pool["ep_next"]

    return gameweek_pool


def _load_gameweek_pool(gameweek: int, season: str) -> pd.DataFrame:
    if gameweek == 1:
        gameweek_pool = _load_gameweek_pool_from_csv(0)
        if gameweek_pool is not None:
            return gameweek_pool

    gameweek_pool = _load_gameweek_pool_from_csv(gameweek)

    if gameweek_pool is None and gameweek > 1:
        print(f"  GW{gameweek} hasn't been played yet. Using GW0 (ep_next data) to optimize for GW{gameweek}...")
        gameweek_pool = _load_gameweek_pool_from_csv(0)

    if gameweek_pool is None:
        print(f"  No 2026-27 data found. Falling back to {season} legacy data...")
        data = load_and_prepare_gw_data(season=season)
        gameweek_pool = data[data["round"] == gameweek].copy()

    if gameweek_pool is None or gameweek_pool.empty:
        raise ValueError(f"No rows found for gameweek {gameweek}.")
    gameweek_pool = _coerce_numeric_columns(gameweek_pool)
    gameweek_pool = _reduce_to_one_row_per_player(gameweek_pool)
    return gameweek_pool.reset_index(drop=True)


def _add_custom_ep_next(gameweek_pool: pd.DataFrame) -> pd.DataFrame:
    """Add gus_ep_next column: total_points / 38."""
    gameweek_pool = gameweek_pool.copy()
    gameweek_pool["total_points"] = pd.to_numeric(gameweek_pool["total_points"], errors="coerce").fillna(0.0)
    gameweek_pool["gus_ep_next"] = gameweek_pool["total_points"] / 38.0
    return gameweek_pool


def _calculate_weighted_next_5_difficulty(team_id: int, fixtures: list, gameweek: int) -> float:
    """Calculate weighted next-5 fixture difficulty (0.8-1.2 range).
    
    Weights: 50% immediate, 25% next, 15% +2, 7% +3, 3% +4
    Returns a multiplier: 1.0 for neutral (all 3s), 0.8 for worst, 1.2 for best.
    """
    # Get all future fixtures for this team, sorted by event
    future_fixtures = [
        f for f in fixtures
        if f.get("event") is not None and f.get("event") >= gameweek and (
            f.get("team_h") == team_id or f.get("team_a") == team_id
        )
    ]
    future_fixtures.sort(key=lambda f: f.get("event", 999))

    # Get difficulties from next 5 fixtures
    difficulties = []
    for fixture in future_fixtures[:5]:
        if fixture.get("team_h") == team_id:
            diff = fixture.get("team_h_difficulty", 3)
        else:
            diff = fixture.get("team_a_difficulty", 3)
        difficulties.append(diff if diff is not None else 3)

    # Pad with neutral (3) if fewer than 5 fixtures
    while len(difficulties) < 5:
        difficulties.append(3)

    # Weighted average: 50%, 25%, 15%, 7%, 3%
    weights = [0.5, 0.25, 0.15, 0.07, 0.03]
    weighted_difficulty = sum(d * w for d, w in zip(difficulties, weights))

    # Convert to 0.8-1.2 multiplier range
    # Worst (all 5s) -> ~4.9 -> 0.8
    # Neutral (all 3s) -> 3.0 -> 1.0
    # Best (all 1s) -> ~1.1 -> 1.2
    multiplier = 1.0 - 0.2 * (weighted_difficulty - 1.0) / 4.0
    return max(0.8, min(1.2, multiplier))  # Clamp to 0.8-1.2


def _add_opponent_difficulty(gameweek_pool: pd.DataFrame, gameweek: int) -> pd.DataFrame:
    """Add opponent difficulty, team names, and adjusted EP columns based on fixture data."""
    gameweek_pool = gameweek_pool.copy()

    # Fetch fixture data and team data
    BASE_URL = "https://fantasy.premierleague.com/api/"
    fixtures = requests.get(BASE_URL + "fixtures/").json()
    bootstrap = requests.get(BASE_URL + "bootstrap-static/").json()
    teams_data = bootstrap["teams"]

    # Create team ID to name mapping
    team_id_to_name = {team["id"]: team["name"] for team in teams_data}

    # Create difficulty mapping: 5->0.8, 4->0.9, 3->1.0, 2->1.1, 1->1.2
    difficulty_multipliers = {5: 0.8, 4: 0.9, 3: 1.0, 2: 1.1, 1: 1.2}

    # Filter fixtures for this gameweek
    gw_fixtures = [f for f in fixtures if f.get("event") == gameweek]

    # Create mappings from team to opponent difficulty and opponent team
    team_difficulty = {}
    team_opponent = {}
    for fixture in gw_fixtures:
        team_h = fixture.get("team_h")
        team_a = fixture.get("team_a")
        h_diff = fixture.get("team_h_difficulty")
        a_diff = fixture.get("team_a_difficulty")

        if team_h is not None and h_diff is not None:
            team_difficulty[team_h] = h_diff
            team_opponent[team_h] = team_a
        if team_a is not None and a_diff is not None:
            team_difficulty[team_a] = a_diff
            team_opponent[team_a] = team_h

    # Add opponent difficulty and team names to gameweek pool
    gameweek_pool["opp_difficulty"] = gameweek_pool["team"].map(team_difficulty)
    gameweek_pool["opp_difficulty"] = gameweek_pool["opp_difficulty"].fillna(3)  # Default to neutral (3)

    gameweek_pool["team_name"] = gameweek_pool["team"].map(team_id_to_name)
    gameweek_pool["opp_team_id"] = gameweek_pool["team"].map(team_opponent)
    gameweek_pool["opp_team_name"] = gameweek_pool["opp_team_id"].map(team_id_to_name)

    # Create multipliers
    gameweek_pool["difficulty_multiplier"] = gameweek_pool["opp_difficulty"].map(difficulty_multipliers)

    # Create opponent-adjusted EP columns
    gameweek_pool["ep_next"] = pd.to_numeric(gameweek_pool["ep_next"], errors="coerce").fillna(0.0)
    gameweek_pool["gus_ep_next"] = pd.to_numeric(gameweek_pool.get("gus_ep_next"), errors="coerce").fillna(0.0)

    gameweek_pool["opp_adj_ep_next"] = gameweek_pool["ep_next"] * gameweek_pool["difficulty_multiplier"]
    gameweek_pool["opp_adj_gus_ep_next"] = gameweek_pool["gus_ep_next"] * gameweek_pool["difficulty_multiplier"]

    # Calculate weighted next-5 difficulty multiplier for advanced opponent adjustment
    gameweek_pool["adv_difficulty_multiplier"] = gameweek_pool["team"].apply(
        lambda team_id: _calculate_weighted_next_5_difficulty(team_id, fixtures, gameweek)
    )

    gameweek_pool["adv_opp_adj_gus_ep_next"] = gameweek_pool["gus_ep_next"] * gameweek_pool["adv_difficulty_multiplier"]

    return gameweek_pool


def _build_and_solve_model_gw1(
    gameweek_pool: pd.DataFrame,
    budget_cap: float,
    ep_column: str = "ep_next",
    bench_gk_min_ppg: float = GK_MIN_PPG,
    bench_gk_min_total: int = GK_MIN_TOTAL_POINTS,
    bench_outfield_min_ppg: float = OUTFIELD_MIN_PPG,
    bench_outfield_min_total: int = OUTFIELD_MIN_TOTAL_POINTS,
    bench_outfield_avg_ppg: float = OUTFIELD_MIN_AVG_PPG,
    bench_outfield_avg_total: int = OUTFIELD_MIN_AVG_TOTAL,
) -> pd.DataFrame:
    """Build and solve GW1 optimizer using specified ep_next column."""
    gameweek_pool = gameweek_pool.copy()
    gameweek_pool["position_order"] = gameweek_pool["position"].map(POSITION_ORDER)

    # Use specified ep column for xP
    if ep_column not in gameweek_pool.columns:
        raise ValueError(f"Column {ep_column} not found in gameweek_pool")

    gameweek_pool["xP"] = gameweek_pool[ep_column]

    model = LpProblem("fpl_team_selection", LpMaximize)
    selected_vars = {
        idx: LpVariable(f"selected_{idx}", lowBound=0, upBound=1, cat="Binary")
        for idx in gameweek_pool.index
    }
    starter_vars = {
        idx: LpVariable(f"starter_{idx}", lowBound=0, upBound=1, cat="Binary")
        for idx in gameweek_pool.index
    }

    # Maximize expected points of starters only (bench doesn't contribute to objective)
    model += lpSum(gameweek_pool.loc[idx, "xP"] * starter_vars[idx] for idx in gameweek_pool.index)

    add_budget_constraint(model, gameweek_pool, selected_vars, budget_cap=budget_cap)
    add_max_3_from_same_team_constraint(model, gameweek_pool, selected_vars)
    add_squad_position_constraints(model, gameweek_pool, selected_vars)
    add_starter_position_constraints(model, gameweek_pool, starter_vars)
    add_team_size_constraints(model, selected_vars, starter_vars)

    # Add bench quality constraint (will skip forced bench players due to their constraints)
    add_gw1_bench_quality_constraint(
        model,
        gameweek_pool,
        selected_vars,
        starter_vars,
        gk_min_ppg=bench_gk_min_ppg,
        gk_min_total_points=bench_gk_min_total,
        outfield_min_ppg=bench_outfield_min_ppg,
        outfield_min_total_points=bench_outfield_min_total,
        outfield_min_avg_ppg=bench_outfield_avg_ppg,
        outfield_min_avg_total=bench_outfield_avg_total,
    )


    for idx in gameweek_pool.index:
        model += starter_vars[idx] <= selected_vars[idx], f"starter_requires_selection_{idx}"

    for _, player_rows in gameweek_pool.groupby("element").groups.items():
        model += lpSum(selected_vars[idx] for idx in player_rows) <= 1

    solve_status = model.solve(PULP_CBC_CMD(msg=False))
    status_name = LpStatus[solve_status]
    if status_name != "Optimal":
        raise RuntimeError(f"Optimiser failed to find an optimal solution. Status: {status_name}")

    gameweek_pool["selected"] = [int(round(value(selected_vars[idx]))) for idx in gameweek_pool.index]
    gameweek_pool["starter"] = [int(round(value(starter_vars[idx]))) for idx in gameweek_pool.index]
    gameweek_pool["bench"] = gameweek_pool["selected"] - gameweek_pool["starter"]

    # Set first starter as captain
    gameweek_pool["captain"] = 0
    first_starter_idx = gameweek_pool[gameweek_pool["starter"] == 1].index[0]
    gameweek_pool.loc[first_starter_idx, "captain"] = 1

    # Add transfer state for GW1
    transfer_state = []
    for idx in gameweek_pool.index:
        if gameweek_pool.loc[idx, "starter"] == 1:
            transfer_state.append("SELECTED → STARTER")
        elif gameweek_pool.loc[idx, "bench"] == 1:
            transfer_state.append("SELECTED → BENCH")
        else:
            transfer_state.append("")
    gameweek_pool["transfer_state"] = transfer_state

    return gameweek_pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dual FPL GW optimiser (API vs Custom EP).")
    parser.add_argument("--gameweek", type=int, default=1, help="Gameweek to optimise.")
    parser.add_argument("--budget-cap", type=float, default=100.0, help="Budget cap in millions.")
    parser.add_argument("--season", type=str, default="2025-26", help="Season folder.")
    parser.add_argument("--prev-squad", type=Path, default=None, help="Previous squad for GW > 1.")
    parser.add_argument(
        "--bench-gk-min-ppg",
        type=float,
        default=GK_MIN_PPG,
        help=f"Minimum points per game for goalkeeper on bench (default: {GK_MIN_PPG}).",
    )
    parser.add_argument(
        "--bench-gk-min-total",
        type=int,
        default=GK_MIN_TOTAL_POINTS,
        help=f"Minimum total points for goalkeeper on bench (default: {GK_MIN_TOTAL_POINTS}).",
    )
    parser.add_argument(
        "--bench-outfield-min-ppg",
        type=float,
        default=OUTFIELD_MIN_PPG,
        help=f"Minimum points per game for each outfield bench player (default: {OUTFIELD_MIN_PPG}).",
    )
    parser.add_argument(
        "--bench-outfield-min-total",
        type=int,
        default=OUTFIELD_MIN_TOTAL_POINTS,
        help=f"Minimum total points for each outfield bench player (default: {OUTFIELD_MIN_TOTAL_POINTS}).",
    )
    parser.add_argument(
        "--bench-outfield-avg-ppg",
        type=float,
        default=OUTFIELD_MIN_AVG_PPG,
        help=f"Minimum average points per game for 3 outfield bench players (default: {OUTFIELD_MIN_AVG_PPG}).",
    )
    parser.add_argument(
        "--bench-outfield-avg-total",
        type=int,
        default=OUTFIELD_MIN_AVG_TOTAL,
        help=f"Minimum average total points for 3 outfield bench players (default: {OUTFIELD_MIN_AVG_TOTAL}).",
    )
    args = parser.parse_args()

    print(f"Loading data for GW{args.gameweek}...")
    gameweek_pool = _load_gameweek_pool(gameweek=args.gameweek, season=args.season)
    gameweek_pool = _add_custom_ep_next(gameweek_pool)
    gameweek_pool = _add_opponent_difficulty(gameweek_pool, args.gameweek)

    output_dir = OPTIMISER_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)


    if args.gameweek == 1:
        # GW1: Simple selection without transfers
        print(f"Running API EP optimizer (GW1)...")
        solution_api = _build_and_solve_model_gw1(
            gameweek_pool, args.budget_cap, ep_column="ep_next",
            bench_gk_min_ppg=args.bench_gk_min_ppg,
            bench_gk_min_total=args.bench_gk_min_total,
            bench_outfield_min_ppg=args.bench_outfield_min_ppg,
            bench_outfield_min_total=args.bench_outfield_min_total,
            bench_outfield_avg_ppg=args.bench_outfield_avg_ppg,
            bench_outfield_avg_total=args.bench_outfield_avg_total,
        )

        print(f"Running Custom EP (gus_ep_next) optimizer (GW1)...")
        solution_gus = _build_and_solve_model_gw1(
            gameweek_pool, args.budget_cap, ep_column="gus_ep_next",
            bench_gk_min_ppg=args.bench_gk_min_ppg,
            bench_gk_min_total=args.bench_gk_min_total,
            bench_outfield_min_ppg=args.bench_outfield_min_ppg,
            bench_outfield_min_total=args.bench_outfield_min_total,
            bench_outfield_avg_ppg=args.bench_outfield_avg_ppg,
            bench_outfield_avg_total=args.bench_outfield_avg_total,
        )

        print(f"Running Opponent-Adjusted Custom EP optimizer (GW1)...")
        solution_opp = _build_and_solve_model_gw1(
            gameweek_pool, args.budget_cap, ep_column="opp_adj_gus_ep_next",
            bench_gk_min_ppg=args.bench_gk_min_ppg,
            bench_gk_min_total=args.bench_gk_min_total,
            bench_outfield_min_ppg=args.bench_outfield_min_ppg,
            bench_outfield_min_total=args.bench_outfield_min_total,
            bench_outfield_avg_ppg=args.bench_outfield_avg_ppg,
            bench_outfield_avg_total=args.bench_outfield_avg_total,
        )

        print(f"Running Advanced Opponent-Adjusted Custom EP optimizer (GW1)...")
        solution_adv_opp = _build_and_solve_model_gw1(
            gameweek_pool, args.budget_cap, ep_column="adv_opp_adj_gus_ep_next",
            bench_gk_min_ppg=args.bench_gk_min_ppg,
            bench_gk_min_total=args.bench_gk_min_total,
            bench_outfield_min_ppg=args.bench_outfield_min_ppg,
            bench_outfield_min_total=args.bench_outfield_min_total,
            bench_outfield_avg_ppg=args.bench_outfield_avg_ppg,
            bench_outfield_avg_total=args.bench_outfield_avg_total,
        )
    else:
        # GW2+: Transfer optimization
        if not args.prev_squad:
            raise ValueError(f"--prev-squad required for GW{args.gameweek}")

        current_squad = SquadState.load(args.prev_squad)
        print(f"Loaded previous squad from {args.prev_squad}")
        print(f"Bank: {current_squad.bank}M, Free transfers: {current_squad.free_transfers}")

        print(f"Running API EP optimizer (GW{args.gameweek})...")
        solution_api = build_transfer_model(
            gameweek_pool,
            current_squad=current_squad,
            free_transfers=current_squad.free_transfers,
            ep_column="ep_next",
        )

        print(f"Running Custom EP (gus_ep_next) optimizer (GW{args.gameweek})...")
        solution_gus = build_transfer_model(
            gameweek_pool,
            current_squad=current_squad,
            free_transfers=current_squad.free_transfers,
            ep_column="gus_ep_next",
        )

        print(f"Running Opponent-Adjusted Custom EP optimizer (GW{args.gameweek})...")
        solution_opp = build_transfer_model(
            gameweek_pool,
            current_squad=current_squad,
            free_transfers=current_squad.free_transfers,
            ep_column="opp_adj_gus_ep_next",
        )

        print(f"Running Advanced Opponent-Adjusted Custom EP optimizer (GW{args.gameweek})...")
        solution_adv_opp = build_transfer_model(
            gameweek_pool,
            current_squad=current_squad,
            free_transfers=current_squad.free_transfers,
            ep_column="adv_opp_adj_gus_ep_next",
        )

    # Save all four squads
    api_starters = solution_api[solution_api["starter"] == 1].sort_values("element")
    api_bench = solution_api[solution_api["bench"] == 1].sort_values("element")
    api_squad_ids = list(api_starters["element"].values) + list(api_bench["element"].values)
    api_squad = SquadState(player_ids=api_squad_ids, gameweek=args.gameweek, bank=0.0, free_transfers=1)
    api_squad.save(output_dir / f"squad_gw{args.gameweek}_api.json")

    gus_starters = solution_gus[solution_gus["starter"] == 1].sort_values("element")
    gus_bench = solution_gus[solution_gus["bench"] == 1].sort_values("element")
    gus_squad_ids = list(gus_starters["element"].values) + list(gus_bench["element"].values)
    gus_squad = SquadState(player_ids=gus_squad_ids, gameweek=args.gameweek, bank=0.0, free_transfers=1)
    gus_squad.save(output_dir / f"squad_gw{args.gameweek}_gus.json")

    opp_starters = solution_opp[solution_opp["starter"] == 1].sort_values("element")
    opp_bench = solution_opp[solution_opp["bench"] == 1].sort_values("element")
    opp_squad_ids = list(opp_starters["element"].values) + list(opp_bench["element"].values)
    opp_squad = SquadState(player_ids=opp_squad_ids, gameweek=args.gameweek, bank=0.0, free_transfers=1)
    opp_squad.save(output_dir / f"squad_gw{args.gameweek}_opp.json")

    adv_opp_starters = solution_adv_opp[solution_adv_opp["starter"] == 1].sort_values("element")
    adv_opp_bench = solution_adv_opp[solution_adv_opp["bench"] == 1].sort_values("element")
    adv_opp_squad_ids = list(adv_opp_starters["element"].values) + list(adv_opp_bench["element"].values)
    adv_opp_squad = SquadState(player_ids=adv_opp_squad_ids, gameweek=args.gameweek, bank=0.0, free_transfers=1)
    adv_opp_squad.save(output_dir / f"squad_gw{args.gameweek}_adv_opp.json")

    # Generate detailed reports
    _write_team_report(solution_api, args.gameweek, output_dir, suffix="_api", ep_column="ep_next")
    _write_team_report(solution_gus, args.gameweek, output_dir, suffix="_gus", ep_column="gus_ep_next")
    _write_team_report(solution_opp, args.gameweek, output_dir, suffix="_opp", ep_column="opp_adj_gus_ep_next")
    _write_team_report(solution_adv_opp, args.gameweek, output_dir, suffix="_adv_opp", ep_column="adv_opp_adj_gus_ep_next")

    # Generate comparison
    _write_quad_comparison(solution_api, solution_gus, solution_opp, solution_adv_opp, args.gameweek, output_dir)

    # Generate fixture difficulty reference
    _write_fixture_reference(gameweek_pool, args.gameweek, output_dir)

    print(f"✅ Quad optimization complete!")


def _format_section_table(section_df: pd.DataFrame, ep_column: str = "ep_next") -> str:
    """Format player section as table, using specified ep column mapped to ep_next."""
    table_df = section_df.copy()

    # Map the appropriate EP column to "ep_next" for display
    if ep_column == "ep_next":
        # API: already correct
        pass
    elif ep_column == "gus_ep_next":
        # Custom: show gus_ep_next as ep_next
        if "gus_ep_next" in table_df.columns:
            table_df["ep_next"] = table_df["gus_ep_next"]
    elif ep_column == "opp_adj_gus_ep_next":
        # Opponent-adjusted custom: show opp_adj_gus_ep_next as ep_next
        if "opp_adj_gus_ep_next" in table_df.columns:
            table_df["ep_next"] = table_df["opp_adj_gus_ep_next"]
    elif ep_column == "adv_opp_adj_gus_ep_next":
        # Advanced opponent-adjusted custom: show adv_opp_adj_gus_ep_next as ep_next
        if "adv_opp_adj_gus_ep_next" in table_df.columns:
            table_df["ep_next"] = table_df["adv_opp_adj_gus_ep_next"]

    cols = [c for c in REPORT_COLUMNS if c in table_df.columns]
    return table_df[cols].to_string(index=False)


def _write_team_report(
    solution_df: pd.DataFrame, gameweek: int, output_dir: Path, suffix: str = "", ep_column: str = "ep_next"
) -> Path:
    """Generate team selection report using specified ep_column."""
    report_path = output_dir / f"optimal_team_gw{gameweek}{suffix}.txt"

    starters = solution_df[solution_df["starter"] == 1].copy()
    bench = solution_df[solution_df["bench"] == 1].copy()

    starters = starters.sort_values(by=["position_order", "team_name", "web_name"], ascending=[True, True, True])
    bench = bench.sort_values(by=["position_order", "team_name", "web_name"], ascending=[True, True, True])

    starter_xp = starters[ep_column].sum()
    squad_value = solution_df.loc[solution_df["selected"] == 1, "value"].sum()

    suffix_label = " (API)" if suffix == "_api" else " (Custom gus_ep_next)" if suffix == "_gus" else ""

    report = (
        f"FPL Optimiser Result{suffix_label} - GW {gameweek}\n"
        f"{'=' * 60}\n"
        f"Squad Value Used: {squad_value:.1f} / 100.0\n"
        f"Expected Points (Starters): {starter_xp:.2f}\n"
        f"Starters: {len(starters)} | Bench: {len(bench)}\n\n"
        f"STARTING XI\n"
        f"{'-' * 60}\n"
        f"{_format_section_table(starters, ep_column=ep_column)}\n\n"
        f"BENCH\n"
        f"{'-' * 60}\n"
        f"{_format_section_table(bench, ep_column=ep_column)}\n"
    )

    report_path.write_text(report, encoding="utf-8")
    return report_path


def _write_fixture_reference(gameweek_pool: pd.DataFrame, gameweek: int, output_dir: Path) -> Path:
    """Generate a reference list of all teams, opponent difficulties, and fixtures."""
    report_path = output_dir / f"fixture_difficulty_gw{gameweek}.txt"

    # Get unique teams with their difficulty and opponent
    fixture_data = (
        gameweek_pool[["team_name", "opp_team_name", "opp_difficulty"]]
        .drop_duplicates(subset=["team_name"])
        .sort_values("opp_difficulty")
    )

    # Create difficulty labels
    difficulty_labels = {1: "Very Easy", 2: "Easy", 3: "Neutral", 4: "Hard", 5: "Very Hard"}

    report = (
        f"FPL FIXTURE DIFFICULTY REFERENCE - GW{gameweek}\n"
        f"{'=' * 80}\n"
        f"Opponent Difficulty Ratings & Fixtures\n\n"
    )

    # Group by difficulty
    for difficulty in sorted(fixture_data["opp_difficulty"].unique()):
        label = difficulty_labels.get(int(difficulty), "Unknown")
        teams_at_difficulty = fixture_data[fixture_data["opp_difficulty"] == difficulty]

        report += (
            f"{label.upper()} (Difficulty {int(difficulty)})\n"
            f"{'-' * 80}\n"
        )

        for _, row in teams_at_difficulty.iterrows():
            report += f"  {row['team_name']:20} vs {row['opp_team_name']:20}\n"

        report += "\n"

    # Summary
    report += (
        f"DIFFICULTY BREAKDOWN\n"
        f"{'-' * 80}\n"
    )
    for difficulty in [1, 2, 3, 4, 5]:
        label = difficulty_labels[difficulty]
        count = len(fixture_data[fixture_data["opp_difficulty"] == difficulty])
        if count > 0:
            report += f"  {label:15} (Difficulty {difficulty}): {count:2} teams\n"

    report_path.write_text(report, encoding="utf-8")
    print(report)
    return report_path


def _write_triple_comparison(
    solution_api: pd.DataFrame, solution_gus: pd.DataFrame, solution_opp: pd.DataFrame, gameweek: int, output_dir: Path
) -> Path:
    """Generate comparison report between API, custom, and opponent-adjusted EP."""
    report_path = output_dir / f"comparison_gw{gameweek}.txt"

    api_starters = solution_api[solution_api["starter"] == 1].copy()
    gus_starters = solution_gus[solution_gus["starter"] == 1].copy()
    opp_starters = solution_opp[solution_opp["starter"] == 1].copy()

    api_xp = api_starters["ep_next"].sum()
    gus_xp = gus_starters["gus_ep_next"].sum()
    opp_xp = opp_starters["opp_adj_gus_ep_next"].sum()

    # Find all three sets
    api_ids = set(api_starters["element"].values)
    gus_ids = set(gus_starters["element"].values)
    opp_ids = set(opp_starters["element"].values)

    # Common across all three
    all_three = api_ids & gus_ids & opp_ids

    # Pairwise overlaps
    api_gus = (api_ids & gus_ids) - all_three
    api_opp = (api_ids & opp_ids) - all_three
    gus_opp = (gus_ids & opp_ids) - all_three

    # Only in one
    only_api = api_ids - gus_ids - opp_ids
    only_gus = gus_ids - api_ids - opp_ids
    only_opp = opp_ids - api_ids - gus_ids

    report = (
        f"FPL TRIPLE OPTIMIZER COMPARISON - GW{gameweek}\n"
        f"{'=' * 80}\n"
        f"Comparison: API ep_next vs Custom gus_ep_next (total_points/38) vs Opponent-Adjusted ep_next\n\n"
        f"SUMMARY\n"
        f"{'-' * 80}\n"
        f"API Expected Points (Starters):              {api_xp:.2f}\n"
        f"Custom Expected Points (Starters):          {gus_xp:.2f}\n"
        f"Opponent-Adjusted Expected Points (Starters): {opp_xp:.2f}\n\n"
        f"Squad Composition:\n"
        f"  All three agree:        {len(all_three)}/11\n"
        f"  API & Custom only:      {len(api_gus)}\n"
        f"  API & Opponent only:    {len(api_opp)}\n"
        f"  Custom & Opponent only: {len(gus_opp)}\n"
        f"  Only API:               {len(only_api)}\n"
        f"  Only Custom:            {len(only_gus)}\n"
        f"  Only Opponent-Adj:      {len(only_opp)}\n\n"
    )

    if all_three:
        report += (
            f"PLAYERS AGREED BY ALL THREE\n"
            f"{'-' * 80}\n"
        )
        all_three_players = solution_api[solution_api["element"].isin(all_three)].sort_values(by="web_name")
        for _, player in all_three_players.iterrows():
            report += (
                f"  {player['web_name']:20} ({player['position']}) PPG: {player['points_per_game']:.2f} Opp Diff: {int(player['opp_difficulty'])}\n"
                f"    API: {player['ep_next']:.2f}, Custom: {player['gus_ep_next']:.2f}, Opp-Adj: {player['opp_adj_gus_ep_next']:.2f}\n"
            )
        report += "\n"

    if api_gus:
        report += (
            f"PLAYERS IN API & CUSTOM (NOT OPP-ADJ)\n"
            f"{'-' * 80}\n"
        )
        for _, player in solution_api[solution_api["element"].isin(api_gus)].sort_values(by="web_name").iterrows():
            report += (
                f"  {player['web_name']:20} ({player['position']}) PPG: {player['points_per_game']:.2f} Opp Diff: {int(player['opp_difficulty'])}\n"
            )
        report += "\n"

    if api_opp:
        report += (
            f"PLAYERS IN API & OPP-ADJ (NOT CUSTOM)\n"
            f"{'-' * 80}\n"
        )
        for _, player in solution_api[solution_api["element"].isin(api_opp)].sort_values(by="web_name").iterrows():
            report += (
                f"  {player['web_name']:20} ({player['position']}) PPG: {player['points_per_game']:.2f} Opp Diff: {int(player['opp_difficulty'])}\n"
            )
        report += "\n"

    if gus_opp:
        report += (
            f"PLAYERS IN CUSTOM & OPP-ADJ (NOT API)\n"
            f"{'-' * 80}\n"
        )
        for _, player in solution_gus[solution_gus["element"].isin(gus_opp)].sort_values(by="web_name").iterrows():
            report += (
                f"  {player['web_name']:20} ({player['position']}) PPG: {player['points_per_game']:.2f} Opp Diff: {int(player['opp_difficulty'])}\n"
            )
        report += "\n"

    if only_api:
        report += (
            f"PLAYERS ONLY IN API\n"
            f"{'-' * 80}\n"
        )
        for _, player in solution_api[solution_api["element"].isin(only_api)].sort_values(by="web_name").iterrows():
            report += (
                f"  {player['web_name']:20} ({player['position']}) PPG: {player['points_per_game']:.2f} Opp Diff: {int(player['opp_difficulty'])}\n"
            )
        report += "\n"

    if only_gus:
        report += (
            f"PLAYERS ONLY IN CUSTOM\n"
            f"{'-' * 80}\n"
        )
        for _, player in solution_gus[solution_gus["element"].isin(only_gus)].sort_values(by="web_name").iterrows():
            report += (
                f"  {player['web_name']:20} ({player['position']}) PPG: {player['points_per_game']:.2f} Opp Diff: {int(player['opp_difficulty'])}\n"
            )
        report += "\n"

    if only_opp:
        report += (
            f"PLAYERS ONLY IN OPPONENT-ADJUSTED\n"
            f"{'-' * 80}\n"
        )
        for _, player in solution_opp[solution_opp["element"].isin(only_opp)].sort_values(by="web_name").iterrows():
            report += (
                f"  {player['web_name']:20} ({player['position']}) PPG: {player['points_per_game']:.2f} Opp Diff: {int(player['opp_difficulty'])}\n"
            )
        report += "\n"

    report_path.write_text(report, encoding="utf-8")
    print(report)
    return report_path





def _write_quad_comparison(
    solution_api: pd.DataFrame,
    solution_gus: pd.DataFrame,
    solution_opp: pd.DataFrame,
    solution_adv_opp: pd.DataFrame,
    gameweek: int,
    output_dir: Path,
) -> Path:
    report_path = output_dir / f"comparison_gw{gameweek}.txt"

    api_starters = solution_api[solution_api["starter"] == 1].copy()
    gus_starters = solution_gus[solution_gus["starter"] == 1].copy()
    opp_starters = solution_opp[solution_opp["starter"] == 1].copy()
    adv_opp_starters = solution_adv_opp[solution_adv_opp["starter"] == 1].copy()

    api_xp = api_starters["ep_next"].sum()
    gus_xp = gus_starters["gus_ep_next"].sum()
    opp_xp = opp_starters["opp_adj_gus_ep_next"].sum()
    adv_opp_xp = adv_opp_starters["adv_opp_adj_gus_ep_next"].sum()

    api_ids = set(api_starters["element"].values)
    gus_ids = set(gus_starters["element"].values)
    opp_ids = set(opp_starters["element"].values)
    adv_opp_ids = set(adv_opp_starters["element"].values)

    all_four = api_ids & gus_ids & opp_ids & adv_opp_ids

    report = f"FPL QUAD OPTIMIZER COMPARISON - GW{gameweek}\n"
    report += f"{'=' * 80}\n"
    report += f"Comparison: API vs Custom vs Opponent-Adjusted vs Advanced Opponent-Adjusted\n\n"
    report += f"SUMMARY\n{'-' * 80}\n"
    report += f"API Expected Points (Starters):                         {api_xp:.2f}\n"
    report += f"Custom Expected Points (Starters):                      {gus_xp:.2f}\n"
    report += f"Opponent-Adjusted Expected Points (Starters):           {opp_xp:.2f}\n"
    report += f"Advanced Opponent-Adjusted Expected Points (Starters):  {adv_opp_xp:.2f}\n\n"
    report += f"Unique starters: {len(api_ids | gus_ids | opp_ids | adv_opp_ids)} total\n"
    report += f"Agreed by all four: {len(all_four)}\n\n"

    if all_four:
        report += f"PLAYERS AGREED BY ALL FOUR\n{'-' * 80}\n"
        all_four_players = solution_api[solution_api["element"].isin(all_four)].sort_values(by="web_name")
        for _, player in all_four_players.iterrows():
            report += f"  {player['web_name']:20} ({player['position']})\n"
            report += f"    API: {player['ep_next']:.2f}, Custom: {player['gus_ep_next']:.2f}, Opp-Adj: {player['opp_adj_gus_ep_next']:.2f}, Adv-Opp: {player['adv_opp_adj_gus_ep_next']:.2f}\n"
        report += "\n"

    only_api = api_ids - gus_ids - opp_ids - adv_opp_ids
    only_gus = gus_ids - api_ids - opp_ids - adv_opp_ids
    only_opp = opp_ids - api_ids - gus_ids - adv_opp_ids
    only_adv = adv_opp_ids - api_ids - gus_ids - opp_ids

    if only_api:
        names = ", ".join([solution_api[solution_api["element"] == eid]["web_name"].values[0] for eid in only_api])
        report += f"ONLY IN API: {names}\n\n"

    if only_gus:
        names = ", ".join([solution_gus[solution_gus["element"] == eid]["web_name"].values[0] for eid in only_gus])
        report += f"ONLY IN CUSTOM: {names}\n\n"

    if only_opp:
        names = ", ".join([solution_opp[solution_opp["element"] == eid]["web_name"].values[0] for eid in only_opp])
        report += f"ONLY IN OPP-ADJUSTED: {names}\n\n"

    if only_adv:
        names = ", ".join([solution_adv_opp[solution_adv_opp["element"] == eid]["web_name"].values[0] for eid in only_adv])
        report += f"ONLY IN ADVANCED OPP-ADJUSTED: {names}\n\n"

    report_path.write_text(report, encoding="utf-8")
    print(report)
    return report_path



if __name__ == "__main__":
    main()
