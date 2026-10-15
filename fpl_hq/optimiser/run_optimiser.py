from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd
from pulp import LpMaximize, LpProblem, LpStatus, LpVariable, PULP_CBC_CMD, lpSum, value

from fpl_hq.optimiser.constraints import (
    add_budget_constraint,
    add_gw1_bench_quality_constraint,
    add_max_3_from_same_team_constraint,
    add_squad_position_constraints,
    add_starter_position_constraints,
    add_team_size_constraints,
)
from fpl_hq.optimiser.prepare_data import load_and_prepare_gw_data
from fpl_hq.optimiser.objective_function import build_starter_xp_objective
from fpl_hq.optimiser.transfer_optimizer import build_transfer_model

from fpl_hq.optimiser.squad_state import SquadState
from fpl_hq.paths import OPTIMISER_OUTPUT_DIR, SEASON_DATA_DIR

POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
REPORT_COLUMNS = [
    "web_name",
    "position",
    "team",
    "ep_next",
    "now_cost",
    "form",
    "points_per_game",
    "total_points",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FPL GW optimiser.")
    parser.add_argument("--gameweek", type=int, default=1, help="Gameweek to optimise.")
    parser.add_argument(
        "--budget-cap",
        type=float,
        default=100.0,
        help="Budget cap in millions (default: 100.0).",
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2025-26",
        help="Season folder under data/vaastav_data.",
    )
    parser.add_argument(
        "--ep-column",
        type=str,
        default="ep_next",
        help="EP column to use for optimization.",
    )
    parser.add_argument(
        "--use-enriched",
        action="store_true",
        help="Use enriched data file.",
    )
    parser.add_argument(
        "--prev-squad",
        type=Path,
        default=None,
        help="Path to previous squad JSON file (required for GW > 1).",
    )
    parser.add_argument(
        "--no-open-report",
        action="store_true",
        help="Do not auto-open the generated report file.",
    )
    return parser.parse_args()


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
        column: "first" for column in gameweek_pool.columns 
        if column not in exclude_cols and column != "id"
    }
    
    aggregation = {
        **first_columns,
    }
    
    # Add sum aggregations for numeric EP columns
    for col in gameweek_pool.columns:
        if col in aggregation or col == "id":
            continue
        if col in {"ep_next", "ep_lag1_opp", "ep_lag3_opp", "ep_lag5_opp", "ep_lag10_opp", "ep_lag5_advopp"}:
            aggregation[col] = "sum"
        elif col == "fixture" and col in gameweek_pool.columns:
            aggregation[col] = lambda values: ",".join(str(v) for v in sorted(set(values.dropna())))
    
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
        gameweek_pool["value"] = gameweek_pool["now_cost"] / 10  # Convert to millions

    if "ep_next" in gameweek_pool.columns and "xP" not in gameweek_pool.columns:
        gameweek_pool["xP"] = gameweek_pool["ep_next"]

    return gameweek_pool


def _load_gameweek_pool(gameweek: int, season: str) -> pd.DataFrame:
    # For GW1, always use GW0 data (which has ep_next predictions for GW1)
    if gameweek == 1:
        gameweek_pool = _load_gameweek_pool_from_csv(0)
        if gameweek_pool is not None:
            return gameweek_pool

    # Try loading from 2026-27 data first
    gameweek_pool = _load_gameweek_pool_from_csv(gameweek)

    # If GW doesn't exist yet, try loading GW0 (team selection with ep_next for next GW)
    if gameweek_pool is None and gameweek > 1:
        print(f"  GW{gameweek} hasn't been played yet. Using GW0 (ep_next data) to optimize for GW{gameweek}...")
        gameweek_pool = _load_gameweek_pool_from_csv(0)

    # Fall back to legacy data if 2026-27 data still not available
    if gameweek_pool is None:
        print(f"  No 2026-27 data found. Falling back to {season} legacy data...")
        data = load_and_prepare_gw_data(season=season)
        gameweek_pool = data[data["round"] == gameweek].copy()

    if gameweek_pool is None or gameweek_pool.empty:
        raise ValueError(f"No rows found for gameweek {gameweek}.")
    gameweek_pool = _coerce_numeric_columns(gameweek_pool)
    gameweek_pool = _reduce_to_one_row_per_player(gameweek_pool)
    return gameweek_pool.reset_index(drop=True)


def _build_and_solve_model(gameweek_pool: pd.DataFrame, budget_cap: float, gameweek: int = 1) -> pd.DataFrame:
    gameweek_pool = gameweek_pool.copy()
    gameweek_pool["position_order"] = gameweek_pool["position"].map(POSITION_ORDER)

    model = LpProblem("fpl_team_selection", LpMaximize)
    selected_vars = {
        idx: LpVariable(f"selected_{idx}", lowBound=0, upBound=1, cat="Binary")
        for idx in gameweek_pool.index
    }
    starter_vars = {
        idx: LpVariable(f"starter_{idx}", lowBound=0, upBound=1, cat="Binary")
        for idx in gameweek_pool.index
    }

    model += build_starter_xp_objective(gameweek_pool, starter_vars)

    add_budget_constraint(model, gameweek_pool, selected_vars, budget_cap=budget_cap)
    add_max_3_from_same_team_constraint(model, gameweek_pool, selected_vars)
    add_squad_position_constraints(model, gameweek_pool, selected_vars)
    add_starter_position_constraints(model, gameweek_pool, starter_vars)
    add_team_size_constraints(model, selected_vars, starter_vars)

    if gameweek == 1:
        # Apply relaxed bench quality constraint (relaxed by 15%)
        add_gw1_bench_quality_constraint(model, gameweek_pool, selected_vars, starter_vars)

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

    # Set first starter as captain by default (can be changed manually)
    gameweek_pool["captain"] = 0
    first_starter_idx = gameweek_pool[gameweek_pool["starter"] == 1].index[0]
    gameweek_pool.loc[first_starter_idx, "captain"] = 1

    # Add transfer state for GW1 (all selected are "initial")
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


def _format_section_table(section_df: pd.DataFrame, include_transfer: bool = False) -> str:
    table_df = section_df.copy()
    table_df["value"] = table_df["value"] / 10.0

    # Add captain marker
    if "captain" in table_df.columns:
        table_df["web_name"] = table_df.apply(
            lambda row: f"{row['web_name']} (C)" if row.get("captain") == 1 else row["web_name"],
            axis=1
        )

    if include_transfer and "transfer_state" in table_df.columns:
        cols = REPORT_COLUMNS + ["transfer_state"]
    else:
        cols = REPORT_COLUMNS

    # Only include columns that exist
    cols = [c for c in cols if c in table_df.columns]
    import sys
    return table_df[cols].to_string(index=False)


def _write_report(
    solution_df: pd.DataFrame, gameweek: int, budget_cap: float, output_dir: Path
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"optimal_team_gw{gameweek}.txt"

    starters = solution_df[solution_df["starter"] == 1].copy()
    bench = solution_df[solution_df["bench"] == 1].copy()

    starters = starters.sort_values(by=["position_order", "xP"], ascending=[True, False])
    bench = bench.sort_values(by=["position_order", "xP"], ascending=[True, False])

    squad_value = (solution_df.loc[solution_df["selected"] == 1, "value"].sum())
    starter_xp = starters["xP"].sum()

    # Extract transfer information
    transfers_in = solution_df[
        solution_df["transfer_state"].str.contains("IN", na=False)
    ].sort_values(by=["position_order"])

    transfers_out = solution_df[
        solution_df["transfer_state"].str.contains("OUT", na=False)
    ].sort_values(by=["position_order"])

    report = (
        f"FPL Optimiser Result - GW {gameweek}\n"
        f"{'=' * 60}\n"
        f"Squad Value Used: {squad_value:.1f} / {budget_cap:.1f}\n"
        f"Expected Points (Starters): {starter_xp:.2f}\n"
        f"Starters: {len(starters)} | Bench: {len(bench)}\n\n"
    )

    # Transfer summary
    if not transfers_out.empty or not transfers_in.empty:
        report += (
            f"TRANSFERS\n"
            f"{'-' * 60}\n"
        )

        # Pair transfers: out → in
        num_transfers = max(len(transfers_out), len(transfers_in))
        for i in range(num_transfers):
            out_name = transfers_out['web_name'].iloc[i] if i < len(transfers_out) else "-"
            in_name = transfers_in['web_name'].iloc[i] if i < len(transfers_in) else "-"

            if out_name != "-" and in_name != "-":
                report += f"  {out_name} → {in_name}\n"
            elif out_name != "-":
                report += f"  OUT: {out_name}\n"
            elif in_name != "-":
                report += f"  IN: {in_name}\n"

        report += "\n"

    report += (
        f"STARTING XI\n"
        f"{'-' * 60}\n"
        f"{_format_section_table(starters, include_transfer=True)}\n\n"
        f"BENCH\n"
        f"{'-' * 60}\n"
        f"{_format_section_table(bench, include_transfer=True)}\n"
    )

    report_path.write_text(report, encoding="utf-8")
    return report_path


def _open_report(report_path: Path) -> None:
    system_name = platform.system()
    if system_name == "Darwin":
        subprocess.run(["open", str(report_path)], check=True)
        return
    if system_name == "Windows":
        subprocess.run(["cmd", "/c", "start", "", str(report_path)], check=True)
        return
    subprocess.run(["xdg-open", str(report_path)], check=True)



def _simple_squad_selection(gameweek_pool: pd.DataFrame) -> pd.DataFrame:
    """Fallback: select best 15 players with basic constraints."""
    gameweek_pool = gameweek_pool.copy()
    gameweek_pool['value'] = gameweek_pool['now_cost'] / 10
    gameweek_pool['ep_next'] = pd.to_numeric(gameweek_pool['ep_next'], errors='coerce').fillna(0)
    
    selected = []
    gk_count = def_count = mid_count = fwd_count = 0
    
    for _, player in gameweek_pool.nlargest(100, 'ep_next').iterrows():
        pos = player['element_type']
        if pos == 1 and gk_count < 2:
            selected.append(player.name)
            gk_count += 1
        elif pos == 2 and def_count < 5:
            selected.append(player.name)
            def_count += 1
        elif pos == 3 and mid_count < 5:
            selected.append(player.name)
            mid_count += 1
        elif pos == 4 and fwd_count < 3:
            selected.append(player.name)
            fwd_count += 1
        
        if len(selected) == 15:
            break
    
    selected_data = gameweek_pool.loc[selected].copy()
    selected_data['selected'] = 1
    selected_data['starter'] = 1
    selected_data['bench'] = 0
    selected_data['captain'] = 0
    selected_data['transfer_state'] = 'FALLBACK'
    
    # Add position_order for sorting later
    position_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    element_type_to_position = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    selected_data['position'] = selected_data['element_type'].map(element_type_to_position)
    selected_data['position_order'] = selected_data['position'].map(position_order)
    
    first_idx = selected_data.index[0]
    selected_data.loc[first_idx, 'captain'] = 1
    
    return selected_data


def main() -> None:
    args = _parse_args()
    gameweek_pool = _load_gameweek_pool(gameweek=args.gameweek, season=args.season)

    # Handle GW1 vs GW > 1
    if args.gameweek == 1:
        # GW1: unlimited transfers, use standard optimizer with bench quality constraint
        solution_df = _build_and_solve_model(gameweek_pool, budget_cap=args.budget_cap, gameweek=args.gameweek)
        if solution_df is None:
            raise RuntimeError(f"Failed to find optimal solution for GW{args.gameweek}")
        starting_xi = solution_df[solution_df["starter"] == 1].sort_values("element")
        bench = solution_df[solution_df["bench"] == 1].sort_values("element")
    else:
        # GW > 1: load previous squad and use transfer optimizer
        if not args.prev_squad:
            raise ValueError(f"--prev-squad required for GW{args.gameweek}")

        current_squad = SquadState.load(args.prev_squad)
        print(f"  Loaded previous squad from {args.prev_squad}")
        print(f"  Bank: {current_squad.bank}M, Free transfers: {current_squad.free_transfers}")

        solution_df = build_transfer_model(
            gameweek_pool,
            current_squad=current_squad,
            budget_cap=args.budget_cap,
            free_transfers=current_squad.free_transfers,
        )
        if solution_df is None:
            print(f"  ⚠️  Transfer optimizer failed, using fallback selection...")
            solution_df = _simple_squad_selection(gameweek_pool)
        starting_xi = solution_df[solution_df["starter"] == 1].sort_values("element")
        bench = solution_df[solution_df["bench"] == 1].sort_values("element")

    # Save squad for next gameweek
    squad_ids = list(starting_xi["element"].values) + list(bench["element"].values)
    next_squad = SquadState(
        player_ids=squad_ids,
        gameweek=args.gameweek,
        bank=0.0,  # TODO: calculate remaining bank from transfers
        free_transfers=1,  # Reset to 1 for next GW
    )

    squad_file = OPTIMISER_OUTPUT_DIR / f"squad_gw{args.gameweek}.json"
    next_squad.save(squad_file)

    # Write report
    output_dir = OPTIMISER_OUTPUT_DIR
    report_path = _write_report(
        solution_df=solution_df,
        gameweek=args.gameweek,
        budget_cap=args.budget_cap,
        output_dir=output_dir,
    )

    print(f"✅ Optimiser complete. Report: {report_path}")
    print(f"💾 Squad saved to: {squad_file}")
    if args.no_open_report:
        return
    _open_report(report_path)


if __name__ == "__main__":
    main()