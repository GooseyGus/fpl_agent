"""Select a 15-player FPL squad by maximising consensus scores."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests
from pulp import LpMaximize, LpProblem, LpStatus, LpVariable, PULP_CBC_CMD, lpSum

from fpl_hq.paths import OPTIMISER_OUTPUT_DIR, gw_report_dir

from fpl_hq.optimiser.constraints import (
    add_budget_constraint,
    add_max_3_from_same_team_constraint,
    add_squad_position_constraints,
)

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
POSITION_BY_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
DEFAULT_INPUT = gw_report_dir(4) / "gw_4_consensus.csv"
DEFAULT_OUTPUT = OPTIMISER_OUTPUT_DIR / "consensus_squad.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select an FPL squad that maximises total consensus score."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Consensus CSV containing player_id and Consensus (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination for the selected 15 players (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--budget-cap",
        type=float,
        default=100,
        help="Squad budget in millions (default: 100.0).",
    )
    return parser.parse_args()


def _load_consensus(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Consensus CSV not found: {path}")

    consensus = pd.read_csv(path)
    required_columns = {"player_id", "Consensus"}
    missing_columns = required_columns.difference(consensus.columns)
    if missing_columns:
        raise ValueError(
            f"Consensus CSV is missing columns: {', '.join(sorted(missing_columns))}"
        )

    consensus = consensus[["player_id", "Consensus"]].copy()
    consensus["player_id"] = pd.to_numeric(consensus["player_id"], errors="raise").astype(int)
    consensus["Consensus"] = pd.to_numeric(consensus["Consensus"], errors="raise")

    duplicate_ids = consensus.loc[consensus["player_id"].duplicated(), "player_id"].tolist()
    if duplicate_ids:
        raise ValueError(f"Duplicate player_id values in consensus CSV: {duplicate_ids}")
    if consensus["Consensus"].isna().any():
        raise ValueError("Consensus scores cannot be blank.")

    return consensus


def _fetch_player_metadata() -> pd.DataFrame:
    response = requests.get(FPL_BOOTSTRAP_URL, timeout=30)
    response.raise_for_status()
    bootstrap = response.json()

    teams = {team["id"]: team["name"] for team in bootstrap["teams"]}
    players = pd.DataFrame(bootstrap["elements"])
    players = players[
        ["id", "first_name", "second_name", "web_name", "element_type", "team", "now_cost"]
    ].copy()
    players["position"] = players["element_type"].map(POSITION_BY_TYPE)
    players["team_name"] = players["team"].map(teams)
    players["price"] = players["now_cost"] / 10.0
    players["full_name"] = (
        players["first_name"].fillna("").str.strip()
        + " "
        + players["second_name"].fillna("").str.strip()
    ).str.strip()

    return players.rename(columns={"id": "player_id"})


def _build_player_pool(consensus: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    pool = consensus.merge(players, on="player_id", how="left", validate="one_to_one")
    missing_ids = pool.loc[pool["web_name"].isna(), "player_id"].tolist()
    if missing_ids:
        missing_text = ", ".join(str(player_id) for player_id in missing_ids)
        raise ValueError(f"Player IDs not found in the current FPL API: {missing_text}")

    if pool["position"].isna().any():
        raise ValueError("The FPL API returned an unknown player position.")

    # Names expected by the repository's shared squad constraints.
    pool["value"] = pool["price"]
    return pool.reset_index(drop=True)


def _optimise_squad(pool: pd.DataFrame, budget_cap: float) -> pd.DataFrame:
    model = LpProblem("consensus_squad_selection", LpMaximize)
    row_indices = list(range(len(pool)))
    consensus_scores: dict[int, float] = {
        idx: score
        for idx, score in enumerate(pool["Consensus"].astype(float).tolist())
    }
    selected: dict[int, LpVariable] = {
        idx: LpVariable(f"selected_{idx}", lowBound=0, upBound=1, cat="Binary")
        for idx in row_indices
    }

    model += lpSum(
        consensus_scores[idx] * selected[idx] for idx in row_indices
    ), "total_consensus"
    model += lpSum(selected.values()) == 15, "total_squad_size"
    add_budget_constraint(model, pool, selected, budget_cap=budget_cap)
    add_max_3_from_same_team_constraint(model, pool, selected)
    add_squad_position_constraints(model, pool, selected)

    solve_status = model.solve(PULP_CBC_CMD(msg=False))
    status_name = LpStatus[solve_status]
    if status_name != "Optimal":
        raise RuntimeError(
            "Consensus optimiser failed to find an optimal 15-player squad. "
            f"Status: {status_name}"
        )

    selected_indices = [
        idx for idx in row_indices if (selected[idx].varValue or 0.0) > 0.5
    ]
    squad = pool.loc[selected_indices].copy()
    squad["position_order"] = squad["position"].map(POSITION_ORDER)
    squad = squad.sort_values(
        ["position_order", "Consensus", "price", "web_name"],
        ascending=[True, False, True, True],
    )
    return squad.reset_index(drop=True)


def _write_output(squad: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "player_id",
        "web_name",
        "full_name",
        "position",
        "team_name",
        "price",
        "Consensus",
    ]
    squad[output_columns].to_csv(output_path, index=False, float_format="%.2f")


def main() -> None:
    args = _parse_args()
    consensus = _load_consensus(args.input)
    players = _fetch_player_metadata()
    pool = _build_player_pool(consensus, players)
    squad = _optimise_squad(pool, budget_cap=args.budget_cap)
    _write_output(squad, args.output)

    display_columns = ["web_name", "position", "team_name", "price", "Consensus"]
    print(squad[display_columns].to_string(index=False))
    print(f"\nPlayers: {len(squad)}")
    print(f"Total price: £{squad['price'].sum():.1f}m / £{args.budget_cap:.1f}m")
    print(f"Total consensus: {squad['Consensus'].sum():.2f}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
