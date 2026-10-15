from __future__ import annotations

import pandas as pd
from pulp import LpProblem, LpVariable, lpSum

SQUAD_POSITION_COUNTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTER_POSITION_MIN = {"GK": 1, "DEF": 3, "MID": 3, "FWD": 1}
STARTER_POSITION_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}


def add_squad_position_constraints(
    problem: LpProblem,
    gameweek_pool: pd.DataFrame,
    selected_vars: dict[int, LpVariable],
) -> None:
    for position, required_count in SQUAD_POSITION_COUNTS.items():
        position_rows = gameweek_pool.index[gameweek_pool["position"] == position]
        problem += (
            lpSum(selected_vars[idx] for idx in position_rows) == required_count
        ), f"squad_{position.lower()}_count"


def add_starter_position_constraints(
    problem: LpProblem,
    gameweek_pool: pd.DataFrame,
    starter_vars: dict[int, LpVariable],
) -> None:
    for position, min_count in STARTER_POSITION_MIN.items():
        position_rows = gameweek_pool.index[gameweek_pool["position"] == position]
        problem += (
            lpSum(starter_vars[idx] for idx in position_rows) >= min_count
        ), f"starter_{position.lower()}_min"

    for position, max_count in STARTER_POSITION_MAX.items():
        position_rows = gameweek_pool.index[gameweek_pool["position"] == position]
        problem += (
            lpSum(starter_vars[idx] for idx in position_rows) <= max_count
        ), f"starter_{position.lower()}_max"