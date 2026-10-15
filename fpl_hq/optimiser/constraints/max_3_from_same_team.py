from __future__ import annotations

import pandas as pd
from pulp import LpProblem, LpVariable, lpSum


def add_max_3_from_same_team_constraint(
    problem: LpProblem,
    gameweek_pool: pd.DataFrame,
    selected_vars: dict[int, LpVariable],
) -> None:
    for team_name, team_rows in gameweek_pool.groupby("team").groups.items():
        problem += (
            lpSum(selected_vars[idx] for idx in team_rows) <= 3
        ), f"max_3_from_{team_name}"