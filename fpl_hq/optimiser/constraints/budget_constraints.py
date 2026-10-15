from __future__ import annotations

import pandas as pd
from pulp import LpProblem, LpVariable, lpSum


def add_budget_constraint(
    problem: LpProblem,
    gameweek_pool: pd.DataFrame,
    selected_vars: dict[int, LpVariable],
    budget_cap: float = 100.0,
    value_scale: float = 1.0,
) -> None:
    problem += (
        lpSum(
            (float(gameweek_pool.loc[idx, "value"]) / value_scale) * selected_vars[idx]
            for idx in gameweek_pool.index
        )
        <= budget_cap
    ), "budget_cap"