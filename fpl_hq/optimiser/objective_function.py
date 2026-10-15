from __future__ import annotations

import pandas as pd
from pulp import LpVariable, lpSum


def build_starter_xp_objective(
    gameweek_pool: pd.DataFrame, starter_vars: dict[int, LpVariable]
):
    # Use ep_next (expected points for next gameweek) if available, otherwise xP
    ep_column = "ep_next" if "ep_next" in gameweek_pool.columns else "xP"
    return lpSum(
        float(gameweek_pool.loc[idx, ep_column]) * starter_vars[idx]
        for idx in gameweek_pool.index
    )