from __future__ import annotations

import pandas as pd
from pulp import LpProblem, LpVariable, lpSum

# Bench quality constraint defaults (relaxed by 15%)
GK_MIN_PPG = 3.9 * 0.85  # 3.315
GK_MIN_TOTAL_POINTS = int(100 * 0.85)  # 85
OUTFIELD_MIN_PPG = 3.0 * 0.85  # 2.55
OUTFIELD_MIN_TOTAL_POINTS = int(80 * 0.85)  # 68
OUTFIELD_MIN_AVG_PPG = 4.3 * 0.85  # 3.655
OUTFIELD_MIN_AVG_TOTAL = int(135 * 0.85)  # 114.75


def add_gw1_bench_quality_constraint(
    problem: LpProblem,
    gameweek_pool: pd.DataFrame,
    selected_vars: dict[int, LpVariable],
    starter_vars: dict[int, LpVariable],
    gk_min_ppg: float = GK_MIN_PPG,
    gk_min_total_points: int = GK_MIN_TOTAL_POINTS,
    outfield_min_ppg: float = OUTFIELD_MIN_PPG,
    outfield_min_total_points: int = OUTFIELD_MIN_TOTAL_POINTS,
    outfield_min_avg_ppg: float = OUTFIELD_MIN_AVG_PPG,
    outfield_min_avg_total: int = OUTFIELD_MIN_AVG_TOTAL,
) -> None:
    """Ensure bench meets quality thresholds.

    Bench composition (1 GK, 3 outfield):
    - GK: must meet individual min_ppg and min_total_points
    - 3 Outfield: each player must meet individual outfield_min_ppg and outfield_min_total_points
    - 3 Outfield: collectively must average at least outfield_min_avg_ppg and outfield_min_avg_total
      (i.e., sum of 3 players / 3 >= threshold, or sum >= threshold * 3)
    """
    gameweek_pool = gameweek_pool.copy()
    gameweek_pool["points_per_game"] = pd.to_numeric(
        gameweek_pool["points_per_game"], errors="coerce"
    ).fillna(0.0)
    gameweek_pool["total_points"] = pd.to_numeric(
        gameweek_pool["total_points"], errors="coerce"
    ).fillna(0.0)

    # Separate GK and outfield players
    gk_players = []
    outfield_players = []

    for idx in gameweek_pool.index:
        position = gameweek_pool.loc[idx, "position"]
        ppg = gameweek_pool.loc[idx, "points_per_game"]
        total_points = gameweek_pool.loc[idx, "total_points"]
        bench_var = selected_vars[idx] - starter_vars[idx]

        if position == "GK":
            gk_players.append((idx, ppg, total_points, bench_var))
        else:
            outfield_players.append((idx, ppg, total_points, bench_var))

    # GK constraint: individual minimums
    for idx, ppg, total_points, bench_var in gk_players:
        # If on bench, must meet individual thresholds
        if ppg < gk_min_ppg or total_points < gk_min_total_points:  # type: ignore
            # Can't be on bench: selected - starter <= 0
            problem += (
                selected_vars[idx] - starter_vars[idx] <= 0
            ), f"gk_bench_quality_{idx}"

    # Outfield constraint: individual minimums for each bench player
    for idx, ppg, total_points, bench_var in outfield_players:
        # If on bench, must meet individual thresholds
        if ppg < outfield_min_ppg or total_points < outfield_min_total_points:  # type: ignore
            # Can't be on bench: selected - starter <= 0
            problem += (
                selected_vars[idx] - starter_vars[idx] <= 0
            ), f"outfield_bench_quality_{idx}"

    # Outfield constraint: average minimums for the 3 bench players
    if outfield_players:
        # Sum of PPG for all outfield bench players >= avg_ppg * 3
        outfield_ppg_sum = lpSum(
            ppg * bench_var for _, ppg, _, bench_var in outfield_players
        )
        outfield_count = lpSum(
            bench_var for _, _, _, bench_var in outfield_players
        )
        # When 3 outfield are on bench: sum >= outfield_min_avg_ppg * 3
        problem += (
            outfield_ppg_sum >= outfield_min_avg_ppg * outfield_count
        ), "outfield_bench_avg_ppg"

        # Sum of total_points for outfield bench >= avg_total * 3
        outfield_total_sum = lpSum(
            total_points * bench_var for _, _, total_points, bench_var in outfield_players
        )
        problem += (
            outfield_total_sum >= outfield_min_avg_total * outfield_count
        ), "outfield_bench_avg_total"
