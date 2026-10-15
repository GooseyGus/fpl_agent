from __future__ import annotations

import pandas as pd
from pulp import LpProblem, LpVariable, lpSum


def add_force_bench_constraint(
    problem: LpProblem,
    gameweek_pool: pd.DataFrame,
    selected_vars: dict[int, LpVariable],
    starter_vars: dict[int, LpVariable],
    player_names: list[str],
) -> None:
    """Force specified players onto the bench (selected but not starting).

    Args:
        problem: The LP problem
        gameweek_pool: The player dataframe
        selected_vars: Dictionary of selection variables
        starter_vars: Dictionary of starter variables
        player_names: List of player names (web_name) to force onto bench
    """
    if not player_names:
        return

    for player_name in player_names:
        # Find player(s) matching the name
        matching_players = gameweek_pool[gameweek_pool["web_name"].str.lower() == player_name.lower()]

        if matching_players.empty:
            print(f"  ⚠️  Player '{player_name}' not found in pool")
            continue

        for idx in matching_players.index:
            # Force: selected = 1, starter = 0 (so bench = 1)
            problem += selected_vars[idx] == 1, f"force_bench_selected_{idx}"
            problem += starter_vars[idx] == 0, f"force_bench_not_starter_{idx}"
            print(f"  ✓ Forcing {player_name} onto bench")
