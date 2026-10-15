from __future__ import annotations

from pulp import LpProblem, LpVariable, lpSum


def add_team_size_constraints(
    problem: LpProblem,
    selected_vars: dict[int, LpVariable],
    starter_vars: dict[int, LpVariable],
    total_squad_size: int = 15,
    starter_size: int = 11,
) -> None:
    problem += lpSum(selected_vars.values()) == total_squad_size, "total_squad_size"
    problem += lpSum(starter_vars.values()) == starter_size, "starter_size"