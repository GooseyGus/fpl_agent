"""Transfer optimization for gameweeks after GW1 - following the working notebook approach."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, PULP_CBC_CMD, lpSum, LpBinary, LpStatus

from fpl_hq.optimiser.squad_state import SquadState


class TransferConstraints:
    """Handles all constraint logic for transfer optimization."""

    def get_decision_variables(self, df: pd.DataFrame) -> dict:
        """Create all transfer decision variables."""
        return {
            "transfer_in_starting_free": LpVariable.dicts("transfer_in_starting_free", df.index, 0, 1, LpBinary),
            "transfer_in_starting_paid": LpVariable.dicts("transfer_in_starting_paid", df.index, 0, 1, LpBinary),
            "transfer_out_starting_free": LpVariable.dicts("transfer_out_starting_free", df.index, 0, 1, LpBinary),
            "transfer_out_starting_paid": LpVariable.dicts("transfer_out_starting_paid", df.index, 0, 1, LpBinary),
            "stay_starting": LpVariable.dicts("stay_starting", df.index, 0, 1, LpBinary),
            "transfer_in_bench_free": LpVariable.dicts("transfer_in_bench_free", df.index, 0, 1, LpBinary),
            "transfer_in_bench_paid": LpVariable.dicts("transfer_in_bench_paid", df.index, 0, 1, LpBinary),
            "transfer_out_bench_free": LpVariable.dicts("transfer_out_bench_free", df.index, 0, 1, LpBinary),
            "transfer_out_bench_paid": LpVariable.dicts("transfer_out_bench_paid", df.index, 0, 1, LpBinary),
            "stay_bench": LpVariable.dicts("stay_bench", df.index, 0, 1, LpBinary),
            "bench_to_starting": LpVariable.dicts("bench_to_starting", df.index, 0, 1, LpBinary),
            "starting_to_bench": LpVariable.dicts("starting_to_bench", df.index, 0, 1, LpBinary),
        }

    def set_objective_function(self, df: pd.DataFrame, decision_variables: dict, prob: LpProblem, ep_column: str = "ep_next") -> None:
        """Maximize xP from starting players only."""
        prob += lpSum(
            decision_variables["transfer_in_starting_free"][i] * df.loc[i, ep_column]
            + decision_variables["transfer_in_starting_paid"][i] * (df.loc[i, ep_column] - 4)
            + decision_variables["stay_starting"][i] * df.loc[i, ep_column]
            + decision_variables["bench_to_starting"][i] * df.loc[i, ep_column]
            - decision_variables["transfer_in_bench_paid"][i] * 4
            for i in df.index
        ), "TotalExpectedPoints"

    def apply_formation_constraints(self, df: pd.DataFrame, decision_variables: dict, prob: LpProblem) -> None:
        """Apply squad size and position constraints."""
        # Squad size: 11 starters, 4 bench
        prob += (
            lpSum(
                decision_variables["transfer_in_starting_free"][i]
                + decision_variables["transfer_in_starting_paid"][i]
                + decision_variables["stay_starting"][i]
                + decision_variables["bench_to_starting"][i]
                for i in df.index
            )
            == 11
        ), "11Starters"

        prob += (
            lpSum(
                decision_variables["transfer_in_bench_free"][i]
                + decision_variables["transfer_in_bench_paid"][i]
                + decision_variables["stay_bench"][i]
                + decision_variables["starting_to_bench"][i]
                for i in df.index
            )
            == 4
        ), "4Bench"

        # Position constraints
        positions = ["GK", "DEF", "MID", "FWD"]
        min_starting = [1, 3, 3, 1]
        max_starting = [1, 5, 5, 3]
        total = [2, 5, 5, 3]

        for pos, min_start, max_start, tot in zip(positions, min_starting, max_starting, total):
            # Minimum starters per position
            prob += (
                lpSum(
                    decision_variables["transfer_in_starting_free"][i]
                    + decision_variables["transfer_in_starting_paid"][i]
                    + decision_variables["stay_starting"][i]
                    + decision_variables["bench_to_starting"][i]
                    for i in df[df["position"] == pos].index
                )
                >= min_start
            ), f"Min{min_start}Starting{pos}"

            # Maximum starters per position
            prob += (
                lpSum(
                    decision_variables["transfer_in_starting_free"][i]
                    + decision_variables["transfer_in_starting_paid"][i]
                    + decision_variables["stay_starting"][i]
                    + decision_variables["bench_to_starting"][i]
                    for i in df[df["position"] == pos].index
                )
                <= max_start
            ), f"Max{max_start}Starting{pos}"

            # Total per position
            prob += (
                lpSum(
                    decision_variables["transfer_in_starting_free"][i]
                    + decision_variables["transfer_in_starting_paid"][i]
                    + decision_variables["stay_starting"][i]
                    + decision_variables["bench_to_starting"][i]
                    + decision_variables["transfer_in_bench_free"][i]
                    + decision_variables["transfer_in_bench_paid"][i]
                    + decision_variables["stay_bench"][i]
                    + decision_variables["starting_to_bench"][i]
                    for i in df[df["position"] == pos].index
                )
                == tot
            ), f"Total{tot}{pos}"

    def apply_budget_constraints(self, df: pd.DataFrame, decision_variables: dict, prob: LpProblem, budget: float) -> None:
        """Apply budget constraint."""
        prob += (
            lpSum(
                decision_variables["transfer_in_starting_free"][i] * df.loc[i, "value"]
                + decision_variables["transfer_in_starting_paid"][i] * df.loc[i, "value"]
                + decision_variables["transfer_in_bench_free"][i] * df.loc[i, "value"]
                + decision_variables["transfer_in_bench_paid"][i] * df.loc[i, "value"]
                + decision_variables["stay_starting"][i] * df.loc[i, "value"]
                + decision_variables["stay_bench"][i] * df.loc[i, "value"]
                + decision_variables["bench_to_starting"][i] * df.loc[i, "value"]
                + decision_variables["starting_to_bench"][i] * df.loc[i, "value"]
                for i in df.index
            )
            <= budget
        ), "Budget"

    def apply_team_constraints(self, df: pd.DataFrame, decision_variables: dict, prob: LpProblem) -> None:
        """Max 3 players from same team."""
        for team in df["team"].unique():
            prob += (
                lpSum(
                    decision_variables["transfer_in_starting_free"][i]
                    + decision_variables["transfer_in_starting_paid"][i]
                    + decision_variables["stay_starting"][i]
                    + decision_variables["bench_to_starting"][i]
                    + decision_variables["transfer_in_bench_free"][i]
                    + decision_variables["transfer_in_bench_paid"][i]
                    + decision_variables["stay_bench"][i]
                    + decision_variables["starting_to_bench"][i]
                    for i in df[df["team"] == team].index
                )
                <= 3
            ), f"Max3FromTeam_{team}"

    def apply_player_constraints(
        self,
        df: pd.DataFrame,
        decision_variables: dict,
        prob: LpProblem,
        previous_squad: SquadState,
        free_transfers: int,
    ) -> None:
        """Apply constraints linking to previous squad."""
        # Each player can only be in one state
        for i in df.index:
            prob += (
                decision_variables["transfer_in_starting_free"][i]
                + decision_variables["transfer_in_starting_paid"][i]
                + decision_variables["stay_starting"][i]
                + decision_variables["bench_to_starting"][i]
                + decision_variables["transfer_in_bench_free"][i]
                + decision_variables["transfer_in_bench_paid"][i]
                + decision_variables["stay_bench"][i]
                + decision_variables["starting_to_bench"][i]
                + decision_variables["transfer_out_starting_free"][i]
                + decision_variables["transfer_out_starting_paid"][i]
                + decision_variables["transfer_out_bench_free"][i]
                + decision_variables["transfer_out_bench_paid"][i]
                <= 1
            ), f"Exclusive_{i}"

        # Get indices of previous starters and bench
        previous_starter_ids = set(previous_squad.starters)
        previous_bench_ids = set(previous_squad.bench)

        starter_indices = df[df["element"].isin(previous_starter_ids)].index
        bench_indices = df[df["element"].isin(previous_bench_ids)].index

        # Only previous starters can have "stay_starting" or transfer_out from starting
        for index in df.index:
            if index not in starter_indices:
                prob += decision_variables["stay_starting"][index] == 0
                prob += decision_variables["transfer_out_starting_free"][index] == 0
                prob += decision_variables["transfer_out_starting_paid"][index] == 0
                prob += decision_variables["starting_to_bench"][index] == 0

        # Only previous bench can have "stay_bench" or transfer_out from bench
        for index in df.index:
            if index not in bench_indices:
                prob += decision_variables["stay_bench"][index] == 0
                prob += decision_variables["transfer_out_bench_free"][index] == 0
                prob += decision_variables["transfer_out_bench_paid"][index] == 0
                prob += decision_variables["bench_to_starting"][index] == 0

        # Transfer balance: in = out for each transfer type
        prob += (
            lpSum(decision_variables["transfer_out_starting_free"][i] for i in df.index)
            == lpSum(decision_variables["transfer_in_starting_free"][i] for i in df.index)
        ), "StartingFreeTransfersBalance"

        prob += (
            lpSum(decision_variables["transfer_out_starting_paid"][i] for i in df.index)
            == lpSum(decision_variables["transfer_in_starting_paid"][i] for i in df.index)
        ), "StartingPaidTransfersBalance"

        prob += (
            lpSum(decision_variables["transfer_out_bench_free"][i] for i in df.index)
            == lpSum(decision_variables["transfer_in_bench_free"][i] for i in df.index)
        ), "BenchFreeTransfersBalance"

        prob += (
            lpSum(decision_variables["transfer_out_bench_paid"][i] for i in df.index)
            == lpSum(decision_variables["transfer_in_bench_paid"][i] for i in df.index)
        ), "BenchPaidTransfersBalance"

        # Swap balance constraint removed for flexibility

        # Free transfer limits
        prob += (
            lpSum(
                decision_variables["transfer_in_starting_free"][i]
                + decision_variables["transfer_in_bench_free"][i]
                for i in df.index
            )
            <= free_transfers
        ), "TotalTransferInFreeLimit"

        prob += (
            lpSum(
                decision_variables["transfer_out_starting_free"][i]
                + decision_variables["transfer_out_bench_free"][i]
                for i in df.index
            )
            <= free_transfers
        ), "TotalTransferOutFreeLimit"


def build_transfer_model(
    gameweek_pool: pd.DataFrame,
    current_squad: SquadState,
    budget_cap: float = 100.0,
    free_transfers: int = 1,
    ep_column: str = "ep_next",
) -> pd.DataFrame:
    """Build and solve transfer optimization model following the notebook approach."""
    gameweek_pool = gameweek_pool.copy()
    gameweek_pool["value"] = gameweek_pool["now_cost"] / 10
    gameweek_pool["ep_next"] = pd.to_numeric(gameweek_pool["ep_next"], errors="coerce").fillna(0.0)
    if ep_column != "ep_next" and ep_column in gameweek_pool.columns:
        gameweek_pool[ep_column] = pd.to_numeric(gameweek_pool[ep_column], errors="coerce").fillna(0.0)

    # Build model
    prob = LpProblem("TransferSquadSelection", LpMaximize)
    constraints = TransferConstraints()

    decision_variables = constraints.get_decision_variables(gameweek_pool)
    constraints.set_objective_function(gameweek_pool, decision_variables, prob, ep_column=ep_column)
    constraints.apply_formation_constraints(gameweek_pool, decision_variables, prob)
    constraints.apply_budget_constraints(gameweek_pool, decision_variables, prob, budget_cap)
    constraints.apply_team_constraints(gameweek_pool, decision_variables, prob)
    constraints.apply_player_constraints(gameweek_pool, decision_variables, prob, current_squad, free_transfers)

    # Solve
    solve_status = prob.solve(PULP_CBC_CMD(msg=False))
    status_name = LpStatus[solve_status]

    if status_name != "Optimal":
        print(f"  ⚠️  Solver status: {status_name}. Model infeasible.")
        return None

    # Extract solution
    gameweek_pool["selected"] = 0
    gameweek_pool["starter"] = 0
    gameweek_pool["bench"] = 0
    gameweek_pool["captain"] = 0
    gameweek_pool["transfer_state"] = ""

    for idx in gameweek_pool.index:
        player_id = gameweek_pool.loc[idx, "element"]
        in_prev_starters = player_id in current_squad.starters
        in_prev_bench = player_id in current_squad.bench

        # Determine state and transfer status
        if int(decision_variables["stay_starting"][idx].varValue) == 1:
            gameweek_pool.loc[idx, "selected"] = 1
            gameweek_pool.loc[idx, "starter"] = 1
            gameweek_pool.loc[idx, "transfer_state"] = "STARTER (kept)"
        elif int(decision_variables["transfer_in_starting_free"][idx].varValue) == 1:
            gameweek_pool.loc[idx, "selected"] = 1
            gameweek_pool.loc[idx, "starter"] = 1
            gameweek_pool.loc[idx, "transfer_state"] = "IN → STARTER (free)"
        elif int(decision_variables["transfer_in_starting_paid"][idx].varValue) == 1:
            gameweek_pool.loc[idx, "selected"] = 1
            gameweek_pool.loc[idx, "starter"] = 1
            gameweek_pool.loc[idx, "transfer_state"] = "IN → STARTER (paid, -4pts)"
        elif int(decision_variables["bench_to_starting"][idx].varValue) == 1:
            gameweek_pool.loc[idx, "selected"] = 1
            gameweek_pool.loc[idx, "starter"] = 1
            gameweek_pool.loc[idx, "transfer_state"] = "BENCH → STARTER"
        elif int(decision_variables["stay_bench"][idx].varValue) == 1:
            gameweek_pool.loc[idx, "selected"] = 1
            gameweek_pool.loc[idx, "bench"] = 1
            gameweek_pool.loc[idx, "transfer_state"] = "BENCH (kept)"
        elif int(decision_variables["transfer_in_bench_free"][idx].varValue) == 1:
            gameweek_pool.loc[idx, "selected"] = 1
            gameweek_pool.loc[idx, "bench"] = 1
            gameweek_pool.loc[idx, "transfer_state"] = "IN → BENCH (free)"
        elif int(decision_variables["transfer_in_bench_paid"][idx].varValue) == 1:
            gameweek_pool.loc[idx, "selected"] = 1
            gameweek_pool.loc[idx, "bench"] = 1
            gameweek_pool.loc[idx, "transfer_state"] = "IN → BENCH (paid, -4pts)"
        elif int(decision_variables["transfer_out_starting_free"][idx].varValue) == 1:
            gameweek_pool.loc[idx, "transfer_state"] = "OUT (free transfer)"
        elif int(decision_variables["transfer_out_starting_paid"][idx].varValue) == 1:
            gameweek_pool.loc[idx, "transfer_state"] = "OUT (paid transfer, -4pts)"
        elif int(decision_variables["transfer_out_bench_free"][idx].varValue) == 1:
            gameweek_pool.loc[idx, "transfer_state"] = "OUT (free transfer)"
        elif int(decision_variables["transfer_out_bench_paid"][idx].varValue) == 1:
            gameweek_pool.loc[idx, "transfer_state"] = "OUT (paid transfer, -4pts)"
        elif int(decision_variables["starting_to_bench"][idx].varValue) == 1:
            gameweek_pool.loc[idx, "selected"] = 1
            gameweek_pool.loc[idx, "bench"] = 1
            gameweek_pool.loc[idx, "transfer_state"] = "STARTER → BENCH"

    # Add reporting columns
    POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    gameweek_pool["position_order"] = gameweek_pool["position"].map(POSITION_ORDER)
    gameweek_pool["xP"] = gameweek_pool["ep_next"]

    return gameweek_pool.reset_index(drop=True)
