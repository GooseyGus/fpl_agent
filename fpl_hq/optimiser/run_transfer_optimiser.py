"""Mid-season transfers that maximise the consensus score of your starting XI.

Reads your real squad, bank, free transfers and selling prices from the FPL API, and
the gameweek's consensus from the report CSV. Unmentioned players sit at the neutral
prior. Each transfer must raise the starters' total by `--min-gain`, and a paid
transfer by `--min-gain` plus `--hit-cost`, both in consensus points.

    python -m fpl_hq.optimiser.run_transfer_optimiser --gameweek 7
"""

from __future__ import annotations

import argparse

import pandas as pd
import requests

from fpl_hq.gather.my_team import DEFAULT_TEAM_ID, fetch_my_squad
from fpl_hq.optimiser.transfer_optimizer import build_transfer_model
from fpl_hq.paths import OPTIMISER_OUTPUT_DIR, gw_report_dir
from fpl_hq.reports.deterministic_report import NEUTRAL_STANCE

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
POSITION_BY_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
SCORE_COLUMN = "combined_consensus"
# Players below this chance of playing score zero, so they are never picked to start.
MIN_CHANCE_OF_PLAYING = 50


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimise this week's transfers for consensus.")
    parser.add_argument("--gameweek", type=int, required=True, help="Upcoming gameweek; its consensus CSV must exist.")
    parser.add_argument("--team-id", type=int, default=DEFAULT_TEAM_ID)
    parser.add_argument("--free-transfers", type=int, help="Override the free transfers worked out from your history.")
    parser.add_argument("--bank", type=float, help="Override the bank in millions.")
    parser.add_argument("--min-gain", type=float, default=0.5, help="Consensus gain each transfer must add (default 0.5).")
    parser.add_argument("--hit-cost", type=float, default=1.0, help="Extra consensus cost of a -4 transfer (default 1.0).")
    return parser.parse_args()


def _player_pool(gameweek: int) -> pd.DataFrame:
    response = requests.get(FPL_BOOTSTRAP_URL, timeout=30)
    response.raise_for_status()
    bootstrap = response.json()
    teams = {team["id"]: team["short_name"] for team in bootstrap["teams"]}
    pool = pd.DataFrame(bootstrap["elements"])[
        ["id", "web_name", "element_type", "team", "now_cost", "ep_next", "chance_of_playing_next_round"]
    ].rename(columns={"id": "element"})
    pool["position"] = pool["element_type"].map(POSITION_BY_TYPE)
    pool["team_name"] = pool["team"].map(teams)

    consensus = pd.read_csv(gw_report_dir(gameweek) / f"gw_{gameweek}_consensus.csv")[["player_id", SCORE_COLUMN]]
    pool = pool.merge(consensus, left_on="element", right_on="player_id", how="left").drop(columns="player_id")
    pool["consensus"] = pool[SCORE_COLUMN].fillna(NEUTRAL_STANCE)
    chance = pd.to_numeric(pool["chance_of_playing_next_round"], errors="coerce")
    pool["score"] = pool["consensus"].where(chance.isna() | (chance >= MIN_CHANCE_OF_PLAYING), 0.0)
    return pool


def _describe(solution: pd.DataFrame, baseline: pd.DataFrame, squad, budget: float) -> str:
    selected = solution[solution["selected"] == 1].copy()
    selected["position_order"] = selected["position"].map(POSITION_ORDER)
    starters = selected[selected["starter"] == 1].sort_values(["position_order", "score"], ascending=[True, False])
    bench = selected[selected["bench"] == 1].sort_values("position_order")
    # Sorted by position so an outgoing defender lines up with an incoming one where possible.
    sold = solution[solution["transfer_state"].str.startswith("OUT")].sort_values("position", key=lambda p: p.map(POSITION_ORDER))
    bought = selected[selected["transfer_state"].str.startswith("IN")].sort_values("position_order")
    captain, vice = starters.sort_values("score", ascending=False)["web_name"].head(2)
    hits = int(bought["transfer_state"].str.contains("paid").sum())

    def row(player) -> str:
        flag = "" if player["score"] > 0 else "  (doubtful: scored 0)"
        return f"  {player['position']:3s} {player['web_name']:16s} {player['team_name']:4s} £{player['value']:.1f}m  consensus {player['consensus']:.2f}{flag}"

    total = float(starters["score"].sum())
    best_without = float(baseline.loc[baseline["starter"] == 1, "score"].sum())
    lines = [f"GW{squad.gameweek}: {squad.free_transfers} free transfer(s), budget £{budget:.1f}m", ""]
    if sold.empty:
        lines.append("No transfers: none clears the minimum gain.")
    else:
        lines.append(f"Transfers ({len(bought)}, {hits} hit(s)):")
        for (_, out_player), (_, in_player) in zip(sold.iterrows(), bought.iterrows()):
            lines.append(f"  OUT {out_player['web_name']:16s} £{out_player['value']:.1f}m  ->  IN {in_player['web_name']:16s} £{in_player['value']:.1f}m")
    lines += ["", "Starting XI:", *[row(p) for _, p in starters.iterrows()], "", "Bench:", *[row(p) for _, p in bench.iterrows()]]
    lines += [
        "",
        f"Captain: {captain}   Vice: {vice}",
        f"XI consensus: {total:.2f} (best XI without transfers: {best_without:.2f}, gain {total - best_without:+.2f})",
        f"Money left: £{budget - float(selected['value'].sum()):.1f}m",
    ]
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    squad = fetch_my_squad(args.team_id)
    if squad.gameweek != args.gameweek:
        print(f"Warning: the next FPL deadline is GW{squad.gameweek}, but --gameweek is {args.gameweek}.")
    if args.free_transfers is not None:
        squad.free_transfers = args.free_transfers
    if args.bank is not None:
        squad.bank = args.bank
    budget = squad.bank + sum(squad.selling_prices.values())

    pool = _player_pool(args.gameweek)
    options = dict(current_squad=squad, ep_column="score", min_gain=args.min_gain, hit_cost=args.hit_cost)
    solution = build_transfer_model(pool, free_transfers=squad.free_transfers, **options)
    baseline = build_transfer_model(pool, free_transfers=0, **{**options, "hit_cost": 1e6})
    if solution is None or baseline is None:
        raise RuntimeError("The transfer model is infeasible; check the squad, bank and free transfers.")

    report = _describe(solution, baseline, squad, budget)
    print(report)
    output_path = OPTIMISER_OUTPUT_DIR / f"transfer_plan_gw{args.gameweek}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
