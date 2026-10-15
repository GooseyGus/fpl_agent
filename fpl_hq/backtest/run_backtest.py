"""Score pre-deadline signals against final gameweek points.

Run from the project root:

    python -m fpl_hq.backtest.run_backtest
    python -m fpl_hq.backtest.run_backtest --gameweeks 3 4 5 --output backtest/results.csv
"""

from __future__ import annotations

import argparse
import warnings
import sys
from pathlib import Path

import pandas as pd

from fpl_hq.backtest.data import (
    finished_gameweeks,
    has_numeric_extractions,
    load_consensus,
    load_final_points,
    load_snapshot,
)
from fpl_hq.backtest.signals import SIGNALS, Evidence
from fpl_hq.consensus.bradley_terry import fit_gameweek, load_comparisons
from fpl_hq.paths import GAMEWEEKS_DIR


POSITION_LABELS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
TOP_K = 10
MIN_POSITION_SIZE = 5
# compared: players in at least one comparison, so consensus and BT are scored on the same set.
UNIVERSES = ("compared", "mentioned", "active")


def build_frame(gameweek: int) -> pd.DataFrame:
    """One row per player: universe flags, every signal, and final points."""
    players = load_snapshot(gameweek)
    consensus = load_consensus(gameweek, players)
    final_points = load_final_points(gameweek)
    ids_by_name = {player["name"]: int(player_id) for player_id, player in players.items()}
    stances = [(item["player"]["name"], int(e["stance"])) for item in consensus.values() for e in item["evidence"]]
    bt_fit = fit_gameweek(gameweek, stances)
    bt_scores = {ids_by_name[name]: value for name, value in bt_fit.strengths.items()}
    compared = {ids_by_name[name] for name in {c.winner for c in load_comparisons(gameweek)} | {c.loser for c in load_comparisons(gameweek)}}

    rows = []
    for raw_id, player in players.items():
        player_id = int(raw_id)
        item = consensus.get(player_id)
        evidence = Evidence(item, bt_scores.get(player_id))
        recent = player.get("recent_form", {}).get("last_3_gameweeks", {})
        rows.append(
            {
                "player_id": player_id,
                "name": player["name"],
                "position": POSITION_LABELS.get(player.get("position"), "?"),
                "mentioned": item is not None,
                "compared": player_id in compared,
                "active": int(recent.get("appearances", 0) or 0) > 0 or item is not None or player_id in compared,
                "points": final_points.get(player_id, 0),
                **{name: signal(player, evidence) for name, signal in SIGNALS.items()},
            }
        )
    return pd.DataFrame(rows)


def spearman(left: pd.Series, right: pd.Series) -> float:
    """Spearman correlation as Pearson on average ranks (pandas' built-in needs scipy).

    NaN when a signal is constant within the group, e.g. every goalkeeper unmentioned.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return left.rank().corr(right.rank())


def score(frame: pd.DataFrame, signal: str) -> dict[str, float]:
    by_position = [
        spearman(group[signal], group["points"])
        for _, group in frame.groupby("position")
        if len(group) >= MIN_POSITION_SIZE
    ]
    top = frame.sort_values([signal, "player_id"], ascending=[False, True]).head(TOP_K)
    return {
        "spearman": spearman(frame[signal], frame["points"]),
        "position_spearman": pd.Series(by_position).mean(),
        f"top{TOP_K}_points": top["points"].mean(),
    }


def evaluate(gameweeks: list[int]) -> pd.DataFrame:
    results = []
    for gameweek in gameweeks:
        print(f"GW{gameweek}", file=sys.stderr)
        frame = build_frame(gameweek)
        for universe in UNIVERSES:
            subset = frame[frame[universe]]
            for signal in SIGNALS:
                results.append(
                    {"gameweek": gameweek, "universe": universe, "players": len(subset), "signal": signal, **score(subset, signal)}
                )
            # What picking at random from this universe would score.
            results.append(
                {"gameweek": gameweek, "universe": universe, "players": len(subset), "signal": "(random)",
                 "spearman": 0.0, "position_spearman": 0.0, f"top{TOP_K}_points": subset["points"].mean()}
            )
    return pd.DataFrame(results)


def default_gameweeks() -> list[int]:
    finished = finished_gameweeks()
    candidates = sorted(int(path.name.removeprefix("gw")) for path in GAMEWEEKS_DIR.glob("gw[0-9]*"))
    return [gameweek for gameweek in candidates if gameweek in finished and has_numeric_extractions(gameweek)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest consensus and baseline signals against final points.")
    parser.add_argument("--gameweeks", type=int, nargs="+", help="Defaults to every finished gameweek with extractions.")
    parser.add_argument("--output", type=Path, help="Optional CSV path for the per-gameweek results.")
    args = parser.parse_args()

    gameweeks = args.gameweeks or default_gameweeks()
    if not gameweeks:
        raise SystemExit("No finished gameweeks with extractions to backtest.")
    results = evaluate(gameweeks)

    metrics = ["spearman", "position_spearman", f"top{TOP_K}_points"]
    pd.set_option("display.width", 160)
    for universe, group in results.groupby("universe", sort=False):
        print(f"\n=== {universe} players, per gameweek ===")
        print(group.pivot(index="signal", columns="gameweek", values=metrics).round(2).to_string())
        print(f"\n=== {universe} players, mean over GW{', GW'.join(map(str, gameweeks))} ===")
        print(group.groupby("signal")[metrics].mean().sort_values("position_spearman", ascending=False).round(3).to_string())

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
