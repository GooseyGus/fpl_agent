"""Fetch your current FPL squad, bank, free transfers and selling prices from the public API.

Only completed gameweeks are public, so transfers already made for the upcoming
deadline are not visible. Override the free transfers or bank if they look wrong.

    python -m fpl_hq.gather.my_team
"""

from __future__ import annotations

import argparse
from typing import Any

import requests

from fpl_hq.optimiser.squad_state import SquadState

FPL_API = "https://fantasy.premierleague.com/api"
DEFAULT_TEAM_ID = 3123442
MAX_FREE_TRANSFERS = 5
SQUAD_CHIPS = {"wildcard", "freehit"}


def _get(path: str) -> Any:
    response = requests.get(f"{FPL_API}/{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def free_transfers(history: dict[str, Any]) -> int:
    """Free transfers for the next deadline.

    One free transfer after the first gameweek, plus one per gameweek up to the cap.
    Transfers spend them, and Wildcard and Free Hit weeks leave the balance untouched.
    """
    chip_events = {chip["event"] for chip in history["chips"] if chip["name"] in SQUAD_CHIPS}
    available = None
    for event in history["current"]:
        if available is None:
            available = 1
            continue
        if event["event"] not in chip_events:
            available = max(available - event["event_transfers"], 0)
        available = min(available + 1, MAX_FREE_TRANSFERS)
    return available or 1


def selling_price(purchase: int, now: int) -> int:
    """FPL keeps half of any rise (rounded down) and all of any fall, in tenths of a million."""
    return purchase + (now - purchase) // 2 if now > purchase else now


def fetch_my_squad(team_id: int = DEFAULT_TEAM_ID) -> SquadState:
    """The squad going into the next deadline, ordered as 11 starters then 4 bench."""
    bootstrap = _get("bootstrap-static/")
    elements = {element["id"]: element for element in bootstrap["elements"]}
    next_gameweek = next(event["id"] for event in bootstrap["events"] if event["is_next"])
    history = _get(f"entry/{team_id}/history/")
    transfers = _get(f"entry/{team_id}/transfers/")

    # A Free Hit squad reverts afterwards, so read the squad from the gameweek before it.
    free_hit_events = {chip["event"] for chip in history["chips"] if chip["name"] == "freehit"}
    last_event = history["current"][-1]["event"]
    squad_event = last_event - 1 if last_event in free_hit_events else last_event
    picks = sorted(_get(f"entry/{team_id}/event/{squad_event}/picks/")["picks"], key=lambda pick: pick["position"])
    bank = history["current"][-1]["bank"]

    prices = {}
    for pick in picks:
        element = elements[pick["element"]]
        buys = [t for t in transfers if t["element_in"] == pick["element"] and t["event"] not in free_hit_events]
        # Players held since the first deadline were bought at their start-of-season price.
        purchase = max(buys, key=lambda t: t["time"])["element_in_cost"] if buys else element["now_cost"] - element["cost_change_start"]
        prices[pick["element"]] = selling_price(purchase, element["now_cost"]) / 10

    return SquadState(
        player_ids=[pick["element"] for pick in picks],
        gameweek=next_gameweek,
        bank=bank / 10,
        free_transfers=free_transfers(history),
        selling_prices=prices,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Show your current FPL squad, bank and free transfers.")
    parser.add_argument("--team-id", type=int, default=DEFAULT_TEAM_ID)
    args = parser.parse_args()
    squad = fetch_my_squad(args.team_id)
    names = {element["id"]: element["web_name"] for element in _get("bootstrap-static/")["elements"]}
    print(squad)
    for slot, player_id in enumerate(squad.player_ids, start=1):
        role = "XI " if slot <= 11 else "SUB"
        print(f"  {role} {names[player_id]:16s} sell £{squad.selling_prices[player_id]:.1f}m")
    print(f"Budget: £{squad.bank + sum(squad.selling_prices.values()):.1f}m (bank £{squad.bank:.1f}m)")


if __name__ == "__main__":
    main()
