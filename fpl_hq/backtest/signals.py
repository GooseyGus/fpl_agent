"""Pre-deadline signals to score against final points.

Every signal reads only the gameweek's own snapshot, so nothing from after the
deadline can leak in. Higher is always better.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from fpl_hq.reports.deterministic_report import NEUTRAL_STANCE


Player = dict[str, Any]


@dataclass(frozen=True)
class Evidence:
    """What the creators said about one player before the deadline."""

    consensus: dict[str, Any] | None
    bt: float | None  # Bradley-Terry log-strength: anchored stances plus comparisons


def _last_3_points_per_appearance(player: Player) -> float:
    form = player.get("recent_form", {}).get("last_3_gameweeks", {})
    return float(form.get("averages_per_appearance", {}).get("total_points", 0) or 0)


def _fixture_difficulty(player: Player) -> float | None:
    fixture = player.get("upcoming_fixture") or {}
    difficulty = fixture.get("difficulty")
    return float(difficulty) if difficulty is not None else None


def consensus(player: Player, evidence: Evidence) -> float:
    # Unmentioned players sit at the neutral prior, matching how shrinkage treats them.
    return evidence.consensus["consensus_score"] if evidence.consensus else NEUTRAL_STANCE


def bt(player: Player, evidence: Evidence) -> float:
    # Players with no stance or comparison sit at the neutral anchor, 0.
    return evidence.bt if evidence.bt is not None else 0.0


def ownership(player: Player, _: Evidence) -> float:
    return float(player.get("ownership_percent", 0) or 0)


def net_transfers(player: Player, _: Evidence) -> float:
    return float(player.get("transfers_in_event", 0) or 0) - float(player.get("transfers_out_event", 0) or 0)


def form(player: Player, _: Evidence) -> float:
    return _last_3_points_per_appearance(player)


def form_fixture(player: Player, _: Evidence) -> float:
    """Recent points scaled by a mild fixture adjustment: FDR 2 -> +10%, FDR 5 -> -20%."""
    difficulty = _fixture_difficulty(player)
    if difficulty is None:
        return 0.0
    return _last_3_points_per_appearance(player) * (1 + 0.1 * (3 - difficulty))


SIGNALS: dict[str, Callable[[Player, Evidence], float]] = {
    "consensus": consensus,
    "bt": bt,
    "ownership": ownership,
    "net_transfers": net_transfers,
    "form": form,
    "form_fixture": form_fixture,
}
