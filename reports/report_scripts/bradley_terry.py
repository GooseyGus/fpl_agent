"""Bradley-Terry player strengths from extracted head-to-head comparisons.

Uses Davidson's (1970) extension so draws are modelled directly. For players i
and j with log-strengths a and b, and a log draw parameter d:

    P(i better) = e^a / Z,   P(j better) = e^b / Z,   P(equal) = e^(d + (a+b)/2) / Z

A Gaussian prior on each log-strength pulls rarely compared players toward 0
(average), the same role PRIOR_WEIGHT plays for stance consensus. The penalised
log-likelihood is concave, so Newton's method finds the unique maximum.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CERTAINTY_WEIGHTS = {1: 0.5, 2: 1.0}
ANCHOR = "__neutral__"
# stance -> (outcome against the neutral anchor, weight)
STANCE_ANCHORS = {5: ("win", 1.0), 4: ("win", 0.5), 3: ("draw", 1.0), 2: ("loss", 0.5), 1: ("loss", 1.0)}
PRIOR_SD = 1.0
DRAW_PRIOR_SD = 10.0


@dataclass(frozen=True)
class Comparison:
    winner: str  # for a draw, either player
    loser: str
    draw: bool
    weight: float


@dataclass(frozen=True)
class Fit:
    strengths: dict[str, float]
    draw_parameter: float
    comparison_counts: dict[str, int]


def anchor_comparisons(stances: Iterable[tuple[str, int]]) -> list[Comparison]:
    """Turn 1-5 stances into matches against one fixed neutral player.

    Comparisons happen within shortlists (budget keepers against budget keepers,
    premiums against premiums), so on their own they cannot place one shortlist
    above another. Anchoring every recommendation to the same reference puts all
    players on one scale, and comparisons then refine the order within it.
    """
    comparisons = []
    for player, stance in stances:
        outcome, weight = STANCE_ANCHORS[stance]
        winner, loser = (ANCHOR, player) if outcome == "loss" else (player, ANCHOR)
        comparisons.append(Comparison(winner, loser, outcome == "draw", weight))
    return comparisons


def fit(comparisons: Iterable[Comparison], prior_sd: float = PRIOR_SD, max_iter: int = 100) -> Fit:
    comparisons = list(comparisons)
    players = sorted({name for c in comparisons for name in (c.winner, c.loser)})
    if not players:
        return Fit({}, 1.0, {})
    index = {name: position for position, name in enumerate(players)}
    n = len(players)
    i = np.array([index[c.winner] for c in comparisons])
    j = np.array([index[c.loser] for c in comparisons])
    draw = np.array([c.draw for c in comparisons])
    weight = np.array([c.weight for c in comparisons])
    precision = np.r_[np.full(n, 1 / prior_sd**2), 1 / DRAW_PRIOR_SD**2]

    def objective(params: np.ndarray) -> float:
        theta, d = params[:n], params[n]
        logits = np.stack([theta[i], theta[j], d + 0.5 * (theta[i] + theta[j])], axis=1)
        observed = np.where(draw, logits[:, 2], logits[:, 0])
        log_z = np.logaddexp.reduce(logits, axis=1)
        return float(np.sum(weight * (observed - log_z)) - 0.5 * np.sum(precision * params**2))

    params = np.zeros(n + 1)
    for _ in range(max_iter):
        theta, d = params[:n], params[n]
        logits = np.stack([theta[i], theta[j], d + 0.5 * (theta[i] + theta[j])], axis=1)
        p = np.exp(logits - np.logaddexp.reduce(logits, axis=1, keepdims=True))
        p_win, p_lose, p_draw = p[:, 0], p[:, 1], p[:, 2]

        # Each comparison touches three parameters: (theta_i, theta_j, d). The logit
        # gradients are u1 = (1,0,0), u2 = (0,1,0), u3 = (.5,.5,1).
        mean_u = np.stack([p_win + 0.5 * p_draw, p_lose + 0.5 * p_draw, p_draw], axis=1)
        observed_u = np.where(draw[:, None], [0.5, 0.5, 1.0], [1.0, 0.0, 0.0])
        local_grad = weight[:, None] * (observed_u - mean_u)
        # E[u u^T] - E[u] E[u]^T, the per-comparison Fisher information.
        second = np.zeros((len(comparisons), 3, 3))
        second[:, 0, 0] = p_win + 0.25 * p_draw
        second[:, 1, 1] = p_lose + 0.25 * p_draw
        second[:, 0, 1] = second[:, 1, 0] = 0.25 * p_draw
        second[:, 0, 2] = second[:, 2, 0] = second[:, 1, 2] = second[:, 2, 1] = 0.5 * p_draw
        second[:, 2, 2] = p_draw
        local_info = weight[:, None, None] * (second - mean_u[:, :, None] * mean_u[:, None, :])

        slots = np.stack([i, j, np.full_like(i, n)], axis=1)
        grad = np.zeros(n + 1)
        np.add.at(grad, slots, local_grad)
        grad -= precision * params
        info = np.diag(precision)
        np.add.at(info, (slots[:, :, None], slots[:, None, :]), local_info)

        step = np.linalg.solve(info, grad)
        current, scale = objective(params), 1.0
        while objective(params + scale * step) < current - 1e-12 and scale > 1e-6:
            scale /= 2
        params = params + scale * step
        if np.max(np.abs(scale * step)) < 1e-8:
            break

    counts = {name: 0 for name in players}
    for c in comparisons:
        counts[c.winner] += 1
        counts[c.loser] += 1
    return Fit(
        strengths={name: float(params[index[name]]) for name in players},
        draw_parameter=float(np.exp(params[n])),
        comparison_counts=counts,
    )


def load_comparisons(gameweek: int, horizons: set[str] | None = None) -> list[Comparison]:
    path = PROJECT_ROOT / "gw" / f"gw{gameweek}" / "data" / f"gw_{gameweek}_transcript_comparisons.json"
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8")).get("comparisons", [])
    return [
        Comparison(
            winner=record["player_a"],
            loser=record["player_b"],
            draw=record["relation"] == "equal",
            weight=CERTAINTY_WEIGHTS[record["certainty"]],
        )
        for record in records
        if horizons is None or record["horizon"] in horizons
    ]


def fit_gameweek(gameweek: int, stances: Iterable[tuple[str, int]], horizons: set[str] | None = None) -> Fit:
    """Strengths from stances anchored to a neutral player, refined by head-to-head comparisons.

    Being compared at all means a creator treats the player as a live option, so a
    compared player with no stance gets an implicit 3 (a draw with the anchor).
    Without it, players named only as losing alternatives sink below the neutral point.
    """
    stances = list(stances)
    comparisons = load_comparisons(gameweek, horizons)
    with_stance = {name for name, _ in stances}
    compared = {name for c in comparisons for name in (c.winner, c.loser)}
    floors = anchor_comparisons((name, 3) for name in sorted(compared - with_stance))
    result = fit(anchor_comparisons(stances) + floors + comparisons)
    strengths = {name: value for name, value in result.strengths.items() if name != ANCHOR}
    counts = {name: value for name, value in result.comparison_counts.items() if name != ANCHOR}
    return Fit(strengths, result.draw_parameter, counts)
