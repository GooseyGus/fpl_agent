"""Build an FPL report from canonical transcript recommendations and player form."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from reports.report_scripts.fpl_report_v2 import load_name_resolution_helpers


PROJECT_ROOT = Path(__file__).resolve().parent
POSITION_LABELS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
NEUTRAL_STANCE = 3.0
PRIOR_WEIGHT = 2


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def invert_element_lookup(element_lookup: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Return the one-to-one canonical player name to FPL element ID mapping."""
    ids_by_name: dict[str, list[int]] = defaultdict(list)
    for raw_player_id, player in element_lookup.items():
        name = str(player.get("name", "")).strip()
        if not name:
            raise ValueError(f"Element {raw_player_id} has no canonical name.")
        ids_by_name[name].append(int(raw_player_id))

    duplicates = {name: ids for name, ids in ids_by_name.items() if len(ids) > 1}
    if duplicates:
        details = ", ".join(f"{name}: {ids}" for name, ids in sorted(duplicates.items()))
        raise ValueError(f"Canonical names must be unique in element_lookup.json: {details}")

    return {name: ids[0] for name, ids in ids_by_name.items()}


def assign_player_ids(
    recommendations: list[dict[str, Any]],
    players: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign IDs using canonical names plus alias-aware name resolution."""
    helpers = load_name_resolution_helpers(players)
    resolve_supported_player_id = helpers.resolve_supported_player_id
    player_names = {player["name"]: int(player_id) for player_id, player in players.items()}

    unresolved = Counter()
    assigned = []
    for recommendation in recommendations:
        name = str(recommendation.get("player_name", "")).strip()
        player_id = resolve_supported_player_id(recommendation, player_names)
        if player_id is None:
            unresolved[name] += 1
            continue
        assigned.append({**recommendation, "player_id": player_id})

    if unresolved:
        details = "\n".join(
            f"  - {name or '<missing player_name>'}: {count} recommendation(s)"
            for name, count in sorted(unresolved.items())
        )
        raise ValueError(
            "Recommendations contain player names that could not be resolved to canonical FPL names. "
            "Fix the extraction or alias mapping before reporting:\n"
            f"{details}"
        )

    return assigned


def group_recommendations(
    recommendations: list[dict[str, Any]],
    players: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for recommendation in recommendations:
        player_id = int(recommendation["player_id"])
        if str(player_id) not in players:
            raise ValueError(
                f"Player ID {player_id} ({recommendation['player_name']}) is absent from aggregated player form."
            )
        stance = recommendation.get("stance")
        if isinstance(stance, bool) or not isinstance(stance, (int, float)) or not 1 <= stance <= 5:
            raise ValueError(
                f"Invalid stance for {recommendation['player_name']}: {stance!r}; expected 1-5."
            )
        grouped[player_id].append(recommendation)

    result = []
    for player_id, evidence in grouped.items():
        player = players[str(player_id)]
        stances = [float(item["stance"]) for item in evidence]
        sources = sorted({str(item.get("source", "")).strip() for item in evidence if item.get("source")})
        mean_stance = fmean(stances)
        shrunk_stance = (sum(stances) + NEUTRAL_STANCE * PRIOR_WEIGHT) / (
            len(stances) + PRIOR_WEIGHT
        )
        result.append(
            {
                "player_id": player_id,
                "player": player,
                "evidence": evidence,
                "mean_stance": mean_stance,
                "consensus_score": shrunk_stance,
                "stance_deviation": pstdev(stances),
                "stance_counts": Counter(int(stance) for stance in stances),
                "mention_count": len(evidence),
                "sources": sources,
                "source_count": len(sources),
            }
        )
    return result


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "/").replace("\n", " ").strip()


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(markdown_escape(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def format_fixture(player: dict[str, Any]) -> str:
    fixture = player.get("upcoming_fixture")
    if not fixture:
        return "-"
    venue = "H" if fixture.get("is_home") else "A"
    return f"{fixture.get('opponent', '-')} ({venue}, FDR {fixture.get('difficulty', '-')})"


def form_snapshot(player: dict[str, Any]) -> dict[str, Any]:
    form = player.get("recent_form", {}).get("last_3_gameweeks", {})
    totals = form.get("totals", {})
    per_90 = form.get("per_90", {})
    appearances = int(form.get("appearances", 0) or 0)
    points = float(totals.get("total_points", 0) or 0)
    return {
        "appearances": appearances,
        "points_per_appearance": points / appearances if appearances else 0,
        "minutes": int(totals.get("minutes", 0) or 0),
        "xgi_per_90": float(per_90.get("expected_goal_involvements", 0) or 0),
    }


def signal_labels(item: dict[str, Any]) -> list[str]:
    player = item["player"]
    snapshot = form_snapshot(player)
    labels = []
    if item["source_count"] >= 3 and item["mean_stance"] >= 4:
        labels.append("broad positive consensus")
    if item["source_count"] >= 2 and item["mean_stance"] <= 2.5:
        labels.append("multiple negative sources")
    if item["mention_count"] >= 2 and item["stance_deviation"] >= 0.9:
        labels.append("analysts disagree")
    if item["mean_stance"] >= 4 and snapshot["xgi_per_90"] < 0.25 and player.get("position") in {3, 4}:
        labels.append("enthusiasm ahead of recent xGI")
    if item["mean_stance"] >= 4 and snapshot["minutes"] < 180:
        labels.append("positive view with limited recent minutes")
    if player.get("chance_of_playing_next_round") not in {None, 100}:
        labels.append(f"availability {player['chance_of_playing_next_round']}%")
    return labels or ["no exceptional cross-signal"]


def stance_distribution(item: dict[str, Any]) -> str:
    return " ".join(
        f"{stance}:{item['stance_counts'].get(stance, 0)}" for stance in range(1, 6)
    )


def overview_row(item: dict[str, Any]) -> list[Any]:
    player = item["player"]
    snapshot = form_snapshot(player)
    return [
        player["name"],
        player["team"],
        POSITION_LABELS.get(player.get("position"), player.get("position", "-")),
        f"{item['consensus_score']:.2f}",
        f"{item['mean_stance']:.2f}",
        item["source_count"],
        item["mention_count"],
        f"{item['stance_deviation']:.2f}",
        f"{snapshot['points_per_appearance']:.1f}",
        f"{snapshot['xgi_per_90']:.2f}",
        format_fixture(player),
    ]


def _item_index_by_player_id(items: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(item["player_id"]): item for item in items}


def _rank_grouped_items(grouped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        grouped,
        key=lambda item: (item["consensus_score"], item["source_count"], item["mention_count"]),
        reverse=True,
    )


def _source_scores(
    extraction: dict[str, Any],
    players: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    recommendations = extraction.get("recommendations")
    if not isinstance(recommendations, list):
        raise ValueError("Extraction file must contain a recommendations list.")
    assigned = assign_player_ids([item for item in recommendations if isinstance(item, dict)], players)
    grouped = group_recommendations(assigned, players)
    ranked = _rank_grouped_items(grouped)
    sources = sorted({str(item.get("source", "")).strip() for item in assigned if str(item.get("source", "")).strip()})
    if not sources:
        sources = sorted(str(item).strip() for item in extraction.get("sources", []) if str(item).strip())
    return ranked, sources


def _combined_ranked(
    transcript_ranked: list[dict[str, Any]],
    text_ranked: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    combined_map: dict[int, dict[str, Any]] = {}
    for item in transcript_ranked + text_ranked:
        player_id = int(item["player_id"])
        combined_map[player_id] = item
    combined = [item for item in transcript_ranked + text_ranked if item["player_id"] in combined_map]
    grouped_like: dict[int, dict[str, Any]] = {}
    for item in transcript_ranked + text_ranked:
        player_id = int(item["player_id"])
        entry = grouped_like.setdefault(
            player_id,
            {
                "player_id": player_id,
                "player": item["player"],
                "evidence": [],
                "mean_stance": 0.0,
                "consensus_score": 0.0,
                "stance_deviation": 0.0,
                "mention_count": 0,
                "sources": [],
                "source_count": 0,
            },
        )
        entry["evidence"].extend(item.get("evidence", []))
        entry["mention_count"] += int(item.get("mention_count", 0) or 0)
        entry["sources"].extend(item.get("sources", []))
    result: list[dict[str, Any]] = []
    for player_id, entry in grouped_like.items():
        evidence = entry["evidence"]
        stances = [float(item["stance"]) for item in evidence if isinstance(item.get("stance"), (int, float))]
        if not stances:
            continue
        mean_stance = fmean(stances)
        consensus_score = (sum(stances) + NEUTRAL_STANCE * PRIOR_WEIGHT) / (len(stances) + PRIOR_WEIGHT)
        result.append(
            {
                "player_id": player_id,
                "player": entry["player"],
                "evidence": evidence,
                "mean_stance": mean_stance,
                "consensus_score": consensus_score,
                "stance_deviation": pstdev(stances) if len(stances) > 1 else 0.0,
                "stance_counts": Counter(int(stance) for stance in stances),
                "mention_count": len(stances),
                "sources": sorted({str(source).strip() for source in entry["sources"] if str(source).strip()}),
                "source_count": len({str(source).strip() for source in entry["sources"] if str(source).strip()}),
            }
        )
    return _rank_grouped_items(result)


def _markdown_section(title: str, ranked: list[dict[str, Any]], empty_message: str) -> list[str]:
    lines = [f"## {title}", ""]
    lines.append(markdown_table(OVERVIEW_HEADERS, [overview_row(item) for item in ranked]) if ranked else empty_message)
    lines.append("")
    return lines


OVERVIEW_HEADERS = [
    "Player",
    "Team",
    "Pos",
    "Consensus",
    "Raw mean",
    "Sources",
    "Mentions",
    "Disagreement",
    "Pts/app",
    "xGI/90",
    "Fixture",
]


def render_player(item: dict[str, Any]) -> list[str]:
    player = item["player"]
    snapshot = form_snapshot(player)
    position = POSITION_LABELS.get(player.get("position"), player.get("position", "-"))
    run = player.get("fixture_run") or {}
    run_text = (
        f"{float(run['average_difficulty']):.1f} average FDR over {run['num_fixtures']} fixtures"
        if run.get("num_fixtures") and run.get("average_difficulty") is not None
        else "no fixture run available"
    )
    lines = [
        f"### {player['name']}",
        "",
        f"**{player['team']} · {position} · £{float(player.get('price', 0)):.1f}m · "
        f"{float(player.get('ownership_percent', 0)):.1f}% owned**",
        "",
        f"- **Analyst signal:** {item['mean_stance']:.2f}/5 raw mean; "
        f"{item['consensus_score']:.2f}/5 consensus; {item['source_count']} source(s), "
        f"{item['mention_count']} mention(s); distribution `{stance_distribution(item)}`.",
        f"- **Recent form:** {snapshot['points_per_appearance']:.1f} points per appearance, "
        f"{snapshot['xgi_per_90']:.2f} xGI/90, {snapshot['minutes']} minutes over the last three gameweeks.",
        f"- **Fixtures:** {format_fixture(player)}; {run_text}.",
        f"- **Signals:** {', '.join(signal_labels(item))}.",
    ]
    news = str(player.get("news", "")).strip()
    if news:
        lines.append(f"- **Availability note:** {news}")
    lines.extend(["", "**Source evidence**", ""])
    for evidence in sorted(item["evidence"], key=lambda row: (str(row.get("source", "")), -float(row["stance"]))):
        lines.extend(
            [
                f"- **{evidence['stance']}/5 · {markdown_escape(evidence.get('source', 'unknown source'))}:** "
                f"{markdown_escape(evidence.get('sentiment', ''))}",
                f"  > {markdown_escape(evidence.get('supporting_quote', ''))}",
            ]
        )
    lines.append("")
    return lines


def build_report(gameweek: int) -> str:
    data_dir = PROJECT_ROOT / "gw" / f"gw{gameweek}" / "data"
    transcript_extraction_path = data_dir / f"gw_{gameweek}_transcript_extractions.json"
    text_extraction_path = data_dir / f"gw_{gameweek}_text_extractions.json"
    aggregate_path = data_dir / f"gw_{gameweek}_aggregated_player_form.json"
    lookup_path = PROJECT_ROOT / "lookup" / "element_lookup.json"
    for path in (aggregate_path, lookup_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path.relative_to(PROJECT_ROOT)}")

    players = load_json(aggregate_path)
    transcript_extraction = load_json(transcript_extraction_path) if transcript_extraction_path.exists() else {"recommendations": [], "sources": []}
    text_extraction = load_json(text_extraction_path) if text_extraction_path.exists() else {"recommendations": [], "sources": []}

    transcript_ranked, transcript_sources = _source_scores(transcript_extraction, players)
    text_ranked, text_sources = _source_scores(text_extraction, players)
    ranked = _combined_ranked(transcript_ranked, text_ranked)

    strong = [item for item in ranked if item["consensus_score"] >= 3.75 and item["source_count"] >= 2]
    avoid = sorted(
        [item for item in ranked if item["consensus_score"] <= 2.75],
        key=lambda item: (item["consensus_score"], -item["source_count"]),
    )
    divided = sorted(
        [item for item in ranked if item["mention_count"] >= 2 and item["stance_deviation"] >= 0.9],
        key=lambda item: (item["stance_deviation"], item["source_count"]),
        reverse=True,
    )
    under_radar = [
        item for item in ranked
        if item["mean_stance"] >= 4
        and item["source_count"] <= 2
        and float(item["player"].get("ownership_percent", 0) or 0) < 10
    ]

    lines = [
        f"# FPL Gameweek {gameweek} Evidence Report",
        "",
        "This report joins transcript and website-text recommendations to FPL player form by exact player name and ID. "
        "It uses no additional LLM synthesis.",
        "",
        f"- **Inputs:** `{transcript_extraction_path.relative_to(PROJECT_ROOT)}`, "
        f"`{text_extraction_path.relative_to(PROJECT_ROOT)}`, `{aggregate_path.relative_to(PROJECT_ROOT)}`, "
        f"`{lookup_path.relative_to(PROJECT_ROOT)}`",
        f"- **Coverage:** {len(transcript_extraction.get('recommendations', []))} transcript recommendations, "
        f"{len(text_extraction.get('recommendations', []))} text recommendations, {len(ranked)} combined players.",
        f"- **Sources:** {len(transcript_sources)} transcript source(s), {len(text_sources)} text source(s).",
        f"- **Consensus score:** raw 1-5 stances shrunk toward neutral using a {PRIOR_WEIGHT}-mention prior. "
        "The raw mean, source count, distribution, and disagreement remain visible.",
        "",
        *
        _markdown_section(
            "Transcript Consensus",
            transcript_ranked,
            "No transcript players were available.",
        ),
        *
        _markdown_section(
            "Website Text Consensus",
            text_ranked,
            "No website text players were available.",
        ),
        *
        _markdown_section(
            "Combined Consensus",
            ranked,
            "No combined players were available.",
        ),
        "## Strong Consensus",
        "",
        markdown_table(OVERVIEW_HEADERS, [overview_row(item) for item in strong])
        if strong else "No players met the strong-consensus threshold.",
        "",
        "## Players To Avoid",
        "",
        markdown_table(OVERVIEW_HEADERS, [overview_row(item) for item in avoid])
        if avoid else "No players met the avoid threshold.",
        "",
        "## Divided Opinion",
        "",
        markdown_table(OVERVIEW_HEADERS, [overview_row(item) for item in divided])
        if divided else "No players showed substantial source disagreement.",
        "",
        "## Under The Radar",
        "",
        markdown_table(OVERVIEW_HEADERS, [overview_row(item) for item in under_radar])
        if under_radar else "No low-ownership players met the limited-coverage positive threshold.",
        "",
        "## Full Player Index",
        "",
        markdown_table(OVERVIEW_HEADERS, [overview_row(item) for item in ranked]),
        "",
        "## Player Evidence",
        "",
    ]
    for item in ranked:
        lines.extend(render_player(item))
    return "\n".join(lines).rstrip() + "\n"


def build_consensus_csv(gameweek: int) -> str:
    data_dir = PROJECT_ROOT / "gw" / f"gw{gameweek}" / "data"
    transcript_extraction_path = data_dir / f"gw_{gameweek}_transcript_extractions.json"
    text_extraction_path = data_dir / f"gw_{gameweek}_text_extractions.json"
    aggregate_path = data_dir / f"gw_{gameweek}_aggregated_player_form.json"
    lookup_path = PROJECT_ROOT / "lookup" / "element_lookup.json"
    for path in (aggregate_path, lookup_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path.relative_to(PROJECT_ROOT)}")

    players = load_json(aggregate_path)
    transcript_extraction = load_json(transcript_extraction_path) if transcript_extraction_path.exists() else {"recommendations": [], "sources": []}
    text_extraction = load_json(text_extraction_path) if text_extraction_path.exists() else {"recommendations": [], "sources": []}

    transcript_ranked, transcript_sources = _source_scores(transcript_extraction, players)
    text_ranked, text_sources = _source_scores(text_extraction, players)
    ranked = _combined_ranked(transcript_ranked, text_ranked)

    transcript_map = _item_index_by_player_id(transcript_ranked)
    text_map = _item_index_by_player_id(text_ranked)

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "player_id",
            "player_name",
            "team",
            "position",
            "transcript_consensus",
            "transcript_mentions",
            "transcript_sources",
            "text_consensus",
            "text_mentions",
            "text_sources",
            "combined_consensus",
            "combined_mentions",
            "combined_sources",
        ]
    )
    for item in ranked:
        player = item["player"]
        transcript_item = transcript_map.get(item["player_id"])
        text_item = text_map.get(item["player_id"])
        writer.writerow(
            [
                item["player_id"],
                player["name"],
                player["team"],
                POSITION_LABELS.get(player.get("position"), player.get("position", "-")),
                f"{transcript_item['consensus_score']:.2f}" if transcript_item else "",
                transcript_item["mention_count"] if transcript_item else 0,
                transcript_item["source_count"] if transcript_item else 0,
                f"{text_item['consensus_score']:.2f}" if text_item else "",
                text_item["mention_count"] if text_item else 0,
                text_item["source_count"] if text_item else 0,
                f"{item['consensus_score']:.2f}",
                item["mention_count"],
                item["source_count"],
            ]
        )
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic FPL consensus scores.")
    parser.add_argument("--gameweek", type=int, required=True, help="Upcoming gameweek number.")
    parser.add_argument("--output", type=Path, help="Optional output path.")
    args = parser.parse_args()

    report_dir = PROJECT_ROOT / "reports" / f"gw_{args.gameweek}"
    csv_path = args.output or report_dir / f"gw_{args.gameweek}_consensus.csv"
    markdown_path = report_dir / f"gw_{args.gameweek}_report.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(build_consensus_csv(args.gameweek), encoding="utf-8")
    markdown_path.write_text(build_report(args.gameweek), encoding="utf-8")
    print(f"Consensus CSV written to {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Markdown report written to {markdown_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()