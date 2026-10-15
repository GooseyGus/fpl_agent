"""Rank the current FPL squad by overall consensus.

This reads the saved squad picks JSON, joins each player against the consensus
CSV, and prints a ranked table sorted by combined/overall consensus.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


from fpl_hq.paths import PROJECT_ROOT, REPORTS_DIR, gw_dir
REPORTS_DIR = REPORTS_DIR


def default_squad_path(gameweek: int) -> Path:
	# The squad picks available before gameweek N are those from the completed gameweek N-1.
	previous = gameweek - 1
	return gw_dir(previous) / "data" / f"gw_{previous}_my_team_picks.json"


def default_consensus_candidates(gameweek: int) -> list[Path]:
	return [REPORTS_DIR / f"gw_{gameweek}" / f"gw_{gameweek}_consensus.csv"]


def load_json(path: Path) -> Any:
	with path.open(encoding="utf-8") as handle:
		return json.load(handle)


def resolve_consensus_path(gameweek: int, explicit_path: Path | None = None) -> Path:
	if explicit_path is not None:
		if not explicit_path.exists():
			raise FileNotFoundError(f"Missing consensus file: {explicit_path.relative_to(PROJECT_ROOT)}")
		return explicit_path

	candidates = default_consensus_candidates(gameweek)
	for candidate in candidates:
		if candidate.exists():
			return candidate

	searched = "\n".join(f"- {path.relative_to(PROJECT_ROOT)}" for path in candidates)
	raise FileNotFoundError(f"Could not find a consensus CSV. Tried:\n{searched}")


def load_consensus_rows(path: Path) -> dict[int, dict[str, str]]:
	rows: dict[int, dict[str, str]] = {}
	with path.open(newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			if not row.get("player_id"):
				continue
			rows[int(row["player_id"])] = row
	return rows


def format_float(value: str | None) -> str:
	if value is None or value == "":
		return "-"
	try:
		return f"{float(value):.2f}"
	except ValueError:
		return value


def build_ranked_squad(squad_path: Path, consensus_path: Path) -> list[dict[str, Any]]:
	squad_data = load_json(squad_path)
	consensus_rows = load_consensus_rows(consensus_path)

	ranked: list[dict[str, Any]] = []
	for pick in squad_data.get("picks", []):
		player_id = int(pick["element"])
		consensus_row = consensus_rows.get(player_id, {})
		combined_consensus = consensus_row.get("combined_consensus")
		ranked.append(
			{
				"squad_position": int(pick.get("position", 0) or 0),
				"player_id": player_id,
				"player_name": consensus_row.get("player_name", f"Player {player_id}"),
				"team": consensus_row.get("team", "-"),
				"position": consensus_row.get("position", "-"),
				"overall_consensus": float(combined_consensus) if combined_consensus not in (None, "") else -1.0,
				"overall_consensus_display": format_float(combined_consensus),
				"combined_mentions": consensus_row.get("combined_mentions", "0"),
				"combined_sources": consensus_row.get("combined_sources", "0"),
				"is_captain": bool(pick.get("is_captain")),
				"is_vice_captain": bool(pick.get("is_vice_captain")),
				"multiplier": int(pick.get("multiplier", 0) or 0),
			}
		)

	return sorted(
		ranked,
		key=lambda item: (
			item["overall_consensus"],
			item["multiplier"],
			item["squad_position"],
			item["player_name"],
		),
		reverse=True,
	)


def print_ranked_squad(rows: list[dict[str, Any]]) -> None:
	headers = [
		"rank",
		"squad_position",
		"player_name",
		"team",
		"position",
		"overall_consensus",
		"combined_mentions",
		"combined_sources",
		"captain",
		"vice_captain",
		"multiplier",
	]
	print(",".join(headers))
	for index, row in enumerate(rows, start=1):
		print(
			",".join(
				[
					str(index),
					str(row["squad_position"]),
					row["player_name"],
					row["team"],
					row["position"],
					row["overall_consensus_display"],
					str(row["combined_mentions"]),
					str(row["combined_sources"]),
					"yes" if row["is_captain"] else "no",
					"yes" if row["is_vice_captain"] else "no",
					str(row["multiplier"]),
				]
			)
		)


def main() -> None:
	parser = argparse.ArgumentParser(description="Rank the current FPL squad by overall consensus.")
	parser.add_argument("--gameweek", type=int, required=True, help="Upcoming gameweek number.")
	parser.add_argument("--squad-path", type=Path, help="Optional explicit squad JSON path (defaults to the previous gameweek's picks).")
	parser.add_argument("--consensus-path", type=Path, help="Optional explicit consensus CSV path.")
	args = parser.parse_args()

	squad_path = args.squad_path or default_squad_path(args.gameweek)
	if not squad_path.exists():
		raise FileNotFoundError(f"Missing squad file: {squad_path.relative_to(PROJECT_ROOT)}")

	consensus_path = resolve_consensus_path(args.gameweek, args.consensus_path)
	ranked_rows = build_ranked_squad(squad_path, consensus_path)

	print(f"Squad file: {squad_path.relative_to(PROJECT_ROOT)}")
	print(f"Consensus file: {consensus_path.relative_to(PROJECT_ROOT)}")
	print_ranked_squad(ranked_rows)


if __name__ == "__main__":
	main()