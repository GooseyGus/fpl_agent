"""Extract explicit head-to-head player comparisons for Bradley-Terry scoring.

Runs on the same retained analysis text as 4_extraction. The model maps spoken
names onto the canonical roster; anything it leaves unmapped goes through the
alias resolver used by the report.

Backfill a past gameweek from saved files:

    python -m fpl_hq.consensus.transcripts.5_comparisons --gameweek 5
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from fpl_hq.paths import ENV_FILE, PROJECT_ROOT, REFERENCE_DIR, gw_data_dir, gw_dir
from fpl_hq.consensus.llm import MODEL, DETERMINISTIC

COMPARISON_PROMPT_VERSION = "v2"
MAX_CONCURRENT_COMPARISONS = 6

COMPARISON_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "transcript_player_comparisons",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "comparisons": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        # Order matters: quote, then spoken names, then roster names.
                        "properties": {
                            "supporting_quote": {"type": "string"},
                            "player_a_as_spoken": {"type": "string"},
                            "player_b_as_spoken": {"type": "string"},
                            "player_a": {"type": "string"},
                            "player_b": {"type": "string"},
                            "relation": {"type": "string", "enum": ["better", "equal"]},
                            "certainty": {"type": "integer", "enum": [1, 2]},
                            "horizon": {"type": "string", "enum": ["next_gameweek", "longer"]},
                        },
                        "required": [
                            "supporting_quote",
                            "player_a_as_spoken",
                            "player_b_as_spoken",
                            "player_a",
                            "player_b",
                            "relation",
                            "certainty",
                            "horizon",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["comparisons"],
            "additionalProperties": False,
        },
    },
}

COMPARISON_PROMPT_TEMPLATE = """You extract explicit head-to-head comparisons between Fantasy Premier League (FPL) players from transcript analysis.

The upcoming deadline is Gameweek {gameweek}. The text may be grouped under `[SECTION: topic | start-end]` metadata headings; use them as context, never as quotes.

A comparison is a statement by the speaker that one named player will score more FPL points than another named player, or that two named players are of equal value. Extract a comparison only when ALL of these hold:
1. Both players are named, and the speaker evaluates them against each other on expected FPL returns. Price, ownership, or popularity alone is not a comparison.
2. The view is the speaker's own. Skip views attributed to the viewer or to other people ("if you think Haaland is the better premium").
3. Denying certainty is not a comparison. "It's not a guarantee he outscores Bruno" is excluded.
4. "Not better than" or "not miles better than" is `equal`, unless the speaker goes on to say one player is worse.
5. A transfer made to free up funds ("Bruno to Palmer to free up 2 million") is not a comparison, unless the speaker also says the incoming player is better.
6. A ranked list yields only adjacent pairs: first vs second, second vs third, and so on.
7. `equal` needs explicit equivalence language ("as good as", "nothing between them", "toss-up", "could go either way"). Co-mentions such as "both are fine options" are not comparisons.
8. Ignore recaps of past gameweeks unless the speaker turns them into a forward-looking comparison.

Captaincy choices count as comparisons ("I'd captain Haaland over Salah" means Haaland is better for the next gameweek).

Fields, in this order:
- `supporting_quote`: a direct, verbatim quote from ANALYSIS_TEXT naming or clearly referring to both players. Include consecutive sentences if needed.
- `player_a_as_spoken`, `player_b_as_spoken`: each name exactly as it appears in the quote.
- `player_a`, `player_b`: the matching full canonical name copied from PLAYERS, without the team and position. Transcripts are auto-generated, so names are often misheard ("Munos" is "Víctor Muñoz", "Jubury Hall" is "Kiernan Dewsbury-Hall"); use the team and position in PLAYERS and the context of the discussion to pick the right player. Never copy the spoken spelling here unless it is also the canonical name. For `better`, `player_a` is the better player.
- `relation`: `better` or `equal`.
- `certainty`: 1 when hedged ("probably", "I'd lean", "slightly", "on paper"); 2 when clear.
- `horizon`: `next_gameweek` when the comparison is about Gameweek {gameweek}; `longer` for later gameweeks, a run of fixtures, or the season.

Record each distinct comparison once; do not repeat the same pair and horizon within this transcript. If you cannot identify a player in PLAYERS, skip the comparison. Return an empty list when there are no comparisons that satisfy every rule.

SOURCE:
{source}

PLAYERS:
{players}

ANALYSIS_TEXT:
{analysis_text}
"""


def _cache_key(*parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:24]


def _read_cache(cache_dir: Path, key: str):
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_cache(cache_dir: Path, key: str, payload: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _name_key(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in decomposed if character.isalnum())


def _quote_key(value: str) -> str:
    """Compare quotes without case, punctuation, or the inline `[mm:ss]` timestamps models drop."""
    value = re.sub(r"\[\d{1,2}:\d{2}(?::\d{2})?\]", " ", value)
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def player_roster(players: dict[str, dict[str, Any]]) -> list[str]:
    """Roster lines like `Cole Palmer (Chelsea, MID)` from the aggregated player form."""
    positions = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    return sorted(
        f"{player['name']} ({player.get('team', '?')}, {positions.get(player.get('position'), '?')})"
        for player in players.values()
    )


def _normalise_response(
    parsed: dict[str, Any],
    roster: list[str],
    analysis_text: str,
    resolve_alias: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Keep comparisons between two distinct roster players whose quote appears in the text."""
    canonical = {_name_key(line.rsplit(" (", 1)[0]): line.rsplit(" (", 1)[0] for line in roster}

    def resolve(raw_name: str) -> str | None:
        raw_name = raw_name.rsplit(" (", 1)[0].strip()
        name = canonical.get(_name_key(raw_name))
        if name is None and resolve_alias is not None and raw_name:
            name = resolve_alias(raw_name)
        return name

    text_key = _quote_key(analysis_text)
    comparisons: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    dropped = {"unknown_player": 0, "same_player": 0, "quote_not_found": 0, "duplicate": 0}
    for item in parsed.get("comparisons", []):
        if not isinstance(item, dict):
            continue
        player_a = resolve(str(item.get("player_a", ""))) or resolve(str(item.get("player_a_as_spoken", "")))
        player_b = resolve(str(item.get("player_b", ""))) or resolve(str(item.get("player_b_as_spoken", "")))
        relation = item.get("relation")
        certainty = item.get("certainty")
        horizon = item.get("horizon")
        quote = str(item.get("supporting_quote", "")).strip()
        if relation not in {"better", "equal"} or certainty not in {1, 2} or horizon not in {"next_gameweek", "longer"}:
            continue
        if player_a is None or player_b is None:
            dropped["unknown_player"] += 1
            continue
        if player_a == player_b:
            dropped["same_player"] += 1
            continue
        if not quote or _quote_key(quote) not in text_key:
            dropped["quote_not_found"] += 1
            continue
        pair = (player_a, player_b) if relation == "better" else tuple(sorted((player_a, player_b)))
        key = (*pair, relation, horizon)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        comparisons.append(
            {
                "player_a": pair[0],
                "player_b": pair[1],
                "relation": relation,
                "certainty": certainty,
                "horizon": horizon,
                "supporting_quote": quote,
            }
        )
    return {"comparisons": comparisons, "dropped": dropped}


def extract_comparisons(
    client: Any,
    model: str,
    cache_dir: Path,
    source: str,
    analysis_text: str,
    roster: list[str],
    gameweek: int,
    resolve_alias: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    players_context = "\n".join(roster)
    key = _cache_key("comparisons", COMPARISON_PROMPT_VERSION, model, str(gameweek), source, analysis_text, players_context)
    cached = _read_cache(cache_dir, key)
    if cached is None:
        response = client.chat.completions.create(
            model=model,
            response_format=COMPARISON_RESPONSE_FORMAT,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {
                    "role": "user",
                    "content": COMPARISON_PROMPT_TEMPLATE.format(
                        gameweek=gameweek,
                        source=source,
                        players=players_context,
                        analysis_text=analysis_text,
                    ),
                },
            ],
            **DETERMINISTIC,
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
        cached = parsed if isinstance(parsed, dict) else {}
        # Cache the raw response so the normalisation rules can change without new calls.
        _write_cache(cache_dir, key, cached)
    return _normalise_response(cached, roster, analysis_text, resolve_alias)


def write_transcript_comparisons_output(
    output_path: Path,
    *,
    gameweek: int,
    comparisons_by_source: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    comparisons = [
        {"source": source, **comparison}
        for source, result in sorted(comparisons_by_source.items())
        for comparison in result["comparisons"]
    ]
    dropped: dict[str, int] = {}
    for result in comparisons_by_source.values():
        for reason, count in result.get("dropped", {}).items():
            dropped[reason] = dropped.get(reason, 0) + count
    output = {
        "gameweek": gameweek,
        "prompt_version": COMPARISON_PROMPT_VERSION,
        "sources": sorted(comparisons_by_source),
        "counts": {
            "sources": len(comparisons_by_source),
            "comparisons": len(comparisons),
            "dropped": dropped,
        },
        "comparisons": comparisons,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def alias_resolver(players: dict[str, dict[str, Any]]) -> Callable[[str], str | None]:
    """Resolve a spoken name only through an exact `data/reference/name_aliases.json` hit.

    The report's resolver also guesses from last names and fuzzy matches, which is
    fine for single-player recommendations but would silently pair the wrong players
    here ("Ken and Jubury Hall" -> Lewis Hall), so it is not reused.
    """
    aliases = json.loads((REFERENCE_DIR / "name_aliases.json").read_text(encoding="utf-8"))
    roster_names = {player["name"] for player in players.values()}

    def resolve(raw_name: str) -> str | None:
        name = aliases.get(re.sub(r"[^a-z]", "", raw_name.lower()))
        return name if name in roster_names else None

    return resolve


def saved_analysis_texts(gameweek: int) -> dict[str, str]:
    """Retained text per source from the saved selection output, else the raw transcripts.

    The raw fallback exists for gameweeks saved before the selection output was written
    and whose segmentation cache no longer matches the current prompt version.
    """
    gameweek_dir = gw_dir(gameweek)
    selection_path = gameweek_dir / "data" / f"gw_{gameweek}_transcript_selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        return {
            record["source"]: "\n\n".join(
                f"{section['heading']}\n{section['text']}" for section in record.get("retained_sections", [])
            )
            for record in selection.get("source_records", [])
        }
    segmentation = importlib.import_module("fpl_hq.consensus.transcripts.1_segmentation")
    return {
        transcript["source"]: transcript["text"]
        for transcript in segmentation.collect_transcripts(gameweek_dir / "transcripts")
    }


def main() -> None:
    from dotenv import load_dotenv
    from openai import OpenAI

    parser = argparse.ArgumentParser(description="Extract player comparisons for one gameweek from saved files.")
    parser.add_argument("--gameweek", type=int, required=True)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    data_dir = gw_data_dir(args.gameweek)
    cache_dir = gw_dir(args.gameweek) / "transcript_cache" / "comparison_cache"
    players = json.loads((data_dir / f"gw_{args.gameweek}_aggregated_player_form.json").read_text(encoding="utf-8"))
    roster = player_roster(players)
    resolve_alias = alias_resolver(players)
    texts = saved_analysis_texts(args.gameweek)
    inputs = [(source, text) for source, text in sorted(texts.items()) if text.strip()]
    print(f"GW{args.gameweek}: {len(inputs)} source(s) with retained text; roster of {len(roster)} players")

    load_dotenv(ENV_FILE)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env.")
    client = OpenAI(max_retries=5)

    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_COMPARISONS) as executor:
        futures = {
            executor.submit(
                extract_comparisons, client, args.model, cache_dir, source, text, roster, args.gameweek, resolve_alias
            ): source
            for source, text in inputs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            source = futures[future]
            try:
                results[source] = future.result()
                print(f"[{completed}/{len(futures)}] {source}: {len(results[source]['comparisons'])} comparison(s)")
            except Exception as exc:
                print(f"[{completed}/{len(futures)}] FAILED {source}: {type(exc).__name__}: {exc}")

    output_path = data_dir / f"gw_{args.gameweek}_transcript_comparisons.json"
    output = write_transcript_comparisons_output(output_path, gameweek=args.gameweek, comparisons_by_source=results)
    print(f"Wrote {output['counts']['comparisons']} comparison(s) to {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Dropped: {output['counts']['dropped']}")


if __name__ == "__main__":
    main()
