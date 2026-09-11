from __future__ import annotations

# Name resolution work.

import hashlib
import json
import re
from difflib import get_close_matches
from pathlib import Path
from types import SimpleNamespace
from typing import Any

NAME_CORRECTION_PROMPT_VERSION = "v15"
NAME_CORRECTION_CANDIDATE_LIMIT = 5
NAME_CORRECTION_CONTEXT_RADIUS = 1
MAX_CONCURRENT_NAME_RESOLUTION = 6
AUTO_RESOLVE_CONFIDENCE_THRESHOLD = 0.88
AUTO_RESOLVE_MARGIN_THRESHOLD = 0.08

VALID_NAME_CORRECTION_MENTION_TYPES = {
    "recommendation",
    "captain",
    "vice_captain",
    "triple_captain",
    "avoid_captain",
}

NAME_CORRECTION_PROMPT_TEMPLATE = """You analyse a retained FPL-analysis transcript before a later extraction step. Your job is only to resolve transcript-spelled player names to canonical FPL player identities and to suggest alias entries that could be added to the alias script.

The transcript is grouped under `[SECTION: topic | start-end]` metadata headings. Use each broader topic as context when resolving names across nearby paragraphs. These headings were generated during segmentation and are not source transcript text.

This text comes from a raw auto-generated transcript. Player names are ASR-style phonetic renderings of spoken names, not deliberate typed spellings. Treat each raw player mention as a sound-alike transcript surface form. Do not assume normal typo-correction logic unless the phonetic evidence, alias map, or nearby transcript context makes the identity genuinely clear.

You are given:
- a transcript source id
- canonical FPL player ids and names from the element lookup
- a map of common ASR or nickname aliases to canonical names

Identify every player-specific mention that is relevant to transfer, selection, hold/sell, or captaincy discussion. Ignore team-only discussion and generic role terms that do not point to a player.

Return strict JSON with exactly one key: `mentions`.

Each item in `mentions` must have:
- `raw_name`: the player name as it appears in the transcript
- `timestamp`: the [MM:SS] marker of the paragraph containing the mention, or null
- `mention_type`: one of `recommendation`, `captain`, `vice_captain`, `triple_captain`, `avoid_captain`
- `resolution_status`: `resolved` or `unresolved`
- `selected_player_id`: the chosen canonical FPL player id, or null if unresolved
- `selected_player_name`: the chosen canonical FPL player name, or null if unresolved
- `confidence`: a number from 0 to 1
- `ranked_candidates`: an array of up to 5 candidate objects with `player_id`, `player_name`, and `confidence`
- `suggested_aliases`: an array of up to 5 alias suggestions ranked by confidence descending. Each suggestion must have `alias`, `canonical_player_id`, `canonical_player_name`, `confidence`, and `reason`. Only include aliases that would be useful to add to lookup/name_aliases.json.
- `reason`: a short explanation grounded in the alias map, canonical names, and nearby transcript context

Resolution rules:
- Use the alias map when it gives a clear correction.
- Prefer exact canonical player names when they already appear in the transcript.
- Use nearby transcript context to disambiguate phonetic renderings or same-surname players.
- Be conservative with low-confidence phonetic guesses; if the evidence is weak, return `unresolved`.
- If more than one candidate remains materially plausible, mark the mention unresolved.
- Keep `raw_name` faithful to the transcript and do not rewrite quotes.
- Rank both `ranked_candidates` and `suggested_aliases` by confidence descending.
- Suggested aliases should be transcript-faithful, normalized for the alias script when possible, and conservative enough that they would be worth adding to name_aliases.json.
- If no sensible alias exists, return an empty `suggested_aliases` array.

SOURCE:
{source}

ALIASES:
{aliases}

CANONICAL_PLAYERS:
{players}

TRANSCRIPT:
{text}"""


def load_name_aliases(project_root: Path | None = None) -> dict[str, str]:
    root = project_root or Path(__file__).resolve().parent
    alias_path = root / "lookup" / "name_aliases.json"
    with alias_path.open(encoding="utf-8") as file:
        return json.load(file)


def _normalise_lookup_key(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def clamp(value: float, lower: float = 0, upper: float = 100) -> float:
    return max(lower, min(value, upper))


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


def _extract_timestamp(paragraph: str) -> str | None:
    match = re.match(r"\[(\d{1,2}:\d{2})\]", paragraph)
    return match.group(1) if match else None


def build_name_resolution_helpers(
    *,
    name_aliases: dict[str, str],
    canonical_player_names: dict[str, int],
    canonical_player_id_to_name: dict[int, str],
    player_metadata: dict[int, dict[str, Any]],
    candidate_limit: int = NAME_CORRECTION_CANDIDATE_LIMIT,
    context_radius: int = NAME_CORRECTION_CONTEXT_RADIUS,
    auto_resolve_confidence_threshold: float = AUTO_RESOLVE_CONFIDENCE_THRESHOLD,
    auto_resolve_margin_threshold: float = AUTO_RESOLVE_MARGIN_THRESHOLD,
) -> SimpleNamespace:
    def _candidate_record(player_id: int, player_name: str, confidence: float) -> dict:
        return {
            "player_id": player_id,
            "player_name": player_name,
            "confidence": round(clamp(confidence, 0, 1), 4),
        }

    def _normalise_phrase(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()

    def build_timestamp_contexts(text: str) -> dict[str, str]:
        paragraphs = [paragraph.strip() for paragraph in re.split(r"(?=\[\d{1,2}:\d{2}\])", text) if paragraph.strip()]
        contexts: dict[str, str] = {}
        for index, paragraph in enumerate(paragraphs):
            timestamp = _extract_timestamp(paragraph)
            if timestamp is None:
                continue
            start = max(0, index - context_radius)
            end = min(len(paragraphs), index + context_radius + 1)
            contexts[timestamp] = "\n\n".join(paragraphs[start:end])
        return contexts

    team_context_aliases = {
        "arsenal": {"arsenal", "arteta"},
        "aston villa": {"aston villa", "villa", "unai", "emery"},
        "bournemouth": {"bournemouth"},
        "brentford": {"brentford", "brenford"},
        "brighton": {"brighton", "herzler"},
        "chelsea": {"chelsea"},
        "crystal palace": {"crystal palace", "palace", "glazner"},
        "everton": {"everton"},
        "fulham": {"fulham"},
        "hull city": {"hull"},
        "ipswich town": {"ipswich", "ipsswich"},
        "leeds": {"leeds"},
        "liverpool": {"liverpool"},
        "man city": {"man city", "city", "guardiola", "pep"},
        "man utd": {"man utd", "man united", "united", "amorim"},
        "newcastle": {"newcastle", "howe"},
        "nott'm forest": {"forest", "nottingham forest", "nuno"},
        "spurs": {"spurs", "tottenham", "ange"},
        "sunderland": {"sunderland", "sun london"},
    }
    position_context_keywords = {
        1: {"goalkeeper", "goalkeepers", "keeper", "keepers"},
        2: {"defender", "defenders", "centre back", "center back", "left back", "right back"},
        3: {"midfielder", "midfielders", "midfield", "winger", "wingers", "number ten"},
        4: {"forward", "forwards", "striker", "strikers", "number nine", "penalty taker", "penalty takers"},
    }
    attacking_context_keywords = {"penalties", "penalty", "number nine", "striker", "forward"}

    def _team_context_keywords(team_name: str) -> set[str]:
        normalised_team_name = _normalise_phrase(team_name)
        keywords = set(team_context_aliases.get(normalised_team_name, set()))
        if normalised_team_name:
            keywords.add(normalised_team_name)
            for token in normalised_team_name.split():
                if len(token) >= 5:
                    keywords.add(token)
        return {keyword for keyword in keywords if keyword}

    def _extract_context_teams(context_text: str) -> set[str]:
        normalised_context = f" {_normalise_phrase(context_text)} "
        found_teams: set[str] = set()
        for team_name, keywords in team_context_aliases.items():
            if any(f" {keyword} " in normalised_context for keyword in keywords):
                found_teams.add(team_name)
        return found_teams

    def _context_position_bonus(position: int, context_text: str) -> float:
        normalised_context = _normalise_phrase(context_text)
        bonus = 0.0
        if any(keyword in normalised_context for keyword in position_context_keywords.get(position, set())):
            bonus += 0.12 if position == 1 else 0.08
        if position in {3, 4} and any(keyword in normalised_context for keyword in attacking_context_keywords):
            bonus += 0.06
        return bonus

    def _score_candidate_with_context(player_id: int, base_confidence: float, context_text: str) -> float:
        metadata = player_metadata.get(player_id, {})
        position = int(metadata.get("position", 0) or 0)
        team_name = str(metadata.get("team", ""))
        mentioned_teams = _extract_context_teams(context_text)
        bonus = _context_position_bonus(position, context_text)

        team_keywords = _team_context_keywords(team_name)
        normalised_context = f" {_normalise_phrase(context_text)} "
        has_team_match = any(f" {keyword} " in normalised_context for keyword in team_keywords)
        if has_team_match:
            bonus += 0.14
        elif len(mentioned_teams) == 1 and _normalise_phrase(team_name) not in mentioned_teams:
            bonus -= 0.1

        return round(clamp(base_confidence + bonus, 0, 1), 4)

    def resolve_player_name(name: str, player_names: dict[str, int]) -> int | None:
        normalised_names = {re.sub(r"[^a-z ]", "", key.lower()): value for key, value in player_names.items()}
        squashed_input = _normalise_lookup_key(name)
        if squashed_input in name_aliases:
            name = name_aliases[squashed_input]
        normalised_name = re.sub(r"[^a-z ]", "", name.lower())
        if normalised_name in normalised_names:
            return normalised_names[normalised_name]

        last_names: dict[str, list[int]] = {}
        for full_name, player_id in player_names.items():
            last_name = re.sub(r"[^a-z]", "", full_name.lower().split()[-1])
            last_names.setdefault(last_name, []).append(player_id)

        input_last_name = re.sub(r"[^a-z]", "", name.lower().split()[-1])
        if input_last_name in last_names and len(last_names[input_last_name]) == 1:
            return last_names[input_last_name][0]

        close_last_names = get_close_matches(input_last_name, last_names.keys(), n=1, cutoff=0.75)
        if close_last_names and len(last_names[close_last_names[0]]) == 1:
            return last_names[close_last_names[0]][0]

        matches = get_close_matches(normalised_name, normalised_names.keys(), n=1, cutoff=0.78)
        return normalised_names[matches[0]] if matches else None

    def build_name_resolution_candidates(raw_name: str, player_names: dict[str, int], context_text: str = "") -> list[dict]:
        raw_name = raw_name.strip()
        if not raw_name:
            return []

        normalised_full_names: dict[str, list[int]] = {}
        last_names: dict[str, list[int]] = {}
        token_candidates: dict[str, list[int]] = {}
        compact_tail_candidates: dict[str, list[int]] = {}
        for full_name, player_id in player_names.items():
            normalised_full_name = re.sub(r"[^a-z ]", " ", full_name.lower()).strip()
            normalised_full_names.setdefault(normalised_full_name, []).append(player_id)

            name_parts = [part for part in normalised_full_name.split() if part]
            if not name_parts:
                continue
            last_names.setdefault(name_parts[-1], []).append(player_id)
            for part in name_parts:
                if len(part) >= 4:
                    token_candidates.setdefault(part, []).append(player_id)
            if len(name_parts) >= 2:
                compact_tail = "".join(name_parts[1:])
                if len(compact_tail) >= 6:
                    compact_tail_candidates.setdefault(compact_tail, []).append(player_id)

        squashed_input = _normalise_lookup_key(raw_name)
        normalised_input = re.sub(r"[^a-z ]", " ", raw_name.lower()).strip()
        input_parts = [part for part in normalised_input.split() if part]
        input_last_name = input_parts[-1] if input_parts else squashed_input
        compacted_input = "".join(input_parts) if input_parts else squashed_input

        candidates_by_id: dict[int, dict] = {}

        def add_candidate(player_id: int, confidence: float) -> None:
            player_name = canonical_player_id_to_name.get(player_id)
            if player_name is None:
                return
            scored_confidence = _score_candidate_with_context(player_id, confidence, context_text)
            record = _candidate_record(player_id, player_name, scored_confidence)
            existing = candidates_by_id.get(player_id)
            if existing is None or record["confidence"] > existing["confidence"]:
                candidates_by_id[player_id] = record

        alias_name = name_aliases.get(squashed_input)
        if alias_name is not None:
            alias_player_id = player_names.get(alias_name)
            if alias_player_id is not None:
                add_candidate(alias_player_id, 0.99)

        alias_cutoff = 0.82
        for alias_key in get_close_matches(squashed_input, list(name_aliases.keys()), n=3, cutoff=alias_cutoff):
            alias_target_name = name_aliases[alias_key]
            alias_player_id = player_names.get(alias_target_name)
            if alias_player_id is not None:
                add_candidate(alias_player_id, 0.86)
        for alias_key, alias_target_name in name_aliases.items():
            if len(squashed_input) >= 4 and (alias_key.endswith(squashed_input) or squashed_input in alias_key):
                alias_player_id = player_names.get(alias_target_name)
                if alias_player_id is not None:
                    add_candidate(alias_player_id, 0.84)

        full_name_matches = normalised_full_names.get(normalised_input, [])
        if len(full_name_matches) == 1:
            add_candidate(full_name_matches[0], 0.97)

        if len(input_parts) >= 2:
            for close_compact_tail in get_close_matches(
                compacted_input,
                list(compact_tail_candidates.keys()),
                n=3,
                cutoff=0.78,
            ):
                candidate_ids = compact_tail_candidates[close_compact_tail]
                if len(candidate_ids) == 1:
                    add_candidate(candidate_ids[0], 0.9)

            for full_name, player_id in player_names.items():
                full_parts = [part for part in re.sub(r"[^a-z ]", " ", full_name.lower()).split() if part]
                if len(full_parts) < 2:
                    continue
                first_matches = full_parts[0].startswith(input_parts[0]) or input_parts[0].startswith(full_parts[0][:3])
                last_matches = full_parts[-1] == input_parts[-1]
                if first_matches and last_matches:
                    add_candidate(player_id, 0.98)
                elif last_matches and any(part.startswith(input_parts[0]) for part in full_parts[1:-1]):
                    add_candidate(player_id, 0.76)

        if input_last_name in last_names:
            last_name_matches = last_names[input_last_name]
            if len(last_name_matches) == 1:
                add_candidate(last_name_matches[0], 0.95)
            else:
                for player_id in last_name_matches[:candidate_limit]:
                    add_candidate(player_id, 0.72)

        if len(input_last_name) >= 2:
            prefix_matches = [
                player_id
                for last_name, player_ids in last_names.items()
                if last_name.startswith(input_last_name)
                and len(last_name) - len(input_last_name) <= 2
                for player_id in player_ids
            ]
            unique_prefix_matches = sorted(set(prefix_matches))
            if len(unique_prefix_matches) == 1:
                add_candidate(unique_prefix_matches[0], 0.88 if len(input_last_name) <= 3 else 0.9)
            elif len(unique_prefix_matches) > 1:
                for player_id in unique_prefix_matches[:candidate_limit]:
                    add_candidate(player_id, 0.74)

        close_last_name_cutoff = 0.86 if len(input_last_name) > 3 else 0.72
        for close_last_name in get_close_matches(input_last_name, list(last_names.keys()), n=4, cutoff=close_last_name_cutoff):
            if len(input_last_name) <= 3 and not (
                close_last_name.startswith(input_last_name) and len(close_last_name) - len(input_last_name) <= 1
            ):
                continue
            candidate_ids = last_names[close_last_name]
            if len(candidate_ids) == 1:
                add_candidate(candidate_ids[0], 0.82)

        for close_full_name in get_close_matches(normalised_input, list(normalised_full_names.keys()), n=3, cutoff=0.88):
            candidate_ids = normalised_full_names[close_full_name]
            if len(candidate_ids) == 1:
                add_candidate(candidate_ids[0], 0.8)

        if len(input_parts) >= 2:
            for full_name, player_id in player_names.items():
                full_parts = [part for part in re.sub(r"[^a-z ]", " ", full_name.lower()).split() if part]
                if all(any(part.startswith(input_part) or input_part.startswith(part) for part in full_parts) for input_part in input_parts):
                    add_candidate(player_id, 0.74)
        elif input_last_name in token_candidates:
            token_match_ids = sorted(set(token_candidates[input_last_name]))
            if len(token_match_ids) == 1:
                add_candidate(token_match_ids[0], 0.9 if len(input_last_name) >= 6 else 0.82)
            else:
                for player_id in token_match_ids[:candidate_limit]:
                    add_candidate(player_id, 0.68)

        return sorted(
            candidates_by_id.values(),
            key=lambda item: (-item["confidence"], item["player_name"]),
        )[:candidate_limit]

    def build_name_resolution_reason(
        raw_name: str,
        selected_player_name: str | None,
        selected_player_id: int | None,
        candidates: list[dict],
        context_text: str,
    ) -> str:
        squashed_input = _normalise_lookup_key(raw_name)
        alias_name = name_aliases.get(squashed_input)
        team_context = sorted(_extract_context_teams(context_text))
        context_suffix = f" Nearby context mentions {', '.join(team_context)}." if team_context else ""

        if selected_player_name is None:
            if not candidates:
                return f"No supported canonical candidate found for '{raw_name}'.{context_suffix}".strip()
            if len(candidates) == 1:
                candidate = candidates[0]
                return (
                    f"Only a low-confidence phonetic candidate for '{raw_name}' was found "
                    f"({candidate['player_name']} at {candidate['confidence']:.2f}); it stays unresolved.{context_suffix}"
                ).strip()
            return f"No alias match for '{raw_name}' and multiple plausible canonical candidates remain.{context_suffix}".strip()

        if alias_name == selected_player_name:
            return f"Alias '{raw_name}' maps to {selected_player_name}.{context_suffix}".strip()

        if selected_player_id is not None:
            metadata = player_metadata.get(selected_player_id, {})
            team_name = str(metadata.get("team", "")).strip()
            position = int(metadata.get("position", 0) or 0)
            if len(candidates) == 1:
                return (
                    f"Unique supported canonical match for '{raw_name}' resolves to {selected_player_name}"
                    f" ({team_name}, position {position}).{context_suffix}"
                ).strip()

        return f"Best supported canonical candidate for '{raw_name}' is {selected_player_name}.{context_suffix}".strip()

    def should_resolve_candidate(ranked_candidates: list[dict], player_id: int | None) -> bool:
        if player_id is None:
            return False

        selected_candidate = next((candidate for candidate in ranked_candidates if candidate["player_id"] == player_id), None)
        if selected_candidate is None:
            return False
        if selected_candidate["confidence"] < auto_resolve_confidence_threshold:
            return False

        competing_confidences = [
            candidate["confidence"]
            for candidate in ranked_candidates
            if candidate["player_id"] != player_id
        ]
        if not competing_confidences:
            return True
        return selected_candidate["confidence"] - max(competing_confidences) >= auto_resolve_margin_threshold

    def _normalise_alias_suggestion(item: dict, default_player_id: int | None = None, default_player_name: str | None = None) -> dict | None:
        alias = str(item.get("alias", "")).strip()
        if not alias:
            return None

        alias = _normalise_lookup_key(alias)
        if not alias:
            return None

        canonical_player_id = item.get("canonical_player_id", item.get("player_id", default_player_id))
        try:
            canonical_player_id = int(canonical_player_id) if canonical_player_id is not None else None
        except (TypeError, ValueError):
            canonical_player_id = None

        canonical_player_name = str(
            item.get("canonical_player_name", item.get("player_name", default_player_name or ""))
        ).strip()
        if not canonical_player_name and canonical_player_id is not None:
            canonical_player_name = canonical_player_id_to_name.get(canonical_player_id, "")
        if not canonical_player_name:
            return None

        if canonical_player_id is None:
            canonical_player_id = canonical_player_names.get(canonical_player_name)
        if canonical_player_id is None:
            return None

        confidence = clamp(float(item.get("confidence", 0)), 0, 1)
        reason = str(item.get("reason", "")).strip()
        raw_alias = str(item.get("raw_alias", "")).strip() or None
        suggestion = {
            "alias": alias,
            "canonical_player_id": canonical_player_id,
            "canonical_player_name": canonical_player_name,
            "confidence": round(confidence, 4),
            "reason": reason,
        }
        if raw_alias and raw_alias != alias:
            suggestion["raw_alias"] = raw_alias
        return suggestion

    def _fallback_alias_suggestion(raw_name: str, selected_player_id: int | None, selected_player_name: str | None, confidence: float) -> dict | None:
        if selected_player_id is None or selected_player_name is None:
            return None

        alias = _normalise_lookup_key(raw_name)
        if not alias:
            return None
        if name_aliases.get(alias) == selected_player_name:
            return None

        return {
            "alias": alias,
            "canonical_player_id": selected_player_id,
            "canonical_player_name": selected_player_name,
            "confidence": round(clamp(confidence, 0, 1), 4),
            "reason": f"Transcript surface form '{raw_name}' maps to {selected_player_name}.",
        }

    def _normalise_name_correction_record(source: str, item: dict, context_text: str = "") -> dict | None:
        raw_name = str(item.get("raw_name", "")).strip()
        if not raw_name:
            return None

        mention_type = str(item.get("mention_type", "recommendation")).strip().lower()
        if mention_type not in VALID_NAME_CORRECTION_MENTION_TYPES:
            mention_type = "recommendation"

        timestamp = item.get("timestamp")
        timestamp = str(timestamp).strip() if isinstance(timestamp, str) else None
        if timestamp == "":
            timestamp = None
        elif timestamp is not None:
            timestamp = timestamp.strip("[]")

        resolution_status = str(item.get("resolution_status", "unresolved")).strip().lower()
        if resolution_status != "resolved":
            resolution_status = "unresolved"

        deterministic_candidates = build_name_resolution_candidates(raw_name, canonical_player_names, context_text)
        deterministic_candidate_ids = {candidate["player_id"] for candidate in deterministic_candidates}

        model_selected_player_id = item.get("selected_player_id")
        try:
            model_selected_player_id = int(model_selected_player_id) if model_selected_player_id is not None else None
        except (TypeError, ValueError):
            model_selected_player_id = None

        selected_player_id = model_selected_player_id
        if selected_player_id not in canonical_player_id_to_name:
            selected_player_id = None
            resolution_status = "unresolved"

        if deterministic_candidate_ids and selected_player_id not in deterministic_candidate_ids:
            selected_player_id = None
            resolution_status = "unresolved"

        ranked_candidates_by_id = {candidate["player_id"]: candidate for candidate in deterministic_candidates}
        if not deterministic_candidates:
            for candidate in item.get("ranked_candidates", [])[:candidate_limit]:
                candidate_id = candidate.get("player_id")
                try:
                    candidate_id = int(candidate_id)
                except (TypeError, ValueError):
                    continue
                player_name = canonical_player_id_to_name.get(candidate_id)
                if player_name is None:
                    continue
                ranked_candidates_by_id[candidate_id] = _candidate_record(
                    candidate_id,
                    player_name,
                    float(candidate.get("confidence", 0)),
                )

        ranked_candidates = sorted(
            ranked_candidates_by_id.values(),
            key=lambda candidate: (-candidate["confidence"], candidate["player_name"]),
        )[:candidate_limit]

        if selected_player_id is not None and not should_resolve_candidate(ranked_candidates, selected_player_id):
            selected_player_id = None
            resolution_status = "unresolved"
        elif selected_player_id is not None:
            resolution_status = "resolved"

        if selected_player_id is None and deterministic_candidates:
            top_candidate = deterministic_candidates[0]
            if should_resolve_candidate(ranked_candidates, top_candidate["player_id"]):
                selected_player_id = top_candidate["player_id"]
                resolution_status = "resolved"

        selected_player_name = None
        if selected_player_id is not None:
            selected_player_name = canonical_player_id_to_name[selected_player_id]
            selected_candidate_confidence = ranked_candidates_by_id.get(selected_player_id, {"confidence": 0.95})["confidence"]
            selected_candidate = _candidate_record(
                selected_player_id,
                selected_player_name,
                selected_candidate_confidence,
            )
            ranked_candidates_by_id[selected_player_id] = selected_candidate
            ranked_candidates = sorted(
                ranked_candidates_by_id.values(),
                key=lambda candidate: (
                    candidate["player_id"] != selected_player_id,
                    -candidate["confidence"],
                    candidate["player_name"],
                ),
            )[:candidate_limit]
            confidence = ranked_candidates[0]["confidence"]
        else:
            confidence = ranked_candidates[0]["confidence"] if ranked_candidates else 0.0
            resolution_status = "unresolved"

        suggested_aliases = []
        for suggestion in item.get("suggested_aliases", [])[:candidate_limit]:
            normalised_suggestion = _normalise_alias_suggestion(
                suggestion,
                default_player_id=selected_player_id,
                default_player_name=selected_player_name,
            )
            if normalised_suggestion is not None:
                suggested_aliases.append(normalised_suggestion)

        if not suggested_aliases:
            fallback_suggestion = _fallback_alias_suggestion(raw_name, selected_player_id, selected_player_name, confidence)
            if fallback_suggestion is not None:
                suggested_aliases.append(fallback_suggestion)

        suggested_aliases = sorted(
            {
                (suggestion["alias"], suggestion["canonical_player_id"]): suggestion
                for suggestion in suggested_aliases
            }.values(),
            key=lambda suggestion: (-suggestion["confidence"], suggestion["alias"], suggestion["canonical_player_name"]),
        )[:candidate_limit]

        reason = build_name_resolution_reason(
            raw_name,
            selected_player_name,
            selected_player_id,
            ranked_candidates,
            context_text,
        )

        mention_key = _normalise_lookup_key(raw_name) or "unknown"
        mention_id = f"{source}|{timestamp or 'unknown'}|{mention_type}|{mention_key}"

        return {
            "mention_id": mention_id,
            "source": source,
            "timestamp": timestamp,
            "raw_name": raw_name,
            "mention_type": mention_type,
            "resolution_status": resolution_status,
            "selected_player_id": selected_player_id,
            "selected_player_name": selected_player_name,
            "confidence": confidence,
            "ranked_candidates": ranked_candidates,
            "suggested_aliases": suggested_aliases,
            "reason": reason,
        }

    def resolve_transcript_names(client: Any, model: str, cache_dir: Path, source: str, analysis_text: str) -> list[dict]:
        key = _cache_key("name_correction", NAME_CORRECTION_PROMPT_VERSION, model, source, analysis_text)
        cached = _read_cache(cache_dir, key)
        if cached is not None:
            return cached.get("mentions", [])

        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {
                    "role": "user",
                    "content": NAME_CORRECTION_PROMPT_TEMPLATE.format(
                        source=source,
                        aliases=json.dumps(name_aliases, ensure_ascii=False, sort_keys=True),
                        players="\n".join(
                            f"{player_id}: {player_name}"
                            for player_id, player_name in sorted(canonical_player_id_to_name.items())
                        ),
                        text=analysis_text,
                    ),
                },
            ],
            temperature=0,
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
        timestamp_contexts = build_timestamp_contexts(analysis_text)

        mentions_by_id: dict[str, dict] = {}
        for item in parsed.get("mentions", []):
            if not isinstance(item, dict):
                continue
            timestamp = item.get("timestamp")
            normalised_timestamp = str(timestamp).strip().strip("[]") if isinstance(timestamp, str) else None
            context_text = timestamp_contexts.get(normalised_timestamp or "", analysis_text)
            record = _normalise_name_correction_record(source, item, context_text)
            if record is None:
                continue
            existing = mentions_by_id.get(record["mention_id"])
            if existing is None or record["confidence"] > existing["confidence"]:
                mentions_by_id[record["mention_id"]] = record

        mentions = sorted(
            mentions_by_id.values(),
            key=lambda item: (item["timestamp"] or "", item["mention_type"], item["raw_name"]),
        )
        _write_cache(cache_dir, key, {"mentions": mentions})
        return mentions

    def build_name_correction_context(name_corrections: list[dict]) -> str:
        if not name_corrections:
            return "No name-correction decisions were produced for this transcript."

        lines = []
        for correction in name_corrections:
            selected = correction["selected_player_name"] or "UNRESOLVED"
            candidates = ", ".join(
                f"{candidate['player_id']}: {candidate['player_name']} ({candidate['confidence']:.2f})"
                for candidate in correction.get("ranked_candidates", [])
            ) or "none"
            alias_suggestions = ", ".join(
                f"{suggestion['alias']} -> {suggestion['canonical_player_name']} ({suggestion['confidence']:.2f})"
                for suggestion in correction.get("suggested_aliases", [])
            ) or "none"
            lines.append(
                f"- mention_id: {correction['mention_id']} | raw_name: {correction['raw_name']} | "
                f"status: {correction['resolution_status']} | selected: {selected} | "
                f"confidence: {correction['confidence']:.2f} | candidates: {candidates} | "
                f"alias_suggestions: {alias_suggestions} | reason: {correction['reason']}"
            )
        return "\n".join(lines)

    def resolve_supported_player_id(item: dict, player_names: dict[str, int]) -> int | None:
        quote = str(item.get("supporting_quote", ""))
        player_name = str(item.get("player_name", ""))
        player_id = resolve_player_name(player_name, player_names)
        if player_id is not None:
            canonical_name = canonical_player_id_to_name[player_id]
            if canonical_name and re.sub(r"[^a-z ]", " ", canonical_name.lower()).strip() in re.sub(r"[^a-z ]", " ", quote.lower()):
                item["player_name"] = canonical_name
                return player_id

        quote_supported_ids: set[int] = set()
        for alias, canonical_name in name_aliases.items():
            squashed_alias = _normalise_lookup_key(alias)
            normalised_quote = re.sub(r"[^a-z ]", " ", quote.lower())
            if len(squashed_alias) <= 3:
                alias_hit = re.search(rf"(?<![a-z]){re.escape(squashed_alias)}(?![a-z])", normalised_quote) is not None
            else:
                alias_hit = squashed_alias in _normalise_lookup_key(quote)
            if not alias_hit:
                continue
            alias_player_id = resolve_player_name(canonical_name, player_names)
            if alias_player_id is not None:
                quote_supported_ids.add(alias_player_id)

        if len(quote_supported_ids) == 1:
            player_id = next(iter(quote_supported_ids))
            item["player_name"] = canonical_player_id_to_name[player_id]
            return player_id

        return None

    return SimpleNamespace(
        NAME_CORRECTION_PROMPT_VERSION=NAME_CORRECTION_PROMPT_VERSION,
        NAME_CORRECTION_PROMPT_TEMPLATE=NAME_CORRECTION_PROMPT_TEMPLATE,
        NAME_CORRECTION_CANDIDATE_LIMIT=candidate_limit,
        NAME_CORRECTION_CONTEXT_RADIUS=context_radius,
        MAX_CONCURRENT_NAME_RESOLUTION=MAX_CONCURRENT_NAME_RESOLUTION,
        AUTO_RESOLVE_CONFIDENCE_THRESHOLD=auto_resolve_confidence_threshold,
        AUTO_RESOLVE_MARGIN_THRESHOLD=auto_resolve_margin_threshold,
        VALID_NAME_CORRECTION_MENTION_TYPES=VALID_NAME_CORRECTION_MENTION_TYPES,
        resolve_player_name=resolve_player_name,
        build_name_resolution_candidates=build_name_resolution_candidates,
        build_name_resolution_reason=build_name_resolution_reason,
        resolve_transcript_names=resolve_transcript_names,
        build_name_correction_context=build_name_correction_context,
        resolve_supported_player_id=resolve_supported_player_id,
    )