from __future__ import annotations

# Name resolution work.

import hashlib
import json
import re
from difflib import get_close_matches
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fpl_hq.paths import REFERENCE_DIR
from fpl_hq.consensus.llm import DETERMINISTIC

NAME_CORRECTION_PROMPT_VERSION = "v18"
NAME_CORRECTION_RESOLUTION_VERSION = "v6"
NAME_CORRECTION_CANDIDATE_LIMIT = 5
MAX_CONCURRENT_NAME_RESOLUTION = 2
# The response lists every mention, so long videos are resolved in chunks to keep each response small.
NAME_RESOLUTION_CHUNK_CHARS = 20_000
# Confidence is an integer 1-5 scale (see CONFIDENCE_SCALE_LABELS); only 4 or 5 auto-resolve.
AUTO_RESOLVE_CONFIDENCE_THRESHOLD = 4
AUTO_RESOLVE_MARGIN_THRESHOLD = 1

CONFIDENCE_SCALE_LABELS = {
    1: "Definitely not the same player",
    2: "Unlikely to be correct",
    3: "Possibly correct, meaningfully uncertain",
    4: "Most likely correct",
    5: "Certain, no meaningful doubt",
}

NAME_CORRECTION_PROMPT_TEMPLATE = """You analyse a retained FPL-analysis transcript before a later extraction step. Your only job is to resolve transcript-spelled player names to canonical FPL player names.

This text comes from a raw auto-generated transcript. Player names are ASR-style phonetic renderings of spoken names, not deliberate typed spellings. Treat each raw player mention as a sound-alike transcript surface form. Do not assume normal typo-correction logic unless the phonetic evidence or alias map makes the identity genuinely clear.

You are given:
- a transcript source id
- canonical FPL player names from the element lookup
- a map of common ASR or nickname aliases to canonical names

Identify every player-specific mention that is relevant to transfer, selection, hold/sell, or captaincy discussion. Ignore team-only discussion and generic role terms that do not point to a player.

Return strict JSON with exactly one key: `mentions`.

Each item in `mentions` must have:
- `raw_name`: the player name as it appears in the transcript
- `timestamp`: the [MM:SS] marker of the paragraph containing the mention, or null
- `resolution_status`: `resolved` or `unresolved`
- `selected_player_name`: the chosen canonical FPL player name, or null if unresolved
- `confidence`: an integer from 1 to 5 measuring how sure you are the selected (or best candidate) player is correct:
  - 1 = Definitely not the same player; no meaningful phonetic or alias support.
  - 2 = Unlikely to be correct; only a weak or partial resemblance.
  - 3 = Possibly correct, but meaningfully uncertain; plausible without being established.
  - 4 = Most likely correct; strong phonetic or alias support with minimal doubt.
  - 5 = Certain; an exact alias match or unmistakable phonetic match.
- `ranked_candidates`: an array of up to 5 candidate objects with only `player_name` and `confidence` (same 1-5 scale)
- `reason`: one short sentence explaining the decision using phonetic similarity or an alias-map match

Resolution rules:
- Use the alias map when it gives a clear correction.
- A resolved name must be phonetically similar to the raw name unless the alias map gives an exact mapping.
- Only mark `resolution_status` as `resolved` when confidence is 4 or 5; use `unresolved` for confidence 1-3.
- If more than one candidate would independently earn confidence 4 or 5, mark the mention unresolved.
- Keep `raw_name` faithful to the transcript and do not rewrite quotes.
- Candidate order does not matter.

SOURCE:
{source}

ALIASES:
{aliases}

CANONICAL_PLAYERS:
{players}

TRANSCRIPT:
{text}"""


def load_name_aliases() -> dict[str, str]:
    alias_path = REFERENCE_DIR / "name_aliases.json"
    with alias_path.open(encoding="utf-8") as file:
        return json.load(file)


def _normalise_lookup_key(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def clamp(value: float, lower: float = 0, upper: float = 100) -> float:
    return max(lower, min(value, upper))


def _bucket_confidence(score: float) -> int:
    """Map a 0-1 heuristic match score onto the labeled 1-5 confidence scale."""
    if score >= 0.95:
        return 5
    if score >= 0.85:
        return 4
    if score >= 0.7:
        return 3
    if score >= 0.5:
        return 2
    return 1


def _coerce_confidence_scale(value: Any) -> int:
    """Clamp a model-provided confidence value onto the 1-5 scale."""
    try:
        number = round(float(value))
    except (TypeError, ValueError):
        number = 1
    return int(clamp(number, 1, 5))


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
    auto_resolve_confidence_threshold: float = AUTO_RESOLVE_CONFIDENCE_THRESHOLD,
    auto_resolve_margin_threshold: float = AUTO_RESOLVE_MARGIN_THRESHOLD,
) -> SimpleNamespace:
    def _candidate_record(player_id: int, player_name: str, confidence: int) -> dict:
        return {
            "player_id": player_id,
            "player_name": player_name,
            "confidence": int(clamp(round(confidence), 1, 5)),
        }

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

    def build_name_resolution_candidates(raw_name: str, player_names: dict[str, int]) -> list[dict]:
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

        alias_name = name_aliases.get(squashed_input)
        if alias_name is not None:
            alias_player_id = player_names.get(alias_name)
            if alias_player_id is not None:
                return [_candidate_record(alias_player_id, alias_name, 5)]

        exact_name_matches = normalised_full_names.get(normalised_input, [])
        if len(exact_name_matches) == 1:
            exact_player_id = exact_name_matches[0]
            exact_player_name = canonical_player_id_to_name.get(exact_player_id)
            if exact_player_name is not None:
                return [_candidate_record(exact_player_id, exact_player_name, 5)]

        candidates_by_id: dict[int, dict] = {}

        def add_candidate(player_id: int, confidence: float) -> None:
            player_name = canonical_player_id_to_name.get(player_id)
            if player_name is None:
                return
            record = _candidate_record(player_id, player_name, _bucket_confidence(confidence))
            existing = candidates_by_id.get(player_id)
            if existing is None or record["confidence"] > existing["confidence"]:
                candidates_by_id[player_id] = record

        def _alias_cutoff_for_input(input_text: str) -> float:
            if len(input_text) <= 4:
                return 0.72
            if len(input_text) <= 6:
                return 0.78
            return 0.82

        alias_cutoff = _alias_cutoff_for_input(squashed_input)
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
    ) -> str:
        squashed_input = _normalise_lookup_key(raw_name)
        alias_name = name_aliases.get(squashed_input)

        if selected_player_name is None:
            if not candidates:
                return f"No supported canonical candidate found for '{raw_name}'."
            if len(candidates) == 1:
                candidate = candidates[0]
                return (
                    f"Only one supported phonetic candidate for '{raw_name}' was found "
                    f"({candidate['player_name']} at confidence {candidate['confidence']}/5); it stays unresolved."
                )
            return f"No alias match for '{raw_name}' and multiple plausible canonical candidates remain."

        if alias_name == selected_player_name:
            return f"Alias '{raw_name}' maps to {selected_player_name}."

        if selected_player_id is not None:
            metadata = player_metadata.get(selected_player_id, {})
            team_name = str(metadata.get("team", "")).strip()
            position = int(metadata.get("position", 0) or 0)
            if len(candidates) == 1:
                return (
                    f"Unique supported canonical match for '{raw_name}' resolves to {selected_player_name}"
                    f" ({team_name}, position {position})."
                )

        return f"Best supported canonical candidate for '{raw_name}' is {selected_player_name}."

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

    def _normalise_name_correction_record(source: str, item: dict) -> dict | None:
        raw_name = str(item.get("raw_name", "")).strip()
        if not raw_name:
            return None

        timestamp = item.get("timestamp")
        timestamp = str(timestamp).strip() if isinstance(timestamp, str) else None
        if timestamp == "":
            timestamp = None
        elif timestamp is not None:
            timestamp = timestamp.strip("[]")

        model_resolution_status = str(item.get("resolution_status", "unresolved")).strip().lower()
        resolution_status = "resolved" if model_resolution_status == "resolved" else "unresolved"

        deterministic_candidates = build_name_resolution_candidates(raw_name, canonical_player_names)
        deterministic_candidate_ids = {candidate["player_id"] for candidate in deterministic_candidates}

        model_selected_player_name = str(item.get("selected_player_name", "")).strip()
        model_selected_player_id = canonical_player_names.get(model_selected_player_name)
        selected_player_id = model_selected_player_id if resolution_status == "resolved" else None
        if selected_player_id is None:
            selected_player_id = None
            resolution_status = "unresolved"

        if selected_player_id not in deterministic_candidate_ids:
            selected_player_id = None
            resolution_status = "unresolved"

        ranked_candidates_by_id = {candidate["player_id"]: candidate for candidate in deterministic_candidates}
        for candidate in item.get("ranked_candidates", [])[:candidate_limit]:
            if not isinstance(candidate, dict):
                continue
            candidate_name = str(candidate.get("player_name", "")).strip()
            candidate_id = canonical_player_names.get(candidate_name)
            if candidate_id is None or candidate_id in ranked_candidates_by_id:
                continue
            ranked_candidates_by_id[candidate_id] = _candidate_record(
                candidate_id,
                candidate_name,
                _coerce_confidence_scale(candidate.get("confidence", 1)),
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

        if selected_player_id is None and ranked_candidates:
            top_candidate = ranked_candidates[0]
            if should_resolve_candidate(ranked_candidates, top_candidate["player_id"]):
                selected_player_id = top_candidate["player_id"]
                resolution_status = "resolved"

        selected_player_name = None
        if selected_player_id is not None:
            selected_player_name = canonical_player_id_to_name[selected_player_id]
            selected_candidate_confidence = ranked_candidates_by_id.get(
                selected_player_id, {"confidence": AUTO_RESOLVE_CONFIDENCE_THRESHOLD}
            )["confidence"]
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
            confidence = ranked_candidates[0]["confidence"] if ranked_candidates else 1
            resolution_status = "unresolved"

        model_reason = str(item.get("reason", "")).strip()
        decision_changed = selected_player_id != model_selected_player_id or resolution_status != model_resolution_status
        reason = model_reason if model_reason and not decision_changed else build_name_resolution_reason(
            raw_name,
            selected_player_name,
            selected_player_id,
            ranked_candidates,
        )
        public_candidates = [
            {
                "player_name": candidate["player_name"],
                "confidence": candidate["confidence"],
            }
            for candidate in ranked_candidates
            if candidate["player_id"] != selected_player_id
        ][:candidate_limit]

        return {
            "raw_name": raw_name,
            "timestamp": timestamp,
            "resolution_status": resolution_status,
            "selected_player_name": selected_player_name,
            "confidence": confidence,
            "ranked_candidates": public_candidates,
            "reason": reason,
        }

    def _split_for_resolution(analysis_text: str) -> list[str]:
        chunks: list[str] = []
        for section in re.split(r"\n\n(?=\[SECTION: )", analysis_text):
            if chunks and len(chunks[-1]) + len(section) + 2 <= NAME_RESOLUTION_CHUNK_CHARS:
                chunks[-1] = f"{chunks[-1]}\n\n{section}"
            else:
                chunks.append(section)
        return chunks

    def resolve_transcript_names(client: Any, model: str, cache_dir: Path, source: str, analysis_text: str) -> list[dict]:
        mentions_by_key: dict[tuple[str | None, str], dict] = {}
        for chunk in _split_for_resolution(analysis_text):
            for record in _resolve_chunk(client, model, cache_dir, source, chunk):
                mention_key = (record["timestamp"], _normalise_lookup_key(record["raw_name"]))
                existing = mentions_by_key.get(mention_key)
                if existing is None or record["confidence"] > existing["confidence"]:
                    mentions_by_key[mention_key] = record
        return sorted(mentions_by_key.values(), key=lambda item: (item["timestamp"] or "", item["raw_name"]))

    def _resolve_chunk(client: Any, model: str, cache_dir: Path, source: str, analysis_text: str) -> list[dict]:
        aliases_fingerprint = json.dumps(name_aliases, ensure_ascii=False, sort_keys=True)
        key = _cache_key(
            "name_correction",
            NAME_CORRECTION_PROMPT_VERSION,
            NAME_CORRECTION_RESOLUTION_VERSION,
            model,
            source,
            analysis_text,
            aliases_fingerprint,
        )
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
                        players="\n".join(sorted(canonical_player_names)),
                        text=analysis_text,
                    ),
                },
            ],
            **DETERMINISTIC,
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
        mentions_by_key: dict[tuple[str | None, str], dict] = {}
        for item in parsed.get("mentions", []):
            if not isinstance(item, dict):
                continue
            record = _normalise_name_correction_record(source, item)
            if record is None:
                continue
            mention_key = (record["timestamp"], _normalise_lookup_key(record["raw_name"]))
            existing = mentions_by_key.get(mention_key)
            if existing is None or record["confidence"] > existing["confidence"]:
                mentions_by_key[mention_key] = record

        mentions = sorted(
            mentions_by_key.values(),
            key=lambda item: (item["timestamp"] or "", item["raw_name"]),
        )
        _write_cache(cache_dir, key, {"mentions": mentions})
        return mentions

    def resolve_supported_player_id(item: dict, player_names: dict[str, int]) -> int | None:
        quote = str(item.get("supporting_quote", ""))
        player_name = str(item.get("player_name", ""))
        player_id = resolve_player_name(player_name, player_names)
        if player_id is not None:
            item["player_name"] = canonical_player_id_to_name[player_id]
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
        NAME_CORRECTION_RESOLUTION_VERSION=NAME_CORRECTION_RESOLUTION_VERSION,
        NAME_CORRECTION_PROMPT_TEMPLATE=NAME_CORRECTION_PROMPT_TEMPLATE,
        CONFIDENCE_SCALE_LABELS=CONFIDENCE_SCALE_LABELS,
        NAME_CORRECTION_CANDIDATE_LIMIT=candidate_limit,
        MAX_CONCURRENT_NAME_RESOLUTION=MAX_CONCURRENT_NAME_RESOLUTION,
        AUTO_RESOLVE_CONFIDENCE_THRESHOLD=auto_resolve_confidence_threshold,
        AUTO_RESOLVE_MARGIN_THRESHOLD=auto_resolve_margin_threshold,
        resolve_player_name=resolve_player_name,
        build_name_resolution_candidates=build_name_resolution_candidates,
        build_name_resolution_reason=build_name_resolution_reason,
        resolve_transcript_names=resolve_transcript_names,
        resolve_supported_player_id=resolve_supported_player_id,
    )