from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Launched via `streamlit run`, so put the repo root on the path for `fpl_hq`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_hq.paths import PROJECT_ROOT, REFERENCE_DIR, gw_dir


def normalise_alias(raw_name: str) -> str:
    return re.sub(r"[^a-z]", "", raw_name.lower())


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def review_id(cache_file: Path, mention: dict[str, Any]) -> str:
    identity = "|".join(
        (
            cache_file.name,
            str(mention.get("timestamp") or ""),
            str(mention.get("raw_name") or ""),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def load_mentions(cache_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cache_file in sorted(cache_dir.glob("*.json")):
        payload = load_json(cache_file, {})
        for mention in payload.get("mentions", []):
            if not isinstance(mention, dict):
                continue
            records.append(
                {
                    "review_id": review_id(cache_file, mention),
                    "cache_file": cache_file.name,
                    **mention,
                }
            )
    return records


def load_player_names(element_lookup_path: Path) -> list[str]:
    players = load_json(element_lookup_path, {})
    return sorted(
        {
            str(player.get("name", "")).strip()
            for player in players.values()
            if isinstance(player, dict) and str(player.get("name", "")).strip()
        }
    )


def save_review(
    review_path: Path,
    gameweek: int,
    mention: dict[str, Any],
    decision: str,
    selected_player_name: str | None,
) -> None:
    payload = load_json(review_path, {"gameweek": gameweek, "reviews": {}})
    payload["gameweek"] = gameweek
    reviews = payload.setdefault("reviews", {})
    reviews[mention["review_id"]] = {
        "cache_file": mention["cache_file"],
        "raw_name": mention.get("raw_name"),
        "timestamp": mention.get("timestamp"),
        "original_status": mention.get("resolution_status"),
        "original_player_name": mention.get("selected_player_name"),
        "decision": decision,
        "selected_player_name": selected_player_name,
        "reviewed_at": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(review_path, payload)


def restore_review(
    review_path: Path,
    gameweek: int,
    review_id: str,
    restored_review: dict[str, Any] | None,
) -> None:
    payload = load_json(review_path, {"gameweek": gameweek, "reviews": {}})
    payload["gameweek"] = gameweek
    reviews = payload.setdefault("reviews", {})
    if restored_review is None:
        reviews.pop(review_id, None)
    else:
        reviews[review_id] = restored_review
    write_json_atomic(review_path, payload)


def save_alias(alias_path: Path, raw_name: str, player_name: str) -> tuple[str, str | None]:
    aliases = load_json(alias_path, {})
    alias = normalise_alias(raw_name)
    if not alias:
        raise ValueError("The raw name does not produce a usable alias key.")
    previous_player = aliases.get(alias)
    aliases[alias] = player_name
    write_json_atomic(alias_path, aliases)
    return alias, previous_player


def remove_alias(alias_path: Path, alias: str) -> str | None:
    aliases = load_json(alias_path, {})
    previous_player = aliases.pop(alias, None)
    write_json_atomic(alias_path, aliases)
    return previous_player


def restore_alias(alias_path: Path, alias: str, previous_player: str | None) -> None:
    aliases = load_json(alias_path, {})
    if previous_player is None:
        aliases.pop(alias, None)
    else:
        aliases[alias] = previous_player
    write_json_atomic(alias_path, aliases)


def update_alias(alias_path: Path, alias: str, player_name: str) -> None:
    aliases = load_json(alias_path, {})
    aliases[alias] = player_name
    write_json_atomic(alias_path, aliases)


def save_alias_review(
    review_path: Path,
    gameweek: int,
    alias: str,
    original_player_name: str,
    decision: str,
    selected_player_name: str | None,
) -> None:
    payload = load_json(review_path, {"gameweek": gameweek, "reviews": {}, "alias_reviews": {}})
    payload["gameweek"] = gameweek
    alias_reviews = payload.setdefault("alias_reviews", {})
    alias_reviews[alias] = {
        "original_player_name": original_player_name,
        "decision": decision,
        "selected_player_name": selected_player_name,
        "reviewed_at": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(review_path, payload)


def restore_alias_review(
    review_path: Path,
    gameweek: int,
    alias: str,
    restored_review: dict[str, Any] | None,
) -> None:
    payload = load_json(review_path, {"gameweek": gameweek, "reviews": {}, "alias_reviews": {}})
    payload["gameweek"] = gameweek
    alias_reviews = payload.setdefault("alias_reviews", {})
    if restored_review is None:
        alias_reviews.pop(alias, None)
    else:
        alias_reviews[alias] = restored_review
    write_json_atomic(review_path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gameweek", type=int, default=4)
    args, _ = parser.parse_known_args()
    return args


def run_dashboard(default_gameweek: int) -> None:
    import streamlit as st

    st.set_page_config(page_title="Name review", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: #f4f1e8; color: #17231d; }
        [data-testid="stSidebar"] { background: #17231d; }
        [data-testid="stSidebar"] * { color: #f7f2e7; }
        div[data-testid="stMetric"] { border-top: 3px solid #d95f3b; padding-top: 0.75rem; }
        .mention-name { font: 700 2.4rem/1.05 Georgia, serif; color: #17231d; }
        .mention-meta { color: #647067; margin: 0.35rem 0 1.4rem; }
        .reason { border-left: 4px solid #d95f3b; padding: 0.25rem 0 0.25rem 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("Name review")
        gameweek = int(st.number_input("Gameweek", min_value=1, value=default_gameweek, step=1))
        status_filter = st.segmented_control(
            "Resolver status",
            ["All", "Unresolved", "Resolved"],
            default="Unresolved",
        )
        review_filter = st.segmented_control(
            "Review status",
            ["Unreviewed", "Reviewed", "All"],
            default="Unreviewed",
        )

    cache_dir = gw_dir(gameweek) / "transcript_cache" / "name_resolution_cache"
    review_path = gw_dir(gameweek) / "data" / f"gw_{gameweek}_name_resolution_reviews.json"
    alias_path = REFERENCE_DIR / "name_aliases.json"

    mentions = load_mentions(cache_dir)
    reviews_payload = load_json(review_path, {"gameweek": gameweek, "reviews": {}})
    reviews = reviews_payload.get("reviews", {})
    alias_reviews = reviews_payload.get("alias_reviews", {})
    player_names = load_player_names(REFERENCE_DIR / "element_lookup.json")

    if not cache_dir.exists():
        st.error(f"No name-resolution cache found for gameweek {gameweek}.")
        st.stop()
    if not mentions:
        st.warning(f"No mentions found in {cache_dir.relative_to(PROJECT_ROOT)}.")
        st.stop()

    reviewed_count = sum(mention["review_id"] in reviews for mention in mentions)
    unresolved_count = sum(mention.get("resolution_status") == "unresolved" for mention in mentions)
    metric_columns = st.columns(3)
    metric_columns[0].metric("Mentions", len(mentions))
    metric_columns[1].metric("Reviewed", f"{reviewed_count} / {len(mentions)}")
    metric_columns[2].metric("Unresolved", unresolved_count)
    st.progress(reviewed_count / len(mentions))

    last_action_key = f"name_review_last_action_{gameweek}"

    def undo_last_action() -> None:
        last_action = st.session_state.get(last_action_key)
        if not last_action:
            return

        action_kind = last_action.get("kind")
        if action_kind == "mention":
            restore_review(
                review_path,
                gameweek,
                last_action["review_id"],
                last_action.get("previous_review"),
            )
            if last_action.get("alias_changed"):
                restore_alias(
                    alias_path,
                    last_action["alias"],
                    last_action.get("previous_alias_player"),
                )
        elif action_kind == "alias":
            restore_alias_review(
                review_path,
                gameweek,
                last_action["alias"],
                last_action.get("previous_alias_review"),
            )
            if last_action.get("alias_changed"):
                restore_alias(
                    alias_path,
                    last_action["alias"],
                    last_action.get("previous_alias_player"),
                )

        st.session_state.pop(last_action_key, None)

    with st.sidebar:
        last_action = st.session_state.get(last_action_key)
        if last_action:
            st.caption(f"Last action: {last_action.get('summary', 'Saved review')}")
        if st.button("Undo last action", width="stretch", disabled=not last_action):
            undo_last_action()
            st.rerun()

    unresolved_mentions = [
        mention for mention in mentions if mention.get("resolution_status") == "unresolved"
    ]
    unresolved_reviewed = sum(mention["review_id"] in reviews for mention in unresolved_mentions)
    if unresolved_reviewed == len(unresolved_mentions):
        aliases = load_json(alias_path, {})
        pending_aliases = [
            alias
            for alias, player_name in aliases.items()
            if alias_reviews.get(alias, {}).get("selected_player_name") != player_name
        ]

        st.header("Review existing aliases")
        st.caption(
            f"All {len(unresolved_mentions)} unresolved names have been reviewed. "
            "Now check each saved alias mapping."
        )

        if not aliases:
            st.success("There are no existing aliases to review.")
            st.stop()
        if not pending_aliases:
            st.success(f"All {len(aliases)} existing aliases have been reviewed.")
            st.stop()

        alias_order_key = f"alias_review_order_{gameweek}"
        alias_order = [
            alias for alias in st.session_state.get(alias_order_key, []) if alias in pending_aliases
        ]
        new_aliases = [alias for alias in pending_aliases if alias not in alias_order]
        random.shuffle(new_aliases)
        alias_order.extend(new_aliases)
        st.session_state[alias_order_key] = alias_order

        alias = alias_order[0]
        mapped_player = aliases[alias]
        reviewed_alias_count = len(aliases) - len(pending_aliases)
        st.progress(reviewed_alias_count / len(aliases))
        st.markdown(f"<div class='mention-name'>{alias}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='mention-meta'>Alias {reviewed_alias_count + 1} of {len(aliases)}</div>",
            unsafe_allow_html=True,
        )

        st.subheader("Current mapping")
        st.markdown(f"<div class='reason'>{mapped_player}</div>", unsafe_allow_html=True)
        alias_options = ["Select a player…", *player_names]
        alias_default_index = alias_options.index(mapped_player) if mapped_player in alias_options else 0
        alias_player = st.selectbox(
            "Canonical player",
            alias_options,
            index=alias_default_index,
            key=f"alias_player_{gameweek}_{alias}",
        )
        alias_player = None if alias_player == alias_options[0] else alias_player

        confirm_column, correction_column, remove_column = st.columns(3)
        confirm_alias = confirm_column.button(
            "Confirm mapping",
            type="primary",
            width="stretch",
            disabled=alias_player != mapped_player,
        )
        correct_alias = correction_column.button(
            "Save correction",
            width="stretch",
            disabled=alias_player is None or alias_player == mapped_player,
        )
        remove_alias_clicked = remove_column.button("Remove alias", width="stretch")

        if confirm_alias:
            previous_alias_review = dict(alias_reviews.get(alias)) if alias in alias_reviews else None
            save_alias_review(review_path, gameweek, alias, mapped_player, "confirmed", mapped_player)
            st.session_state[last_action_key] = {
                "kind": "alias",
                "alias": alias,
                "previous_alias_review": previous_alias_review,
                "alias_changed": False,
                "summary": f"Confirmed alias {alias}",
            }
            st.rerun()
        if correct_alias and alias_player is not None:
            previous_alias_review = dict(alias_reviews.get(alias)) if alias in alias_reviews else None
            previous_alias_player = aliases.get(alias)
            update_alias(alias_path, alias, alias_player)
            save_alias_review(review_path, gameweek, alias, mapped_player, "corrected", alias_player)
            st.session_state[last_action_key] = {
                "kind": "alias",
                "alias": alias,
                "previous_alias_review": previous_alias_review,
                "alias_changed": True,
                "previous_alias_player": previous_alias_player,
                "summary": f"Corrected alias {alias}",
            }
            st.toast(f"Updated {alias}: {mapped_player} -> {alias_player}")
            st.rerun()
        if remove_alias_clicked:
            previous_alias_review = dict(alias_reviews.get(alias)) if alias in alias_reviews else None
            previous_alias_player = aliases.get(alias)
            remove_alias(alias_path, alias)
            save_alias_review(review_path, gameweek, alias, mapped_player, "removed", None)
            st.session_state[last_action_key] = {
                "kind": "alias",
                "alias": alias,
                "previous_alias_review": previous_alias_review,
                "alias_changed": True,
                "previous_alias_player": previous_alias_player,
                "summary": f"Removed alias {alias}",
            }
            st.toast(f"Removed {alias} -> {mapped_player}")
            st.rerun()
        st.stop()

    filtered_mentions = mentions
    if status_filter != "All":
        expected_status = str(status_filter).lower()
        filtered_mentions = [
            mention for mention in filtered_mentions if mention.get("resolution_status") == expected_status
        ]
    if review_filter != "All":
        want_reviewed = review_filter == "Reviewed"
        filtered_mentions = [
            mention
            for mention in filtered_mentions
            if (mention["review_id"] in reviews) == want_reviewed
        ]

    if not filtered_mentions:
        st.success("Nothing left in this view.")
        st.stop()

    navigation_key = f"review_index_{gameweek}_{status_filter}_{review_filter}"
    current_index = min(st.session_state.get(navigation_key, 0), len(filtered_mentions) - 1)
    st.session_state[navigation_key] = current_index
    mention = filtered_mentions[current_index]
    existing_review = reviews.get(mention["review_id"])

    previous_column, position_column, next_column = st.columns([1, 3, 1])
    if previous_column.button("Previous", width="stretch", disabled=current_index == 0):
        st.session_state[navigation_key] = current_index - 1
        st.rerun()
    position_column.markdown(
        f"<div style='text-align:center;padding:.55rem'>{current_index + 1} of {len(filtered_mentions)}</div>",
        unsafe_allow_html=True,
    )
    if next_column.button("Next", width="stretch", disabled=current_index == len(filtered_mentions) - 1):
        st.session_state[navigation_key] = current_index + 1
        st.rerun()

    raw_name = str(mention.get("raw_name", ""))
    st.markdown(f"<div class='mention-name'>{raw_name}</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='mention-meta'>"
        f"{mention.get('timestamp') or 'No timestamp'} | {mention['cache_file']} | "
        f"resolver: {mention.get('resolution_status', 'unknown')} (confidence {int(mention.get('confidence', 1))}/5)"
        "</div>",
        unsafe_allow_html=True,
    )

    aliases = load_json(alias_path, {})
    alias_key = normalise_alias(raw_name)
    existing_alias = aliases.get(alias_key)
    proposed_player = mention.get("selected_player_name") or existing_alias
    candidates = mention.get("ranked_candidates", [])

    detail_column, decision_column = st.columns([1, 1], gap="large")
    with detail_column:
        st.subheader("Resolver evidence")
        st.markdown(f"<div class='reason'>{mention.get('reason', 'No reason provided.')}</div>", unsafe_allow_html=True)
        if candidates:
            st.dataframe(
                candidates,
                column_config={
                    "player_name": "Candidate",
                    "confidence": st.column_config.NumberColumn("Confidence", min_value=1, max_value=5, format="%d"),
                },
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No alternative candidates were recorded.")

    with decision_column:
        st.subheader("Your decision")
        options = ["Select a player…", *player_names]
        default_player = existing_review.get("selected_player_name") if existing_review else proposed_player
        default_index = options.index(default_player) if default_player in options else 0
        selected_player = st.selectbox(
            "Canonical player",
            options,
            index=default_index,
            key=f"player_{mention['review_id']}",
        )
        selected_player = None if selected_player == options[0] else selected_player

        st.caption(f"Alias key: `{alias_key or 'none'}`")
        if existing_alias and existing_alias != selected_player:
            st.warning(f"This alias currently maps to {existing_alias}. Saving a player mapping will replace it.")
        if existing_review:
            st.info(f"Saved decision: {existing_review['decision'].replace('_', ' ')}")

        confirm_column, correction_column = st.columns(2)
        confirm_clicked = confirm_column.button(
            "Confirm mapping",
            type="primary",
            width="stretch",
            disabled=selected_player is None or selected_player != proposed_player,
        )
        correction_clicked = correction_column.button(
            "Save correction",
            width="stretch",
            disabled=selected_player is None,
        )
        not_player_column, unsure_column = st.columns(2)
        not_player_clicked = not_player_column.button("Not a player", width="stretch")
        unsure_clicked = unsure_column.button("Unsure", width="stretch")

    decision = None
    if confirm_clicked:
        decision = "confirmed"
    elif correction_clicked:
        decision = "corrected"
    elif not_player_clicked:
        decision = "not_a_player"
        selected_player = None
    elif unsure_clicked:
        decision = "unsure"
        selected_player = None

    if decision is not None:
        previous_review = dict(existing_review) if existing_review else None
        alias_changed = False
        previous_alias_player = None
        if decision in {"confirmed", "corrected"} and selected_player is not None:
            alias, previous_alias_player = save_alias(alias_path, raw_name, selected_player)
            alias_changed = True
            if previous_alias_player and previous_alias_player != selected_player:
                st.toast(f"Updated {alias}: {previous_alias_player} -> {selected_player}")
            else:
                st.toast(f"Saved {alias} -> {selected_player}")
        save_review(review_path, gameweek, mention, decision, selected_player)
        st.session_state[last_action_key] = {
            "kind": "mention",
            "review_id": mention["review_id"],
            "alias": normalise_alias(raw_name),
            "previous_review": previous_review,
            "alias_changed": alias_changed,
            "previous_alias_player": previous_alias_player,
            "summary": f"{decision.replace('_', ' ').title()} {raw_name}",
        }
        if review_filter == "Unreviewed":
            st.session_state[navigation_key] = min(current_index, len(filtered_mentions) - 2)
        else:
            st.session_state[navigation_key] = min(current_index + 1, len(filtered_mentions) - 1)
        st.rerun()


if __name__ == "__main__":
    run_dashboard(parse_args().gameweek)