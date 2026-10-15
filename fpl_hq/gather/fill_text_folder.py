"""Discover and save multiple FPL text sources into a gameweek text folder.

This is the one-command batch path for:
- finding candidate FPL articles from the curated source catalog
- filtering them by query and optional gameweek hint
- downloading the article text
- writing it into data/2026-27/gameweeks/gwN/text/
- updating the per-gameweek source manifest

Example:
    python -m fpl_hq.gather.fill_text_folder --query fpl --gameweek 3 --limit 5
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import sys
from pathlib import Path
from typing import Any

import requests

from fpl_hq.paths import gw_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Case-insensitive text to match in RSS title/summary")
    parser.add_argument("--gameweek", type=int, required=True, help="Gameweek folder to write into")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of articles to save")
    parser.add_argument(
        "--feed",
        action="append",
        default=[],
        help="Optional RSS feed URL to search; repeat to bypass curated hybrid discovery with explicit feeds",
    )
    parser.add_argument(
        "--strict-gameweek",
        action="store_true",
        help="Only keep entries that explicitly mention the requested gameweek",
    )
    parser.add_argument(
        "--name-prefix",
        default="batch",
        help="Prefix for the generated transcript filenames",
    )
    return parser


def _parse_kickoff_time(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def get_gameweek_publish_window(gameweek: int) -> tuple[datetime | None, datetime | None]:
    if gameweek < 1:
        raise ValueError("gameweek must be 1 or greater")

    def fetch_event(event_number: int) -> list[dict[str, Any]]:
        response = requests.get(
            f"https://fantasy.premierleague.com/api/fixtures/?event={event_number}",
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected fixtures payload for event={event_number}")
        return payload

    current_fixtures = fetch_event(gameweek)
    current_kickoffs = [
        _parse_kickoff_time(str(fixture["kickoff_time"]))
        for fixture in current_fixtures
        if fixture.get("kickoff_time")
    ]
    if not current_kickoffs:
        raise ValueError(f"No kickoff times returned for event={gameweek}")

    window_end = min(current_kickoffs)
    window_start = None
    if gameweek > 1:
        previous_fixtures = fetch_event(gameweek - 1)
        previous_kickoffs = [
            _parse_kickoff_time(str(fixture["kickoff_time"]))
            for fixture in previous_fixtures
            if fixture.get("kickoff_time")
        ]
        if not previous_kickoffs:
            raise ValueError(f"No kickoff times returned for event={gameweek - 1}")
        window_start = max(previous_kickoffs) - timedelta(days=1)

    return window_start, window_end


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from fpl_hq.gather.discover_text import get_default_hybrid_sources, search_hybrid, search_rss
    from fpl_hq.gather.fetch_text import extract_article_text, write_or_print_result

    published_after, published_before = get_gameweek_publish_window(args.gameweek)
    if args.feed:
        feeds = [
            {"id": str(index), "name": feed_url, "rss_url": feed_url, "base_url": ""}
            for index, feed_url in enumerate(args.feed, start=1)
        ]
        discovered = search_rss(
            feeds,
            args.query,
            args.limit,
            args.gameweek,
            args.strict_gameweek,
            published_after=published_after,
            published_before=published_before,
        )
    else:
        discovered = search_hybrid(
            get_default_hybrid_sources(),
            args.query,
            args.limit,
            args.gameweek,
            args.strict_gameweek,
            published_after=published_after,
            published_before=published_before,
        )
    if not discovered:
        print("No matching entries found in the fixture-date window.")
        return

    text_dir = gw_dir(args.gameweek) / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    seen_links: set[str] = set()
    for index, item in enumerate(discovered, start=1):
        link = item.get("link", "")
        if not link or link in seen_links:
            continue
        seen_links.add(link)

        try:
            article = extract_article_text(link)
            filename = f"{args.name_prefix}_{index:02d}"
            write_or_print_result(article, args.gameweek, text_dir / f"{filename}.txt", None)
            saved += 1
        except Exception as exc:  # noqa: BLE001
            print(f"Skipped {link}: {exc}", file=sys.stderr)

    print(f"Saved {saved} article(s) into data/2026-27/gameweeks/gw{args.gameweek}/text/")


if __name__ == "__main__":
    main()