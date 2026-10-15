"""Discover FPL-related text sources via RSS feeds, sitemaps, or Reddit JSON search.

Usage examples:
    python -m fpl_hq.gather.discover_text rss --query saka --limit 10
    python -m fpl_hq.gather.discover_text hybrid --query saka --gameweek 3 --limit 10
    python -m fpl_hq.gather.discover_text reddit --query wildcard --subreddit FantasyPL --limit 10
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse
from xml.etree import ElementTree

import feedparser
import requests
from dotenv import load_dotenv

USER_AGENT = "fpl-agent/1.0 (+https://github.com/)"
DEFAULT_TIMEOUT_SECONDS = 20
from fpl_hq.paths import REFERENCE_DIR
SOURCE_CATALOG_PATH = REFERENCE_DIR / "text_sources.json"
BLOCKED_URL_SUBSTRINGS = ("ucl",)
BLOCKED_PATH_SUBSTRINGS = ("/category/", "/tag/", "/author/")
DEFAULT_METADATA_FETCH_LIMIT = 30

load_dotenv()


class PageMetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.title_parts: list[str] = []
        self.description = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_attrs = {key.lower(): (value or "") for key, value in attrs}
        if tag.lower() == "title":
            self.in_title = True
            return
        if tag.lower() != "meta" or self.description:
            return

        name = normalized_attrs.get("name", "").lower()
        prop = normalized_attrs.get("property", "").lower()
        if name == "description" or prop == "og:description":
            self.description = normalized_attrs.get("content", "").strip()

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)

    @property
    def title(self) -> str:
        return " ".join(part.strip() for part in self.title_parts if part.strip()).strip()


class PageLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []
        self._current_href: str = ""
        self._current_text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attributes = {key.lower(): (value or "") for key, value in attrs}
        self._current_href = attributes.get("href", "").strip()
        self._current_text_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_href:
            self._current_text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._current_href:
            return
        text = " ".join(part.strip() for part in self._current_text_parts if part.strip()).strip()
        self.links.append({"href": self._current_href, "text": text})
        self._current_href = ""
        self._current_text_parts = []


def get_reddit_session() -> requests.Session:
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("Reddit API credentials are missing. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET.")

    token_response = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=(client_id, client_secret),
        data={"grant_type": "client_credentials"},
        headers={"User-Agent": USER_AGENT},
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    token_response.raise_for_status()
    access_token = token_response.json().get("access_token")
    if not access_token:
        raise ValueError("Reddit OAuth response did not contain an access token")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Authorization": f"Bearer {access_token}",
        }
    )
    return session


def load_source_catalog(path: Path = SOURCE_CATALOG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def get_default_hybrid_sources(path: Path = SOURCE_CATALOG_PATH) -> list[dict[str, Any]]:
    catalog = load_source_catalog(path)
    return [dict(entry) for entry in catalog.get("websites", [])]


def get_default_rss_sources(path: Path = SOURCE_CATALOG_PATH) -> list[dict[str, str]]:
    results = []
    for entry in get_default_hybrid_sources(path):
        rss_url = entry.get("rss_url")
        if not rss_url:
            continue
        results.append(
            {
                "id": str(entry.get("id", "")),
                "name": str(entry.get("name", rss_url)),
                "rss_url": str(rss_url),
                "base_url": str(entry.get("base_url", "")),
            }
        )
    return results


def get_default_page_sources(path: Path = SOURCE_CATALOG_PATH) -> list[dict[str, Any]]:
    return [entry for entry in get_default_hybrid_sources(path) if entry.get("page_urls")]


def url_contains_blocked_substring(url: str) -> bool:
    lowered = url.lower()
    return any(blocked in lowered for blocked in BLOCKED_URL_SUBSTRINGS)


def parse_datetime_string(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(normalized)
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def entry_datetime(entry: dict[str, Any]) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed is None:
        return None
    try:
        return datetime(*parsed[:6], tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def detect_gameweek_match(text: str, gameweek: int) -> bool:
    patterns = [
        rf"\bgw\s*{gameweek}\b",
        rf"\bgameweek\s*{gameweek}\b",
        rf"\bgame\s*week\s*{gameweek}\b",
    ]
    lowered_text = text.lower()
    return any(re.search(pattern, lowered_text) for pattern in patterns)


def within_publish_window(
    published_at: datetime | None,
    published_after: datetime | None,
    published_before: datetime | None,
) -> bool:
    if published_after is None and published_before is None:
        return True
    if published_at is None:
        return False
    if published_after is not None and published_at < published_after:
        return False
    if published_before is not None and published_at >= published_before:
        return False
    return True


def fetch_page_metadata(url: str) -> tuple[str, str]:
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    parser = PageMetadataParser()
    parser.feed(response.text)
    return parser.title, parser.description


def fetch_page_links(url: str) -> tuple[str, str, list[dict[str, str]]]:
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    parser = PageMetadataParser()
    parser.feed(response.text)
    link_parser = PageLinkParser()
    link_parser.feed(response.text)
    return parser.title, parser.description, link_parser.links


def source_url_allowed(source: dict[str, Any], url: str) -> bool:
    if not url or url_contains_blocked_substring(url):
        return False

    allowed_prefixes = [str(value) for value in source.get("allowed_url_prefixes", []) if value]
    if allowed_prefixes and not any(url.startswith(prefix) for prefix in allowed_prefixes):
        return False

    blocked_substrings = [str(value).lower() for value in source.get("blocked_url_substrings", []) if value]
    lowered_url = url.lower()
    if any(blocked in lowered_url for blocked in blocked_substrings):
        return False

    parsed = urlparse(url)
    if parsed.path in ("", "/"):
        return False
    if any(blocked in parsed.path.lower() for blocked in BLOCKED_PATH_SUBSTRINGS):
        return False

    base_url = str(source.get("base_url", ""))
    if base_url and not allowed_prefixes:
        base_netloc = urlparse(base_url).netloc
        if base_netloc and parsed.netloc and parsed.netloc != base_netloc:
            return False

    return True


def parse_sitemap(url: str) -> tuple[str, list[dict[str, Any]]]:
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    root = ElementTree.fromstring(response.content)
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag[1:].split("}", 1)[0]

    def qualified(tag_name: str) -> str:
        return f"{{{namespace}}}{tag_name}" if namespace else tag_name

    root_name = root.tag.rsplit("}", 1)[-1]
    items: list[dict[str, Any]] = []
    if root_name == "sitemapindex":
        for sitemap_node in root.findall(qualified("sitemap")):
            items.append(
                {
                    "loc": sitemap_node.findtext(qualified("loc"), default="").strip(),
                    "lastmod": parse_datetime_string(sitemap_node.findtext(qualified("lastmod"))),
                }
            )
        return "index", items

    if root_name == "urlset":
        for url_node in root.findall(qualified("url")):
            items.append(
                {
                    "loc": url_node.findtext(qualified("loc"), default="").strip(),
                    "lastmod": parse_datetime_string(url_node.findtext(qualified("lastmod"))),
                }
            )
        return "urlset", items

    raise ValueError(f"Unsupported sitemap format at {url}")


def collect_sitemap_candidates(
    source: dict[str, Any],
    published_after: datetime | None,
    published_before: datetime | None,
) -> list[dict[str, Any]]:
    sitemap_urls = [str(url) for url in source.get("sitemap_urls", []) if url]
    if not sitemap_urls:
        return []

    pending = list(sitemap_urls)
    seen_sitemaps: set[str] = set()
    candidates: list[dict[str, Any]] = []
    max_nested_sitemaps = int(source.get("max_nested_sitemaps", 25))

    while pending and len(seen_sitemaps) < max_nested_sitemaps:
        sitemap_url = pending.pop(0)
        if sitemap_url in seen_sitemaps:
            continue
        seen_sitemaps.add(sitemap_url)

        kind, items = parse_sitemap(sitemap_url)
        if kind == "index":
            items.sort(key=lambda item: item.get("lastmod") or datetime.min.replace(tzinfo=UTC), reverse=True)
            for item in items:
                loc = str(item.get("loc", ""))
                if not loc or loc in seen_sitemaps or url_contains_blocked_substring(loc):
                    continue
                pending.append(loc)
            continue

        for item in items:
            url = str(item.get("loc", ""))
            published_at = item.get("lastmod")
            if not source_url_allowed(source, url):
                continue
            if not within_publish_window(published_at, published_after, published_before):
                continue
            candidates.append({"url": url, "published_at": published_at})

    deduped: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        existing = deduped.get(candidate["url"])
        if existing is None or (candidate.get("published_at") or datetime.min.replace(tzinfo=UTC)) > (
            existing.get("published_at") or datetime.min.replace(tzinfo=UTC)
        ):
            deduped[candidate["url"]] = candidate

    ordered = list(deduped.values())
    ordered.sort(key=lambda item: item.get("published_at") or datetime.min.replace(tzinfo=UTC), reverse=True)
    return ordered


def search_rss(
    feeds: list[dict[str, Any]],
    query: str,
    limit: int,
    gameweek: int | None,
    strict_gameweek: bool,
    published_after: datetime | None = None,
    published_before: datetime | None = None,
) -> list[dict[str, Any]]:
    lowered_query = query.lower()
    preferred_results: list[dict[str, Any]] = []
    fallback_results: list[dict[str, Any]] = []

    for feed in feeds:
        feed_url = feed["rss_url"]
        if url_contains_blocked_substring(feed_url):
            continue
        parsed = feedparser.parse(feed_url)
        feed_title = parsed.feed.get("title") or feed.get("name", feed_url)
        for entry in parsed.entries:
            title = str(entry.get("title", "Untitled"))
            summary = str(entry.get("summary", ""))
            link = str(entry.get("link", ""))
            if link and url_contains_blocked_substring(link):
                continue
            haystack = (title + "\n" + summary).lower()
            if lowered_query not in haystack:
                continue

            published_at = entry_datetime(entry)
            if published_after is not None and (published_at is None or published_at < published_after):
                continue
            if published_before is not None and (published_at is None or published_at >= published_before):
                continue

            gameweek_match = detect_gameweek_match(title + "\n" + summary, gameweek) if gameweek else None
            if strict_gameweek and gameweek is not None and not gameweek_match:
                continue

            result = {
                "source": str(feed_title),
                "source_id": str(feed.get("id", "")),
                "title": title,
                "link": link,
                "published": str(entry.get("published", entry.get("updated", ""))),
                "gameweek_match": gameweek_match,
                "discovery_method": "rss",
                "_published_at": published_at,
                "_per_source_limit": int(feed.get("max_results_per_source", 4)),
            }
            if gameweek_match:
                preferred_results.append(result)
            else:
                fallback_results.append(result)

    combined_results = preferred_results + fallback_results
    return combined_results[:limit]


def search_sitemaps(
    sources: list[dict[str, Any]],
    query: str,
    limit: int,
    gameweek: int | None,
    strict_gameweek: bool,
    published_after: datetime | None = None,
    published_before: datetime | None = None,
) -> list[dict[str, Any]]:
    lowered_query = query.lower()
    preferred_results: list[dict[str, Any]] = []
    fallback_results: list[dict[str, Any]] = []

    for source in sources:
        max_metadata_fetches = int(source.get("max_metadata_fetches", max(limit * 3, DEFAULT_METADATA_FETCH_LIMIT)))
        candidates = collect_sitemap_candidates(source, published_after, published_before)
        fetched = 0
        for candidate in candidates:
            if fetched >= max_metadata_fetches:
                break
            fetched += 1

            url = candidate["url"]
            try:
                title, description = fetch_page_metadata(url)
            except (requests.RequestException, ValueError):
                continue

            searchable_text = "\n".join(value for value in [title, description, url] if value)
            if lowered_query not in searchable_text.lower():
                continue

            gameweek_match = detect_gameweek_match(searchable_text, gameweek) if gameweek else None
            if strict_gameweek and gameweek is not None and not gameweek_match:
                continue

            published_at = candidate.get("published_at")
            result = {
                "source": str(source.get("name", source.get("id", "unknown"))),
                "source_id": str(source.get("id", "")),
                "title": title or url,
                "link": url,
                "published": published_at.isoformat() if published_at else "",
                "gameweek_match": gameweek_match,
                "discovery_method": "sitemap",
                "_published_at": published_at,
                "_per_source_limit": int(source.get("max_results_per_source", 4)),
            }
            if gameweek_match:
                preferred_results.append(result)
            else:
                fallback_results.append(result)

    combined_results = preferred_results + fallback_results
    return combined_results[:limit]


def search_page_sources(
    sources: list[dict[str, Any]],
    query: str,
    limit: int,
    gameweek: int | None,
    strict_gameweek: bool,
    published_after: datetime | None = None,
    published_before: datetime | None = None,
) -> list[dict[str, Any]]:
    lowered_query = query.lower()
    preferred_results: list[dict[str, Any]] = []
    fallback_results: list[dict[str, Any]] = []

    for source in sources:
        page_urls = [str(url) for url in source.get("page_urls", []) if url]
        if not page_urls:
            continue
        max_links = int(source.get("max_page_links", max(limit * 3, DEFAULT_METADATA_FETCH_LIMIT)))
        seen_links: set[str] = set()

        for page_url in page_urls:
            if url_contains_blocked_substring(page_url):
                continue
            try:
                page_title, page_description, page_links = fetch_page_links(page_url)
            except (requests.RequestException, ValueError):
                continue

            for link in page_links:
                href = urljoin(page_url, link.get("href", "").strip())
                if href in seen_links or not source_url_allowed(source, href):
                    continue
                seen_links.add(href)

                anchor_text = link.get("text", "").strip()
                searchable_text = "\n".join(value for value in [anchor_text, href, page_title, page_description] if value)
                if lowered_query not in searchable_text.lower():
                    continue

                gameweek_match = detect_gameweek_match(searchable_text, gameweek) if gameweek else None
                if strict_gameweek and gameweek is not None and not gameweek_match:
                    continue

                result = {
                    "source": str(source.get("name", source.get("id", "unknown"))),
                    "source_id": str(source.get("id", "")),
                    "title": anchor_text or page_title or href,
                    "link": href,
                    "published": "",
                    "gameweek_match": gameweek_match,
                    "discovery_method": "page",
                    "_published_at": None,
                    "_per_source_limit": int(source.get("max_results_per_source", 4)),
                }
                if gameweek_match:
                    preferred_results.append(result)
                else:
                    fallback_results.append(result)

                if len(seen_links) >= max_links:
                    break
            if len(seen_links) >= max_links:
                break

    combined_results = preferred_results + fallback_results
    return combined_results[:limit]


def finalize_results(results: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(
        results,
        key=lambda item: (
            1 if item.get("gameweek_match") else 0,
            item.get("_published_at") or datetime.min.replace(tzinfo=UTC),
            1 if item.get("discovery_method") == "rss" else 0,
        ),
        reverse=True,
    )

    deduped: list[dict[str, Any]] = []
    seen_links: set[str] = set()
    source_counts: dict[str, int] = {}
    for item in ordered:
        link = str(item.get("link", ""))
        if not link or link in seen_links:
            continue
        source_id = str(item.get("source_id", ""))
        per_source_limit = int(item.get("_per_source_limit", 4))
        if source_counts.get(source_id, 0) >= per_source_limit:
            continue
        seen_links.add(link)
        source_counts[source_id] = source_counts.get(source_id, 0) + 1
        cleaned = {key: value for key, value in item.items() if not key.startswith("_")}
        deduped.append(cleaned)
        if len(deduped) >= limit:
            break

    return deduped


def search_hybrid(
    sources: list[dict[str, Any]],
    query: str,
    limit: int,
    gameweek: int | None,
    strict_gameweek: bool,
    published_after: datetime | None = None,
    published_before: datetime | None = None,
) -> list[dict[str, Any]]:
    rss_sources = [source for source in sources if source.get("rss_url")]
    sitemap_sources = [source for source in sources if source.get("sitemap_urls")]
    page_sources = [source for source in sources if source.get("page_urls")]
    rss_results = search_rss(
        rss_sources,
        query,
        max(limit * 3, limit),
        gameweek,
        strict_gameweek,
        published_after,
        published_before,
    )
    sitemap_results = search_sitemaps(
        sitemap_sources,
        query,
        max(limit * 4, limit),
        gameweek,
        strict_gameweek,
        published_after,
        published_before,
    )
    page_results = search_page_sources(
        page_sources,
        query,
        max(limit * 3, limit),
        gameweek,
        strict_gameweek,
        published_after,
        published_before,
    )
    return finalize_results(rss_results + sitemap_results + page_results, limit)


def search_reddit(query: str, subreddit: str | None, sort: str, limit: int) -> list[dict[str, Any]]:
    session = get_reddit_session()
    subreddit_path = f"/r/{subreddit}" if subreddit else ""
    url = f"https://oauth.reddit.com{subreddit_path}/search?q={quote_plus(query)}&restrict_sr={'true' if subreddit else 'false'}&sort={sort}&limit={limit}"
    response = session.get(
        url,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()

    results = []
    for child in payload.get("data", {}).get("children", []):
        data = child.get("data", {})
        permalink = data.get("permalink", "")
        results.append(
            {
                "subreddit": data.get("subreddit", ""),
                "title": data.get("title", ""),
                "url": f"https://www.reddit.com{permalink}" if permalink else "",
                "score": data.get("score", 0),
                "comments": data.get("num_comments", 0),
                "author": data.get("author", ""),
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="source_type", required=True)

    rss_parser = subparsers.add_parser("rss", help="Search RSS or Atom feeds for matching entries")
    rss_parser.add_argument("--query", required=True, help="Case-insensitive text to match in title/summary")
    rss_parser.add_argument("--feed", action="append", default=[], help="Feed URL to search; repeat to add more")
    rss_parser.add_argument("--gameweek", type=int, help="Prefer entries whose title/summary explicitly mentions this gameweek")
    rss_parser.add_argument(
        "--strict-gameweek",
        action="store_true",
        help="Drop RSS entries that do not explicitly mention the requested gameweek",
    )
    rss_parser.add_argument("--limit", type=int, default=10, help="Maximum number of matches to print")

    hybrid_parser = subparsers.add_parser("hybrid", help="Search the curated source catalog via RSS plus sitemap discovery")
    hybrid_parser.add_argument("--query", required=True, help="Case-insensitive text to match in titles, summaries, and page metadata")
    hybrid_parser.add_argument("--gameweek", type=int, help="Prefer entries whose metadata explicitly mentions this gameweek")
    hybrid_parser.add_argument(
        "--strict-gameweek",
        action="store_true",
        help="Drop entries that do not explicitly mention the requested gameweek",
    )
    hybrid_parser.add_argument("--limit", type=int, default=10, help="Maximum number of matches to print")

    reddit_parser = subparsers.add_parser("reddit", help="Search Reddit posts via Reddit JSON")
    reddit_parser.add_argument("--query", required=True, help="Reddit search query")
    reddit_parser.add_argument("--subreddit", help="Optional subreddit, e.g. FantasyPL")
    reddit_parser.add_argument(
        "--sort",
        choices=["relevance", "new", "top", "comments"],
        default="relevance",
        help="Reddit result sort",
    )
    reddit_parser.add_argument("--limit", type=int, default=10, help="Maximum number of matches to print")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.source_type == "rss":
            if args.feed:
                feeds = [
                    {"id": slug, "name": feed_url, "rss_url": feed_url, "base_url": ""}
                    for slug, feed_url in enumerate(args.feed, start=1)
                ]
            else:
                feeds = get_default_rss_sources()
            results = search_rss(feeds, args.query, args.limit, args.gameweek, args.strict_gameweek)
        elif args.source_type == "hybrid":
            results = search_hybrid(get_default_hybrid_sources(), args.query, args.limit, args.gameweek, args.strict_gameweek)
        elif args.source_type == "reddit":
            results = search_reddit(args.query, args.subreddit, args.sort, args.limit)
        else:
            raise ValueError(f"Unsupported source type: {args.source_type}")
    except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()