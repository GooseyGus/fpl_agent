"""Fetch text content for the FPL pipeline from URLs, Reddit, RSS entries, or a
gameweek URL list.

Usage examples:
	python -m fpl_hq.gather.fetch_text url 'https://example.com/article' --gameweek 4 --name article_name
	python -m fpl_hq.gather.fetch_text reddit 'https://www.reddit.com/r/FantasyPL/comments/abc123/post_title/' --gameweek 4
	python -m fpl_hq.gather.fetch_text rss-item 'https://feed.url/rss' --match 'Saka' --gameweek 4
	python -m fpl_hq.gather.fetch_text --gameweek 4 --method text_file

Without --gameweek/--out, the extracted text is printed to stdout.
With --gameweek N, the text is saved to data/2026-27/gameweeks/gwN/text/<NAME>.txt.
With --method text_file, URLs are read from data/2026-27/gameweeks/gwN/text/text_urls.txt and each
result is written into the same folder.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
import re
import sys
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import feedparser
import requests
import trafilatura
from dotenv import load_dotenv

from fpl_hq.paths import ENV_FILE, PROJECT_ROOT, gw_dir
USER_AGENT = "fpl-agent/1.0 (+https://github.com/)"

DEFAULT_TIMEOUT_SECONDS = 20

load_dotenv(ENV_FILE)


@dataclass
class ExtractedText:
	source_name: str
	text: str
	metadata: dict[str, Any]


def slugify(value: str) -> str:
	cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
	return cleaned.strip("_") or "text_source"


def normalize_text_url(url: str) -> str:
	candidate = url.strip()
	parsed = urlparse(candidate)
	host = parsed.netloc.lower()
	if "google." in host and parsed.path == "/url":
		params = parse_qs(parsed.query)
		for key in ("url", "q", "u"):
			target = params.get(key, [""])[0].strip()
			if target:
				return unquote(target)
	return candidate


def extract_publication_year(downloaded: str | None, url: str) -> int | None:
	metadata = trafilatura.extract_metadata(downloaded or url)
	date_value = getattr(metadata, "date", None) if metadata is not None else None
	if not date_value:
		return None
	date_text = str(date_value)
	match = re.search(r"\b(19|20)\d{2}\b", date_text)
	if match:
		return int(match.group(0))
	try:
		return datetime.fromisoformat(date_text).year
	except ValueError:
		return None


def load_url_list(path: Path) -> list[str]:
	return [
		line.strip()
		for line in path.read_text(encoding="utf-8").splitlines()
		if line.strip() and not line.lstrip().startswith("#")
	]


def derive_output_path(gameweek: int | None, out: Path | None, name: str | None, fallback_name: str) -> Path | None:
	if out:
		return out
	if not gameweek:
		return None
	text_dir = gw_dir(gameweek) / "text"
	text_dir.mkdir(parents=True, exist_ok=True)
	return text_dir / f"{name or fallback_name}.txt"


def text_dir_for_gameweek(gameweek: int) -> Path:
	text_dir = gw_dir(gameweek) / "text"
	text_dir.mkdir(parents=True, exist_ok=True)
	return text_dir


def manifest_path_for_gameweek(gameweek: int) -> Path:
	data_dir = gw_dir(gameweek) / "data"
	data_dir.mkdir(parents=True, exist_ok=True)
	return data_dir / f"gw_{gameweek}_source_manifest.json"


def load_manifest(path: Path) -> list[dict[str, Any]]:
	if not path.exists():
		return []
	if path.stat().st_size == 0:
		return []
	with path.open(encoding="utf-8") as file:
		try:
			return json.load(file)
		except json.JSONDecodeError:
			return []


def write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
	path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def update_manifest(gameweek: int, out_path: Path, result: ExtractedText) -> None:
	manifest_path = manifest_path_for_gameweek(gameweek)
	records = load_manifest(manifest_path)
	try:
		output_relative = str(out_path.relative_to(PROJECT_ROOT))
	except ValueError:
		output_relative = str(out_path)

	record = {
		"output_file": output_relative,
		"fetched_at_utc": datetime.now(UTC).isoformat(),
		"source_name": result.source_name,
		**result.metadata,
	}

	updated = False
	for index, existing in enumerate(records):
		if existing.get("output_file") == output_relative:
			records[index] = record
			updated = True
			break
	if not updated:
		records.append(record)

	write_manifest(manifest_path, records)


def fetch_html(url: str) -> str:
	response = requests.get(
		url,
		headers={"User-Agent": USER_AGENT},
		timeout=DEFAULT_TIMEOUT_SECONDS,
	)
	response.raise_for_status()
	return response.text


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


def extract_article_text(url: str, min_year: int | None = None) -> ExtractedText:
	resolved_url = normalize_text_url(url)
	parsed_url = urlparse(resolved_url)
	downloaded = trafilatura.fetch_url(resolved_url)
	if downloaded is None:
		raise ValueError(f"Unable to download article content from {resolved_url}")

	publication_year = extract_publication_year(downloaded, resolved_url)
	if min_year is not None:
		if publication_year is None:
			raise ValueError(f"Unable to determine publication year for {resolved_url}")
		if publication_year < min_year:
			raise ValueError(f"Skipping article from {publication_year}: {resolved_url}")

	extracted = trafilatura.extract(
		downloaded,
		include_comments=True,
		include_links=False,
		favor_precision=True,
	)
	if not extracted:
		raise ValueError(f"Unable to extract readable text from {resolved_url}")

	path_tail = parsed_url.path.rstrip("/").rsplit("/", 1)[-1]
	fallback_name = slugify(path_tail or parsed_url.netloc or "article")
	metadata: dict[str, Any] = {
		"source_type": "url",
		"original_url": url,
		"resolved_url": resolved_url,
	}
	if publication_year is not None:
		metadata["publication_year"] = publication_year
	if resolved_url != url:
		metadata["normalized_from_url"] = url
	return ExtractedText(
		source_name=fallback_name,
		text=extracted.strip(),
		metadata=metadata,
	)


def flatten_reddit_comment_tree(children: list[dict[str, Any]], lines: list[str]) -> None:
	for child in children:
		data = child.get("data", {})
		body = data.get("body")
		author = data.get("author")
		if body:
			lines.append(f"[{author}] {body.strip()}")
		replies = data.get("replies")
		if isinstance(replies, dict):
			flatten_reddit_comment_tree(replies.get("data", {}).get("children", []), lines)


def normalize_reddit_json_url(url: str) -> str:
	cleaned = url.rstrip("/")
	if cleaned.endswith(".json"):
		return cleaned
	return f"{cleaned}.json"


def extract_reddit_text(url: str) -> ExtractedText:
	json_url = normalize_reddit_json_url(url)
	try:
		session = get_reddit_session()
		oauth_url = json_url.replace("https://www.reddit.com", "https://oauth.reddit.com")
		response = session.get(oauth_url, timeout=DEFAULT_TIMEOUT_SECONDS)
	except ValueError:
		response = requests.get(
			json_url,
			headers={"User-Agent": USER_AGENT},
			timeout=DEFAULT_TIMEOUT_SECONDS,
		)
	response.raise_for_status()
	payload = response.json()
	if not isinstance(payload, list) or len(payload) < 2:
		raise ValueError(f"Unexpected Reddit response for {url}")

	post_children = payload[0].get("data", {}).get("children", [])
	if not post_children:
		raise ValueError(f"No Reddit post found for {url}")

	post = post_children[0].get("data", {})
	title = unescape(post.get("title", "")).strip()
	selftext = unescape(post.get("selftext", "")).strip()
	author = post.get("author", "unknown")
	subreddit = post.get("subreddit", "unknown")

	lines = [
		f"TITLE: {title}",
		f"SUBREDDIT: r/{subreddit}",
		f"AUTHOR: {author}",
	]
	if selftext:
		lines.extend(["", selftext])

	comment_lines: list[str] = []
	comment_children = payload[1].get("data", {}).get("children", [])
	flatten_reddit_comment_tree(comment_children, comment_lines)
	if comment_lines:
		lines.extend(["", "COMMENTS:", *comment_lines])

	return ExtractedText(
		source_name=slugify(title or post.get("id", "reddit_post")),
		text="\n".join(lines).strip(),
		metadata={
			"source_type": "reddit",
			"original_url": url,
			"title": title,
			"subreddit": subreddit,
			"author": author,
		},
	)


def select_rss_entry(feed_url: str, match: str | None, index: int, gameweek: int | None, strict_gameweek: bool) -> dict[str, Any]:
	feed = feedparser.parse(feed_url)
	if getattr(feed, "bozo", 0):
		parse_exception = getattr(feed, "bozo_exception", None)
		if not feed.entries:
			raise ValueError(f"Unable to parse RSS feed {feed_url}: {parse_exception}")

	entries = list(feed.entries)
	all_entries = list(entries)
	selection_mode = "all"
	if match:
		lowered_match = match.lower()
		matched_entries = [
			entry
			for entry in entries
			if lowered_match in (entry.get("title", "") + "\n" + entry.get("summary", "")).lower()
		]
		if matched_entries:
			entries = matched_entries
			selection_mode = "match"
		elif gameweek is not None and not strict_gameweek:
			entries = all_entries
			selection_mode = "match-fallback-gw"
		elif strict_gameweek:
			raise ValueError(f"No RSS entries matched in {feed_url}")

	if gameweek is not None:
		matching_entries = [
			entry
			for entry in entries
			if any(
				re.search(pattern, (entry.get("title", "") + "\n" + entry.get("summary", "")).lower())
				for pattern in [
					rf"\bgw\s*{gameweek}\b",
					rf"\bgameweek\s*{gameweek}\b",
					rf"\bgame\s*week\s*{gameweek}\b",
				]
			)
		]
		if strict_gameweek:
			entries = matching_entries
			selection_mode = f"gw{gameweek}-strict"
		elif matching_entries:
			remaining_entries = [entry for entry in entries if entry not in matching_entries]
			entries = matching_entries + remaining_entries
			if selection_mode == "all":
				selection_mode = f"gw{gameweek}-preferred"
			elif selection_mode == "match-fallback-gw":
				selection_mode = f"match-fallback-gw{gameweek}"

	if not entries:
		raise ValueError(f"No RSS entries matched in {feed_url}")
	try:
		entry = entries[index]
		entry = dict(entry)
		entry["_selection_mode"] = selection_mode
		return entry
	except IndexError as exc:
		raise ValueError(f"RSS entry index {index} is out of range for {feed_url}") from exc


def extract_rss_entry_text(feed_url: str, match: str | None, index: int, gameweek: int | None, strict_gameweek: bool) -> ExtractedText:
	entry = select_rss_entry(feed_url, match, index, gameweek, strict_gameweek)
	link = entry.get("link")
	if not link:
		raise ValueError(f"RSS entry in {feed_url} did not contain a link")
	article = extract_article_text(link)
	article.metadata.update(
		{
			"source_type": "rss-item",
			"feed_url": feed_url,
			"original_url": link,
			"title": str(entry.get("title", "")),
			"published": str(entry.get("published", entry.get("updated", ""))),
			"selection_mode": str(entry.get("_selection_mode", "unknown")),
			"gameweek_match": gameweek is not None
			and any(
				re.search(pattern, (str(entry.get("title", "")) + "\n" + str(entry.get("summary", ""))).lower())
				for pattern in [
					rf"\bgw\s*{gameweek}\b",
					rf"\bgameweek\s*{gameweek}\b",
					rf"\bgame\s*week\s*{gameweek}\b",
				]
			),
		}
	)
	return article


def write_or_print_result(result: ExtractedText, gameweek: int | None, out: Path | None, name: str | None) -> None:
	out_path = derive_output_path(gameweek, out, name, result.source_name)
	if out_path is None:
		print(result.text)
		return

	out_path.write_text(result.text, encoding="utf-8")
	if gameweek is not None:
		update_manifest(gameweek, out_path, result)
	try:
		relative = out_path.relative_to(PROJECT_ROOT)
	except ValueError:
		relative = out_path
	print(f"Text written to {relative}")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"source_type",
		nargs="?",
		choices=["url", "reddit", "rss-item"],
		help="Source type to fetch in single-item mode",
	)
	parser.add_argument(
		"source_value",
		nargs="?",
		help="URL or feed URL for the selected source type",
	)
	parser.add_argument(
		"--method",
		choices=["single", "text_file"],
		default="single",
		help="Fetch one source or all URLs from data/2026-27/gameweeks/gwN/text/text_urls.txt",
	)
	parser.add_argument("--gameweek", type=int, help="Save into data/2026-27/gameweeks/gw{N}/text/")
	parser.add_argument("--out", type=Path, help="Explicit output file path")
	parser.add_argument("--name", help="Output filename stem")
	parser.add_argument("--min-year", type=int, help="Reject articles published before this year")
	parser.add_argument("--match", help="Case-insensitive title/summary filter")
	parser.add_argument("--index", type=int, default=0, help="Matched entry index to fetch (default: 0)")
	parser.add_argument(
		"--strict-gameweek",
		action="store_true",
		help="Drop RSS entries that do not explicitly mention the requested gameweek",
	)

	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	try:
		if args.method == "text_file":
			if args.source_type or args.source_value:
				parser.error("--method text_file cannot be combined with a source_type or source_value")
			if args.out or args.name or args.match or args.index != 0 or args.strict_gameweek:
				parser.error("--method text_file cannot be combined with --out, --name, --match, --index, or --strict-gameweek")
			if args.gameweek is None:
				parser.error("--method text_file requires --gameweek")

			text_dir = text_dir_for_gameweek(args.gameweek)
			urls_path = text_dir / "text_urls.txt"
			if not urls_path.is_file():
				parser.error(f"URL list not found: {urls_path}")

			urls = load_url_list(urls_path)
			if not urls:
				parser.error(f"No URLs found in {urls_path}")

			unique_urls: list[str] = []
			seen_urls: set[str] = set()
			for url in urls:
				normalized_url = normalize_text_url(url)
				if normalized_url in seen_urls:
					continue
				seen_urls.add(normalized_url)
				unique_urls.append(url)
			duplicate_count = len(urls) - len(unique_urls)
			if duplicate_count:
				print(f"Skipped {duplicate_count} duplicate URL entr{'y' if duplicate_count == 1 else 'ies'}.")

			written = 0
			failed = 0
			for url in unique_urls:
				try:
					result = extract_article_text(url, args.min_year)
				except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
					failed += 1
					print(f"Error: {exc}", file=sys.stderr)
					continue

				out_path = text_dir / f"{result.source_name}.txt"
				if out_path.exists():
					print(f"Skipped existing text file {out_path.relative_to(PROJECT_ROOT)}")
					continue

				out_path.write_text(result.text, encoding="utf-8")
				update_manifest(args.gameweek, out_path, result)
				print(f"Text written to {out_path.relative_to(PROJECT_ROOT)}")
				written += 1

			print(f"Fetched {written} of {len(unique_urls)} text sources.")
			if written == 0 and failed:
				sys.exit(1)
			return

		if not args.source_type or not args.source_value:
			parser.error("source_type and source_value are required unless --method text_file is used")

		if args.source_type == "url":
			result = extract_article_text(args.source_value, args.min_year)
		elif args.source_type == "reddit":
			result = extract_reddit_text(args.source_value)
		elif args.source_type == "rss-item":
			result = extract_rss_entry_text(args.source_value, args.match, args.index, args.gameweek, args.strict_gameweek)
		else:
			raise ValueError(f"Unsupported source type: {args.source_type}")

		write_or_print_result(result, args.gameweek, args.out, args.name)
	except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
		print(f"Error: {exc}", file=sys.stderr)
		sys.exit(1)


if __name__ == "__main__":
	main()
