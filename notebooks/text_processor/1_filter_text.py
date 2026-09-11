"""Run a first-pass FPL filter over gameweek article text files.

This script first builds deterministic labeled segments from each ``*.txt`` file
in ``gw/gwN/text/``. It then asks the LLM to return only the segment ids that
should be excluded, and reconstructs the kept text locally.

Example:
	python notebooks/text_processor/1_filter_text.py --gameweek 3
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "gpt-4.1-mini"
PROMPT_VERSION = "v2"
DEFAULT_SEGMENT_WORDS = 180
DEFAULT_MAX_WORKERS = 4

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Segment:
	id: str
	text: str
	word_count: int
	unit_count: int


@dataclass
class FilteredText:
	source_name: str
	text: str
	metadata: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--gameweek", type=int, required=True, help="Gameweek folder to read from")
	parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use for filtering")
	parser.add_argument(
		"--source-dir",
		type=Path,
		help="Optional explicit source folder; defaults to gw/gwN/text",
	)
	parser.add_argument(
		"--cache-dir",
		type=Path,
		help="Optional explicit cache folder; defaults to gw/gwN/text_cache/filter_cache",
	)
	parser.add_argument(
		"--segment-words",
		type=int,
		default=DEFAULT_SEGMENT_WORDS,
		help="Target word count per deterministic segment",
	)
	parser.add_argument(
		"--max-workers",
		type=int,
		default=DEFAULT_MAX_WORKERS,
		help="Maximum number of files to process in parallel",
	)
	parser.add_argument(
		"--refresh",
		action="store_true",
		help="Ignore cached exclusion decisions and recompute them",
	)
	return parser


def _cache_key(*parts: str) -> str:
	digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
	return digest[:24]


def _read_cache(cache_dir: Path, key: str) -> dict[str, Any] | None:
	path = cache_dir / f"{key}.json"
	if not path.exists():
		return None
	with path.open(encoding="utf-8") as file:
		return json.load(file)


def _write_cache(cache_dir: Path, key: str, payload: dict[str, Any]) -> None:
	cache_dir.mkdir(parents=True, exist_ok=True)
	(cache_dir / f"{key}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _display_path(path: Path) -> str:
	try:
		return str(path.relative_to(PROJECT_ROOT))
	except ValueError:
		return str(path)


def count_words(text: str) -> int:
	return len(text.split())


def split_units(text: str) -> list[str]:
	paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
	if len(paragraphs) > 1:
		return paragraphs
	return [line.strip() for line in text.splitlines() if line.strip()]


def build_segments(text: str, segment_words: int) -> list[Segment]:
	if segment_words < 1:
		raise ValueError("segment_words must be at least 1")

	units = split_units(text)
	segments: list[Segment] = []
	current_units: list[str] = []
	current_word_count = 0

	for unit in units:
		unit_words = count_words(unit)
		if current_units and current_word_count + unit_words > segment_words:
			segments.append(
				Segment(
					id=f"S{len(segments) + 1:03d}",
					text="\n\n".join(current_units).strip(),
					word_count=current_word_count,
					unit_count=len(current_units),
				)
			)
			current_units = []
			current_word_count = 0

		current_units.append(unit)
		current_word_count += unit_words

	if current_units:
		segments.append(
			Segment(
				id=f"S{len(segments) + 1:03d}",
				text="\n\n".join(current_units).strip(),
				word_count=current_word_count,
				unit_count=len(current_units),
			)
		)

	return segments


def _source_hash(text: str) -> str:
	return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _segment_payload(segments: list[Segment]) -> list[dict[str, Any]]:
	return [
		{
			"id": segment.id,
			"word_count": segment.word_count,
			"unit_count": segment.unit_count,
			"text": segment.text,
		}
		for segment in segments
	]


def _decision_cache_path(cache_dir: Path, key: str) -> Path:
	return cache_dir / f"{key}.json"


def load_cached_decision(
	cache_dir: Path,
	key: str,
	model: str,
) -> dict[str, Any] | None:
	path = _decision_cache_path(cache_dir, key)
	if not path.exists():
		return None
	with path.open(encoding="utf-8") as file:
		payload = json.load(file)
	if payload.get("model") != model or payload.get("prompt_version") != PROMPT_VERSION:
		return None
	return payload


def write_decision_cache(cache_dir: Path, key: str, payload: dict[str, Any]) -> None:
	cache_dir.mkdir(parents=True, exist_ok=True)
	_decision_cache_path(cache_dir, key).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


_thread_local = threading.local()


def get_client() -> OpenAI:
	client = getattr(_thread_local, "client", None)
	if client is None:
		client = OpenAI()
		_thread_local.client = client
	return client


def choose_exclusions(
	model: str,
	cache_dir: Path,
	source_name: str,
	text: str,
	segments: list[Segment],
	segment_words: int,
	refresh: bool = False,
) -> dict[str, Any]:
	source_hash = _source_hash(text)
	cache_key = _cache_key("filter", PROMPT_VERSION, model, source_name, text, str(segment_words))
	cached = None if refresh else load_cached_decision(cache_dir, cache_key, model)
	if cached is not None:
		return cached

	segment_json = json.dumps(_segment_payload(segments), ensure_ascii=False)
	response = get_client().chat.completions.create(
		model=model,
		response_format={"type": "json_object"},
		messages=[
			{"role": "system", "content": "Return valid JSON only."},
			{
				"role": "user",
				"content": (
					"You review deterministic segments from a Fantasy Premier League article. "
					"Your job is only to identify segments that should be excluded from a first-pass FPL corpus. "
					"Keep segments that contain FPL strategy, player discussion, transfers, fixtures, captaincy, chips, injuries, stats, or analyst opinion. "
					"Exclude only segments that are intros, sponsor copy, social prompts, comments, navigation, unrelated football news, or filler with no useful FPL content. "
					"Be conservative: if a segment has meaningful FPL content, keep it. "
					"Return JSON with exactly one key: excluded_segment_ids. "
					"excluded_segment_ids must be an array of segment ids to drop, and nothing else.\n\n"
					f"SOURCE:\n{source_name}\n\nSEGMENTS:\n{segment_json}"
				),
			},
		],
		temperature=0,
	)
	parsed = json.loads(response.choices[0].message.content or "{}")
	valid_segment_ids = {segment.id for segment in segments}
	excluded_segment_ids = [
		str(segment_id)
		for segment_id in parsed.get("excluded_segment_ids", [])
		if str(segment_id) in valid_segment_ids
	]
	kept_segment_ids = [segment.id for segment in segments if segment.id not in set(excluded_segment_ids)]
	payload = {
		"source_name": source_name,
		"cache_key": cache_key,
		"source_hash": source_hash,
		"model": model,
		"prompt_version": PROMPT_VERSION,
		"segment_words": segment_words,
		"segment_count": len(segments),
		"excluded_segment_ids": excluded_segment_ids,
		"kept_segment_ids": kept_segment_ids,
	}
	write_decision_cache(cache_dir, cache_key, payload)
	return payload


def reconstruct_text(segments: list[Segment], excluded_segment_ids: list[str]) -> str:
	excluded = set(excluded_segment_ids)
	kept_segments = [segment.text for segment in segments if segment.id not in excluded and segment.text.strip()]
	return "\n\n".join(kept_segments).strip()


def process_file(
	text_path: Path,
	model: str,
	cache_dir: Path,
	segment_words: int,
	refresh: bool,
) -> FilteredText:

	raw_text = text_path.read_text(encoding="utf-8", errors="replace").strip()
	if not raw_text:
		return FilteredText(
			source_name=text_path.stem,
			text="",
			metadata={
				"source_file": text_path.name,
				"segment_count": 0,
				"excluded_count": 0,
				"kept_count": 0,
			},
		)

	segments = build_segments(raw_text, segment_words)
	source_name = text_path.stem
	source_hash = _source_hash(raw_text)
	decision = choose_exclusions(model, cache_dir, source_name, raw_text, segments, segment_words, refresh=refresh)
	filtered_text = reconstruct_text(segments, list(decision.get("excluded_segment_ids", [])))
	excluded_count = len(decision.get("excluded_segment_ids", []))
	kept_count = len(segments) - excluded_count
	return FilteredText(
		source_name=source_name,
		text=filtered_text,
		metadata={
			"source_file": text_path.name,
			"source_hash": source_hash,
			"segment_count": len(segments),
			"excluded_count": excluded_count,
			"kept_count": kept_count,
			"segment_words": segment_words,
			"excluded_segment_ids": list(decision.get("excluded_segment_ids", [])),
			"kept_segment_ids": list(decision.get("kept_segment_ids", [])),
		},
	)


def process_text_folder(
	gameweek: int,
	model: str = DEFAULT_MODEL,
	source_dir: Path | None = None,
	cache_dir: Path | None = None,
	segment_words: int = DEFAULT_SEGMENT_WORDS,
	max_workers: int = DEFAULT_MAX_WORKERS,
	refresh: bool = False,
) -> dict[str, FilteredText]:
	source_dir = source_dir or (PROJECT_ROOT / "gw" / f"gw{gameweek}" / "text")
	cache_dir = cache_dir or (PROJECT_ROOT / "gw" / f"gw{gameweek}" / "text_cache" / "filter_cache")

	if not source_dir.exists():
		raise FileNotFoundError(f"Missing source folder: {_display_path(source_dir)}")
	if max_workers < 1:
		raise ValueError("max_workers must be at least 1")

	text_paths = sorted(source_dir.glob("*.txt"))
	if not text_paths:
		return {}

	cache_dir.mkdir(parents=True, exist_ok=True)
	results: dict[str, FilteredText] = {}
	worker_count = min(max_workers, len(text_paths))
	with ThreadPoolExecutor(max_workers=worker_count) as executor:
		future_map = {
			executor.submit(
				process_file,
				text_path,
				model,
				cache_dir,
				segment_words,
				refresh,
			): text_path
			for text_path in text_paths
		}
		for future in as_completed(future_map):
			result = future.result()
			results[result.source_name] = result

	return dict(sorted(results.items()))


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	results = process_text_folder(
		gameweek=args.gameweek,
		model=args.model,
		source_dir=args.source_dir,
		cache_dir=args.cache_dir,
		segment_words=args.segment_words,
		max_workers=args.max_workers,
		refresh=args.refresh,
	)
	if not results:
		source_dir = args.source_dir or (PROJECT_ROOT / "gw" / f"gw{args.gameweek}" / "text")
		print(f"No text files found in {_display_path(source_dir)}")
		return

	for source_name, filtered in results.items():
		print(
			f"Processed {source_name} "
			f"({filtered.metadata.get('segment_count', 0)} segment(s), "
			f"{filtered.metadata.get('excluded_count', 0)} excluded, "
			f"{filtered.metadata.get('kept_count', 0)} kept)"
		)


if __name__ == "__main__":
	main()