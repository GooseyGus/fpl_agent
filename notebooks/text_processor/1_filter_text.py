"""Run a first-pass FPL filter over gameweek article text files.

This script first builds deterministic labeled segments from each ``*.txt`` file
in ``gw/gwN/text/``. It then asks the LLM to group those segments into broad,
coherent sections, label each section as FPL or non-FPL content, and
reconstruct the kept text locally from the retained sections.

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
PROMPT_VERSION = "v3"
DEFAULT_SEGMENT_WORDS = 180
DEFAULT_MAX_WORKERS = 4
KEEP_SECTION_TYPES = {"FPL_CONTENT"}
VALID_SECTION_TYPES = KEEP_SECTION_TYPES | {"NON_FPL_CONTENT"}

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


def _normalise_response(parsed: dict[str, Any], segment_count: int) -> list[dict[str, Any]]:
	candidates: list[dict[str, Any]] = []
	if isinstance(parsed.get("sections"), list):
		for item in parsed["sections"]:
			if not isinstance(item, dict):
				continue
			raw_start_index = item.get("start_index")
			raw_end_index = item.get("end_index")
			if not isinstance(raw_start_index, (int, str)) or not isinstance(raw_end_index, (int, str)):
				continue
			try:
				start_index = int(raw_start_index)
				end_index = int(raw_end_index)
			except (TypeError, ValueError):
				continue
			if start_index < 0 or end_index < start_index or start_index >= segment_count:
				continue
			section_type = str(item.get("type", "FPL_CONTENT")).upper()
			if section_type not in VALID_SECTION_TYPES:
				section_type = "FPL_CONTENT"
			topic = str(item.get("topic", "")).strip() or section_type.replace("_", " ").title()
			candidates.append(
				{
					"start_index": start_index,
					"end_index": min(end_index, segment_count - 1),
					"type": section_type,
					"topic": topic,
				}
			)

	normalised: list[dict[str, Any]] = []
	next_index = 0
	for candidate in sorted(candidates, key=lambda item: (item["start_index"], item["end_index"])):
		if candidate["end_index"] < next_index:
			continue
		if candidate["start_index"] > next_index:
			normalised.append(
				{
					"start_index": next_index,
					"end_index": candidate["start_index"] - 1,
					"type": "FPL_CONTENT",
					"topic": "Unclassified FPL content",
				}
			)
		candidate["start_index"] = max(candidate["start_index"], next_index)
		normalised.append(candidate)
		next_index = candidate["end_index"] + 1
		if next_index >= segment_count:
			break

	if next_index < segment_count:
		normalised.append(
			{
				"start_index": next_index,
				"end_index": segment_count - 1,
				"type": "FPL_CONTENT",
				"topic": "Unclassified FPL content",
			}
		)
	return normalised


def choose_sections(
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
					"You divide a Fantasy Premier League article into broad, coherent sections and identify the FPL content. "
					"Each input item is a deterministic article segment. Group consecutive segments that belong to the same broader discussion, such as transfer targets, captaincy, fixtures, team news, or sponsor copy. "
					"Prefer a small number of substantial sections over many small ones. Start a new section only when the main subject or content type clearly changes. "
					"Classify each section as FPL_CONTENT for substantive FPL discussion, or NON_FPL_CONTENT for intros, sponsor copy, navigation, social prompts, unrelated football news, and filler with no useful FPL information. "
					"Keep supporting context with the FPL discussion it relates to. When uncertain, use FPL_CONTENT. "
					"Return JSON with exactly one key: sections. "
					"sections must cover every input segment exactly once, in order, with no gaps or overlaps. "
					"Each section must have start_index, end_index, type, and topic. "
					"Do not reproduce, rewrite, or quote segment text. Only report section boundaries, types, and topic labels.\n\n"
					f"SOURCE:\n{source_name}\n\nSEGMENTS:\n{segment_json}"
				),
			},
		],
		temperature=0,
	)
	parsed = json.loads(response.choices[0].message.content or "{}")
	section_specs = _normalise_response(parsed, len(segments))
	payload = {
		"source_name": source_name,
		"cache_key": cache_key,
		"source_hash": source_hash,
		"model": model,
		"prompt_version": PROMPT_VERSION,
		"segment_words": segment_words,
		"segment_count": len(segments),
		"section_count": len(section_specs),
		"retained_section_count": sum(1 for section in section_specs if section["type"] in KEEP_SECTION_TYPES),
		"removed_section_count": sum(1 for section in section_specs if section["type"] not in KEEP_SECTION_TYPES),
		"sections": section_specs,
	}
	write_decision_cache(cache_dir, cache_key, payload)
	return payload


def reconstruct_text(segments: list[Segment], sections: list[dict[str, Any]]) -> str:
	retained_sections: list[str] = []
	for section in sections:
		if section.get("type") not in KEEP_SECTION_TYPES:
			continue
		start_index = int(section["start_index"])
		end_index = int(section["end_index"])
		retained_text = "\n\n".join(
			segment.text for segment in segments[start_index : end_index + 1] if segment.text.strip()
		).strip()
		if retained_text:
			retained_sections.append(retained_text)
	return "\n\n".join(retained_sections).strip()


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
				"section_count": 0,
				"retained_section_count": 0,
				"removed_section_count": 0,
				"segment_words": segment_words,
				"sections": [],
			},
		)

	segments = build_segments(raw_text, segment_words)
	source_name = text_path.stem
	source_hash = _source_hash(raw_text)
	decision = choose_sections(model, cache_dir, source_name, raw_text, segments, segment_words, refresh=refresh)
	section_specs = list(decision.get("sections", []))
	filtered_text = reconstruct_text(segments, section_specs)
	section_records = [
		{
			**section,
			"text": "\n\n".join(
				segment.text for segment in segments[section["start_index"] : section["end_index"] + 1] if segment.text.strip()
			).strip(),
		}
		for section in section_specs
	]
	retained_section_count = sum(1 for section in section_specs if section.get("type") in KEEP_SECTION_TYPES)
	removed_section_count = sum(1 for section in section_specs if section.get("type") not in KEEP_SECTION_TYPES)
	return FilteredText(
		source_name=source_name,
		text=filtered_text,
		metadata={
			"source_file": text_path.name,
			"source_hash": source_hash,
			"segment_count": len(segments),
			"section_count": len(section_specs),
			"retained_section_count": retained_section_count,
			"removed_section_count": removed_section_count,
			"segment_words": segment_words,
			"sections": section_records,
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

	text_paths = [text_path for text_path in sorted(source_dir.glob("*.txt")) if text_path.name != "text_urls.txt"]
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
			f"{filtered.metadata.get('section_count', 0)} section(s), "
			f"{filtered.metadata.get('retained_section_count', 0)} retained, "
			f"{filtered.metadata.get('removed_section_count', 0)} removed)"
		)


if __name__ == "__main__":
	main()