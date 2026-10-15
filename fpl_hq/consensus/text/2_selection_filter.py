from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from fpl_hq.paths import ENV_FILE
DEFAULT_MODEL = "gpt-4.1-mini"
FILTER_PROMPT_VERSION = "v1"
DEFAULT_MAX_WORKERS = 4
KEEP_SECTION_TYPES = {"FPL_CONTENT"}

FILTER_RESPONSE_FORMAT = {
	"type": "json_schema",
	"json_schema": {
		"name": "text_section_filter",
		"strict": True,
		"schema": {
			"type": "object",
			"properties": {
				"sections": {
					"type": "array",
					"items": {
						"type": "object",
						"properties": {
							"section_index": {"type": "integer", "minimum": 0},
							"retain": {"type": "boolean"},
							"reason": {"type": "string"},
						},
						"required": ["section_index", "retain", "reason"],
						"additionalProperties": False,
					},
				},
			},
			"required": ["sections"],
			"additionalProperties": False,
		},
	},
}

FILTER_PROMPT_TEMPLATE = """You filter retained Fantasy Premier League (FPL) article sections before recommendation extraction.

Each input item is one section from a previous broad segmentation pass. Your job is only to decide whether the section should remain for upcoming-gameweek recommendation extraction.

Keep a section only if it is materially actionable for the upcoming gameweek:
- current buy, sell, hold, start, bench, captain, transfer, wildcard, punt, or team-selection advice
- a current judgment that affects a near-term decision
- a retrospective result that is explicitly used to justify a near-term decision

Drop a section if it is only:
- a recap of the previous gameweek
- score, rank, points, haul, or blank commentary with no current decision attached
- hindsight commentary that does not change the next move
- generic narrative filler, sponsor copy, or historical context

If a section is mixed, keep it only when the current or future-facing decision is materially present.
If you are unsure, keep the section.

Return strict JSON with exactly one key: `sections`.

Each item in `sections` must include:
- `section_index`: zero-based index of the input section
- `retain`: boolean
- `reason`: a short explanation

Do not rewrite section text. Do not invent content. Preserve the input order exactly.

SOURCE:
{source}

SECTIONS:
{sections}
"""


@dataclass
class SelectedText:
	source_name: str
	text: str
	metadata: dict[str, Any]


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


def _get_client() -> OpenAI:
	client = getattr(_thread_local, "client", None)
	if client is None:
		client = OpenAI()
		_thread_local.client = client
	return client


_thread_local = threading.local()
load_dotenv(ENV_FILE)


def _source_name(source: Any) -> str:
	if hasattr(source, "source_name"):
		return str(getattr(source, "source_name"))
	if isinstance(source, dict):
		return str(source.get("source_name", "")).strip()
	return ""


def _source_text(source: Any) -> str:
	if hasattr(source, "text"):
		return str(getattr(source, "text"))
	if isinstance(source, dict):
		return str(source.get("text", ""))
	return ""


def _source_metadata(source: Any) -> dict[str, Any]:
	if hasattr(source, "metadata"):
		metadata = getattr(source, "metadata")
		return metadata if isinstance(metadata, dict) else {}
	if isinstance(source, dict):
		metadata = source.get("metadata", {})
		return metadata if isinstance(metadata, dict) else {}
	return {}


def _section_payload(source: Any) -> list[dict[str, Any]]:
	metadata = _source_metadata(source)
	sections = metadata.get("sections", [])
	payload: list[dict[str, Any]] = []
	if not isinstance(sections, list):
		return payload
	for section in sections:
		if not isinstance(section, dict):
			continue
		section_type = str(section.get("type", "FPL_CONTENT")).strip() or "FPL_CONTENT"
		if section_type not in KEEP_SECTION_TYPES:
			continue
		text = str(section.get("text", "")).strip()
		if not text:
			continue
		payload.append(
			{
				"section_index": len(payload),
				"type": section_type,
				"topic": str(section.get("topic", "")).strip(),
				"text": text,
			}
		)
	return payload


def _normalise_response(parsed: dict[str, Any], section_count: int) -> list[dict[str, Any]]:
	decisions = [
		{
			"section_index": index,
			"retain": True,
			"reason": "Defaulted to keep because no valid decision was returned.",
		}
		for index in range(section_count)
	]
	if section_count <= 0:
		return decisions

	for item in parsed.get("sections", []):
		if not isinstance(item, dict):
			continue
		raw_section_index = item.get("section_index")
		if not isinstance(raw_section_index, (int, str)):
			continue
		try:
			section_index = int(raw_section_index)
		except (TypeError, ValueError):
			continue
		if section_index < 0 or section_index >= section_count:
			continue
		retain = bool(item.get("retain"))
		reason = str(item.get("reason", "")).strip()
		if not reason:
			reason = "Retain because the section is actionable." if retain else "Drop because the section is retrospective only."
		decisions[section_index] = {
			"section_index": section_index,
			"retain": retain,
			"reason": reason,
		}
	return decisions


def select_text_source(
	model: str,
	cache_dir: Path,
	source_name: str,
	source: Any,
	refresh: bool = False,
) -> SelectedText:
	sections = _section_payload(source)
	if not sections:
		return SelectedText(
			source_name=source_name,
			text="",
			metadata={
				"source_name": source_name,
				"section_count": 0,
				"retained_section_count": 0,
				"removed_section_count": 0,
				"sections": [],
			},
		)

	section_json = json.dumps(sections, ensure_ascii=False, sort_keys=True)
	key = _cache_key("text-selection", FILTER_PROMPT_VERSION, model, source_name, section_json)
	cached = None if refresh else _read_cache(cache_dir, key)
	if cached is not None:
		decisions = _normalise_response(cached if isinstance(cached, dict) else {}, len(sections))
	else:
		response = _get_client().chat.completions.create(
			model=model,
			response_format=FILTER_RESPONSE_FORMAT,
			messages=[
				{"role": "system", "content": "Return valid JSON only."},
				{
					"role": "user",
					"content": FILTER_PROMPT_TEMPLATE.format(
						source=source_name,
						sections=json.dumps(sections, ensure_ascii=False, indent=2),
					),
				},
			],
			temperature=0,
		)
		parsed = json.loads(response.choices[0].message.content or "{}")
		decisions = _normalise_response(parsed if isinstance(parsed, dict) else {}, len(sections))
		_write_cache(cache_dir, key, {"sections": decisions})

	retained_sections: list[dict[str, Any]] = []
	removed_sections: list[dict[str, Any]] = []
	for section, decision in zip(sections, decisions):
		section_record = {
			"section_index": decision["section_index"],
			"type": section["type"],
			"topic": section["topic"],
			"text": section["text"],
			"retain": decision["retain"],
			"reason": decision["reason"],
		}
		if decision["retain"]:
			retained_sections.append(section_record)
		else:
			removed_sections.append(section_record)

	selected_text = "\n\n".join(section["text"] for section in retained_sections).strip()
	return SelectedText(
		source_name=source_name,
		text=selected_text,
		metadata={
			"source_name": source_name,
			"section_count": len(sections),
			"retained_section_count": len(retained_sections),
			"removed_section_count": len(removed_sections),
			"sections": retained_sections + removed_sections,
			"retained_sections": retained_sections,
			"removed_sections": removed_sections,
		},
	)


def select_text_sources(
	filtered_text_sources: dict[str, Any],
	cache_dir: Path,
	model: str = DEFAULT_MODEL,
	max_workers: int = DEFAULT_MAX_WORKERS,
	refresh: bool = False,
) -> dict[str, SelectedText]:
	if max_workers < 1:
		raise ValueError("max_workers must be at least 1")

	sources = sorted(filtered_text_sources.items())
	if not sources:
		return {}

	results: dict[str, SelectedText] = {}
	worker_count = min(max_workers, len(sources))
	with ThreadPoolExecutor(max_workers=worker_count) as executor:
		future_map = {
			executor.submit(select_text_source, model, cache_dir, source_name, source, refresh): source_name
			for source_name, source in sources
		}
		for future in as_completed(future_map):
			result = future.result()
			results[result.source_name] = result

	return dict(sorted(results.items()))
