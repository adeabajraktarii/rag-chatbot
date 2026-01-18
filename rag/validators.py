from __future__ import annotations

import json
from typing import Any

from rag.prompts import DONT_KNOW


def parse_json_or_none(raw: str) -> dict | None:
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def unique_sources(sources: list[Any]) -> list[dict]:
    seen: set[tuple[str, int]] = set()
    uniq: list[dict] = []

    for s in sources:
        if not isinstance(s, dict):
            continue
        d = s.get("doc")
        p = s.get("page")
        if not isinstance(d, str) or not d:
            continue
        if not isinstance(p, int):
            continue

        key = (d, p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append({"doc": d, "page": p})

    return uniq


def confidence_from_sources(num_sources: int) -> str:
    if num_sources <= 0:
        return "low"
    if num_sources <= 2:
        return "medium"
    return "high"


def normalize_result(answer: Any, sources: Any, quotes: Any) -> tuple[str, list[dict], list[dict]]:
    # Answer
    if not isinstance(answer, str) or not answer.strip():
        answer = DONT_KNOW
    answer = answer.strip()

    # Sources
    if not isinstance(sources, list):
        sources = []
    sources = unique_sources(sources)

    # Quotes (keep as-is for now; UI can render it)
    if not isinstance(quotes, list):
        quotes = []
    quotes = [q for q in quotes if isinstance(q, dict)]

    return answer, sources, quotes
