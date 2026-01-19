from __future__ import annotations

import re
from typing import Optional


BARRIER_TERMS = (
    "adoption",
    "unintended",
    "consequence",
    "consequences",
    "barrier",
    "barriers",
    "challenge",
    "challenges",
    "limitation",
    "limitations",
    "obstacle",
    "obstacles",
    "constraint",
    "constraints",
    "implementation",
    "implementing",
    "adoption",
    "workflow",
    "reimbursement",
    "infrastructure",
    "regulatory",
    "privacy",
    "security",
    "training",
    "licensure",
    "access",
    "connectivity",
)

BARRIERISH_RE = re.compile(
    r"\b(barrier(s)?|challenge(s)?|limitation(s)?|obstacle(s)?|constraint(s)?|implement(ation|ing)?|adoption|unintended consequence(s)?)\b",
    re.IGNORECASE,
)


RETRIEVAL_BOOST = (
    "Focus: challenges, barriers, limitations, obstacles, provider acceptance, workflow, reimbursement, "
    "infrastructure, privacy, security, licensure, regulatory constraints."
)


def is_barrierish(question: str) -> bool:
    return bool(BARRIERISH_RE.search(question or ""))


def _matches_filters(
    item: dict,
    doc_filter: Optional[list[str]],
    year_filter: Optional[int],
    category_filter: Optional[str],
    topic_filter: Optional[list[str]],
) -> bool:
    if doc_filter and item.get("doc") not in doc_filter:
        return False

    if year_filter:
        item_year = item.get("year")
        if item_year is not None:
            if int(item_year) != int(year_filter):
                return False
        else:
            if not item.get("doc", "").startswith(f"{year_filter}_"):
                return False

    if category_filter and item.get("category") != category_filter:
        return False

    if topic_filter:
        item_topics = set(item.get("topics", []) or [])
        if item_topics.isdisjoint(set(topic_filter)):
            return False

    return True


def keyword_fallback_contexts(
    meta: list[dict],
    top_k: int,
    doc_filter: Optional[list[str]] = None,
    year_filter: Optional[int] = None,
    category_filter: Optional[str] = None,
    topic_filter: Optional[list[str]] = None,
) -> list[dict]:
    """
    Lexical (keyword-based) fallback retrieval:
    - scores chunks by number of barrier terms matched
    - applies filters (doc/year/category/topic)
    - returns contexts in the same shape as retriever results
    """
    scored: list[tuple[int, int, dict]] = []

    for item in meta:
        if not _matches_filters(item, doc_filter, year_filter, category_filter, topic_filter):
            continue

        text = (item.get("text") or "").lower()
        if not text:
            continue

        hits = sum(1 for term in BARRIER_TERMS if term in text)
        if hits > 0:
            scored.append((hits, len(text), item))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    take_n = max(top_k, 8)
    results: list[dict] = []

    for hits, _, item in scored[:take_n]:
        results.append(
            {
                "score": float(hits),
                "doc": item["doc"],
                "page": item["page"],
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "year": item.get("year"),
                "topics": item.get("topics", []),
                "category": item.get("category"),
            }
        )

    return results
