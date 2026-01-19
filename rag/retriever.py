from typing import Optional

import faiss
import numpy as np

from rag.openai_client import embed_text


def _expand_query(query: str) -> str:
    """
    Tiny query expansion to improve retrieval for vague wording.
    This does NOT change grounding, it only helps retrieval find better chunks.
    """
    q = (query or "").strip()
    q_lower = q.lower()

    expansions: list[str] = []

    if (
        "prior authorization" in q_lower
        or "prior auth" in q_lower
        or "utilization management" in q_lower
        or "prior-authorization" in q_lower
    ):
        expansions.extend(
            [
                "delays in care",
                "administrative burden",
                "paperwork",
                "denials",
                "appeals",
                "cost",
                "access",
                "quality",
                "patient outcomes",
            ]
        )

    if (
        "pay for performance" in q_lower
        or "pay-for-performance" in q_lower
        or "p4p" in q_lower
        or "performance-based payment" in q_lower
        or "performance based payment" in q_lower
        or "performance-based reimbursement" in q_lower
    ):
        expansions.extend(
            [
                "financial incentives",
                "provider incentives",
                "quality measures",
                "performance metrics",
                "outcomes",
                "process measures",
                "service delivery",
                "unintended consequences",
                "equity",
                "gaming",
            ]
        )

    if any(w in q_lower for w in ["impact", "impacts", "effect", "effects", "consequence", "consequences"]):
        expansions.extend(["trade-offs", "benefits", "risks", "barriers", "limitations"])

    if expansions:
        seen = set()
        uniq: list[str] = []
        for e in expansions:
            if e not in seen:
                seen.add(e)
                uniq.append(e)
        expansions = uniq

    if not expansions:
        return q

    return q + "\n\nHelpful retrieval hints:\n- " + "\n- ".join(expansions)


def retrieve(
    query: str,
    index: faiss.Index,
    meta: list[dict],
    top_k: int = 5,
    doc_filter: Optional[list[str]] = None,
    year_filter: Optional[int] = None,
    category_filter: Optional[str] = None,
    topic_filter: Optional[list[str]] = None,
) -> list[dict]:
    """
    Vector retrieval with optional metadata filters.

    Fixes vs previous version:
    - topic_filter is "ANY overlap" (not "must contain all")
    - year_filter uses stored metadata year (not filename prefix)
    - oversample is larger so filters don't starve results
    """
    query = _expand_query(query)

    q = np.array([embed_text(query)], dtype="float32")
    faiss.normalize_L2(q)

    oversample = min(len(meta), max(top_k * 30, 200))
    scores, ids = index.search(q, oversample)

    results: list[dict] = []
    topic_set = set(topic_filter) if topic_filter else None

    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(meta):
            continue

        item = meta[idx]

        if doc_filter and item.get("doc") not in doc_filter:
            continue

        if year_filter and item.get("year") != year_filter:
            continue

        if category_filter and item.get("category") != category_filter:
            continue

        if topic_set:
            item_topics = set(item.get("topics", []))
            if item_topics.isdisjoint(topic_set):
                continue

        results.append(
            {
                "score": float(score),
                "doc": item["doc"],
                "page": item["page"],
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "year": item.get("year"),
                "topics": item.get("topics", []),
                "category": item.get("category"),
            }
        )

        if len(results) >= top_k:
            break

    return results
