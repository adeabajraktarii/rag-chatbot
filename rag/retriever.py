from typing import Optional
import faiss
import numpy as np

from rag.openai_client import embed_text


def _expand_query(query: str) -> str:
    """
    Tiny query expansion to improve retrieval for vague words like 'impacts'.
    This does NOT change grounding, it only helps retrieval find better chunks.
    """
    q = (query or "").strip()
    q_lower = q.lower()

    expansions = []

    # Prior authorization / utilization management
    if "prior authorization" in q_lower or "prior auth" in q_lower or "pa " in q_lower:
        expansions.append("delays in care")
        expansions.append("administrative burden")
        expansions.append("cost")
        expansions.append("access")
        expansions.append("quality")

    # Generic 'impact' wording
    if "impact" in q_lower or "impacts" in q_lower or "effect" in q_lower or "effects" in q_lower:
        expansions.append("consequences")
        expansions.append("barriers")
        expansions.append("limitations")

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
    # ✅ Expand query slightly to improve embedding-based search
    query = _expand_query(query)

    q = np.array([embed_text(query)], dtype="float32")
    faiss.normalize_L2(q)

    oversample = min(len(meta), max(top_k * 10, 50))
    scores, ids = index.search(q, oversample)

    results: list[dict] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(meta):
            continue

        item = meta[idx]

        if doc_filter and item.get("doc") not in doc_filter:
            continue
        if year_filter and not item.get("doc", "").startswith(f"{year_filter}_"):
            continue
        if category_filter and item.get("category") != category_filter:
            continue
        if topic_filter:
            item_topics = set(item.get("topics", []))
            if not set(topic_filter).issubset(item_topics):
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

