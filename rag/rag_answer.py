from __future__ import annotations

from typing import Optional

import faiss

from rag.barriers import BARRIER_TERMS, RETRIEVAL_BOOST, is_barrierish
from rag.guardrails import looks_like_prompt_injection
from rag.openai_client import chat
from rag.prompts import DONT_KNOW, build_prompt
from rag.retriever import retrieve
from rag.validators import confidence_from_sources, normalize_result, parse_json_or_none


_TELEMED_ANCHOR_TERMS = ("telemedicine", "telehealth", "tele-med", "tele-health", "virtual care", "telecare")


def _question_is_telemedicine_related(question: str) -> bool:
    q = (question or "").lower()
    return any(t in q for t in _TELEMED_ANCHOR_TERMS)


def _context_is_telemedicine_related(c: dict) -> bool:
    # Prefer metadata topic match if available
    topics = set((c.get("topics") or []))
    if "telemedicine" in topics:
        return True

    # Fallback to text match
    text = (c.get("text") or "").lower()
    return ("telemedicine" in text) or ("telehealth" in text) or ("tele-med" in text) or ("tele-health" in text)


def _apply_telemedicine_anchor(question: str, contexts: list[dict]) -> list[dict]:
    """
    If user question is clearly about telemedicine/telehealth,
    keep only telemedicine-related contexts so unrelated 'limitations' don't hijack.
    """
    if not _question_is_telemedicine_related(question):
        return contexts
    return [c for c in contexts if _context_is_telemedicine_related(c)]


def _matches_filters(
    item: dict,
    doc_filter: Optional[list[str]],
    year_filter: Optional[int],
    category_filter: Optional[str],
    topic_filter: Optional[list[str]],
) -> bool:
    if doc_filter and item.get("doc") not in doc_filter:
        return False
    if year_filter and not item.get("doc", "").startswith(f"{year_filter}_"):
        return False
    if category_filter and item.get("category") != category_filter:
        return False
    if topic_filter:
        item_topics = set(item.get("topics", []))
        # Current behavior = must include ALL selected topics
        if not set(topic_filter).issubset(item_topics):
            return False
    return True


def _make_context(item: dict, score: float) -> dict:
    return {
        "score": float(score),  # debug-only score (keyword hits)
        "doc": item["doc"],
        "page": item["page"],
        "chunk_id": item["chunk_id"],
        "text": item["text"],
        "year": item.get("year"),
        "topics": item.get("topics", []),
        "category": item.get("category"),
    }


def _keyword_fallback_contexts(
    meta: list[dict],
    top_k: int,
    doc_filter: Optional[list[str]],
    year_filter: Optional[int],
    category_filter: Optional[str],
    topic_filter: Optional[list[str]],
) -> list[dict]:
    scored: list[tuple[int, dict]] = []

    for item in meta:
        if not _matches_filters(item, doc_filter, year_filter, category_filter, topic_filter):
            continue

        text = (item.get("text") or "").lower()
        if not text:
            continue

        hits = sum(1 for term in BARRIER_TERMS if term in text)
        if hits:
            scored.append((hits, item))

    scored.sort(key=lambda x: (x[0], len(x[1].get("text", ""))), reverse=True)

    take_n = max(top_k, 8)
    return [_make_context(item, hits) for hits, item in scored[:take_n]]


def _drop_injections(contexts: list[dict]) -> list[dict]:
    safe: list[dict] = []
    for c in contexts:
        if looks_like_prompt_injection(c.get("text", "")):
            continue
        safe.append(c)
    return safe


def _run_llm(question: str, contexts: list[dict], memory_block: str) -> dict:
    prompt = build_prompt(memory_block + question, contexts)
    raw = chat(prompt).strip()

    data = parse_json_or_none(raw)
    if data is None:
        return {"answer": DONT_KNOW, "sources": [], "quotes": [], "confidence": "low"}

    answer, sources, quotes = normalize_result(
        data.get("answer"),
        data.get("sources", []),
        data.get("quotes", []),
    )

    # STRICT grounding: answer must have sources (unless DONT_KNOW)
    if answer != DONT_KNOW and len(sources) == 0:
        return {"answer": DONT_KNOW, "sources": [], "quotes": [], "confidence": "low"}

    if answer == DONT_KNOW:
        return {"answer": DONT_KNOW, "sources": [], "quotes": [], "confidence": "low"}

    return {
        "answer": answer,
        "sources": sources,
        "quotes": quotes,
        "confidence": confidence_from_sources(len(sources)),
    }


def answer_question_structured(
    question: str,
    index: faiss.Index,
    meta: list[dict],
    top_k: int = 5,
    history: Optional[list[str]] = None,
    doc_filter: Optional[list[str]] = None,
    year_filter: Optional[int] = None,
    category_filter: Optional[str] = None,
    topic_filter: Optional[list[str]] = None,
) -> dict:
    """
    Returns:
      {
        "answer": str,
        "sources": [{"doc": str, "page": int}],
        "quotes": [{"quote": str, "source_index": int}],
        "confidence": "high|medium|low"
      }
    """

    # ✅ NEW: Block malicious user prompt-injection attempts early
    if looks_like_prompt_injection(question):
        return {"answer": DONT_KNOW, "sources": [], "quotes": [], "confidence": "low"}

    history = history or []
    last_questions = history[-2:]

    retrieval_query = question
    if last_questions:
        retrieval_query += "\n\nPrevious questions:\n" + "\n".join(last_questions)

    barrierish = is_barrierish(question)
    if barrierish:
        retrieval_query += "\n\n" + RETRIEVAL_BOOST

    # Pull more context for barrier-ish questions
    effective_top_k = max(top_k, 10) if barrierish else top_k

    contexts = retrieve(
        retrieval_query,
        index=index,
        meta=meta,
        top_k=effective_top_k,
        doc_filter=doc_filter,
        year_filter=year_filter,
        category_filter=category_filter,
        topic_filter=topic_filter,
    )

    safe_contexts = _drop_injections(contexts)

    # ✅ NEW: Apply telemedicine anchor (prevents unrelated limitation answers)
    safe_contexts = _apply_telemedicine_anchor(question, safe_contexts)

    # If FAISS retrieval yields nothing safe, try lexical fallback
    if not safe_contexts:
        fallback = _keyword_fallback_contexts(
            meta=meta,
            top_k=effective_top_k,
            doc_filter=doc_filter,
            year_filter=year_filter,
            category_filter=category_filter,
            topic_filter=topic_filter,
        )
        fallback = _apply_telemedicine_anchor(question, fallback)

        if not fallback:
            return {"answer": DONT_KNOW, "sources": [], "quotes": [], "confidence": "low"}

        safe_contexts = fallback

    memory_block = ""
    if last_questions:
        memory_block = (
            "Conversation memory (previous user questions):\n"
            + "\n".join(f"- {q}" for q in last_questions)
            + "\n\n"
        )

    # Pass 1
    result = _run_llm(question, safe_contexts, memory_block)

    # If "don't know" on barriers-type question, retry once with keyword fallback
    if result["answer"] == DONT_KNOW and barrierish:
        fallback = _keyword_fallback_contexts(
            meta=meta,
            top_k=effective_top_k,
            doc_filter=doc_filter,
            year_filter=year_filter,
            category_filter=category_filter,
            topic_filter=topic_filter,
        )
        fallback = _apply_telemedicine_anchor(question, fallback)

        if fallback:
            result = _run_llm(question, fallback, memory_block)

    return result


def answer_question(
    question: str,
    index: faiss.Index,
    meta: list[dict],
    top_k: int = 5,
    history: Optional[list[str]] = None,
    doc_filter: Optional[list[str]] = None,
    year_filter: Optional[int] = None,
    category_filter: Optional[str] = None,
    topic_filter: Optional[list[str]] = None,
) -> str:
    return answer_question_structured(
        question=question,
        index=index,
        meta=meta,
        top_k=top_k,
        history=history,
        doc_filter=doc_filter,
        year_filter=year_filter,
        category_filter=category_filter,
        topic_filter=topic_filter,
    )["answer"]

