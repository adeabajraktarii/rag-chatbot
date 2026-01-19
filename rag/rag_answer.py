from __future__ import annotations

from typing import Optional

import faiss

from rag.barriers import RETRIEVAL_BOOST, is_barrierish, keyword_fallback_contexts
from rag.guardrails import looks_like_prompt_injection
from rag.openai_client import chat
from rag.prompts import DONT_KNOW, build_prompt
from rag.retriever import retrieve
from rag.validators import confidence_from_sources, normalize_result, parse_json_or_none


_TELEMED_TERMS = ("telemedicine", "telehealth", "tele-med", "tele-health", "virtual care", "telecare")
_COMPARE_TERMS = ("compare", "versus", "vs", "overlap", "difference", "differences", "similarities", "similarity")

_TELEMED_BOOST = (
    "Focus: telemedicine/telehealth in primary care; workflow; reimbursement/payment; licensure; regulation; "
    "privacy/security; connectivity/infrastructure; training/provider acceptance; access."
)

_COMPARE_BOOST = (
    "Focus: comparison question. Retrieve evidence for BOTH topics if possible. "
    "If comparing interoperability and telemedicine, retrieve both families."
)


def _is_telemed_q(q: str) -> bool:
    ql = (q or "").lower()
    return any(t in ql for t in _TELEMED_TERMS)


def _is_compare_q(q: str) -> bool:
    ql = (q or "").lower()
    return any(t in ql for t in _COMPARE_TERMS)


def _telemed_context_score(c: dict) -> int:
    doc = (c.get("doc") or "").lower()
    text = (c.get("text") or "").lower()
    topics = set(c.get("topics", []) or [])
    score = 0
    if "telemedicine" in doc or "telehealth" in doc:
        score += 3
    if "telemedicine" in topics:
        score += 3
    if any(t in text for t in ("telemedicine", "telehealth", "tele-med", "tele-health", "virtual care")):
        score += 1
    return score


def _prefer_telemed_contexts(question: str, contexts: list[dict]) -> list[dict]:
    if not _is_telemed_q(question) or not contexts:
        return contexts
    return sorted(contexts, key=_telemed_context_score, reverse=True)


def _drop_injections(contexts: list[dict]) -> list[dict]:
    return [c for c in contexts if not looks_like_prompt_injection(c.get("text", ""))]


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

    # STRICT grounding
    if answer == DONT_KNOW or len(sources) == 0:
        return {"answer": DONT_KNOW, "sources": [], "quotes": [], "confidence": "low"}

    return {
        "answer": answer,
        "sources": sources,
        "quotes": quotes,
        "confidence": confidence_from_sources(len(sources)),
    }


def _build_memory(history: Optional[list[str]]) -> tuple[list[str], str]:
    history = history or []
    last_questions = history[-2:]
    if not last_questions:
        return [], ""
    memory_block = (
        "Conversation memory (previous user questions):\n"
        + "\n".join(f"- {q}" for q in last_questions)
        + "\n\n"
    )
    return last_questions, memory_block


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
    if looks_like_prompt_injection(question):
        return {"answer": DONT_KNOW, "sources": [], "quotes": [], "confidence": "low"}

    last_questions, memory_block = _build_memory(history)

    retrieval_query = (question or "").strip()
    if last_questions:
        retrieval_query += "\n\nPrevious questions:\n" + "\n".join(last_questions)

    barrierish = is_barrierish(question)
    telemed_q = _is_telemed_q(question)
    compare_q = _is_compare_q(question)

    if barrierish:
        retrieval_query += "\n\n" + RETRIEVAL_BOOST
    if telemed_q:
        retrieval_query += "\n\n" + _TELEMED_BOOST
    if compare_q:
        retrieval_query += "\n\n" + _COMPARE_BOOST

    effective_top_k = top_k
    if barrierish:
        effective_top_k = max(effective_top_k, 10)
    if compare_q:
        effective_top_k = max(effective_top_k, 20)

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

    contexts = _prefer_telemed_contexts(question, _drop_injections(contexts))

    # Fallback if nothing retrieved
    if not contexts:
        fallback = keyword_fallback_contexts(
            meta=meta,
            top_k=effective_top_k,
            doc_filter=doc_filter,
            year_filter=year_filter,
            category_filter=category_filter,
            topic_filter=topic_filter,
        )
        contexts = _prefer_telemed_contexts(question, fallback)

    if not contexts:
        return {"answer": DONT_KNOW, "sources": [], "quotes": [], "confidence": "low"}

    # For compare questions: force a grounded 2-part answer + cautious overlap
    if compare_q:
        question = (
            question
            + "\n\nAnswer format required:\n"
            + "1) Interoperability barriers (with sources)\n"
            + "2) Telemedicine barriers (with sources)\n"
            + "3) Overlap (ONLY if overlap is explicitly supported by the provided sources; otherwise say 'Overlap not explicitly supported')\n"
        )

    return _run_llm(question, contexts, memory_block)


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

