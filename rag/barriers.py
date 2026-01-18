# rag/barriers.py
from __future__ import annotations

import re

BARRIER_TERMS = (
    "barrier", "barriers",
    "challenge", "challenges",
    "limitation", "limitations",
    "obstacle", "obstacles",
    "constraint", "constraints",
    "implementation", "implementing",
    "adoption",
    "workflow",
    "reimbursement",
    "infrastructure",
    "regulatory",
    "privacy", "security",
    "training",
    "licensure",
    "access", "connectivity",
)

BARRIERISH_RE = re.compile(
    r"\b(barrier(s)?|challenge(s)?|limitation(s)?|obstacle(s)?|constraint(s)?|implement(ation|ing)?|adoption)\b",
    re.IGNORECASE,
)

RETRIEVAL_BOOST = (
    "Focus: challenges, barriers, limitations, obstacles, provider acceptance, workflow, reimbursement, "
    "infrastructure, privacy, security, licensure, regulatory constraints."
)


def is_barrierish(question: str) -> bool:
    return bool(BARRIERISH_RE.search(question))
