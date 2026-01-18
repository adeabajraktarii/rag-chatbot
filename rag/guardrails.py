import re

# Patterns that often indicate prompt-injection attempts or instruction hijacking
INJECTION_PATTERNS = [
    r"ignore (all|previous) instructions",
    r"system prompt",
    r"developer message",
    r"you are chatgpt",
    r"do anything now",
    r"jailbreak",
    r"override",
    r"follow these instructions instead",
    r"repeat the prompt",
    r"reveal.*(policy|rules|instructions|prompt)",
    r"show.*(policy|rules|instructions|prompt)",
]

def looks_like_prompt_injection(text: str) -> bool:
    """
    Lightweight heuristic to detect prompt-injection patterns.
    Used BOTH for:
      - filtering retrieved document chunks
      - rejecting malicious user questions
    """
    t = (text or "").lower()
    return any(re.search(p, t) for p in INJECTION_PATTERNS)

