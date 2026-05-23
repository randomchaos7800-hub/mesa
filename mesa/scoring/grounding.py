"""Grounding checks used by v2 answer scoring."""

import re

from mesa.scoring.deterministic import normalize_text

_REFUSAL_PATTERNS = [
    r"don.t know",
    r"do not know",
    r"no record",
    r"no information",
    r"not mentioned",
    r"not provided",
    r"not in my memory",
    r"i don.t have",
    r"cannot find",
    r"can.t find",
    r"unable to find",
]


def extract_claim_tokens(text: str) -> set[str]:
    """Extract concrete answer tokens that should usually be grounded."""
    normalized = normalize_text(text)
    tokens = set()
    tokens.update(re.findall(r"\b\d{1,10}(?:\.\d{1,4}){0,3}\b", normalized))
    tokens.update(re.findall(r"\b[a-z]+-[a-z0-9.-]+\b", normalized))
    tokens.update(re.findall(r"\b[a-z]{3,}\b", normalized))
    stop_words = {
        "about", "account", "any", "bank", "but", "don", "dont", "have", "information",
        "its", "know", "maybe", "number", "probably", "provided", "shared", "that",
        "the", "there", "was", "because", "which", "threshold",
    }
    return {token for token in tokens if token not in stop_words}


def find_unsupported_claims(answer: str, evidence_texts: list[str]) -> list[str]:
    """Return concrete answer tokens that do not appear in the evidence."""
    if not answer.strip():
        return []
    evidence_norm = normalize_text(" ".join(evidence_texts))
    unsupported = []
    for token in sorted(extract_claim_tokens(answer)):
        if len(token) <= 2:
            continue
        if token not in evidence_norm:
            unsupported.append(token)
    return unsupported


def is_refusal_answer(text: str) -> bool:
    """Return True if the answer contains a refusal phrase."""
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in _REFUSAL_PATTERNS)


def is_clean_abstention(answer: str, evidence_texts: list[str]) -> tuple[bool, list[str]]:
    """Return True only if the answer abstains without unsupported speculation."""
    if not is_refusal_answer(answer):
        return False, ["missing_refusal_phrase"]
    unsupported = find_unsupported_claims(answer, evidence_texts)
    return len(unsupported) == 0, unsupported
