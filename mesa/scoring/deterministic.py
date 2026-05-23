"""Deterministic normalization and fact-matching helpers for v2 scoring."""

import re

_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "jun": "06", "jul": "07", "aug": "08", "sep": "09", "oct": "10",
    "nov": "11", "dec": "12",
}


def normalize_dates(text: str) -> str:
    """Convert common date formats to ISO-like strings for comparison."""
    text = re.sub(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december"
        r"|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b\.?\s+(\d{1,2}),?\s+(\d{4})",
        lambda m: f"{m.group(3)}-{_MONTH_MAP[m.group(1).lower()]}-{int(m.group(2)):02d}",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(\d{4})/(\d{2})/(\d{2})\b", r"\1-\2-\3", text)
    text = re.sub(
        r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b",
        lambda m: f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}",
        text,
    )
    return text


def normalize_text(text: str) -> str:
    """Lowercase, normalize dates, strip punctuation, and collapse whitespace."""
    text = normalize_dates(text)
    text = re.sub(r"[*_`#>\[\]()~]", "", text)
    text = re.sub(r"[^\w\s.-]", " ", text)
    return re.sub(r"\s+", " ", text).lower().strip()


def matches_expected(answer: str, expected_values: list[str]) -> bool:
    """Return True if any expected value is contained in the normalized answer."""
    norm_answer = normalize_text(answer)
    if not norm_answer:
        return False
    for value in expected_values:
        norm_value = normalize_text(value)
        if norm_value and norm_value in norm_answer:
            return True
    return False


def contains_all_required(answer: str, required_values: list[str]) -> tuple[bool, list[str]]:
    """Return whether all required normalized substrings are present."""
    normalized_answer = normalize_text(answer)
    missing = []
    for value in required_values:
        normalized_value = normalize_text(value)
        if normalized_value and normalized_value not in normalized_answer:
            missing.append(value)
    return len(missing) == 0, missing


def contains_forbidden(answer: str, forbidden_values: list[str]) -> list[str]:
    """Return normalized forbidden substrings present in the answer."""
    normalized_answer = normalize_text(answer)
    present = []
    for value in forbidden_values:
        normalized_value = normalize_text(value)
        if normalized_value and normalized_value in normalized_answer:
            present.append(value)
    return present


def match_fact_ids(texts: list[str], atomic_facts: list[dict]) -> set[str]:
    """Return gold fact IDs matched by the provided texts."""
    matched = set()
    normalized_texts = [normalize_text(text) for text in texts]
    for fact in atomic_facts:
        candidates = [fact.get("text", "")]
        candidates.extend(fact.get("aliases", []))
        slots = fact.get("slots", {})
        candidates.extend(str(value) for value in slots.values())
        normalized_candidates = [normalize_text(candidate) for candidate in candidates if candidate]
        for fact_text in normalized_candidates:
            if fact_text and any(fact_text in text or text in fact_text for text in normalized_texts if text):
                matched.add(fact["fact_id"])
                break
    return matched
