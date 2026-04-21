"""MMEB scoring functions.

Three scoring dimensions:
  exact_match   — normalized substring/word-overlap (0.0–1.0)
  rouge1_f1     — ROUGE-1 F1 score (0.0–1.0)
  llm_judge     — local model binary verdict (0.0 or 1.0)

Composite formula:
  With LLM judge:    0.4 * exact + 0.3 * llm_judge + 0.3 * rouge1
  Without LLM judge: 0.55 * exact + 0.45 * rouge1

Special handling:
  adversarial — scored via is_refusal() instead of exact match;
                without LLM judge, composite = refusal score directly
  temporal    — date formats normalized before comparison
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "jun": "06", "jul": "07", "aug": "08", "sep": "09", "oct": "10",
    "nov": "11", "dec": "12",
}


def _normalize_dates(text: str) -> str:
    """Convert common date formats to ISO 8601 (YYYY-MM-DD) for comparison.

    Handles:
      "March 1, 2026" → "2026-03-01"
      "Mar 1 2026"    → "2026-03-01"
      "2026/03/01"    → "2026-03-01"
      "03/01/2026"    → "2026-03-01"
    """
    # "Month D, YYYY" or "Month D YYYY"
    text = re.sub(
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december'
        r'|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b\.?\s+(\d{1,2}),?\s+(\d{4})',
        lambda m: f"{m.group(3)}-{_MONTH_MAP[m.group(1).lower()]}-{int(m.group(2)):02d}",
        text,
        flags=re.IGNORECASE,
    )
    # YYYY/MM/DD → YYYY-MM-DD
    text = re.sub(r'\b(\d{4})/(\d{2})/(\d{2})\b', r'\1-\2-\3', text)
    # MM/DD/YYYY → YYYY-MM-DD
    text = re.sub(
        r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',
        lambda m: f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}",
        text,
    )
    return text


def _normalize(text: str) -> str:
    """Lowercase, normalize dates, strip markdown, collapse whitespace, remove punctuation."""
    text = _normalize_dates(text)
    text = re.sub(r'\*+|_+|`+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).lower().strip()


# ---------------------------------------------------------------------------
# Adversarial refusal detection
# ---------------------------------------------------------------------------

_REFUSAL_PATTERNS = [
    r"don.t know",
    r"do not know",
    r"no record",
    r"not mentioned",
    r"haven.t mentioned",
    r"hasn.t mentioned",
    r"no information",
    r"not in my memory",
    r"not stored",
    r"not aware",
    r"can.t find",
    r"cannot find",
    r"unable to find",
    r"no data",
    r"not provided",
    r"never (said|told|mentioned|shared)",
    r"i.m not sure",
    r"i don.t have",
    r"there.s no",
    r"there is no",
]


def is_refusal(text: str) -> float:
    """Return 1.0 if the text is a refusal/uncertainty response, 0.0 if it asserts a fact."""
    lower = text.lower()
    for pat in _REFUSAL_PATTERNS:
        if re.search(pat, lower):
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------

def _score_one(pred_norm: str, gold_norm: str) -> float:
    """Score normalized pred against a single normalized gold string."""
    if not gold_norm:
        return 0.0
    if gold_norm in pred_norm:
        return 1.0
    gold_words = set(gold_norm.split())
    pred_words = set(pred_norm.split())
    if not gold_words:
        return 0.0
    overlap = len(gold_words & pred_words) / len(gold_words)
    if overlap >= 0.8:
        return 0.8
    if overlap >= 0.5:
        return 0.5
    return 0.0


def exact_match(predicted: str, expected: str) -> float:
    """Exact/substring/word-overlap score (0.0–1.0).

    Splits expected on sentence boundaries so multi-answer strings like
    "7 days. 8 days (also acceptable)." score as a pass if either is correct.
    """
    pred = _normalize(predicted)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', expected) if s.strip()]
    if not sentences:
        return 0.0
    best = max(_score_one(pred, _normalize(s)) for s in sentences)
    return max(best, _score_one(pred, _normalize(expected)))


# ---------------------------------------------------------------------------
# ROUGE-1 F1
# ---------------------------------------------------------------------------

def rouge1_f1(predicted: str, expected: str) -> float:
    """ROUGE-1 F1 using the rouge-score library. Falls back to 0.0 on import error."""
    try:
        from rouge_score import rouge_scorer  # type: ignore
    except ImportError:
        logger.warning("rouge-score not installed — ROUGE-1 scores will be 0.0")
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    result = scorer.score(expected, predicted)
    return round(result["rouge1"].fmeasure, 4)


# ---------------------------------------------------------------------------
# LLM judge (local tower)
# ---------------------------------------------------------------------------

LLM_JUDGE_PROMPT = """You are a strict but fair memory evaluator.

Question: {question}
Expected answer: {expected}
AI answer: {predicted}

Does the AI answer correctly address the question based on the expected answer?
Answer YES if the key information is present (exact wording not required).
Answer NO if the answer is wrong, hallucinated, or refuses without cause.

Respond with JSON only: {{"verdict": "YES" or "NO", "reason": "<one sentence>"}}"""


def llm_judge(
    predicted: str,
    expected: str,
    question: str,
    client,
    model: str = "local",
) -> dict:
    """LLM-as-judge using the local inference tower.

    Returns {"score": float, "verdict": str, "reason": str}.
    score is 1.0 for YES, 0.0 for NO, None on error.
    """
    prompt = LLM_JUDGE_PROMPT.format(
        question=question,
        expected=expected,
        predicted=predicted,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        text = text.strip()
        text = re.sub(r",\s*([}\]])", r"\1", text)
        data = json.loads(text)
        verdict = str(data.get("verdict", "NO")).strip().upper()
        score = 1.0 if verdict == "YES" else 0.0
        return {"score": score, "verdict": verdict, "reason": data.get("reason", "")}
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return {"score": None, "verdict": "ERROR", "reason": str(e)}


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def composite(
    exact: float,
    rouge: float,
    judge: Optional[float],
    use_llm_judge: bool = True,
) -> float:
    """Weighted composite score.

    With judge:    0.4 * exact + 0.3 * judge + 0.3 * rouge
    Without judge: 0.55 * exact + 0.45 * rouge
    """
    if use_llm_judge and judge is not None:
        return round(0.4 * exact + 0.3 * judge + 0.3 * rouge, 4)
    return round(0.55 * exact + 0.45 * rouge, 4)
