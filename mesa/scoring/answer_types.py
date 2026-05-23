"""Typed v2 answer scorers."""

from mesa.scoring.deterministic import (
    contains_all_required,
    contains_forbidden,
    matches_expected,
    normalize_text,
)
from mesa.scoring.grounding import find_unsupported_claims, is_clean_abstention


def score_single_fact_answer(answer: str, gold_answer: dict, evidence_texts: list[str]) -> dict:
    """Score a single-fact answer using deterministic inclusion checks."""
    must_include = gold_answer.get("must_include", [])
    must_not_include = gold_answer.get("must_not_include", [])

    includes_ok, missing = contains_all_required(answer, must_include)
    forbidden = contains_forbidden(answer, must_not_include)
    correct = matches_expected(answer, gold_answer.get("canonical_answers", []))
    if must_include:
        correct = correct and includes_ok

    unsupported = find_unsupported_claims(answer, evidence_texts)
    grounded = len(unsupported) == 0
    return {
        "correct": correct and not forbidden,
        "grounded": grounded,
        "unsupported_claims": unsupported,
        "missing_required": missing,
        "forbidden_mentions": forbidden,
        "abstention_correct": None,
    }


def score_abstention_answer(answer: str, gold_answer: dict, evidence_texts: list[str]) -> dict:
    """Score an abstention answer, rejecting speculative refusals."""
    clean, details = is_clean_abstention(answer, evidence_texts)
    return {
        "correct": clean and gold_answer.get("abstention_expected") is True,
        "grounded": clean,
        "unsupported_claims": [] if clean else details,
        "missing_required": [],
        "forbidden_mentions": [],
        "abstention_correct": clean,
    }


def score_temporal_answer(answer: str, gold_answer: dict, evidence_texts: list[str]) -> dict:
    """Score a temporal answer via normalized date equivalence plus grounding."""
    includes_ok, missing = contains_all_required(answer, gold_answer.get("must_include", []))
    forbidden = contains_forbidden(answer, gold_answer.get("must_not_include", []))
    correct = matches_expected(answer, gold_answer.get("canonical_answers", []))
    if gold_answer.get("must_include"):
        correct = correct and includes_ok

    unsupported = find_unsupported_claims(answer, evidence_texts)
    grounded = len(unsupported) == 0
    return {
        "correct": correct and not forbidden,
        "grounded": grounded,
        "unsupported_claims": unsupported,
        "missing_required": missing,
        "forbidden_mentions": forbidden,
        "abstention_correct": None,
    }


def score_update_current_answer(answer: str, gold_answer: dict, evidence_texts: list[str]) -> dict:
    """Score an update answer, requiring current facts and rejecting stale ones."""
    includes_ok, missing = contains_all_required(answer, gold_answer.get("must_include", []))
    forbidden = contains_forbidden(answer, gold_answer.get("must_not_include", []))
    correct = matches_expected(answer, gold_answer.get("canonical_answers", []))
    if gold_answer.get("must_include"):
        correct = correct and includes_ok

    unsupported = find_unsupported_claims(answer, evidence_texts)
    grounded = len(unsupported) == 0
    return {
        "correct": correct and not forbidden,
        "grounded": grounded,
        "unsupported_claims": unsupported,
        "missing_required": missing,
        "forbidden_mentions": forbidden,
        "abstention_correct": None,
    }


def score_update_interference_answer(answer: str, gold_answer: dict, evidence_texts: list[str]) -> dict:
    """Score an interference answer, requiring the original fact and rejecting confusers."""
    includes_ok, missing = contains_all_required(answer, gold_answer.get("must_include", []))
    forbidden = contains_forbidden(answer, gold_answer.get("must_not_include", []))
    correct = matches_expected(answer, gold_answer.get("canonical_answers", []))
    if gold_answer.get("must_include"):
        correct = correct and includes_ok

    unsupported = find_unsupported_claims(answer, evidence_texts)
    grounded = len(unsupported) == 0
    return {
        "correct": correct and not forbidden,
        "grounded": grounded,
        "unsupported_claims": unsupported,
        "missing_required": missing,
        "forbidden_mentions": forbidden,
        "abstention_correct": None,
    }


def score_multi_fact_answer(answer: str, gold_answer: dict, evidence_texts: list[str]) -> dict:
    """Score a multi-fact answer by required coverage plus grounding."""
    includes_ok, missing = contains_all_required(answer, gold_answer.get("must_include", []))
    forbidden = contains_forbidden(answer, gold_answer.get("must_not_include", []))
    correct = matches_expected(answer, gold_answer.get("canonical_answers", []))
    if gold_answer.get("must_include"):
        correct = correct or includes_ok

    unsupported = find_unsupported_claims(answer, evidence_texts)
    grounded = len(unsupported) == 0
    return {
        "correct": correct and includes_ok and not forbidden,
        "grounded": grounded,
        "unsupported_claims": unsupported,
        "missing_required": missing,
        "forbidden_mentions": forbidden,
        "abstention_correct": None,
    }


def score_causal_answer(answer: str, gold_answer: dict, evidence_texts: list[str]) -> dict:
    """Score a causal answer by requiring all causal components."""
    return score_multi_fact_answer(answer, gold_answer, evidence_texts)
