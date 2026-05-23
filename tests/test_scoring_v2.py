"""Tests for v2 deterministic scoring and grounding behavior."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mesa.scoring.answer_types import (
    score_abstention_answer,
    score_causal_answer,
    score_multi_fact_answer,
    score_single_fact_answer,
    score_temporal_answer,
    score_update_interference_answer,
    score_update_current_answer,
)
from mesa.scoring.deterministic import match_fact_ids, normalize_dates
from mesa.scoring.grounding import find_unsupported_claims, is_clean_abstention


class TestDeterministicV2:
    def test_normalize_dates_equivalent(self):
        assert normalize_dates("March 1, 2026") == "2026-03-01"

    def test_match_fact_ids_from_alias_and_slot_value(self):
        facts = [
            {
                "fact_id": "f1",
                "text": "The VPN IP of homebase is 10.0.0.10.",
                "aliases": ["VPN IP is 10.0.0.10"],
                "slots": {"value": "10.0.0.10"},
            }
        ]
        matched = match_fact_ids(["The VPN IP is 10.0.0.10 on homebase"], facts)
        assert matched == {"f1"}


class TestGroundingV2:
    def test_find_unsupported_claims_detects_extra_number(self):
        unsupported = find_unsupported_claims(
            "I don't know, but maybe the number is 12345.",
            ["There is no bank account number in memory."],
        )
        assert "12345" in unsupported

    def test_clean_abstention_rejects_speculation(self):
        clean, details = is_clean_abstention(
            "I don't know, but it's probably 12345.",
            ["No bank account number was shared."],
        )
        assert clean is False
        assert "12345" in details


class TestAnswerTypeScoringV2:
    def test_single_fact_correct_and_grounded(self):
        result = score_single_fact_answer(
            "homebase, 10.0.0.10",
            {
                "canonical_answers": ["homebase, 10.0.0.10"],
                "must_include": ["homebase", "10.0.0.10"],
                "must_not_include": [],
            },
            ["homebase 10.0.0.10"],
        )
        assert result["correct"] is True
        assert result["grounded"] is True

    def test_single_fact_fails_unsupported_claim(self):
        result = score_single_fact_answer(
            "homebase, 10.0.0.10, serial 9999",
            {
                "canonical_answers": ["homebase, 10.0.0.10"],
                "must_include": ["homebase", "10.0.0.10"],
                "must_not_include": [],
            },
            ["homebase 10.0.0.10"],
        )
        assert result["grounded"] is False
        assert "9999" in result["unsupported_claims"]

    def test_abstention_accepts_clean_refusal(self):
        result = score_abstention_answer(
            "I don't know.",
            {
                "canonical_answers": ["I don't know"],
                "must_include": [],
                "must_not_include": [],
                "abstention_expected": True,
            },
            ["No account number was provided."],
        )
        assert result["correct"] is True
        assert result["abstention_correct"] is True

    def test_temporal_normalizes_equivalent_dates(self):
        result = score_temporal_answer(
            "March 1, 2026",
            {
                "canonical_answers": ["2026-03-01"],
                "must_include": ["2026-03-01"],
                "must_not_include": [],
            },
            ["The migration started on 2026-03-01."],
        )
        assert result["correct"] is True
        assert result["grounded"] is True

    def test_update_current_rejects_stale_fact(self):
        result = score_update_current_answer(
            "cloud-api",
            {
                "canonical_answers": ["local-llm"],
                "must_include": ["local-llm"],
                "must_not_include": ["cloud-api"],
            },
            ["Current model is local-llm. Previous model was cloud-api."],
        )
        assert result["correct"] is False
        assert "cloud-api" in result["forbidden_mentions"]

    def test_update_interference_rejects_confuser(self):
        result = score_update_interference_answer(
            "XJ-88-A",
            {
                "canonical_answers": ["XJ-99-B"],
                "must_include": ["XJ-99-B"],
                "must_not_include": ["XJ-88-A", "XJ-77-C"],
            },
            ["Primary server serial number is XJ-99-B.", "Backup drive serial number is XJ-88-A."],
        )
        assert result["correct"] is False
        assert "XJ-88-A" in result["forbidden_mentions"]

    def test_multi_fact_requires_all_components(self):
        result = score_multi_fact_answer(
            "Gemma-4 26B",
            {
                "canonical_answers": ["Gemma-4 26B, 70 tokens per second"],
                "must_include": ["Gemma-4 26B", "70 tokens per second"],
                "must_not_include": [],
            },
            ["Gemma-4 26B", "70 tokens per second"],
        )
        assert result["correct"] is False
        assert "70 tokens per second" in result["missing_required"]

    def test_causal_requires_all_components(self):
        result = score_causal_answer(
            "Because disk usage is at 85%.",
            {
                "canonical_answers": ["Because disk usage is at 85%, which exceeds the 80% threshold."],
                "must_include": ["85%", "80%"],
                "must_not_include": [],
            },
            ["threshold is 80%", "disk usage is 85%"],
        )
        assert result["correct"] is False
        assert "80%" in " ".join(result["missing_required"])
