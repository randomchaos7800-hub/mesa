"""Tests for the standalone semantic equivalence scorer.

mesa/semantic_scorer.py has no callers elsewhere in the package (it's not
wired into any adapter or runner path) and previously carried only an
ad-hoc `if __name__ == '__main__':` self-test that pytest never executes.
These cases are ported from that self-test so the scoring logic actually
gets regression coverage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from mesa.semantic_scorer import semantic_equivalence


class TestSemanticEquivalence:
    @pytest.mark.parametrize(
        "expected,actual,min_score",
        [
            ("The server is homeserver", "homeserver is the server", 0.7),
            ("The server is homeserver", "The server is homeserver", 1.0),
            (
                "Ibuprofen was prescribed for dehydration headaches",
                "They were prescribed ibuprofen for the headache",
                0.55,
            ),
            (
                "Kernel update failed causing outage",
                "The kernel update failed and took the server down",
                0.55,
            ),
            ("$30/month", "thirty dollars a month", 0.3),
            ("homeserver", "I don't know", 0.0),
        ],
    )
    def test_meets_minimum_score(self, expected, actual, min_score):
        score = semantic_equivalence(expected, actual)
        assert score >= min_score - 0.05

    def test_exact_match_scores_one(self):
        assert semantic_equivalence("homeserver", "homeserver") == 1.0

    def test_empty_expected_scores_zero(self):
        assert semantic_equivalence("", "anything") == 0.0

    def test_empty_actual_scores_zero(self):
        assert semantic_equivalence("anything", "") == 0.0

    def test_score_bounded_between_zero_and_one(self):
        score = semantic_equivalence("The kernel panic happened at 3am", "totally unrelated text")
        assert 0.0 <= score <= 1.0
