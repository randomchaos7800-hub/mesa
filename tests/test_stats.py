"""Tests for MESA statistical reporting helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mesa.scoring.stats import bootstrap_mean_ci, mean


class TestStatsHelpers:
    def test_mean_handles_booleans(self):
        assert mean([True, False, True]) == 2 / 3

    def test_mean_skips_none(self):
        assert mean([1.0, None, 0.0]) == 0.5

    def test_bootstrap_ci_returns_bounds(self):
        ci = bootstrap_mean_ci([True, False, True, True], iterations=100, seed=1)
        assert ci is not None
        assert 0.0 <= ci["lower"] <= ci["upper"] <= 1.0
        assert ci["n"] == 4

    def test_bootstrap_singleton(self):
        ci = bootstrap_mean_ci([True], iterations=50, seed=1)
        assert ci == {
            "mean": 1.0,
            "lower": 1.0,
            "upper": 1.0,
            "n": 1,
            "iterations": 50,
            "confidence": 0.95,
        }
