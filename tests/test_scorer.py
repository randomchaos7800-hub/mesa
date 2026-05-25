"""Unit tests for mesa.scorer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from types import SimpleNamespace
from unittest.mock import patch
from mesa.scorer import (
    exact_match, rouge1_f1, composite, is_refusal, llm_judge,
    _normalize, _normalize_dates,
)

try:
    from rouge_score import rouge_scorer as _rouge_scorer  # noqa: F401
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


class TestNormalize:
    def test_lowercase(self):
        assert _normalize("Hello World") == "hello world"

    def test_strips_markdown(self):
        result = _normalize("**bold** and _italic_")
        assert "bold" in result and "italic" in result
        assert "**" not in result

    def test_collapses_whitespace(self):
        assert _normalize("foo   bar") == "foo bar"

    def test_strips_punctuation(self):
        result = _normalize("hello, world!")
        assert "," not in result and "!" not in result


class TestNormalizeDates:
    def test_month_day_year(self):
        assert "2026-03-01" in _normalize_dates("March 1, 2026")

    def test_abbreviated_month(self):
        assert "2026-03-01" in _normalize_dates("Mar 1 2026")

    def test_slash_format(self):
        assert "2026-03-01" in _normalize_dates("2026/03/01")

    def test_us_slash_format(self):
        assert "2026-03-01" in _normalize_dates("03/01/2026")

    def test_already_iso(self):
        assert "2026-03-01" in _normalize_dates("2026-03-01")

    def test_temporal_equivalence(self):
        assert _normalize("March 1, 2026") == _normalize("2026-03-01")

    def test_temporal_exact_match(self):
        assert exact_match("the user mentioned this on March 1, 2026", "2026-03-01") == 1.0
        assert exact_match("that was 2026-03-01", "March 1, 2026") == 1.0


class TestExactMatch:
    def test_perfect_match(self):
        assert exact_match("homeserver", "homeserver") == 1.0

    def test_substring_match(self):
        assert exact_match("the server is homeserver at 10.0.0.1", "homeserver") == 1.0

    def test_no_match(self):
        assert exact_match("I don't know", "homeserver") == 0.0

    def test_partial_word_overlap(self):
        assert exact_match("supergemma gemma model", "supergemma llama model") >= 0.5

    def test_high_word_overlap(self):
        assert exact_match("supergemma model rtx tower fast", "supergemma model rtx tower local") >= 0.8

    def test_multi_answer_expected(self):
        expected = "7 days. 8 days (also acceptable)."
        assert exact_match("that was 7 days ago", expected) == 1.0

    def test_case_insensitive(self):
        assert exact_match("HOMESERVER", "homeserver") == 1.0

    def test_empty_expected(self):
        assert exact_match("anything", "") == 0.0


@pytest.mark.skipif(not ROUGE_AVAILABLE, reason="rouge-score not installed")
class TestRouge1F1:
    def test_identical(self):
        assert rouge1_f1("hello world foo", "hello world foo") == 1.0

    def test_partial(self):
        assert 0.0 < rouge1_f1("the cat sat on the mat", "the cat sat") < 1.0

    def test_no_overlap(self):
        assert rouge1_f1("completely different text", "nothing matches at all") < 0.3


class TestComposite:
    def test_with_judge(self):
        assert composite(1.0, 1.0, 1.0, use_llm_judge=True) == 1.0

    def test_with_judge_partial(self):
        result = composite(0.8, 0.6, 1.0, use_llm_judge=True)
        assert result == round(0.4 * 0.8 + 0.3 * 1.0 + 0.3 * 0.6, 4)

    def test_without_judge(self):
        assert composite(1.0, 1.0, None, use_llm_judge=False) == 1.0

    def test_without_judge_partial(self):
        result = composite(0.8, 0.6, None, use_llm_judge=False)
        assert result == round(0.55 * 0.8 + 0.45 * 0.6, 4)

    def test_judge_none_fallback(self):
        result = composite(0.8, 0.6, None, use_llm_judge=True)
        assert result == round(0.55 * 0.8 + 0.45 * 0.6, 4)

    def test_zero(self):
        assert composite(0.0, 0.0, 0.0, use_llm_judge=True) == 0.0

    def test_range(self):
        for e in [0.0, 0.5, 1.0]:
            for r in [0.0, 0.5, 1.0]:
                assert 0.0 <= composite(e, r, None, use_llm_judge=False) <= 1.0


class TestIsRefusal:
    def test_explicit(self):
        assert is_refusal("I don't know") == 1.0

    def test_no_record(self):
        assert is_refusal("There's no record of that in my memory.") == 1.0

    def test_hasnt_mentioned(self):
        assert is_refusal("The user hasn't mentioned that to me.") == 1.0

    def test_confident_answer(self):
        assert is_refusal("The user's employee ID is 12345.") == 0.0

    def test_hallucinated_number(self):
        assert is_refusal("The ID number is 9847-XQ.") == 0.0

    def test_case_insensitive(self):
        assert is_refusal("I DON'T KNOW") == 1.0


class TestRougeFallback:
    def test_fallback_identical(self):
        with patch.dict("sys.modules", {"rouge_score": None}):
            assert rouge1_f1("hello world foo", "hello world foo") == 1.0

    def test_fallback_partial(self):
        with patch.dict("sys.modules", {"rouge_score": None}):
            score = rouge1_f1("the cat sat on the mat", "the cat sat")
            assert 0.0 < score < 1.0


class _FakeCompletionClient:
    def __init__(self, content):
        self._content = content
        self.last_kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


class TestLlmJudge:
    def test_parses_json_response(self):
        client = _FakeCompletionClient('{"grade": 2, "reason": "Core fact is right."}')
        result = llm_judge("homeserver", "homeserver", "What is the server?", client)
        assert result == {
            "score": round(2 / 3.0, 4),
            "grade": 2,
            "verdict": "PASS",
            "reason": "Core fact is right.",
        }
        assert client.last_kwargs["temperature"] == 0
        assert client.last_kwargs["response_format"] == {"type": "json_object"}
        assert client.last_kwargs["messages"][0]["role"] == "system"
        assert "<candidate_answer>" in client.last_kwargs["messages"][1]["content"]

    def test_extracts_json_from_fenced_response(self):
        client = _FakeCompletionClient(
            '```json\n{"grade": 1, "reason": "Some overlap."}\n```\nextra text ignored'
        )
        result = llm_judge("foo", "bar", "question", client)
        assert result["grade"] == 1
        assert result["verdict"] == "FAIL"

    def test_returns_error_on_invalid_response(self):
        client = _FakeCompletionClient("not json at all")
        result = llm_judge("foo", "bar", "question", client)
        assert result["score"] is None
        assert result["grade"] is None
        assert result["verdict"] == "ERROR"
