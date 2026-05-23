"""Tests for runner._inject dispatch (single-session vs multi-session)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from mesa.core.types import AnswerTrace, MemoryWrite, RetrievedMemory
from mesa.runner import _is_multi_session, _inject
from mesa.runner import run_benchmark_v2
from mesa.adapter import MemoryAdapter


# ---------------------------------------------------------------------------
# Minimal concrete adapter for testing
# ---------------------------------------------------------------------------

class _TrackingAdapter(MemoryAdapter):
    def __init__(self):
        self.inject_calls = []
        self.inject_session_calls = []

    def reset(self):
        self.inject_calls = []
        self.inject_session_calls = []

    def inject(self, turns):
        self.inject_calls.append(turns)

    def inject_session(self, turns, session_date=None):
        self.inject_session_calls.append((turns, session_date))

    def ask(self, question):
        return ""


SINGLE_SESSION = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"},
]

MULTI_SESSION = [
    {"date": "2026-01-01", "turns": [{"role": "user", "content": "Day one fact."}]},
    {"date": "2026-01-07", "turns": [{"role": "user", "content": "Day seven update."}]},
]


# ---------------------------------------------------------------------------
# _is_multi_session
# ---------------------------------------------------------------------------

class TestIsMultiSession:
    def test_empty_is_not_multi(self):
        assert _is_multi_session([]) is False

    def test_turn_list_is_not_multi(self):
        assert _is_multi_session(SINGLE_SESSION) is False

    def test_session_list_is_multi(self):
        assert _is_multi_session(MULTI_SESSION) is True

    def test_single_session_object_is_multi(self):
        # Even a one-element list with a "turns" key is multi-session format
        assert _is_multi_session([{"date": "2026-01-01", "turns": []}]) is True


# ---------------------------------------------------------------------------
# _inject dispatch
# ---------------------------------------------------------------------------

class TestInjectDispatch:
    def test_single_session_calls_inject(self):
        adapter = _TrackingAdapter()
        _inject(adapter, SINGLE_SESSION)
        assert len(adapter.inject_calls) == 1
        assert adapter.inject_calls[0] == SINGLE_SESSION
        assert adapter.inject_session_calls == []

    def test_multi_session_calls_inject_session(self):
        adapter = _TrackingAdapter()
        _inject(adapter, MULTI_SESSION)
        assert adapter.inject_calls == []
        assert len(adapter.inject_session_calls) == 2

    def test_multi_session_passes_date(self):
        adapter = _TrackingAdapter()
        _inject(adapter, MULTI_SESSION)
        dates = [c[1] for c in adapter.inject_session_calls]
        assert dates == ["2026-01-01", "2026-01-07"]

    def test_multi_session_passes_turns(self):
        adapter = _TrackingAdapter()
        _inject(adapter, MULTI_SESSION)
        turns0 = adapter.inject_session_calls[0][0]
        assert turns0 == MULTI_SESSION[0]["turns"]

    def test_empty_sessions_calls_inject(self):
        adapter = _TrackingAdapter()
        _inject(adapter, [])
        assert len(adapter.inject_calls) == 1
        assert adapter.inject_calls[0] == []


# ---------------------------------------------------------------------------
# Default inject_session falls back to inject
# ---------------------------------------------------------------------------

class _DefaultAdapter(MemoryAdapter):
    """Does not override inject_session — tests the default delegation."""
    def __init__(self):
        self.inject_calls = []

    def reset(self):
        self.inject_calls = []

    def inject(self, turns):
        self.inject_calls.append(turns)

    def ask(self, question):
        return ""


class TestDefaultInjectSession:
    def test_default_inject_session_delegates_to_inject(self):
        adapter = _DefaultAdapter()
        turns = [{"role": "user", "content": "hello"}]
        adapter.inject_session(turns, session_date="2026-01-01")
        assert adapter.inject_calls == [turns]

    def test_multi_session_via_default_inject_session(self):
        adapter = _DefaultAdapter()
        _inject(adapter, MULTI_SESSION)
        assert len(adapter.inject_calls) == 2
        assert adapter.inject_calls[0] == MULTI_SESSION[0]["turns"]
        assert adapter.inject_calls[1] == MULTI_SESSION[1]["turns"]


class _TraceAdapter(MemoryAdapter):
    def __init__(self):
        self.injected = []

    def reset(self):
        self.injected = []

    def inject(self, turns):
        self.injected.extend(turns)

    def ask(self, question):
        return "fallback answer"

    def get_writes(self):
        return [MemoryWrite(memory_id="w1", text="stored fact")]

    def ask_with_trace(self, question):
        return AnswerTrace(
            answer="trace answer",
            retrieved=[RetrievedMemory(memory_id="r1", text="retrieved fact", score=0.9)],
            metadata={"path": "trace"},
        )


class _CorrectSingleFactAdapter(MemoryAdapter):
    def reset(self):
        pass

    def inject(self, turns):
        pass

    def get_writes(self):
        return [
            MemoryWrite(memory_id="f1", text="The user's primary home server is homebase."),
            MemoryWrite(memory_id="f2", text="The VPN IP of homebase is 10.0.0.10."),
        ]

    def ask(self, question):
        return "homebase, 10.0.0.10"

    def ask_with_trace(self, question):
        return AnswerTrace(
            answer="homebase, 10.0.0.10",
            retrieved=[
                RetrievedMemory(memory_id="f1", text="The user's primary home server is homebase."),
                RetrievedMemory(memory_id="f2", text="The VPN IP of homebase is 10.0.0.10."),
            ],
            metadata={},
        )


class _SpeculativeAbstentionAdapter(MemoryAdapter):
    def reset(self):
        pass

    def inject(self, turns):
        pass

    def ask(self, question):
        return "I don't know, but maybe it's 12345."

    def ask_with_trace(self, question):
        return AnswerTrace(answer=self.ask(question), retrieved=[], metadata={})


class _TemporalAdapter(MemoryAdapter):
    def reset(self):
        pass

    def inject(self, turns):
        pass

    def get_writes(self):
        return [MemoryWrite(memory_id="f1", text="The migration started on 2026-03-01.")]

    def ask(self, question):
        return "March 1, 2026"

    def ask_with_trace(self, question):
        return AnswerTrace(
            answer="March 1, 2026",
            retrieved=[RetrievedMemory(memory_id="f1", text="The migration started on 2026-03-01.")],
            metadata={},
        )


class _StaleUpdateAdapter(MemoryAdapter):
    def reset(self):
        pass

    def inject(self, turns):
        pass

    def get_writes(self):
        return [
            MemoryWrite(memory_id="f1", text="The assistant was using cloud-api for inference."),
            MemoryWrite(memory_id="f2", text="The assistant is currently using local-llm (Gemma-4 26B) for inference."),
        ]

    def ask(self, question):
        return "cloud-api"

    def ask_with_trace(self, question):
        return AnswerTrace(
            answer="cloud-api",
            retrieved=[RetrievedMemory(memory_id="f1", text="The assistant was using cloud-api for inference.")],
            metadata={},
        )


class _NoisyRetrievalAdapter(MemoryAdapter):
    def reset(self):
        pass

    def inject(self, turns):
        pass

    def get_writes(self):
        return [
            MemoryWrite(memory_id="f1", text="The user's primary home server is homebase."),
            MemoryWrite(memory_id="f2", text="The VPN IP of homebase is 10.0.0.10."),
            MemoryWrite(memory_id="junk", text="The coffee mug is blue."),
        ]

    def ask(self, question):
        return "homebase, 10.0.0.10"

    def ask_with_trace(self, question):
        return AnswerTrace(
            answer="homebase, 10.0.0.10",
            retrieved=[
                RetrievedMemory(memory_id="f1", text="The user's primary home server is homebase."),
                RetrievedMemory(memory_id="f2", text="The VPN IP of homebase is 10.0.0.10."),
                RetrievedMemory(memory_id="junk", text="The coffee mug is blue."),
            ],
            metadata={},
        )


class _InterferenceConfuserAdapter(MemoryAdapter):
    def reset(self):
        pass

    def inject(self, turns):
        pass

    def get_writes(self):
        return [
            MemoryWrite(memory_id="f1", text="The primary server serial number is XJ-99-B."),
            MemoryWrite(memory_id="f2", text="The backup drive serial number is XJ-88-A."),
            MemoryWrite(memory_id="f3", text="The external array serial number is XJ-77-C."),
        ]

    def ask(self, question):
        return "XJ-88-A"

    def ask_with_trace(self, question):
        return AnswerTrace(
            answer="XJ-88-A",
            retrieved=[RetrievedMemory(memory_id="f2", text="The backup drive serial number is XJ-88-A.")],
            metadata={},
        )


class _MultiFactPartialAdapter(MemoryAdapter):
    def reset(self):
        pass

    def inject(self, turns):
        pass

    def get_writes(self):
        return [
            MemoryWrite(memory_id="f1", text="The user is running Gemma-4 26B."),
            MemoryWrite(memory_id="f2", text="The model is running at 70 tokens per second."),
        ]

    def ask(self, question):
        return "Gemma-4 26B"

    def ask_with_trace(self, question):
        return AnswerTrace(
            answer="Gemma-4 26B",
            retrieved=[
                RetrievedMemory(memory_id="f1", text="The user is running Gemma-4 26B."),
                RetrievedMemory(memory_id="f2", text="The model is running at 70 tokens per second."),
            ],
            metadata={},
        )


class _CausalCompleteAdapter(MemoryAdapter):
    def reset(self):
        pass

    def inject(self, turns):
        pass

    def get_writes(self):
        return [
            MemoryWrite(memory_id="f1", text="The backup script triggers when disk usage exceeds 80%."),
            MemoryWrite(memory_id="f2", text="Disk usage is currently 85%."),
        ]

    def ask(self, question):
        return "Because disk usage is at 85%, which exceeds the 80% threshold that triggers the backup script."

    def ask_with_trace(self, question):
        return AnswerTrace(
            answer=self.ask(question),
            retrieved=[
                RetrievedMemory(memory_id="f1", text="The backup script triggers when disk usage exceeds 80%."),
                RetrievedMemory(memory_id="f2", text="Disk usage is currently 85%."),
            ],
            metadata={},
        )


class _PreferenceAdapter(MemoryAdapter):
    def reset(self):
        pass

    def inject(self, turns):
        pass

    def get_writes(self):
        return [MemoryWrite(memory_id="f1", text="The user prefers terse responses.")]

    def ask(self, question):
        return "terse"

    def ask_with_trace(self, question):
        return AnswerTrace(
            answer="terse",
            retrieved=[RetrievedMemory(memory_id="f1", text="The user prefers terse responses.")],
            metadata={},
        )


class _ConstraintDistractorAdapter(MemoryAdapter):
    def reset(self):
        pass

    def inject(self, turns):
        pass

    def get_writes(self):
        return [
            MemoryWrite(memory_id="f1", text="The cloud storage budget cap is $20/month with no exceptions."),
            MemoryWrite(memory_id="f2", text="S3 costs $35/month for 2TB."),
        ]

    def ask(self, question):
        return "$35/month"

    def ask_with_trace(self, question):
        return AnswerTrace(
            answer="$35/month",
            retrieved=[RetrievedMemory(memory_id="f2", text="S3 costs $35/month for 2TB.")],
            metadata={},
        )


class _LegacyOnlyAdapter(MemoryAdapter):
    def __init__(self):
        self.injected = []

    def reset(self):
        self.injected = []

    def inject(self, turns):
        self.injected.extend(turns)

    def ask(self, question):
        return "legacy answer"

    def stored_facts(self):
        return ["legacy fact"]


class _OpaqueAdapter(MemoryAdapter):
    def reset(self):
        pass

    def inject(self, turns):
        pass

    def ask(self, question):
        return "opaque answer"

    def ask_with_trace(self, question):
        return None

    def get_writes(self):
        return None

    def stored_facts(self):
        return None


class TestRunBenchmarkV2:
    DATASET_V2 = Path(__file__).parent.parent / "dataset" / "fixtures" / "sample_v2.json"

    def test_prefers_trace_hooks(self):
        adapter = _TraceAdapter()
        summary = run_benchmark_v2(
            adapter=adapter,
            dataset_path=self.DATASET_V2,
            quiet=True,
            limit=1,
        )
        result = summary["results"][0]
        assert result["observable"] is True
        assert result["storage"]["writes"][0]["text"] == "stored fact"
        assert result["retrieval"]["retrieved"][0]["text"] == "retrieved fact"
        assert result["answer"]["text"] == "trace answer"

    def test_falls_back_to_legacy_hooks(self):
        adapter = _LegacyOnlyAdapter()
        summary = run_benchmark_v2(
            adapter=adapter,
            dataset_path=self.DATASET_V2,
            quiet=True,
            limit=1,
        )
        result = summary["results"][0]
        assert result["observable"] is True
        assert result["storage"]["writes"][0]["metadata"]["source"] == "stored_facts"
        assert result["retrieval"]["retrieved"] == []
        assert result["answer"]["text"] == "legacy answer"

    def test_marks_non_observable_adapter(self):
        adapter = _OpaqueAdapter()
        summary = run_benchmark_v2(
            adapter=adapter,
            dataset_path=self.DATASET_V2,
            quiet=True,
            limit=1,
        )
        assert summary["results"][0]["observable"] is False

    def test_trace_required_rejects_opaque_adapter(self):
        adapter = _OpaqueAdapter()
        with pytest.raises(ValueError, match="does not expose trace hooks required"):
            run_benchmark_v2(
                adapter=adapter,
                dataset_path=self.DATASET_V2,
                quiet=True,
                limit=1,
                trace_required=True,
            )

    def test_scores_single_fact_answer(self):
        adapter = _CorrectSingleFactAdapter()
        summary = run_benchmark_v2(
            adapter=adapter,
            dataset_path=self.DATASET_V2,
            quiet=True,
            limit=1,
        )
        result = summary["results"][0]
        assert result["storage"]["metrics"]["required_fact_recall"] == 1.0
        assert result["retrieval"]["metrics"]["required_fact_recall"] == 1.0
        assert result["answer"]["metrics"]["correct"] is True
        assert result["answer"]["metrics"]["grounded"] is True
        assert result["failures"] == []

    def test_rejects_speculative_abstention(self):
        adapter = _SpeculativeAbstentionAdapter()
        summary = run_benchmark_v2(
            adapter=adapter,
            dataset_path=self.DATASET_V2,
            quiet=True,
            limit=2,
        )
        result = summary["results"][1]
        assert result["answer"]["metrics"]["correct"] is False
        assert result["answer"]["metrics"]["abstention_correct"] is False
        assert "unclean_abstention" in result["failures"]

    def test_scores_temporal_answer(self):
        adapter = _TemporalAdapter()
        summary = run_benchmark_v2(
            adapter=adapter,
            dataset_path=self.DATASET_V2,
            quiet=True,
            limit=3,
        )
        result = summary["results"][2]
        assert result["answer"]["metrics"]["correct"] is True
        assert result["answer"]["metrics"]["grounded"] is True

    def test_rejects_stale_update_answer(self):
        adapter = _StaleUpdateAdapter()
        summary = run_benchmark_v2(
            adapter=adapter,
            dataset_path=self.DATASET_V2,
            quiet=True,
            limit=4,
        )
        result = summary["results"][3]
        assert result["answer"]["metrics"]["correct"] is False
        assert "incorrect_answer" in result["failures"]
        assert "retrieved_forbidden_fact" in result["failures"]

    def test_penalizes_extra_storage_and_retrieval(self):
        adapter = _NoisyRetrievalAdapter()
        summary = run_benchmark_v2(
            adapter=adapter,
            dataset_path=self.DATASET_V2,
            quiet=True,
            limit=1,
        )
        result = summary["results"][0]
        assert result["storage"]["metrics"]["required_fact_precision"] == 0.6667
        assert result["retrieval"]["metrics"]["required_fact_precision"] == 0.6667
        assert result["storage"]["metrics"]["unannotated_write_count"] == 1
        assert result["retrieval"]["metrics"]["unannotated_retrieval_count"] == 1
        assert "stored_extra_fact" in result["failures"]
        assert "retrieved_extra_fact" in result["failures"]

    def test_rejects_interference_confuser(self):
        adapter = _InterferenceConfuserAdapter()
        summary = run_benchmark_v2(
            adapter=adapter,
            dataset_path=self.DATASET_V2,
            quiet=True,
            limit=5,
        )
        result = summary["results"][4]
        assert result["answer"]["metrics"]["correct"] is False
        assert "incorrect_answer" in result["failures"]
        assert "retrieved_forbidden_fact" in result["failures"]

    def test_rejects_partial_multi_fact_answer(self):
        adapter = _MultiFactPartialAdapter()
        summary = run_benchmark_v2(
            adapter=adapter,
            dataset_path=self.DATASET_V2,
            quiet=True,
            limit=6,
        )
        result = summary["results"][5]
        assert result["answer"]["metrics"]["correct"] is False
        assert "incorrect_answer" in result["failures"]

    def test_accepts_complete_causal_answer(self):
        adapter = _CausalCompleteAdapter()
        summary = run_benchmark_v2(
            adapter=adapter,
            dataset_path=self.DATASET_V2,
            quiet=True,
            limit=7,
        )
        result = summary["results"][6]
        assert result["answer"]["metrics"]["correct"] is True
        assert result["answer"]["metrics"]["grounded"] is True

    def test_accepts_preference_answer(self):
        adapter = _PreferenceAdapter()
        summary = run_benchmark_v2(
            adapter=adapter,
            dataset_path=self.DATASET_V2,
            quiet=True,
            limit=8,
        )
        result = summary["results"][7]
        assert result["answer"]["metrics"]["correct"] is True

    def test_rejects_constraint_distractor(self):
        adapter = _ConstraintDistractorAdapter()
        summary = run_benchmark_v2(
            adapter=adapter,
            dataset_path=self.DATASET_V2,
            quiet=True,
            limit=9,
        )
        result = summary["results"][8]
        assert result["answer"]["metrics"]["correct"] is False
        assert "incorrect_answer" in result["failures"]
        assert "retrieved_forbidden_fact" in result["failures"]

    def test_summary_is_populated(self):
        adapter = _CorrectSingleFactAdapter()
        summary = run_benchmark_v2(
            adapter=adapter,
            dataset_path=self.DATASET_V2,
            quiet=True,
            limit=1,
        )
        assert summary["summary"]["answer"]["correct_rate"] == 1.0
        assert "recall/single" in summary["summary"]["by_type"]
