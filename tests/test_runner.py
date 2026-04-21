"""Tests for runner._inject dispatch (single-session vs multi-session)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from mesa.runner import _is_multi_session, _inject
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
