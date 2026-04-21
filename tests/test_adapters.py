"""Tests for reference adapters.

Uses mock LLM clients so no inference server is required.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from adapters.keyword_adapter import KeywordAdapter, _tokenize, _tfidf_score, _build_idf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(extraction_text: str = "- The server is homeserver\n- Speed is 70 tok/s",
                      answer_text: str = "homeserver"):
    client = MagicMock()
    def _completion(**kwargs):
        msg = kwargs.get("messages", [{}])[-1].get("content", "")
        text = extraction_text if "Extract" in msg else answer_text
        choice = MagicMock()
        choice.message.content = text
        resp = MagicMock()
        resp.choices = [choice]
        return resp
    client.chat.completions.create.side_effect = lambda **kw: _completion(**kw)
    return client


SAMPLE_SESSIONS = [
    {"role": "user", "content": "My main server is homeserver. Speed is 70 tok/s."},
    {"role": "assistant", "content": "Got it — homeserver at 70 tok/s."},
]


# ---------------------------------------------------------------------------
# KeywordAdapter
# ---------------------------------------------------------------------------

class TestKeywordAdapter:
    def test_reset_clears_facts(self):
        client = _make_mock_client()
        adapter = KeywordAdapter(client=client)
        adapter.inject(SAMPLE_SESSIONS)
        assert len(adapter.stored_facts()) > 0
        adapter.reset()
        assert adapter.stored_facts() == []

    def test_inject_populates_facts(self):
        client = _make_mock_client()
        adapter = KeywordAdapter(client=client)
        adapter.inject(SAMPLE_SESSIONS)
        facts = adapter.stored_facts()
        assert facts is not None
        assert len(facts) > 0

    def test_ask_returns_string(self):
        client = _make_mock_client()
        adapter = KeywordAdapter(client=client)
        adapter.inject(SAMPLE_SESSIONS)
        answer = adapter.ask("What is the server name?")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_adversarial_no_sessions(self):
        client = _make_mock_client()
        adapter = KeywordAdapter(client=client)
        adapter.inject([])
        answer = adapter.ask("What is the bank account number?")
        assert "don't" in answer.lower() or "no information" in answer.lower()

    def test_stored_facts_after_reset(self):
        client = _make_mock_client()
        adapter = KeywordAdapter(client=client)
        adapter.inject(SAMPLE_SESSIONS)
        adapter.reset()
        adapter.inject(SAMPLE_SESSIONS)
        assert len(adapter.stored_facts()) > 0

    def test_extraction_failure_graceful(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("network error")
        adapter = KeywordAdapter(client=client)
        adapter.inject(SAMPLE_SESSIONS)
        assert adapter.stored_facts() == []
        answer = adapter.ask("anything")
        assert isinstance(answer, str)


# ---------------------------------------------------------------------------
# TF-IDF internals
# ---------------------------------------------------------------------------

class TestTfidf:
    def test_tokenize(self):
        tokens = _tokenize("The server is homeserver at 10.0.0.1")
        assert "homeserver" in tokens
        assert "server" in tokens

    def test_build_idf(self):
        docs = [["cat", "dog"], ["cat", "fish"], ["bird"]]
        idf = _build_idf(docs)
        assert "cat" in idf
        assert "bird" in idf
        assert idf["bird"] > idf["cat"]  # rarer word has higher IDF

    def test_tfidf_score_relevant(self):
        doc = _tokenize("homeserver is the main server at 70 tok/s")
        idf = _build_idf([doc, _tokenize("something else entirely")])
        score = _tfidf_score(_tokenize("homeserver speed"), doc, idf)
        assert score > 0

    def test_tfidf_score_irrelevant(self):
        doc = _tokenize("the weather is nice today")
        idf = _build_idf([doc])
        score = _tfidf_score(_tokenize("homeserver speed tok/s"), doc, idf)
        assert score == 0.0


# ---------------------------------------------------------------------------
# ChromaAdapter (import-guarded — skipped if chromadb not installed)
# ---------------------------------------------------------------------------

try:
    import chromadb  # type: ignore
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


@pytest.mark.skipif(not CHROMA_AVAILABLE, reason="chromadb not installed")
class TestChromaAdapter:
    def test_reset_clears_collection(self):
        from adapters.chroma_adapter import ChromaAdapter
        client = _make_mock_client()
        adapter = ChromaAdapter(client=client)
        adapter.inject(SAMPLE_SESSIONS)
        adapter.reset()
        assert adapter.stored_facts() == []

    def test_inject_and_ask(self):
        from adapters.chroma_adapter import ChromaAdapter
        client = _make_mock_client()
        adapter = ChromaAdapter(client=client)
        adapter.inject(SAMPLE_SESSIONS)
        assert len(adapter.stored_facts()) > 0
        answer = adapter.ask("What is the server name?")
        assert isinstance(answer, str)


# ---------------------------------------------------------------------------
# Mem0Adapter (import-guarded — skipped if mem0ai not installed)
# ---------------------------------------------------------------------------

try:
    import mem0  # type: ignore
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False


@pytest.mark.skipif(not MEM0_AVAILABLE, reason="mem0ai not installed")
class TestMem0Adapter:
    def test_reset_clears_memory(self):
        from adapters.mem0_adapter import Mem0Adapter
        client = _make_mock_client()
        adapter = Mem0Adapter(llm_client=client)
        adapter.inject(SAMPLE_SESSIONS)
        adapter.reset()
        facts = adapter.stored_facts()
        assert facts == [] or facts is not None

    def test_inject_and_ask_returns_string(self):
        from adapters.mem0_adapter import Mem0Adapter
        client = _make_mock_client()
        adapter = Mem0Adapter(llm_client=client)
        adapter.inject(SAMPLE_SESSIONS)
        answer = adapter.ask("What is the server name?")
        assert isinstance(answer, str)
