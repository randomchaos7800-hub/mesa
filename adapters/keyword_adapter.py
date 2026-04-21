"""KeywordAdapter — LLM extraction + TF-IDF keyword retrieval.

Pipeline:
  inject()  → LLM extracts discrete facts as bullet points from sessions
  ask()     → TF-IDF scores stored facts against the question, top-k fed to LLM
  stored_facts() → return extracted bullets for diagnostics

Dependencies: openai (already a MESA dep). No vector DB required.

This is the simplest non-trivial adapter — useful as a reference baseline
and for systems that store memory as plain text (markdown files, SQLite text,
etc.). Swap out _retrieve() for your own retrieval logic.

Usage:
    from openai import OpenAI
    from adapters.keyword_adapter import KeywordAdapter

    client = OpenAI(base_url="http://localhost:8081/v1", api_key="none")
    adapter = KeywordAdapter(client=client, model="local")
    results = run_benchmark(adapter=adapter, no_llm_judge=True)
"""

import re
from collections import Counter
from math import log
from typing import Optional

from mesa.adapter import MemoryAdapter

_EXTRACT_PROMPT = """Extract all discrete facts from this conversation as a bullet-point list.
Each bullet should be one self-contained fact (entity, value, preference, rule, or event).
Be specific. Do not summarize. Do not invent facts not stated.

CONVERSATION:
{conversation}

Return only the bullet list, one fact per line starting with "- "."""

_ANSWER_PROMPT = """You are a memory-backed assistant. Answer the question using ONLY the provided memory facts.
If the answer is not in the facts, say "I don't have that information."

MEMORY FACTS:
{facts}

QUESTION: {question}

Answer concisely (1-3 sentences):"""


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b[a-z0-9_\-]+\b', text.lower())


def _tfidf_score(query_tokens: list[str], doc_tokens: list[str], idf: dict[str, float]) -> float:
    tf = Counter(doc_tokens)
    total = len(doc_tokens) or 1
    return sum(idf.get(t, 0) * (tf[t] / total) for t in query_tokens if t in tf)


def _build_idf(docs: list[list[str]]) -> dict[str, float]:
    n = len(docs) or 1
    df: Counter = Counter()
    for doc in docs:
        df.update(set(doc))
    return {term: log((n + 1) / (count + 1)) + 1 for term, count in df.items()}


class KeywordAdapter(MemoryAdapter):
    """LLM extraction + TF-IDF retrieval.

    Extracts facts from injected sessions using a local LLM, stores them as
    plain strings, then retrieves the most relevant ones via TF-IDF cosine
    similarity to answer questions.

    Args:
        client: OpenAI-compatible client instance.
        model: Model name to use for extraction and answering.
        top_k: Number of facts to retrieve per question.
        max_extract_tokens: Max tokens for extraction call.
        max_answer_tokens: Max tokens for answer call.
    """

    def __init__(
        self,
        client,
        model: str = "local",
        top_k: int = 5,
        max_extract_tokens: int = 512,
        max_answer_tokens: int = 256,
    ):
        self._client = client
        self._model = model
        self._top_k = top_k
        self._max_extract = max_extract_tokens
        self._max_answer = max_answer_tokens
        self._facts: list[str] = []

    def reset(self) -> None:
        self._facts = []

    def inject(self, sessions: list[dict]) -> None:
        if not sessions:
            return
        conversation = "\n\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in sessions
        )
        prompt = _EXTRACT_PROMPT.format(conversation=conversation)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_extract,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content or ""
            for line in raw.splitlines():
                line = line.strip().lstrip("-• ").strip()
                if len(line) > 10:
                    self._facts.append(line)
        except Exception as e:
            # Extraction failure — facts stay empty, ask() will refuse
            import logging
            logging.getLogger(__name__).warning(f"Extraction failed: {e}")

    def ask(self, question: str) -> str:
        if not self._facts:
            return "I don't have any information about that."

        # TF-IDF retrieval
        tokenized_facts = [_tokenize(f) for f in self._facts]
        idf = _build_idf(tokenized_facts)
        q_tokens = _tokenize(question)
        scores = [
            (_tfidf_score(q_tokens, doc, idf), fact)
            for doc, fact in zip(tokenized_facts, self._facts)
        ]
        scores.sort(reverse=True)
        top_facts = [fact for _, fact in scores[: self._top_k] if _ > 0]

        if not top_facts:
            top_facts = self._facts[: self._top_k]

        facts_block = "\n".join(f"- {f}" for f in top_facts)
        prompt = _ANSWER_PROMPT.format(facts=facts_block, question=question)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_answer,
                messages=[{"role": "user", "content": prompt}],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Answer generation failed: {e}")
            return "I don't have that information."

    def stored_facts(self) -> list[str] | None:
        return list(self._facts)
