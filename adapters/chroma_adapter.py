"""ChromaAdapter — ChromaDB vector store + LLM answering.

Pipeline:
  inject()  → LLM extracts facts, embeds them via Chroma's default embedder,
               stores in an in-memory ChromaDB collection
  ask()     → embeds question, retrieves top-k nearest facts, LLM synthesizes
  stored_facts() → returns all stored fact strings for diagnostics

Dependencies:
    pip install chromadb openai

Chroma uses its own default embedding function (all-MiniLM-L6-v2 via
sentence-transformers) unless you override it. No API key required for
embeddings when using the default local model.

Usage:
    from openai import OpenAI
    from adapters.chroma_adapter import ChromaAdapter

    client = OpenAI(base_url="http://localhost:8081/v1", api_key="none")
    adapter = ChromaAdapter(client=client, model="local")
    results = run_benchmark(adapter=adapter, no_llm_judge=True)

Custom embedding function (e.g. OpenAI embeddings):
    import chromadb.utils.embedding_functions as ef
    emb_fn = ef.OpenAIEmbeddingFunction(api_key="...", model_name="text-embedding-3-small")
    adapter = ChromaAdapter(client=client, model="local", embedding_fn=emb_fn)
"""

import uuid
import re
from typing import Optional

from mesa.adapter import MemoryAdapter

_EXTRACT_PROMPT = """Extract all discrete facts from this conversation as a bullet-point list.
Each bullet should be one self-contained fact. Be specific. Do not invent.

CONVERSATION:
{conversation}

Return only the bullet list, one fact per line starting with "- "."""

_ANSWER_PROMPT = """Answer the question using ONLY the provided facts.
If the answer is not present, say "I don't have that information."

FACTS:
{facts}

QUESTION: {question}

Answer concisely (1-3 sentences):"""


class ChromaAdapter(MemoryAdapter):
    """ChromaDB vector store memory adapter.

    Extracts facts from sessions using a local LLM, stores them in an
    in-memory ChromaDB collection, and retrieves the most semantically
    relevant facts to answer questions.

    Args:
        client: OpenAI-compatible client for extraction and answering.
        model: Model name for LLM calls.
        embedding_fn: Chroma embedding function. Defaults to Chroma's built-in
                      (all-MiniLM-L6-v2). Pass a custom function for OpenAI
                      or other embedding providers.
        top_k: Number of facts to retrieve per question.
        max_extract_tokens: Max tokens for fact extraction.
        max_answer_tokens: Max tokens for answer generation.
    """

    def __init__(
        self,
        client,
        model: str = "local",
        embedding_fn=None,
        top_k: int = 5,
        max_extract_tokens: int = 512,
        max_answer_tokens: int = 256,
    ):
        try:
            import chromadb  # type: ignore
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaAdapter. Install it with: pip install chromadb"
            )
        self._chromadb = chromadb
        self._client = client
        self._model = model
        self._embedding_fn = embedding_fn
        self._top_k = top_k
        self._max_extract = max_extract_tokens
        self._max_answer = max_answer_tokens
        self._chroma_client = None
        self._collection = None
        self._facts: list[str] = []
        self._init_collection()

    def _init_collection(self) -> None:
        self._chroma_client = self._chromadb.Client()
        kwargs = {}
        if self._embedding_fn:
            kwargs["embedding_function"] = self._embedding_fn
        self._collection = self._chroma_client.create_collection(
            name=f"mesa_{uuid.uuid4().hex[:8]}",
            **kwargs,
        )
        self._facts = []

    def reset(self) -> None:
        # Drop the old collection and create a fresh one
        try:
            self._chroma_client.delete_collection(self._collection.name)
        except Exception:
            pass
        self._init_collection()

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
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Extraction failed: {e}")
            return

        facts = []
        for line in raw.splitlines():
            line = line.strip().lstrip("-• ").strip()
            if len(line) > 10:
                facts.append(line)

        if not facts:
            return

        self._facts = facts
        self._collection.add(
            documents=facts,
            ids=[f"fact_{i}" for i in range(len(facts))],
        )

    def ask(self, question: str) -> str:
        if not self._facts:
            return "I don't have any information about that."

        try:
            n = min(self._top_k, len(self._facts))
            results = self._collection.query(
                query_texts=[question],
                n_results=n,
            )
            retrieved = results["documents"][0] if results["documents"] else []
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Chroma query failed: {e}")
            retrieved = self._facts[: self._top_k]

        if not retrieved:
            return "I don't have any information about that."

        facts_block = "\n".join(f"- {f}" for f in retrieved)
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
