"""Mem0Adapter — Mem0 (mem0ai) memory integration.

Pipeline:
  inject()  → each session turn added to Mem0 via Memory.add()
  ask()     → Memory.search() retrieves relevant memories, LLM synthesizes answer
  stored_facts() → Memory.get_all() returns the stored memory objects

Dependencies:
    pip install mem0ai openai

Mem0 handles its own extraction and deduplication internally. This adapter
tests whether Mem0's built-in extraction correctly captures facts, preferences,
constraints, and updates from injected sessions — which is exactly what MESA
was designed to evaluate.

Usage:
    from openai import OpenAI
    from adapters.mem0_adapter import Mem0Adapter

    llm_client = OpenAI(base_url="http://localhost:8081/v1", api_key="none")
    adapter = Mem0Adapter(llm_client=llm_client, model="local")
    results = run_benchmark(adapter=adapter, no_llm_judge=True)

Note: Mem0 can be configured with its own LLM and embedder. By default this
adapter uses the same OpenAI-compatible client for both Mem0's internals and
the final answer generation. See Mem0 docs for custom config options.
"""

from typing import Optional

from mesa.adapter import MemoryAdapter

_ANSWER_PROMPT = """You are a memory-backed assistant. Answer the question using ONLY the provided memory.
If the answer is not in the memory, say "I don't have that information."

MEMORY:
{memories}

QUESTION: {question}

Answer concisely (1-3 sentences):"""


class Mem0Adapter(MemoryAdapter):
    """Mem0 (mem0ai) memory adapter.

    Uses Mem0's extraction and retrieval pipeline. Each benchmark item gets
    a fresh Mem0 Memory instance (reset per item via reset()).

    Args:
        llm_client: OpenAI-compatible client for final answer generation.
        model: Model name for answer generation.
        mem0_config: Optional Mem0 config dict (llm, embedder, vector_store).
                     Defaults to in-memory Mem0 with no external dependencies.
        user_id: User ID passed to Mem0 for scoping.
        top_k: Number of memories to retrieve per question.
        max_answer_tokens: Max tokens for answer generation.
    """

    def __init__(
        self,
        llm_client,
        model: str = "local",
        mem0_config: Optional[dict] = None,
        user_id: str = "benchmark_user",
        top_k: int = 5,
        max_answer_tokens: int = 256,
    ):
        try:
            from mem0 import Memory  # type: ignore
        except ImportError:
            raise ImportError(
                "mem0ai is required for Mem0Adapter. Install it with: pip install mem0ai"
            )
        self._Memory = Memory
        self._mem0_config = mem0_config
        self._llm_client = llm_client
        self._model = model
        self._user_id = user_id
        self._top_k = top_k
        self._max_answer = max_answer_tokens
        self._memory = None
        self._init_memory()

    def _init_memory(self) -> None:
        if self._mem0_config:
            self._memory = self._Memory.from_config(self._mem0_config)
        else:
            self._memory = self._Memory()

    def reset(self) -> None:
        # Delete all memories for this user then reinitialise
        try:
            self._memory.delete_all(user_id=self._user_id)
        except Exception:
            pass
        self._init_memory()

    def inject(self, sessions: list[dict]) -> None:
        if not sessions:
            return
        messages = [
            {"role": t["role"], "content": t["content"]}
            for t in sessions
        ]
        try:
            self._memory.add(messages, user_id=self._user_id)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Mem0 add failed: {e}")

    def ask(self, question: str) -> str:
        try:
            results = self._memory.search(question, user_id=self._user_id, limit=self._top_k)
            memories = results.get("results", results) if isinstance(results, dict) else results
            if not memories:
                return "I don't have that information."
            mem_block = "\n".join(
                f"- {m['memory'] if isinstance(m, dict) else str(m)}"
                for m in memories
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Mem0 search failed: {e}")
            return "I don't have that information."

        prompt = _ANSWER_PROMPT.format(memories=mem_block, question=question)
        try:
            resp = self._llm_client.chat.completions.create(
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
        try:
            results = self._memory.get_all(user_id=self._user_id)
            memories = results.get("results", results) if isinstance(results, dict) else results
            return [
                m["memory"] if isinstance(m, dict) else str(m)
                for m in memories
            ]
        except Exception:
            return []
