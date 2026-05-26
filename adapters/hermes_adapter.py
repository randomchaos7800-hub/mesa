"""HermesAdapter — benchmarks Hermes (AIAgent) via conversation_history injection.

Pipeline:
  reset()   → clears stored turns
  inject()  → stores turns as OpenAI-format message dicts
  ask()     → creates a fresh AIAgent pointed at the local tower proxy,
              passes stored turns as conversation_history, calls run_conversation()

This tests Hermes' ability to read a conversation history and answer questions
from it — the "long-context + agent reasoning" mode, not active write/recall
memory. Write/recall memory testing requires a running Hermes gateway instance.

Usage:
    from adapters.hermes_adapter import HermesAdapter
    from mesa.runner import run_benchmark_v2
    results = run_benchmark_v2(adapter=HermesAdapter(), quiet=False)

Requirements:
    - /home/dino/hermes must be on sys.path (added automatically below)
    - Tower proxy reachable at http://100.120.50.35:8010/v1
    - Hermes venv at /home/dino/hermes/venv
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Hermes repo must be importable
HERMES_ROOT = Path.home() / "hermes"
if str(HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_ROOT))

from mesa.adapter import MemoryAdapter
from mesa.core.types import AnswerTrace, MemoryWrite, RetrievedMemory

logger = logging.getLogger(__name__)

LOCAL_BASE_URL = "http://100.120.50.35:8010/v1"
LOCAL_MODEL = "local"


class HermesAdapter(MemoryAdapter):
    """Hermes AIAgent driven via conversation_history injection.

    Each ask() creates a fresh, ephemeral AIAgent with all toolsets disabled.
    Stored turns are passed as conversation_history so the agent has full
    context. No writes to Hermes' state DB.
    """

    def __init__(
        self,
        base_url: str = LOCAL_BASE_URL,
        model: str = LOCAL_MODEL,
        max_iterations: int = 3,
    ):
        self._base_url = base_url
        self._model = model
        self._max_iterations = max_iterations
        self._turns: list[dict] = []

    def reset(self) -> None:
        self._turns = []

    def inject(self, sessions: list[dict]) -> None:
        """Store conversation turns as OpenAI-format message dicts."""
        self._turns = []
        for turn in sessions:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if content:
                self._turns.append({"role": role, "content": content})

    def ask(self, question: str) -> str:
        try:
            from run_agent import AIAgent
        except ImportError as e:
            logger.error(f"Cannot import Hermes AIAgent: {e}")
            return "I don't have that information."

        try:
            agent = AIAgent(
                base_url=self._base_url,
                api_key="local",          # must be set with base_url to skip provider router
                model=self._model,
                max_iterations=self._max_iterations,
                # Disable all toolsets — pure LLM reasoning, no tool calls
                enabled_toolsets=[],
                save_trajectories=False,
                quiet_mode=True,
                verbose_logging=False,
            )
            result = agent.run_conversation(
                user_message=question,
                conversation_history=list(self._turns) if self._turns else None,
            )
            return result.get("final_response", "").strip() or "I don't have that information."
        except Exception as e:
            logger.warning(f"HermesAdapter.ask() failed: {e}")
            return "I don't have that information."

    def get_writes(self) -> list[MemoryWrite] | None:
        # No write trace — Hermes doesn't expose explicit write events in this mode
        return None

    def ask_with_trace(self, question: str) -> AnswerTrace | None:
        answer = self.ask(question)
        # Treat the injected context as the "retrieved" chunk for grounding
        retrieved = []
        if self._turns:
            context = "\n".join(
                f"{t['role'].upper()}: {t['content']}" for t in self._turns
            )
            retrieved.append(RetrievedMemory(memory_id="hermes-context", text=context))
        return AnswerTrace(answer=answer, retrieved=retrieved, metadata={"adapter": "HermesAdapter"})

    def stored_facts(self) -> list[str] | None:
        return [t["content"] for t in self._turns if t.get("role") == "user"]
