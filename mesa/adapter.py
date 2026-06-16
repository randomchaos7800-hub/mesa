"""MemoryAdapter interface — plug your memory system in here."""

from abc import ABC, abstractmethod
from typing import Optional

from mesa.core.types import AnswerTrace, MemoryWrite


class MemoryAdapter(ABC):
    """Abstract base class for plugging a memory system into MESA.

    Implement all three methods to benchmark your system:

        class MyAdapter(MemoryAdapter):
            def reset(self):
                self.memory.clear()

            def inject(self, sessions):
                for turn in sessions:
                    self.memory.store(turn["role"], turn["content"])

            def ask(self, question):
                return self.memory.query(question)

    For multi-session items (facts spread across multiple conversations on
    different dates), the runner calls inject_session() once per session in
    chronological order. The default implementation delegates to inject(),
    ignoring the date. Override inject_session() if your system needs per-session
    metadata (e.g. to tag memories with their conversation date).

    See examples/ for a runnable minimal implementation.
    """

    @abstractmethod
    def reset(self) -> None:
        """Clear all memory state.

        Called before each benchmark item. Every item starts with a clean slate.
        """

    @abstractmethod
    def inject(self, turns: list[dict]) -> None:
        """Inject a single conversation session into memory.

        Args:
            turns: List of {"role": "user"|"assistant", "content": str} dicts.
                   Empty list for adversarial items (nothing should be remembered).

        Called once per benchmark item after reset() (single-session items), or
        once per session in order (multi-session items via inject_session()).
        """

    def inject_session(self, turns: list[dict], session_date: Optional[str] = None) -> None:
        """Inject one session of a multi-session item.

        Args:
            turns: List of {"role": "user"|"assistant", "content": str} dicts.
            session_date: ISO date string ("YYYY-MM-DD") for this session, or None.

        The default implementation ignores the date and calls inject(). Override
        if your system tags memories with conversation dates for temporal reasoning.
        """
        self.inject(turns)

    @abstractmethod
    def ask(self, question: str) -> str:
        """Ask a question and return the memory-informed answer.

        Args:
            question: Natural language question about the injected sessions.

        Returns:
            The system's answer string. For adversarial items (empty sessions),
            a correct system should return a refusal/uncertainty response.
        """

    def get_writes(self) -> list[MemoryWrite] | None:
        """Return structured memory writes for v2 observable runs.

        Override this in adapters that can expose the exact memory units written
        during inject()/inject_session(). The v2 runner prefers this hook over
        stored_facts() because it preserves IDs and metadata.
        """
        return None

    def ask_with_trace(self, question: str) -> AnswerTrace | None:
        """Return the answer plus any retrieval trace for v2 runs.

        The default implementation wraps ask() and exposes no retrieval trace.
        """
        return AnswerTrace(answer=self.ask(question), retrieved=None, metadata={})

    def get_retrieved_context(self, question: str) -> list[str] | None:
        """Return retrieved context strings for debugging and analysis.

        This is a lightweight compatibility/debugging hook for adapters that can
        expose retrieved memory strings without constructing a full trace object.
        """
        trace = self.ask_with_trace(question)
        if trace is None or trace.retrieved is None:
            return None
        return [item.text for item in trace.retrieved]

    def stored_facts(self) -> list[str] | None:
        """Return a list of facts/memories stored after inject().

        Optional — override this to expose diagnostics. The runner will include
        the result in each item's result dict under "stored_facts". Useful for
        diagnosing whether low scores are caused by extraction failure (nothing
        stored) vs retrieval failure (fact stored but not returned).

        Compatibility-only diagnostics hook for legacy adapters. Prefer
        get_writes() for new adapters and official v2 benchmark runs.

        Returns None by default (diagnostics not available).
        """
        return None

    # --- Scope / contamination contract (Improvement #2) ---
    # Adapters declare what "memory" they are allowed to use beyond the current
    # benchmark item's injected sessions. This is critical for real production
    # systems (Mike, etc in full mode) that have persistent state + tools.
    #
    # Valid values:
    #   "pure_injection"   — only facts from the sessions passed to this run's
    #                        inject()/inject_session(). No prior memory, no tools
    #                        that can reach outside the harness.
    #   "full_production"  — the system under test may use its normal persistent
    #                        memory, knowledge graph, tools, lighthouse, etc.
    #                        (default for real companion relays)
    #
    # The runner rejects official runs when scope is not pure.

    scope: str = "full_production"

    def get_scope(self) -> str:
        """Return the declared memory scope for this adapter instance."""
        return getattr(self, "scope", "full_production")
