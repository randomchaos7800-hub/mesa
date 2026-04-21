"""MemoryAdapter interface — plug your memory system in here."""

from abc import ABC, abstractmethod
from typing import Optional


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

    def stored_facts(self) -> list[str] | None:
        """Return a list of facts/memories stored after inject().

        Optional — override this to expose diagnostics. The runner will include
        the result in each item's result dict under "stored_facts". Useful for
        diagnosing whether low scores are caused by extraction failure (nothing
        stored) vs retrieval failure (fact stored but not returned).

        Returns None by default (diagnostics not available).
        """
        return None
