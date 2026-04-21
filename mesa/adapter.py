"""MemoryAdapter interface — plug your memory system in here."""

from abc import ABC, abstractmethod


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

    See examples/ for a runnable minimal implementation.
    """

    @abstractmethod
    def reset(self) -> None:
        """Clear all memory state.

        Called before each benchmark item. Every item starts with a clean slate.
        """

    @abstractmethod
    def inject(self, sessions: list[dict]) -> None:
        """Inject conversation sessions into memory.

        Args:
            sessions: List of {"role": "user"|"assistant", "content": str} dicts.
                      Empty list for adversarial items (nothing should be remembered).

        Called once per benchmark item after reset() and before ask().
        """

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
