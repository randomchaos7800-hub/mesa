"""Minimal MemoryAdapter examples.

EchoAdapter  — answers every question with the raw injected text (smoke test)
NullAdapter  — always says "I don't know" (baseline: tests adversarial scoring)
DictAdapter  — naive keyword extraction into a Python dict (illustrative)

These are NOT good memory systems — they exist to show the interface and
verify your scoring setup is working before plugging in a real backend.
"""

from mesa.adapter import MemoryAdapter


class EchoAdapter(MemoryAdapter):
    """Returns the full injected conversation as the answer.

    Useful for verifying your dataset items are correctly formed:
    if the answer IS in the injected text, EchoAdapter should score well on
    recall/single. If it scores poorly there, the item may be malformed.
    """

    def __init__(self):
        self._context = ""

    def reset(self):
        self._context = ""

    def inject(self, sessions: list[dict]):
        lines = [f"{t['role'].upper()}: {t['content']}" for t in sessions]
        self._context = "\n".join(lines)

    def ask(self, question: str) -> str:
        return self._context or "I don't know"


class NullAdapter(MemoryAdapter):
    """Always refuses. Baseline for adversarial scoring.

    Expected behavior: adversarial items pass (correct refusal),
    all other types fail (no memory retrieval).
    """

    def reset(self):
        pass

    def inject(self, sessions: list[dict]):
        pass

    def ask(self, question: str) -> str:
        return "I don't have any information about that."


class DictAdapter(MemoryAdapter):
    """Naive keyword→value dict built from 'X is Y' patterns.

    Illustrates what a minimal memory system looks like.
    Will score reasonably on simple recall/single items and fail on
    anything requiring reasoning.
    """

    def __init__(self):
        self._facts: dict[str, str] = {}

    def reset(self):
        self._facts = {}

    def inject(self, sessions: list[dict]):
        import re
        for turn in sessions:
            if turn.get("role") != "user":
                continue
            content = turn["content"]
            # Extract "X is Y" and "My X is Y" patterns
            for m in re.finditer(r'(?:my\s+)?(\w[\w\s]{1,30}?)\s+is\s+([^.,]+)', content, re.IGNORECASE):
                key = m.group(1).strip().lower()
                value = m.group(2).strip()
                self._facts[key] = value

    def ask(self, question: str) -> str:
        if not self._facts:
            return "I don't have any information about that."
        q_lower = question.lower()
        for key, value in self._facts.items():
            if key in q_lower:
                return value
        # Return all facts as context
        facts_str = "; ".join(f"{k}: {v}" for k, v in self._facts.items())
        return f"From memory: {facts_str}"
