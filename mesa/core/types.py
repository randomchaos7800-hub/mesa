"""Structured trace types for observable benchmark runs."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryWrite:
    """A memory item written during injection."""

    memory_id: str | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedMemory:
    """A memory item surfaced during answer generation."""

    memory_id: str | None
    text: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerTrace:
    """Answer output plus optional retrieval trace."""

    answer: str
    retrieved: list[RetrievedMemory] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkTrace:
    """Full observable trace for one benchmark item."""

    writes: list[MemoryWrite] | None
    answer_trace: AnswerTrace
