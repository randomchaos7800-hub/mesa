"""MESA — Memory Eval Suite for Agents.

Benchmark tooling for personal AI memory systems.
Supports a legacy composite scorer (`run_benchmark`) and an observable
diagnostic path (`run_benchmark_v2`) for storage/retrieval/answer metrics.
"""

from mesa.adapter import MemoryAdapter
from mesa.runner import run_benchmark, run_benchmark_v2
from mesa.scorer import exact_match, rouge1_f1, composite, is_refusal

__version__ = "0.3.2"
__all__ = [
    "MemoryAdapter",
    "run_benchmark",
    "run_benchmark_v2",
    "exact_match",
    "rouge1_f1",
    "composite",
    "is_refusal",
]
