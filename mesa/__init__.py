"""MESA — Memory Eval Suite for Agents.

Benchmark tooling for personal AI memory systems.
Supports a legacy composite scorer (`run_benchmark`) and an observable
diagnostic path (`run_benchmark_v2`) for storage/retrieval/answer metrics.
"""

from mesa.adapter import MemoryAdapter
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


def run_benchmark(*args, **kwargs):
    """Lazily import the legacy runner to avoid package import side effects."""
    from mesa.runner import run_benchmark as _run_benchmark

    return _run_benchmark(*args, **kwargs)


def run_benchmark_v2(*args, **kwargs):
    """Lazily import the v2 runner to avoid package import side effects."""
    from mesa.runner import run_benchmark_v2 as _run_benchmark_v2

    return _run_benchmark_v2(*args, **kwargs)
