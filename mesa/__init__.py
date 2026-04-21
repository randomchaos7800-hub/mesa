"""MESA — Memory Eval Suite for Agents.

A reproducible benchmark framework for personal AI memory systems.
Plug in any memory backend via MemoryAdapter and score it against
the gold dataset of real-world memory scenarios.
"""

from mesa.adapter import MemoryAdapter
from mesa.scorer import exact_match, rouge1_f1, composite, is_refusal

__version__ = "0.1.0"
__all__ = ["MemoryAdapter", "exact_match", "rouge1_f1", "composite", "is_refusal"]
