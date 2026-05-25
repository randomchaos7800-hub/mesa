"""Statistical helpers for MESA benchmark reporting."""

from __future__ import annotations

import math
import random


def _coerce_numeric(value: float | bool | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def mean(values: list[float | bool | None]) -> float | None:
    normalized = [_coerce_numeric(value) for value in values]
    filtered = [value for value in normalized if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def bootstrap_mean_ci(
    values: list[float | bool | None],
    confidence: float = 0.95,
    iterations: int = 500,
    seed: int = 0,
) -> dict[str, float | int] | None:
    """Return a bootstrap confidence interval around the sample mean."""
    normalized = [_coerce_numeric(value) for value in values]
    filtered = [value for value in normalized if value is not None]
    if not filtered:
        return None
    if len(filtered) == 1:
        value = round(filtered[0], 4)
        return {
            "mean": value,
            "lower": value,
            "upper": value,
            "n": 1,
            "iterations": iterations,
            "confidence": confidence,
        }
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        draw = [filtered[rng.randrange(len(filtered))] for _ in range(len(filtered))]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    alpha = 1.0 - confidence
    lower_index = max(0, math.floor((alpha / 2) * (len(samples) - 1)))
    upper_index = min(len(samples) - 1, math.ceil((1 - alpha / 2) * (len(samples) - 1)))
    return {
        "mean": round(sum(filtered) / len(filtered), 4),
        "lower": round(samples[lower_index], 4),
        "upper": round(samples[upper_index], 4),
        "n": len(filtered),
        "iterations": iterations,
        "confidence": confidence,
    }
