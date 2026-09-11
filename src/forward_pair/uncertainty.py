"""Uncertainty estimates and conservative utility for target proxies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class BootstrapSummary:
    mean: float
    standard_error: float
    lower: float
    upper: float
    replicates: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": float(self.mean),
            "standard_error": float(self.standard_error),
            "lower": float(self.lower),
            "upper": float(self.upper),
            "replicates": [float(value) for value in self.replicates],
        }


def bootstrap_mean(
    values: Iterable[float],
    *,
    seed: int,
    replicates: int,
    confidence: float,
) -> BootstrapSummary:
    """Summarize a mean with a deterministic non-parametric bootstrap."""

    samples = np.asarray(tuple(values), dtype=np.float64)
    if samples.ndim != 1 or not len(samples):
        raise ValueError("bootstrap values must be a non-empty one-dimensional set")
    if not np.isfinite(samples).all():
        raise ValueError("bootstrap values must be finite")
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must be between zero and one")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(samples), size=(replicates, len(samples)))
    replicate_means = samples[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(replicate_means, (tail, 1.0 - tail))
    return BootstrapSummary(
        mean=float(samples.mean()),
        standard_error=float(replicate_means.std(ddof=0)),
        lower=float(lower),
        upper=float(upper),
        replicates=tuple(float(value) for value in replicate_means),
    )


def risk_adjusted_utility(
    summary: BootstrapSummary,
    *,
    beta: float,
    cardinality: int,
    size_penalty: float,
) -> float:
    """Return conservative proxy utility with an explicit size penalty."""

    if beta < 0.0:
        raise ValueError("risk beta must be non-negative")
    if cardinality < 1:
        raise ValueError("portfolio cardinality must be positive")
    if size_penalty < 0.0:
        raise ValueError("portfolio size_penalty must be non-negative")
    values = (
        summary.mean,
        summary.standard_error,
        summary.lower,
        summary.upper,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("bootstrap summary statistics must be finite")
    return float(
        summary.mean
        - beta * summary.standard_error
        - size_penalty * (cardinality - 1)
    )


def paired_temporal_gate(candidate, reference, *, seed, replicates, confidence,
                         comparisons, min_gain):
    """Empirical paired-window gate; no coverage guarantee under temporal drift.

    Compare identical time windows, adjust nominal confidence for the number
    of comparisons, and require positive gain in the most recent window.
    """
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.ndim != 1 or candidate.shape != reference.shape or len(candidate) < 2:
        raise ValueError("paired windows must have matching lengths of at least two")
    if comparisons < 1 or not 0 < confidence < 1 or min_gain < 0:
        raise ValueError("invalid paired gate configuration")
    differences = candidate - reference
    summary = bootstrap_mean(differences, seed=seed, replicates=replicates,
                             confidence=1 - (1 - confidence) / comparisons)
    return {
        "window_deltas": differences.tolist(),
        "mean_delta": summary.mean,
        "lower_delta": summary.lower,
        "latest_delta": float(differences[-1]),
        "accepted": bool(summary.lower > min_gain and differences[-1] > min_gain),
        "nominal_adjusted_confidence": 1 - (1 - confidence) / comparisons,
        "interpretation": "empirical heuristic; temporal coverage not guaranteed",
    }
