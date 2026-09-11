"""Small, testable utilities shared by the efficiency benchmark."""

from __future__ import annotations

import statistics
from pathlib import PurePath
from typing import Iterable


def summarize_durations(durations: Iterable[float], *, queries: int) -> dict[str, float | int]:
    values = [float(value) for value in durations]
    if not values or any(value < 0.0 for value in values):
        raise ValueError("durations must contain non-negative measurements")
    if queries <= 0:
        raise ValueError("queries must be positive")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "repetitions": len(values),
        "seconds_mean": mean,
        "seconds_std": std,
        "milliseconds_per_query_mean": 1000.0 * mean / queries,
        "queries_per_second": queries / mean if mean > 0.0 else float("inf"),
    }


def classify_pipeline_stage(checkpoint_path: str) -> str:
    parts = tuple(part.lower() for part in PurePath(checkpoint_path.replace("\\", "/")).parts)
    if "candidates" in parts:
        return "source_selection"
    if "final" in parts and "all_sources" in parts:
        return "control_branch"
    if "final" in parts and ("rasp" in parts or "forward_pair" in parts):
        return "source_pretraining"
    return "unclassified"
