"""Candidate policies used by manuscript control experiments."""

from __future__ import annotations

import itertools
from collections.abc import Mapping

import numpy as np


def exhaustive_subsets(source_pool: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    if not source_pool or len(source_pool) != len(set(source_pool)):
        raise ValueError("source_pool must contain unique sources")
    return tuple(
        subset
        for size in range(1, len(source_pool) + 1)
        for subset in itertools.combinations(source_pool, size)
    )


def exact_random_subsets(
    source_pool: tuple[str, ...], *, count: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    candidates = exhaustive_subsets(source_pool)
    if count < 1 or count > len(candidates):
        raise ValueError("count must be between one and the exhaustive candidate count")
    indices = np.random.default_rng(seed).choice(
        len(candidates), size=count, replace=False
    )
    return tuple(candidates[int(index)] for index in indices)


def cardinality_matched_random(
    source_pool: tuple[str, ...], *, size: int, seed: int
) -> tuple[str, ...]:
    if size < 1 or size > len(source_pool):
        raise ValueError("size must be within the source pool")
    indices = np.random.default_rng(seed).choice(
        len(source_pool), size=size, replace=False
    )
    return tuple(source_pool[int(index)] for index in sorted(indices))


def standard_greedy_from_scores(
    scores: Mapping[tuple[str, ...], float],
    source_pool: tuple[str, ...],
    *,
    margin: float = 0.002,
) -> tuple[str, ...]:
    if margin < 0:
        raise ValueError("margin must be non-negative")
    position = {source: index for index, source in enumerate(source_pool)}

    def canonical(subset: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(subset, key=position.__getitem__))

    canonical_scores = {canonical(tuple(key)): float(value) for key, value in scores.items()}
    missing = [(source,) for source in source_pool if (source,) not in canonical_scores]
    if missing:
        raise ValueError(f"missing singleton scores: {missing}")
    selected = min(
        ((source,) for source in source_pool),
        key=lambda subset: (-canonical_scores[subset], subset),
    )
    while len(selected) < len(source_pool):
        expansions = [
            canonical((*selected, source))
            for source in source_pool
            if source not in selected
        ]
        missing_expansions = [subset for subset in expansions if subset not in canonical_scores]
        if missing_expansions:
            raise ValueError(f"missing greedy expansion scores: {missing_expansions}")
        best = min(
            expansions, key=lambda subset: (-canonical_scores[subset], subset)
        )
        if canonical_scores[best] - canonical_scores[selected] + 1e-12 < margin:
            break
        selected = best
    return selected
