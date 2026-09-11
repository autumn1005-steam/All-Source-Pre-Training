"""Deterministic budgeted beam search over source-domain portfolios."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping


SourceSet = tuple[str, ...]


def _canonical(sources: SourceSet) -> SourceSet:
    canonical = tuple(sorted(str(source) for source in sources))
    if not canonical or len(canonical) != len(set(canonical)):
        raise ValueError(f"source set must be non-empty and unique: {sources}")
    return canonical


def _rank(
    candidates: set[SourceSet], utilities: Mapping[SourceSet, float]
) -> list[SourceSet]:
    return sorted(candidates, key=lambda item: (-utilities[item], item))


@dataclass(frozen=True)
class PortfolioSearchState:
    sources: SourceSet
    beam_width: int
    max_sources: int
    candidate_budget: int
    frontier: tuple[SourceSet, ...]
    explored: tuple[SourceSet, ...]
    accepted: tuple[SourceSet, ...]
    pending: tuple[SourceSet, ...]
    utilities: dict[SourceSet, float]
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": list(self.sources),
            "beam_width": self.beam_width,
            "max_sources": self.max_sources,
            "candidate_budget": self.candidate_budget,
            "frontier": [list(candidate) for candidate in self.frontier],
            "explored": [list(candidate) for candidate in self.explored],
            "accepted": [list(candidate) for candidate in self.accepted],
            "pending": [list(candidate) for candidate in self.pending],
            "utilities": {
                "+".join(candidate): utility
                for candidate, utility in sorted(self.utilities.items())
            },
            "stop_reason": self.stop_reason,
        }


def initialize_search(
    sources: SourceSet,
    *,
    beam_width: int,
    max_sources: int,
    candidate_budget: int,
) -> PortfolioSearchState:
    canonical_sources = _canonical(sources)
    if len(canonical_sources) < 2:
        raise ValueError("portfolio search requires at least two sources")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if not 1 <= max_sources <= len(canonical_sources):
        raise ValueError("max_sources must be between one and the source count")
    if candidate_budget < len(canonical_sources):
        raise ValueError("candidate_budget must cover all singleton sources")
    singletons = tuple((source,) for source in canonical_sources)
    return PortfolioSearchState(
        sources=canonical_sources,
        beam_width=beam_width,
        max_sources=max_sources,
        candidate_budget=candidate_budget,
        frontier=(),
        explored=(),
        accepted=(),
        pending=singletons,
        utilities={},
    )


def _best_parent_utility(
    candidate: SourceSet, utilities: Mapping[SourceSet, float]
) -> float:
    parents = [
        candidate[:index] + candidate[index + 1 :]
        for index in range(len(candidate))
    ]
    available = [utilities[parent] for parent in parents if parent in utilities]
    if not available:
        raise ValueError(f"candidate has no evaluated direct parent: {candidate}")
    return max(available)


def advance_search(
    state: PortfolioSearchState,
    candidate_utilities: Mapping[SourceSet, float],
    *,
    min_gain: float,
) -> PortfolioSearchState:
    if state.stop_reason is not None:
        raise ValueError(f"portfolio search already stopped: {state.stop_reason}")
    if min_gain < 0.0:
        raise ValueError("min_gain must be non-negative")
    normalized: dict[SourceSet, float] = {}
    for candidate, utility in candidate_utilities.items():
        canonical = _canonical(candidate)
        if canonical in normalized:
            raise ValueError(f"duplicate utility for candidate: {canonical}")
        value = float(utility)
        if not math.isfinite(value):
            raise ValueError(f"candidate utility must be finite: {canonical}")
        normalized[canonical] = value
    missing = set(state.pending).difference(normalized)
    if missing:
        raise ValueError(f"missing utilities for pending candidates: {sorted(missing)}")
    unexpected = set(normalized).difference(state.pending)
    if unexpected:
        raise ValueError(f"unexpected utilities for candidates: {sorted(unexpected)}")

    utilities = {**state.utilities, **normalized}
    current = set(state.pending)
    depth = len(state.pending[0]) if state.pending else 0
    if depth == 1:
        survivors = _rank(current, utilities)[: state.beam_width]
    else:
        eligible = {
            candidate
            for candidate in current
            if utilities[candidate] - _best_parent_utility(candidate, utilities)
            + 1e-12
            >= min_gain
        }
        survivors = _rank(eligible, utilities)[: state.beam_width]

    explored = tuple((*state.explored, *state.pending))
    accepted = tuple((*state.accepted, *survivors))
    if not survivors:
        return replace(
            state,
            frontier=(),
            explored=explored,
            accepted=accepted,
            pending=(),
            utilities=utilities,
            stop_reason="no_gain",
        )
    if depth >= state.max_sources:
        return replace(
            state,
            frontier=tuple(survivors),
            explored=explored,
            accepted=accepted,
            pending=(),
            utilities=utilities,
            stop_reason="max_sources",
        )

    explored_set = set(explored)
    expansions: list[SourceSet] = []
    seen: set[SourceSet] = set()
    for parent in survivors:
        for source in state.sources:
            if source in parent:
                continue
            candidate = _canonical((*parent, source))
            if candidate not in explored_set and candidate not in seen:
                seen.add(candidate)
                expansions.append(candidate)
    remaining = state.candidate_budget - len(explored)
    pending = tuple(expansions[: max(remaining, 0)])
    stop_reason = None
    if not pending:
        stop_reason = "candidate_budget" if remaining <= 0 else "search_exhausted"
    return replace(
        state,
        frontier=tuple(survivors),
        explored=explored,
        accepted=accepted,
        pending=pending,
        utilities=utilities,
        stop_reason=stop_reason,
    )


def select_portfolio(state: PortfolioSearchState) -> SourceSet:
    if not state.accepted:
        raise ValueError("portfolio search has no accepted candidates")
    return min(
        state.accepted,
        key=lambda candidate: (-state.utilities[candidate], candidate),
    )
