"""Deterministic target-conditioned Forward-Pair selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class CandidateScore:
    config: str
    sources: tuple[str, ...]
    proxy_ap: float
    proxy_auc: float
    seed: int
    target: str
    config_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "sources": list(self.sources),
            "proxy_ap": self.proxy_ap,
            "proxy_auc": self.proxy_auc,
            "seed": self.seed,
            "target": self.target,
            "config_digest": self.config_digest,
        }


@dataclass(frozen=True)
class SelectionResult:
    target: str
    seed: int
    config_digest: str
    candidate_sources: tuple[str, ...]
    anchor_source: str
    anchor_proxy_ap: float
    best_pair: tuple[str, str]
    best_pair_proxy_ap: float
    pair_gain: float
    margin: float
    selected_sources: tuple[str, ...]
    candidate_count: int
    singleton_scores: dict[str, float]
    anchor_pair_scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "forward_pair",
            "target": self.target,
            "seed": self.seed,
            "config_digest": self.config_digest,
            "candidate_sources": list(self.candidate_sources),
            "anchor_source": self.anchor_source,
            "anchor_proxy_ap": self.anchor_proxy_ap,
            "best_pair": list(self.best_pair),
            "best_pair_proxy_ap": self.best_pair_proxy_ap,
            "pair_gain": self.pair_gain,
            "margin": self.margin,
            "selected_sources": list(self.selected_sources),
            "candidate_count": self.candidate_count,
            "singleton_scores": self.singleton_scores,
            "anchor_pair_scores": self.anchor_pair_scores,
            "information_boundary": "target_training_period_only",
        }


def _validate_common_metadata(scores: list[CandidateScore]) -> None:
    if not scores:
        raise ValueError("candidate scores cannot be empty")
    for field in ("seed", "target", "config_digest"):
        values = {getattr(score, field) for score in scores}
        if len(values) != 1:
            raise ValueError(f"candidate scores have inconsistent {field}")
    if any(
        not math.isfinite(score.proxy_ap) or not math.isfinite(score.proxy_auc)
        for score in scores
    ):
        raise ValueError("candidate proxy scores must be finite")
    configs = [score.config for score in scores]
    if len(configs) != len(set(configs)):
        raise ValueError("candidate score configs must be unique")


def select_forward_pair(
    scores: Iterable[CandidateScore], source_pool: tuple[str, ...], margin: float = 0.002
) -> SelectionResult:
    candidates = list(scores)
    _validate_common_metadata(candidates)
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if len(source_pool) < 2 or len(source_pool) != len(set(source_pool)):
        raise ValueError("source_pool requires at least two unique sources")
    pool = set(source_pool)
    by_sources: dict[frozenset[str], CandidateScore] = {}
    for score in candidates:
        key = frozenset(score.sources)
        if len(key) != len(score.sources) or len(key) not in {1, 2}:
            raise ValueError(f"invalid candidate source set: {score.sources}")
        if not key.issubset(pool):
            raise ValueError(f"candidate contains source outside pool: {score.sources}")
        if key in by_sources:
            raise ValueError(f"duplicate candidate source set: {score.sources}")
        by_sources[key] = score

    singleton: dict[str, CandidateScore] = {}
    for source in source_pool:
        score = by_sources.get(frozenset((source,)))
        if score is None:
            raise ValueError(f"missing candidate singleton: {source}")
        singleton[source] = score
    anchor = min(source_pool, key=lambda name: (-singleton[name].proxy_ap, name))
    partner_scores: dict[str, CandidateScore] = {}
    for partner in source_pool:
        if partner == anchor:
            continue
        pair = by_sources.get(frozenset((anchor, partner)))
        if pair is None:
            raise ValueError(f"missing candidate pair: {anchor}+{partner}")
        partner_scores[partner] = pair
    expected_keys = {frozenset((source,)) for source in source_pool} | {
        frozenset((anchor, partner)) for partner in source_pool if partner != anchor
    }
    extra = set(by_sources).difference(expected_keys)
    if extra:
        raise ValueError(f"unexpected candidate source sets: {sorted(map(sorted, extra))}")
    expected_count = 2 * len(source_pool) - 1
    if len(candidates) != expected_count:
        raise ValueError(
            f"Forward-Pair requires exactly {expected_count} candidates, got {len(candidates)}"
        )

    best_partner = min(
        partner_scores,
        key=lambda name: (-partner_scores[name].proxy_ap, name),
    )
    anchor_ap = singleton[anchor].proxy_ap
    best_pair_ap = partner_scores[best_partner].proxy_ap
    gain = best_pair_ap - anchor_ap
    selected = (anchor, best_partner) if gain + 1e-12 >= margin else (anchor,)
    first = candidates[0]
    return SelectionResult(
        target=first.target,
        seed=first.seed,
        config_digest=first.config_digest,
        candidate_sources=tuple(source_pool),
        anchor_source=anchor,
        anchor_proxy_ap=anchor_ap,
        best_pair=(anchor, best_partner),
        best_pair_proxy_ap=best_pair_ap,
        pair_gain=gain,
        margin=margin,
        selected_sources=selected,
        candidate_count=expected_count,
        singleton_scores={name: singleton[name].proxy_ap for name in source_pool},
        anchor_pair_scores={
            name: partner_scores[name].proxy_ap for name in partner_scores
        },
    )
