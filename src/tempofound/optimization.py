"""Optimization primitives for target-aware multi-source pre-training."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch


def source_affinities(
    selected: tuple[str, ...],
    singleton_utilities: Mapping[str, float],
    portfolio_utilities: Mapping[tuple[str, ...], float],
    *,
    temperature: float,
) -> dict[str, float]:
    """Convert singleton strength and marginal contribution into weights."""

    sources = tuple(sorted(selected))
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("selected sources must be non-empty and unique")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("affinity temperature must be finite and positive")
    missing = set(sources).difference(singleton_utilities)
    if missing:
        raise ValueError(f"missing singleton utilities: {sorted(missing)}")
    canonical_portfolios = {
        tuple(sorted(candidate)): float(utility)
        for candidate, utility in portfolio_utilities.items()
    }
    full_utility = canonical_portfolios.get(sources)
    scores: list[float] = []
    for source in sources:
        singleton = float(singleton_utilities[source])
        if not math.isfinite(singleton):
            raise ValueError(f"singleton utility must be finite: {source}")
        score = singleton
        if full_utility is not None and len(sources) > 1:
            parent = tuple(candidate for candidate in sources if candidate != source)
            parent_utility = canonical_portfolios.get(parent)
            if parent_utility is not None:
                score += full_utility - parent_utility
        if not math.isfinite(score):
            raise ValueError(f"affinity score must be finite: {source}")
        scores.append(score)

    logits = np.asarray(scores, dtype=np.float64) / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    return {
        source: float(weight) for source, weight in zip(sources, weights, strict=True)
    }


def allocate_weighted_batch_budget(
    capacities: Mapping[str, int],
    total_budget: int,
    weights: Mapping[str, float],
    *,
    minimum: int = 1,
) -> dict[str, int]:
    """Allocate an integer cap with deterministic largest remainders."""

    if not capacities:
        raise ValueError("capacities cannot be empty")
    if set(capacities) != set(weights):
        raise ValueError("capacities and weights must contain the same sources")
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    if minimum < 0:
        raise ValueError("minimum allocation must be non-negative")
    names = tuple(sorted(capacities))
    normalized_capacities: dict[str, int] = {}
    normalized_weights: dict[str, float] = {}
    for name in names:
        capacity = capacities[name]
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 0:
            raise ValueError(f"capacity must be a non-negative integer: {name}")
        weight = float(weights[name])
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"weight must be finite and non-negative: {name}")
        normalized_capacities[name] = int(capacity)
        normalized_weights[name] = weight
    if sum(normalized_weights.values()) <= 0.0:
        raise ValueError("at least one source weight must be positive")

    target = min(total_budget, sum(normalized_capacities.values()))
    allocation = {
        name: min(minimum, normalized_capacities[name]) for name in names
    }
    if sum(allocation.values()) > target:
        raise ValueError("total_budget cannot satisfy the minimum allocations")
    remaining = target - sum(allocation.values())
    while remaining:
        available = [
            name
            for name in names
            if allocation[name] < normalized_capacities[name]
        ]
        if not available:
            break
        weight_sum = sum(normalized_weights[name] for name in available)
        effective_weights = {
            name: (
                normalized_weights[name] / weight_sum
                if weight_sum > 0.0
                else 1.0 / len(available)
            )
            for name in available
        }
        quotas = {name: remaining * effective_weights[name] for name in available}
        floor_grants = {
            name: min(
                int(math.floor(quotas[name])),
                normalized_capacities[name] - allocation[name],
            )
            for name in available
        }
        granted = sum(floor_grants.values())
        if granted:
            for name, grant in floor_grants.items():
                allocation[name] += grant
            remaining -= granted
            continue

        by_remainder = sorted(
            available,
            key=lambda name: (-(quotas[name] - math.floor(quotas[name])), name),
        )
        for name in by_remainder:
            if not remaining:
                break
            if allocation[name] < normalized_capacities[name]:
                allocation[name] += 1
                remaining -= 1
    return allocation


def project_conflicting_gradients(
    gradients: Sequence[torch.Tensor],
    *,
    eps: float = 1e-12,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """Symmetrically remove pairwise conflicts using original gradients.

    Each gradient is projected against every *original* peer that has a
    negative dot product with its current projected value.  Using originals as
    projection normals makes the operation deterministic and independent of
    mutations to other output vectors.
    """

    if not gradients:
        raise ValueError("gradients must be non-empty")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")

    size = gradients[0].numel()
    device = gradients[0].device
    dtype = gradients[0].dtype
    originals: list[torch.Tensor] = []
    shapes: list[torch.Size] = []
    for gradient in gradients:
        if not gradient.is_floating_point():
            raise ValueError("gradients must use a floating point dtype")
        if gradient.numel() != size:
            raise ValueError("gradients must contain the same number of elements")
        if gradient.device != device or gradient.dtype != dtype:
            raise ValueError("gradients must share a device and dtype")
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError("gradients must be finite")
        originals.append(gradient.reshape(-1).clone())
        shapes.append(gradient.shape)

    total_pairs = len(originals) * (len(originals) - 1) // 2
    before_cosines: list[float] = []
    conflict_pairs = 0
    for left in range(len(originals)):
        for right in range(left + 1, len(originals)):
            dot = torch.dot(originals[left], originals[right])
            if float(dot.detach().cpu()) < 0.0:
                conflict_pairs += 1
            before_cosines.append(
                _safe_cosine(originals[left], originals[right], eps=eps)
            )

    projected = [gradient.clone() for gradient in originals]
    for source_index in range(len(projected)):
        for peer_index in range(len(originals)):
            if source_index == peer_index:
                continue
            peer = originals[peer_index]
            dot = torch.dot(projected[source_index], peer)
            peer_norm_sq = torch.dot(peer, peer)
            if (
                float(dot.detach().cpu()) < 0.0
                and float(peer_norm_sq.detach().cpu()) > eps
            ):
                projected[source_index] = (
                    projected[source_index] - dot * peer / peer_norm_sq
                )

    after_cosines = [
        _safe_cosine(projected[left], projected[right], eps=eps)
        for left in range(len(projected))
        for right in range(left + 1, len(projected))
    ]
    stats = {
        "conflict_pairs": float(conflict_pairs),
        "total_pairs": float(total_pairs),
        "conflict_rate": (
            float(conflict_pairs / total_pairs) if total_pairs else 0.0
        ),
        "mean_cosine_before": (
            float(np.mean(before_cosines)) if before_cosines else 0.0
        ),
        "mean_cosine_after": (
            float(np.mean(after_cosines)) if after_cosines else 0.0
        ),
    }
    return [
        gradient.reshape(shape)
        for gradient, shape in zip(projected, shapes, strict=True)
    ], stats


def _safe_cosine(left: torch.Tensor, right: torch.Tensor, *, eps: float) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    denominator_value = float(denominator.detach().cpu())
    if denominator_value <= eps:
        return 0.0
    cosine = torch.dot(left, right) / denominator
    return float(cosine.detach().cpu())
