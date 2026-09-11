"""Fair allocation of a strict global pre-training update budget."""

from __future__ import annotations

from collections.abc import Mapping


def allocate_batch_budget(
    capacities: Mapping[str, int], total_budget: int
) -> dict[str, int]:
    if total_budget < 0:
        raise ValueError("total_budget must be non-negative")
    if any(capacity < 0 for capacity in capacities.values()):
        raise ValueError("batch capacities must be non-negative")
    allocation = {name: 0 for name in capacities}
    remaining = min(total_budget, sum(capacities.values()))
    active = list(capacities)
    while remaining and active:
        next_active: list[str] = []
        for name in active:
            if remaining == 0:
                break
            if allocation[name] < capacities[name]:
                allocation[name] += 1
                remaining -= 1
            if allocation[name] < capacities[name]:
                next_active.append(name)
        active = next_active
    return allocation
