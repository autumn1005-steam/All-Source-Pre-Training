import pytest

from tempofound.budget import allocate_batch_budget


def test_k6_global_budget_is_exact() -> None:
    allocation = allocate_batch_budget(
        {f"source_{index}": 100 for index in range(6)}, total_budget=160
    )

    assert sum(allocation.values()) == 160
    assert sorted(allocation.values()) == [26, 26, 27, 27, 27, 27]


def test_small_domain_capacity_is_reassigned() -> None:
    allocation = allocate_batch_budget(
        {"small": 2, "large_a": 100, "large_b": 100}, total_budget=20
    )

    assert allocation["small"] == 2
    assert sum(allocation.values()) == 20
    assert sorted([allocation["large_a"], allocation["large_b"]]) == [9, 9]


def test_budget_larger_than_total_capacity_uses_every_available_batch() -> None:
    assert allocate_batch_budget({"a": 2, "b": 3}, 20) == {"a": 2, "b": 3}


def test_negative_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        allocate_batch_budget({"a": -1}, 4)
