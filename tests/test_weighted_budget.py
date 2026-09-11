import pytest

from tempofound.optimization import (
    allocate_weighted_batch_budget,
    source_affinities,
)


def test_weighted_budget_uses_largest_remainder_under_capacity() -> None:
    got = allocate_weighted_batch_budget(
        {"a": 10, "b": 10, "c": 2},
        10,
        {"a": 0.5, "b": 0.3, "c": 0.2},
        minimum=1,
    )

    assert got == {"a": 5, "b": 3, "c": 2}
    assert sum(got.values()) == 10


def test_weighted_budget_redistributes_saturated_source_quota() -> None:
    got = allocate_weighted_batch_budget(
        {"a": 1, "b": 10}, 6, {"a": 0.9, "b": 0.1}, minimum=0
    )

    assert got == {"a": 1, "b": 5}


def test_weighted_budget_is_deterministic_on_equal_remainders() -> None:
    got = allocate_weighted_batch_budget(
        {"b": 5, "a": 5, "c": 5},
        5,
        {"a": 1.0, "b": 1.0, "c": 1.0},
        minimum=0,
    )

    assert got == {"a": 2, "b": 2, "c": 1}


def test_budget_caps_at_total_available_capacity() -> None:
    got = allocate_weighted_batch_budget(
        {"a": 2, "b": 1}, 10, {"a": 0.5, "b": 0.5}, minimum=1
    )

    assert got == {"a": 2, "b": 1}


def test_affinities_are_positive_and_normalized() -> None:
    got = source_affinities(
        ("a", "b"),
        {"a": 0.8, "b": 0.7},
        {("a", "b"): 0.9, ("a",): 0.8, ("b",): 0.7},
        temperature=0.1,
    )

    assert set(got) == {"a", "b"}
    assert sum(got.values()) == pytest.approx(1.0)
    assert all(value > 0 for value in got.values())
    assert got["a"] > got["b"]


def test_affinity_ties_are_uniform() -> None:
    got = source_affinities(
        ("a", "b"),
        {"a": 0.5, "b": 0.5},
        {("a", "b"): 0.6},
        temperature=1.0,
    )

    assert got == {"a": pytest.approx(0.5), "b": pytest.approx(0.5)}


def test_weighted_budget_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError, match="same sources"):
        allocate_weighted_batch_budget(
            {"a": 2, "b": 2}, 2, {"a": 1.0}, minimum=0
        )


def test_affinities_require_every_singleton_utility() -> None:
    with pytest.raises(ValueError, match="singleton"):
        source_affinities(
            ("a", "b"), {"a": 0.5}, {("a", "b"): 0.6}, temperature=1.0
        )
