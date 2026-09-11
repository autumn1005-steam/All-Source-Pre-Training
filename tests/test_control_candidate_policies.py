from forward_pair.controls import (
    cardinality_matched_random,
    exhaustive_subsets,
    exact_random_subsets,
    standard_greedy_from_scores,
)


def test_k4_exhaustive_control_contains_all_fifteen_nonempty_subsets() -> None:
    subsets = exhaustive_subsets(("a", "b", "c", "d"))

    assert len(subsets) == 15
    assert ("a",) in subsets
    assert ("a", "b", "c", "d") in subsets


def test_random7_is_exact_unique_and_deterministic() -> None:
    first = exact_random_subsets(("a", "b", "c", "d"), count=7, seed=42)
    second = exact_random_subsets(("a", "b", "c", "d"), count=7, seed=42)

    assert first == second
    assert len(first) == len(set(first)) == 7


def test_cardinality_matched_random_has_requested_size() -> None:
    subset = cardinality_matched_random(("a", "b", "c", "d"), size=2, seed=3)

    assert len(subset) == 2
    assert set(subset).issubset({"a", "b", "c", "d"})


def test_standard_greedy_uses_higher_order_gain_until_margin_fails() -> None:
    scores = {
        ("a",): 0.80,
        ("b",): 0.75,
        ("c",): 0.70,
        ("a", "b"): 0.82,
        ("a", "c"): 0.81,
        ("a", "b", "c"): 0.821,
    }

    result = standard_greedy_from_scores(scores, ("a", "b", "c"), margin=0.002)

    assert result == ("a", "b")
