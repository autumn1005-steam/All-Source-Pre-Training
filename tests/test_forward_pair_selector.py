import pytest

from forward_pair.selector import CandidateScore, select_forward_pair


def _score(config: str, sources: tuple[str, ...], ap: float) -> CandidateScore:
    return CandidateScore(
        config=config,
        sources=sources,
        proxy_ap=ap,
        proxy_auc=ap - 0.02,
        seed=42,
        target="target",
        config_digest="digest-1",
    )


def test_selector_uses_k_singletons_and_k_minus_one_anchor_pairs() -> None:
    scores = [
        _score("a", ("a",), 0.70),
        _score("b", ("b",), 0.80),
        _score("c", ("c",), 0.75),
        _score("d", ("d",), 0.74),
        _score("pair_a_b", ("a", "b"), 0.801),
        _score("pair_b_c", ("b", "c"), 0.805),
        _score("pair_b_d", ("b", "d"), 0.820),
    ]

    result = select_forward_pair(scores, ("a", "b", "c", "d"), margin=0.002)

    assert result.anchor_source == "b"
    assert result.best_pair == ("b", "d")
    assert result.selected_sources == ("b", "d")
    assert result.candidate_count == 7
    assert result.pair_gain == pytest.approx(0.02)


def test_pair_below_margin_keeps_singleton() -> None:
    scores = [
        _score("a", ("a",), 0.80),
        _score("b", ("b",), 0.70),
        _score("pair_a_b", ("a", "b"), 0.8019),
    ]

    result = select_forward_pair(scores, ("a", "b"), margin=0.002)

    assert result.selected_sources == ("a",)


def test_ties_are_broken_lexicographically() -> None:
    scores = [
        _score("a", ("a",), 0.80),
        _score("b", ("b",), 0.80),
        _score("c", ("c",), 0.70),
        _score("pair_a_b", ("a", "b"), 0.81),
        _score("pair_a_c", ("a", "c"), 0.81),
    ]

    result = select_forward_pair(scores, ("c", "b", "a"), margin=0.002)

    assert result.anchor_source == "a"
    assert result.best_pair == ("a", "b")


def test_missing_anchor_pair_is_rejected() -> None:
    scores = [
        _score("a", ("a",), 0.80),
        _score("b", ("b",), 0.70),
        _score("c", ("c",), 0.60),
        _score("pair_a_b", ("a", "b"), 0.81),
    ]

    with pytest.raises(ValueError, match="missing candidate"):
        select_forward_pair(scores, ("a", "b", "c"), margin=0.002)
