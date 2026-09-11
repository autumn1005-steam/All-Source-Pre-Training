import pytest

from forward_pair.portfolio import (
    advance_search,
    initialize_search,
    select_portfolio,
)


def test_beam_finds_non_anchor_pair() -> None:
    state = initialize_search(
        ("a", "b", "c"),
        beam_width=2,
        max_sources=2,
        candidate_budget=7,
    )

    state = advance_search(
        state,
        {("a",): 0.80, ("b",): 0.79, ("c",): 0.60},
        min_gain=0.01,
    )

    assert set(state.pending) == {("a", "b"), ("a", "c"), ("b", "c")}

    state = advance_search(
        state,
        {("a", "b"): 0.78, ("a", "c"): 0.81, ("b", "c"): 0.85},
        min_gain=0.01,
    )

    assert select_portfolio(state) == ("b", "c")
    assert state.stop_reason == "max_sources"


def test_candidate_budget_is_never_exceeded() -> None:
    state = initialize_search(
        ("a", "b", "c", "d"),
        beam_width=3,
        max_sources=3,
        candidate_budget=6,
    )

    state = advance_search(
        state,
        {("a",): 0.8, ("b",): 0.7, ("c",): 0.6, ("d",): 0.5},
        min_gain=0.0,
    )

    assert len(state.explored) + len(state.pending) == 6
    assert state.pending == (("a", "b"), ("a", "c"))


def test_no_risk_adjusted_gain_stops_expansion() -> None:
    state = initialize_search(
        ("a", "b", "c"), beam_width=1, max_sources=3, candidate_budget=8
    )
    state = advance_search(
        state,
        {("a",): 0.8, ("b",): 0.7, ("c",): 0.6},
        min_gain=0.01,
    )
    state = advance_search(
        state,
        {("a", "b"): 0.805, ("a", "c"): 0.79},
        min_gain=0.01,
    )

    assert state.pending == ()
    assert state.stop_reason == "no_gain"
    assert select_portfolio(state) == ("a",)


def test_ties_are_broken_by_canonical_source_tuple() -> None:
    state = initialize_search(
        ("c", "b", "a"), beam_width=2, max_sources=1, candidate_budget=3
    )

    state = advance_search(
        state,
        {("c",): 0.8, ("b",): 0.8, ("a",): 0.8},
        min_gain=0.0,
    )

    assert state.frontier == (("a",), ("b",))
    assert select_portfolio(state) == ("a",)


def test_advance_requires_every_pending_score() -> None:
    state = initialize_search(
        ("a", "b"), beam_width=1, max_sources=2, candidate_budget=3
    )

    with pytest.raises(ValueError, match="missing utilities"):
        advance_search(state, {("a",): 0.8}, min_gain=0.0)


@pytest.mark.parametrize(
    ("beam_width", "max_sources", "candidate_budget", "message"),
    [
        (0, 2, 3, "beam_width"),
        (1, 0, 3, "max_sources"),
        (1, 3, 3, "max_sources"),
        (1, 2, 1, "candidate_budget"),
    ],
)
def test_search_rejects_invalid_limits(
    beam_width: int,
    max_sources: int,
    candidate_budget: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        initialize_search(
            ("a", "b"),
            beam_width=beam_width,
            max_sources=max_sources,
            candidate_budget=candidate_budget,
        )
