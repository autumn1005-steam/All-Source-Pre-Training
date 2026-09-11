import pytest

from forward_pair.uncertainty import (
    BootstrapSummary,
    bootstrap_mean,
    risk_adjusted_utility,
)


def test_degenerate_bootstrap_has_zero_uncertainty() -> None:
    summary = bootstrap_mean(
        [0.8, 0.8, 0.8], seed=3, replicates=100, confidence=0.95
    )

    assert summary.mean == pytest.approx(0.8)
    assert summary.standard_error == pytest.approx(0.0)
    assert summary.lower == pytest.approx(0.8)
    assert summary.upper == pytest.approx(0.8)
    assert len(summary.replicates) == 100


def test_bootstrap_is_deterministic_for_the_same_seed() -> None:
    first = bootstrap_mean(
        [0.6, 0.7, 0.9], seed=13, replicates=64, confidence=0.9
    )
    second = bootstrap_mean(
        [0.6, 0.7, 0.9], seed=13, replicates=64, confidence=0.9
    )

    assert first == second
    assert first.lower <= first.mean <= first.upper
    assert first.standard_error > 0.0


def test_risk_utility_penalizes_uncertainty_and_size() -> None:
    summary = BootstrapSummary(0.8, 0.05, 0.7, 0.9, (0.7, 0.9))

    utility = risk_adjusted_utility(
        summary, beta=2.0, cardinality=3, size_penalty=0.01
    )

    assert utility == pytest.approx(0.68)


@pytest.mark.parametrize(
    ("values", "replicates", "confidence", "message"),
    [
        ([], 10, 0.95, "non-empty"),
        ([0.8, float("nan")], 10, 0.95, "finite"),
        ([0.8], 0, 0.95, "replicates"),
        ([0.8], 10, 1.0, "confidence"),
    ],
)
def test_bootstrap_rejects_invalid_inputs(
    values: list[float], replicates: int, confidence: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        bootstrap_mean(
            values,
            seed=1,
            replicates=replicates,
            confidence=confidence,
        )


def test_summary_serializes_without_numpy_scalars() -> None:
    summary = bootstrap_mean(
        [0.5, 0.75], seed=5, replicates=8, confidence=0.8
    )

    payload = summary.to_dict()

    assert set(payload) == {
        "mean",
        "standard_error",
        "lower",
        "upper",
        "replicates",
    }
    assert all(isinstance(value, float) for value in payload["replicates"])


def test_risk_utility_rejects_invalid_penalties() -> None:
    summary = BootstrapSummary(0.8, 0.05, 0.7, 0.9, (0.7, 0.9))

    with pytest.raises(ValueError, match="beta"):
        risk_adjusted_utility(
            summary, beta=-1.0, cardinality=1, size_penalty=0.0
        )
    with pytest.raises(ValueError, match="cardinality"):
        risk_adjusted_utility(
            summary, beta=1.0, cardinality=0, size_penalty=0.0
        )
