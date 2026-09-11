import numpy as np
import pytest

from tempofound.sampling import TemporalNegativeSampler


def test_random_negatives_never_return_observed_edge_at_or_before_query() -> None:
    sampler = TemporalNegativeSampler.from_events(
        np.array([1, 1, 2, 2, 2, 2, 2]),
        np.array([2, 3, 4, 5, 6, 7, 8]),
        np.array([1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        seed=7,
    )

    got = sampler.sample(
        np.array([1]), np.array([3.0]), count=4, policy="random"
    )

    assert got.shape == (1, 4)
    assert len(set(got[0])) == 4
    assert not set(got[0]).intersection({2, 3})


def test_future_edge_is_not_treated_as_observed_at_query_time() -> None:
    sampler = TemporalNegativeSampler.from_events(
        np.array([1, 1, 2]),
        np.array([2, 3, 4]),
        np.array([1.0, 10.0, 1.0]),
        seed=3,
    )

    got = sampler.valid_candidates(1, 5.0, policy="random")

    assert 2 not in got
    assert 3 in got


def test_historical_negatives_come_from_previous_edges_but_not_source_history() -> None:
    sampler = TemporalNegativeSampler.from_events(
        np.array([1, 2, 3]),
        np.array([2, 4, 5]),
        np.array([1.0, 2.0, 3.0]),
        seed=9,
    )

    got = sampler.sample(
        np.array([1]), np.array([3.5]), count=2, policy="historical"
    )

    assert set(got[0]) == {4, 5}


def test_inductive_negatives_use_destinations_first_seen_after_training() -> None:
    sampler = TemporalNegativeSampler.from_events(
        np.array([1, 2, 3]),
        np.array([2, 4, 5]),
        np.array([1.0, 4.0, 5.0]),
        seed=11,
        training_end=2.0,
    )

    got = sampler.sample(
        np.array([1]), np.array([6.0]), count=2, policy="inductive"
    )

    assert set(got[0]) == {4, 5}


def test_stable_query_ids_make_sampling_independent_of_batching() -> None:
    sampler = TemporalNegativeSampler.from_events(
        np.array([1, 2, 3, 4, 5, 6]),
        np.array([2, 3, 4, 5, 6, 7]),
        np.arange(1.0, 7.0),
        seed=17,
    )
    sources = np.array([1, 2, 3])
    timestamps = np.array([7.0, 7.0, 7.0])
    query_ids = np.array([101, 102, 103])

    together = sampler.sample(
        sources,
        timestamps,
        count=3,
        policy="random",
        query_ids=query_ids,
    )
    separately = np.concatenate(
        [
            sampler.sample(
                sources[index : index + 1],
                timestamps[index : index + 1],
                count=3,
                policy="random",
                query_ids=query_ids[index : index + 1],
            )
            for index in range(3)
        ],
        axis=0,
    )

    assert np.array_equal(together, separately)


def test_sampling_fails_when_valid_pool_is_empty() -> None:
    sampler = TemporalNegativeSampler.from_events(
        np.array([1, 1]),
        np.array([2, 3]),
        np.array([1.0, 2.0]),
        seed=5,
    )

    with pytest.raises(ValueError, match="source=1.*policy=random.*available=0"):
        sampler.sample(np.array([1]), np.array([3.0]), count=1, policy="random")


def test_invalid_policy_is_rejected() -> None:
    sampler = TemporalNegativeSampler.from_events(
        np.array([1]), np.array([2]), np.array([1.0]), seed=1
    )

    with pytest.raises(ValueError, match="policy"):
        sampler.sample(np.array([1]), np.array([2.0]), count=1, policy="easy")
