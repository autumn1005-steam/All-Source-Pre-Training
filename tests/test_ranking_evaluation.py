from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from tempofound.evaluation import evaluate_link_ranking, reciprocal_ranks
from tempofound.sampling import TemporalNegativeSampler


class _IdentityNodeModel(nn.Module):
    def encode_nodes(
        self,
        name: str,
        finder: object,
        node_ids: np.ndarray,
        query_times: np.ndarray,
    ) -> torch.Tensor:
        del name, finder, query_times
        return torch.as_tensor(node_ids, dtype=torch.float32).unsqueeze(-1)


class _DestinationScoreHead(nn.Module):
    def forward(
        self, source: torch.Tensor, destination: torch.Tensor
    ) -> torch.Tensor:
        del source
        return destination.squeeze(-1)


def _fixture() -> tuple[
    nn.Module,
    nn.Module,
    str,
    dict[str, object],
    str,
    TemporalNegativeSampler,
]:
    sources = np.array([9, 9, 9, 1, 2, 3])
    destinations = np.array([1, 2, 3, 4, 5, 6])
    timestamps = np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
    data: dict[str, object] = {
        "neighbor_finder": object(),
        "test_src": sources[-3:],
        "test_dst": destinations[-3:],
        "test_ts": timestamps[-3:],
        "test_eidx": np.array([103, 104, 105]),
    }
    sampler = TemporalNegativeSampler.from_events(
        sources, destinations, timestamps, seed=23
    )
    return (
        _IdentityNodeModel(),
        _DestinationScoreHead(),
        "target",
        data,
        "test",
        sampler,
    )


def test_reciprocal_rank_averages_optimistic_and_pessimistic_ties() -> None:
    positive = torch.tensor([2.0, 1.0])
    negative = torch.tensor([[0.0, 3.0], [1.0, 0.0]])

    got = reciprocal_ranks(positive, negative)

    assert torch.allclose(got, torch.tensor([0.5, 2.0 / 3.0]))


def test_reciprocal_rank_rejects_misaligned_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        reciprocal_ranks(torch.ones(2), torch.ones(3, 4))


def test_streaming_ranking_matches_single_batch() -> None:
    fixture = _fixture()

    single = evaluate_link_ranking(
        *fixture,
        batch_size=128,
        negatives_per_positive=3,
        policy="historical",
    )
    streamed = evaluate_link_ranking(
        *fixture,
        batch_size=2,
        negatives_per_positive=3,
        policy="historical",
    )

    assert streamed["mrr"] == pytest.approx(single["mrr"])
    assert streamed["mrr"] == pytest.approx(1.0)
    assert streamed["num_positives"] == 3
    assert streamed["negatives_per_positive"] == 3
    assert streamed["negative_policy"] == "historical"
    assert streamed["total_queries"] == 3
    assert streamed["valid_queries"] == 3
    assert streamed["skipped_queries"] == 0
    assert streamed["fallback_count"] == 0
    assert streamed["collision_rejection_count"] >= 0


def test_ranking_evaluator_rejects_empty_split() -> None:
    model, head, target, data, split, sampler = _fixture()
    data["test_src"] = np.array([], dtype=np.int64)
    data["test_dst"] = np.array([], dtype=np.int64)
    data["test_ts"] = np.array([], dtype=np.float64)
    data["test_eidx"] = np.array([], dtype=np.int64)

    with pytest.raises(ValueError, match="empty"):
        evaluate_link_ranking(
            model,
            head,
            target,
            data,
            split,
            sampler,
            batch_size=2,
            negatives_per_positive=3,
            policy="historical",
        )


def test_ranking_evaluator_reports_sparse_skips_and_fallbacks() -> None:
    model = _IdentityNodeModel()
    head = _DestinationScoreHead()
    data = {
        "neighbor_finder": object(),
        "test_src": np.array([9, 9]),
        "test_dst": np.array([1, 2]),
        "test_ts": np.array([0.5, 10.0]),
        "test_eidx": np.array([1, 2]),
    }
    sampler = TemporalNegativeSampler.from_events(
        np.array([9, 9, 9]),
        np.array([1, 2, 3]),
        np.array([1.0, 2.0, 3.0]),
        seed=7,
    )
    metrics = evaluate_link_ranking(
        model,
        head,
        "target",
        data,
        "test",
        sampler,
        batch_size=2,
        negatives_per_positive=1,
        policy="historical",
        fallback_policy="random",
    )
    assert metrics["total_queries"] == 2
    assert metrics["valid_queries"] == 1
    assert metrics["skipped_queries"] == 1
    assert metrics["fallback_count"] == 1
