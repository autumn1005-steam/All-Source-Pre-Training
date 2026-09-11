import numpy as np
import pytest
import torch

from tempofound.graph import NeighborFinder
from tempofound.model import (
    DatasetSpec,
    TemporalGraphFoundationModel,
    edge_direction_consistency_loss,
)


def _synthetic_model() -> tuple[TemporalGraphFoundationModel, NeighborFinder]:
    specs = (
        DatasetSpec("source_a", raw_node_dim=3, raw_edge_dim=2, n_nodes=4, n_edges=4),
        DatasetSpec("source_b", raw_node_dim=5, raw_edge_dim=3, n_nodes=5, n_edges=3),
    )
    model = TemporalGraphFoundationModel(
        specs,
        shared_dim=8,
        time_dim=4,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        num_neighbors=2,
        time_encoding_mode="paper_sincos",
    )
    model.load_dataset_features(
        "source_a",
        np.arange(15, dtype=np.float32).reshape(5, 3) / 10.0,
        np.arange(10, dtype=np.float32).reshape(5, 2) / 10.0,
    )
    model.load_dataset_features(
        "source_b",
        np.arange(30, dtype=np.float32).reshape(6, 5) / 10.0,
        np.arange(12, dtype=np.float32).reshape(4, 3) / 10.0,
    )
    finder = NeighborFinder.from_events(
        sources=np.array([1, 2, 1, 3]),
        destinations=np.array([2, 3, 3, 4]),
        edge_indices=np.array([1, 2, 3, 4]),
        timestamps=np.array([1.0, 2.0, 2.5, 3.0]),
    )
    return model, finder


def test_edc_matches_manuscript_formula() -> None:
    forward = torch.tensor([0.0, 1.5, -0.5])
    reverse = torch.tensor([0.0, -0.25, 0.75])
    expected = torch.mean((torch.sigmoid(forward) - torch.sigmoid(reverse)) ** 2)

    actual = edge_direction_consistency_loss(forward, reverse)

    assert torch.allclose(actual, expected)
    assert edge_direction_consistency_loss(forward, forward).item() == 0.0


def test_heterogeneous_adapters_produce_shared_dimension() -> None:
    model, finder = _synthetic_model()

    encoded = model.encode_nodes(
        "source_a", finder, np.array([1, 2]), np.array([2.5, 3.5])
    )

    assert encoded.shape == (2, 8)
    assert torch.isfinite(encoded).all()


def test_model_returns_five_pretext_losses_and_explicit_domain_loss() -> None:
    model, finder = _synthetic_model()
    batch = {
        "src_idx": np.array([1, 2]),
        "dst_idx": np.array([2, 3]),
        "neg_dst_idx": np.array([4, 1]),
        "cut_time": np.array([2.5, 3.5]),
        "time_gaps": np.array([1.5, 0.5], dtype=np.float32),
        "ngh_stats": np.array(
            [[1.0, 0.5, 0.7, 1.0, 0.5], [0.7, 0.4, 0.4, 0.4, 0.5]],
            dtype=np.float32,
        ),
        "node_props": np.array(
            [[1.0, 0.5, 0.7, 1.0, 0.5], [0.7, 0.4, 0.4, 0.4, 0.5]],
            dtype=np.float32,
        ),
    }

    losses = model.pretrain_forward("source_a", finder, batch)
    total = model.weighted_pretrain_loss(losses)
    total.backward()

    assert set(losses) == {"mtlp", "tnp", "edc", "npp", "tip", "domain"}
    assert all(loss.ndim == 0 and torch.isfinite(loss) for loss in losses.values())
    assert torch.isfinite(total)
    assert model.task_log_vars.grad is not None


def test_tip_loss_backpropagates_to_time_head() -> None:
    model, finder = _synthetic_model()
    batch = {
        "src_idx": np.array([1, 2]),
        "dst_idx": np.array([2, 3]),
        "neg_dst_idx": np.array([4, 1]),
        "cut_time": np.array([2.5, 3.5]),
        "time_gaps": np.array([1.5, 0.5], dtype=np.float32),
        "ngh_stats": np.ones((2, 5), dtype=np.float32),
        "node_props": np.ones((2, 5), dtype=np.float32),
    }

    losses = model.pretrain_forward("source_a", finder, batch)
    losses["tip"].backward()

    assert torch.isfinite(losses["tip"])
    assert any(
        parameter.grad is not None
        for parameter in model.time_shift_head.parameters()
    )


def test_domain_loss_coefficient_is_explicit() -> None:
    model, _ = _synthetic_model()
    losses = {
        name: torch.tensor(value)
        for name, value in {
            "mtlp": 1.0,
            "tnp": 2.0,
            "edc": 3.0,
            "npp": 4.0,
            "tip": 5.0,
            "domain": 7.0,
        }.items()
    }

    without_domain = model.weighted_pretrain_loss(losses, domain_coefficient=0.0)
    with_domain = model.weighted_pretrain_loss(losses, domain_coefficient=0.25)

    assert (with_domain - without_domain).item() == pytest.approx(1.75)


def test_neighborhood_statistics_are_causal() -> None:
    model, finder = _synthetic_model()

    stats = model.compute_neighborhood_stats(
        finder, np.array([1]), np.array([2.5])
    )

    assert stats.shape == (1, 5)
    assert np.isclose(stats[0, 0], np.log1p(1))
    assert np.isclose(stats[0, 1], np.log1p(1.5))
    assert stats[0, 4] == 0.5


def test_graphmixer_backbone_uses_same_adapter_and_output_contract() -> None:
    spec = DatasetSpec("source", 3, 2, 4, 4)
    model = TemporalGraphFoundationModel(
        (spec,),
        shared_dim=8,
        time_dim=4,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        num_neighbors=2,
        time_encoding_mode="paper_sincos",
        backbone="graphmixer",
    )
    model.load_dataset_features(
        "source",
        np.arange(15, dtype=np.float32).reshape(5, 3) / 10.0,
        np.arange(10, dtype=np.float32).reshape(5, 2) / 10.0,
    )
    finder = NeighborFinder.from_events(
        np.array([1, 2, 1]),
        np.array([2, 3, 3]),
        np.array([1, 2, 3]),
        np.array([1.0, 2.0, 2.5]),
    )

    encoded = model.encode_nodes("source", finder, np.array([1]), np.array([3.0]))

    assert encoded.shape == (1, 8)
    assert model.backbone == "graphmixer"
    assert torch.isfinite(encoded).all()
