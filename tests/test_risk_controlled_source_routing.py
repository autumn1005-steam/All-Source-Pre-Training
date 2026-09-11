import numpy as np
import torch

from forward_pair.proxy import extend_model_for_target
from tempofound.checkpoint import build_checkpoint, load_foundation_checkpoint_payload
from tempofound.model import DatasetSpec, TemporalGraphFoundationModel
from tempofound.temporal_experts import RiskControlledSourceRouter


def _spec(name: str) -> DatasetSpec:
    return DatasetSpec(name=name, raw_node_dim=2, raw_edge_dim=2, n_nodes=4, n_edges=4)


def _target_data():
    return {
        "node_features": np.zeros((5, 2), dtype=np.float32),
        "edge_features": np.zeros((5, 2), dtype=np.float32),
    }


def test_unseen_target_reuses_source_experts_without_creating_target_expert():
    model = TemporalGraphFoundationModel(
        (_spec("a"), _spec("b")),
        shared_dim=8,
        time_dim=4,
        num_layers=1,
        num_heads=2,
        num_neighbors=2,
        dropout=0.0,
        source_experts=True,
        expert_identity_init=True,
        risk_controlled_routing=True,
    )
    extended = extend_model_for_target(model, "target", _target_data())
    assert tuple(extended.source_expert_bank.experts.keys()) == ("a", "b")
    assert "target" not in extended.source_expert_bank.experts
    assert extended.source_router is not None
    assert extended.source_router.names == ("a", "b")


def test_legacy_source_expert_extension_is_preserved_for_reproducibility():
    model = TemporalGraphFoundationModel(
        (_spec("a"), _spec("b")),
        shared_dim=8,
        time_dim=4,
        num_layers=1,
        num_heads=2,
        num_neighbors=2,
        dropout=0.0,
        source_experts=True,
        expert_identity_init=True,
        risk_controlled_routing=False,
    )
    extended = extend_model_for_target(model, "target", _target_data())
    assert "target" in extended.source_expert_bank.experts
    assert extended.source_router is None


def test_router_prior_controls_initial_source_mixture():
    router = RiskControlledSourceRouter(
        ("a", "b"), 4, 0.0, prior_strength=4.0, entropy_penalty=0.0, abstain_bias=5.0
    )
    embeddings = torch.zeros(3, 4)
    stats = torch.ones(3, 5)
    deltas = torch.stack((torch.ones(3, 4), -torch.ones(3, 4)), dim=1)

    router.set_prior({"a": 0.99, "b": 0.01})
    _, diag_a = router(embeddings, stats, deltas, return_diagnostics=True)
    router.set_prior({"a": 0.01, "b": 0.99})
    _, diag_b = router(embeddings, stats, deltas, return_diagnostics=True)

    assert torch.all(diag_a["source_weights"][:, 0] > 0.9)
    assert torch.all(diag_b["source_weights"][:, 1] > 0.9)


def test_router_cold_start_is_exact_identity():
    router = RiskControlledSourceRouter(("a", "b"), 4, 0.0, abstain_bias=5.0)
    embeddings = torch.randn(5, 4)
    stats = torch.zeros(5, 5)
    deltas = torch.randn(5, 2, 4)
    output = router(embeddings, stats, deltas)
    assert torch.allclose(output, embeddings)


def test_router_can_disable_local_acceptance_without_changing_source_routing():
    router = RiskControlledSourceRouter(
        ("a", "b"), 4, 0.0, abstain_bias=-20.0, local_acceptance=False
    )
    embeddings = torch.zeros(3, 4)
    stats = torch.ones(3, 5)
    deltas = torch.stack((torch.ones(3, 4), -torch.ones(3, 4)), dim=1)
    router.set_prior({"a": 0.9, "b": 0.1})

    _, diagnostics = router(embeddings, stats, deltas, return_diagnostics=True)

    assert torch.allclose(
        diagnostics["local_accept_probability"],
        torch.ones_like(diagnostics["local_accept_probability"]),
    )


def test_risk_router_checkpoint_roundtrip_is_strictly_loadable():
    model = TemporalGraphFoundationModel(
        (_spec("a"), _spec("b")),
        shared_dim=8,
        time_dim=4,
        num_layers=1,
        num_heads=2,
        num_neighbors=2,
        dropout=0.0,
        source_experts=True,
        expert_identity_init=True,
        risk_controlled_routing=True,
    )
    model.set_source_routing_prior({"a": 0.8, "b": 0.2})
    payload = build_checkpoint(
        model,
        time_encoding_mode="paper_sincos",
        model_config=model.model_config(),
    )
    restored = load_foundation_checkpoint_payload(payload)
    assert restored.risk_controlled_routing
    assert restored.source_expert_names == ("a", "b")
    assert torch.allclose(restored.source_router.prior_logits, model.source_router.prior_logits)
