import torch
from tempofound.temporal_experts import TemporalResidualExperts


def test_cold_start_bypass_and_active_experts_receive_gradients():
    torch.manual_seed(42)
    model = TemporalResidualExperts(8, 4, 0)
    inputs = torch.randn(3, 8, requires_grad=True)
    stats = torch.tensor([[0., 0, 0, 0, 0], [1., 1, 2, 3, 1], [1., 4, 5, 6, 1]])
    output, weights = model(inputs, stats, return_weights=True)
    assert torch.equal(output[0], inputs[0])
    assert torch.allclose(weights.sum(-1), torch.ones(3))
    assert not torch.allclose(output[1:], inputs[1:])
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(expert[0].weight.grad.abs().sum() > 0 for expert in model.experts)


def test_history_changes_routing_for_identical_embeddings():
    torch.manual_seed(42)
    model = TemporalResidualExperts(8, 4, 0).eval()
    inputs = torch.ones(2, 8)
    stats = torch.tensor([[1., 1, 2, 3, 1], [1., 4, 5, 6, 1]])
    _, weights = model(inputs, stats, return_weights=True)
    assert not torch.allclose(weights[0], weights[1])


def test_identity_experts_preserve_initial_backbone_and_rng_and_can_learn():
    import numpy as np
    from tempofound.model import DatasetSpec, TemporalGraphFoundationModel
    from tempofound.graph import NeighborFinder
    from tempofound.checkpoint import build_checkpoint, load_foundation_checkpoint_payload
    spec = (DatasetSpec('a', 3, 2, 4, 3),)
    kwargs = dict(shared_dim=8, time_dim=4, num_layers=1, num_heads=2, dropout=0., num_neighbors=2)
    torch.manual_seed(42)
    base = TemporalGraphFoundationModel(spec, **kwargs).eval()
    reference_rng = torch.get_rng_state()
    torch.manual_seed(42)
    model = TemporalGraphFoundationModel(spec, **kwargs, temporal_experts=4, expert_identity_init=True).eval()
    assert torch.equal(reference_rng, torch.get_rng_state())
    finder = NeighborFinder.from_events(np.array([1, 1]), np.array([2, 3]),
                                        np.array([1, 2]), np.array([1., 2.]))
    ids, times = np.array([1, 2]), np.array([3., 3.])
    initial = model.encode_nodes('a', finder, ids, times)
    assert torch.equal(initial, base.encode_nodes('a', finder, ids, times))
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.temporal_mixture.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.SGD(model.temporal_mixture.parameters(), lr=.1)
    model.encode_nodes('a', finder, ids, times).square().mean().backward()
    assert all(expert[-1].weight.grad.abs().sum() > 0 for expert in model.temporal_mixture.experts)
    optimizer.step()
    changed = model.encode_nodes('a', finder, ids, times)
    assert not torch.equal(changed, initial)
    payload = build_checkpoint(model, time_encoding_mode='paper_sincos', model_config=model.model_config())
    restored = load_foundation_checkpoint_payload(payload).eval()
    assert torch.equal(changed, restored.encode_nodes('a', finder, ids, times))
