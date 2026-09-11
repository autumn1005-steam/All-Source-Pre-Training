import numpy as np
import torch

from tempofound.temporal_experts import RiskControlledSourceRouter
from tempofound.transfer import _bootstrap_gate_from_gains, _pairwise_ranking_loss


def test_target_gate_zero_forces_identity():
    torch.manual_seed(0)
    router = RiskControlledSourceRouter(("a", "b"), dim=4, dropout=0.0)
    router.set_prior({"a": 0.7, "b": 0.3})
    router.set_target_gate(0.0)
    embeddings = torch.randn(3, 4)
    history = torch.tensor([[1.0, 0, 0, 0, 1.0]] * 3)
    deltas = torch.randn(3, 2, 4)
    result, diagnostics = router(
        embeddings, history, deltas, return_diagnostics=True
    )
    assert torch.allclose(result, embeddings)
    assert torch.allclose(diagnostics["accept_probability"], torch.zeros(3, 1))
    assert torch.allclose(diagnostics["target_gate"], torch.zeros(3, 1))


def test_bootstrap_soft_gate_rejects_consistent_negative_gain():
    gains = np.asarray([-0.03, -0.02, -0.01, -0.04, -0.02] * 20)
    diagnostics = _bootstrap_gate_from_gains(
        gains,
        windows=5,
        replicates=500,
        confidence=0.95,
        min_gain=0.0,
        temperature=0.01,
        mode="lcb_soft",
        seed=7,
    )
    assert diagnostics["target_gate_lcb"] < 0.0
    assert diagnostics["target_gate"] < 0.2


def test_bootstrap_soft_gate_accepts_consistent_positive_gain():
    gains = np.asarray([0.03, 0.02, 0.01, 0.04, 0.02] * 20)
    diagnostics = _bootstrap_gate_from_gains(
        gains,
        windows=5,
        replicates=500,
        confidence=0.95,
        min_gain=0.0,
        temperature=0.01,
        mode="lcb_soft",
        seed=7,
    )
    assert diagnostics["target_gate_lcb"] > 0.0
    assert diagnostics["target_gate"] > 0.8


def test_pairwise_loss_is_per_event_and_prefers_larger_margin():
    pos = torch.tensor([3.0, 0.0, 1.0])
    neg = torch.tensor([0.0, 1.0, 1.0])
    losses = _pairwise_ranking_loss(pos, neg)
    assert losses.shape == (3,)
    assert losses[0] < losses[2] < losses[1]
