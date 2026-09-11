import math

import pytest
import torch

from tempofound.layers import HarmonicTimeEncoder


def test_paper_sincos_matches_normalized_manuscript_formula() -> None:
    encoder = HarmonicTimeEncoder(4, mode="paper_sincos")
    with torch.no_grad():
        encoder.basis_freq.copy_(torch.tensor([1.0, 2.0, 7.0, 8.0]))
        encoder.phase.zero_()

    actual = encoder(torch.tensor([[0.5]]))
    expected = 0.5 * torch.tensor(
        [[[math.cos(0.5), math.cos(1.0), math.sin(0.5), math.sin(1.0)]]]
    )

    assert actual.shape == (1, 1, 4)
    assert torch.allclose(actual, expected, atol=1e-7)


def test_legacy_mode_reproduces_all_cosine_channels_without_normalization() -> None:
    encoder = HarmonicTimeEncoder(4, mode="legacy_cos")
    with torch.no_grad():
        encoder.basis_freq.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        encoder.phase.zero_()

    actual = encoder(torch.zeros(1, 2))

    assert torch.equal(actual, torch.ones(1, 2, 4))


def test_paper_mode_rejects_odd_dimension() -> None:
    with pytest.raises(ValueError, match="even time_dim"):
        HarmonicTimeEncoder(3, mode="paper_sincos")
