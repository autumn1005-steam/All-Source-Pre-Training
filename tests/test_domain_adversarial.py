import pytest
import torch

from tempofound.layers import gradient_reverse


def test_gradient_reversal_changes_only_backward_sign() -> None:
    value = torch.tensor([2.0], requires_grad=True)

    output = gradient_reverse(value, 0.5)

    assert output.item() == 2.0
    output.sum().backward()
    assert value.grad is not None
    assert value.grad.item() == pytest.approx(-0.5)


def test_gradient_reversal_composes_with_upstream_gradient() -> None:
    value = torch.tensor([2.0, -3.0], requires_grad=True)

    (gradient_reverse(value, 2.0) * torch.tensor([3.0, 4.0])).sum().backward()

    assert value.grad is not None
    assert torch.equal(value.grad, torch.tensor([-6.0, -8.0]))


@pytest.mark.parametrize("scale", [-1.0, float("nan"), float("inf")])
def test_gradient_reversal_rejects_invalid_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="scale"):
        gradient_reverse(torch.ones(1), scale)
