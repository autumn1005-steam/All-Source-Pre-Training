import math

import pytest
import torch

from tempofound.optimization import project_conflicting_gradients


def test_opposed_gradients_are_projected() -> None:
    projected, stats = project_conflicting_gradients(
        [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0])]
    )

    assert torch.dot(projected[0], projected[1]) >= -1e-6
    assert stats["conflict_pairs"] == 1.0
    assert stats["total_pairs"] == 1.0
    assert stats["conflict_rate"] == pytest.approx(1.0)
    assert stats["mean_cosine_after"] >= stats["mean_cosine_before"]


def test_aligned_gradients_are_unchanged() -> None:
    original = [torch.tensor([1.0, 0.0]), torch.tensor([2.0, 1.0])]
    projected, stats = project_conflicting_gradients(original)

    assert all(torch.equal(a, b) for a, b in zip(projected, original, strict=True))
    assert stats["conflict_pairs"] == 0.0


def test_projection_preserves_shapes_and_does_not_mutate_inputs() -> None:
    first = torch.tensor([[1.0, 0.0]])
    second = torch.tensor([[-1.0], [1.0]])
    snapshots = [first.clone(), second.clone()]

    projected, _ = project_conflicting_gradients([first, second])

    assert [value.shape for value in projected] == [first.shape, second.shape]
    assert torch.equal(first, snapshots[0])
    assert torch.equal(second, snapshots[1])


def test_zero_gradient_produces_finite_statistics() -> None:
    projected, stats = project_conflicting_gradients(
        [torch.zeros(2), torch.tensor([1.0, 0.0])]
    )

    assert torch.equal(projected[0], torch.zeros(2))
    assert all(math.isfinite(value) for value in stats.values())
    assert stats["mean_cosine_before"] == pytest.approx(0.0)


def test_single_gradient_has_zero_pair_statistics() -> None:
    projected, stats = project_conflicting_gradients([torch.tensor([1.0])])

    assert torch.equal(projected[0], torch.tensor([1.0]))
    assert stats == {
        "conflict_pairs": 0.0,
        "total_pairs": 0.0,
        "conflict_rate": 0.0,
        "mean_cosine_before": 0.0,
        "mean_cosine_after": 0.0,
    }


@pytest.mark.parametrize(
    ("gradients", "message"),
    [
        ([], "non-empty"),
        ([torch.ones(2), torch.ones(3)], "same number"),
        ([torch.ones(2, dtype=torch.int64)], "floating point"),
        ([torch.tensor([float("nan")])], "finite"),
    ],
)
def test_invalid_gradients_are_rejected(
    gradients: list[torch.Tensor], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        project_conflicting_gradients(gradients)


def test_invalid_epsilon_is_rejected() -> None:
    with pytest.raises(ValueError, match="eps"):
        project_conflicting_gradients([torch.ones(1)], eps=0.0)
