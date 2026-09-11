import numpy as np

from tempofound.config import ExperimentConfig
from tempofound.transfer import _bootstrap_gate_from_gains, _chronological_target_partition


def _config(**overrides):
    base = dict(
        name="v3-test",
        datasets=("a", "b", "target"),
        targets=("target",),
        source_overrides={"target": ("a", "b")},
        selection_method="forward_pair",
        target_gate_mode="holdout_lcb_hard",
        target_calibration_fraction=0.2,
        target_calibration_min_events=2,
        target_gate_windows=2,
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def test_chronological_partition_reserves_latest_target_train_suffix():
    data = {
        "train_src": np.arange(10),
        "train_ts": np.asarray([5, 1, 9, 3, 7, 2, 10, 4, 8, 6], dtype=float),
    }
    opt, cal = _chronological_target_partition(data, _config())
    assert len(opt) == 8
    assert len(cal) == 2
    assert max(data["train_ts"][opt]) < min(data["train_ts"][cal])


def test_force_route_still_uses_held_out_partition():
    data = {"train_src": np.arange(10), "train_ts": np.arange(10, dtype=float)}
    opt, cal = _chronological_target_partition(data, _config(target_gate_mode="force_route"))
    assert len(opt) == 8
    assert len(cal) == 2


def test_holdout_soft_gate_is_exact_zero_when_lcb_not_positive():
    out = _bootstrap_gate_from_gains(
        np.asarray([-0.02, -0.01, 0.0, 0.001]),
        windows=2,
        replicates=256,
        confidence=0.95,
        min_gain=0.0,
        temperature=0.01,
        mode="holdout_lcb_soft",
        seed=7,
    )
    assert out["target_gate_lcb"] <= 0.0
    assert out["target_gate"] == 0.0


def test_holdout_hard_gate_accepts_only_positive_lcb():
    positive = _bootstrap_gate_from_gains(
        np.asarray([0.03, 0.04, 0.05, 0.06]),
        windows=2,
        replicates=256,
        confidence=0.95,
        min_gain=0.0,
        temperature=0.01,
        mode="holdout_lcb_hard",
        seed=11,
    )
    negative = _bootstrap_gate_from_gains(
        np.asarray([-0.03, -0.04, -0.05, -0.06]),
        windows=2,
        replicates=256,
        confidence=0.95,
        min_gain=0.0,
        temperature=0.01,
        mode="holdout_lcb_hard",
        seed=11,
    )
    assert positive["target_gate"] == 1.0
    assert negative["target_gate"] == 0.0


def test_force_identity_and_force_route_are_deterministic_controls():
    gains = np.asarray([0.1, 0.1, 0.1, 0.1])
    common = dict(
        gains=gains,
        windows=2,
        replicates=64,
        confidence=0.95,
        min_gain=0.0,
        temperature=0.01,
        seed=3,
    )
    assert _bootstrap_gate_from_gains(mode="force_identity", **common)["target_gate"] == 0.0
    assert _bootstrap_gate_from_gains(mode="force_route", **common)["target_gate"] == 1.0
