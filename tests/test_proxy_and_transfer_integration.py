from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forward_pair.proxy import score_checkpoint_proxy, score_checkpoint_proxy_windows
from tempofound.config import ExperimentConfig
from tempofound.pretrain import run_pretraining
from tempofound.transfer import run_transfer


def _write_dataset(root: Path, name: str, phase: int) -> None:
    count = 24
    order = np.arange(count)
    frame = pd.DataFrame(
        {
            "u": ((order + phase) % 5) + 1,
            "i": ((order + phase + 1) % 5) + 1,
            "idx": order + 1,
            "ts": order.astype(float) + 1.0,
            "label": np.zeros(count, dtype=int),
        }
    )
    frame.to_csv(root / f"ml_{name}.csv", index=False)
    np.save(
        root / f"ml_{name}_node.npy",
        np.arange(18, dtype=np.float32).reshape(6, 3) / (20.0 + phase),
    )
    np.save(
        root / f"ml_{name}.npy",
        np.arange(50, dtype=np.float32).reshape(25, 2) / (30.0 + phase),
    )


def test_proxy_and_frozen_transfer_run_on_cpu_without_held_out_selection_leakage(
    tmp_path,
) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    for phase, name in enumerate(("source_a", "source_b", "target")):
        _write_dataset(processed, name, phase)
    config = ExperimentConfig.from_mapping(
        {
            "name": "tiny",
            "datasets": ["source_a", "source_b", "target"],
            "targets": ["target"],
            "seeds": [42],
            "model": {
                "shared_dim": 8,
                "time_dim": 4,
                "num_layers": 1,
                "num_heads": 2,
                "num_neighbors": 2,
                "dropout": 0.0,
            },
            "training": {
                "pretrain_epochs": 1,
                "transfer_epochs": 1,
                "batch_size": 8,
                "total_batches_per_epoch": 2,
            },
            "proxy": {
                "calibration_fraction": 0.8,
                "calibration_steps": 1,
                "proxy_batch_size": 4,
                "proxy_batches": 1,
                "margin": 0.002,
            },
        }
    )
    checkpoint = tmp_path / "source_model.pth"
    run_pretraining(
        config,
        ("source_a", "source_b"),
        42,
        processed,
        checkpoint,
        "cpu",
    )

    proxy = score_checkpoint_proxy(
        checkpoint,
        target="target",
        processed_dir=processed,
        seed=42,
        batch_size=4,
        calibration_steps=1,
        probe_batches=1,
        calibration_fraction=0.8,
        device="cpu",
    )
    transfer = run_transfer(
        config,
        checkpoint_path=checkpoint,
        target="target",
        seed=42,
        processed_dir=processed,
        mode="frozen",
        device="cpu",
    )
    window_proxy = score_checkpoint_proxy_windows(
        checkpoint,
        target="target",
        processed_dir=processed,
        seed=42,
        batch_size=4,
        calibration_steps=1,
        probe_batches=1,
        calibration_fraction=0.7,
        proxy_windows=2,
        bootstrap_replicates=16,
        bootstrap_confidence=0.9,
        risk_beta=1.0,
        size_penalty=0.0,
        cardinality=2,
        device="cpu",
    )
    ranking_transfer = run_transfer(
        config,
        checkpoint_path=checkpoint,
        target="target",
        seed=42,
        processed_dir=processed,
        mode="frozen",
        device="cpu",
        evaluation_protocol="ranking",
        negative_policy="historical",
        negatives_per_positive=3,
        evaluation_batch_size=2,
    )

    assert 0.0 <= proxy["proxy_ap"] <= 1.0
    assert proxy["information_boundary"] == "target_training_period_only"
    assert len(window_proxy["window_proxy_ap"]) == 2
    assert window_proxy["bootstrap"]["mean"] == pytest.approx(
        sum(window_proxy["window_proxy_ap"]) / 2.0
    )
    assert np.isfinite(window_proxy["risk_adjusted_utility"])
    assert 0.0 <= transfer["test_ap"] <= 1.0
    assert 0.0 <= transfer["test_auc"] <= 1.0
    assert transfer["mode"] == "frozen"
    assert 0.0 <= ranking_transfer["test_mrr"] <= 1.0
    assert ranking_transfer["evaluation_protocol"] == "ranking"
    assert ranking_transfer["negative_policy"] == "historical"
    assert ranking_transfer["negatives_per_positive"] == 3
