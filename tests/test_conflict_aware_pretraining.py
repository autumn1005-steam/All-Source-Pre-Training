from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tempofound.config import ExperimentConfig
from tempofound.pretrain import run_pretraining


def _write_dataset(root: Path, name: str, offset: float) -> None:
    event_count = 24
    frame = pd.DataFrame(
        {
            "u": (np.arange(event_count) % 5) + 1,
            "i": ((np.arange(event_count) + 1) % 5) + 1,
            "idx": np.arange(1, event_count + 1),
            "ts": np.arange(1, event_count + 1, dtype=float) + offset,
            "label": np.zeros(event_count, dtype=int),
        }
    )
    frame.to_csv(root / f"ml_{name}.csv", index=False)
    np.save(
        root / f"ml_{name}_node.npy",
        np.arange(18, dtype=np.float32).reshape(6, 3) / 20.0,
    )
    np.save(
        root / f"ml_{name}.npy",
        np.arange(50, dtype=np.float32).reshape(25, 2) / 30.0,
    )


def test_conflict_aware_pretraining_respects_weighted_global_budget(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_dataset(processed, "source_a", 0.0)
    _write_dataset(processed, "source_b", 0.1)
    config = ExperimentConfig.from_mapping(
        {
            "name": "conflict-aware-tiny",
            "datasets": ["source_a", "source_b", "target"],
            "targets": ["target"],
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
                "batch_size": 8,
                "total_batches_per_epoch": 3,
                "pretraining_strategy": "conflict_aware",
                "minimum_source_batches": 1,
            },
        }
    )
    checkpoint = tmp_path / "model.pth"

    summary = run_pretraining(
        config,
        ("source_a", "source_b"),
        7,
        processed,
        checkpoint,
        source_weights={"source_a": 0.9, "source_b": 0.1},
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert summary["epoch_batch_counts"] == [3]
    assert summary["source_batch_counts"] == {"source_a": 2, "source_b": 1}
    assert summary["optimizer_step_counts"] == [2]
    assert summary["gradient_pairs"] == 1
    assert 0.0 <= summary["conflict_rate"] <= 1.0
    assert np.isfinite(summary["mean_cosine_before"])
    assert np.isfinite(summary["mean_cosine_after"])
    assert payload["metadata"]["pretraining"] == summary


def test_legacy_strategy_keeps_uniform_round_robin_budget(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_dataset(processed, "source_a", 0.0)
    _write_dataset(processed, "source_b", 0.1)
    config = ExperimentConfig.from_mapping(
        {
            "name": "legacy-tiny",
            "datasets": ["source_a", "source_b", "target"],
            "targets": ["target"],
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
                "batch_size": 8,
                "total_batches_per_epoch": 2,
                "pretraining_strategy": "legacy_sequential",
            },
        }
    )

    summary = run_pretraining(
        config,
        ("source_a", "source_b"),
        9,
        processed,
        tmp_path / "legacy.pth",
        source_weights={"source_a": 0.99, "source_b": 0.01},
    )

    assert summary["source_batch_counts"] == {"source_a": 1, "source_b": 1}
    assert summary["optimizer_step_counts"] == [2]
    assert summary["gradient_pairs"] == 0
