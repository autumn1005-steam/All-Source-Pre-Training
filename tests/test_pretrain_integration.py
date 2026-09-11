from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tempofound.config import ExperimentConfig
from tempofound.pretrain import run_pretraining


def _write_processed_dataset(root: Path, name: str, offset: float) -> None:
    event_count = 24
    sources = (np.arange(event_count) % 5) + 1
    destinations = ((np.arange(event_count) + 1) % 5) + 1
    frame = pd.DataFrame(
        {
            "u": sources,
            "i": destinations,
            "idx": np.arange(1, event_count + 1),
            "ts": np.arange(1, event_count + 1, dtype=float) + offset,
            "label": np.zeros(event_count, dtype=int),
        }
    )
    frame.to_csv(root / f"ml_{name}.csv", index=False)
    node_features = np.arange(18, dtype=np.float32).reshape(6, 3) / 20.0
    edge_features = np.arange(50, dtype=np.float32).reshape(25, 2) / 30.0
    np.save(root / f"ml_{name}_node.npy", node_features)
    np.save(root / f"ml_{name}.npy", edge_features)


def test_smoke_pretraining_writes_versioned_checkpoint_with_exact_budget(tmp_path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    for index, name in enumerate(("source_a", "source_b")):
        _write_processed_dataset(processed, name, float(index) / 10.0)
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
        }
    )
    checkpoint = tmp_path / "model.pth"

    summary = run_pretraining(
        config,
        sources=("source_a", "source_b"),
        seed=42,
        processed_dir=processed,
        output_path=checkpoint,
        device="cpu",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert checkpoint.is_file()
    assert payload["format_version"] == 3
    assert payload["time_encoding_mode"] == "paper_sincos"
    assert summary["epoch_batch_counts"] == [2]
    assert summary["source_batch_counts"] == {"source_a": 1, "source_b": 1}
